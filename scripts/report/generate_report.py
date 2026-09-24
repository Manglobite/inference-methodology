#!/usr/bin/env python3
"""Aggregate raw run results of an inference study into Markdown and JSON.

Read-only: the script only reads `<results-dir>/*/result.json` and writes
`<docs-dir>/results-tables.md` plus `<docs-dir>/results.json`. Standard library
only.

Configuration (`study.json` by default, `--config` overrides it) drives the
profile list, per-language labels, the study title, the control role, the
planned A/B orders and the ladder percentages. Auto-detection of profiles,
labels and A/B orders from the raw runs (every profile becomes `main`, label =
profile name, generic title) is used only when no explicit `--config` is given
and `./study.json` is absent. A configuration file that exists but is
unreadable or invalid is a hard error (`exit 2`), even when it is the default
`./study.json`; an explicitly requested `--config` must exist. The report
always states whether the configuration was used.

Canonical result (METHODOLOGY.md section 11.1): among `mode == "ladder"`,
`status == "ok"` runs, repetitions are aggregated per `target_pct` step.
Canonicality is decided **per step**: a step enters the aggregate only when it
is comparable — `completion_is_fixed == true`, `finish_reason == "length"` and
`step_comparable != false` (the optional field is absent for legacy runs).
For each step the median plus min/max of `prefill_tokens_per_second`,
`decode_tokens_per_second`, `cache_hit_fraction`, `elapsed_s` and (when present)
`overhead_s` are reported together with `n` and the list of `run_id`s. The
canonical minimum is applied per step: a step with fewer than 3 runs is marked
limited even when the profile total is larger; the profile total is reported as
well. The 0% warm-up step is excluded from the aggregate. Excluded steps are
not silently dropped: they are listed with the reason. Runs are classified
`fixed` (all steps comparable), `mixed` (partly) or `variable` (none); the
`mixed` and `variable` runs are reported in separate sections.

Usage from the case root:
    python3 scripts/report/generate_report.py
    python3 .../generate_report.py --results-dir results --docs-dir docs
"""

import argparse
import csv
import datetime as dt
import itertools
import json
import re
import statistics
import sys
from pathlib import Path

MISSING = "\u2014"
MIN_CANONICAL_N = 3
WARMUP_PCT = 0
PATH_PLACEHOLDER = "<path>"
DEFAULT_CONFIG_PATH = Path("./study.json")
DEFAULT_AB_ORDERS = ["ABAB", "ABBABAA"]
DEFAULT_TITLE = {"ru": "Результаты исследования", "en": "Study results"}
BASE_METRIC_KEYS = (
    "prefill_tokens_per_second",
    "decode_tokens_per_second",
    "cache_hit_fraction",
    "elapsed_s",
)
# Energy metrics are aggregated exactly like the base metrics (median+min/max,
# None-safe). They are None for legacy steps without telemetry or step
# timestamps and are then ignored by `step_stats`.
ENERGY_METRIC_KEYS = (
    "energy_j",
    "energy_dynamic_j",
    "power_avg_w",
    "energy_per_output_token_j",
    "energy_per_input_token_j",
    "energy_per_output_token_j_dynamic",
)
# `energy_samples` is a scalar count and is aggregated exactly like the metrics
# above, but it is kept out of `ENERGY_METRIC_KEYS` so that "is there any energy
# data" detection stays based on actual energy/power values (a step without
# telemetry legitimately reports `energy_samples == 0`).
ENERGY_SAMPLE_METRIC_KEYS = ("energy_samples",)
# `gpu_energy_j` is a per-GPU mapping (`{gpu_index: energy_j}`) and is never
# aggregated as a scalar; the GPU breakdown stays only in the raw runs.
METRIC_KEYS = BASE_METRIC_KEYS + ENERGY_METRIC_KEYS + ENERGY_SAMPLE_METRIC_KEYS
TELEMETRY_FILENAME = "telemetry.csv"
# Width of the explicit idle window probed right before the first request
# (model already loaded, GPU idle). The total watts are reduced by their median
# over this window, which is robust to a single low sample.
IDLE_WINDOW_S = 10.0
_GPU_POWER_RE = re.compile(r"^gpu(\d+)_power_w$")
_TS_FORMATS = ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ")

_ABS_PREFIXES = []

# Absolute path token (embedded anywhere in a string). The leading slash must
# not be preceded by a word character, a dot or another slash so relative
# fragments (`a/b`, `./x`), URLs and already-relative paths are left alone; the
# token excludes whitespace and common delimiters.
_ABS_PATH_RE = re.compile(
    r"(?<![\w./])/(?:[^\s/\"'`|,;:()\[\]{}<>]+/)*[^\s/\"'`|,;:()\[\]{}<>]+/?"
)


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def scrub_absolute_paths(text):
    """Reduce any absolute path left in `text` to `<path>/<basename>`.

    Paths under the case/repo prefixes are already made relative by the callers;
    this guards every other host prefix (`/home/`, `/mnt/`, `/opt/`, `/tmp/`, …)
    that the publication barrier rejects.
    """
    def replace(match):
        basename = match.group(0).rstrip("/").rsplit("/", 1)[-1]
        return f"{PATH_PLACEHOLDER}/{basename}" if basename else PATH_PLACEHOLDER

    return _ABS_PATH_RE.sub(replace, text)


def scrub_text(value):
    """Strip case/repo prefixes and any remaining absolute path in a string."""
    if not isinstance(value, str):
        return value
    for prefix, replacement in _ABS_PREFIXES:
        value = value.replace(prefix, replacement)
    return scrub_absolute_paths(value)


def scrub_value(value):
    """Recursively sanitize a structured value: strings via `scrub_text`.

    Dict keys are preserved; lists/tuples become lists of sanitized elements;
    any other value is returned unchanged.
    """
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        return {key: scrub_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_value(item) for item in value]
    return value


def relative_path_text(value):
    """Return a case-relative path string, never an absolute host path."""
    if not value:
        return value
    text = str(value)
    for prefix, replacement in _ABS_PREFIXES:
        if text.startswith(prefix):
            return scrub_absolute_paths(replacement + text[len(prefix):])
        exact = prefix.rstrip("/")
        if exact and text == exact:
            return scrub_absolute_paths(replacement or ".")
    return scrub_absolute_paths(text)


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def cell(value):
    if value is None or value == "":
        return MISSING
    return str(value).replace("|", "\\|").replace("\n", " ")


def number(value, digits=2):
    if value is None:
        return MISSING
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return MISSING


def integer(value):
    if value is None:
        return MISSING
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return MISSING


def step_is_comparable(step):
    """A step enters the canonical aggregate only when generation was fixed.

    `completion_is_fixed` must be true and `finish_reason` must be `length`;
    the optional `step_comparable` flag (new runner) can only further exclude
    (`is False`). Legacy steps without the flag rely on the first two fields.
    """
    if step.get("completion_is_fixed") is not True:
        return False
    if step.get("finish_reason") != "length":
        return False
    if step.get("step_comparable") is False:
        return False
    return True


def step_exclusion_reason(step):
    """Human-readable reason why a step fails the per-step comparability test."""
    if step.get("completion_is_fixed") is not True:
        return "completion_is_fixed != true"
    if step.get("finish_reason") != "length":
        return f"finish_reason={step.get('finish_reason')!r} != 'length'"
    if step.get("step_comparable") is False:
        return "step_comparable=false"
    return "not comparable"


def generation_status_for_steps(steps):
    """`fixed` if all steps are comparable, `variable` if none is, `mixed` when
    only part of them is; `unknown` when the run has no steps (for example smoke)."""
    if not steps:
        return "unknown"
    comparable = sum(1 for step in steps if step_is_comparable(step))
    if comparable == 0:
        return "variable"
    if comparable == len(steps):
        return "fixed"
    return "mixed"


def run_generation_status(run):
    return run.get("generation_status") or generation_status_for_steps(run.get("steps") or [])


# --- telemetry, energy and power -------------------------------------------

def parse_utc(value):
    """Parse an ISO-8601 UTC timestamp as written by the runner.

    Accepts `%Y-%m-%dT%H:%M:%S.%fZ` (millisecond precision) and the fractional
    seconds-less variant. Returns a timezone-aware UTC `datetime`, or None when
    the value is absent or not parseable.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    for fmt in _TS_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
    return None


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_LEGACY_TELEMETRY_WARNED = False


def load_telemetry(path):
    """Read `<path>/telemetry.csv` into a list of power samples.

    Each sample is `(datetime_utc, watts_total, {gpu_index: watts})`; only
    `timestamp_utc` and every `gpuN_power_w` column are used. Non-numeric or
    empty cells are skipped (treated as absent, never as a failure); a row
    without a parseable timestamp is skipped. The list is sorted by timestamp.
    Returns `[]` when the file is missing or unreadable.

    The mandatory `schema_version` column is enforced (METHODOLOGY.md section
    10): when the header carries it and its value is not `1`, the file is **not
    interpreted** — `[]` is returned and a warning is printed to stderr. A file
    without the column is treated as legacy schema 1 and processed as before;
    that fallback is warned about only once per process.
    """
    global _LEGACY_TELEMETRY_WARNED
    path = Path(path)
    if not path.is_file():
        return []
    try:
        handle = path.open("r", newline="", encoding="utf-8")
    except OSError:
        return []
    points = []
    with handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        rows = reader
        if "schema_version" in reader.fieldnames:
            first = next(reader, None)
            if first is None:
                return []
            version = first.get("schema_version")
            if _to_float(version) != 1:
                print(
                    f"warning: telemetry schema_version={version} != 1; "
                    "energy not computed",
                    file=sys.stderr,
                )
                return []
            rows = itertools.chain([first], reader)
        elif not _LEGACY_TELEMETRY_WARNED:
            _LEGACY_TELEMETRY_WARNED = True
            print(
                "warning: telemetry has no schema_version column; treating as "
                "legacy schema 1",
                file=sys.stderr,
            )
        power_columns = {}
        for name in reader.fieldnames:
            match = _GPU_POWER_RE.match(name)
            if match:
                power_columns[name] = int(match.group(1))
        for row in rows:
            timestamp = parse_utc(row.get("timestamp_utc"))
            if timestamp is None:
                continue
            gpu_watts = {}
            total = 0.0
            for name, index in power_columns.items():
                watts = _to_float(row.get(name))
                if watts is None:
                    continue
                gpu_watts[index] = watts
                total += watts
            points.append((timestamp, total, gpu_watts))
    points.sort(key=lambda point: point[0])
    return points


def _interpolate(series, moment):
    """Power at `moment` from a sorted `[(datetime, value)]` series.

    Linear interpolation between the bracketing samples, clamped to the value
    of the nearest sample outside the sampled range.
    """
    first_time, first_value = series[0]
    if moment <= first_time:
        return first_value
    last_time, last_value = series[-1]
    if moment >= last_time:
        return last_value
    for index in range(1, len(series)):
        time, value = series[index]
        if time >= moment:
            prev_time, prev_value = series[index - 1]
            span = (time - prev_time).total_seconds()
            if span <= 0:
                return value
            fraction = (moment - prev_time).total_seconds() / span
            return prev_value + (value - prev_value) * fraction
    return last_value


def _integrate_series_result(series, t0, t1):
    """Trapezoid integral of a sorted `[(datetime, value)]` series over [t0,t1].

    Samples inside the window are integrated over their actual `dt`. The two
    edge segments (`t0` to the first in-window sample and the last in-window
    sample to `t1`) use linear interpolation, clamped to the nearest sample
    outside the window. A window with zero or one in-window sample is still
    integrated: both boundaries are interpolated from the out-of-window
    neighbours (points before `t0` and after `t1`), which covers short steps
    that fall between two telemetry ticks. `energy_j=None` is returned only when
    the window is degenerate (`t1 <= t0`) or when fewer than two in-window
    samples exist and there is no out-of-window sample to interpolate the
    boundaries with (an empty series included); the number of in-window samples
    is always reported.
    """
    series = sorted(series, key=lambda item: item[0])
    if t1 <= t0:
        return {"energy_j": None, "power_avg_w": None, "samples": 0}
    window = [(time, value) for time, value in series if t0 <= time <= t1]
    if len(window) < 2:
        outside = [
            (time, value) for time, value in series if time < t0 or time > t1
        ]
        if not outside:
            return {"energy_j": None, "power_avg_w": None, "samples": len(window)}
    energy = 0.0
    prev_time = t0
    prev_value = _interpolate(series, t0)
    for time, value in window:
        energy += (time - prev_time).total_seconds() * (prev_value + value) / 2.0
        prev_time, prev_value = time, value
    energy += (t1 - prev_time).total_seconds() * (prev_value + _interpolate(series, t1)) / 2.0
    duration = (t1 - t0).total_seconds()
    return {
        "energy_j": energy,
        "power_avg_w": energy / duration if duration > 0 else None,
        "samples": len(window),
    }


def integrate_power(points, t0, t1):
    """Integrate total GPU power over `[t0, t1]` (both inclusive).

    Returns `{"energy_j": float|null, "power_avg_w": float|null, "samples": int}`.
    `power_avg_w` is `energy_j / (t1 - t0)`. Edge handling (including the
    short-window interpolation) is documented in `_integrate_series_result`.
    """
    series = [(point[0], point[1]) for point in points]
    return _integrate_series_result(series, t0, t1)


def integrate_dynamic_power(points, t0, t1, baseline_w):
    """Integrate `max(0, P_total - baseline_w)` over `[t0, t1]`.

    Same shape as `integrate_power`; `energy_j` is None when `baseline_w` is
    None.
    """
    if baseline_w is None:
        return {"energy_j": None, "power_avg_w": None, "samples": 0}
    series = [(point[0], max(0.0, point[1] - baseline_w)) for point in points]
    return _integrate_series_result(series, t0, t1)


def integrate_gpu_energy(points, t0, t1):
    """Per-GPU energy over `[t0, t1]` as `{gpu_index: energy_j|None}`."""
    indices = sorted({index for point in points for index in point[2]})
    result = {}
    for index in indices:
        series = [(point[0], point[2].get(index, 0.0)) for point in points]
        result[index] = _integrate_series_result(series, t0, t1)["energy_j"]
    return result


def idle_baseline_w(points, window_end=None, window_s=IDLE_WINDOW_S):
    """Idle (baseline) total power estimate.

    With an explicit idle window ending at `window_end` (the first request's
    `started_at_utc`, i.e. the model is loaded and the GPU is idle), the
    baseline is the **median** of the total watts over
    `[window_end - window_s, window_end]`. The median is robust to a single
    downward outlier, unlike the run-wide minimum, which one low sample shifts.
    When fewer than two samples fall inside that window, or when `window_end`
    is None, the previous behaviour is kept as a fallback: the minimum total
    watts over the whole run.

    Returns `{"baseline_w": float|None, "source": "idle_window_median"|
    "run_minimum_fallback", "samples": int, "window_s": float}`; `baseline_w`
    is None when there are no samples.
    """
    source = "run_minimum_fallback"
    samples = 0
    baseline_w = None
    if window_end is not None and points:
        window_start = window_end - dt.timedelta(seconds=window_s)
        window_watts = [
            point[1] for point in points
            if window_start <= point[0] <= window_end
        ]
        if len(window_watts) >= 2:
            baseline_w = statistics.median(window_watts)
            source = "idle_window_median"
            samples = len(window_watts)
    if baseline_w is None and points:
        baseline_w = min(point[1] for point in points)
        samples = len(points)
    return {
        "baseline_w": baseline_w,
        "source": source,
        "samples": samples,
        "window_s": window_s,
    }


def _per_token(energy_j, tokens):
    if energy_j is None or not is_number(tokens) or tokens == 0:
        return None
    return energy_j / float(tokens)


def step_energy_fields(step, points, baseline_w):
    """Energy/power fields for one step, all None when the window is missing.

    Uses `started_at_utc`/`finished_at_utc`; a step without both timestamps (or
    without telemetry) gets the empty shape and never raises. `gpu_energy_j` is
    the only non-scalar field (a per-GPU mapping); it is never aggregated and
    stays only in the raw runs.
    """
    empty = {
        "energy_j": None,
        "power_avg_w": None,
        "energy_dynamic_j": None,
        "energy_per_output_token_j": None,
        "energy_per_input_token_j": None,
        "energy_per_output_token_j_dynamic": None,
        "energy_samples": 0,
        "gpu_energy_j": {},
    }
    started = parse_utc(step.get("started_at_utc"))
    finished = parse_utc(step.get("finished_at_utc"))
    if started is None or finished is None or finished <= started:
        return empty
    total = integrate_power(points, started, finished)
    dynamic = integrate_dynamic_power(points, started, finished, baseline_w)
    energy_j = total["energy_j"]
    energy_dynamic_j = dynamic["energy_j"]
    completion_tokens = step.get("completion_tokens")
    evaluated_tokens = step.get("evaluated_tokens")
    gpu_energy = integrate_gpu_energy(points, started, finished)
    fields = dict(empty)
    fields.update({
        "energy_j": energy_j,
        "power_avg_w": total["power_avg_w"],
        "energy_dynamic_j": energy_dynamic_j,
        "energy_per_output_token_j": _per_token(energy_j, completion_tokens),
        "energy_per_input_token_j": _per_token(energy_j, evaluated_tokens),
        "energy_per_output_token_j_dynamic": _per_token(energy_dynamic_j, completion_tokens),
        "energy_samples": total["samples"],
        "gpu_energy_j": {
            index: value for index, value in gpu_energy.items() if value is not None
        },
    })
    return fields


_ENERGY_SCALAR_FIELDS = (
    "energy_j",
    "power_avg_w",
    "energy_dynamic_j",
    "energy_per_output_token_j",
    "energy_per_input_token_j",
    "energy_per_output_token_j_dynamic",
)


def _fresh_energy_value(key, value):
    """Whether a freshly computed energy field should replace a stored one.

    A scalar `None` means "no telemetry/timestamps", so the stored value is
    kept. `energy_samples` is an int (0 when absent) and `gpu_energy_j` a
    mapping, so emptiness is the signal there instead.
    """
    if key == "energy_samples":
        return is_number(value) and value > 0
    if key == "gpu_energy_j":
        return isinstance(value, dict) and len(value) > 0
    return value is not None


def enrich_step_energy(steps, run_dir):
    """Return copies of `steps` enriched with energy fields.

    Derived energy fields (`step_energy_fields`) and the idle-baseline metadata
    are owned by this aggregator: they are recomputed on every report
    regeneration, so a stale value already stored in result.json (computed
    against an older baseline) is refreshed. A stored value is kept only when
    the fresh computation has no data (None / zero samples / empty mapping),
    which preserves fields when telemetry or timestamps are missing. Fields
    written by the runner are never overwritten. Non-dict entries pass through
    unchanged. Telemetry is read once per run.
    """
    points = load_telemetry(run_dir / TELEMETRY_FILENAME)
    start_times = [
        parse_utc(step.get("started_at_utc"))
        for step in steps
        if isinstance(step, dict)
    ]
    start_times = [value for value in start_times if value is not None]
    window_end = min(start_times) if start_times else None
    baseline = idle_baseline_w(points, window_end=window_end)
    baseline_w = baseline["baseline_w"]
    enriched = []
    for step in steps:
        if not isinstance(step, dict):
            enriched.append(step)
            continue
        item = dict(step)
        for key, value in step_energy_fields(item, points, baseline_w).items():
            if key in _ENERGY_SCALAR_FIELDS or key in ("energy_samples", "gpu_energy_j"):
                if _fresh_energy_value(key, value):
                    item[key] = value
            else:
                item.setdefault(key, value)
        item["idle_baseline_w"] = baseline["baseline_w"]
        item["idle_baseline_source"] = baseline["source"]
        item["idle_window_s"] = baseline["window_s"]
        enriched.append(item)
    return enriched


def collect_run(run_dir):
    result = load_json(run_dir / "result.json")
    if not isinstance(result, dict):
        return None
    steps = result.get("steps")
    if not isinstance(steps, list):
        steps = []
    steps = enrich_step_energy(steps, run_dir)
    run_id = result.get("run_id", run_dir.name)
    status_reasons = result.get("status_reasons")
    if not isinstance(status_reasons, list):
        status_reasons = []
    return {
        "run_id": scrub_value(run_id),
        "run_path": f"results/{relative_path_text(run_id)}",
        "status": result.get("status", "failed" if result.get("error") else "unknown"),
        "error": scrub_value(result.get("error")),
        "status_reasons": [scrub_value(reason) for reason in status_reasons],
        "profile": result.get("profile"),
        "mode": result.get("mode"),
        "generation_status": generation_status_for_steps(steps),
        "model_path": relative_path_text(result.get("model_path")),
        "binary": relative_path_text(result.get("binary")),
        "profile_path": relative_path_text(result.get("profile_path")),
        "case_dir": relative_path_text(result.get("case_dir")),
        "repo_root": relative_path_text(result.get("repo_root")),
        "results_dir": relative_path_text(result.get("results_dir")),
        "build_variant": result.get("build_variant"),
        "ctx_size": result.get("ctx_size"),
        "parallel": result.get("parallel"),
        "cache_type_k": result.get("cache_type_k"),
        "cache_type_v": result.get("cache_type_v"),
        "n_predict": result.get("n_predict"),
        "seed": result.get("seed"),
        "port": result.get("port"),
        "order": result.get("order"),
        "steps": steps,
    }


def collect_runs(results_dir):
    if not results_dir.is_dir():
        return []
    runs = []
    for run_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        run = collect_run(run_dir)
        if run is not None:
            runs.append(run)
    return runs


# --- configuration ---------------------------------------------------------

def normalize_label(value, default):
    if isinstance(value, dict):
        ru = value.get("ru") or value.get("en") or default
        en = value.get("en") or value.get("ru") or default
        return {"ru": str(ru), "en": str(en)}
    if isinstance(value, str) and value:
        return {"ru": value, "en": value}
    return {"ru": default, "en": default}


def normalize_title(value):
    if isinstance(value, dict):
        ru = value.get("ru") or value.get("en") or DEFAULT_TITLE["ru"]
        en = value.get("en") or value.get("ru") or DEFAULT_TITLE["en"]
        return {"ru": str(ru), "en": str(en)}
    if isinstance(value, str) and value:
        return {"ru": value, "en": value}
    return dict(DEFAULT_TITLE)


def detect_config(runs):
    names = sorted({run["profile"] for run in runs if run["profile"]})
    profiles = [
        {"name": name, "label": {"ru": name, "en": name}, "role": "main"}
        for name in names
    ]
    ab_orders = sorted({
        run["order"] for run in runs
        if run["mode"] == "ab-sequence" and run["order"]
    })
    if not ab_orders:
        ab_orders = list(DEFAULT_AB_ORDERS)
    ab_planned = sorted({
        run["profile"] for run in runs
        if run["mode"] == "ab-sequence" and run["profile"]
    })
    pcts = set()
    for run in runs:
        if run["mode"] != "ladder":
            continue
        for step in run["steps"]:
            pct = step.get("target_pct")
            if is_number(pct):
                pcts.add(int(pct))
    return {
        "title": dict(DEFAULT_TITLE),
        "profiles": profiles,
        "ab_planned": ab_planned,
        "ab_orders": ab_orders,
        "ladder_pcts": sorted(pcts),
    }


def normalized_config(raw, runs):
    detected = detect_config(runs)
    if not isinstance(raw, dict):
        return detected

    profiles = []
    seen = set()
    for item in raw.get("profiles") or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        role = item.get("role")
        profiles.append({
            "name": str(name),
            "label": normalize_label(item.get("label"), str(name)),
            "role": role if role in ("main", "control") else "main",
        })
    if not profiles:
        profiles = detected["profiles"]

    ab_orders = raw.get("ab_orders")
    if not (isinstance(ab_orders, list) and ab_orders):
        ab_orders = detected["ab_orders"]
    ab_planned = raw.get("ab_planned")
    if not (isinstance(ab_planned, list) and ab_planned):
        ab_planned = detected["ab_planned"]
    ladder_pcts = raw.get("ladder_pcts")
    if not (isinstance(ladder_pcts, list) and ladder_pcts):
        ladder_pcts = detected["ladder_pcts"]

    return {
        "title": normalize_title(raw.get("title")),
        "profiles": profiles,
        "ab_planned": list(ab_planned),
        "ab_orders": list(ab_orders),
        "ladder_pcts": list(ladder_pcts),
    }


def profile_names(cfg, role=None):
    return [
        item["name"] for item in cfg["profiles"]
        if role is None or item["role"] == role
    ]


def profile_role(cfg, name):
    for item in cfg["profiles"]:
        if item["name"] == name:
            return item["role"]
    return "main"


def profile_label(cfg, name, lang):
    for item in cfg["profiles"]:
        if item["name"] == name:
            return item["label"].get(lang) or item["label"].get("ru") or name
    return name


# --- canonical aggregation -------------------------------------------------

def step_stats(values):
    numbers = [float(value) for value in values if is_number(value)]
    if not numbers:
        return None
    return {
        "n": len(numbers),
        "median": statistics.median(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def aggregate_profile(runs, name):
    """Aggregate comparable `ok` ladder steps of one profile per `target_pct`.

    Canonicality is per step: only comparable steps enter the median/min/max,
    while every non-comparable step of a candidate run is recorded in
    `excluded` with its reason. A run counts towards the profile `n` when it
    contributes at least one comparable step. A single run contributes at most
    one entry per `target_pct`: a second comparable step of the same run for the
    same percentage is refused as `duplicate target_pct for run`, so `n` can
    never be inflated by one run alone.
    """
    candidates = [
        run for run in runs
        if run["profile"] == name
        and run["mode"] == "ladder"
        and run["status"] == "ok"
    ]
    candidates.sort(key=lambda run: run["run_id"])

    by_pct = {}
    excluded = []
    contributing = set()
    for run in candidates:
        seen_pcts = set()
        for step in run["steps"] or []:
            pct = step.get("target_pct")
            if not is_number(pct) or int(pct) == WARMUP_PCT:
                continue
            pct = int(pct)
            if step_is_comparable(step):
                entries = by_pct.setdefault(pct, {})
                if pct in seen_pcts or run["run_id"] in entries:
                    excluded.append({
                        "run_id": run["run_id"],
                        "target_pct": pct,
                        "reason": "duplicate target_pct for run",
                        "finish_reason": step.get("finish_reason"),
                        "completion_tokens": step.get("completion_tokens"),
                    })
                    continue
                seen_pcts.add(pct)
                entries[run["run_id"]] = step
                contributing.add(run["run_id"])
            else:
                excluded.append({
                    "run_id": run["run_id"],
                    "target_pct": pct,
                    "reason": step_exclusion_reason(step),
                    "finish_reason": step.get("finish_reason"),
                    "completion_tokens": step.get("completion_tokens"),
                })

    steps = []
    for pct in sorted(by_pct):
        entries = by_pct[pct]
        run_ids = sorted(entries)
        steps.append({
            "target_pct": pct,
            "n": len(run_ids),
            "limited": len(run_ids) < MIN_CANONICAL_N,
            "run_ids": run_ids,
            "metrics": {
                key: step_stats([entries[run_id].get(key) for run_id in run_ids])
                for key in METRIC_KEYS
            },
            "overhead": step_stats([entries[run_id].get("overhead_s") for run_id in run_ids]),
        })

    n = len(contributing)
    return {
        "n": n,
        "limited": n < MIN_CANONICAL_N,
        "run_ids": sorted(contributing),
        "steps": steps,
        "excluded": excluded,
    }


def aggregate_profiles(runs, cfg):
    return {name: aggregate_profile(runs, name) for name in profile_names(cfg)}


# --- paired A/B delta ------------------------------------------------------

def _sign(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def ab_delta_series(runs):
    """Paired A/B deltas, keyed by `(profile, order)`.

    Pairs live inside a single `ab-sequence` run: for one `target_pct` the
    comparable steps of sessions `A` and `B` are matched via
    `step_is_comparable` (the warm-up `0 %` step is excluded, as everywhere
    else). `delta_pct = 100 * (B - A) / A`; a pair is skipped when A is None or
    zero, or when either side is not a number. Values are grouped per metric
    (`decode_tokens_per_second` / `prefill_tokens_per_second`) and per
    `target_pct`; different orders and percentages are never mixed.

    Returns `{(profile, order): {"decode": [...], "prefill": [...],
    "excluded_duplicates": [...]}}` where each metric list holds, per
    `target_pct`, `{"target_pct", "median_delta_pct", "n_pairs",
    "sign_consistency", "pairs": [{"run_id", "a", "b", "delta_pct"}]}`.
    `sign_consistency` is the share of pairs whose delta sign matches the sign
    of the median. A second comparable step of the same run and session for one
    `target_pct` is a duplicate: it is never used for pairing and is recorded in
    `excluded_duplicates` as `{"run_id", "target_pct", "session",
    "reason": "duplicate session step"}`.
    """
    buckets = {}
    for run in runs:
        if run.get("mode") != "ab-sequence":
            continue
        key = (run.get("profile"), run.get("order"))
        bucket = buckets.setdefault(
            key, {"decode": {}, "prefill": {}, "excluded_duplicates": []}
        )
        metric_pairs = bucket
        by_pct = {}
        for step in run.get("steps") or []:
            pct = step.get("target_pct")
            if not is_number(pct) or int(pct) == WARMUP_PCT:
                continue
            if not step_is_comparable(step):
                continue
            session = step.get("session")
            if session not in ("A", "B"):
                continue
            sessions = by_pct.setdefault(int(pct), {})
            if session in sessions:
                bucket["excluded_duplicates"].append({
                    "run_id": run.get("run_id"),
                    "target_pct": int(pct),
                    "session": session,
                    "reason": "duplicate session step",
                })
                continue
            sessions[session] = step
        for pct in sorted(by_pct):
            sessions = by_pct[pct]
            if "A" not in sessions or "B" not in sessions:
                continue
            for metric, field in (
                ("decode", "decode_tokens_per_second"),
                ("prefill", "prefill_tokens_per_second"),
            ):
                a = sessions["A"].get(field)
                b = sessions["B"].get(field)
                if not is_number(a) or not is_number(b) or a == 0:
                    continue
                metric_pairs[metric].setdefault(pct, []).append({
                    "run_id": run.get("run_id"),
                    "a": float(a),
                    "b": float(b),
                    "delta_pct": 100.0 * (float(b) - float(a)) / float(a),
                })

    series = {}
    for key, metric_pairs in buckets.items():
        entry = {
            "excluded_duplicates": sorted(
                metric_pairs["excluded_duplicates"],
                key=lambda item: (
                    str(item.get("run_id")),
                    item.get("target_pct"),
                    str(item.get("session")),
                ),
            ),
        }
        for metric in ("decode", "prefill"):
            items = []
            for pct in sorted(metric_pairs[metric]):
                pairs = sorted(
                    metric_pairs[metric][pct],
                    key=lambda pair: str(pair.get("run_id")),
                )
                deltas = [pair["delta_pct"] for pair in pairs]
                median = statistics.median(deltas)
                sign = _sign(median)
                consistent = sum(1 for delta in deltas if _sign(delta) == sign)
                items.append({
                    "target_pct": pct,
                    "median_delta_pct": median,
                    "n_pairs": len(pairs),
                    "sign_consistency": consistent / len(pairs) if pairs else 0.0,
                    "pairs": pairs,
                })
            entry[metric] = items
        series[key] = entry
    return series


def ab_delta_key(profile, order):
    """Stable JSON key for one `(profile, order)` pair."""
    return "{}|{}".format(
        profile if profile is not None else "",
        order if order is not None else "",
    )


def serialize_ab_delta(ab_delta):
    """JSON-serializable form: string key `"profile|order"` -> series dict."""
    return {
        ab_delta_key(profile, order): series
        for (profile, order), series in ab_delta.items()
    }


# --- markdown rendering ----------------------------------------------------

def stat_text(stat, digits):
    if not stat:
        return MISSING
    median = number(stat["median"], digits)
    if stat["min"] == stat["max"]:
        return median
    return f"{median} ({number(stat['min'], digits)}\u2013{number(stat['max'], digits)})"


def stat_summary_cell(aggregate):
    n = aggregate["n"]
    if n == 0:
        return MISSING
    if aggregate["limited"]:
        return f"limited (n={n})"
    return f"n={n}"


def render_variants(runs, cfg, aggregation):
    lines = [
        "| variant | role | ladder runs | ab-sequence runs | canonical n | aggregation "
        "| last status | ctx | KV | model |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in profile_names(cfg):
        variant_runs = [run for run in runs if run["profile"] == name]
        ladder_runs = [run for run in variant_runs if run["mode"] == "ladder"]
        ab_runs = [run for run in variant_runs if run["mode"] == "ab-sequence"]
        last = variant_runs[-1] if variant_runs else None
        kv = f"{cell(last['cache_type_k'])}/{cell(last['cache_type_v'])}" if last else MISSING
        lines.append("| " + " | ".join([
            cell(f"{profile_label(cfg, name, 'ru')} / {profile_label(cfg, name, 'en')}"),
            cell(profile_role(cfg, name)),
            integer(len(ladder_runs)) if variant_runs else MISSING,
            integer(len(ab_runs)) if variant_runs else MISSING,
            integer(aggregation[name]["n"]),
            cell(stat_summary_cell(aggregation[name])),
            cell(last["status"]) if last else MISSING,
            integer(last["ctx_size"]) if last else MISSING,
            cell(kv),
            cell(last["model_path"]) if last else MISSING,
        ]) + " |")
    return lines


def render_coverage(runs, cfg, aggregation):
    orders = list(cfg["ab_orders"])
    header = (
        ["variant", "role", "ladder runs", "fixed ok", "mixed ok", "fixed non-ok",
         "variable"]
        + [f"ab {order}" for order in orders]
        + ["coverage"]
    )
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for name in profile_names(cfg):
        variant_runs = [run for run in runs if run["profile"] == name]
        ladder_runs = [run for run in variant_runs if run["mode"] == "ladder"]
        fixed = [run for run in ladder_runs if run_generation_status(run) == "fixed"]
        mixed = [run for run in ladder_runs if run_generation_status(run) == "mixed"]
        fixed_ok = [run for run in fixed if run["status"] == "ok"]
        mixed_ok = [run for run in mixed if run["status"] == "ok"]
        fixed_other = [run for run in fixed if run["status"] != "ok"]
        variable = [run for run in ladder_runs if run_generation_status(run) == "variable"]
        if name in cfg["ab_planned"]:
            marks = [
                any(
                    run["mode"] == "ab-sequence" and run["status"] == "ok" and run["order"] == order
                    for run in variant_runs
                )
                for order in orders
            ]
            ab_cells = ["OK" if mark else "MISSING" for mark in marks]
            ab_ok = all(marks)
        else:
            ab_cells = ["N/A"] * len(orders)
            ab_ok = True
        n = aggregation[name]["n"]
        step_limited = [
            step["target_pct"] for step in aggregation[name]["steps"]
            if step["limited"]
        ]
        if not (fixed_ok or mixed_ok):
            coverage = "MISSING"
        elif n < MIN_CANONICAL_N or step_limited:
            coverage = "LIMITED"
        elif not ab_ok:
            coverage = "INCOMPLETE"
        else:
            coverage = "OK"
        lines.append("| " + " | ".join(
            [cell(name), cell(profile_role(cfg, name)), integer(len(ladder_runs)),
             integer(len(fixed_ok)), integer(len(mixed_ok)), integer(len(fixed_other)),
             integer(len(variable))]
            + ab_cells + [coverage]
        ) + " |")
    return lines


def render_summary(runs):
    lines = [
        "| run_id | path | profile | mode | status | generation | ctx | KV | parallel "
        "| n_predict | seed | steps | completion_tokens | completion_tokens_expected | error |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        kv = f"{cell(run['cache_type_k'])}/{cell(run['cache_type_v'])}"
        lines.append("| " + " | ".join([
            cell(run["run_id"]),
            cell(run["run_path"]),
            cell(run["profile"]),
            cell(run["mode"]),
            cell(run["status"]),
            cell(run["generation_status"]),
            integer(run["ctx_size"]),
            cell(kv),
            integer(run["parallel"]),
            integer(run["n_predict"]),
            integer(run["seed"]),
            integer(len(run["steps"])),
            completion_summary(run, "completion_tokens"),
            completion_summary(run, "completion_tokens_expected"),
            cell(run["error"]),
        ]) + " |")
    return lines


def completion_summary(run, key):
    values = sorted({
        int(step[key])
        for step in run["steps"]
        if isinstance(step.get(key), int) and not isinstance(step.get(key), bool)
    })
    if not values:
        return MISSING
    if len(values) == 1:
        return str(values[0])
    return f"{values[0]}\u2013{values[-1]}"


def render_ladder(aggregation, cfg, profiles, include_limited=False, limited_only=False):
    """Render the per-step aggregate table.

    `include_limited` controls whether steps below `MIN_CANONICAL_N` appear in
    the table (main section excludes them; the limited section includes them).
    `limited_only` restricts the table to those limited steps. The `overhead_s`
    column is only added when the field is present on at least one aggregated
    step.
    """
    selected = []
    for name in profiles:
        aggregate = aggregation.get(name)
        if not aggregate:
            continue
        for step in aggregate["steps"]:
            if limited_only and not step["limited"]:
                continue
            if not include_limited and not limited_only and step["limited"]:
                continue
            selected.append((name, step))
    show_overhead = any(step["overhead"] for _, step in selected)
    header = (
        "| profile | role | target_pct | n | prefill tok/s (min\u2013max) "
        "| decode tok/s (min\u2013max) | cache_hit_fraction (min\u2013max) "
        "| elapsed_s (min\u2013max) "
        + ("| overhead_s (min\u2013max) " if show_overhead else "")
        + "| run_id |"
    )
    column_count = 10 if show_overhead else 9
    lines = [header, "|" + "|".join(["---"] * column_count) + "|"]
    for name, step in selected:
        metrics = step["metrics"]
        row = [
            cell(f"{profile_label(cfg, name, 'ru')} / {profile_label(cfg, name, 'en')}"),
            cell(profile_role(cfg, name)),
            integer(step["target_pct"]),
            cell(stat_summary_cell(step)),
            stat_text(metrics["prefill_tokens_per_second"], 2),
            stat_text(metrics["decode_tokens_per_second"], 2),
            stat_text(metrics["cache_hit_fraction"], 4),
            stat_text(metrics["elapsed_s"], 2),
        ]
        if show_overhead:
            row.append(stat_text(step["overhead"], 3))
        row.append(cell(" ".join(step["run_ids"])))
        lines.append("| " + " | ".join(row) + " |")
    if not selected:
        lines.append("| " + " | ".join([MISSING] * column_count) + " |")
    return lines


def render_energy(aggregation, cfg, profiles):
    """Per-step energy/power table (median with min/max, like the ladder).

    Returns a single honest no-data line when no aggregated step carries any
    energy metric (legacy runs or missing telemetry); otherwise a table with
    total energy, dynamic energy, average power and the per-token energies.
    """
    selected = []
    for name in profiles:
        aggregate = aggregation.get(name)
        if not aggregate:
            continue
        for step in aggregate["steps"]:
            selected.append((name, step))
    has_energy = any(
        any(step["metrics"].get(key) is not None for key in ENERGY_METRIC_KEYS)
        for _, step in selected
    )
    if not has_energy:
        return ["Нет данных по энергии/мощности (нет `telemetry.csv` или "
                "`started_at_utc`/`finished_at_utc`). / No energy/power data "
                "(no `telemetry.csv` or step timestamps)."]
    header = (
        "| profile | role | target_pct | n | energy_j (min\u2013max) "
        "| energy_dynamic_j (min\u2013max) | power_avg_w (min\u2013max) "
        "| J/output token (min\u2013max) | J/input token (min\u2013max) "
        "| J/output token dynamic (min\u2013max) | run_id |"
    )
    column_count = 11
    lines = [header, "|" + "|".join(["---"] * column_count) + "|"]
    for name, step in selected:
        metrics = step["metrics"]
        row = [
            cell(f"{profile_label(cfg, name, 'ru')} / {profile_label(cfg, name, 'en')}"),
            cell(profile_role(cfg, name)),
            integer(step["target_pct"]),
            cell(stat_summary_cell(step)),
            stat_text(metrics.get("energy_j"), 1),
            stat_text(metrics.get("energy_dynamic_j"), 1),
            stat_text(metrics.get("power_avg_w"), 1),
            stat_text(metrics.get("energy_per_output_token_j"), 4),
            stat_text(metrics.get("energy_per_input_token_j"), 6),
            stat_text(metrics.get("energy_per_output_token_j_dynamic"), 4),
            cell(" ".join(step["run_ids"])),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_limited_and_excluded(runs, cfg, aggregation):
    lines = []
    for name in profile_names(cfg):
        aggregate = aggregation[name]
        if aggregate["n"] == 0:
            lines.append(
                f"- `{cell(name)}`: нет канонических прогонов (`ok` + comparable "
                "ladder). / no canonical runs (`ok` + comparable ladder)."
            )
        else:
            step_limited = [
                step for step in aggregate["steps"] if step["limited"]
            ]
            if step_limited:
                details = ", ".join(
                    f"{step['target_pct']} % (n={step['n']}): "
                    f"prefill={stat_text(step['metrics']['prefill_tokens_per_second'], 2)}, "
                    f"decode={stat_text(step['metrics']['decode_tokens_per_second'], 2)}"
                    for step in step_limited
                )
                lines.append(
                    f"- `{cell(name)}`: ступени limited: {details} \u2014 "
                    f"агрегат сохранён, но неканонический. Всего канонических "
                    f"прогонов n={aggregate['n']}, нужно \u2265 {MIN_CANONICAL_N} "
                    "на ступень. / steps limited: " + details + "; kept but "
                    "non-canonical."
                )
            elif aggregate["limited"]:
                lines.append(
                    f"- `{cell(name)}`: limited (n={aggregate['n']}, нужно "
                    f"\u2265 {MIN_CANONICAL_N}) \u2014 агрегат сохранён, но "
                    "неканонический. / limited (n="
                    f"{aggregate['n']}), kept but non-canonical."
                )
    for name in profile_names(cfg):
        for item in aggregation[name]["excluded"]:
            lines.append(
                f"- `{cell(item['run_id'])}` (profile `{cell(name)}`, "
                f"target_pct={item['target_pct']}): ступень исключена из "
                f"агрегата \u2014 {cell(item['reason'])} (finish_reason="
                f"{cell(item['finish_reason'])}, completion_tokens="
                f"{integer(item['completion_tokens'])}). / step excluded from "
                f"the aggregate \u2014 {cell(item['reason'])}."
            )
    for run in runs:
        if run["mode"] != "ladder" or run_generation_status(run) == "variable":
            continue
        if run["status"] == "ok":
            continue
        lines.append(
            f"- `{cell(run['run_id'])}` (`{cell(run['run_path'])}`, profile "
            f"`{cell(run['profile'])}`): ladder run excluded from the aggregate, "
            f"status `{cell(run['status'])}`."
        )
    if not lines:
        lines.append("Нет ограниченных агрегатов и исключённых ступеней/прогонов. / "
                     "No limited aggregates or excluded steps/runs.")
    return lines


def render_variable_runs(runs):
    variable_runs = [
        run for run in runs
        if run_generation_status(run) in ("variable", "mixed")
    ]
    lines = [
        "Прогоны ниже имеют переменную длину генерации "
        "(`completion_is_fixed != true` или `finish_reason != length`): "
        "генерация останавливалась сама (`finish_reason = stop`) на разной "
        "длине, поэтому **decode tok/s несопоставим** между прогонами. "
        "Перечислены только несопоставимые ступени; у `mixed`-прогонов "
        "сопоставимые ступени входят в агрегат выше. / Runs with variable "
        "generation length; only non-comparable steps are listed; comparable "
        "steps of `mixed` runs are part of the aggregate above.",
        "",
        "| run_id | path | profile | mode | order | gen | step | target_pct "
        "| cache_hit_fraction | completion_tokens | finish_reason | prefill tok/s "
        "| reason |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in variable_runs:
        generation = run_generation_status(run)
        for step in run["steps"]:
            if step_is_comparable(step):
                continue
            lines.append("| " + " | ".join([
                cell(run["run_id"]),
                cell(run["run_path"]),
                cell(run["profile"]),
                cell(run["mode"]),
                cell(run["order"]),
                cell(generation),
                integer(step.get("step")),
                integer(step.get("target_pct")),
                number(step.get("cache_hit_fraction"), digits=4),
                integer(step.get("completion_tokens")),
                cell(step.get("finish_reason")),
                number(step.get("prefill_tokens_per_second")),
                cell(step_exclusion_reason(step)),
            ]) + " |")
    if not variable_runs:
        lines.append("| " + " | ".join([MISSING] * 13) + " |")
    return lines


def render_ab(runs):
    lines = [
        "| run_id | profile | order | generation | order_index | session | step | target_pct "
        "| prompt_tokens | evaluated_tokens | cache_hit_fraction | prefill tok/s "
        "| decode tok/s | completion_tokens | completion_tokens_expected | finish_reason |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        if run["mode"] != "ab-sequence":
            continue
        generation = run_generation_status(run)
        for step in run["steps"]:
            lines.append("| " + " | ".join([
                cell(run["run_id"]),
                cell(run["profile"]),
                cell(run["order"]),
                cell(generation),
                integer(step.get("order_index")),
                cell(step.get("session")),
                integer(step.get("step")),
                integer(step.get("target_pct")),
                integer(step.get("prompt_tokens")),
                integer(step.get("evaluated_tokens")),
                number(step.get("cache_hit_fraction"), digits=4),
                number(step.get("prefill_tokens_per_second")),
                number(step.get("decode_tokens_per_second")),
                integer(step.get("completion_tokens")),
                integer(step.get("completion_tokens_expected")),
                cell(step.get("finish_reason")),
            ]) + " |")
    return lines


def render_ab_delta(ab_delta):
    """Paired A/B delta tables, one per metric (decode then prefill).

    Rows are `(profile, order, target_pct)`; columns carry the median delta,
    the pair count, the sign consistency and the contributing run ids. Returns
    an honest no-data line when nothing was paired.
    """
    if not ab_delta:
        return ["Нет данных для парного A/B-Δ (нужны `ab-sequence`-прогоны с "
                "сопоставимыми ступенями A и B). / No paired A/B delta data "
                "(requires `ab-sequence` runs with comparable A and B steps)."]
    lines = []
    for metric in ("decode", "prefill"):
        lines += [
            f"### Δ {metric}, % (B\u2212A)/A",
            "",
            "| profile | order | target_pct | median Δ, % | n pairs "
            "| sign consistency | run_id |",
            "|---|---|---|---|---|---|---|",
        ]
        rows = []
        for key in sorted(ab_delta, key=lambda item: (str(item[0]), str(item[1]))):
            profile, order = key
            for item in ab_delta[key].get(metric) or []:
                run_ids = " ".join(
                    str(pair.get("run_id")) for pair in item["pairs"]
                )
                rows.append("| " + " | ".join([
                    cell(profile),
                    cell(order),
                    integer(item["target_pct"]),
                    number(item["median_delta_pct"], 3),
                    integer(item["n_pairs"]),
                    number(item["sign_consistency"], 3),
                    cell(run_ids),
                ]) + " |")
        if rows:
            lines += rows
        else:
            lines.append("| " + " | ".join([MISSING] * 7) + " |")
        lines.append("")
    return lines


def render_timing_divergence(runs):
    """Compact report of API-vs-log timing divergence across all steps.

    Uses the optional `timing_delta_pct` field (`{"prefill": .., "decode": ..}`)
    written by the runner; when no step carries it, says so explicitly.
    """
    count = 0
    max_by_metric = {"prefill": None, "decode": None}
    for run in runs:
        for step in run["steps"] or []:
            delta = step.get("timing_delta_pct")
            if not isinstance(delta, dict):
                continue
            count += 1
            for metric in max_by_metric:
                value = delta.get(metric)
                if not is_number(value):
                    continue
                magnitude = abs(float(value))
                current = max_by_metric[metric]
                if current is None or magnitude > current:
                    max_by_metric[metric] = magnitude
    if count == 0:
        return ["Нет данных о расхождении источников таймингов "
                "(`timing_delta_pct` отсутствует). / No timing-source "
                "divergence data (`timing_delta_pct` absent)."]
    def render_metric(metric):
        value = max_by_metric[metric]
        return number(value, 4) + " %" if value is not None else MISSING
    return [
        f"Ступеней с `timing_delta_pct`: {count}. Максимальное |расхождение| "
        f"API-vs-лог: prefill {render_metric('prefill')}, "
        f"decode {render_metric('decode')} (знак не учитывается). / Steps with "
        "`timing_delta_pct`: " + str(count) + "; max |API-vs-log divergence|: "
        f"prefill {render_metric('prefill')}, decode {render_metric('decode')}."
    ]


def render_markdown(runs, cfg, aggregation, config_note):
    main = profile_names(cfg, "main")
    control = profile_names(cfg, "control")
    lines = [
        f"# {cfg['title']['ru']} / {cfg['title']['en']}",
        "",
        f"Прогонов / runs: {len(runs)}",
        "",
        "Профили / profiles: "
        + ", ".join(
            f"{name} ({profile_role(cfg, name)})" for name in profile_names(cfg)
        ) + ".",
        f"Планируемые A/B-порядки / planned A/B orders: {', '.join(cfg['ab_orders'])}.",
        f"Ступени лесенки / ladder steps: "
        + ", ".join(str(pct) for pct in cfg["ladder_pcts"]) + ".",
        "Источник конфигурации / configuration source: " + config_note + ".",
        "",
        f"Канонический результат / canonical result: агрегат по ступеням "
        f"`target_pct`; каноничность — по каждой ступени: при `n < {MIN_CANONICAL_N}` "
        "на ступени ступень помечается limited, даже если суммарно прогонов больше. "
        "Суммарный `n` по профилю приводится отдельно. Агрегация — медиана + "
        "min/max, с указанием `n` (per-step n/min/max в `results.json`) и списка "
        "`run_id`. Ступень "
        f"`{WARMUP_PCT} %` (прогрев) в агрегат не входит. `variable`-прогоны и "
        "неуспешные (`failed`/`timeout`/`non_comparable`) в агрегат не попадают. / "
        "Canonical result is the median aggregate per step; a step with fewer "
        f"than {MIN_CANONICAL_N} runs is limited, even when the total is larger; "
        "the warm-up step is excluded.",
        "",
        "## Сводка по вариантам / Variant summary",
        "",
    ]
    lines += render_variants(runs, cfg, aggregation)
    lines += ["", "## Покрытие профилей / Profile coverage", ""]
    lines += render_coverage(runs, cfg, aggregation)
    lines += ["", "## Прогоны / Runs", ""]
    lines += render_summary(runs)
    lines += [
        "",
        "## Контекстная лесенка, агрегат (основные профили) "
        "/ Context ladder, aggregate (main profiles)",
        "",
    ]
    lines += render_ladder(aggregation, cfg, main, include_limited=False)
    lines += [
        "",
        "## Контрольные профили / Control profiles",
        "",
        "Профили с ролью `control` (обход/контроль). На графиках они рисуются "
        "пунктиром; в основную серию не входят. / Profiles with role `control` "
        "(workaround/control); dashed on the figures; not part of the main series.",
        "",
    ]
    lines += render_ladder(aggregation, cfg, control, include_limited=False)
    lines += [
        "",
        "## Ограниченные агрегаты и исключённые прогоны "
        "/ Limited aggregates and excluded runs",
        "",
    ]
    lines += render_ladder(aggregation, cfg, profile_names(cfg), limited_only=True)
    lines += [""]
    lines += render_limited_and_excluded(runs, cfg, aggregation)
    lines += [
        "",
        "## Энергия и мощность / Energy and power",
        "",
        "Полная энергия и средняя мощность ступени по `telemetry.csv`: интеграл "
        "`gpuN_power_w` трапециями по фактическим `dt` в окне "
        "`started_at_utc`/`finished_at_utc`. `energy_dynamic_j` вычитает "
        "baseline из idle-окна (медиана суммарной мощности за `IDLE_WINDOW_S` "
        "перед первым запросом, fallback — минимум по прогону; источник — "
        "`idle_baseline_source`); per-token энергии делят энергию на "
        "`completion_tokens`/`evaluated_tokens`. Медиана + min/max, как в "
        "лесенке. / Per-step energy and average power from `telemetry.csv` "
        "(trapezoid integral over the step window); dynamic energy subtracts an "
        "idle-window baseline (median total power over `IDLE_WINDOW_S` before the "
        "first request, falling back to the run minimum; source in "
        "`idle_baseline_source`); per-token energies divide by the token counts.",
        "",
    ]
    lines += render_energy(aggregation, cfg, profile_names(cfg))
    lines += [
        "",
        "## Расхождение источников таймингов / Timing source divergence",
        "",
    ]
    lines += render_timing_divergence(runs)
    lines += [
        "",
        "## Прогоны с переменной генерацией / Variable-generation runs",
        "",
    ]
    lines += render_variable_runs(runs)
    lines += [
        "",
        "## A/B-кеш / A/B cache reuse",
        "",
        "Строки сгруппированы по прогону, порядку (`order`) и сессии (`session`). "
        "`cache_hit_fraction` — вычисленная доля промпта из кеша (`1 - evaluated/prompt`). / "
        "A/B rows grouped by run, order and session; cache hit fraction is computed.",
        "",
    ]
    lines += render_ab(runs)
    lines += [
        "",
        "## Парный A/B-Δ / Paired A/B delta",
        "",
        "Для одного прогона `ab-sequence`, порядка (`order`) и `target_pct` "
        "сопоставляются comparable-ступени сессий A и B; Δ = 100·(B−A)/A. "
        "`sign_consistency` — доля пар, знак Δ которых совпадает со знаком "
        "медианы. Ступень прогрева `0 %` исключена. / Within one `ab-sequence` "
        "run, order and `target_pct`, comparable A and B steps are paired; "
        "delta = 100·(B−A)/A; the warm-up `0 %` step is excluded.",
        "",
    ]
    lines += render_ab_delta(ab_delta_series(runs))
    lines += [
        "",
        "Канонический источник скоростей — сырой тайминг `server.log` (log-first); "
        "API `timings.*` — вторичное подтверждение, сохраняется отдельно "
        "(`*_api`/`*_log`), расхождение фиксируется в `timing_delta_pct`. "
        "`cache_hit_fraction = 1 - evaluated/prompt`. Оценка качества не "
        "выполняется. Режимы `smoke` в агрегированную лесенку не попадают. / "
        "Canonical speeds come from the raw `server.log` timing (log-first); "
        "API `timings.*` is a secondary confirmation, stored separately "
        "(`*_api`/`*_log`), with divergence recorded in `timing_delta_pct`. "
        "`cache_hit_fraction = 1 - evaluated/prompt`. No quality assessment is "
        "performed. `smoke` modes are excluded from the aggregated ladder.",
        "",
    ]
    return "\n".join(lines) + "\n"


def build_payload(results_dir, runs, cfg, aggregation):
    return {
        "results_dir": relative_results_dir(results_dir),
        "config": cfg,
        "aggregation": {
            "method": "median+min/max",
            "min_canonical_n": MIN_CANONICAL_N,
            "warmup_pct_excluded": WARMUP_PCT,
            "profiles": {
                name: {
                    "role": profile_role(cfg, name),
                    "label": {
                        "ru": profile_label(cfg, name, "ru"),
                        "en": profile_label(cfg, name, "en"),
                    },
                    "n": aggregate["n"],
                    "limited": aggregate["limited"],
                    "run_ids": aggregate["run_ids"],
                    "steps": aggregate["steps"],
                    "excluded": aggregate["excluded"],
                }
                for name, aggregate in aggregation.items()
            },
        },
        "ab_orders": list(cfg["ab_orders"]),
        "ab_delta": serialize_ab_delta(ab_delta_series(runs)),
        "runs": runs,
    }


def resolve_case_paths(results_dir, case_dir, repo_root):
    resolved = results_dir.resolve()
    resolved_case = case_dir.resolve() if case_dir else resolved.parent
    resolved_repo = repo_root.resolve() if repo_root else Path.cwd().resolve()
    return resolved_case, resolved_repo


_ABS_CASE_DIR = Path.cwd()


def configure_paths(results_dir, case_dir=None, repo_root=None):
    """Resolve and install the case/repo prefixes used to scrub output paths."""
    global _ABS_PREFIXES, _ABS_CASE_DIR
    resolved_case, resolved_repo = resolve_case_paths(results_dir, case_dir, repo_root)
    _ABS_CASE_DIR = resolved_case
    _ABS_PREFIXES = [
        (str(resolved_case) + "/", ""),
        (str(resolved_repo) + "/", ""),
    ]
    return resolved_case, resolved_repo


def relative_results_dir(results_dir):
    try:
        return str(Path(results_dir).resolve().relative_to(_ABS_CASE_DIR))
    except (ValueError, OSError):
        return Path(results_dir).name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("./results"))
    parser.add_argument("--docs-dir", type=Path, default=Path("./docs"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--case-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        print(f"results dir not found: {args.results_dir}", file=sys.stderr)
        return 2

    configure_paths(args.results_dir, args.case_dir, args.repo_root)

    runs = collect_runs(args.results_dir)

    raw_config = None
    config_path = args.config if args.config is not None else DEFAULT_CONFIG_PATH
    config_explicit = args.config is not None
    if config_path.is_file():
        raw_config = load_json(config_path)
        if raw_config is None or not isinstance(raw_config, dict):
            print(
                f"invalid config file: {relative_path_text(config_path)}",
                file=sys.stderr,
            )
            return 2
        config_note = f"config file {relative_path_text(config_path)}"
    elif config_explicit:
        print(
            f"config file not found: {relative_path_text(config_path)}",
            file=sys.stderr,
        )
        return 2
    else:
        config_note = "auto-detected (no ./study.json; config not used)"
    cfg = normalized_config(raw_config, runs)
    aggregation = aggregate_profiles(runs, cfg)

    args.docs_dir.mkdir(parents=True, exist_ok=True)
    (args.docs_dir / "results-tables.md").write_text(
        render_markdown(runs, cfg, aggregation, config_note), encoding="utf-8"
    )
    (args.docs_dir / "results.json").write_text(
        json.dumps(build_payload(args.results_dir, runs, cfg, aggregation),
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"runs: {len(runs)}; profiles: {len(cfg['profiles'])}; wrote "
        f"{args.docs_dir / 'results-tables.md'} and {args.docs_dir / 'results.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
