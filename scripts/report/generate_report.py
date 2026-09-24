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
`status == "ok"` runs with `completion_is_fixed == true`, repetitions are
aggregated per `target_pct` step. For each step the median plus min/max of
`prefill_tokens_per_second`, `decode_tokens_per_second`, `cache_hit_fraction`
and `elapsed_s` are reported together with `n` and the list of `run_id`s. The
canonical minimum is applied per step: a step with fewer than 3 runs is marked
limited even when the profile total is larger; the profile total is reported as
well. The 0% warm-up step is excluded from the aggregate. Fixed and variable
generation runs are never mixed; variable runs are reported in a separate
section.

Usage from the case root:
    python3 scripts/report/generate_report.py
    python3 .../generate_report.py --results-dir results --docs-dir docs
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

MISSING = "\u2014"
MIN_CANONICAL_N = 3
WARMUP_PCT = 0
DEFAULT_CONFIG_PATH = Path("./study.json")
DEFAULT_AB_ORDERS = ["ABAB", "ABBABAA"]
DEFAULT_TITLE = {"ru": "Результаты исследования", "en": "Study results"}
METRIC_KEYS = (
    "prefill_tokens_per_second",
    "decode_tokens_per_second",
    "cache_hit_fraction",
    "elapsed_s",
)

_ABS_PREFIXES = []


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def scrub_text(value):
    """Strip absolute case/repo prefixes embedded anywhere in a string."""
    if not isinstance(value, str):
        return value
    for prefix, replacement in _ABS_PREFIXES:
        value = value.replace(prefix, replacement)
    return value


def relative_path_text(value):
    """Return a case-relative path string, never an absolute host path."""
    if not value:
        return value
    text = str(value)
    for prefix, replacement in _ABS_PREFIXES:
        if text.startswith(prefix):
            return replacement + text[len(prefix):]
    return text


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


def generation_status_for_steps(steps):
    """`fixed` if any step is flagged, `variable` if steps exist but none is,
    `unknown` when the run has no steps (for example smoke)."""
    if not steps:
        return "unknown"
    if any(step.get("completion_is_fixed") is True for step in steps):
        return "fixed"
    return "variable"


def run_generation_status(run):
    return run.get("generation_status") or generation_status_for_steps(run.get("steps") or [])


def collect_run(run_dir):
    result = load_json(run_dir / "result.json")
    if not isinstance(result, dict):
        return None
    steps = result.get("steps")
    if not isinstance(steps, list):
        steps = []
    run_id = result.get("run_id", run_dir.name)
    return {
        "run_id": run_id,
        "run_path": f"results/{run_id}",
        "status": result.get("status", "failed" if result.get("error") else "unknown"),
        "error": scrub_text(result.get("error")),
        "profile": result.get("profile"),
        "mode": result.get("mode"),
        "generation_status": generation_status_for_steps(steps),
        "model_path": relative_path_text(result.get("model_path")),
        "binary": relative_path_text(result.get("binary")),
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
    """Aggregate fixed `ok` ladder runs of one profile per `target_pct` step."""
    candidates = [
        run for run in runs
        if run["profile"] == name
        and run["mode"] == "ladder"
        and run["status"] == "ok"
        and run_generation_status(run) == "fixed"
    ]
    candidates.sort(key=lambda run: run["run_id"])

    by_pct = {}
    for run in candidates:
        for step in run["steps"] or []:
            pct = step.get("target_pct")
            if not is_number(pct) or int(pct) == WARMUP_PCT:
                continue
            by_pct.setdefault(int(pct), []).append((run["run_id"], step))

    steps = []
    for pct in sorted(by_pct):
        entries = by_pct[pct]
        n = len(entries)
        steps.append({
            "target_pct": pct,
            "n": n,
            "limited": n < MIN_CANONICAL_N,
            "run_ids": sorted({run_id for run_id, _ in entries}),
            "metrics": {
                key: step_stats([step.get(key) for _, step in entries])
                for key in METRIC_KEYS
            },
        })

    n = len(candidates)
    return {
        "n": n,
        "limited": n < MIN_CANONICAL_N,
        "run_ids": [run["run_id"] for run in candidates],
        "steps": steps,
    }


def aggregate_profiles(runs, cfg):
    return {name: aggregate_profile(runs, name) for name in profile_names(cfg)}


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
        ["variant", "role", "ladder runs", "fixed ok", "fixed non-ok", "variable"]
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
        fixed_ok = [run for run in fixed if run["status"] == "ok"]
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
        if not fixed_ok:
            coverage = "MISSING"
        elif n < MIN_CANONICAL_N or step_limited:
            coverage = "LIMITED"
        elif not ab_ok:
            coverage = "INCOMPLETE"
        else:
            coverage = "OK"
        lines.append("| " + " | ".join(
            [cell(name), cell(profile_role(cfg, name)), integer(len(ladder_runs)),
             integer(len(fixed_ok)), integer(len(fixed_other)), integer(len(variable))]
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


def render_ladder(aggregation, cfg, profiles):
    lines = [
        "| profile | role | target_pct | n | prefill tok/s (min\u2013max) "
        "| decode tok/s (min\u2013max) | cache_hit_fraction (min\u2013max) "
        "| elapsed_s (min\u2013max) | run_id |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in profiles:
        aggregate = aggregation.get(name)
        if not aggregate or not aggregate["steps"]:
            continue
        for step in aggregate["steps"]:
            metrics = step["metrics"]
            lines.append("| " + " | ".join([
                cell(f"{profile_label(cfg, name, 'ru')} / {profile_label(cfg, name, 'en')}"),
                cell(profile_role(cfg, name)),
                integer(step["target_pct"]),
                cell(stat_summary_cell(step)),
                stat_text(metrics["prefill_tokens_per_second"], 2),
                stat_text(metrics["decode_tokens_per_second"], 2),
                stat_text(metrics["cache_hit_fraction"], 4),
                stat_text(metrics["elapsed_s"], 2),
                cell(" ".join(step["run_ids"])),
            ]) + " |")
    if not any(aggregation.get(name) and aggregation[name]["steps"] for name in profiles):
        lines.append("| " + " | ".join([MISSING] * 9) + " |")
    return lines


def render_limited_and_excluded(runs, cfg, aggregation):
    lines = []
    for name in profile_names(cfg):
        aggregate = aggregation[name]
        if aggregate["n"] == 0:
            lines.append(
                f"- `{cell(name)}`: нет канонических прогонов (`ok` + fixed ladder). / "
                "no canonical runs (`ok` + fixed ladder)."
            )
        else:
            step_limited = [
                step for step in aggregate["steps"] if step["limited"]
            ]
            if step_limited:
                details = ", ".join(
                    f"{step['target_pct']} % (n={step['n']})" for step in step_limited
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
    for run in runs:
        if run["mode"] != "ladder" or run_generation_status(run) != "fixed":
            continue
        if run["status"] == "ok":
            continue
        lines.append(
            f"- `{cell(run['run_id'])}` (`{cell(run['run_path'])}`, profile "
            f"`{cell(run['profile'])}`): fixed ladder excluded from the aggregate, "
            f"status `{cell(run['status'])}`."
        )
    if not lines:
        lines.append("Нет ограниченных агрегатов и исключённых fixed-прогонов. / "
                     "No limited aggregates or excluded fixed runs.")
    return lines


def render_variable_runs(runs):
    variable_runs = [run for run in runs if run_generation_status(run) == "variable"]
    lines = [
        "Прогоны ниже имеют переменную длину генерации "
        "(`completion_is_fixed != true`): генерация останавливалась сама "
        "(`finish_reason = stop`) на разной длине, поэтому **decode tok/s "
        "несопоставим** между прогонами. Они сохраняются для prefill/cache и "
        "раздела контроля, но не входят в агрегированную лесенку. / Runs with "
        "variable generation length; decode is not comparable; kept for "
        "prefill/cache only.",
        "",
        "| run_id | path | profile | mode | order | step | target_pct "
        "| cache_hit_fraction | completion_tokens | finish_reason | prefill tok/s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in variable_runs:
        for step in run["steps"]:
            lines.append("| " + " | ".join([
                cell(run["run_id"]),
                cell(run["run_path"]),
                cell(run["profile"]),
                cell(run["mode"]),
                cell(run["order"]),
                integer(step.get("step")),
                integer(step.get("target_pct")),
                number(step.get("cache_hit_fraction"), digits=4),
                integer(step.get("completion_tokens")),
                cell(step.get("finish_reason")),
                number(step.get("prefill_tokens_per_second")),
            ]) + " |")
    if not variable_runs:
        lines.append("| " + " | ".join([MISSING] * 11) + " |")
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
    lines += render_ladder(aggregation, cfg, main)
    lines += [
        "",
        "## Контрольные профили / Control profiles",
        "",
        "Профили с ролью `control` (обход/контроль). На графиках они рисуются "
        "пунктиром; в основную серию не входят. / Profiles with role `control` "
        "(workaround/control); dashed on the figures; not part of the main series.",
        "",
    ]
    lines += render_ladder(aggregation, cfg, control)
    lines += [
        "",
        "## Ограниченные агрегаты и исключённые прогоны "
        "/ Limited aggregates and excluded runs",
        "",
    ]
    lines += render_limited_and_excluded(runs, cfg, aggregation)
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
        "Скорости взяты из timings ответа сервера; `cache_hit_fraction = "
        "1 - evaluated/prompt`. Оценка качества не выполняется. Режимы `smoke` "
        "в агрегированную лесенку не попадают.",
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
                }
                for name, aggregate in aggregation.items()
            },
        },
        "ab_orders": list(cfg["ab_orders"]),
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
