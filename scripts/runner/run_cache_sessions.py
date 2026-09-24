#!/usr/bin/env python3
"""Run context-ladder and A/B cache-reuse measurements.

Single-slot llama-server sessions at a long context: the `ladder` mode grows
one conversation cumulatively up to 80% of the configured context, the
`ab-sequence` mode alternates two sessions by a caller-supplied order string to
show how prompt caching behaves when two contexts share one slot. The `smoke`
mode starts the server, validates `/props`, issues one tiny chat request and
stops the server, recording GPU snapshots and parsed server.log markers.

Before every server start the runner may drop a GPU frequency lock with
`nvidia-smi -i <devices> -rgc` and record the clocks before/after. The device
list is taken from `--clock-reset-devices` or the profile key
`clock_reset_devices`; when neither is set the clocks are left untouched and the
skip is recorded in `result.json`. This happens because an external GPU
power-management service may lock GPU clocks to a low value while requests
bypass that service.

Only the standard library is required. Host/GPU telemetry (schema version 1,
METHODOLOGY section 10) and the llama-server process memory are sampled every
0.5 s into `telemetry.csv`; each run writes `result.json`. A run is preserved on
failure: the status is `failed` or `timeout`, and a run whose invariants break
(`/props` mismatch, host offload) is `non_comparable` with the reasons in
`status_reasons`.

`--profile` is resolved against `--case-dir` first and against the current
working directory second; a relative path found in neither is a clear error.

Usage:
    python3 run_cache_sessions.py --profile profiles/model.json --mode smoke
    python3 run_cache_sessions.py --profile profiles/model.json --mode ladder \
        --case-dir . --repo-root ..
    python3 run_cache_sessions.py --profile profiles/model.json \
        --mode ab-sequence --order ABBA
"""

import argparse
import csv
import datetime as dt
import itertools
import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_TIMEOUT = "timeout"
STATUS_NON_COMPARABLE = "non_comparable"

STEP_TIMEOUT_S = 7200
HEALTH_TIMEOUT_S = 900
TELEMETRY_INTERVAL_S = 0.5
TELEMETRY_SCHEMA_VERSION = 1
LADDER_PCTS = (10, 30, 60, 80)
SMOKE_MAX_TOKENS = 32
SMOKE_PROMPT = "2+2"
# Host buffers at or below this (MiB) are rounding noise and count as zero.
OFFLOAD_NONZERO_MIB = 0.01
CLOCKS_QUERY = ["nvidia-smi", "--query-gpu=index,clocks.current.graphics,pstate", "--format=csv,noheader,nounits"]
GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,temperature.gpu,utilization.gpu,memory.used,power.draw,clocks.sm,pstate",
    "--format=csv,noheader,nounits",
]
GPU_VALUE_COUNT = 6
SMOKE_GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,name,memory.used,clocks.current.graphics,pstate",
    "--format=csv",
]
SMOKE_GPU_FIELDS = ("index", "name", "memory.used", "clocks.current.graphics", "pstate")
LOG_MARKER_PATTERN = re.compile(
    r"error|warn|unknown|missing|not supported|unsupported|skipping",
    re.IGNORECASE,
)
LOG_BUFFER_PATTERNS = (
    ("KV self size", re.compile(r"KV self size\s*=\s*(?P<value>[^\n,]+)")),
    ("KV buffer size", re.compile(r"KV buffer size\s*=\s*(?P<value>[^\n,]+)")),
    ("model buffer size", re.compile(
        r"(?<!CPU_Mapped )(?<!CUDA[0-9] )model buffer size\s*=\s*(?P<value>[^\n,]+)"
    )),
    ("compute buffer size", re.compile(r"compute buffer size\s*=\s*(?P<value>[^\n,]+)")),
    ("RS buffer size", re.compile(r"RS buffer size\s*=\s*(?P<value>[^\n,]+)")),
    ("CPU_Mapped model buffer size", re.compile(r"CPU_Mapped model buffer size\s*=\s*(?P<value>[^\n,]+)")),
    ("CUDA model buffer size", re.compile(r"CUDA[0-9]+ model buffer size\s*=\s*(?P<value>[^\n,]+)")),
    ("n_ctx_per_seq", re.compile(r"n_ctx_per_seq\s*[=(:]?\s*(?P<value>\d+)")),
    ("n_ctx", re.compile(r"\bn_ctx\s*=\s*(?P<value>\d+)")),
    ("flash_attn", re.compile(r"flash_attn\s*=\s*(?P<value>\S+)")),
)
# `offloaded X/Y layers to GPU` with X < Y is the direct partial-offload marker.
OFFLOAD_LAYERS_PATTERN = re.compile(
    r"offloaded\s+(?P<offloaded>\d+)/(?P<total>\d+)\s+layers to GPU"
)
# KV on a CPU/host device is a host-KV marker; `CUDA_Host` output/compute buffers
# are normal staging and deliberately not matched here.
HOST_KV_PATTERN = re.compile(
    r"(?:CUDA_Host|CPU[0-9]?)\s+KV buffer size\s*=\s*(?P<value>[^\n,]+)"
)
# A plain `CPU model buffer size` (not the mmap-backed `CPU_Mapped` line) means
# weights actually held on the CPU.
HOST_MODEL_PATTERN = re.compile(
    r"CPU[0-9]?\s+model buffer size\s*=\s*(?P<value>[^\n,]+)"
)
# `CPU_Mapped model buffer size` is mmap-backed weight storage, not offload.
CPU_MAPPED_MODEL_PATTERN = re.compile(
    r"CPU_Mapped model buffer size\s*=\s*(?P<value>[^\n,]+)"
)
PEVAL_PATTERN = re.compile(
    r"prompt eval time =\s*(?P<milliseconds>[\d.]+) ms /\s*(?P<tokens>\d+) tokens "
    r"\([^,]+,\s*(?P<tokens_per_second>[\d.]+) tokens per second\)"
)
EVAL_PATTERN = re.compile(
    r"(?<!prompt )eval time =\s*(?P<milliseconds>[\d.]+) ms /\s*(?P<tokens>\d+) tokens "
    r"\([^,]+,\s*(?P<tokens_per_second>[\d.]+) tokens per second\)"
)
FILLER_UNIT = (
    "Измерительный текст для последовательного расширения одного диалога: "
    "сохраняй порядок фактов и состояние контекста, не сокращай и не "
    "пересказывай предыдущие фрагменты.\n"
)
SESSION_MARKERS = {"A": "Сессия A: начало приватного контекста.\n", "B": "Сессия B: начало приватного контекста.\n"}
_RUN_ID_COUNTER = itertools.count()


def classify_status(exc):
    """Map an exception to a run status: timeouts vs everything else."""
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return STATUS_TIMEOUT
    return STATUS_FAILED


def invariant_reason(code, detail):
    """Structured status reason for an invariant violation (non-comparable)."""
    return {"kind": "invariant", "code": code, "detail": detail}


def exception_reason(code, detail):
    """Structured status reason for an exception that ended the run."""
    return {"kind": "exception", "code": code, "detail": detail}


def mark_non_comparable(report, reasons):
    """Append non-comparability reasons; downgrade ok -> non_comparable only."""
    for reason in reasons:
        report["status_reasons"].append(reason)
        report["invariants_failed"] = True
        report["non_comparable_was_flagged"] = True
    if reasons and report["status"] == STATUS_OK:
        report["status"] = STATUS_NON_COMPARABLE


def step_comparable(entry, prefix_ok):
    """A step is comparable only when decoding was full-length and in order."""
    return (
        entry.get("completion_is_fixed") is True
        and entry.get("finish_reason") == "length"
        and prefix_ok is True
    )


def timing_delta_pct(api_value, log_value):
    """Percent divergence of an API timing from the log timing; None if absent."""
    if api_value is None or log_value is None or log_value == 0:
        return None
    return round(100.0 * (api_value - log_value) / log_value, 3)


def canonical_speed_from_sources(api_value, log_value, api_tokens, log_tokens):
    """Pick the canonical speed and its source for one timing metric.

    The raw server.log timing is canonical (METHODOLOGY section 8). It is
    accepted when its token count matches this request's own token count
    (`confirmed=True`); when the API exposes no token count there is nothing to
    cross-check against, so the log is still accepted as primary
    (`confirmed=False`). A log block whose token count differs from a known API
    token count is a foreign/previous block and is rejected as a mismatch, in
    which case the API timing is the fallback. Returns {"value", "source",
    "mismatch", "confirmed"}, where `source` is "log", "api" or None.
    """
    if log_value is not None and log_tokens is not None:
        if api_tokens is None:
            return {"value": log_value, "source": "log", "mismatch": False, "confirmed": False}
        if api_tokens == log_tokens:
            return {"value": log_value, "source": "log", "mismatch": False, "confirmed": True}
        if api_value is not None:
            return {"value": api_value, "source": "api", "mismatch": True, "confirmed": False}
        return {"value": None, "source": None, "mismatch": True, "confirmed": False}
    if api_value is not None:
        return {"value": api_value, "source": "api", "mismatch": False, "confirmed": False}
    return {"value": None, "source": None, "mismatch": False, "confirmed": False}


def request_json(base_url, path, payload=None, timeout=STEP_TIMEOUT_S):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def token_count(base_url, text):
    return len(request_json(base_url, "/tokenize", {"content": text}, timeout=900)["tokens"])


def wait_ready(base_url, process, timeout_s=HEALTH_TIMEOUT_S):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with status {process.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise TimeoutError(f"server did not become ready at {base_url} within {timeout_s}s")


def stop_process(process, graceful_timeout_s=120):
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=graceful_timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=30)
        return
    except (subprocess.TimeoutExpired, ProcessLookupError):
        pass
    if process.poll() is not None:
        return
    process.kill()
    process.wait(timeout=20)


def proc_memory_kib(pid):
    values = {}
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"Rss", "Pss", "Swap"}:
                values[key.lower()] = int(value.strip().split()[0])
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return values


def find_llama_server_pids():
    """All live llama-server PIDs found via /proc/<pid>/exe."""
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            target = os.readlink(entry / "exe")
        except OSError:
            continue
        if os.path.basename(target) == "llama-server":
            pids.append(int(entry.name))
    return sorted(pids)


def aggregate_llama_memory(pids):
    """Sum RSS/PSS/swap over every llama-server process."""
    totals = {"rss": 0, "pss": 0, "swap": 0}
    observed = False
    for pid in pids:
        values = proc_memory_kib(pid)
        if not values:
            continue
        observed = True
        for key in totals:
            totals[key] += values.get(key, 0)
    if not observed:
        return {"rss": "", "pss": "", "swap": ""}
    return totals


def host_cpu_ticks():
    try:
        fields = Path("/proc/stat").read_text().splitlines()[0].split()
        values = [int(value) for value in fields[1:]]
        return sum(values), values[3] + values[4]
    except (IndexError, ValueError):
        return 0, 0


def host_cpu_per_core_ticks():
    """Return a list of (total, idle) tick pairs, one per CPU core."""
    cores = []
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        if not fields or not re.fullmatch(r"cpu\d+", fields[0]):
            continue
        try:
            values = [int(value) for value in fields[1:]]
        except ValueError:
            continue
        cores.append((sum(values), values[3] + values[4]))
    return cores


def cpu_core_fields():
    """Column names for the per-core CPU load, matching /proc/stat order."""
    count = len(host_cpu_per_core_ticks()) or (os.cpu_count() or 0)
    return [f"cpu{index}_pct" for index in range(count)]


def host_memory_kib():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            values[key] = int(value.strip().split()[0])
    return values


def cpu_temp_c():
    """Return (temperature_c, thermal_zone_name) for the warmest valid zone."""
    best = None
    best_zone = ""
    base = Path("/sys/class/thermal")
    if not base.is_dir():
        return "", ""
    for zone in sorted(base.iterdir()):
        if not zone.name.startswith("thermal_zone"):
            continue
        try:
            value = int((zone / "temp").read_text().strip())
        except (OSError, ValueError):
            continue
        if value > 1000:
            value /= 1000.0
        if 0 < value < 130 and (best is None or value > best):
            best = value
            best_zone = zone.name
    return (best, best_zone) if best is not None else ("", "")


def gpu_snapshot():
    """Map GPU index -> [temp, util, mem_used_mib, power_w, sm_clock_mhz, pstate]."""
    try:
        output = subprocess.check_output(GPU_QUERY, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return {}
    gpus = {}
    for line in output.splitlines():
        row = next(csv.reader([line]))
        if len(row) < GPU_VALUE_COUNT + 1:
            continue
        gpus[row[0].strip()] = [value.strip() for value in row[1 : GPU_VALUE_COUNT + 1]]
    return gpus


def gpu_indices():
    """GPU indices in nvidia-smi order; falls back to `nvidia-smi -L`.

    An empty list means no GPU telemetry is available (nvidia-smi absent or
    failing); the collector then writes no gpu columns and never crashes.
    """
    indices = list(gpu_snapshot())
    if indices:
        return indices
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in output.splitlines():
        match = re.match(r"GPU\s+(\d+)", line.strip())
        if match and match.group(1) not in found:
            found.append(match.group(1))
    return found


def query_clocks():
    """Snapshot graphics clock and P-state for every GPU as a list of rows."""
    try:
        output = subprocess.check_output(CLOCKS_QUERY, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return [{"error": str(exc)}]
    rows = []
    for line in output.splitlines():
        row = next(csv.reader([line]))
        if len(row) < 3:
            continue
        rows.append({
            "index": row[0].strip(),
            "clocks.current.graphics": row[1].strip(),
            "pstate": row[2].strip(),
        })
    return rows


def smoke_gpu_snapshot():
    """Snapshot index/name/memory/clock/pstate with the header-bearing CSV format."""
    try:
        output = subprocess.check_output(SMOKE_GPU_QUERY, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": str(exc), "rows": []}
    rows = []
    for row in csv.reader(output.splitlines()):
        if not row or row[0].strip() == "index":
            continue
        rows.append({name: value.strip() for name, value in zip(SMOKE_GPU_FIELDS, row)})
    return {"rows": rows}


def reset_cmp_clocks(devices):
    """Drop the frequency lock on `devices`; ignore absent/unavailable GPUs."""
    command = ["nvidia-smi", "-i", str(devices), "-rgc"]
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": command, "returncode": None, "error": str(exc)}
    output = (result.stdout or "") + (result.stderr or "")
    return {
        "command": command,
        "returncode": result.returncode,
        "output": output.strip(),
    }


def collect_clock_reset(devices, prefix="clocks"):
    """Run the frequency-lock reset and record the clocks around it once.

    When no device list is configured the reset is skipped and the skip is
    recorded, so the clocks are never touched implicitly.
    """
    if not devices:
        return {
            f"{prefix}_reset_skipped": (
                "no clock_reset_devices configured; GPU clocks were left untouched"
            ),
        }
    return {
        f"{prefix}_before_reset": query_clocks(),
        f"{prefix}_reset": reset_cmp_clocks(devices),
        f"{prefix}_after_reset": query_clocks(),
    }


def _clean_buffer_value(value):
    """Return a size in MiB when the value carries a byte unit, else as-is."""
    value = value.strip().strip(",").strip()
    match = re.match(r"(?P<number>[\d.]+)\s*(?P<unit>MiB|KiB|GiB)", value)
    if match:
        number = float(match.group("number"))
        unit = match.group("unit")
        if unit == "KiB":
            number /= 1024.0
        elif unit == "GiB":
            number *= 1024.0
        return round(number, 3)
    return value


def parse_server_log(text):
    """Extract buffer sizes and warning/error markers from a server.log."""
    buffers = {}
    sizes = {}
    for key, pattern in LOG_BUFFER_PATTERNS:
        matches = [match.group("value") for match in pattern.finditer(text)]
        if not matches:
            continue
        raw_value = matches[-1].strip()
        sizes[key] = raw_value
        buffers[key] = _clean_buffer_value(raw_value)
    markers = []
    for line in text.splitlines():
        if not LOG_MARKER_PATTERN.search(line):
            continue
        stripped = line.strip()
        if not stripped:
            continue
        markers.append(stripped)
    result = {
        "sizes": sizes,
        "markers": markers,
        "marker_count": len(markers),
        "log_buffers": buffers,
    }
    if not buffers:
        result["log_buffers_note"] = (
            "no buffer-size lines found in server.log; run smoke with --verbose"
        )
    return result


def snapshot_props(base_url):
    props = request_json(base_url, "/props", timeout=60)
    chat_template = props.get("chat_template")
    settings = props.get("default_generation_settings") or {}
    return {
        "n_ctx": props.get("n_ctx") or settings.get("n_ctx"),
        "total_slots": props.get("total_slots"),
        "model_path": props.get("model_path"),
        "chat_template_present": bool(chat_template),
        "chat_template_length": len(chat_template) if isinstance(chat_template, str) else None,
        "default_generation_settings_n_ctx": settings.get("n_ctx"),
        "raw_keys": sorted(props.keys()),
    }


def check_props(props, ctx_size, parallel):
    """Compare /props against the profile; a mismatch makes a run non-comparable."""
    actual_n_ctx = props.get("n_ctx")
    actual_total_slots = props.get("total_slots")
    n_ctx_ok = actual_n_ctx == ctx_size
    total_slots_ok = actual_total_slots == parallel
    reasons = []
    if not n_ctx_ok:
        reasons.append(invariant_reason(
            "props_n_ctx_mismatch",
            f"props n_ctx {actual_n_ctx!r} != profile ctx_size {ctx_size!r}",
        ))
    if not total_slots_ok:
        reasons.append(invariant_reason(
            "props_total_slots_mismatch",
            f"props total_slots {actual_total_slots!r} != profile parallel {parallel!r}",
        ))
    return {
        "ok": n_ctx_ok and total_slots_ok,
        "n_ctx": {"actual": actual_n_ctx, "expected": ctx_size, "ok": n_ctx_ok},
        "total_slots": {
            "actual": actual_total_slots,
            "expected": parallel,
            "ok": total_slots_ok,
        },
        "props": props,
        "reasons": reasons,
    }


def _nonzero_mib(raw_value):
    """True when a buffer-size string is above the rounding-noise floor."""
    cleaned = _clean_buffer_value(raw_value)
    try:
        number = float(cleaned)
    except (TypeError, ValueError):
        return False
    return number > OFFLOAD_NONZERO_MIB


def check_offload(log_text):
    """Detect actual host offload from a server.log; non-comparable if found.

    `offloaded X/Y layers to GPU` with X < Y is by itself an invariant
    violation: the layers below Y keep their weights outside the GPU. CPU-held
    model/KV buffers (`CPU model buffer size`, host `KV buffer size`) are
    independent signals of offload. `CPU_Mapped model buffer size` is
    mmap-backed weight storage and is recorded for information only, never a
    reason. `CUDA_Host` output/compute buffers are normal staging.
    """
    parsed = parse_server_log(log_text)
    buffers = parsed.get("log_buffers") or {}
    mapped_matches = CPU_MAPPED_MODEL_PATTERN.findall(log_text)
    mapped_raw = mapped_matches[-1].strip() if mapped_matches else None
    mapped_mib = _clean_buffer_value(mapped_raw) if mapped_raw is not None else None
    partial_layers = [
        f"{match.group('offloaded')}/{match.group('total')}"
        for match in OFFLOAD_LAYERS_PATTERN.finditer(log_text)
        if int(match.group("offloaded")) < int(match.group("total"))
    ]
    host_kv_values = [value.strip().strip(",") for value in HOST_KV_PATTERN.findall(log_text)]
    host_model_values = [value.strip().strip(",") for value in HOST_MODEL_PATTERN.findall(log_text)]
    host_kv = any(_nonzero_mib(value) for value in host_kv_values)
    host_model = any(_nonzero_mib(value) for value in host_model_values)
    reasons = []
    if partial_layers:
        reasons.append(invariant_reason(
            "partial_layer_offload",
            f"offloaded layers below total: {partial_layers}",
        ))
    if host_model:
        reasons.append(invariant_reason(
            "host_model_buffer",
            f"CPU model buffer size present: {host_model_values}",
        ))
    if host_kv:
        reasons.append(invariant_reason(
            "host_kv_buffer",
            f"host KV buffer size present: {host_kv_values}",
        ))
    return {
        "ok": not reasons,
        "cpu_mapped_mib": mapped_mib,
        "partial_layer_offloads": partial_layers,
        "host_kv_buffer": host_kv,
        "host_model_buffer": host_model,
        "log_buffers": buffers,
        "reasons": reasons,
    }


def run_smoke(base_url, profile):
    model = profile["served_model_name"]
    props = snapshot_props(base_url)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": SMOKE_PROMPT}],
        "max_tokens": SMOKE_MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
        # Generate exactly max_tokens so decode tok/s is measured on a comparable sample.
        "ignore_eos": True,
    }
    started = time.perf_counter()
    response = request_json(base_url, "/v1/chat/completions", payload, timeout=STEP_TIMEOUT_S)
    elapsed = time.perf_counter() - started
    choices = response.get("choices") or [{}]
    message = choices[0].get("message") or {}
    return {
        "props": props,
        "request": payload,
        "content": message.get("content"),
        "finish_reason": choices[0].get("finish_reason"),
        "usage": response.get("usage"),
        "timings": response.get("timings"),
        "elapsed_s": round(elapsed, 3),
    }


def telemetry_loop(output, server_pid, stop_event):
    gpu_ids = gpu_indices()
    fields = [
        "schema_version", "timestamp_utc", "cpu_pct", "cpu_temp_c", "cpu_temp_source",
        "ram_used_gib", "ram_total_gib", "swap_used_gib",
    ]
    fields += cpu_core_fields()
    for index in gpu_ids:
        fields += [
            f"gpu{index}_temp_c",
            f"gpu{index}_util_pct",
            f"gpu{index}_mem_used_mib",
            f"gpu{index}_power_w",
            f"gpu{index}_sm_clock_mhz",
            f"gpu{index}_pstate",
        ]
    fields += ["server_rss_kib", "server_pss_kib", "server_swap_kib", "llama_pids"]
    previous_host = host_cpu_ticks()
    previous_cores = host_cpu_per_core_ticks()
    warned_indices = set()
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        while not stop_event.is_set():
            started = time.monotonic()
            host_total, host_idle = host_cpu_ticks()
            old_total, old_idle = previous_host
            total_delta = host_total - old_total
            cpu_pct = 100.0 * (total_delta - (host_idle - old_idle)) / total_delta if total_delta > 0 else 0.0
            previous_host = (host_total, host_idle)
            cores = host_cpu_per_core_ticks()
            host = host_memory_kib()
            total_kib = host.get("MemTotal", 0)
            available_kib = host.get("MemAvailable", 0)
            temp_c, temp_zone = cpu_temp_c()
            row = {
                "schema_version": TELEMETRY_SCHEMA_VERSION,
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                "cpu_pct": f"{cpu_pct:.1f}",
                "cpu_temp_c": temp_c,
                "cpu_temp_source": temp_zone,
                "ram_used_gib": round((total_kib - available_kib) / 1024 / 1024, 2),
                "ram_total_gib": round(total_kib / 1024 / 1024, 2),
                "swap_used_gib": round((host.get("SwapTotal", 0) - host.get("SwapFree", 0)) / 1024 / 1024, 2),
            }
            for index, (core_total, core_idle) in enumerate(cores):
                old_core = previous_cores[index] if index < len(previous_cores) else (core_total, core_idle)
                delta = core_total - old_core[0]
                idle_delta = core_idle - old_core[1]
                core_pct = 100.0 * (delta - idle_delta) / delta if delta > 0 else 0.0
                row[f"cpu{index}_pct"] = f"{core_pct:.1f}"
            previous_cores = cores
            gpus = gpu_snapshot()
            for index in gpu_ids:
                values = gpus.get(index, [""] * GPU_VALUE_COUNT)
                row[f"gpu{index}_temp_c"] = values[0]
                row[f"gpu{index}_util_pct"] = values[1]
                row[f"gpu{index}_mem_used_mib"] = values[2]
                row[f"gpu{index}_power_w"] = values[3]
                row[f"gpu{index}_sm_clock_mhz"] = values[4]
                row[f"gpu{index}_pstate"] = values[5]
            for index in gpus:
                if index not in gpu_ids and index not in warned_indices:
                    warned_indices.add(index)
                    print(
                        f"warning: GPU {index} appeared after start; not in telemetry header",
                        flush=True,
                    )
            pids = find_llama_server_pids()
            if server_pid not in pids:
                pids = sorted(set(pids + [server_pid]))
            memory = aggregate_llama_memory(pids)
            row["server_rss_kib"] = memory.get("rss", "")
            row["server_pss_kib"] = memory.get("pss", "")
            row["server_swap_kib"] = memory.get("swap", "")
            row["llama_pids"] = " ".join(str(pid) for pid in pids)
            writer.writerow(row)
            file.flush()
            time.sleep(max(0.0, TELEMETRY_INTERVAL_S - (time.monotonic() - started)))


def make_text(base_url, prefix, target_tokens):
    """Build a deterministic text of at most target_tokens tokens.

    The text is `prefix` followed by repetitions of FILLER_UNIT, trimmed by a
    binary search on the byte length so the token count never exceeds the
    target. Since every later prompt is built from the same prefix and unit, it
    naturally extends the earlier one.
    """
    if target_tokens <= 0:
        text = prefix
        return text, token_count(base_url, text)
    low, high = 0, max(1, target_tokens * len(FILLER_UNIT))
    while low < high:
        middle = (low + high + 1) // 2
        candidate = prefix + (FILLER_UNIT * (middle // len(FILLER_UNIT) + 1))[:middle]
        if token_count(base_url, candidate) <= target_tokens:
            low = middle
        else:
            high = middle - 1
    text = prefix + (FILLER_UNIT * (low // len(FILLER_UNIT) + 1))[:low]
    return text, token_count(base_url, text)


def extract_metrics(response, max_tokens=None):
    usage = response.get("usage") or {}
    timings = response.get("timings") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens") or timings.get("predicted_n")
    completion_is_fixed = None if max_tokens is None else completion_tokens == max_tokens
    evaluated_tokens = timings.get("prompt_n")
    if evaluated_tokens is None:
        evaluated_tokens = response.get("tokens_evaluated")
    prefill_tps = timings.get("prompt_per_second")
    if prefill_tps is None and evaluated_tokens and timings.get("prompt_ms"):
        prefill_tps = evaluated_tokens / (timings["prompt_ms"] / 1000.0)
    decode_tps = timings.get("predicted_per_second")
    if decode_tps is None and completion_tokens and timings.get("predicted_ms"):
        decode_tps = completion_tokens / (timings["predicted_ms"] / 1000.0)
    cache_hit = None
    if evaluated_tokens is not None and prompt_tokens:
        cache_hit = max(0.0, 1.0 - evaluated_tokens / prompt_tokens)
    cached_tokens_api = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    cache_n_timings = timings.get("cache_n")
    cache_hit_api = None
    if cached_tokens_api is not None and prompt_tokens:
        cache_hit_api = max(0.0, min(1.0, cached_tokens_api / prompt_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "completion_tokens_expected": max_tokens,
        "completion_is_fixed": completion_is_fixed,
        "evaluated_tokens": evaluated_tokens,
        "cache_hit_fraction": round(cache_hit, 4) if cache_hit is not None else None,
        "cached_tokens_api": cached_tokens_api,
        "cache_n_timings": cache_n_timings,
        "cache_hit_fraction_api": round(cache_hit_api, 4) if cache_hit_api is not None else None,
        "prefill_tokens_per_second": round(prefill_tps, 3) if prefill_tps is not None else None,
        "decode_tokens_per_second": round(decode_tps, 3) if decode_tps is not None else None,
    }


def parse_log_timings(text):
    """Extract the last prompt-eval/eval timing block from a server.log tail."""
    result = {}
    for match in PEVAL_PATTERN.finditer(text):
        result["prompt_eval_tokens"] = int(match.group("tokens"))
        result["prompt_eval_tokens_per_second"] = float(match.group("tokens_per_second"))
    for match in EVAL_PATTERN.finditer(text):
        result["eval_tokens"] = int(match.group("tokens"))
        result["eval_tokens_per_second"] = float(match.group("tokens_per_second"))
    return result


def run_request(base_url, model, prompt, max_tokens, seed, log_path, cache_prompt=True):
    """Issue one chat request and attach canonical (log-first) timing metrics.

    The canonical `prefill_tokens_per_second`/`decode_tokens_per_second` come
    from the raw server.log block when its token count matches this request's
    own token count, or when the API exposes no token count to cross-check
    (METHODOLOGY section 8); a known mismatch falls back to the API timing.
    Both sources are kept explicitly and `timings_source` records which one is
    canonical and whether the log was confirmed by an independent API count.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "cache_prompt": cache_prompt,
        "stream": False,
        # Generate exactly max_tokens so decode tok/s is measured on a comparable sample.
        "ignore_eos": True,
    }
    offset = log_path.stat().st_size if log_path.exists() else 0
    started = time.perf_counter()
    started_utc = dt.datetime.now(dt.timezone.utc)
    response = request_json(base_url, "/v1/chat/completions", payload, timeout=STEP_TIMEOUT_S)
    finished_utc = dt.datetime.now(dt.timezone.utc)
    elapsed = time.perf_counter() - started
    entry = extract_metrics(response, max_tokens)
    entry["started_at_utc"] = started_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    entry["finished_at_utc"] = finished_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    entry["elapsed_s"] = round(elapsed, 3)
    choices = response.get("choices") or [{}]
    entry["finish_reason"] = choices[0].get("finish_reason")
    timings = response.get("timings") or {}
    time.sleep(0.4)
    with log_path.open() as handle:
        handle.seek(offset)
        log_timings = parse_log_timings(handle.read())
    api_prefill = timings.get("prompt_per_second")
    if api_prefill is None and entry["evaluated_tokens"] and timings.get("prompt_ms"):
        api_prefill = entry["evaluated_tokens"] / (timings["prompt_ms"] / 1000.0)
    api_decode = timings.get("predicted_per_second")
    if api_decode is None and entry["completion_tokens"] and timings.get("predicted_ms"):
        api_decode = entry["completion_tokens"] / (timings["predicted_ms"] / 1000.0)
    log_prefill = log_timings.get("prompt_eval_tokens_per_second")
    log_decode = log_timings.get("eval_tokens_per_second")
    log_prefill_tokens = log_timings.get("prompt_eval_tokens")
    log_decode_tokens = log_timings.get("eval_tokens")
    # Capture the API token counts before the log backfill below, so the
    # canonical-source guard compares like with like (METHODOLOGY section 8):
    # the log is primary, confirmed only when an independent API count matches.
    api_eval_tokens = entry["evaluated_tokens"]
    api_completion_tokens = entry["completion_tokens"]
    prefill = canonical_speed_from_sources(
        api_prefill,
        log_prefill,
        api_eval_tokens,
        log_prefill_tokens,
    )
    decode = canonical_speed_from_sources(
        api_decode,
        log_decode,
        api_completion_tokens,
        log_decode_tokens,
    )
    entry["prefill_tokens_per_second"] = (
        round(prefill["value"], 3) if prefill["value"] is not None else None
    )
    entry["decode_tokens_per_second"] = (
        round(decode["value"], 3) if decode["value"] is not None else None
    )
    entry["prefill_tokens_per_second_api"] = round(api_prefill, 3) if api_prefill is not None else None
    entry["prefill_tokens_per_second_log"] = round(log_prefill, 3) if log_prefill is not None else None
    entry["decode_tokens_per_second_api"] = round(api_decode, 3) if api_decode is not None else None
    entry["decode_tokens_per_second_log"] = round(log_decode, 3) if log_decode is not None else None
    entry["timings_source"] = {
        "prefill": prefill["source"],
        "decode": decode["source"],
        "confirmed": {"prefill": prefill["confirmed"], "decode": decode["confirmed"]},
    }
    entry["timing_delta_pct"] = {
        "prefill": timing_delta_pct(api_prefill, log_prefill),
        "decode": timing_delta_pct(api_decode, log_decode),
    }
    entry["log_timing_mismatch"] = prefill["mismatch"] or decode["mismatch"]
    entry["log_timing_mismatch_details"] = {
        "prefill": {
            "mismatch": prefill["mismatch"],
            "expected_tokens": api_eval_tokens if api_eval_tokens is not None else log_prefill_tokens,
            "log_tokens": log_prefill_tokens,
        },
        "decode": {
            "mismatch": decode["mismatch"],
            "expected_tokens": api_completion_tokens if api_completion_tokens is not None else log_decode_tokens,
            "log_tokens": log_decode_tokens,
        },
    }
    entry["client_ttft_s"] = None
    prompt_ms = timings.get("prompt_ms")
    predicted_ms = timings.get("predicted_ms")
    if prompt_ms is not None and predicted_ms is not None:
        entry["overhead_s"] = round(elapsed - (prompt_ms + predicted_ms) / 1000.0, 3)
    else:
        entry["overhead_s"] = None
    if log_timings:
        if entry["evaluated_tokens"] is None and "prompt_eval_tokens" in log_timings:
            entry["evaluated_tokens"] = log_timings["prompt_eval_tokens"]
            if entry["prompt_tokens"]:
                entry["cache_hit_fraction"] = round(
                    max(0.0, 1.0 - entry["evaluated_tokens"] / entry["prompt_tokens"]), 4
                )
        entry["log_timings"] = log_timings
    return entry


def build_targets(ctx_size, ladder_pcts=None):
    pcts = LADDER_PCTS if ladder_pcts is None else tuple(ladder_pcts)
    return [{"step": 0, "target_pct": 0, "target_tokens": 0}] + [
        {"step": index, "target_pct": pct, "target_tokens": int(ctx_size * pct / 100)}
        for index, pct in enumerate(pcts, start=1)
    ]


def run_ladder(base_url, profile, targets, seed, log_path, records, cache_prompt=True):
    model = profile["served_model_name"]
    max_tokens = int(profile.get("n_predict", 128))
    prompt = "Привет"
    prompt_tokens = token_count(base_url, prompt)
    previous_prompt = None
    for target in targets:
        if target["target_tokens"] > 0:
            prompt, prompt_tokens = make_text(base_url, "Привет", target["target_tokens"])
        prefix_ok = True if previous_prompt is None else prompt.startswith(previous_prompt)
        entry = run_request(base_url, model, prompt, max_tokens, seed, log_path, cache_prompt)
        entry.update({
            "step": target["step"],
            "target_pct": target["target_pct"],
            "target_tokens": target["target_tokens"],
            "raw_prompt_tokens": prompt_tokens,
            "prompt_prefix_ok": prefix_ok,
        })
        entry["step_comparable"] = step_comparable(entry, prefix_ok)
        previous_prompt = prompt
        records.append(entry)
        print(
            f"[ladder] step {entry['step']} pct={entry['target_pct']} "
            f"prompt={entry['prompt_tokens']} eval={entry['evaluated_tokens']} "
            f"cache={entry['cache_hit_fraction']} "
            f"prefill={entry['prefill_tokens_per_second']} "
            f"decode={entry['decode_tokens_per_second']}",
            flush=True,
        )
    return records


def run_ab_sequence(base_url, profile, targets, order, seed, log_path, records, cache_prompt=True):
    """Interleave two sessions, one ladder rung per visit, following `order`.

    `order` is a repeating pattern (e.g. `ABAB`, `ABBABAA`); it is cycled
    until both sessions have consumed every rung of the shared ladder.
    """
    model = profile["served_model_name"]
    max_tokens = int(profile.get("n_predict", 128))
    sessions = sorted(set(order))
    if sessions != ["A", "B"]:
        raise SystemExit(f"--order must contain exactly sessions A and B, got {order!r}")
    prompts = {
        "A": SESSION_MARKERS["A"] + "Привет",
        "B": SESSION_MARKERS["B"] + "Привет",
    }
    prompt_tokens = {"A": token_count(base_url, prompts["A"]), "B": token_count(base_url, prompts["B"])}
    previous_prompts = {"A": None, "B": None}
    counters = {"A": 0, "B": 0}
    for order_index, session in enumerate(itertools.cycle(order)):
        if all(counters[name] >= len(targets) for name in sessions):
            break
        if counters[session] >= len(targets):
            continue
        target = targets[counters[session]]
        counters[session] += 1
        if target["target_tokens"] > 0:
            prompts[session], prompt_tokens[session] = make_text(
                base_url, SESSION_MARKERS[session] + "Привет", target["target_tokens"]
            )
        earlier = previous_prompts[session]
        prefix_ok = True if earlier is None else prompts[session].startswith(earlier)
        entry = run_request(base_url, model, prompts[session], max_tokens, seed, log_path, cache_prompt)
        entry.update({
            "order_index": order_index,
            "session": session,
            "step": target["step"],
            "target_pct": target["target_pct"],
            "target_tokens": target["target_tokens"],
            "raw_prompt_tokens": prompt_tokens[session],
            "prompt_prefix_ok": prefix_ok,
        })
        entry["step_comparable"] = step_comparable(entry, prefix_ok)
        previous_prompts[session] = prompts[session]
        records.append(entry)
        print(
            f"[ab] {order_index} session={session} step={entry['step']} pct={entry['target_pct']} "
            f"prompt={entry['prompt_tokens']} eval={entry['evaluated_tokens']} "
            f"cache={entry['cache_hit_fraction']} "
            f"prefill={entry['prefill_tokens_per_second']} "
            f"decode={entry['decode_tokens_per_second']}",
            flush=True,
        )
    return {"order": order}


def resolve_command(command, port, n_predict, repo_root):
    resolved = []
    for item in command:
        resolved.append(item.replace("<repo>", str(repo_root)))
    for index, item in enumerate(resolved):
        if item == "--port" and index + 1 < len(resolved):
            resolved[index + 1] = str(port)
    if "--n-predict" not in resolved:
        resolved += ["--n-predict", str(n_predict)]
    return resolved


def resolve_profile_path(value, case_dir):
    """Resolve --profile: absolute as-is, relative against case_dir then cwd."""
    if value.is_absolute():
        candidate = value
        if candidate.is_file():
            return candidate.resolve()
        raise SystemExit(f"profile not found: {candidate}")
    for base in (case_dir, Path.cwd()):
        candidate = base / value
        if candidate.is_file():
            return candidate.resolve()
    raise SystemExit(
        f"profile not found: {value} (looked in {case_dir} and {Path.cwd()})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("smoke", "ladder", "ab-sequence"))
    parser.add_argument("--case-dir", type=Path, default=None, help="case directory (default: cwd)")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="root replacing <repo> in the profile command (default: --case-dir)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="where results/<run_id>/ is written (default: <case-dir>/results)",
    )
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--order", default="ABAB", help="session order for --mode ab-sequence")
    parser.add_argument(
        "--clock-reset-devices",
        default=None,
        help="comma-separated GPU indices for `nvidia-smi -i <devices> -rgc`; "
        "falls back to the profile key clock_reset_devices; when neither is set "
        "the clocks are left untouched",
    )
    args = parser.parse_args()

    case_dir = (args.case_dir or Path.cwd()).resolve()
    repo_root = (args.repo_root or case_dir).resolve()
    results_dir = (args.results_dir or (case_dir / "results")).resolve()

    profile_path = resolve_profile_path(args.profile, case_dir)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    port = args.port or int(profile.get("port", 18091))
    base_url = f"http://127.0.0.1:{port}"
    ctx_size = int(profile["ctx_size"])
    n_predict = int(profile.get("n_predict", 128))
    command = resolve_command(profile["command"], port, n_predict, repo_root)
    if args.mode == "smoke" and "--verbose" not in command:
        command = command + ["--verbose"]
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in profile.get("command_env", {}).items()})

    clock_reset_devices = args.clock_reset_devices or profile.get("clock_reset_devices")
    if clock_reset_devices is not None:
        clock_reset_devices = str(clock_reset_devices)

    run_id = (
        f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')[:-3]}"
        f"-{profile['name']}-{args.mode}-{os.getpid()}-{next(_RUN_ID_COUNTER)}"
    )
    result_dir = results_dir / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    (result_dir / "command.json").write_text(json.dumps({
        "command": command,
        "command_env": profile.get("command_env", {}),
        "profile_path": str(profile_path),
        "case_dir": str(case_dir),
        "repo_root": str(repo_root),
        "results_dir": str(results_dir),
        "clock_reset_devices": clock_reset_devices,
        "port": port,
        "seed": args.seed,
        "mode": args.mode,
        "order": args.order if args.mode == "ab-sequence" else None,
    }, indent=2, ensure_ascii=False) + "\n")
    (result_dir / "profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n")

    ladder_pcts = profile.get("ladder_pcts")
    if ladder_pcts is not None:
        ladder_pcts = [int(pct) for pct in ladder_pcts]
    targets = build_targets(ctx_size, ladder_pcts)
    cache_prompt = bool(profile.get("cache_prompt", True))
    telemetry_stop = threading.Event()
    telemetry = None
    process = None
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    report = {
        "run_id": run_id,
        "started_at": started_at,
        "status": STATUS_OK,
        "status_reasons": [],
        "invariants_failed": False,
        "non_comparable_was_flagged": False,
        "error": None,
        "profile": profile["name"],
        "profile_path": str(profile_path),
        "case_dir": str(case_dir),
        "repo_root": str(repo_root),
        "results_dir": str(results_dir),
        "clock_reset_devices": clock_reset_devices,
        "mode": args.mode,
        "model_path": profile["model_path"],
        "binary": profile["binary"],
        "build_variant": profile.get("build_variant"),
        "ctx_size": ctx_size,
        "parallel": int(profile.get("parallel", 1)),
        "cache_type_k": profile.get("cache_type_k"),
        "cache_type_v": profile.get("cache_type_v"),
        "cache_prompt": cache_prompt,
        "n_predict": int(profile.get("n_predict", 128)),
        "seed": args.seed,
        "port": port,
        "targets": targets,
        "ladder_pcts": [target["target_pct"] for target in targets[1:]],
        "telemetry_csv": "telemetry.csv",
        "steps": [],
    }
    report.update(collect_clock_reset(clock_reset_devices))

    smoke = None
    try:
        with (result_dir / "server.log").open("w") as server_log:
            process = subprocess.Popen(
                command, cwd=repo_root, env=environment, stdout=server_log, stderr=subprocess.STDOUT
            )
            telemetry = threading.Thread(
                target=telemetry_loop,
                args=(result_dir / "telemetry.csv", process.pid, telemetry_stop),
                daemon=True,
            )
            telemetry.start()
            wait_ready(base_url, process)
            if args.mode == "smoke":
                smoke = {"gpu_before": smoke_gpu_snapshot()}
                report["smoke"] = smoke
                smoke.update(run_smoke(base_url, profile))
                smoke["gpu_after"] = smoke_gpu_snapshot()
            else:
                props_check = check_props(snapshot_props(base_url), ctx_size, report["parallel"])
                report["props_check"] = props_check
                if not props_check["ok"]:
                    # Fail closed: a mismatched /props makes every later step
                    # non-comparable, so skip the heavy ladder/ab requests and
                    # keep `steps` empty; result.json still carries the reason.
                    mark_non_comparable(report, props_check["reasons"])
                    report["measurement_skipped"] = "props_check_failed"
                else:
                    if args.mode == "ladder":
                        run_ladder(
                            base_url, profile, targets, args.seed, result_dir / "server.log",
                            report["steps"], cache_prompt,
                        )
                    else:
                        report["order"] = args.order
                        run_ab_sequence(
                            base_url, profile, targets, args.order, args.seed, result_dir / "server.log",
                            report["steps"], cache_prompt,
                        )
                    log_path = result_dir / "server.log"
                    log_text = (
                        log_path.read_text(encoding="utf-8", errors="replace")
                        if log_path.exists() else ""
                    )
                    offload_check = check_offload(log_text)
                    report["offload_check"] = offload_check
                    mark_non_comparable(report, offload_check["reasons"])
    except Exception as exc:  # noqa: BLE001
        report["status"] = classify_status(exc)
        report["error"] = str(exc)
        report["status_reasons"].append(exception_reason(
            report["status"],
            f"{report['status']}: {exc}",
        ))
        raise
    finally:
        telemetry_stop.set()
        if telemetry is not None:
            telemetry.join(timeout=15)
        stop_process(process)
        if args.mode == "smoke":
            log_path = result_dir / "server.log"
            if smoke is not None:
                smoke["server_log"] = parse_server_log(
                    log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
                )
            report.update(collect_clock_reset(clock_reset_devices, prefix="clocks_after_finish"))
        report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        (result_dir / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"Results: {result_dir}", flush=True)


if __name__ == "__main__":
    main()
