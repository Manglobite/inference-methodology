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
0.5 s into `telemetry.csv`; each run writes `result.json` (failed runs are
preserved with status=failed and the error).

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
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

STEP_TIMEOUT_S = 7200
HEALTH_TIMEOUT_S = 900
TELEMETRY_INTERVAL_S = 0.5
TELEMETRY_SCHEMA_VERSION = 1
LADDER_PCTS = (10, 30, 60, 80)
SMOKE_MAX_TOKENS = 32
SMOKE_PROMPT = "2+2"
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


def run_request(base_url, model, prompt, max_tokens, seed, log_path):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "cache_prompt": True,
        "stream": False,
        # Generate exactly max_tokens so decode tok/s is measured on a comparable sample.
        "ignore_eos": True,
    }
    offset = log_path.stat().st_size if log_path.exists() else 0
    started = time.perf_counter()
    response = request_json(base_url, "/v1/chat/completions", payload, timeout=STEP_TIMEOUT_S)
    elapsed = time.perf_counter() - started
    entry = extract_metrics(response, max_tokens)
    entry["elapsed_s"] = round(elapsed, 3)
    choices = response.get("choices") or [{}]
    entry["finish_reason"] = choices[0].get("finish_reason")
    time.sleep(0.4)
    with log_path.open() as handle:
        handle.seek(offset)
        log_timings = parse_log_timings(handle.read())
    if log_timings:
        if entry["evaluated_tokens"] is None and "prompt_eval_tokens" in log_timings:
            entry["evaluated_tokens"] = log_timings["prompt_eval_tokens"]
            if entry["prompt_tokens"]:
                entry["cache_hit_fraction"] = round(
                    max(0.0, 1.0 - entry["evaluated_tokens"] / entry["prompt_tokens"]), 4
                )
        if entry["prefill_tokens_per_second"] is None and "prompt_eval_tokens_per_second" in log_timings:
            entry["prefill_tokens_per_second"] = round(log_timings["prompt_eval_tokens_per_second"], 3)
        if entry["decode_tokens_per_second"] is None and "eval_tokens_per_second" in log_timings:
            entry["decode_tokens_per_second"] = round(log_timings["eval_tokens_per_second"], 3)
        entry["log_timings"] = log_timings
    return entry


def build_targets(ctx_size):
    return [{"step": 0, "target_pct": 0, "target_tokens": 0}] + [
        {"step": index, "target_pct": pct, "target_tokens": int(ctx_size * pct / 100)}
        for index, pct in enumerate(LADDER_PCTS, start=1)
    ]


def run_ladder(base_url, profile, targets, seed, log_path):
    model = profile["served_model_name"]
    max_tokens = int(profile.get("n_predict", 128))
    records = []
    prompt = "Привет"
    prompt_tokens = token_count(base_url, prompt)
    previous_prompt = None
    for target in targets:
        if target["target_tokens"] > 0:
            prompt, prompt_tokens = make_text(base_url, "Привет", target["target_tokens"])
        prefix_ok = True if previous_prompt is None else prompt.startswith(previous_prompt)
        entry = run_request(base_url, model, prompt, max_tokens, seed, log_path)
        entry.update({
            "step": target["step"],
            "target_pct": target["target_pct"],
            "target_tokens": target["target_tokens"],
            "raw_prompt_tokens": prompt_tokens,
            "prompt_prefix_ok": prefix_ok,
        })
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


def run_ab_sequence(base_url, profile, targets, order, seed, log_path):
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
    records = []
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
        entry = run_request(base_url, model, prompts[session], max_tokens, seed, log_path)
        entry.update({
            "order_index": order_index,
            "session": session,
            "step": target["step"],
            "target_pct": target["target_pct"],
            "target_tokens": target["target_tokens"],
            "raw_prompt_tokens": prompt_tokens[session],
            "prompt_prefix_ok": prefix_ok,
        })
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
    return {"order": order, "steps": records}


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

    run_id = f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{profile['name']}-{args.mode}"
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

    targets = build_targets(ctx_size)
    telemetry_stop = threading.Event()
    telemetry = None
    process = None
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    report = {
        "run_id": run_id,
        "started_at": started_at,
        "status": "ok",
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
        "n_predict": int(profile.get("n_predict", 128)),
        "seed": args.seed,
        "port": port,
        "targets": targets,
        "ladder_pcts": list(LADDER_PCTS),
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
            elif args.mode == "ladder":
                report["steps"] = run_ladder(base_url, profile, targets, args.seed, result_dir / "server.log")
            else:
                report.update(run_ab_sequence(base_url, profile, targets, args.order, args.seed, result_dir / "server.log"))
    except Exception as exc:  # noqa: BLE001
        report["status"] = "failed"
        report["error"] = str(exc)
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
