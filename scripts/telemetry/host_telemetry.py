#!/usr/bin/env python3
"""Sample host and GPU telemetry to a CSV until terminated.

The column set follows METHODOLOGY section 10 (Telemetry), schema version 1:
host CPU/RAM/swap, per-core CPU load, per-GPU temperature/utilization/memory/
power/SM clock/pstate, and the RSS/PSS/swap of the server process. The number
of GPUs and CPU cores is detected dynamically, so the file can be reused on a
different rig.

SM clocks before/after a run are not written to the CSV (they belong to
result.json); they are reported on stderr and, optionally, to a JSON file.

Standard library only. Terminate with SIGTERM/SIGINT to stop sampling.
"""

import argparse
import csv
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

SCHEMA_VERSION = 1
DEFAULT_INTERVAL_S = 0.5
DEFAULT_PROC_NAME = "llama-server"
GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,temperature.gpu,utilization.gpu,memory.used,power.draw,clocks.sm,pstate",
    "--format=csv,noheader,nounits",
]
GPU_VALUE_COUNT = 6
GPU_CLOCK_OFFSET = 4


def host_cpu_ticks():
    try:
        fields = Path("/proc/stat").read_text().splitlines()[0].split()
        values = [int(value) for value in fields[1:]]
        return sum(values), values[3] + values[4]
    except (IndexError, ValueError, OSError):
        return 0, 0


def host_cpu_per_core_ticks():
    """Return a list of (total, idle) tick pairs, one per logical CPU."""
    cores = []
    try:
        lines = Path("/proc/stat").read_text().splitlines()
    except OSError:
        return cores
    for line in lines:
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
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                values[key] = int(value.strip().split()[0])
    except (OSError, ValueError):
        pass
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
    """GPU indices in nvidia-smi order; falls back to `nvidia-smi -L`."""
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


def gpu_sm_clocks(gpus):
    return {index: values[GPU_CLOCK_OFFSET] for index, values in gpus.items()}


def proc_memory_kib(pid):
    values = {}
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"Rss", "Pss", "Swap"}:
                values[key.lower()] = int(value.strip().split()[0])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        pass
    return values


def find_server_pids(proc_name):
    """All live PIDs whose executable basename equals proc_name."""
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            target = os.readlink(entry / "exe")
        except OSError:
            continue
        if os.path.basename(target) == proc_name:
            pids.append(int(entry.name))
    return sorted(pids)


def server_pids(pid, proc_name):
    """Tracked server PIDs: an explicit --pid takes priority over --proc-name."""
    if pid is not None:
        return [pid] if Path(f"/proc/{pid}").exists() else []
    return find_server_pids(proc_name)


def aggregate_server_memory(pids):
    """Sum RSS/PSS/swap over every tracked server process."""
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Sample CPU/RAM/GPU telemetry to CSV")
    parser.add_argument("path", help="output CSV path")
    parser.add_argument(
        "interval",
        nargs="?",
        type=float,
        default=DEFAULT_INTERVAL_S,
        help="seconds between samples (default 0.5)",
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="server PID to track; takes priority over --proc-name",
    )
    parser.add_argument(
        "--proc-name",
        default=DEFAULT_PROC_NAME,
        help="server executable name to track (default: llama-server)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the output file if it already exists",
    )
    parser.add_argument(
        "--clocks-json",
        default="",
        help="optional path for a JSON file with SM clocks before/after",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()

    if os.path.exists(args.path) and not args.force:
        print(
            f"error: output file already exists: {args.path} (use --force to overwrite)",
            file=sys.stderr,
        )
        return 2

    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    gpu_ids = gpu_indices()
    fields = [
        "schema_version",
        "timestamp_utc",
        "cpu_pct",
        "cpu_temp_c",
        "cpu_temp_source",
        "ram_used_gib",
        "ram_total_gib",
        "swap_used_gib",
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

    clocks_before = gpu_sm_clocks(gpu_snapshot())
    print(f"clocks_sm_before={json.dumps(clocks_before, sort_keys=True)}", file=sys.stderr)

    previous_host = host_cpu_ticks()
    previous_cores = host_cpu_per_core_ticks()
    warned_indices = set()

    try:
        handle = open(args.path, "w", newline="")
    except OSError as error:
        print(f"error: cannot open output file: {args.path}: {error}", file=sys.stderr)
        return 2

    with handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        while running:
            started = time.monotonic()
            host_total, host_idle = host_cpu_ticks()
            old_total, old_idle = previous_host
            total_delta = host_total - old_total
            cpu_pct = (
                100.0 * (total_delta - (host_idle - old_idle)) / total_delta
                if total_delta > 0
                else 0.0
            )
            previous_host = (host_total, host_idle)

            cores = host_cpu_per_core_ticks()
            host = host_memory_kib()
            total_kib = host.get("MemTotal", 0)
            available_kib = host.get("MemAvailable", 0)
            temp_c, temp_zone = cpu_temp_c()

            row = {
                "schema_version": SCHEMA_VERSION,
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
                + "Z",
                "cpu_pct": f"{cpu_pct:.1f}",
                "cpu_temp_c": temp_c,
                "cpu_temp_source": temp_zone,
                "ram_used_gib": round((total_kib - available_kib) / 1024 / 1024, 2),
                "ram_total_gib": round(total_kib / 1024 / 1024, 2),
                "swap_used_gib": round(
                    (host.get("SwapTotal", 0) - host.get("SwapFree", 0)) / 1024 / 1024, 2
                ),
            }

            for index, (core_total, core_idle) in enumerate(cores):
                old_core = (
                    previous_cores[index] if index < len(previous_cores) else (core_total, core_idle)
                )
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
                        f"warning: GPU {index} appeared after start; not in CSV header",
                        file=sys.stderr,
                    )

            pids = server_pids(args.pid, args.proc_name)
            memory = aggregate_server_memory(pids)
            row["server_rss_kib"] = memory.get("rss", "")
            row["server_pss_kib"] = memory.get("pss", "")
            row["server_swap_kib"] = memory.get("swap", "")
            row["llama_pids"] = " ".join(str(pid) for pid in pids)

            writer.writerow(row)
            handle.flush()
            time.sleep(max(0.0, args.interval - (time.monotonic() - started)))

    clocks_after = gpu_sm_clocks(gpu_snapshot())
    print(f"clocks_sm_after={json.dumps(clocks_after, sort_keys=True)}", file=sys.stderr)

    if args.clocks_json:
        try:
            with open(args.clocks_json, "w") as handle:
                json.dump(
                    {"clocks_sm_before": clocks_before, "clocks_sm_after": clocks_after},
                    handle,
                    indent=2,
                )
                handle.write("\n")
        except OSError as error:
            print(f"error: cannot write clocks JSON: {args.clocks_json}: {error}", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
