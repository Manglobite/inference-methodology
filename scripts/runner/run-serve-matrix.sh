#!/usr/bin/env bash
# Portable entry point for starting a study profile.
#
# Usage (from the case root; the script is copied into <case>/scripts/):
#   bash scripts/run-serve-matrix.sh <profile> [smoke|up]
#
#   smoke (default) start the server, validate /props, run one tiny chat
#                   request, stop the server and write results/<id>/result.json;
#   up              start the server detached and leave it running.
#
# <profile> is a profile file name without the .json suffix found in
# <case>/profiles/. The case directory is the parent of this script's directory;
# the repository root defaults to the case directory and can be overridden with
# REPO_ROOT.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CASE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="${REPO_ROOT:-$CASE_DIR}"
cd "$ROOT_DIR"

find_script() {
  # Locate a methodology script in the canonical tree and in a copied case.
  local name="$1" candidate
  for candidate in \
    "$SCRIPT_DIR/$name" \
    "$SCRIPT_DIR/../report/$name" \
    "$SCRIPT_DIR/../publish/$name" \
    "$CASE_DIR/scripts/$name"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

require_script() {
  local name="$1" path
  if ! path="$(find_script "$name")"; then
    echo "script not found: $name (looked in $SCRIPT_DIR, $SCRIPT_DIR/../report, $SCRIPT_DIR/../publish, $CASE_DIR/scripts)" >&2
    exit 2
  fi
  printf '%s\n' "$path"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: bash scripts/run-serve-matrix.sh <profile> [smoke|up]" >&2
  exit 2
fi

PROFILE="$1"
MODE="${2:-smoke}"
PROFILE_JSON="$CASE_DIR/profiles/$PROFILE.json"
RUNNER="$(require_script run_cache_sessions.py)"
HUB_IDLE="$CASE_DIR/scripts/hub-idle.sh"

if [[ ! -f "$PROFILE_JSON" ]]; then
  echo "profile not found: $PROFILE_JSON" >&2
  exit 2
fi

run_runner() {
  local log status result_dir line
  if [[ -f "$HUB_IDLE" ]]; then
    bash "$HUB_IDLE" off || true
  fi
  log="$(mktemp "${TMPDIR:-/tmp}/serve-matrix.XXXXXX")"
  set +e
  python3 "$RUNNER" "$@" 2>&1 | tee "$log"
  status="${PIPESTATUS[0]}"
  set -e
  result_dir=""
  while IFS= read -r line; do
    case "$line" in
      "Results: "*) result_dir="${line#Results: }" ;;
    esac
  done <"$log"
  rm -f "$log"
  if [[ "$status" -ne 0 ]]; then
    echo "runner failed with status $status" >&2
    exit "$status"
  fi
  if [[ -n "$result_dir" && -f "$result_dir/result.json" ]]; then
    echo "result.json: $result_dir/result.json"
  else
    echo "result.json was not reported by the runner" >&2
    exit 1
  fi
}

case "$MODE" in
  smoke)
    run_runner --profile "$PROFILE_JSON" --mode smoke \
      --case-dir "$CASE_DIR" --repo-root "$ROOT_DIR"
    ;;
  up)
    NOTES_DIR="$CASE_DIR/notes"
    mkdir -p "$NOTES_DIR"
    LOG_FILE="$NOTES_DIR/serve-$PROFILE.log"
    PID_FILE="$NOTES_DIR/serve-$PROFILE.pid"

    # One llama-server per host is assumed here: the runner also tracks every
    # `llama-server` process via /proc, so a second instance would pollute its
    # memory aggregation. The profile port is additionally checked below.
    if pgrep -x 'llama-server' >/dev/null 2>&1; then
      echo "another llama-server is already running; stop it first" >&2
      pgrep -ax 'llama-server' >&2 || true
      exit 1
    fi

    read -r PORT LAUNCH_CMD < <(python3 - "$PROFILE_JSON" "$ROOT_DIR" "$PID_FILE" <<'PY'
import json
import shlex
import sys

profile_path, root, pid_file = sys.argv[1], sys.argv[2], sys.argv[3]
with open(profile_path, encoding="utf-8") as handle:
    profile = json.loads(handle.read())

command = [str(item).replace("<repo>", root) for item in profile["command"]]
environment = profile.get("command_env", {})
parts = [f"{key}={shlex.quote(str(value))}" for key, value in environment.items()]
parts += [shlex.quote(item) for item in command]
launch = "echo $$ > " + shlex.quote(pid_file) + " && cd " + shlex.quote(root) + " && exec env " + " ".join(parts)
print(profile.get("port", 18091), launch)
PY
) || true
    if [[ -z "${PORT:-}" || -z "${LAUNCH_CMD:-}" ]]; then
      echo "failed to derive launch command from $PROFILE_JSON" >&2
      exit 1
    fi
    if python3 - "$PORT" <<'PY'
import socket
import sys

sock = socket.socket()
try:
    if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0:
        sys.exit(0)
finally:
    sock.close()
sys.exit(1)
PY
    then
      echo "port $PORT is already in use; stop the server on it or change the profile port" >&2
      exit 1
    fi
    HEALTH_URL="http://127.0.0.1:$PORT/health"
    rm -f "$PID_FILE"

    setsid nohup bash -c "$LAUNCH_CMD" >"$LOG_FILE" 2>&1 &

    SERVER_PID=""
    for _ in $(seq 1 100); do
      if [[ -s "$PID_FILE" ]]; then
        SERVER_PID="$(cat "$PID_FILE")"
        break
      fi
      sleep 0.1
    done
    if [[ -z "$SERVER_PID" ]]; then
      echo "server pid was not recorded; see $LOG_FILE" >&2
      exit 1
    fi

    if python3 - "$PORT" "$SERVER_PID" <<'PY'
import os
import sys
import time
import urllib.error
import urllib.request

port, pid = int(sys.argv[1]), int(sys.argv[2])
url = f"http://127.0.0.1:{port}/health"
deadline = time.monotonic() + 900
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2):
            sys.exit(0)
    except (OSError, urllib.error.URLError):
        try:
            os.kill(pid, 0)
        except OSError:
            sys.exit(1)
    time.sleep(2)
sys.exit(1)
PY
    then
      echo "server is up: pid=$SERVER_PID port=$PORT health=$HEALTH_URL"
      echo "log: $LOG_FILE"
      echo "pid file: $PID_FILE"
    else
      echo "server failed to become healthy (pid=$SERVER_PID, port=$PORT); see $LOG_FILE" >&2
      exit 1
    fi
    ;;
  *)
    echo "unknown mode: $MODE (expected smoke or up)" >&2
    exit 2
    ;;
esac
