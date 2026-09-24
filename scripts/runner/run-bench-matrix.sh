#!/usr/bin/env bash
# Portable cache-benchmark entry point for a study case.
#
# Usage (from the case root; the script is copied into <case>/scripts/):
#   bash scripts/run-bench-matrix.sh <profile> <ladder|ab-abab|ab-abbabaa>
#   bash scripts/run-bench-matrix.sh report   # aggregate results into docs/
#   bash scripts/run-bench-matrix.sh plot     # render SVG figures (ru + en)
#   bash scripts/run-bench-matrix.sh all      # report, then plot
#   bash scripts/run-bench-matrix.sh check    # run the publication barrier
#   bash scripts/run-bench-matrix.sh sanitize # report sanitizable markers (dry-run)
#   bash scripts/run-bench-matrix.sh sanitize-fix # rewrite sanitizable markers in place
#
# <profile> is a profile file name without the .json suffix found in
# <case>/profiles/. The case directory is the parent of this script's directory;
# the repository root defaults to the case directory and can be overridden with
# REPO_ROOT.
#
# The runner starts and stops its own llama-server before returning, so this
# script only exits after the server has been stopped. The report/plot/all
# modes only run the python aggregators and never touch a server.
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

usage() {
  echo "usage: bash scripts/run-bench-matrix.sh <profile> <ladder|ab-abab|ab-abbabaa>" >&2
  echo "       bash scripts/run-bench-matrix.sh report|plot|all" >&2
  echo "       bash scripts/run-bench-matrix.sh check" >&2
  echo "       bash scripts/run-bench-matrix.sh sanitize|sanitize-fix" >&2
}

if [[ $# -eq 1 && "$1" =~ ^(report|plot|all)$ ]]; then
  RESULTS_DIR="$CASE_DIR/results"
  DOCS_DIR="$CASE_DIR/docs"
  FIGURES_DIR="$DOCS_DIR/figures"
  CONFIG_FILE="$CASE_DIR/study.json"
  REPORT_SCRIPT="$(require_script generate_report.py)"
  PLOT_SCRIPT="$(require_script plot_context_curves.py)"

  run_report() {
    local args=(
      --results-dir "$RESULTS_DIR"
      --docs-dir "$DOCS_DIR"
      --case-dir "$CASE_DIR"
      --repo-root "$ROOT_DIR"
    )
    if [[ -f "$CONFIG_FILE" ]]; then
      args+=(--config "$CONFIG_FILE")
    fi
    python3 "$REPORT_SCRIPT" "${args[@]}"
    echo "wrote $DOCS_DIR/results-tables.md"
    echo "wrote $DOCS_DIR/results.json"
  }

  run_plot() {
    mkdir -p "$FIGURES_DIR"
    local lang
    for lang in ru en; do
      local args=(
        --results-dir "$RESULTS_DIR"
        --out-dir "$FIGURES_DIR"
        --case-dir "$CASE_DIR"
        --repo-root "$ROOT_DIR"
        --lang "$lang"
      )
      if [[ -f "$CONFIG_FILE" ]]; then
        args+=(--config "$CONFIG_FILE")
      fi
      python3 "$PLOT_SCRIPT" "${args[@]}"
    done
    for lang in ru en; do
      for name in prefill-vs-context decode-vs-context ab-cache-hit \
        power-vs-load energy-per-token; do
        if [[ -f "$FIGURES_DIR/$name.$lang.svg" ]]; then
          echo "wrote $FIGURES_DIR/$name.$lang.svg"
        fi
      done
    done
  }

  case "$1" in
    report) run_report ;;
    plot) run_plot ;;
    all)
      run_report
      run_plot
      ;;
  esac
  exit 0
fi

if [[ $# -eq 1 && "$1" =~ ^(check|sanitize|sanitize-fix)$ ]]; then
  case "$1" in
    check)
      CHECK_SCRIPT="$(require_script check-public.sh)"
      set +e
      bash "$CHECK_SCRIPT"
      status=$?
      set -e
      if [[ "$status" -eq 2 ]]; then
        echo "barrier failed" >&2
      fi
      exit "$status"
      ;;
    sanitize)
      SANITIZE_SCRIPT="$(require_script sanitize-results.sh)"
      set +e
      bash "$SANITIZE_SCRIPT"
      status=$?
      set -e
      exit "$status"
      ;;
    sanitize-fix)
      SANITIZE_SCRIPT="$(require_script sanitize-results.sh)"
      echo "applying in-place sanitization to results"
      set +e
      bash "$SANITIZE_SCRIPT" --in-place
      status=$?
      set -e
      exit "$status"
      ;;
  esac
fi

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

PROFILE="$1"
MODE="$2"
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
  log="$(mktemp "${TMPDIR:-/tmp}/bench-matrix.XXXXXX")"
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
  ladder)
    run_runner --profile "$PROFILE_JSON" --mode ladder \
      --case-dir "$CASE_DIR" --repo-root "$ROOT_DIR"
    ;;
  ab-abab)
    run_runner --profile "$PROFILE_JSON" --mode ab-sequence --order ABAB \
      --case-dir "$CASE_DIR" --repo-root "$ROOT_DIR"
    ;;
  ab-abbabaa)
    run_runner --profile "$PROFILE_JSON" --mode ab-sequence --order ABBABAA \
      --case-dir "$CASE_DIR" --repo-root "$ROOT_DIR"
    ;;
  *)
    echo "unknown mode: $MODE (expected ladder, ab-abab or ab-abbabaa)" >&2
    exit 2
    ;;
esac
