#!/usr/bin/env bash
# Portable publication barrier without git: scan the case tree recursively for
# private markers. Every file is walked except the excluded directories below
# (.git/, build/, dist/, __pycache__/, node_modules/, *.pyc); it is not an
# allowlist. results/ is included on purpose (it is published after sanitizing).
# Run before publishing or pushing the case.
#
# Fail-closed: a scan error or any match exits 2; exit 0 only when the set is
# clean.
#
# The patterns are generic on purpose: no node-specific account, hostname or
# service names are baked in. The baseline is shared with sanitize-results.sh:
# local absolute paths /home/<user>/, /mnt/, /root/ and macOS /Users/<user>/,
# RFC1918 ranges, the secret pattern sk-[A-Za-z0-9_-]{13,}, the
# `Authorization: Bearer` header and `api_key`/`secret`/`token` assignments.
# Hostnames and internal identifiers are NOT detected by this baseline -- add
# them via PUBLIC_DENY_FILE and/or manual review. Extend the
# barrier without editing this script:
#   * PUBLIC_DENY_FILE=/path/to/deny.txt -- extra grep -E patterns, one per line;
#   * .public-allow -- per-line grep -E exceptions inside the case tree.
#
# Only the specific RFC1918 ranges are scanned (10/8, 172.16/12, 192.168/16);
# no blanket four-octet IPv4 pattern is used. A leading llama.cpp progress
# timestamp `H.MM.mmm.uuu` (its first field may be `10`, i.e. inside 10/8) can
# still match RFC1918, so an IP hit is confirmed only after the leading
# timestamp token is stripped; the other patterns are checked against the raw
# line and are not weakened.

set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

SELF="./scripts/check-public.sh"
ALLOW_FILE=".public-allow"

# --- Patterns -----------------------------------------------------------------
OCTET='(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])'
RFC1918="\\b10\\.${OCTET}\\.${OCTET}\\.${OCTET}\\b"
RFC1918="${RFC1918}|\\b172\\.(1[6-9]|2[0-9]|3[01])\\.${OCTET}\\.${OCTET}\\b"
RFC1918="${RFC1918}|\\b192\\.168\\.${OCTET}\\.${OCTET}\\b"

# Leading llama.cpp progress timestamp `H.MM.mmm.uuu`; its first field may be
# `10`, which collides with RFC1918 10/8.
TS_PREFIX='^[[:space:]]*[0-9]{1,3}\.[0-9]{2}\.[0-9]{3}\.[0-9]{3}'

# RFC1918 is scanned separately so that a leading timestamp token can be
# stripped before an IP hit is confirmed; the patterns below still see the raw
# line and keep their full strength.
#
# Secret threshold: `sk-` followed by at least 13 key-body characters
# (`[A-Za-z0-9_-]`). This exact pattern is shared with sanitize-results.sh so
# both tools agree on the same tree; keep them in sync.
PATTERNS=(
  '/home/[^/]+/'
  '/mnt/'
  '/root/'
  '/Users/[^/]+/'
  'sk-[A-Za-z0-9_-]{13,}'
  'Authorization:[[:space:]]*Bearer'
  '(api[_-]?key|secret|token)[[:space:]]*[:=]'
)

if [[ -n "${PUBLIC_DENY_FILE:-}" ]]; then
  if [[ ! -r "$PUBLIC_DENY_FILE" ]]; then
    echo "FAIL: PUBLIC_DENY_FILE is not readable: $PUBLIC_DENY_FILE" >&2
    exit 2
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    PATTERNS+=("$line")
  done < "$PUBLIC_DENY_FILE"
fi

COMBINED=""
for pattern in "${PATTERNS[@]}"; do
  if [[ -z "$COMBINED" ]]; then
    COMBINED="$pattern"
  else
    COMBINED="${COMBINED}|${pattern}"
  fi
done

ALLOW=""
if [[ -r "$ALLOW_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    if [[ -z "$ALLOW" ]]; then
      ALLOW="$line"
    else
      ALLOW="${ALLOW}|${line}"
    fi
  done < "$ALLOW_FILE"
fi

# --- Collect files ------------------------------------------------------------
FILE_LIST="$(mktemp "${TMPDIR:-/tmp}/check-public.XXXXXX")"
trap 'rm -f "$FILE_LIST"' EXIT

if ! find . -type f \
  ! -path '*/.git/*' \
  ! -path '*/__pycache__/*' \
  ! -path '*/node_modules/*' \
  ! -path '*/build/*' \
  ! -path '*/dist/*' \
  ! -name '*.pyc' \
  ! -path "$SELF" \
  ! -path "./$ALLOW_FILE" \
  -print0 > "$FILE_LIST"; then
  echo "FAIL: scan error (could not enumerate files)" >&2
  exit 2
fi

FILES=()
while IFS= read -r -d '' file; do
  FILES+=("$file")
done < "$FILE_LIST"

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "OK: no private markers found"
  exit 0
fi

# --- Scan ---------------------------------------------------------------------
set +e
hits="$(grep -nHIiE --binary-files=without-match -e "$COMBINED" -- "${FILES[@]}" 2>/dev/null)"
status=$?
ip_hits="$(grep -nHIiE --binary-files=without-match -e "$RFC1918" -- "${FILES[@]}" 2>/dev/null)"
ip_status=$?
set -e

if [[ "$status" -eq 2 || "$ip_status" -eq 2 ]]; then
  echo "FAIL: scan error (unreadable input)" >&2
  exit 2
fi

if [[ "$status" -ne 0 && -n "$hits" ]] || [[ "$ip_status" -ne 0 && -n "$ip_hits" ]]; then
  echo "FAIL: scan error (unexpected grep status $status/$ip_status)" >&2
  exit 2
fi

# --- Drop timestamp-only IP hits ----------------------------------------------
# Keep an IP hit only if a real address survives after a leading llama.cpp
# timestamp token is stripped: a log line that starts with a timestamp at hour
# `10` is a false positive, while a real 192.168.x.x later in the same line is
# still reported.
ip_filtered=""
if [[ -n "$ip_hits" ]]; then
  while IFS= read -r hit; do
    [[ -z "$hit" ]] && continue
    rest="${hit#*:}"
    text="${rest#*:}"
    stripped="$text"
    if [[ "$text" =~ $TS_PREFIX ]]; then
      ts="${BASH_REMATCH[0]}"
      stripped="${text#"$ts"}"
    fi
    if grep -qiE --binary-files=without-match -e "$RFC1918" <<< "$stripped"; then
      if [[ -z "$ip_filtered" ]]; then
        ip_filtered="$hit"
      else
        ip_filtered="${ip_filtered}"$'\n'"${hit}"
      fi
    fi
  done <<< "$ip_hits"
fi

if [[ -n "$ip_filtered" ]]; then
  if [[ -z "$hits" ]]; then
    hits="$ip_filtered"
  else
    hits="${hits}"$'\n'"${ip_filtered}"
  fi
fi

# --- Filter allowlist ---------------------------------------------------------
if [[ -n "$ALLOW" && -n "$hits" ]]; then
  filtered=""
  while IFS= read -r hit; do
    [[ -z "$hit" ]] && continue
    rest="${hit#*:}"
    text="${rest#*:}"
    if grep -qiE --binary-files=without-match -e "$ALLOW" <<< "$text"; then
      continue
    fi
    if [[ -z "$filtered" ]]; then
      filtered="$hit"
    else
      filtered="${filtered}"$'\n'"${hit}"
    fi
  done <<< "$hits"
  hits="$filtered"
fi

if [[ -n "$hits" ]]; then
  printf '%s\n' "$hits" >&2
  echo "FAIL: private markers found" >&2
  exit 2
fi

echo "OK: no private markers found"
