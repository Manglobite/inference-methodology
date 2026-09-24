#!/usr/bin/env bash
# Sanitize publication artefacts: replace private paths, mount points, RFC1918
# addresses, secrets and caller-supplied private markers with portable
# placeholders.
#
# Baseline local-path markers are generic: /home/<user>/, /mnt/, /root/ and
# macOS /Users/<user>/ -- the same set check-public.sh uses.
#
# Portable baseline: no node-specific account, hostname or absolute repository
# path is baked in. Node specifics are injected at run time through the
# environment:
#   * SANITIZE_USER=<name>       account name -> <user>
#   * SANITIZE_REPO_ROOT=<path>  repository root -> <repo>
# Extra markers come from a file:
#   * --patterns FILE / SANITIZE_PATTERNS_FILE, one `pattern<TAB>replacement`
#     per line (a single `|` separator is accepted as a fallback). Blank lines
#     and `#` comments are ignored. Patterns are extended regular expressions;
#     user patterns are applied after the base patterns, the base patterns stay
#     in force.
#
# Fail-closed by default: without --in-place it only reports matches and exits
# non-zero when any are found. Pass --in-place to rewrite the matched files.
#
# A leading llama.cpp progress timestamp (H.MM.mmm.uuu) is guarded before the
# private-IPv4 substitutions and restored afterwards, so it is never mistaken
# for (or rewritten as) an RFC1918 address.
set -euo pipefail

IN_PLACE=0
PATTERNS_FILE="${SANITIZE_PATTERNS_FILE:-}"
PATHS=()
EXPLICIT_PATHS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-place)
      IN_PLACE=1
      shift
      ;;
    --patterns)
      if [[ $# -lt 2 ]]; then
        echo "FAIL: --patterns needs a file argument" >&2
        exit 2
      fi
      PATTERNS_FILE="$2"
      shift 2
      ;;
    --patterns=*)
      PATTERNS_FILE="${1#--patterns=}"
      shift
      ;;
    -h|--help)
      echo "usage: sanitize-results.sh [--in-place] [--patterns FILE] [path ...]"
      echo "defaults: results notes docs prompts plus top-level *.md/*.json/*.csv/*.log"
      echo "user patterns: 'pattern<TAB>replacement' per line (a single '|' is accepted)"
      exit 0
      ;;
    *)
      PATHS+=("$1")
      EXPLICIT_PATHS=1
      shift
      ;;
  esac
done

if [[ -n "$PATTERNS_FILE" && "$PATTERNS_FILE" != /* ]]; then
  PATTERNS_FILE="$PWD/$PATTERNS_FILE"
fi

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

if [[ ${#PATHS[@]} -eq 0 ]]; then
  PATHS=(results notes docs prompts)
  for file in ./*.md ./*.json ./*.csv ./*.log; do
    if [[ -f "$file" ]]; then
      PATHS+=("$file")
    fi
  done
fi

if [[ "$EXPLICIT_PATHS" -eq 1 ]]; then
  for path in "${PATHS[@]}"; do
    if [[ ! -e "$path" ]]; then
      echo "FAIL: path does not exist: $path" >&2
      exit 2
    fi
  done
else
  for path in "${PATHS[@]}"; do
    if [[ ! -e "$path" ]]; then
      echo "note: default path not present, skipped: $path"
    fi
  done
fi

# IPv4 with validated octets (0-255) restricted to RFC1918 private ranges.
# Loopback (127.0.0.0/8) is deliberately not a marker.
OCTET='(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])'
IPV4_10="10\.${OCTET}\.${OCTET}\.${OCTET}"
IPV4_172="172\.(1[6-9]|2[0-9]|3[01])\.${OCTET}\.${OCTET}"
IPV4_192="192\.168\.${OCTET}\.${OCTET}"
PRIVATE_IP="\\b${IPV4_10}\\b|\\b${IPV4_172}\\b|\\b${IPV4_192}\\b"
# Secret threshold: `sk-` followed by at least 13 key-body characters
# (`[A-Za-z0-9_-]`). This exact pattern is shared with check-public.sh so both
# tools agree on the same tree; keep them in sync.
KEY_RE='sk-[A-Za-z0-9_-]{13,}'
BEARER_RE='Authorization:[[:space:]]*Bearer[[:space:]]+[A-Za-z0-9._~+/=-]+'

# Leading llama.cpp progress timestamp, e.g. 0.29.793.287. When every group is
# <= 255 it visually resembles an RFC1918 address but is not one, so it is
# stripped from the line before the private-IP scan.
TS_PATTERN='^[[:space:]]*[0-9]+\.[0-9]{2}\.[0-9]{3}\.[0-9]{3}'

# Pick a sed delimiter that occurs in neither the pattern nor the replacement.
choose_delim() {
  local d
  for d in '|' '~' '#' '@' '%' '^' '!'; do
    if [[ "$1" != *"$d"* && "$2" != *"$d"* ]]; then
      printf '%s' "$d"
      return 0
    fi
  done
  return 1
}

build_subst() {
  local pattern="$1" replacement="$2" delim repl_esc
  if ! delim="$(choose_delim "$pattern" "$replacement")"; then
    echo "FAIL: no safe delimiter for pattern: $pattern" >&2
    exit 2
  fi
  repl_esc="$(printf '%s' "$replacement" | sed 's/[&\\]/\\&/g')"
  printf 's%s%s%s%s%sg' "$delim" "$pattern" "$delim" "$repl_esc" "$delim"
}

# Escape ERE metacharacters so environment values are matched literally.
ere_escape() {
  printf '%s' "$1" | sed 's/[][\.^$*+?(){}|]/\\&/g'
}

# --- Caller-supplied patterns -------------------------------------------------
CUSTOM_PATTERNS=()
CUSTOM_REPLACEMENTS=()
if [[ -n "$PATTERNS_FILE" ]]; then
  if [[ ! -r "$PATTERNS_FILE" ]]; then
    echo "FAIL: patterns file is not readable: $PATTERNS_FILE" >&2
    exit 2
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    if [[ "$line" == *$'\t'* ]]; then
      pattern="${line%%$'\t'*}"
      replacement="${line#*$'\t'}"
    elif [[ "$line" == *"|"* ]]; then
      pattern="${line%%|*}"
      replacement="${line#*|}"
    else
      echo "FAIL: bad pattern line (expected 'pattern<TAB>replacement'): $line" >&2
      exit 2
    fi
    CUSTOM_PATTERNS+=("$pattern")
    CUSTOM_REPLACEMENTS+=("$replacement")
  done < "$PATTERNS_FILE"
fi

collect_files() {
  local path file
  for path in "${PATHS[@]}"; do
    [[ -e "$path" ]] || continue
    if [[ -f "$path" ]]; then
      printf '%s\n' "$path"
      continue
    fi
    while IFS= read -r file; do
      case "$file" in
        *.json|*.jsonl|*.csv|*.log|*.md|*.txt|*.sh|*.py) printf '%s\n' "$file" ;;
      esac
    done < <(find "$path" -type f \
      ! -path '*/.git/*' ! -path '*/.raw/*' ! -path '*/__pycache__/*' \
      ! -name '*.pyc' -print)
  done
}

# Non-IP markers: local absolute paths (/home/<user>, /mnt, /root, macOS
# /Users/<user>), the optional account name and repository root, secrets and
# caller-supplied patterns.
build_non_ip_patterns() {
  printf '%s\n' '/home/[^/]+' '/mnt/' '/root/' '/Users/[^/]+' "$KEY_RE" "$BEARER_RE"
  if [[ -n "${SANITIZE_REPO_ROOT:-}" ]]; then
    printf '%s\n' "$(ere_escape "$SANITIZE_REPO_ROOT")"
  fi
  if [[ -n "${SANITIZE_USER:-}" ]]; then
    printf '%s\n' "$(ere_escape "$SANITIZE_USER")"
  fi
  local i
  if [[ ${#CUSTOM_PATTERNS[@]} -gt 0 ]]; then
    for i in "${!CUSTOM_PATTERNS[@]}"; do
      printf '%s\n' "${CUSTOM_PATTERNS[$i]}"
    done
  fi
}

# Private-IP scan that ignores a leading llama.cpp timestamp.
scan_private_ips() {
  local file out status hit number line stripped
  for file in "$@"; do
    set +e
    out="$(grep -nH --binary-files=without-match -E "$PRIVATE_IP" "$file" 2>/dev/null)"
    status=$?
    set -e
    if [[ "$status" -eq 2 ]]; then
      echo "FAIL: scan error (unreadable input)" >&2
      return 2
    fi
    [[ -n "$out" ]] || continue
    while IFS= read -r hit; do
      number="${hit#"$file":}"
      number="${number%%:*}"
      line="$(sed -n "${number}p" "$file")"
      stripped="$(printf '%s\n' "$line" | sed -E "s|${TS_PATTERN}| |")"
      if printf '%s\n' "$stripped" | grep -qE --binary-files=without-match "$PRIVATE_IP"; then
        printf '%s\n' "$hit"
      fi
    done <<< "$out"
  done
}

non_ip_args=()
while IFS= read -r pattern; do
  non_ip_args+=(-e "$pattern")
done < <(build_non_ip_patterns)

if [[ "$IN_PLACE" -eq 0 ]]; then
  files=()
  while IFS= read -r file; do
    [[ -n "$file" ]] && files+=("$file")
  done < <(collect_files)

  hits=""
  if [[ ${#files[@]} -gt 0 ]]; then
    set +e
    hits="$(grep -nH --binary-files=without-match -E "${non_ip_args[@]}" "${files[@]}" 2>/dev/null)"
    status=$?
    set -e
    if [[ "$status" -eq 2 ]]; then
      echo "FAIL: scan error (unreadable input)" >&2
      exit 2
    fi
    set +e
    ip_hits="$(scan_private_ips "${files[@]}")"
    ip_status=$?
    set -e
    if [[ "$ip_status" -eq 2 ]]; then
      exit 2
    fi
    if [[ -n "$ip_hits" ]]; then
      if [[ -n "$hits" ]]; then
        hits="${hits}"$'\n'"${ip_hits}"
      else
        hits="$ip_hits"
      fi
    fi
  fi

  printf 'checked %s file(s)\n' "${#files[@]}"

  if [[ -n "$hits" ]]; then
    printf '%s\n' "$hits"
    printf 'found %s matching line(s)\n' "$(printf '%s\n' "$hits" | wc -l)"
    echo "FAIL: private markers found (report-only); re-run with --in-place to rewrite" >&2
    exit 2
  fi
  echo "no private markers found"
  exit 0
fi

base_args=()
if [[ -n "${SANITIZE_REPO_ROOT:-}" ]]; then
  base_args+=(-e "$(build_subst "$(ere_escape "$SANITIZE_REPO_ROOT")" '<repo>')")
fi
base_args+=(-e "$(build_subst '/home/[^/]+' '<home>')")
base_args+=(-e "$(build_subst '/mnt/' '<mount>/')")
base_args+=(-e "$(build_subst '/root/' '<root>/')")
base_args+=(-e "$(build_subst '/Users/[^/]+' '<home>')")
if [[ -n "${SANITIZE_USER:-}" ]]; then
  base_args+=(-e "$(build_subst "$(ere_escape "$SANITIZE_USER")" '<user>')")
fi
base_args+=(-e "$(build_subst "$KEY_RE" '<secret>')")
base_args+=(-e "$(build_subst "$BEARER_RE" '<secret>')")

custom_args=()
if [[ ${#CUSTOM_PATTERNS[@]} -gt 0 ]]; then
  for i in "${!CUSTOM_PATTERNS[@]}"; do
    custom_args+=(-e "$(build_subst "${CUSTOM_PATTERNS[$i]}" "${CUSTOM_REPLACEMENTS[$i]}")")
  done
fi

# Guard the leading llama.cpp timestamp behind a sentinel (dots turned into
# dashes) so the private-IP substitutions cannot rewrite it, then restore it
# after the IP pass. This keeps the timestamp intact while still rewriting real
# RFC1918 addresses that appear elsewhere on the same line.
TS_GUARD="s~^([[:space:]]*)([0-9]+)\.([0-9]{2})\.([0-9]{3})\.([0-9]{3})~\1@@TS@@\2-\3-\4-\5~"
TS_RESTORE="s~@@TS@@([0-9]+)-([0-9]{2})-([0-9]{3})-([0-9]{3})~@@TS@@\1.\2.\3.\4~"

matched_lines=0
changed_files=0
while IFS= read -r file; do
  [[ -n "$file" ]] || continue
  line_matches="$(grep -Ec --binary-files=without-match "${non_ip_args[@]}" "$file" || true)"
  ip_lines="$(scan_private_ips "$file" || true)"
  ip_matches=0
  if [[ -n "$ip_lines" ]]; then
    ip_matches="$(printf '%s\n' "$ip_lines" | wc -l)"
  fi
  matched_lines=$((matched_lines + line_matches + ip_matches))
  before_hash="$(sha256sum "$file" | cut -d' ' -f1)"
  sed -i -E "${base_args[@]}" "$file"
  if [[ ${#custom_args[@]} -gt 0 ]]; then
    sed -i -E "${custom_args[@]}" "$file"
  fi
  sed -i -E \
    -e "$TS_GUARD" \
    -e "s~${IPV4_10}~<ip>~g" \
    -e "s~${IPV4_172}~<ip>~g" \
    -e "s~${IPV4_192}~<ip>~g" \
    -e "$TS_RESTORE" \
    -e "s~@@TS@@~~" \
    "$file"
  after_hash="$(sha256sum "$file" | cut -d' ' -f1)"
  if [[ "$before_hash" != "$after_hash" ]]; then
    changed_files=$((changed_files + 1))
  fi
done < <(collect_files)
echo "sanitized in place: ${PATHS[*]}"
echo "matching lines replaced: $matched_lines"
echo "files changed: $changed_files"
