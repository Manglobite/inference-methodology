# Template for a publishable inference study

[Русский](TEMPLATE.md) | **English**

The working skeleton is next to this file: `templates/research-case/`.

## 1. Purpose

This document defines the **minimally sufficient** set of information for a
publishable LLM inference study: the data with which a third-party reader can
**reproduce** the runs and understand the conclusions **without the author**.
The template describes:

- what must always be present in a study directory;
- the format of profiles, raw runs and aggregates;
- what is published and what is not (weights, secrets, private paths);
- how the publication barrier works and how to adapt the template.

The "minimally sufficient" principle: fewer artefacts, but each one complete and
verifiable, is better than many incomplete ones. If a piece of information is
missing, it is stated explicitly rather than silently omitted.

The general measurement methodology is described in
[METHODOLOGY.md](METHODOLOGY.md). The description of the concrete node/rig is
internal material ([hardware_info/HARDWARE.md](hardware_info/HARDWARE.md)) and
is **not published**; a study describes its own rig in its local `HARDWARE.md`
(template — `templates/research-case/HARDWARE.template.md`).

## 2. Required publication contents (checklist)

| # | Section | Where it lives | Required |
| ---: | --- | --- | --- |
| 1 | Goal and hypotheses | `README.md` ("Goal" section) | yes |
| 2 | Model: source/URL, commit, sha256, size, architecture, license | `README.md`, `RUNBOOK.md`, `LICENSE-NOTICE.md` | yes |
| 3 | Hardware | `HARDWARE.md` | yes |
| 4 | Runtime: engine, commit, build flags, manifest, binary sha256 | `RUNBOOK.md`, `LICENSE-NOTICE.md`, manifests outside the case | yes |
| 5 | Methodology | link to `METHODOLOGY.md` + local details in `README.md` | yes |
| 6 | Run/reproduction | `RUNBOOK.md` | yes |
| 7 | Chronology and negative results | `CHRONOLOGY.md` | yes |
| 8 | Profiles | `profiles/*.json` | yes |
| 9 | Prompts | `prompts/*.json` | yes |
| 10 | Scripts | `scripts/` (copied from `methodology/scripts/<category>/`) | yes |
| 11 | Raw runs | `results/<run_id>/` | yes |
| 12 | Telemetry | `results/<run_id>/telemetry.csv` | yes |
| 13 | Aggregate tables + machine-readable summary | `docs/results-tables.md`, `docs/results.json` | yes |
| 14 | Figures | `docs/figures/*.svg` | conditional: a figure is mandatory **when the corresponding measured series exists**. `plot_context_curves.py` builds five figures: the base ones `prefill-vs-context`, `decode-vs-context` (if a ladder was measured) and `ab-cache-hit` (under A/B), plus `power-vs-load` and `energy-per-token` when an energy series (power telemetry) exists. Additional per-axis plots (layout/ubatch/speculation) are built separately by the author when a series exists. If the series was not measured, the figure is not built and the reason is recorded in `README.md` |
| 15 | Licenses | `LICENSE-NOTICE.md` | yes |
| 16 | Publication barrier | `scripts/check-public.sh`, `scripts/sanitize-results.sh` (both copied from `methodology/scripts/publish/`) | yes |

Additionally (as needed): `CHECKLIST.md` — study acceptance and publication;
`data/` — small input data (sha256, extracted curves, summary CSVs).

> **The skeleton is a frame, not a finished case.** `templates/research-case/`
> contains only document templates, an example profile, an example config
> (`study.example.json`) and placeholder directories (`profiles/`, `prompts/`,
> `results/`, `docs/`, `scripts/` with a single `README.md`). The required
> artefacts — real profiles, prompts, scripts and results — are added during
> adaptation of the case (section 9 and `CHECKLIST.md`, sections A/B). Scripts
> are **not shipped with the skeleton**: they are copied from
> `methodology/scripts/<category>/` (section 3,
> `methodology/scripts/README.md`). Empty directories do not satisfy the contents
> requirements.

## 3. Study directory structure

```text
<case>/
├── README.md / README.en.md      <- report: goal, model, methodology, results (RU is primary, EN is a translation)
├── HARDWARE.md                   <- rig configuration and minimum to reproduce
├── RUNBOOK.md                    <- weight/runtime provenance, command order, troubleshooting
├── CHRONOLOGY.md                 <- phases, decisions and negative results
├── LICENSE-NOTICE.md             <- licenses for the engine, patches, model, scripts
├── CHECKLIST.md                  <- acceptance and publication checklist
├── study.json                    <- (optional) aggregator/plotter config (profiles, roles, A/B, steps)
├── .gitignore                    <- what is not committed (weights, runtimes, local drafts)
├── profiles/                     <- JSON profiles of configurations (runner input)
│   └── <profile>.json
├── prompts/                      <- JSON case prompts (runner input)
│   └── <prompt>.json
├── scripts/                      <- filled by copying from methodology/scripts/<category>/
│   ├── run_cache_sessions.py     <- runner/ (runner; run-*-matrix.sh wrappers next to it)
│   ├── run-serve-matrix.sh       <- runner/ (serve a profile: smoke/up)
│   ├── run-bench-matrix.sh       <- runner/ (ladder/A-B + report/plot/check/sanitize)
│   ├── host_telemetry.py         <- telemetry/ (when separate collection is needed)
│   ├── generate_report.py        <- report/  (aggregation into docs/)
│   ├── plot_context_curves.py    <- report/  (figures into docs/figures/)
│   ├── check-public.sh           <- publish/ (publication barrier)
│   └── sanitize-results.sh       <- publish/ (sanitizer)
├── data/                         <- small input data (sha256, curves, summaries)
│   └── model-checksums.txt
├── docs/                         <- aggregates (generated)
│   ├── results-tables.md
│   ├── results.json
│   └── figures/
│       └── <metric>.svg
└── results/                      <- raw runs (one directory per run)
    └── <run_id>/
        ├── result.json
        ├── profile.json
        ├── command.json
        ├── server.log
        └── telemetry.csv
```

Notes:

- **`README.md` is the primary human-readable document.** All conclusions and
  numbers live here; the other files are the source or the detail.
- **`HARDWARE.md`** — the rig, device binding and the minimum at which
  reproduction makes sense.
- **`RUNBOOK.md`** — the exact command order and provenance (weights, runtime,
  manifests).
- **`CHRONOLOGY.md`** — what was tried, why, what did not work (negative results
  are preserved, not deleted).
- **`profiles/`** — declarative inputs; they use the `<repo>` placeholder instead
  of absolute paths.
- **`prompts/`** — texts/structures of every prompt used; mandatory (see §4.2
  and the "Prompts policy" below).
- **`scripts/`** — all executable code; it refers to relative paths. The
  skeleton ships **no** scripts: `<case>/scripts/` is filled by copying from the
  methodology's canonical portable versions
  (`methodology/scripts/<category>/`: `runner/`, `telemetry/`, `report/`,
  `publish/`) during adaptation; categories and exact sources are in
  `methodology/scripts/README.md` and
  `templates/research-case/scripts/README.md`.
- **`study.json`** — optional aggregator/plotter config: title, list of profiles
  with roles, planned A/B and ladder steps (see §4.5). When absent, the
  profiles/labels/orders are auto-detected from the raw runs.
- **`data/`** — only small text/CSV artefacts; weights are not placed here.
- **`docs/`** — generated aggregates and figures; the aggregator never
  overwrites raw runs.
- **`results/`** — raw runs; failed runs are preserved as is.

## 4. File formats

### 4.1. Profile `profiles/<name>.json`

A profile describes one configuration and is the runner input. Fields:

| Field | Type | Purpose |
| --- | --- | --- |
| `name` | string | Profile name (matches the file name without `.json`) |
| `engine` | string | Inference engine (`llama.cpp`, `vllm`, …) |
| `model_path` | string | Path to the weights **relative to the repo root** |
| `served_model_name` | string | Model alias in the API |
| `binary` | string | Path to the server binary |
| `build_variant` | string | Build variant name (manifest key) |
| `gpu` | object | `{pci_bus_id, cuda_visible_devices, llama_device}` |
| `ctx_size` | int | Context size |
| `parallel` | int | Number of parallel slots |
| `cache_type_k` | string | KV cache type for K |
| `cache_type_v` | string | KV cache type for V |
| `seed` | int | Fixed generation seed (decode comparability) |
| `temperature` | number | Generation temperature (use `0` for measurements) |
| `ignore_eos` | bool | Do not stop at EOS; generate exactly `n_predict` tokens |
| `n_predict` | int | Number of generated tokens (fixed for comparability) |
| `port` | int | Server port |
| `command_env` | object | Launch environment variables |
| `command` | array[string] | Actual command; `<repo>` is the repo-root placeholder |
| `note` | string | Human-readable note about the profile's purpose |
| `vram_note` | string | Measured VRAM layout (buffers, peak, headroom) |

Generation is fixed explicitly: `seed`, `temperature: 0`, `ignore_eos: true`
and `n_predict` (correctness markers — `finish_reason = length`,
`completion_is_fixed = true`). `--seed` and `--n-predict` are duplicated in
`command`; `temperature`/`ignore_eos` are sent by the runner in the API request
body (`llama-server` treats them as per-request parameters). Mode flags that
matter for comparability and reproduction also belong to `command` and the
profile: `--fit off`, `--kv-offload`, `--flash-attn on`, `--cache-prompt`,
`--cache-ram`, `--kv-unified`, `--timeout`, `--metrics`, `--reasoning off`.

The prompt-cache mode is set by the profile key `cache_prompt` (`false` is the
cold mode). In cold mode the runner automatically checks the "cache hit ≈ 0"
invariant: if `cache_hit_fraction` exceeds the `cold_cache_hit_tolerance`
threshold (profile value, default `0.005`) or the cache hit is unknown, the step
is not comparable and the reason is recorded in `step_comparable_reason`
(`cold_cache_hit_exceeded` / `cold_cache_hit_unknown`); the threshold itself is
stored in `result.json` (`cold_cache_hit_tolerance`).

> **The full actual `command` is the source of truth.** The summary fields
> (`ctx_size`, `cache_type_*`, `seed`, `n_predict`, mode flags) must match the
> command; if they disagree, the command wins and the fields must be fixed.

Example — `templates/research-case/profiles/example.json` (valid JSON with a
`_note` field containing fill-in hints; expanded to the completeness of a real
reference profile).

### 4.2. Prompts `prompts/*.json`

Prompts are mandatory, even if the case "does not use" any. The `prompts/`
directory records everything that generated the inputs, so a reader can
reproduce the requests:

- prompt texts/structures for every step and mode (JSON);
- ladder-generator configuration (filler unit/`FILLER_UNIT`, prefix, token
  truncation method — see `METHODOLOGY.md` §4.3);
- request parameters (role, message format, `max_tokens`).

If there are no user prompts, attach a file describing how the inputs were
built (what generated the prompt, which steps, which tokenizer/`/tokenize`
service). An empty `prompts/` is not allowed.

### 4.3. Run contents `results/<run_id>/`

`<run_id>` is `<timestamp>-<profile>-<mode>`, e.g.
`20260101-120000-<profile>-ladder`.

| File | Purpose |
| --- | --- |
| `result.json` | Run outcome: status (`ok`/`failed`/`timeout`/`non_comparable`), error, structured `status_reasons`, steps (prefill/decode, cache-hit, fixed generation, `finish_reason`, `step_comparable`, `timing_delta_pct`, `overhead_s`), `props_check`/`offload_check`, telemetry maxima, clocks |
| `profile.json` | Copy of the profile used (records the input as is) |
| `command.json` | Actual command, environment, port, seed, mode/order |
| `server.log` | Server stdout+stderr (buffers, warnings, errors) |
| `telemetry.csv` | Samples (typically 0.5 s): RAM/swap, CPU, temperature, per-GPU temperature/utilization/memory/power; process memory |

Failed runs are preserved with the `failed`/`timeout`/`non_comparable` status and
the error text/structured cause; they are neither deleted nor overwritten. A run
with a broken invariant is marked `non_comparable` (fail-closed): on a `/props`
mismatch the heavy requests are skipped, and the reasons go to `status_reasons`,
`props_check`/`offload_check` (host offload; `CPU_Mapped` is mmap-backed weights,
not an offload marker).

### 4.4. Aggregates `docs/`

| File | Purpose |
| --- | --- |
| `docs/results-tables.md` | Human-readable summary tables over the canonical runs |
| `docs/results.json` | Machine-readable summary: list of runs, canonical/control, steps; per-step energy metrics (`energy_j`, `power_avg_w`, `energy_per_output_token_j`, …) and the paired A/B delta (`ab_delta`, key `"profile\|order"`) |
| `docs/figures/*.svg` | Five auto-figures of `plot_context_curves.py`: `prefill-vs-context`, `decode-vs-context`, `ab-cache-hit`; `power-vs-load`, `energy-per-token` (the last two only when an energy series exists); per language — `*.ru.svg` / `*.en.svg` |

The aggregators are **read-only** with respect to `results/`: they only read raw
runs and write to `docs/`. The canonical result is the **aggregate over
repetitions**, and canonicality is decided **per step** (not per run): a step
enters the aggregate only when it is comparable — `completion_is_fixed = true`,
`finish_reason = length` and `step_comparable != false` (for new runs the
`step_comparable` field also covers `prompt_prefix_ok`; the aggregator checks
the flag itself). The
norm is **n ≥ 3** unique successful (`ok`) runs **per step**, aggregation —
**median + min/max**; `docs/results.json` records `n` and the list of `run_id`s
of all runs included in the step. Steps with **n < 3** are marked `limited` and
**do not enter the main tables/figures** — they are shown in a separate section.
The generation status of a run is `fixed` (all steps comparable), `mixed` (part)
or `variable` (none); for `mixed` runs the comparable steps enter the aggregate.
A single run's contribution to one `target_pct` is not duplicated; the profile
`n` = the number of runs with **≥ 1** comparable step. The warm-up step `0 %`
does not enter the aggregate. The selection rule and the list of
canonical/control runs are recorded in `docs/results.json`; the norm is in
`METHODOLOGY.md` §11.1 (canonical run selection and aggregation), figures —
§11.2.

> **Aggregation is done by the aggregator, not the runner.** The runner only
> writes raw runs (`result.json` with steps and `step_comparable`); the portable
> `generate_report.py` aggregates repetitions **per step** and marks as `limited`
> whatever falls short of **n ≥ 3**.

**Energy is an estimate from telemetry.** When `telemetry.csv` carries per-GPU
power (`gpuN_power_w`), the aggregator computes the step energy: `energy_j` is
the trapezoid integral of power over the actual `dt` inside the
`started_at_utc`/`finished_at_utc` window; `power_avg_w` is the average power;
`energy_per_output_token_j` / `energy_per_input_token_j` are joules per
output/input token (J/request is `energy_j`); `energy_dynamic_j` subtracts an
idle-window baseline — the median total power over `IDLE_WINDOW_S` = 10 s before
the first request (fallback — the run minimum; source in `idle_baseline_source`,
fields `idle_baseline_w` / `idle_window_s`). This is an **estimate** from
telemetry, not a
direct meter reading. Step labels `started_at_utc`/`finished_at_utc` are
required: without them or without telemetry there is no energy — this is **not a
blocker**, and the report prints an explicit "no data" line. The report section
is "Energy and power" (`render_energy`); the per-step metrics land in
`docs/results.json`.

**Paired A/B delta.** When `ab-sequence` runs exist, the aggregator pairs the
comparable steps of sessions A and B within one run, order and `target_pct` and
computes delta = 100·(B−A)/A over the matching steps (median delta, `n_pairs`,
`sign_consistency`); the `0 %` warm-up step is excluded. The result is the
"Paired A/B delta" report section (`render_ab_delta`) and `ab_delta` in
`docs/results.json` (key `"profile|order"`).

### 4.5. Study config `study.json`

`study.json` (example — `templates/research-case/study.example.json`) is an
optional input for the aggregator `generate_report.py` and the plotter
`plot_context_curves.py`; it defines the report and figure presentation. Fields:

| Field | Type | Purpose |
| --- | --- | --- |
| `title` | object | Study title `{ru, en}` (report header, figure captions) |
| `profiles` | array | List of variants: `{name, label{ru,en}, role}`; `name` matches `profile` in `results/<run_id>/result.json`, `role` is `main` (main series) or `control` (control series) |
| `ab_planned` | array[string] | Profiles for which the A/B cache-hit series is required |
| `ab_orders` | array[string] | Expected A/B order strings (e.g. `ABAB`, `ABBABAA`) |
| `ladder_pcts` | array[int] | Context-fill steps (percent of `ctx`) the ladder is expected to cover |

The file's absence is **allowed**: the aggregator and plotter auto-detect
profiles, labels, A/B orders and steps from the raw runs (every profile becomes
`main`). The lists `ab_planned`/`ab_orders`/`ladder_pcts` are optional as well.
The config is looked up as `./study.json`; the path is overridden with
`--config`.

## 5. What is published and what is not

| Artefact | Publication |
| --- | --- |
| Model weights | **never commit**; publish only URL, commit, size and sha256 |
| Runtimes/binaries/builds | never commit; publish the manifest (commit, flags, sha256) and the build script |
| Raw logs and `results/` | only **after sanitizing the publication copy** (`scripts/sanitize-results.sh --in-place`; the script is copied from `methodology/scripts/publish/`). `--in-place` overwrites files, so apply it to the **publication copy** or make an archive/backup of the original `results/` first — private original runs must not be lost |
| Secrets, tokens, keys | **never** (neither in texts nor in logs) |
| Absolute home paths | **never**; only relative paths and `<repo>` |
| Host names, internal IPs | **never** |
| Proprietary drivers/modules | **never**; only the source link and tag |
| Scripts, profiles, documentation | published "as is" (usually MIT), with a notice in `LICENSE-NOTICE.md` |

Third-party component licenses (engine, patches, model) are collected in
`LICENSE-NOTICE.md`; for each one — license, source and exactly what is reused.

## 6. Publication barrier

Before publishing, the directory goes through two tools. Both are copied into
`<case>/scripts/` from `methodology/scripts/publish/` (see
`methodology/scripts/README.md`):

1. **`scripts/check-public.sh`** — a recursive walk of the case tree excluding
   service directories (`.git/`, `build/`, `dist/`, `__pycache__/`,
   `node_modules/`, `*.pyc`). For each file — `grep` over generic patterns: home
   paths (`/home/<user>/`, `/root/`, `/Users/<user>/`), mount points (`/mnt/`),
   private IPv4 (RFC1918: `10/8`, `172.16/12`, `192.168/16`), secrets matching
   the exact `sk-[A-Za-z0-9_-]{13,}` pattern, the `Authorization: Bearer`
   header and `api_key`/`secret`/`token` assignments. Returns **exit 2** on a hit
   or a scan error and **exit 0** only when clean (fail-closed). Patterns are
   extended via `PUBLIC_DENY_FILE`, false positives are dropped via
   `.public-allow`; run with `bash scripts/check-public.sh`. **Host names and
   internal identifiers are not covered by the baseline:** add them via
   `PUBLIC_DENY_FILE` and/or review them manually during publication review.
2. **`scripts/sanitize-results.sh [--in-place]`** — without the flag it only
   reports hits and fails (exit 2); with `--in-place` it rewrites private paths,
   names, hosts and secrets in `results/`, `docs/`, `prompts/` and top-level
   `*.md/*.json/*.csv/*.log`. Apply `--in-place` to the **publication copy** of
   the tree (or make an archive/backup of the original `results/` before running):
   the command is destructive and may erase private original runs.

Policy and exceptions:

- The sanitizer and the barrier must reach the **same verdict** on the same tree
  — detection patterns are kept in sync.
- **Loopback `127.0.0.0/8` is not a private marker** (the server listens
  locally).
- **llama.cpp progress timestamps of the form `H.MM.mmm.uuu`** visually resemble
  an IPv4 address but are not one; the first timestamp field may fall into `10`,
  i.e. into the `10/8` range. Therefore the generic IPv4 pattern is not applied
  at all, and the barrier's RFC1918 scan is protected by a **guard**: the leading
  timestamp token is stripped from the line and an IP hit is confirmed only if a
  real address remains afterwards (the other patterns are checked against the raw
  line and are not weakened). During sanitizing the leading timestamp is hidden
   behind a temporary marker. An example of this guard is implemented in the
   methodology's `methodology/scripts/publish/check-public.sh`.
- Internal project markers (control-plane package/directory names) do not get
  published and are replaced with placeholders.

## 7. Proposed `.gitignore`

Full text (the same file lives in the skeleton as
`templates/research-case/.gitignore`):

```gitignore
# Model weights are never committed: publish URL + sha256 only.
data/models/
models/
*.gguf
*.safetensors

# Runtimes and build trees are reproducible from source; do not commit binaries.
# Binary patterns are scoped to build/runtime directories so they do not mask
# small text artefacts elsewhere in the tree.
tools/llama-cpp/
llama.cpp/runtime/
llama.cpp/src/
llama.cpp/runtime-*/
llama.cpp/src/build/
llama.cpp/src/build-*/
dist/
build/
dist/**/*.o
dist/**/*.so
dist/**/*.so.*
dist/**/*.a
build/**/*.o
build/**/*.so
build/**/*.so.*
build/**/*.a
tools/**/*.o
tools/**/*.so
tools/**/*.so.*
tools/**/*.a

# Raw benchmark runs ARE committed (published after sanitizing), so results/
# is not ignored by default. Unsanitized local drafts live in results.local/
# and are never committed. Before committing, sanitize results with the case's
# own sanitize-results.sh (copied from methodology/scripts/publish/ into
# scripts/; see scripts/README.md) -- the skeleton does not ship that script.
results.local/

# The study config (study.json) is tracked: it is published metadata for the
# report/figures and is intentionally absent from this ignore list.

# Python cache and bytecode.
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# Local virtual environments and caches.
.venv/
venv/
.cache/

# Temporary and editor files.
*.tmp
*.swp
*~
.DS_Store
```

Raw runs are **published** by default (after sanitizing, as in any published
case): `results/` is not ignored, and unsanitized drafts are kept in
`results.local/`. Before committing, run the sanitizer (the script is copied from
`methodology/scripts/publish/`) on the **publication copy** of the tree (or make
an archive/backup of the original `results/` first — `--in-place` is
destructive):

```bash
bash scripts/sanitize-results.sh --in-place
bash scripts/check-public.sh
```

## 8. Publication checklist

1. `HARDWARE.md` is filled in: CPU, RAM, disk, OS, driver, toolchain, GPU table
   (model, PCI bus, VRAM, compute, role).
2. Model provenance is recorded: URL, commit, size and sha256 of each file; the
   weights are absent from the tree.
3. Runtime provenance is recorded: engine, commit, build flags, manifest, binary
   sha256; the binaries are absent from the tree.
4. `RUNBOOK.md` reproduces the full command order and includes troubleshooting.
5. `CHRONOLOGY.md` contains phases, decisions and negative results.
6. Profiles in `profiles/` are valid, real (not `example.json`) and use
   `<repo>`/relative paths; generation is fixed (`seed`, `temperature: 0`,
   `ignore_eos`, `n_predict`).
7. `prompts/*.json` are filled in: every prompt used and the ladder-generator
   configuration; if there are no user prompts, a file describing how the
   inputs were built is attached.
8. Scripts are copied from `methodology/scripts/<category>/` into
   `<case>/scripts/` and run from the case directory: runner + wrappers
   (`runner/`), telemetry in place, aggregator and figures (`report/`), barrier
   (`check-public.sh`) and sanitizer (`sanitize-results.sh`) — from `publish/`;
   see `methodology/scripts/README.md`.
9. `study.json` (if used) describes the title, profiles with roles,
   `ab_planned`/`ab_orders` and `ladder_pcts`; when absent, auto-detection from
   the raw runs yields correct profiles and labels.
10. `docs/results-tables.md`, `docs/results.json` and figures are generated from
    the canonical runs. The canonical result is the aggregate over repetitions
    **per step** (per-step canonicality): a step enters when it is comparable
    (`completion_is_fixed` + `finish_reason = length` + `step_comparable`) and
    has **n ≥ 3** unique `ok` runs at that step (**median + min/max**, with `n`
    and the list of `run_id`s recorded); steps with `n < 3` are marked `limited`
    and shown separately, outside the main tables/figures; the run status is
     `fixed`/`mixed`/`variable`. A figure is mandatory **when the corresponding
     measured series exists**: the five
     automatically generated figures of `plot_context_curves.py`
     (`prefill-vs-context`, `decode-vs-context` — if a ladder was measured;
     `ab-cache-hit` — under A/B; `power-vs-load`, `energy-per-token` — when an
     energy series exists), while additional per-axis plots
     (layout/ubatch/speculation) are built separately by the author when a series
     exists. If the series was not measured, the figure is not built and the
     reason is recorded in `README.md`.
11. `bash scripts/sanitize-results.sh` (without the flag) finds no markers.
12. `bash scripts/sanitize-results.sh --in-place` is applied to the **publication
    copy** of the tree (or an archive/backup of the original `results/` was made
    before it) so that private original runs are not lost.
13. `bash scripts/check-public.sh` returns `OK: no private markers found`.
14. `LICENSE-NOTICE.md` is verified: engine, patches, model, scripts.
15. Published texts contain only relative paths and placeholders; host names,
    internal IPs and secrets are absent.
16. (Conditional: only for the monorepo; not applicable to a standalone
    repository.) The case index (`inference_cases/README.ru.md`) is updated with
    a link to the new case.

## 9. How to adapt the template to your model/hardware

The skeleton is a **frame**: copy `templates/research-case/` into a new case
directory, then **add the required artefacts** and fill in the placeholders.

Fill in:

| What | Replace with |
| --- | --- |
| `<MODEL>` | the model name throughout the documents |
| `<HARDWARE>` | a short rig name (e.g. `2xCMP50HX+RTX2080Ti`) |
| `<CTX>` | the context size (e.g. `262144`) |
| `<RUNTIME>` | the runtime variant name (manifest key) |
| `<repo>` | the root of your inference repository at run time |
| `profiles/example.json` | real configuration profiles |
| `study.example.json` | `study.json` with your profiles/roles (or delete the file — auto-detection) |
| `prompts/.gitkeep` | real `prompts/*.json` (every prompt used) |

Copy the scripts from the methodology into `<case>/scripts/` (absent from the
skeleton):

- runner and wrappers — `methodology/scripts/runner/`
  (`run_cache_sessions.py`, `run-serve-matrix.sh`, `run-bench-matrix.sh`);
- telemetry — `methodology/scripts/telemetry/host_telemetry.py` (when separate
  collection is needed; the runner collects telemetry itself);
- aggregator and figures — `methodology/scripts/report/`
  (`generate_report.py`, `plot_context_curves.py`; a figure is built only **when
  the corresponding measured series exists** — see §2, §8);
- barrier and sanitizer — `methodology/scripts/publish/` (`check-public.sh`,
  `sanitize-results.sh`) — always needed.

Then add real profiles, prompts and raw runs (`results/`).

Then verify:

- `HARDWARE.md` — real PCI bus IDs, VRAM, compute capability, driver;
- `RUNBOOK.md` — actual download and sha256 verification commands;
- `profiles/` — model/runtime paths and `CUDA_VISIBLE_DEVICES` binding;
- `scripts/` — telemetry thresholds, ports, run modes;
- `LICENSE-NOTICE.md` — the licenses of your specific model and patches;
- `check-public.sh` / `sanitize-results.sh` — for your private markers (and keep
  the barrier's llama.cpp timestamp guard).

Once filled in, go through the checklist in section 8.
