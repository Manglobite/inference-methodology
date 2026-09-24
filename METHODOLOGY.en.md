# Methodology for local LLM inference research

[Русский](METHODOLOGY.md) | **English**

> A portable, hardware-agnostic methodology. It lets another researcher or agent
> on **different hardware** and with a **different model** conduct a maximally
> objectively comparable study of local LLM inference. All concrete numbers,
> names, and paths of a stand below are **examples only** (marked); what is
> mandatory is the principles, the protocol, and the set of artifacts, not the
> values.

- **Audience.** An executing agent (runs the measurements and writes the
  artifacts) and a human researcher (sets the task, interprets, publishes).
- **How to use.** Read sections 1–2, go through the checklist in section 0, then
  run the experiment following sections 3–12. Sections 13–16 are for before
  publication and when analyzing results.
- **Cross-references.** `hardware_info/HARDWARE.md` — description of a concrete
  stand and the parameter matrix (fill it in for your hardware); `TEMPLATE.md` —
  packaging of a published case.
- **Tools.** The portable methodology scripts live in `scripts/` by category:
  `scripts/runner/` (runs), `scripts/telemetry/` (telemetry), `scripts/report/`
  (aggregation and plots), `scripts/publish/` (barrier and sanitizer). They are
  not tied to hardware or a case and are copied into `<case>/scripts/`
  (see `scripts/README.md`).
- **Placeholders.** `<MODEL>` — model; `<MODEL_PATH>` — path to the weights file;
  `<CTX>` — configured context; `<GPU>` — accelerator, `<GPU_LIST>` — list of
  accelerators (for example, `nvidia-smi -i <GPU_LIST>`); `<RUNTIME>` — runtime
  build/variant, `<SERVER_BINARY>` — path to the server binary; `<PORT>` — server
  port; `<DEVICE_LIST>` — device order for `--device`; `<SEED>` — fixed seed;
  `<N_PREDICT>` — fixed generation length; `<CONTROL_PLANE_URL>` — address of your
  node's control plane; `<RESULTS_DIR>` — results directory; `<CACHE_RAM>` — host
  prompt-cache size (`--cache-ram`).

**Confidence labels** (mandatory for every effect statement):

| Label | Meaning |
| --- | --- |
| **measured** | a direct value from an instrument, server timing, or telemetry |
| **computed** | arithmetic over measured values (Δ, percentages, headroom) |
| **interpretation** | a causal explanation not directly following from the data |

## 0. Agent quick start (checklist)

1. Check free `<GPU>` and disk space (`nvidia-smi`, `df -h`).
2. Run the stand adapters — stop the production backend and disable
   frequency/VRAM parking, only if applicable on your stand (section 2).
3. Record provenance: commit and build flags of `<RUNTIME>`, sha256 and size of
   `<MODEL>`, driver and CUDA versions, `<GPU>` topology/order, CPU/RAM/OS.
4. Build the configuration profile: explicitly separate fixed and varied
   parameters (section 3).
5. Run a smoke probe: load, `/props` (`n_ctx = <CTX>`), one short request, peak
   VRAM.
6. Make sure the model **and** the KV cache fit entirely in VRAM (section 5);
   otherwise do not measure speeds until this is resolved.
7. Run the `ladder` on the first configuration (section 4).
8. For comparable configurations, use A/B mode in one order, then in the reverse
   order (`ABAB` and `ABBABAA`), within one session.
9. Repeat the key runs: a canonical result requires **n ≥ 3** successful (`ok`)
   runs per **step** (per-step canonicality, section 11.1). Preserve **all** runs,
   including failed ones, with a status (`ok | failed | timeout |
   non_comparable`, section 7).
10. Produce the machine-readable summary and tables, build the mandatory plots,
    label measured/computed/interpretation, and pass the publication barrier
    (sections 11, 15).

## 1. Research invariants

| Invariant | Why | How to verify |
| --- | --- | --- |
| One heavy process at a time | exclude mutual distortion of runs | `nvidia-smi` before start; the runner refuses to start a second server |
| Model **and** KV entirely in VRAM | do not measure eviction instead of speed | `/props`, log lines, VRAM headroom (section 5) |
| Parallelism 1 (`--parallel 1`) | stable timings, one slot and prompt cache | profile flag; `/props` → `total_slots` |
| Determinism: `temperature 0`, fixed `seed`, fixed generation length | comparable decode across runs | `ignore_eos` + `n_predict`; `finish_reason = length` |
| One axis at a time | do not mix effects | the comparison differs in exactly one parameter |
| A/B interleaving within one session | remove drift of clocks/temperatures between "eras" | order `ABAB`/`ABBABAA`, not "first all A, then all B" |
| Raw server timings are the primary metric | the engine's own metric, not the network's | parse `prompt eval time` / `eval time` from `server.log` |
| HTTP end-to-end is the confirmatory metric | catches overhead and timeouts | client `elapsed` next to server timings |
| Telemetry always | peak VRAM, headroom, temperatures, RAM/swap | `telemetry.csv` (section 10) |
| Failed runs are not deleted | preserve the negative result and the cause | run directory with a status (`failed`/`timeout`/`non_comparable`) |
| Fact separated from interpretation | readability and verifiability | measured/computed/interpretation labels |

**Rule.** If an invariant condition is violated, the result is marked with the
`non_comparable` status and does not enter the main tables/plots (but is preserved
as is). The full run-status policy is in section 7.

## 2. Preflight

The order is mandatory; each item is recorded in the run artifacts.

### 2.1. Mandatory preflight checks

1. **Free `<GPU>`.** `nvidia-smi -i <GPU_LIST>` — no third-party heavy processes;
   VRAM free. One heavy process at a time.
2. **Disk.** `df -h` for the root and separately for the volume holding the
   weights (`<MODEL_PATH>`): enough for `<MODEL>` plus build/run artifacts.
3. **Smoke probe.** Start the server, check `/props` (`n_ctx = <CTX>`,
   flash-attention), send one short request, capture peak VRAM, stop the server.
   Protection against OOM on multi-hour runs.
4. **Provenance** (record before the start, into a machine-readable file):

| Field | What to record |
| --- | --- |
| Runtime | commit, branch/fork, build flags, binary path and sha256, `<RUNTIME>` variant |
| Model | source, revision commit, file name, size in bytes, sha256 |
| Driver/CUDA | driver version, CUDA version, build architectures (`CMAKE_CUDA_ARCHITECTURES`) |
| Topology | list of `<GPU>`: model, VRAM, PCI bus id, compute capability, links (PCIe gen/width, P2P) |
| Device order | `CUDA_DEVICE_ORDER=PCI_BUS_ID` and the actual `--device` |
| Host | CPU (model/thread count), RAM, swap, OS/kernel |

### 2.2. Stand adapters (only if applicable on your stand)

These steps depend on the infrastructure of a particular node; they may be absent
on another stand.

1. **Stop the production backend.** Through your node's control plane
   (`<CONTROL_PLANE_URL>`): find the PID and stop it. If there is no control
   plane, stop it manually. Do not touch production without the owner's decision.
2. **Disable idle frequency parking.** If the control plane parks `<GPU>` when
   idle (for example, down to 300 MHz) and holds VRAM, disable parking for the
   duration of the series and wake the cards. **Interpretation:** with a parked
   card all measurements are invalid. Restore the previous policy after the
   series. Apply the clock reset `nvidia-smi -i <GPU_LIST> -rgc` only if it is
   applicable and permitted on your stand.

## 3. Experiment axes

| Axis | What varies | What must be fixed for a comparison |
| --- | --- | --- |
| KV quantization | `--cache-type-k` / `--cache-type-v` (`f16`, `q8_0`, `q4_0`, …) | weights, `<CTX>`, layout, runtime |
| Weight quantization | `<MODEL>` file (`Q4_K_M`, `Q8_0`, …) | KV quantization, `<CTX>`, layout |
| Layout across `<GPU>` | number of cards, `--split-mode`, `--tensor-split`, `--device` | weights, KV, `<CTX>` |
| µbatch | `--ubatch-size` | weights, KV, `<CTX>`, other env |
| Speculation/MTP | enablement and depth (`--spec-draft-n-max`) | baseline decode without speculation as control |
| Context size | `<CTX>` | weights, KV quantization, layout |
| Runtime/build variant | `<RUNTIME>` (baseline vs patch) | model, KV, `<CTX>`, layout |
| Device order | `--device` (order of cards) | everything else; verify byte-identical output |
| Prompt cache | `cache_prompt` (`true`/`false`) | weights, KV, `<CTX>`, layout; with `false` — the `cache_hit ≈ 0` invariant |

**One-change rule.** One comparison — exactly one difference. If a sweep changes
several parameters at once, it is a **confound**: its contribution is not
isolated, and such a conclusion is marked as non-comparable (see section 16).

## 4. The ladder principle (ladder) — central section

**Definition.** `ladder` is a sequence of requests in **one server session** in
which context fill grows in rungs, and metrics are captured at each rung.

### 4.1. Rungs

- Rungs: **0 / 10 / 30 / 60 / 80 %** of `<CTX>`.
- Rung **0 %** is a short warm-up (example: "Hello", ~12 tokens). It is
  **excluded from analysis**: it shows a one-off startup effect. The warm-up token
  count is expected/verifiable, not a universal constant; record the actual token
  counts of the warm-up.
- Rungs 10–80 % enter the analysis. Example (not mandatory): with
  `<CTX>` = 262144 this is 26214 / 78643 / 157286 / 209715 tokens.

### 4.2. Cumulativity of rungs

The rungs run in **one session** and use a **shared prefix**: each next rung
extends the previous prompt. These are increments (example: +10 / +20 / +30 /
+20 pp of fill), not independent levels. **Hypothesis (expected, verifiable):**
because of the shared prefix, `cache_hit_fraction` grows across rungs. This is not
a universal fact: record the actual token counts of every rung and **record
non-monotonicity** if it is observed.

### 4.3. Deterministic prompt assembly

- The prompt is built by extending a prefix: a base prefix + a repeating filler,
  trimmed so that the token count does not exceed the target.
- The token count is verified against the server (`/tokenize`), not estimated
  from bytes.
- For each rung it is checked that the new prompt starts with the previous one
  (`prompt_prefix_ok` flag). This guarantees prefix reuse.
- The assembly must be reproducible: the same input yields the same prompt.

### 4.4. Semantics of `cache_hit_fraction`

- `cache_hit_fraction` is the fraction of the prompt taken from cache:
  `1 − evaluated_tokens / prompt_tokens` (or `cache_n / (cache_n + prompt_n)`).
- **Hypothesis (expected, verifiable):** the fraction grows across cumulative
  rungs because the prefix is reused; at the first rung 0 is expected. This is not
  a guaranteed fact: record the actual token counts and record non-monotonicity.
- Configurations can be compared by speed **only at equal prompt length** (equal
  fill). Do not compare prefill/decode values across different rungs.
- **Optional cold mode.** The profile may set `cache_prompt = false` (a
  configurable axis, section 3). Then a hard invariant applies: `cache_hit ≈ 0`
  at every rung; when it does not hold, the rung is marked `non_comparable`. Do
  not expand cold mode into a full protocol — only as a separate comparison axis.

### 4.5. Adapting to your `<CTX>`

- The rung fractions (10/30/60/80 %) are portable; absolute lengths are not.
- The rung grid may be set by the profile (`ladder_pcts`, read by the runner from
  the profile; see also `study.json`, section 11.2); if it is set, it is
  **primary** over the default fractions, and auto-detection is used only in its
  absence.
- With a small `<CTX>`, the upper rungs may hit the slot limit: change the grid
  (for example, add a rung) or set rungs in tokens.
- In the report, state both the fraction and the actual token count
  (`prompt_tokens`).

### 4.6. A/B mode

- Two sessions (A and B) alternate **in one slot** following a given order; the
  pattern is applied cyclically until both sessions have passed every rung of the
  ladder.
- The order is set by the caller: the runner accepts `--order` (default `ABAB`).
  The normative set of orders for an A/B series is `ABAB` and `ABBABAA`; running
  **both** contract orders is required.
- A/B comparison is done **at matching rungs**.
- An A/B cache-hit measurement is valid only if both sessions fit into the host
  prompt cache (`--cache-ram`). If they do not fit, the measurement runs with
  eviction, and A/B is not performed for that configuration; only a single ladder
  is captured.
- **Measured** (example): with a KV quantization where the two sessions together
  are smaller than `--cache-ram`, cache-hit values are identical for A and B and
  for both orders.
- **Counterbalancing for comparing configurations.** For A/B of **different
  configurations** (a flag change with a server restart) alternate `A,B,A,B`, not
  "all A, then all B": consecutive blocks accumulate clock/temperature drift. For
  session A/B (the order of requests within one session as the axis), the orders
  `ABAB`/`ABBABAA` remain as above.

## 5. Placing the model and KV entirely in VRAM

### 5.1. How to check

- Full offload flag (example: `-ngl all`); all layers on `<GPU>`.
- `/props` → `n_ctx` matches `<CTX>`.
- `server.log` lines: `model buffer size`, `KV self size` / `KV buffer size`,
  `compute buffer size`, `RS buffer size`, plus per-device sizes
  (`CUDA0 model buffer size`, …).
- **For every device**, verify the VRAM allocation (model + KV + compute/RS)
  against the log and telemetry: the sum of buffers from the log must be
  consistent with the card specification and the observed VRAM; any discrepancy
  is recorded.
- **The sign of partial offload is the layer count:** the line
  `offloaded X/Y layers to GPU`; with X < Y the configuration is marked as
  **partial offload**, not "full VRAM", and its speeds are not compared with
  full offload.
- **Informational fields (NOT a sign of partial offload):** a non-empty
  `CPU_Mapped model buffer size` — this is memory-mapped (mmap) weights, not
  CPU inference; `CUDA_Host output/compute buffer size` — also not a sign.
  Record them as informational fields.
- **Signs that the model/KV buffer is on the CPU:** a model or KV buffer on the
  CPU (not `CPU_Mapped`, not `CUDA_Host`) with a non-zero value (threshold >
  0.01 MiB). On any such sign the configuration is marked as **partial
  offload**, not "full VRAM", and its speeds are not compared with full
  offload.
- Peak VRAM and headroom — from telemetry: peak = the maximum `memory.used` per
  card over the run; **computed:** headroom = card VRAM − peak VRAM. **VRAM
  headroom ≥ 0 at all rungs of the ladder**, not only at the maximum; on eviction
  or partial offload the run is marked with the `non_comparable` status
  (section 7).

### 5.2. What to do if it does not fit

1. Lower the KV quantization (for example, `f16` → `q8_0` → `q4_0`).
2. Reduce `<CTX>`.
3. Redistribute layers (`--tensor-split`, `--split-mode layer`) or add a card.
4. As a last resort — a separate card for the model.

After each step, run the smoke probe again and check headroom. The narrow card is
the one that will fail with OOM on a longer prompt; use it as the reference.

### 5.3. Honest labeling

- **Partial offload** (`offloaded X/Y` with X < Y; or a model/KV buffer on the
  CPU, other than `CPU_Mapped`/`CUDA_Host`) — state it explicitly in the report;
  do not compare such numbers with full offload. `CPU_Mapped` is mmap of the
  weights, not a sign.
- **Host prompt-cache eviction** (lines like
  `making room for prompt cache entry, removing oldest entry`) — expected with
  large KV; note that cache-hit is distorted in this mode.
- **OOM** — the run is preserved as `failed`, and the cause is recorded.

## 6. Parallelism 1

- `--parallel 1` — one slot. **Interpretation:** with a single slot there is no
  request contention for the slot/batch, timings are stable, and the prompt cache
  and A/B behave predictably.
- Slots and prompt cache: A/B sessions share one slot; `--cache-prompt`,
  `--cache-ram`, `--cache-idle-slots` control KV eviction to RAM.
- **Concurrency is a separate axis.** Do not mix measurements at `--parallel 1`
  and `concurrency 1` with multi-threaded/parallel requests (`--parallel 2..N`):
  that is a different experiment with a different answer (section 3).

## 7. Measurement protocol

**Sequence:** `smoke` → `ladder` → `A/B`.

1. **Smoke** before each wave/series: load, `/props`, peak VRAM.
2. **Ladder** — the main "speed versus fill" curve.
3. **A/B** — control of prompt-cache reuse and order (where both sessions fit
   into the host cache).

### 7.1. Run statuses

Every run has exactly one status:

| Status | Condition | In main tables/plots |
| --- | --- | --- |
| `ok` | the run completed, section 1 invariants hold | yes (by comparable steps) |
| `failed` | error, OOM, server crash, absence of expected timings | no (preserved as is) |
| `timeout` | client/server timeout, abort before completion | no (preserved as is) |
| `non_comparable` | the run completed but a section 1 invariant is violated; the status is set by the runner | no (preserved as is) |

- **Division of responsibility.** The runner sets `timeout`/`failed` by the
  exception type (a timeout is `timeout`, everything else is `failed`) and
  **flags** invariant violations (status `non_comparable` + structured reasons in
  `status_reasons`, of kind `kind=invariant|exception`). **Step** comparability
  is expressed by the runner as the `step_comparable` flag
  (= `completion_is_fixed` AND `finish_reason = length` AND
  `prompt_prefix_ok`), not by the run status. Canonicality is built by the
  aggregator **per step** from `step_comparable`.
- **Step, not run.** A single ladder run may contain both comparable and
  non-comparable steps. Therefore a run's generation status is `fixed` (all steps
  comparable), `mixed` (some) or `variable` (none); the aggregator builds the
  canon by comparable steps, while `mixed`/`variable` runs are listed in a
  separate section.
- **TIMEOUT is separated from FAILED.** A timeout is not a crash but an abort by
  time; it is a separate `timeout` status, not `failed`.
- `non_comparable` runs are not "bad": they are preserved but do not enter the
  main tables and plots.
- **Unavailable mandatory telemetry fields.** If a mandatory telemetry field
  (section 10) was not collected — an instrument/interface is unavailable — the
  run is marked `non_comparable` when this affects comparability (for example,
  peak/headroom VRAM or temperatures cannot be verified). Silently skipping
  mandatory fields is **not allowed**: the fact and the cause are recorded in
  `result.json` and the report. Empty optional fields are allowed and do not
  change the status.

**Fixed generation.** `temperature 0`, fixed `seed`, `ignore_eos = true` and a
fixed `n_predict` (example: `<N_PREDICT>`, e.g. 128 tokens). The sign of
correctness is `finish_reason = length` and `completion_is_fixed = true`. A step
with `finish_reason != length` or `completion_is_fixed != true` gets
`step_comparable = false` and does not enter the aggregate (sections 7.1, 11.1).
**Interpretation:** without a fixed length, decode is not comparable across runs.

**Repeats and warm-up.** A canonical result for a step requires **n ≥ 3**
successful (`ok`) runs at that step (per-step canonicality, section 11.1). For
every series state `n` and the identifiers (`run_id`) of all included runs;
aggregation is **median + min/max** (section 11). At **n < 3** the step is marked
limited/non-canonical: it is preserved but not used for the main conclusions and
not presented as canonical. The warm-up request (rung 0 %) does not enter the
analysis.

**Isolation of long cases.** Long prompts within one server process share a
prefix, and prefill is understated (**prompt-cache contamination**). Measure long
cases as separate runs with `--repetitions 1`.

**Client timeouts.** The request timeout must exceed the expected prefill.
Example (not mandatory): a ~250k prefill takes ~950–1000 s, so a hard timeout of
900 s aborts the run. Configure the client timeout and the server `--timeout` for
your `<CTX>`; TIMEOUT ≠ failure — it is a separate status.

## 8. Metrics and their definitions

| Metric | Definition | Source | Type |
| --- | --- | --- | --- |
| prefill tok/s | prompt eval tokens / prompt eval time | `prompt eval time = … (… tokens per second)` in `server.log`; `timings.prompt_per_second` | measured |
| decode tok/s | eval tokens / eval time | `eval time = … (… tokens per second)`; `timings.predicted_per_second` | measured |
| TTFT | time to first token | streaming client | measured |
| elapsed | total request time | client, end-to-end | measured |
| `cache_hit_fraction` | fraction of the prompt from cache | `1 − evaluated/prompt`; `cache_n/(cache_n+prompt_n)` | computed |
| peak VRAM | maximum `memory.used` per card | `telemetry.csv` | measured |
| headroom | card VRAM − peak VRAM | telemetry + specification | computed |
| acceptance | `draft_n_accepted / draft_n`; mean draft length | `timings` / `draft acceptance` in the log | measured |
| average power (`power_avg_w`) | total `<GPU>` power integrated over the request window / window duration | `telemetry.csv` (`gpuN_power_w`); aggregator | computed (estimate) |
| request energy (`energy_j`) | integral of total power over the request window (trapezoids) | `telemetry.csv`; aggregator | computed (estimate) |
| dynamic energy (`energy_dynamic_j`) | integral of `max(0, P − base)`; `base` = the minimum total power over the run | `telemetry.csv`; aggregator | computed (estimate) |
| J/output-token (`energy_per_output_token_j`) | `energy_j / completion_tokens` | aggregator | computed (estimate) |
| J/input-token (`energy_per_input_token_j`) | `energy_j / evaluated_tokens` | aggregator | computed (estimate) |

- **Energy/power is an estimate.** The `gpuN_power_w` telemetry of consumer and
  mining cards is coarse, and the request window is given by the
  `started_at_utc`/`finished_at_utc` pair and approximated by edge interpolation
  of power. The aggregator computes the energy metrics when `telemetry.csv` and
  the timestamps are present (sections 10, 11.1); otherwise the fields are empty.
- **The canonical source of every metric** is given in the "Source" column. When
  both API fields `timings.*` and log lines are present, preserve **both** values
  and check the divergence between them.
- **The primacy of raw server timings is stated explicitly.** The canonical value
  is the raw server timing (the log line); the structured API fields `timings.*`
  are preserved as secondary confirmation, and their divergence from the log is
  recorded in the report. HTTP end-to-end (`elapsed`) is the confirmatory metric.
- **Result of the source audit (`docs/timing-source-audit.md`).** The API
  `timings.*` and the raw `log_timings` are numerically **identical**: max |delta|
  = 0.025%, with no pair at |delta| ≥ 0.1% (0/220); the divergence is only the
  loss of precision when the log line is printed to 2 decimals. Therefore the
  canon does not change: the raw log is primary, the API is secondary
  confirmation. The divergence is recorded in the `timing_delta_pct` field;
  significance thresholds: |delta| ≥ 0.1% — significant; 0.05–0.1% — grey zone
  (re-check); < 0.05% — normal (print rounding).
- If a timing field is absent in this build, take the equivalent (example:
  `raw_server_timing.tg` may be absent — take decode from
  `eval.tokens_per_second`).
- `started_at` in the report may mean the time the report was assembled, not the
  run start: do not use it as the start marker.

## 9. Server logging

- `server.log` is preserved **without filtering** (server stdout+stderr) — it is
  the primary source of timings and warnings.
- Parse: `prompt eval time`, `eval time`, `n_decoded`/`tg`, `draft acceptance`,
  buffer sizes (model/KV/compute/RS), `n_ctx`, `flash_attn`, prompt-cache
  eviction lines.
- **Expected warnings** (do not treat as errors):

| Warning | Meaning |
| --- | --- |
| no permission to change clocks (`clocks_reset.returncode: 4`, "does not have permission to change clocks") | GPU clocks are not changed by the process — expected |
| CORS / missing API-key | the server listens locally; a key is not needed |
| reasoning preserve "enabled by default" | the profile sets `--reasoning off`; does not affect measurements |
| `making room for prompt cache entry, removing oldest entry` | host prompt-cache eviction with large KV — expected |
| `rope_finetuned = unknown` | a GGUF metadata field; not an error and not an architecture warning |
| `srv stop: cancel task, id_task = …` | normal server shutdown at the end of a run |

- **What NOT to treat as an error:** the expected warnings from the table above.
- **What to treat as an error:** OOM, server crash, absence of expected timings,
  `Xid`/`AER` in the kernel log.

## 10. Telemetry

- **Normative format (schema version 1).** A single `telemetry.csv` format with a
  header; cadence 0.5 s; samples are written for the entire run.
- **Machine-readable schema version.** The first CSV column/field is
  `schema_version` with the value `1`. The consumer **must** check
  `schema_version` before interpreting; on a mismatch the file is **not
  interpreted** as schema 1.
- **The number of `<GPU>` is dynamic:** the `gpuN_*` columns repeat for every
  device `N` from `<GPU_LIST>`, not hard-coded `gpu0`–`gpu2`.
- **Columns** (M — mandatory, o — optional); the exact normative list:

| Group | Columns |
| --- | --- |
| Schema | `schema_version` (M; always `1`, first column) |
| Time | `timestamp_utc` (M, ISO-8601 UTC) |
| Host | `cpu_pct` (M); `cpu_temp_c` (M) + `cpu_temp_source` (M); `ram_used_gib` (M), `ram_total_gib` (M), `swap_used_gib` (M) |
| Per-core | `cpuN_pct` (M; dynamic, one per logical CPU: `cpu0_pct`, `cpu1_pct`, …) |
| Per-`<GPU>` | `gpuN_temp_c`, `gpuN_util_pct`, `gpuN_mem_used_mib`, `gpuN_power_w`, `gpuN_sm_clock_mhz` (M); `gpuN_pstate` (o) |
| Server process | `server_rss_kib`, `server_pss_kib`, `server_swap_kib` (M; sum over server processes); `llama_pids` (o) |

- **Clock snapshots are metadata, not telemetry.** `clocks_sm_before` /
  `clocks_sm_after` are recorded **once per run** in `result.json` (metadata),
  and **not** in `telemetry.csv`.
- **Value format:** CSV with a header; numeric values with fixed precision. Empty
  values are allowed **only for optional** fields; a mandatory field with an
  unavailable instrument/interface is not skipped silently — see the status rule
  in section 7.1.
- **Time synchronization:** timestamps in UTC so that telemetry can be matched
  against `server.log`.
- **`gpuN_power_w` is the energy source.** The integral of total power over a
  step (section 8) is computed by the aggregator from these columns; the request
  window is given by the step's `started_at_utc`/`finished_at_utc` pair (UTC,
  synchronized with `timestamp_utc`), while the energy itself is computed by the
  aggregator (`scripts/report/generate_report.py`), **not** by the telemetry
  collector. The column schema does not change.
- **Clocks:** capture `clocks.sm` (and P-state) before and after the run; idle
  parking of `<GPU>` makes measurements invalid.
- **Canonical collector.** `scripts/telemetry/host_telemetry.py` is the portable
  methodology collector (copied into `<case>/scripts/`); it implements schema 1
  and is invoked by the runner or standalone. Any other collector that does not
  gather per-core and SM clock and does not write a UTC timestamp **does not
  conform** to the schema and is not used as a telemetry source.

## 11. Aggregation and plots

### 11.1. Canonical run selection and aggregation

The canonical value for a configuration/rung is the **aggregate over repeats**,
not an individual run. Canonicality is decided **per step** (not per run): a step
enters the aggregate only when it is comparable by `step_comparable`
(`completion_is_fixed == true` AND `finish_reason = "length"` AND
`step_comparable != false`). The `step_comparable` field is optional: the runner
forms it as (`completion_is_fixed` AND `finish_reason = length` AND
`prompt_prefix_ok`), so for **new** runs all three conditions apply. The
aggregator (`scripts/report/generate_report.py`, `step_is_comparable`) does
**not** check `prompt_prefix_ok` directly and, for **legacy** runs without the
`step_comparable` field, relies on the first two conditions.

1. Exclude runs with the `failed`, `timeout`, `non_comparable` statuses, and
   within `ok` runs — steps that fail `step_comparable`.
2. The median/min/max are computed **only over comparable steps**; a canonical
   result for a step requires **n ≥ 3** unique runs at that step. Excluded steps
   are **not dropped silently** but listed with the reason
   (status/`finish_reason`/`completion_is_fixed`).
3. Steps with **n < 3** are marked `limited` and **do not enter the main tables
   and plots** — they are shown in a separate section.
4. A step's `n` = the number of **unique** runs at that step; one run's
   contribution to one `target_pct` is not duplicated. The profile `n` = the
   number of runs with **≥ 1** comparable step.
5. Document `n` and the list of `run_id`s of all included runs.

A run's generation status is `fixed`/`mixed`/`variable` (section 7.1): for
`mixed` runs the comparable steps enter the aggregate, while the excluded ones are
listed separately.

The aggregation is performed by the portable `scripts/report/generate_report.py`
(copied into `<case>/scripts/`): it reads the raw runs, excludes
`failed`/`timeout`/`non_comparable`, applies `step_comparable` per step, and
computes the median and min/max over repeats, documenting `n` and `run_id`;
limited steps are shown separately. The normative minimum is aggregation over
repeats, as described above.

The energy metrics (`energy_j`, `energy_dynamic_j`, `power_avg_w`,
`energy_per_output_token_j`, `energy_per_input_token_j`,
`energy_per_output_token_j_dynamic`) are aggregated exactly like the base
metrics: median + min/max over comparable steps, with `n` and `run_id`; they
enter the machine-readable `results.json` summary (per-step `metrics`) and a
separate report table. Empty values (no `telemetry.csv` or no timestamps) are
ignored, not treated as zero.

**Paired A/B delta.** The A/B comparison is computed not from separate A and B
medians but from pairs of matching steps within a single `ab-sequence` run: for
one `target_pct` and order (`order`) the comparable steps of sessions A and B
are paired, `delta_pct = 100·(B−A)/A`; the pair aggregate is the median Δ, the
number of pairs `n_pairs` and `sign_consistency` (the share of pairs whose Δ
sign matches the sign of the median). The warm-up `0 %` step is excluded;
different orders and percentages are never mixed. The result is stored in
`results.json` under the key `"profile|order"`.

### 11.2. Plots

Mandatoriness rule: a plot is mandatory **if the corresponding measured series
was measured**; if the series was not measured, the plot is not built, and the
reason is recorded in the report. The absence of a mandatory plot when the series
exists requires an explanation. Every plot — RU and EN.

Plots fall into two groups.

**(a) Automatically generated figures.** The plotter
`scripts/report/plot_context_curves.py` builds five figures (files
`<name>.<lang>.svg`): the base three under their existing conditions, and two
consumption figures (`power-vs-load`, `energy-per-token`) when an energy
series exists; otherwise a warning is printed and the figure is not built.

| Figure (file name) | Condition of being mandatory |
| --- | --- |
| `prefill-vs-context` | if the ladder was measured (the core of the methodology) |
| `decode-vs-context` | if the ladder was measured (the core of the methodology) |
| `ab-cache-hit` | with an A/B series |
| `power-vs-load` (average power versus fill) | with an energy series |
| `energy-per-token` (J/output-token versus fill) | with an energy series |

**(b) Additional per-axis plots.** The author builds them **separately** (outside
`plot_context_curves.py`) when the corresponding series exists:

| Plot | Condition of being mandatory |
| --- | --- |
| layout vs prefill | with a layout series |
| ubatch (prefill/decode versus `--ubatch-size`) | with a ubatch series |
| speculation depth (decode/acceptance versus depth) | with a speculation series |

- If the series was not measured, the corresponding plot is not built, and the
  reason is recorded in the report.
- Format — **SVG** (no external dependencies), separate files for RU and EN.
- **Machine-readable summary** `results.json` and tables `results-tables.md` are
  generated from the raw runs; aggregators are read-only with respect to the runs.
- **Labeling rules:** separate measured from computed (percentages, Δ, headroom —
  "computed"); state the configuration, `<CTX>`, KV quantization, and runtime; do
  not smooth curves without a note.
- **Portable aggregators.** `scripts/report/generate_report.py` (the
  `results.json` summary + `results-tables.md` tables) and
  `scripts/report/plot_context_curves.py` (plots) are copied into
  `<case>/scripts/`. Series, labels and roles (`main`/`control`) come from the
  `study.json` config (or are auto-detected from the raw runs when the config is
  absent); there are no hard-coded profiles or labels.
- **The `study.json` config.** It sets the study title, the profile list
  (`name`/`label`/`role`), `ab_planned` (profiles with a mandatory A/B series),
  `ab_orders` (expected orders) and `ladder_pcts` (ladder rungs as % of `<CTX>`).
  Each list is optional: when the config is absent the aggregators auto-detect
  profiles, orders and rungs, and every profile becomes `main`.

## 12. Run and research artifacts

**Each run** writes a directory `results/<run_id>/`:

| File | Content |
| --- | --- |
| `result.json` | `run_id`, status (`ok`/`failed`/`timeout`/`non_comparable`), error, rungs (prefill/decode, cache-hit, `completion_is_fixed`, `finish_reason`), telemetry maxima, `clocks_*` |
| `profile.json` | copy of the used profile |
| `command.json` | actual command, environment, port, seed, mode/order |
| `server.log` | server stdout+stderr (buffers, warnings, eviction) |
| `telemetry.csv` | samples every 0.5 s (section 10) |

**The research** additionally contains:

- aggregated summaries (`results.json`, `results-tables.md`) and plots;
- a chronology of phases (what was measured and when, including negative results);
- provenance of the model and runtimes (sha256, commit, build flags);
- runs with the `failed`/`timeout`/`non_comparable` statuses and the cause text.

## 13. Portability

| Category | What it covers | Comparability |
| --- | --- | --- |
| Must be recorded | model/`<GPU>` (model, VRAM, count, topology, links), CPU/RAM, driver, CUDA, `<RUNTIME>` commit, `<MODEL>` sha256 | without it comparison is impossible |
| Hardware-dependent (local sweep) | `--ubatch-size`, `--tensor-split`, speculation depth, device order | the optimum cannot be transferred, only re-swept |
| Comparable across stands | only identically defined normalized indicators with matching semantics (for example, relative Δ between configurations within one stand, an identically defined `cache_hit_fraction`) | relatively |
| Not comparable across stands | absolute tok/s, the shape of the performance curve (depends on runtime/driver/clocks/tokenizer), optimal layout, OOM thresholds | locally only |

**Conclusion.** Compare predominantly **within one stand**; across stands only
identically defined normalized indicators with matching semantics are comparable.
Absolute numbers and the curve shape are only valid within one stand, one runtime,
and one clock configuration.

## 14. Agent workflow

- **Roles:** orchestrator (`keeper`) — plans and accepts; `sys-probe` —
  hardware/environment; `bench-rig` — runs; `runtime-build` — engine build;
  `model-fetch` — weight download/verification; `code-smith` — code;
  `net-researcher` — external sources; `reviewer` — verification.
- **Separation:** execution, verification, and interpretation are different
  roles. An executor does not verify itself; a reviewer does not edit code.
- **Task contract:** goal → confirmed context → absolute path → scope and
  prohibition to exceed it → task and constraints → acceptance criteria →
  response format.
- **Parallelism ≤3**, only without overlapping files/contracts; background — with
  a task registry and a deadline.
- **Failed runs are preserved.** Fact vs interpretation: the runner returns
  verbatim (PASS/FAIL/TIMEOUT, exit code, stdout); the caller interprets.
- **Git mutations** (commit/push/checkout/clean) — only on an explicit user
  request.

## 15. Research acceptance criteria (checklist)

- [ ] Configuration completeness: all declared axes and levels are covered.
- [ ] All runs are preserved, including failed ones, with the
      `ok`/`failed`/`timeout`/`non_comparable` statuses.
- [ ] Repeats **≥ 3** successful (`ok`) per **step** for a canonical result; for
      every series `n` and `run_id`s are stated; aggregation is median + min/max.
      At n < 3 the step is marked limited/non-canonical, and excluded steps are
      listed with the reason.
- [ ] Telemetry exists for every run; peak VRAM and headroom are recorded at all
      rungs.
- [ ] Telemetry conforms to the normative format (section 10), including
      `schema_version = 1`; mandatory fields are not skipped silently.
- [ ] Plots are built in RU and EN; the mandatory set is complete (or its absence
      is explained).
- [ ] Provenance of the model and runtimes is recorded (sha256, commit, flags).
- [ ] Facts are separated from interpretations by measured/computed/interpretation
      labels.
- [ ] Comparisons are aligned (one axis at a time; equal prompt length; full
      offload).
- [ ] The publication barrier is passed (no private paths, secrets, internal IPs).
- [ ] The machine-readable summary and tables are generated from the raw runs.

## 16. Typical mistakes

Collected from completed cases; each with a symptom and a remedy. The numbers
below are **case-specific examples** from concrete artifacts (see
`inference_cases/`), not portable facts.

1. **Decode on a short generation.** The metric is invalid (too few tokens).
   Remedy: a long generation and `ignore_eos` + `n_predict`.
2. **Prompt-cache contamination.** Long cases within one process share a prefix,
   and prefill is understated. Remedy: separate runs, `--repetitions 1`.
3. **Idle frequency parking.** All measurements are invalid. Case-specific
   example: 300 MHz (see the artifact). Remedy: disable parking, capture
   `clocks.sm` before/after.
4. **Client timeout.** Shorter than the prefill time — an abort before
   completion. Remedy: a per-case timeout, not a hard-coded value.
5. **Fixed speculation/MTP overhead.** Case-specific example: ~2.8 s per request
   (see the artifact), which hurts short additions to a long context. Remedy:
   separate the fixed overhead from the marginal cost.
6. **Host prompt-cache eviction.** KV larger than `--cache-ram` — A/B cache-hit is
   distorted. Remedy: A/B only when both sessions fit.
7. **OOM at a large µbatch.** Case-specific example: `--ubatch-size 1024` at a
   large `<CTX>` gives OOM on the narrow card (see the artifact). Remedy: a local
   µbatch sweep and VRAM headroom control.
8. **Incomplete offload.** Some layers on the CPU (`offloaded X/Y` with X < Y; or
   a model/KV buffer on the CPU, other than `CPU_Mapped`/`CUDA_Host`) — speeds are
   not comparable. `CPU_Mapped` is mmap of the weights, not a sign of CPU
   inference. Remedy: check buffers and headroom before measurements.
9. **Confounded sweeps.** Several parameters change at once (example: µbatch
   together with KV quantization and environment variables). Remedy: one axis at a
   time; otherwise mark the conclusion as non-comparable.

---

The concrete description of your stand is in `hardware_info/HARDWARE.md`;
packaging of a published case is in `TEMPLATE.md`.

---

**Revision log.** 2026-09-23 — following independent review: run-status policy
(`ok | failed | timeout | non_comparable`); repeats and aggregation (median +
min/max, `n` and `run_id`); canonical run selection; conditionally mandatory
plots; strengthened full-KV-in-VRAM verification; normative telemetry format
(schema version 1) and the non-conformance of the legacy `host_telemetry.py`;
placeholders instead of local addresses; separation of mandatory preflight checks
and stand adapters; clarifications on warm-up/rung 0, metric sources, A/B orders,
TIMEOUT, cross-stand comparability, and case-specific examples.

2026-09-23 — following the second review: a single repeat norm (**n ≥ 3**
successful for a canonical result; n < 3 is limited/non-canonical); a
machine-readable telemetry schema version (`schema_version = 1`, mandatory
consumer check; the exact column list; `clocks_sm_*` are `result.json` metadata);
the `non_comparable` status for unavailable mandatory telemetry fields; a unified
formulation of plot mandatory-ness (when the measured series exists); labeling of
`cache_hit_fraction` growth as a verifiable hypothesis.

2026-09-24 — moving the scripts into the methodology and generalizing: all scripts
were moved to the portable `scripts/runner/`, `scripts/telemetry/`,
`scripts/report/`, `scripts/publish/` and are copied into `<case>/scripts/`; the
canonical telemetry collector is `scripts/telemetry/host_telemetry.py` (schema 1);
`scripts/report/generate_report.py` and `scripts/report/plot_context_curves.py`
take series/labels/roles from `study.json` (or auto-detect them), with no
hard-coded profiles; the last-run selection limitation was removed (aggregation
over repeats); the `hardware_info/` index entry was updated.

2026-09-24 — following the second document review: in §11.2 the plots are
explicitly split into (a) the three automatically generated figures of
`plot_context_curves.py` (`prefill-vs-context`, `decode-vs-context`,
`ab-cache-hit`) and (b) additional per-axis plots (layout/ubatch/speculation
depth) that the author builds separately; the mandatoriness rule when a measured
series exists is preserved.

2026-09-24 — synchronization with the fixed automation and the audit facts: in
§5.1 the partial-offload sign was corrected (`offloaded X/Y` with X < Y; or a
model/KV buffer on the CPU, other than `CPU_Mapped`/`CUDA_Host`; `CPU_Mapped` is
mmap of the weights, not a sign) and §5.3/§16.8 updated; in §7.1 the
responsibilities of the runner (`timeout`/`failed` by exception type, invariant
flagging, `step_comparable`) and the aggregator (per-step canon) were separated,
and the `mixed` status added; in §8 the audit fact from
`docs/timing-source-audit.md` was recorded (API and `log_timings` identical;
`timing_delta_pct` thresholds); in §11.1 the canon was moved to per-step
(comparable steps, limited steps outside the main tables/plots, `n` = unique
runs); in §3 the `cache_prompt` axis was added and a cold mode in §4.4; in §4.6
counterbalancing for comparing configurations was added; in §4.5 the primacy of
the profile rung grid `ladder_pcts` was noted.

2026-09-24 — norm for P2-b (paired A/B delta) and P2-d (energy/power + two
consumption figures): in §8 energy metrics were added (`power_avg_w`,
`energy_j`, `energy_dynamic_j`, J/output-token, J/input-token) with the
`gpuN_power_w` source and the "computed/estimate" label; in §10 it was recorded
that `gpuN_power_w` is the energy source, the request window is the step's
`started_at_utc`/`finished_at_utc` pair (UTC, synchronized with `timestamp_utc`),
and the energy is computed by the aggregator, not the collector; in §11.1 it was
stated that the energy metrics are aggregated like the base ones (median +
min/max) and enter `results.json`, while the paired A/B delta is computed from
matching steps within a single `ab-sequence` run (`Δ = 100·(B−A)/A`, median Δ,
`n_pairs`, `sign_consistency`, key `"profile|order"`), not from separate A and B
medians; in §11.2 the `power-vs-load` and `energy-per-token`
auto-figures were added to group (a) (when an energy series exists).

2026-09-24 — synchronizing the step-comparability wording with the aggregator
code: in §11.1 the comparable condition was brought to the actual one
(`completion_is_fixed == true` AND `finish_reason = "length"` AND
`step_comparable != false`); it was clarified that `prompt_prefix_ok` is part of
`step_comparable`, which the runner forms (for new runs), while the aggregator
(`step_is_comparable`) does not check it directly and, for legacy runs without
the `step_comparable` field, relies on the first two conditions. §7.1 is
consistent: there `prompt_prefix_ok` is already described as part of the
runner's `step_comparable` definition, not as a direct aggregator condition.
