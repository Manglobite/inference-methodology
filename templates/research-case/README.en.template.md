# <MODEL> on <HARDWARE>: <short research summary>

[Русский](README.md) | **English**

> Template. Replace every `<...>` placeholder and fill the sections with facts.
> "FILL IN" notes say exactly what is required. Do not leave placeholders in the
> published version.

## Goal

FILL IN: which hypothesis the study tests and what the reader must be able to
reproduce. For example:

1. how **<MODEL>** behaves at context **<CTX>** on the **<HARDWARE>** rig;
2. how prefill/decode change as the context fills;
3. what <variable factor> provides.

## Model

| Field | Value |
| --- | --- |
| Model | `<MODEL>` |
| GGUF source | `<HF-repo>` (HF), commit `<commit>` |
| File(s) | `<file>.gguf` — `<bytes>` B, sha256 `<sha256>` |
| Architecture | FILL IN (layers, attention types, GQA, head_dim) |
| Native context | `<CTX>` |
| License | FILL IN |

Weights are **not** included in the repository: only URL, size and sha256 are
reproduced (see [RUNBOOK.md](RUNBOOK.md), [LICENSE-NOTICE.md](LICENSE-NOTICE.md)).

## Rig and software

- GPU: `<HARDWARE>`;
- CPU/RAM/OS/driver/toolchain: see [HARDWARE.md](HARDWARE.md);
- runtime: `<RUNTIME>`, engine/commit/flags — in [RUNBOOK.md](RUNBOOK.md).

## Methodology

FILL IN: run modes (smoke/ladder/A-B), fixed generation (seed,
`temperature: 0`, `ignore_eos`, `n_predict`), device binding, prompt-cache mode
(`cache_prompt`; when `false` — the cold mode with automatic checking of the
`cold_cache_hit_tolerance` threshold and the `step_comparable_reason`),
telemetry collection. The general methodology is in
[METHODOLOGY.md](../../METHODOLOGY.md).

> The link `../../METHODOLOGY.md` is correct **for the skeleton**
> (`methodology/templates/research-case/`). **After copying the case into
> `inference_cases/<case>/`**, the path changes to
> `../methodology/METHODOLOGY.md`. Re-check the link after copying; under any
> other layout, replace it with a stable URL/commit of the methodology (a tag) so
> the link does not look broken and the reader does not depend on the original
> tree layout.

## Variants

| # | Profile | Weights | KV | Device | Role |
| ---: | --- | --- | --- | --- | --- |
| 1 | `<profile>` | FILL IN | FILL IN | FILL IN | FILL IN |

## Results

FILL IN: summary table over the canonical runs. Sources —
`docs/results-tables.md` (human-readable tables) and `docs/results.json`
(machine-readable summary with canonical/control runs); raw runs —
`results/<run_id>/`. The canonical result is the aggregate over repetitions
**per step** (per-step canonicality): a step enters only when it is comparable
(`completion_is_fixed` + `finish_reason = length` + `step_comparable`) and has
**n ≥ 3** unique `ok` runs (**median + min/max**, with `n` and the list of
`run_id`s recorded); steps with `n < 3` are marked `limited` and shown
separately, outside the main tables/figures. The generation status of a run is
`fixed`/`mixed`/`variable`. The norm is in
[METHODOLOGY.md](../../METHODOLOGY.md) §11.1 (for the skeleton; after copying the
case the path is `../methodology/METHODOLOGY.md`). Mark derived values
(percentages, deltas) as **computed**, and keep interpretation separate from
facts.

When an A/B series exists, report the paired A/B delta (delta = 100·(B−A)/A over
the matching steps, median delta, `n_pairs`, `sign_consistency`) from the report
and `docs/results.json` (`ab_delta`, key `"profile|order"`). When power telemetry
exists, give the step energy estimate (J/request = `energy_j`, J/output-token
from `telemetry.csv`; the report's "Energy and power" section), marked as an
**estimate**; `energy_dynamic_j` subtracts an idle-window baseline (median total
power over `IDLE_WINDOW_S` before the first request, falling back to the run
minimum; source in `idle_baseline_source`); without power telemetry this is not a
blocker — record an explicit "no data".

## Figures

FILL IN: links to `docs/figures/*.svg`. A figure is mandatory **when the
corresponding measured series exists**; if the series was not measured, the
figure is not built and the reason is recorded here (see
[METHODOLOGY.md](../../METHODOLOGY.md) §11.2 — for the skeleton; after copying the
case the path is `../methodology/METHODOLOGY.md`).

## Negative results

FILL IN: what was tested and did not hold (kept on purpose).

## Limitations

FILL IN: what was not measured, and under what conditions numbers are not
comparable.

## Reproduction

Briefly (full order in [RUNBOOK.md](RUNBOOK.md)); the scripts are first copied
from `methodology/scripts/<category>/` into `scripts/` (see
[scripts/README.md](scripts/README.md)):

```bash
# 1) preflight: free GPUs, disk, disable the node idle policy
# 2) smoke-check the profile
# 3) runs (ladder / A-B)
# 4) aggregation: tables and figures
```

## Repository layout

```text
<case>/
├── README.md / README.en.md
├── HARDWARE.md
├── RUNBOOK.md
├── CHRONOLOGY.md
├── LICENSE-NOTICE.md
├── study.json (optional)
├── profiles/
├── prompts/
├── scripts/
├── data/
├── docs/
└── results/
```

## Publication

FILL IN after passing the barrier: `scripts/check-public.sh` (expects
`OK: no private markers found`), sanitize the publication copy with
`scripts/sanitize-results.sh --in-place` (both scripts are copied from
`methodology/scripts/publish/`). The `--in-place` sanitizer is destructive: apply
it to the publication copy of the tree or make an archive/backup of the original
`results/` first. The barrier covers `/home/<user>/`, `/root/`, `/Users/<user>/`,
`/mnt/`, RFC1918, `sk-[A-Za-z0-9_-]{13,}`, `Authorization: Bearer` and `api_key`/`secret`/`token`.
Host names are not caught by the baseline — add them via `PUBLIC_DENY_FILE`.
Published texts contain only relative paths; absolute home paths, host names and
internal IPs are not published. Model weights are not committed.
