# DI-CC: Decomposition-Induced Context-Memory Conflict

Code and data accompanying a paper on **Decomposition-Induced Context-Memory
Conflict (DI-CC)** — a failure mode in decompose-then-verify fact-checking
pipelines (like FActScore) where the decomposer LLM silently substitutes its
own parametric belief for what a source passage actually says, producing an
atomic claim that contradicts the source it was supposed to summarize
faithfully.

This repository lets you reproduce the paper's headline results and the
robustness checks added during revision:

- **H0 (mechanistic, primary result):** a linear probe trained *only* on
  classical context-memory conflict data (NQ-Swap), and never exposed to
  decomposition output, still detects DI-CC in decomposition activations at
  n=64, under two independent layer-selection criteria.
- **H3 (mitigation):** context-aware decoding (CAD) reduces the DI-CC rate,
  at a severe completeness cost (including outright identity fabrication) on
  coreference-heavy inputs.
- **Robustness checks:** a class-imbalance precision/recall characterization
  of the H0 probe (multi-seed), a CAD contrast-strength (α) sensitivity
  sweep, and a SelfCheckGPT-style self-consistency baseline comparison.

This is a **research artifact**, not a general-purpose library: it exists to
let you re-run and inspect the exact pipeline behind the paper's numbers, not
to be installed as a dependency in other projects.

## Headline Results

| Result | Finding |
|---|---|
| H0 — mechanistic transfer (**primary**, n=64) | A probe trained only on NQ-Swap detects DI-CC in decomposition activations: AUC 0.881 [0.841, 0.916] vs. legit, AUC 0.863 [0.806, 0.913] vs. DI-UE (variance-ratio layer selection). Holds under accuracy-based selection too (0.858 / 0.893). |
| H0 — pilot (n=14, original 40-person pool) | Same test at the original, smaller sample: AUC 0.893–0.917 vs. legit, 0.806–0.934 vs. DI-UE, depending on layer-selection criterion. Retained for comparison, not the primary result. |
| H3 — CAD mitigation | Context-aware decoding reduces the DI-CC rate from 4.16% to 2.57% (McNemar's test, p = 0.000041), but 19.4% of coreference-heavy decompositions fail to parse under CAD — 84% of those failures involve the decomposer fabricating a different person's identity, not just omitting detail. Not a deployment-ready mitigation as implemented. |
| Class-imbalance robustness | At a realistic training-class ratio (~6.8% positive), the H0 probe's fixed-threshold conflict-class recall collapses from 0.808±0.039 (balanced) to 0.247±0.190 (imbalanced), averaged over 50 random seeds — AUC alone does not reveal this. |
| SelfCheckGPT baseline | On the same n=64 dataset, SelfCheckGPT-style self-consistency sampling fails to detect DI-CC at all (AUC 0.508 vs. legit, 0.142 — wrong-signed — vs. DI-UE), because DI-CC's defining stability makes it recur consistently across resamples, the opposite of what self-consistency methods flag. |
| Cross-model validation — Mistral-7B-Instruct-v0.3 | H0 does **not** replicate: AUC 0.397 [0.344, 0.449] vs. legit, 0.308 [0.238, 0.385] vs. DI-UE — both below chance. Checked and ruled out as explanations: statistical power (CI is narrow, not wide), layer-selection criterion (variance-ratio selects the identical layer), and labeling quality (manual review of all 74 DI-CC cases finds 71/74 correct; excluding the 3 mislabels makes the result more negative, not less). |
| Cross-model validation — Falcon-7B-Instruct | Inconclusive, not a third confirming/disconfirming data point: the same elicitation protocol yields only 10 DI-CC instances (vs. 74 for Mistral), so the resulting CI is uninformative (0.367–0.669, straddling chance). Falcon's underlying NQ-Swap probe is also markedly weaker (test_acc 0.605 vs. Mistral's 0.756) due to Falcon's lower closed-book knowledge yield (19.5% vs. Mistral's 70.8%). |

Full statistics are written under `outputs/reports/` by the corresponding
stage script (see table in [Running the Pipeline](#running-the-pipeline)
below); the reports behind the numbers above are also frozen in this repo
under `outputs/reports/` so you can inspect them without rerunning anything.

## What's Included

This release covers the core **7B pipeline**: NQ-Swap probe training, Phase 0
mechanistic validation (stages 1–7), the n=64 primary-result pipeline
(stages 28, 29, 32, 33), the H3 CAD mitigation study (stages 24–27, 30–31),
and four revision-round robustness checks (stages 34–41: class-imbalance
precision/recall, CAD α sensitivity, SelfCheckGPT baseline). The decomposer
was also tested at 3B and 14B scale in the paper (Phase 2 scale ablation),
but that ablation's code is not part of this staged release.

It also covers the cross-model-family validation (stages 1–7 rerun end-to-end
against Mistral-7B-Instruct-v0.3 and Falcon-7B-Instruct via `PHASE0_SCALE`,
stage 42's bootstrap-CI diagnostic, stage 43's variance-ratio layer-selection
re-check, and stage 44's manual-mislabel-exclusion sensitivity recompute).

```
config.py                 # central path/model config (env-var driven)
phase0/                   # library code: data loading, decomposition, probing
  data/                   #   bio_entities.py (191-person pool), nq_swap.py
  decomposition/          #   decompose.py (incl. CAD), entailment.py, labeling.py
  activations/            #   hidden-state extraction from the decomposer model
  probe/                  #   layer-sweep + logistic-regression probe training
  transfer_eval/          #   AUC + permutation test / bootstrap CI evaluation
  utils/                  #   answer matching, keep-awake utility
scripts/                  # numbered pipeline stages, see below
data/
  factscore_subclaims_40person.jsonl   # frozen 40-person bio+decomposition data (H0 pilot)
  h3_bios_191person.jsonl              # frozen 191-person bio pool (H3, also source of H0 primary)
  h3_selfcheckgpt_resamples.jsonl      # frozen N=10 temperature-sampled resamples (SelfCheckGPT baseline)
outputs/probes/
  probe_layer22.joblib             # frozen probe checkpoint, accuracy-selected layer (7B)
  probe_layer20.joblib             # frozen probe checkpoint, variance-ratio-selected layer (7B)
  probe_layer18_mistral7b.joblib   # frozen probe checkpoint (Mistral-7B cross-model validation)
  probe_layer28_falcon7b.joblib    # frozen probe checkpoint (Falcon-7B cross-model validation)
outputs/reports/           # frozen small (<5KB) JSON reports behind every number above
```

## Prerequisites

- Python 3.10+ (the codebase uses `int | None`-style union type hints)
- A CUDA GPU — the decomposer model is loaded in 4-bit (NF4) via
  `bitsandbytes`, and the NLI labeling model also runs on GPU by default.
  Stages that only touch already-extracted activations (`03`, `07`, `30`,
  `31`) do not need a GPU.

## Quick Start

```bash
git clone https://github.com/hollis-png/di-cc-decomposition-conflict.git
cd di-cc-decomposition-conflict
./setup.sh
source .venv/bin/activate
```

`setup.sh` creates a virtualenv and installs `requirements.txt`. There is no
`.env` file to configure for the core pipeline — the only environment
variables the code reads (`PHASE0_SCALE`, `PHASE0_DEVICE`) have safe
defaults (see [Configuration](#configuration) below).

## Running the Pipeline

All scripts are run from the repository root, in order, e.g.
`python scripts/01_build_nq_swap.py`. Output paths (under `data/processed/`
and `outputs/`) are created automatically and are all defined centrally in
`config.py`.

### H0 — mechanistic transfer (the headline result)

```bash
python scripts/01_build_nq_swap.py              # subsample NQ-Swap, no GPU
python scripts/02_extract_nqswap_activations.py  # needs GPU
python scripts/03_train_probe.py                 # layer sweep + train probe, no GPU
python scripts/04_run_decomposition.py           # generate + decompose 40 bios, needs GPU
python scripts/05_label_spans.py                 # NLI-label every subclaim, needs GPU
python scripts/06_extract_decomp_activations.py  # extract decomposition activations, needs GPU
python scripts/07_evaluate_transfer.py           # apply the frozen probe -> AUC, no GPU
```

`04_run_decomposition.py` regenerates biographies from the decomposer model,
which is stochastic even in closed-book, greedy-decode mode across
hardware/library versions. To reproduce the paper's numbers **exactly**,
skip stage 04 and instead copy the frozen data file into place before
running stage 05:

```bash
mkdir -p data/processed
cp data/factscore_subclaims_40person.jsonl data/processed/factscore_subclaims.jsonl
python scripts/05_label_spans.py
python scripts/06_extract_decomp_activations.py
python scripts/07_evaluate_transfer.py
```

Both `01`–`07` and `24`–`25` scripts support `--limit N` for a quick smoke
test before committing to a full (GPU-hours-long) run.

### H0 — primary result (n=64, reuses H3's baseline pool)

The paper's actual headline AUCs (0.881 / 0.863) come from applying the H0
methodology to the larger 191-entity H3 pool's non-CAD (`*_baseline`)
decompositions, not the original 40-person pilot above. This needs the H3
bio pool and baseline decompositions/labels (see the H3 section below) to
already exist, plus a frozen NQ-Swap probe-training run (stages 1–2):

```bash
python scripts/28_extract_h3_baseline_activations.py   # needs GPU
python scripts/29_evaluate_h3_baseline_transfer.py      # accuracy-layer AUC, no GPU
python scripts/32_evaluate_h3_baseline_transfer_variance_ratio.py  # variance-ratio-layer AUC (primary), no GPU
python scripts/33_bootstrap_ci_all_h0_results.py        # bootstrap 95% CIs for all 4 (sample x criterion) configs, no GPU
```

### H3 — CAD mitigation

```bash
python scripts/24_generate_h3_bios.py    # generate the 191-person bio pool, needs GPU, resumable
python scripts/25_decompose_h3.py        # decode under baseline + CAD, needs GPU, resumable
python scripts/26_label_h3.py            # NLI-label every subclaim, needs GPU
python scripts/27_evaluate_h3.py         # McNemar's test -> does CAD reduce DI-CC? no GPU
```

To reproduce exactly without regenerating the bio pool, skip stage 24 and
copy the frozen pool into place first:

```bash
mkdir -p data/processed
cp data/h3_bios_191person.jsonl data/processed/h3_bios.jsonl
python scripts/25_decompose_h3.py
```

Stage 25 still needs a GPU (it decodes both baseline and CAD conditions for
all 191 people) and is resumable — it appends completed records and skips
ones already written, so it's safe to interrupt and restart. It can take a
long time for the full pool across all conditions.

### Alternative layer-selection criterion (robustness check)

Stages 30–31 re-run the H0 transfer evaluation using a variance-ratio layer
selection criterion (Burger et al. 2024) instead of held-out accuracy
(stage 3's method), reusing activations already extracted by stages 2 and 6
— no GPU and no re-decomposition needed:

```bash
python scripts/30_train_probe_variance_ratio.py
python scripts/31_evaluate_transfer_variance_ratio.py
```

### Class-imbalance robustness check

The H0 probe above is trained on perfectly balanced NQ-Swap data (413
conflict / 413 no-conflict). Real deployment prevalence is far lower
(DI-CC is ~3.2% of atomic claims — see the paper's Section 5). These
scripts retrain the probe at a downsampled, closer-to-deployment class
ratio and report fixed-threshold precision/recall (not just AUC) — the
headline finding is that AUC looks fine under imbalance, but fixed-threshold
conflict-class recall collapses. Needs the frozen `outputs/activations/`
files from stages 2 and 28; no GPU needed for these scripts themselves:

```bash
python scripts/34_probe_imbalanced_robustness.py         # first pass: AUC only (superseded below)
python scripts/38_imbalanced_probe_precision_recall.py   # adds precision/recall/confusion matrix, single seed
python scripts/41_imbalanced_probe_multiseed.py           # repeats across 50 seeds -> distribution, not a point estimate
```

Run `41` if you only run one of these three — it supersedes `34` and `38`
with a proper multi-seed characterization (mean/std/range) rather than a
single, potentially noisy split.

### CAD contrast-strength (α) sensitivity sweep

The CAD mitigation above uses the literature-recommended α=1.0. These
scripts sweep α ∈ {0.5, 1.5} on the full 191-entity pool to check whether a
different value trades off the primary suppression effect against the
side-effect severity better than the default:

```bash
python scripts/35_h3_alpha_sensitivity.py         # decode at alpha in {0.5, 1.5}, needs GPU, resumable, long-running
python scripts/36_label_h3_alpha_sensitivity.py   # NLI-label every subclaim, needs GPU
python scripts/37_evaluate_h3_alpha_sensitivity.py # McNemar's + side-effect stats per alpha, no GPU
```

Only the `_cad` conditions are re-decoded (`_baseline` doesn't depend on
α and is reused from stage 25's output), but this is still 191 people × 2
conditions × 2 α values ≈ 750 decodes — expect several hours on a single
GPU.

### SelfCheckGPT baseline comparison

Compares the mechanistic probe against SelfCheckGPT-style self-consistency
sampling (Manakul et al., 2023) on the same n=64 primary dataset, using the
same permutation-test/bootstrap-CI methodology, so the two detectors are
directly comparable:

```bash
python scripts/39_selfcheckgpt_resample.py   # N=10 temperature resamples per (person, condition), needs GPU, resumable, long-running
python scripts/40_selfcheckgpt_evaluate.py   # NLI-based inconsistency scoring + AUC, needs GPU (NLI model only)
```

Stage 39 resamples all 329 (person, condition) pairs behind the n=64
dataset × 10 samples each ≈ 3,290 generations — expect several hours. To
skip it, copy the frozen resamples into place first:

```bash
mkdir -p data/processed
cp data/h3_selfcheckgpt_resamples.jsonl data/processed/h3_selfcheckgpt_resamples.jsonl
python scripts/40_selfcheckgpt_evaluate.py
```

## Why the Data Files Are Frozen

`data/factscore_subclaims_40person.jsonl` (75 bio records / 443 subclaims)
and `data/h3_bios_191person.jsonl` (191 bio records) are pinned outputs of
stages 4 and 24 respectively. Decomposer generation is closed-book and
greedy, but still not guaranteed bit-identical across GPU/driver/library
versions — freezing these files lets anyone reproduce the exact subclaim
counts and downstream statistics reported in the paper without re-running
the (multi-hour, GPU-bound) generation stages.

The two frozen probe checkpoints in `outputs/probes/` (`probe_layer22.joblib`,
accuracy-selected; `probe_layer20.joblib`, variance-ratio-selected) are the
trained artifacts of stage 3 / stage 30 and let you skip straight to stage 7
/ 31 if you also use the frozen NQ-Swap activations you extract via stage 2.

`data/h3_selfcheckgpt_resamples.jsonl` (3,290 temperature-sampled
decompositions) is the pinned output of stage 39, for the same reason: it's
several hours of GPU-bound, stochastic generation, and freezing it lets
anyone reproduce the SelfCheckGPT baseline's exact numbers by running only
stage 40.

The small JSON reports under `outputs/reports/` (all under 5KB) are also
committed directly, unlike the large `outputs/activations/*.npz` files
(hundreds of MB, regenerable via stages 2/6/28 but deliberately not
version-controlled) — this lets you inspect every number cited above without
running anything at all.

## Configuration

The pipeline is configured via `config.py`, which reads two environment
variables:

| Variable | Required | Default | Description |
|---|---|---|---|
| `PHASE0_SCALE` | No | `7b` | Decomposer scale: `3b`, `7b`, or `14b`. Only `7b` is validated by the data/checkpoints in this release. |
| `PHASE0_DEVICE` | No | `cuda:0` | Torch device string for both the decomposer and NLI models. |

`FACTSCORE_UNLABELED_DIR` and `FACTSCORE_LABELED_DIR` are also read from the
environment, but only used by H1/H2 scripts that are **not** part of this
staged release (they require a separately-downloaded copy of the
[FActScore](https://github.com/shmsw25/FActScore) dataset). They are not
needed to reproduce the H0 or H3 results above.

## Models Used

- **Decomposer:** `Qwen/Qwen2.5-7B-Instruct`, loaded 4-bit (NF4) via
  `bitsandbytes`.
- **NLI / entailment (span labeling):**
  `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`.
- **NQ-Swap dataset:** `pminervini/NQ-Swap` on the Hugging Face Hub.

All three are downloaded automatically on first use via the Hugging Face
Hub / `datasets` library.

## Citation

```bibtex
@misc{dicc2026,
  title  = {Decomposition-Induced Context-Memory Conflict},
  author = {Yen Yu Feng},
  year   = {2026},
  note   = {TODO: add venue / arXiv link once available}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
