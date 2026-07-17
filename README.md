# DI-CC: Decomposition-Induced Context-Memory Conflict

Code and data accompanying a paper on **Decomposition-Induced Context-Memory
Conflict (DI-CC)** — a failure mode in decompose-then-verify fact-checking
pipelines (like FActScore) where the decomposer LLM silently substitutes its
own parametric belief for what a source passage actually says, producing an
atomic claim that contradicts the source it was supposed to summarize
faithfully.

This repository lets you reproduce the paper's two headline results:

- **H0 (mechanistic):** a linear probe trained *only* on classical
  context-memory conflict data (NQ-Swap), and never exposed to decomposition
  output, still detects DI-CC in decomposition activations.
- **H3 (mitigation):** context-aware decoding (CAD) reduces the DI-CC rate,
  at a completeness cost on coreference-heavy inputs.

This is a **research artifact**, not a general-purpose library: it exists to
let you re-run and inspect the exact pipeline behind the paper's numbers, not
to be installed as a dependency in other projects.

## Headline Results

| Result | Finding |
|---|---|
| H0 — mechanistic transfer | A probe trained only on NQ-Swap (classical context-memory conflict) detects DI-CC in decomposition activations with AUC ≈ 0.86–0.88 (n=64), despite never seeing decomposition data during training. |
| H3 — CAD mitigation | Context-aware decoding reduces the DI-CC rate from 4.16% to 2.57% (McNemar's test, p = 0.000041), but costs a 19.4% structural parsing failure rate on coreference-heavy (vague/elliptical) inputs. |

Full statistics are written to `outputs/reports/transfer_eval.json` and
`outputs/reports/h3_mitigation.json` when you run stages 7 and 27
respectively.

## What's Included

This release covers the core **7B pipeline**: NQ-Swap probe training, Phase 0
mechanistic validation (stages 1–7), and the H3 CAD mitigation study (stages
24–27, 30–31). The decomposer was also tested at 3B and 14B scale in the
paper, but that ablation is not part of this staged release.

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
  factscore_subclaims_40person.jsonl   # frozen 40-person bio+decomposition data (H0)
  h3_bios_191person.jsonl              # frozen 191-person bio pool (H3)
outputs/probes/
  probe_layer22.joblib     # frozen probe checkpoint, accuracy-selected layer
  probe_layer20.joblib     # frozen probe checkpoint, variance-ratio-selected layer
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
