"""Shared configuration for the Phase 0 mechanistic pre-validation pipeline
behind Decomposition-Induced Context-Memory Conflict (DI-CC). See README.md
for the pipeline overview and how the numbered scripts/ stages fit together.
"""
import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent

# --- Model / scale (Phase 2 scale ablation) ---
# Set PHASE0_SCALE=3b or PHASE0_SCALE=14b before running any stage script to
# rerun the Phase 0 pipeline (stages 1-7) against a different decomposer
# scale. Unset (default "7b") reproduces the original, already-validated
# Phase 0 run unchanged -- existing 7b output paths are untouched so prior
# results stay reproducible.
PHASE0_SCALE = os.environ.get("PHASE0_SCALE", "7b")
_SCALE_MODELS = {
    "3b": ("Qwen/Qwen2.5-3B-Instruct", 36),
    "7b": ("Qwen/Qwen2.5-7B-Instruct", 28),
    "14b": ("Qwen/Qwen2.5-14B-Instruct", 48),
}
if PHASE0_SCALE not in _SCALE_MODELS:
    raise ValueError(f"PHASE0_SCALE must be one of {list(_SCALE_MODELS)}, got {PHASE0_SCALE!r}")
DECOMPOSER_MODEL_NAME, NUM_HIDDEN_LAYERS = _SCALE_MODELS[PHASE0_SCALE]
NUM_HIDDEN_STATES = NUM_HIDDEN_LAYERS + 1  # + embedding output, index 0

BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16
BNB_4BIT_QUANT_TYPE = "nf4"

DEVICE = os.environ.get("PHASE0_DEVICE", "cuda:0")

# --- Layer sweep candidates for the probe (Phase 0, step 2) ---
# Sweep every 2 layers across the back half of the network first; classic
# CM-conflict / fact-recall signals concentrate in mid-to-late layers
# (cf. Pham_2026_WhereKnowledgeCollides, Zhao_2024_SAE_Representation).
PROBE_LAYER_CANDIDATES = list(range(8, NUM_HIDDEN_STATES, 2))

# --- NLI model for entailment checks (R |= c_i) ---
NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

# --- Embedding model for closed-book recoverability / semantic checks ---
# Reuses BGE, consistent with CREDENCE_2026's Semantic-F1 choice.
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
RECOVERABILITY_SIM_THRESHOLD = 0.80  # cosine sim threshold, see decomposition/closed_book_probe.py

# --- Paths ---
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS_PROBES = PROJECT_ROOT / "outputs" / "probes"
OUTPUTS_ACTIVATIONS = PROJECT_ROOT / "outputs" / "activations"
OUTPUTS_REPORTS = PROJECT_ROOT / "outputs" / "reports"

for _dir in (DATA_RAW, DATA_PROCESSED, OUTPUTS_PROBES, OUTPUTS_ACTIVATIONS, OUTPUTS_REPORTS):
    _dir.mkdir(parents=True, exist_ok=True)


def _scale_path(path: Path) -> Path:
    """Suffix a Phase-0-core-pipeline path with the active scale, e.g.
    'transfer_eval.json' -> 'transfer_eval_14b.json'. No-op at the default
    "7b" scale so existing, already-validated outputs stay at their
    original paths and are never overwritten by a rerun.
    """
    if PHASE0_SCALE == "7b":
        return path
    return path.with_name(f"{path.stem}_{PHASE0_SCALE}{path.suffix}")


NQ_SWAP_JSONL = _scale_path(DATA_PROCESSED / "nq_swap.jsonl")
NQ_SWAP_ACTIVATIONS = _scale_path(OUTPUTS_ACTIVATIONS / "nq_swap_activations.npz")
PROBE_CHECKPOINT = _scale_path(OUTPUTS_PROBES / "probe_layer{layer}.joblib")
PROBE_SWEEP_REPORT = _scale_path(OUTPUTS_REPORTS / "layer_sweep.json")

FACTSCORE_SUBCLAIMS_JSONL = _scale_path(DATA_PROCESSED / "factscore_subclaims.jsonl")
LABELED_SPANS_JSONL = _scale_path(DATA_PROCESSED / "labeled_spans.jsonl")
DECOMP_ACTIVATIONS = _scale_path(OUTPUTS_ACTIVATIONS / "decomp_activations.npz")
TRANSFER_EVAL_REPORT = _scale_path(OUTPUTS_REPORTS / "transfer_eval.json")

# --- Layer-selection reanalysis (2026-07-16): variance-ratio criterion
# (Burger et al. 2024) as an alternative to accuracy-based layer selection,
# tested after Phase 2 found the accuracy-selected layer's transfer signal
# non-monotonic across scale. Reuses the SAME NQ-Swap and decomposition
# activations already on disk for each scale -- no GPU/decomposer needed. ---
PROBE_SWEEP_REPORT_VARRATIO = _scale_path(OUTPUTS_REPORTS / "layer_sweep_varratio.json")
TRANSFER_EVAL_REPORT_VARRATIO = _scale_path(OUTPUTS_REPORTS / "transfer_eval_varratio.json")

# --- Phase 1 / H1 (natural injection rate, uses the real FActScore release:
# https://github.com/shmsw25/FActScore -- download separately and point
# these env vars at your local copy; not needed for the core Phase 0
# pipeline, only for H1/H2) ---
FACTSCORE_UNLABELED_DIR = Path(os.environ.get("FACTSCORE_UNLABELED_DIR", "./data/factscore/unlabeled"))
H1_N_ENTITIES = 100  # scaled up from 30-entity pilot: natural injection rate
                      # was only 9/360 bios, too sparse for stable regression
SPACY_MODEL_NAME = "en_core_web_sm"

H1_BIOS_JSONL = DATA_PROCESSED / "h1_bios.jsonl"
H1_ELLIPTICITY_JSONL = DATA_PROCESSED / "h1_ellipticity.jsonl"
H1_SUBCLAIMS_JSONL = DATA_PROCESSED / "h1_subclaims.jsonl"
H1_LABELED_JSONL = DATA_PROCESSED / "h1_labeled_spans.jsonl"
H1_REGRESSION_REPORT = OUTPUTS_REPORTS / "h1_regression.json"

# --- Phase 1 / H2 simplified (calibrate automated NLI labeling against
# FActScore's human-atomic-facts annotations) ---
FACTSCORE_LABELED_DIR = Path(os.environ.get("FACTSCORE_LABELED_DIR", "./data/factscore/labeled"))
H2_N_ENTITIES = 40  # pilot scale, same reasoning as H1's pilot

H2_SENTENCES_JSONL = DATA_PROCESSED / "h2_sentences.jsonl"
H2_SUBCLAIMS_JSONL = DATA_PROCESSED / "h2_subclaims.jsonl"
H2_LABELED_JSONL = DATA_PROCESSED / "h2_labeled_spans.jsonl"
H2_CALIBRATION_REPORT = OUTPUTS_REPORTS / "h2_calibration.json"

# --- Phase 0 replication on real FActScore data (2026-07-15) ---
# Same mechanistic test as Phase 0 stages 4-7, but using real data/labeled
# sentences that contain >=1 human-verified NS ("not supported by Wikipedia")
# atomic fact as the naturally-occurring "R has wrong content" condition,
# instead of Phase 0's synthetic year-perturbation. Reuses the frozen
# layer-22 probe from stage 3 unchanged.
#
# First attempt sampled uniformly across FActScore's real fame distribution
# and got DI-CC=0/150 sentences -- FActScore's entities skew far more
# obscure than Phase 0's hand-picked (Einstein, Gandhi, ...) figures, and
# DI-CC requires the decomposer to actually have confident parametric
# knowledge to draw a correction from. Archived as *_random.jsonl. Now
# restricted to FActScore's own "very freq" fame tier (36 entities, 302
# NS-containing sentences total -- use the whole pool, no further
# subsampling) to match Phase 0's original entity-fame condition.
H0_REAL_CATEGORY_FILTER = "very freq"
H0_REAL_N_SENTENCES = 302

H0_REAL_SENTENCES_JSONL = DATA_PROCESSED / "h0_real_sentences.jsonl"
H0_REAL_SUBCLAIMS_JSONL = DATA_PROCESSED / "h0_real_subclaims.jsonl"
H0_REAL_LABELED_JSONL = DATA_PROCESSED / "h0_real_labeled_spans.jsonl"
H0_REAL_ACTIVATIONS = OUTPUTS_ACTIVATIONS / "h0_real_activations.npz"
H0_REAL_TRANSFER_REPORT = OUTPUTS_REPORTS / "h0_real_transfer_eval.json"

# --- Phase 1 / H3 (CAD mitigation, see README.md) ---
# 191-entity pool (expanded 2026-07-15 from Phase 0's original 40; see
# phase0/data/bio_entities.py docstring). Needs a paired before/after
# comparison on the same entities, and 40 only yielded 14 baseline DI-CC
# cases -- too few for a paired test to detect a reduction.
H3_ALPHA = 1.0  # matches decompose.CAD_ALPHA_DEFAULT; kept here too for
                 # visibility in the report-writing stage (27) without
                 # importing the decomposition module just for a constant.
H3_BIOS_JSONL = DATA_PROCESSED / "h3_bios.jsonl"
H3_DECOMPOSITIONS_JSONL = DATA_PROCESSED / "h3_decompositions.jsonl"
H3_LABELED_JSONL = DATA_PROCESSED / "h3_labeled_spans.jsonl"
H3_MITIGATION_REPORT = OUTPUTS_REPORTS / "h3_mitigation.json"

# --- H0 sample-size expansion (2026-07-16): reuse H3's perturbed_baseline /
# vague_baseline conditions (same methodology as Phase 0, 191-entity pool)
# to replicate the stage 6/7 H0 transfer-eval test at n~64 DI-CC instead of
# the original 40-entity pool's n=14. See scripts/28, 29. ---
H3_BASELINE_ACTIVATIONS = OUTPUTS_ACTIVATIONS / "h3_baseline_activations.npz"
H3_BASELINE_TRANSFER_REPORT = OUTPUTS_REPORTS / "h3_baseline_transfer_eval.json"

# --- Misc ---
RANDOM_SEED = 42
