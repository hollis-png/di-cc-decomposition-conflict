"""Stage 28 (H0 sample-size expansion): reuse H3's existing 191-entity
decompositions and NLI labels -- specifically the perturbed_baseline and
vague_baseline conditions only (not the CAD conditions) -- to extract
activations for the same H0 transfer-eval test Phase 0 stage 6/7 ran on the
original 40-entity pool (n=14 DI-CC). No new decomposer calls: bios,
decompositions, and labels already exist from H3; this only extracts the
hidden states, which H3 never needed. Needs GPU.

Drafted by gemma-2-9b-it (LM Studio), reviewed and finalized by Claude.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from phase0.activations.extract import hidden_states_at_char_position, load_decomposer_model
from phase0.decomposition.decompose import DECOMPOSE_PROMPT, DECOMPOSE_PROMPT_CORRECT

_PROMPT_BY_VARIANT = {"vague": DECOMPOSE_PROMPT, "perturbed": DECOMPOSE_PROMPT_CORRECT}
_BASELINE_CONDITIONS = {"perturbed_baseline", "vague_baseline"}


def main():
    decompositions = {}
    with open(config.H3_DECOMPOSITIONS_JSONL, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            decompositions[(rec["person"], rec["condition"])] = rec["bio"]

    spans = []
    with open(config.H3_LABELED_JSONL, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["label"] != "excluded" and rec["condition"] in _BASELINE_CONDITIONS:
                spans.append(rec)
    print(f"Loaded {len(spans)} non-excluded baseline-condition labeled spans")

    tokenizer, model = load_decomposer_model()
    print("Model loaded")

    activations = []
    meta_records = []
    for i, span in enumerate(spans):
        person = span["person"]
        condition = span["condition"]
        prompt_template = _PROMPT_BY_VARIANT[span["bio_variant"]]
        bio = decompositions[(person, condition)]
        messages = [{"role": "user", "content": prompt_template.format(bio=bio)}]
        char_end = span["span"][1]
        act = hidden_states_at_char_position(tokenizer, model, messages, span["raw_decomposition"], char_end)
        activations.append(act)
        meta_records.append(
            {
                "row_id": i,
                "person": person,
                "condition": condition,
                "claim_text": span["claim_text"],
                "label": span["label"],
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(spans)}")

    activations_arr = np.stack(activations, axis=0)
    labels_arr = np.array([s["label"] for s in spans])
    np.savez_compressed(config.H3_BASELINE_ACTIVATIONS, activations=activations_arr, labels=labels_arr)
    print(f"Saved activations {activations_arr.shape} to {config.H3_BASELINE_ACTIVATIONS}")

    meta_path = config.H3_BASELINE_ACTIVATIONS.with_suffix(".meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for rec in meta_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"Label distribution: {Counter(labels_arr.tolist())}")


if __name__ == "__main__":
    main()
