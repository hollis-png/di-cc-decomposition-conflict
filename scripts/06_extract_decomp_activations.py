"""Stage 6: extract the decomposer's hidden states at the end of every
labeled sub-claim span (excluding "excluded"), for all layers. These are the
three-way (a)/(b)/(c) transfer-test groups: DI-CC, legit, DI-UE (design doc
Phase 0 step 3). Needs GPU.
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


def main():
    spans = []
    with open(config.LABELED_SPANS_JSONL, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["label"] != "excluded":
                spans.append(rec)
    print(f"Loaded {len(spans)} non-excluded labeled spans")

    tokenizer, model = load_decomposer_model()
    print("Model loaded")

    activations = []
    meta_records = []
    for i, span in enumerate(spans):
        prompt_template = _PROMPT_BY_VARIANT[span["bio_variant"]]
        messages = [{"role": "user", "content": prompt_template.format(bio=span["bio"])}]
        char_end = span["span"][1]
        act = hidden_states_at_char_position(tokenizer, model, messages, span["raw_decomposition"], char_end)
        activations.append(act)
        meta_records.append(
            {
                "row_id": i,
                "person": span["person"],
                "claim_text": span["claim_text"],
                "label": span["label"],
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(spans)}")

    activations_arr = np.stack(activations, axis=0)
    labels_arr = np.array([s["label"] for s in spans])
    np.savez_compressed(config.DECOMP_ACTIVATIONS, activations=activations_arr, labels=labels_arr)
    print(f"Saved activations {activations_arr.shape} to {config.DECOMP_ACTIVATIONS}")

    meta_path = config.DECOMP_ACTIVATIONS.with_suffix(".meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for rec in meta_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"Label distribution: {Counter(labels_arr.tolist())}")


if __name__ == "__main__":
    main()
