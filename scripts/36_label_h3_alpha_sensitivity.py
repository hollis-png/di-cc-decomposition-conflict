"""Stage 36 (CAD alpha sensitivity, labeling): label every
subclaim from stage 35 (h3_alpha_sensitivity.jsonl, alpha in {0.5, 1.5}) as
legit/DI-CC/DI-UE/excluded, identical logic to stage 26. Needs GPU (NLI model
only).

Reports label counts broken down by (condition, alpha) so the primary
DI-CC-rate effect and the coreference-resolution side effect can be read off
directly for each new alpha value.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from phase0.decomposition.entailment import load_nli_model
from phase0.decomposition.labeling import label_subclaim
from phase0.utils.keep_awake import prevent_sleep

prevent_sleep()

H3_ALPHA_SENSITIVITY_JSONL = config.DATA_PROCESSED / "h3_alpha_sensitivity.jsonl"
H3_ALPHA_SENSITIVITY_LABELED_JSONL = config.DATA_PROCESSED / "h3_alpha_sensitivity_labeled.jsonl"


def main():
    records = []
    with open(H3_ALPHA_SENSITIVITY_JSONL, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} decomposition records")

    nli_tokenizer, nli_model, id2label = load_nli_model()
    print(f"NLI model loaded, labels: {id2label}")

    labeled_spans = []
    counts_by_key: dict[tuple[str, float], dict[str, int]] = {}
    for i, rec in enumerate(records):
        condition = rec["condition"]
        alpha = rec["alpha"]
        key = (condition, alpha)
        counts = counts_by_key.setdefault(key, {"legit": 0, "DI-CC": 0, "DI-UE": 0, "excluded": 0})
        for claim in rec["subclaims"]:
            label = label_subclaim(
                nli_tokenizer, nli_model, id2label,
                bio_text=rec["bio"], knowledge_dump=rec["knowledge_dump"], claim_text=claim["text"],
            )
            counts[label] += 1
            labeled_spans.append(
                {
                    "person": rec["person"],
                    "condition": condition,
                    "alpha": alpha,
                    "bio_variant": rec["bio_variant"],
                    "raw_decomposition": rec["raw_decomposition"],
                    "claim_text": claim["text"],
                    "span": claim["span"],
                    "label": label,
                }
            )
        if (i + 1) % 100 == 0:
            print(f"  labeled {i + 1}/{len(records)} records")

    with open(H3_ALPHA_SENSITIVITY_LABELED_JSONL, "w", encoding="utf-8") as f:
        for rec in labeled_spans:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Label counts by (condition, alpha):")
    for (condition, alpha), counts in sorted(counts_by_key.items()):
        print(f"  {condition}, alpha={alpha}: {counts}")
    print(f"Wrote {len(labeled_spans)} labeled spans to {H3_ALPHA_SENSITIVITY_LABELED_JSONL}")


if __name__ == "__main__":
    main()
