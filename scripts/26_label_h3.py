"""Stage 26 (H3 CAD mitigation): label every subclaim from stage 25 as
legit/DI-CC/DI-UE/excluded, identical logic to Phase 0 stage 5 / H0-real
stage 19. Needs GPU (NLI model only).

Reports label counts broken down by condition (perturbed_baseline,
perturbed_cad, vague_baseline, vague_cad) so the primary DI-CC-rate effect
and the coreference-resolution side effect can both be read off directly.
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


def main():
    records = []
    with open(config.H3_DECOMPOSITIONS_JSONL, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} decomposition records")

    nli_tokenizer, nli_model, id2label = load_nli_model()
    print(f"NLI model loaded, labels: {id2label}")

    labeled_spans = []
    counts_by_condition: dict[str, dict[str, int]] = {}
    for i, rec in enumerate(records):
        condition = rec["condition"]
        counts = counts_by_condition.setdefault(condition, {"legit": 0, "DI-CC": 0, "DI-UE": 0, "excluded": 0})
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
                    "bio_variant": rec["bio_variant"],
                    "decode_mode": rec["decode_mode"],
                    "raw_decomposition": rec["raw_decomposition"],
                    "claim_text": claim["text"],
                    "span": claim["span"],
                    "label": label,
                }
            )
        if (i + 1) % 100 == 0:
            print(f"  labeled {i + 1}/{len(records)} records")

    with open(config.H3_LABELED_JSONL, "w", encoding="utf-8") as f:
        for rec in labeled_spans:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Label counts by condition:")
    for condition, counts in counts_by_condition.items():
        print(f"  {condition}: {counts}")
    print(f"Wrote {len(labeled_spans)} labeled spans to {config.H3_LABELED_JSONL}")


if __name__ == "__main__":
    main()
