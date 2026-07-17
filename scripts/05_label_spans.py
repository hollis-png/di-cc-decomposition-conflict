"""Stage 5: label every sub-claim from stage 4 as "legit" / "DI-CC" / "DI-UE"
/ "excluded" using the NLI-based entailment + recoverability checks (design
doc section 2, Phase 0 step 3). Needs GPU (NLI model), but much lighter than
the decomposer -- runs fine alongside or after it.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from phase0.decomposition.entailment import load_nli_model
from phase0.decomposition.labeling import label_subclaim


def main():
    records = []
    with open(config.FACTSCORE_SUBCLAIMS_JSONL, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} people")

    nli_tokenizer, nli_model, id2label = load_nli_model()
    print(f"NLI model loaded, labels: {id2label}")

    labeled_spans = []
    counts = {"legit": 0, "DI-CC": 0, "DI-UE": 0, "excluded": 0}
    for rec in records:
        for claim in rec["subclaims"]:
            label = label_subclaim(
                nli_tokenizer, nli_model, id2label,
                bio_text=rec["bio"], knowledge_dump=rec["knowledge_dump"], claim_text=claim["text"],
            )
            counts[label] += 1
            labeled_spans.append(
                {
                    "person": rec["person"],
                    "bio_variant": rec["bio_variant"],
                    "bio": rec["bio"],
                    "raw_decomposition": rec["raw_decomposition"],
                    "claim_text": claim["text"],
                    "span": claim["span"],
                    "label": label,
                }
            )

    with open(config.LABELED_SPANS_JSONL, "w", encoding="utf-8") as f:
        for rec in labeled_spans:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Label counts: {counts}")
    print(f"Wrote {len(labeled_spans)} labeled spans to {config.LABELED_SPANS_JSONL}")


if __name__ == "__main__":
    main()
