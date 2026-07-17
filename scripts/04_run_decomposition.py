"""Stage 4: for each person, generate a closed-book biography R, generate a
broader closed-book knowledge dump (used later for the recoverability
check), and decompose R into atomic sub-claims. Needs GPU.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from phase0.activations.extract import load_decomposer_model
from phase0.data.bio_entities import BIO_ENTITIES
from phase0.decomposition.decompose import (
    DECOMPOSE_PROMPT_CORRECT,
    decompose_into_subclaims,
    generate_bio,
    generate_knowledge_dump,
    generate_specific_bio,
    perturb_one_year,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    people = BIO_ENTITIES[: args.limit] if args.limit else BIO_ENTITIES
    print(f"Processing {len(people)} people")

    tokenizer, model = load_decomposer_model()
    print("Model loaded")

    records = []
    t0 = time.time()
    for i, person in enumerate(people):
        knowledge_dump = generate_knowledge_dump(tokenizer, model, person)

        # Variant 1: vague bio -> decomposer fills unstated gaps -> DI-UE-prone
        vague_bio = generate_bio(tokenizer, model, person)
        vague_subclaims, vague_raw = decompose_into_subclaims(tokenizer, model, vague_bio)
        records.append(
            {
                "person": person,
                "bio_variant": "vague",
                "bio": vague_bio,
                "knowledge_dump": knowledge_dump,
                "raw_decomposition": vague_raw,
                "subclaims": [{"text": c["text"], "span": list(c["span"])} for c in vague_subclaims],
            }
        )

        # Variant 2: specific bio with one year swapped (NQ-Swap-style
        # substitution) -> decomposer may "correct" it -> DI-CC-prone
        specific_bio = generate_specific_bio(tokenizer, model, person)
        perturbed_bio = perturb_one_year(specific_bio, seed=i)
        n_perturbed_subclaims = 0
        if perturbed_bio is not None:
            perturbed_subclaims, perturbed_raw = decompose_into_subclaims(
                tokenizer, model, perturbed_bio, prompt_template=DECOMPOSE_PROMPT_CORRECT
            )
            n_perturbed_subclaims = len(perturbed_subclaims)
            records.append(
                {
                    "person": person,
                    "bio_variant": "perturbed",
                    "bio": perturbed_bio,
                    "knowledge_dump": knowledge_dump,
                    "raw_decomposition": perturbed_raw,
                    "subclaims": [{"text": c["text"], "span": list(c["span"])} for c in perturbed_subclaims],
                }
            )

        print(f"  [{i + 1}/{len(people)}] {person}: {len(vague_subclaims)} vague + "
              f"{n_perturbed_subclaims} perturbed subclaims ({time.time() - t0:.1f}s elapsed)")

    with open(config.FACTSCORE_SUBCLAIMS_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    total_claims = sum(len(r["subclaims"]) for r in records)
    print(f"Wrote {len(records)} bio records / {total_claims} subclaims to {config.FACTSCORE_SUBCLAIMS_JSONL}")


if __name__ == "__main__":
    main()
