"""Stage 24 (H3 CAD mitigation): generate a vague bio, a perturbed
(year-tampered) bio, and a knowledge dump for the expanded ~191-entity
pool. Bio generation only -- decomposition (baseline vs CAD) happens in
stage 25, kept separate so bios are generated exactly once regardless of
how many decode conditions run against them.

Resumable (append+flush+skip-done) + sleep prevention + UTF-8 stdout.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from phase0.activations.extract import load_decomposer_model
from phase0.data.bio_entities import BIO_ENTITIES
from phase0.decomposition.decompose import (
    generate_bio,
    generate_knowledge_dump,
    generate_specific_bio,
    perturb_one_year,
)
from phase0.utils.keep_awake import prevent_sleep

prevent_sleep()

# stable per-person seed regardless of resume order (matches stage 4's
# original seed=i semantics, indexed against the full entity list)
_ENTITY_SEED = {person: idx for idx, person in enumerate(BIO_ENTITIES)}


def load_done_people() -> set[str]:
    done = set()
    if not config.H3_BIOS_JSONL.exists():
        return done
    with open(config.H3_BIOS_JSONL, encoding="utf-8") as f:
        for line in f:
            done.add(json.loads(line)["person"])
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    people = BIO_ENTITIES[: args.limit] if args.limit else BIO_ENTITIES
    done = load_done_people()
    remaining = [p for p in people if p not in done]
    print(f"Loaded {len(people)} target people; {len(done)} already done; {len(remaining)} remaining")

    if not remaining:
        print("Nothing left to do.")
        return

    tokenizer, model = load_decomposer_model()
    print("Model loaded")

    t0 = time.time()
    n_written = 0
    n_no_year = 0
    with open(config.H3_BIOS_JSONL, "a", encoding="utf-8") as out_f:
        for i, person in enumerate(remaining):
            knowledge_dump = generate_knowledge_dump(tokenizer, model, person)
            vague_bio = generate_bio(tokenizer, model, person)
            specific_bio = generate_specific_bio(tokenizer, model, person)
            perturbed_bio = perturb_one_year(specific_bio, seed=_ENTITY_SEED[person])

            if perturbed_bio is None:
                n_no_year += 1

            record = {
                "person": person,
                "knowledge_dump": knowledge_dump,
                "vague_bio": vague_bio,
                "specific_bio": specific_bio,
                "perturbed_bio": perturbed_bio,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            n_written += 1

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [{i + 1}/{len(remaining)}] {person} "
                      f"({elapsed:.1f}s elapsed, {n_no_year} no-year so far)")

    print(f"Appended {n_written} bio records to {config.H3_BIOS_JSONL} "
          f"({n_no_year} had no perturbable year)")


if __name__ == "__main__":
    main()
