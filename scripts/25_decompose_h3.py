"""Stage 25 (H3 CAD mitigation): for each entity from stage 24, decompose
the perturbed bio and the vague bio under two decode conditions each --
baseline (decompose_into_subclaims) and CAD (decompose_with_cad) --
producing up to 4 records per entity (fewer if no perturbed bio was
available, i.e. no year found to tamper).

perturbed_baseline / perturbed_cad measure the primary DI-CC-rate effect.
vague_baseline / vague_cad measure the coreference-resolution side effect
(see README.md).

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
from phase0.decomposition.decompose import (
    DECOMPOSE_PROMPT,
    DECOMPOSE_PROMPT_CORRECT,
    decompose_into_subclaims,
    decompose_with_cad,
)
from phase0.utils.keep_awake import prevent_sleep

prevent_sleep()

# (bio_field, decompose_prompt, condition_label)
CONDITIONS = [
    ("perturbed_bio", DECOMPOSE_PROMPT_CORRECT, "perturbed_baseline"),
    ("perturbed_bio", DECOMPOSE_PROMPT_CORRECT, "perturbed_cad"),
    ("vague_bio", DECOMPOSE_PROMPT, "vague_baseline"),
    ("vague_bio", DECOMPOSE_PROMPT, "vague_cad"),
]


def load_done_keys() -> set[tuple[str, str]]:
    done = set()
    if not config.H3_DECOMPOSITIONS_JSONL.exists():
        return done
    with open(config.H3_DECOMPOSITIONS_JSONL, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            done.add((rec["person"], rec["condition"]))
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process first N bio records (smoke test)")
    args = parser.parse_args()

    bios = []
    with open(config.H3_BIOS_JSONL, encoding="utf-8") as f:
        for line in f:
            bios.append(json.loads(line))
    if args.limit is not None:
        bios = bios[: args.limit]

    work = []
    for rec in bios:
        for bio_field, prompt, condition in CONDITIONS:
            if rec.get(bio_field):
                work.append((rec, bio_field, prompt, condition))

    done_keys = load_done_keys()
    remaining = [w for w in work if (w[0]["person"], w[3]) not in done_keys]
    print(f"Loaded {len(bios)} bio records / {len(work)} target decompositions; "
          f"{len(done_keys)} already done; {len(remaining)} remaining")

    if not remaining:
        print("Nothing left to do.")
        return

    tokenizer, model = load_decomposer_model()
    print("Model loaded")

    t0 = time.time()
    n_written = 0
    with open(config.H3_DECOMPOSITIONS_JSONL, "a", encoding="utf-8") as out_f:
        for i, (rec, bio_field, prompt, condition) in enumerate(remaining):
            person = rec["person"]
            bio_text = rec[bio_field]

            if condition.endswith("_cad"):
                subclaims, raw = decompose_with_cad(tokenizer, model, person, bio_text, prompt_template=prompt)
            else:
                subclaims, raw = decompose_into_subclaims(tokenizer, model, bio_text, prompt_template=prompt)

            out_rec = {
                "person": person,
                "condition": condition,
                "bio_variant": "perturbed" if bio_field == "perturbed_bio" else "vague",
                "decode_mode": "cad" if condition.endswith("_cad") else "baseline",
                "bio": bio_text,
                "knowledge_dump": rec["knowledge_dump"],
                "raw_decomposition": raw,
                "subclaims": [{"text": c["text"], "span": list(c["span"])} for c in subclaims],
            }
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            out_f.flush()
            n_written += 1

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = elapsed / (i + 1)
                remaining_est_min = rate * (len(remaining) - i - 1) / 60
                print(f"  [{i + 1}/{len(remaining)}] {person} ({condition}): {len(subclaims)} subclaims "
                      f"({elapsed:.1f}s elapsed, ~{remaining_est_min:.1f}min remaining)")

    print(f"Appended {n_written} decomposition records to {config.H3_DECOMPOSITIONS_JSONL}")


if __name__ == "__main__":
    main()
