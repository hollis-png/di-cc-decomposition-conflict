"""Stage 35 (CAD alpha sensitivity sweep): rerun H3's CAD-decoded conditions
(perturbed_cad for the primary DI-CC suppression effect, vague_cad for the
coreference/identity-drift side effect documented in README.md) at
alpha in {0.5, 1.5}, alongside the existing alpha=1.0 result already in
h3_decompositions.jsonl. Only the two _cad conditions are rerun --
_baseline conditions don't depend on alpha and are reused unchanged from
the existing H3 data.

Resumable by (person, condition, alpha), same pattern as stage 25, so a
--limit N smoke run and a later full run never redo completed work.
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
from phase0.decomposition.decompose import DECOMPOSE_PROMPT, DECOMPOSE_PROMPT_CORRECT, decompose_with_cad
from phase0.utils.keep_awake import prevent_sleep

prevent_sleep()

ALPHAS = [0.5, 1.5]
OUT_PATH = config.DATA_PROCESSED / "h3_alpha_sensitivity.jsonl"

# (bio_field, prompt, condition_label) -- same two _cad conditions as stage 25.
CONDITIONS = [
    ("perturbed_bio", DECOMPOSE_PROMPT_CORRECT, "perturbed_cad"),
    ("vague_bio", DECOMPOSE_PROMPT, "vague_cad"),
]


def load_done_keys() -> set[tuple[str, str, float]]:
    done = set()
    if not OUT_PATH.exists():
        return done
    with open(OUT_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            done.add((rec["person"], rec["condition"], rec["alpha"]))
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
            if not rec.get(bio_field):
                continue
            for alpha in ALPHAS:
                work.append((rec, bio_field, prompt, condition, alpha))

    done_keys = load_done_keys()
    remaining = [w for w in work if (w[0]["person"], w[3], w[4]) not in done_keys]
    print(f"Loaded {len(bios)} bio records (--limit={args.limit}) / {len(work)} target decompositions; "
          f"{len(done_keys)} already done; {len(remaining)} remaining")

    if not remaining:
        print("Nothing left to do.")
        return

    tokenizer, model = load_decomposer_model()
    print("Model loaded")

    t0 = time.time()
    n_written = 0
    with open(OUT_PATH, "a", encoding="utf-8") as out_f:
        for i, (rec, bio_field, prompt, condition, alpha) in enumerate(remaining):
            person = rec["person"]
            bio_text = rec[bio_field]

            subclaims, raw = decompose_with_cad(tokenizer, model, person, bio_text, prompt_template=prompt, alpha=alpha)

            out_rec = {
                "person": person,
                "condition": condition,
                "alpha": alpha,
                "bio_variant": "perturbed" if bio_field == "perturbed_bio" else "vague",
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
                print(f"  [{i + 1}/{len(remaining)}] {person} ({condition}, alpha={alpha}): {len(subclaims)} subclaims "
                      f"({elapsed:.1f}s elapsed, ~{remaining_est_min:.1f}min remaining)")

    print(f"Appended {n_written} decomposition records to {OUT_PATH}")


if __name__ == "__main__":
    main()
