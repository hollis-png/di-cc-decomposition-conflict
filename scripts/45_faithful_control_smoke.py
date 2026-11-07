"""Stage 45 (smoke test): instruction-matched (faithful) and length-matched
(sham) controls for the perturbed-bio decomposition.

Runs three conditions on the first N perturbed bios and labels every subclaim
with the same label_subclaim logic as stage 26:
  perturbed_correct   DECOMPOSE_PROMPT_CORRECT (re-run, for a paired reference)
  perturbed_faithful  DECOMPOSE_PROMPT_FAITHFUL
  perturbed_sham      DECOMPOSE_PROMPT_SHAM

Read-only w.r.t. existing outputs: writes only
outputs/reports/faithful_control_smoke.jsonl (overwritten each run).
Sequential model loading (decomposer, then NLI) to fit a 12GB GPU.

Smoke test only: with small N the DI-CC counts are too few for any
significance test. Read them as a sanity check on direction and parse rate.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

import config
from phase0.activations.extract import load_decomposer_model
from phase0.decomposition.control_prompts import (
    DECOMPOSE_PROMPT_FAITHFUL,
    DECOMPOSE_PROMPT_SHAM,
    middle_word_counts,
)
from phase0.decomposition.decompose import DECOMPOSE_PROMPT_CORRECT, decompose_into_subclaims
from phase0.decomposition.entailment import load_nli_model
from phase0.decomposition.labeling import label_subclaim

CONDITIONS = [
    ("perturbed_correct", DECOMPOSE_PROMPT_CORRECT),
    ("perturbed_faithful", DECOMPOSE_PROMPT_FAITHFUL),
    ("perturbed_sham", DECOMPOSE_PROMPT_SHAM),
]
DEFAULT_BIOS = Path(__file__).parent.parent / "data" / "h3_bios_191person.jsonl"
OUT_PATH = config.OUTPUTS_REPORTS / "faithful_control_smoke.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="First N entities with a perturbed bio")
    parser.add_argument("--bios", type=Path, default=DEFAULT_BIOS)
    args = parser.parse_args()

    print("varying-sentence word counts:", middle_word_counts())

    bios = []
    with open(args.bios, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("perturbed_bio"):
                bios.append(rec)
            if len(bios) >= args.limit:
                break
    print(f"Using {len(bios)} entities x {len(CONDITIONS)} conditions")

    tokenizer, model = load_decomposer_model()
    decomps = []
    for i, rec in enumerate(bios):
        for condition, prompt in CONDITIONS:
            subclaims, raw = decompose_into_subclaims(tokenizer, model, rec["perturbed_bio"], prompt_template=prompt)
            decomps.append((rec, condition, subclaims, raw))
        print(f"  decomposed {i + 1}/{len(bios)}")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    nli_tokenizer, nli_model, id2label = load_nli_model()
    counts: dict[str, dict[str, int]] = {}
    parse_fail: dict[str, int] = {c: 0 for c, _ in CONDITIONS}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        for rec, condition, subclaims, raw in decomps:
            c = counts.setdefault(condition, {"legit": 0, "DI-CC": 0, "DI-UE": 0, "excluded": 0})
            if not subclaims:
                parse_fail[condition] += 1
            labeled = []
            for claim in subclaims:
                label = label_subclaim(
                    nli_tokenizer, nli_model, id2label,
                    bio_text=rec["perturbed_bio"], knowledge_dump=rec["knowledge_dump"], claim_text=claim["text"],
                )
                c[label] += 1
                labeled.append({"text": claim["text"], "label": label})
            out_f.write(json.dumps({
                "person": rec["person"], "condition": condition, "bio": rec["perturbed_bio"],
                "raw_decomposition": raw, "labeled_subclaims": labeled,
            }, ensure_ascii=False) + "\n")

    print("\n=== label counts by condition (subclaim level) ===")
    for condition, _ in CONDITIONS:
        c = counts.get(condition, {})
        n = sum(c.values())
        rate = c.get("DI-CC", 0) / n if n else float("nan")
        print(f"{condition:20s} n_claims={n:4d} {c}  DI-CC rate={rate:.3%}  parse_fail_entities={parse_fail[condition]}/{len(bios)}")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
