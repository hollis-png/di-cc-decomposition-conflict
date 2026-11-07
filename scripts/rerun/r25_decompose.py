"""Rerun of stage 25 (new file; original scripts untouched). Decomposes the
191-entity pool under the conditions needed for the 2026-11 self-review checks.
Conditions: perturbed_baseline / vague_baseline (paper's H3 baselines),
perturbed_faithful / perturbed_sham (instruction-matched controls from
control_prompts.py), perturbed_cad (paper's CAD, alpha=1.0).
Resumable. Greedy decoding, same 4-bit NF4 Qwen2.5-7B as the paper.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from phase0.activations.extract import load_decomposer_model
from phase0.decomposition.control_prompts import DECOMPOSE_PROMPT_FAITHFUL, DECOMPOSE_PROMPT_SHAM
from phase0.decomposition.decompose import (DECOMPOSE_PROMPT, DECOMPOSE_PROMPT_CORRECT,
    decompose_into_subclaims, decompose_with_cad)

ALL = {
    "perturbed_baseline": ("perturbed_bio", DECOMPOSE_PROMPT_CORRECT),
    "vague_baseline": ("vague_bio", DECOMPOSE_PROMPT),
    "perturbed_faithful": ("perturbed_bio", DECOMPOSE_PROMPT_FAITHFUL),
    "perturbed_sham": ("perturbed_bio", DECOMPOSE_PROMPT_SHAM),
    "perturbed_cad": ("perturbed_bio", DECOMPOSE_PROMPT_CORRECT),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    bios = [json.loads(l) for l in open(config.H3_BIOS_JSONL, encoding="utf-8")]
    if a.limit: bios = bios[:a.limit]
    out = config.H3_DECOMPOSITIONS_JSONL
    done = set()
    if out.exists():
        for l in open(out, encoding="utf-8"):
            r = json.loads(l); done.add((r["person"], r["condition"]))
    work = [(r, c) for c in a.conditions for r in bios if r.get(ALL[c][0]) and (r["person"], c) not in done]
    print(f"{len(work)} decompositions to do ({len(done)} done)", flush=True)
    if not work: return
    tok, model = load_decomposer_model()
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i, (rec, cond) in enumerate(work):
            field, prompt = ALL[cond]
            if cond.endswith("_cad"):
                subs, raw = decompose_with_cad(tok, model, rec["person"], rec[field], prompt_template=prompt)
            else:
                subs, raw = decompose_into_subclaims(tok, model, rec[field], prompt_template=prompt)
            f.write(json.dumps({"person": rec["person"], "condition": cond,
                "bio_variant": "perturbed" if field == "perturbed_bio" else "vague",
                "decode_mode": "cad" if cond.endswith("_cad") else "baseline",
                "bio": rec[field], "knowledge_dump": rec["knowledge_dump"], "raw_decomposition": raw,
                "subclaims": [{"text": c["text"], "span": list(c["span"])} for c in subs]}, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                el = time.time() - t0
                print(f"  [{i+1}/{len(work)}] {cond} {el:.0f}s, ~{el/(i+1)*(len(work)-i-1)/60:.1f}min left", flush=True)

if __name__ == "__main__":
    main()
