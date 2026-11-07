"""Rerun of stage 28 (new file): activations for EVERY labeled claim (including
'excluded', which the original stage 28 skipped) under the given conditions.
Saved float16 to keep size manageable. Position and prompt reconstruction are
identical to stage 28 (hidden_states_at_char_position).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from phase0.activations.extract import hidden_states_at_char_position, load_decomposer_model
from phase0.decomposition.control_prompts import DECOMPOSE_PROMPT_FAITHFUL, DECOMPOSE_PROMPT_SHAM
from phase0.decomposition.decompose import DECOMPOSE_PROMPT, DECOMPOSE_PROMPT_CORRECT

PROMPT = {"perturbed_baseline": DECOMPOSE_PROMPT_CORRECT, "vague_baseline": DECOMPOSE_PROMPT,
          "perturbed_faithful": DECOMPOSE_PROMPT_FAITHFUL, "perturbed_sham": DECOMPOSE_PROMPT_SHAM,
          "perturbed_cad": DECOMPOSE_PROMPT_CORRECT}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bio = {}
    for l in open(config.H3_DECOMPOSITIONS_JSONL, encoding="utf-8"):
        r = json.loads(l); bio[(r["person"], r["condition"])] = r["bio"]
    spans = [json.loads(l) for l in open(config.H3_LABELED_JSONL, encoding="utf-8")]
    spans = [s for s in spans if s["condition"] in a.conditions]
    print(f"{len(spans)} claims", flush=True)
    tok, model = load_decomposer_model()
    acts, meta = [], []
    for i, s in enumerate(spans):
        msgs = [{"role": "user", "content": PROMPT[s["condition"]].format(bio=bio[(s["person"], s["condition"])])}]
        acts.append(hidden_states_at_char_position(tok, model, msgs, s["raw_decomposition"], s["span"][1]).astype(np.float16))
        meta.append({"row_id": i, "person": s["person"], "condition": s["condition"],
                     "claim_text": s["claim_text"], "label": s["label"]})
        if (i + 1) % 200 == 0: print(f"  {i+1}/{len(spans)}", flush=True)
    np.savez_compressed(a.out, activations=np.stack(acts), labels=np.array([m["label"] for m in meta]))
    with open(a.out + ".meta.jsonl", "w", encoding="utf-8") as f:
        for m in meta: f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print("saved", a.out, flush=True)

if __name__ == "__main__":
    main()
