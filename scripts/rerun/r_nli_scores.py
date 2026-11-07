"""Per-claim NLI probabilities for (source bio, claim) and (knowledge dump, claim).
The DI-CC/DI-UE/legit/excluded labels are deterministic functions of the argmax of
exactly these two NLI calls, so these scores are the 'NLI baseline detector'."""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from phase0.decomposition.entailment import load_nli_model

@torch.no_grad()
def probs(tok, model, id2label, premise, hyp):
    x = tok(premise, hyp, return_tensors="pt", truncation=True, max_length=512).to(config.DEVICE)
    p = torch.softmax(model(**x).logits[0].float(), -1).cpu().tolist()
    return {id2label[i]: p[i] for i in range(len(p))}

def main():
    ctx = {}
    for l in open(config.H3_DECOMPOSITIONS_JSONL, encoding="utf-8"):
        r = json.loads(l); ctx[(r["person"], r["condition"])] = (r["bio"], r["knowledge_dump"])
    spans = [json.loads(l) for l in open(config.H3_LABELED_JSONL, encoding="utf-8")]
    tok, model, id2label = load_nli_model()
    out = config.OUTPUTS_REPORTS / "rerun_nli_scores.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for i, s in enumerate(spans):
            bio, dump = ctx[(s["person"], s["condition"])]
            f.write(json.dumps({"person": s["person"], "condition": s["condition"], "claim_text": s["claim_text"],
                "label": s["label"], "nli_src": probs(tok, model, id2label, bio, s["claim_text"]),
                "nli_dump": probs(tok, model, id2label, dump, s["claim_text"])}, ensure_ascii=False) + "\n")
            if (i + 1) % 500 == 0: print(f"  {i+1}/{len(spans)}", flush=True)
    print("saved", out)

if __name__ == "__main__":
    main()
