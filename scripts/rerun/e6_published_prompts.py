"""E6: decomposition instructions close to published pipelines, on the perturbed biographies (greedy).
 perturbed_nopron : the project's own DECOMPOSE_PROMPT_NEUTRAL (no correct/fill language), whole biography.
 perturbed_factscore : FActScore-style sentence-level few-shot prompt, demos from shmsw25/FActScore demos/demons.json
   (first 8 fixed demos; the original also retrieves a BM25-matched demo for biographies, which we omit).
Usage (from the project dir): e6_published_prompts.py DEMONS_JSON"""
import json, re, sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path.cwd())); sys.path.insert(0, str(Path.cwd() / "rerun"))
import r25_decompose
import phase0.decomposition.decompose as D
from phase0.activations.extract import generate_short_answer

demons = json.load(open(sys.argv[1], encoding="utf-8"))
HEAD = "Please breakdown the following sentence into independent facts: "
FEWSHOT = "".join(HEAD + s + "\n" + "".join(f"- {f}\n" for f in demons[s]) + "\n" for s in list(demons)[:8])
_SENT = re.compile(r"(?<=[.!?])\s+")

def decompose_factscore(tokenizer, model, bio, prompt_template=None):
    facts = []
    for sent in _SENT.split(bio.strip()):
        if not sent.strip():
            continue
        # FActScore uses a completion-style LM: feed the few-shot text raw (no chat template) and read the continuation
        inp = tokenizer(FEWSHOT + HEAD + sent + "\n", return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        raw = raw.split(HEAD)[0].strip().split("\n\n")[0]  # the model may continue with further demo-style blocks: keep only the first block
        facts += [l.lstrip("-•* ").strip() for l in raw.split("\n") if l.strip().startswith(("-", "•", "*")) and l.lstrip("-•* ").strip()]
    text = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1))
    return D._parse_numbered_subclaims(text), text

r25_decompose.ALL["perturbed_nopron"] = ("perturbed_bio", D.DECOMPOSE_PROMPT_NEUTRAL)
r25_decompose.ALL["perturbed_factscore"] = ("perturbed_bio", "FACTSCORE")
r25_decompose.decompose_into_subclaims = (lambda orig: (lambda tok, model, bio, prompt_template=D.DECOMPOSE_PROMPT:
    decompose_factscore(tok, model, bio) if prompt_template == "FACTSCORE" else orig(tok, model, bio, prompt_template=prompt_template)))(r25_decompose.decompose_into_subclaims)
sys.argv = ["r25", "--conditions"] + (sys.argv[2:] or ["perturbed_nopron", "perturbed_factscore"])
r25_decompose.main()
