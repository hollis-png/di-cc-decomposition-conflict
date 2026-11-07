"""E3: r25 decomposition (perturbed_baseline) with optional temperature sampling. Usage: e3_decompose.py greedy|sample SEED"""
import sys
from pathlib import Path
mode, seed = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, str(Path.cwd())); sys.path.insert(0, str(Path.cwd() / "rerun"))
import torch
import phase0.decomposition.decompose as D
from phase0.activations.extract import _build_chat_inputs
if mode == "sample":
    torch.manual_seed(seed)
    def _gen(tokenizer, model, messages, max_new_tokens=16):
        inputs = _build_chat_inputs(tokenizer, messages)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=1.0, top_k=0,
                             pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    D.generate_short_answer = _gen
import r25_decompose
sys.argv = ["r25", "--conditions", "perturbed_baseline"]
r25_decompose.main()
