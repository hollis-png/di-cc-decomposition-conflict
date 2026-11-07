"""Run a stage script with the double-BOS fix: tokenize chat-templated text with
add_special_tokens=False (the template already contains <s> for Mistral)."""
import runpy, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import config
import phase0.activations.extract as E

def _build_chat_inputs(tokenizer, messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(config.DEVICE)

import numpy as np, torch
@torch.no_grad()
def hidden_states_at_char_position(tokenizer, model, messages, response_text, char_end):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt + response_text[:char_end], add_special_tokens=False, return_tensors="pt").to(config.DEVICE)
    out = model(**inputs)
    return np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states], axis=0)

@torch.no_grad()
def hidden_states_at_last_token(tokenizer, model, messages):
    inputs = _build_chat_inputs(tokenizer, messages)
    out = model(**inputs)
    return np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states], axis=0)

E._build_chat_inputs = _build_chat_inputs
E.hidden_states_at_char_position = hidden_states_at_char_position
E.hidden_states_at_last_token = hidden_states_at_last_token
script = sys.argv[1]; sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
