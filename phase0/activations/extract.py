"""Model loading + hidden-state extraction utilities shared by every stage
that touches the decomposer model (NQ-Swap probe data, decomposition, transfer
eval). Kept generic: callers decide which token position / layer to use.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import config


def load_decomposer_model():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=config.BNB_4BIT_COMPUTE_DTYPE,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.DECOMPOSER_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.DECOMPOSER_MODEL_NAME,
        quantization_config=quant_config,
        device_map=config.DEVICE,
        output_hidden_states=True,
    )
    model.eval()
    return tokenizer, model


def _build_chat_inputs(tokenizer, messages: list[dict]):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(prompt, return_tensors="pt").to(config.DEVICE)


@torch.no_grad()
def hidden_states_at_last_token(tokenizer, model, messages: list[dict]) -> np.ndarray:
    """Return an (num_hidden_states, hidden_size) array: the residual-stream
    activation at the final prompt token (i.e. right at the assistant
    generation-start marker), for every layer. This is the model's "decision
    state" right before it would start generating an answer -- the standard
    probing position for context-vs-parametric-knowledge conflict (cf.
    Zhao_2024_SAE_Representation, Pham_2026_WhereKnowledgeCollides).
    `messages` is a chat-format list, e.g. [{"role": "user", "content": ...}].
    """
    inputs = _build_chat_inputs(tokenizer, messages)
    outputs = model(**inputs)
    # outputs.hidden_states: tuple of (num_hidden_states,) each (1, seq_len, hidden)
    last_token_states = [h[0, -1, :].float().cpu().numpy() for h in outputs.hidden_states]
    return np.stack(last_token_states, axis=0)


@torch.no_grad()
def hidden_states_at_char_position(tokenizer, model, messages: list[dict], response_text: str, char_end: int) -> np.ndarray:
    """Teacher-forced variant of hidden_states_at_last_token: reconstructs
    prompt + response[:char_end] (i.e. the response truncated right after a
    given sub-claim ends) and returns the (num_hidden_states, hidden_size)
    activation at that truncation's last token. Used to extract the
    "decision state" for a specific sub-claim produced during decomposition
    (design doc Phase 0 step 3, groups a/b/c).
    """
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = prompt + response_text[:char_end]
    inputs = tokenizer(full_text, return_tensors="pt").to(config.DEVICE)
    outputs = model(**inputs)
    last_token_states = [h[0, -1, :].float().cpu().numpy() for h in outputs.hidden_states]
    return np.stack(last_token_states, axis=0)


@torch.no_grad()
def generate_short_answer(tokenizer, model, messages: list[dict], max_new_tokens: int = 16) -> str:
    inputs = _build_chat_inputs(tokenizer, messages)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
