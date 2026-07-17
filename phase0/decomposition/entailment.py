"""NLI wrapper used for both the R |= c_i entailment check and the closed-
book recoverability check (design doc section 1.2). Reusing one 3-way NLI
model for both purposes: entailment(R, c_i) determines "legit" vs
"contradiction"/"neutral"; entailment(closed_book_dump, c_i) determines
whether the added content is recoverable from the decomposer's own
parametric knowledge.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config


def load_nli_model():
    tokenizer = AutoTokenizer.from_pretrained(config.NLI_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(config.NLI_MODEL_NAME)
    model.to(config.DEVICE)
    model.eval()
    # label2id varies by checkpoint; read it instead of hardcoding indices.
    id2label = {v: k.lower() for k, v in model.config.label2id.items()}
    return tokenizer, model, id2label


@torch.no_grad()
def nli_label(tokenizer, model, id2label, premise: str, hypothesis: str) -> str:
    """Returns one of 'entailment', 'neutral', 'contradiction'."""
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(config.DEVICE)
    logits = model(**inputs).logits[0]
    pred_id = int(logits.argmax())
    return id2label[pred_id]
