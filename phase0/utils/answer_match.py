"""SQuAD/NQ-style normalized answer matching.

Used to decide whether the model's closed-book answer agrees with a gold
answer string, i.e. to compute a_m == org_answer (design doc section 1.2).
"""
import re
import string


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def answer_matches(prediction: str, gold_answers: list[str]) -> bool:
    """True if the normalized prediction exact-matches, or word-boundary-safe
    contains/is-contained-by, any normalized gold answer. Short-answer QA
    predictions are often a superset (e.g. "23 episodes" vs gold "23"), so
    containment is included; padding with spaces enforces whole-word
    boundaries so e.g. gold "an" cannot match inside predicted "band".
    """
    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return False
    pred_padded = f" {pred_norm} "
    for gold in gold_answers:
        gold_norm = normalize_answer(gold)
        if not gold_norm:
            continue
        gold_padded = f" {gold_norm} "
        if pred_norm == gold_norm or gold_padded in pred_padded or pred_padded in gold_padded:
            return True
    return False
