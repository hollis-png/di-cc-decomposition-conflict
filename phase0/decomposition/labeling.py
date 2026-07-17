"""Combine the R |= c_i entailment check with the closed-book recoverability
check into the three-way Phase 0 label (design doc section 2, Phase 0 step 3):

  entailment(R, c_i) == entailment                              -> "legit"
  entailment(R, c_i) == contradiction  AND recoverable(c_i)      -> "DI-CC"
  entailment(R, c_i) == neutral        AND recoverable(c_i)      -> "DI-UE"
  not recoverable(c_i)  (contradiction or neutral, but not
                         traceable to the decomposer's own
                         parametric knowledge)                   -> "excluded"

"recoverable" is approximated as entailment(closed_book_dump, c_i) ==
entailment, i.e. the decomposer's own free-form parametric recall about the
subject already contains/entails this claim.
"""
from phase0.decomposition.entailment import nli_label


def label_subclaim(nli_tokenizer, nli_model, id2label, bio_text: str, knowledge_dump: str, claim_text: str) -> str:
    r_label = nli_label(nli_tokenizer, nli_model, id2label, premise=bio_text, hypothesis=claim_text)
    if r_label == "entailment":
        return "legit"

    recoverable = nli_label(nli_tokenizer, nli_model, id2label, premise=knowledge_dump, hypothesis=claim_text) == "entailment"
    if not recoverable:
        return "excluded"
    return "DI-CC" if r_label == "contradiction" else "DI-UE"
