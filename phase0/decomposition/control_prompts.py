"""Control prompts for the instruction-matched / length-matched checks on the
perturbed-bio decomposition (added after reviewing Chen et al. 2026, CDD).

DECOMPOSE_PROMPT_CORRECT (decompose.py) explicitly asks the decomposer to
correct a wrong stated fact from its own knowledge, so part of the DI-CC rate
on perturbed bios may come from following that instruction rather than from
decomposition itself. These two variants keep the same skeleton and differ
only in the one "Before finalizing each claim ..." sentence:

- FAITHFUL: instruction-matched. Same position and intent, but tells the
  model to stay with the biography. What DI-CC survives an explicit ban?
- SHAM: length-matched. Same position and length, but the sentence is
  semantically unrelated to facts. Is DI-CC just a prompt-length effect?

The shared prefix/suffix is copied from DECOMPOSE_PROMPT_CORRECT verbatim.
"""

_PREFIX = (
    "Decompose the following biography into a numbered list of atomic "
    "factual claims. Each claim must be a single self-contained sentence "
    "with no pronouns (resolve every \"he\"/\"she\"/\"it\"/\"his\"/\"her\" to "
    "the actual name). "
)
_SUFFIX = " Output only the numbered list, nothing else.\n\nBiography:\n{bio}"

_CORRECT_MIDDLE = (
    "Before finalizing each claim, check it against what "
    "you actually know: if a stated date, place, or fact is incorrect, "
    "output the corrected version instead of repeating the biography's "
    "wrong statement."
)

DECOMPOSE_PROMPT_FAITHFUL = (
    _PREFIX
    + "Before finalizing each claim, check it against the biography only: "
    "even if a stated date, place, or fact seems incorrect to you, output "
    "exactly what the biography states and do not correct, replace, or add "
    "any detail from your own knowledge."
    + _SUFFIX
)

DECOMPOSE_PROMPT_SHAM = (
    _PREFIX
    + "Before finalizing each claim, check its formatting: make sure it "
    "begins with a capital letter, ends with a period, and contains both "
    "a subject and a verb, and rewrite the claim if any of these is "
    "missing from it."
    + _SUFFIX
)


def middle_word_counts() -> dict[str, int]:
    """Word counts of the varying sentence, to confirm the sham is length-matched."""
    def middle(p: str) -> str:
        return p[len(_PREFIX): -len(_SUFFIX)]
    return {
        "correct": len(_CORRECT_MIDDLE.split()),
        "faithful": len(middle(DECOMPOSE_PROMPT_FAITHFUL).split()),
        "sham": len(middle(DECOMPOSE_PROMPT_SHAM).split()),
    }
