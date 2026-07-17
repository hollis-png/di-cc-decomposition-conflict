"""Load the NQ-Swap classic context-memory-conflict dataset.

Source: pminervini/NQ-Swap on the HF Hub (Longpre et al. 2021 construction,
same dataset used by Shi_2024_CAD and Zhao_2024_SAE_Representation).
"""
import random
from dataclasses import dataclass

from datasets import load_dataset


@dataclass
class NQSwapExample:
    example_id: int
    question: str
    org_context: str
    org_answer: list[str]
    sub_context: str
    sub_answer: list[str]


def load_nq_swap_examples(n: int | None = None, seed: int = 42) -> list[NQSwapExample]:
    """Load NQ-Swap dev split, optionally subsampled to `n` examples."""
    ds = load_dataset("pminervini/NQ-Swap")["dev"]
    indices = list(range(len(ds)))
    if n is not None and n < len(indices):
        random.Random(seed).shuffle(indices)
        indices = indices[:n]
    examples = []
    for idx in indices:
        row = ds[idx]
        examples.append(
            NQSwapExample(
                example_id=idx,
                question=row["question"],
                org_context=row["org_context"],
                org_answer=row["org_answer"],
                sub_context=row["sub_context"],
                sub_answer=row["sub_answer"],
            )
        )
    return examples
