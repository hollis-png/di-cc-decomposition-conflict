"""Stage 1: pull a subsample of NQ-Swap and write it to data/processed/nq_swap.jsonl.

No GPU needed. See N_SAMPLES below.
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from phase0.data.nq_swap import load_nq_swap_examples

N_SAMPLES = 1500  # dev split has 4746 rows total; step 2 filters further by
                   # closed-book agreement, so oversample here.


def main():
    examples = load_nq_swap_examples(n=N_SAMPLES, seed=config.RANDOM_SEED)
    with open(config.NQ_SWAP_JSONL, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} examples to {config.NQ_SWAP_JSONL}")


if __name__ == "__main__":
    main()
