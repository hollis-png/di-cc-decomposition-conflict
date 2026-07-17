"""Stage 2: for every NQ-Swap example, (a) query the decomposer model closed-
book to find its parametric answer a_m, (b) keep only examples where a_m
agrees with org_answer (so conflict/no-conflict is cleanly defined per the
design doc's a_c != a_m / a_c == a_m criterion), (c) extract last-token
hidden states for both the sub_context (conflict) and org_context
(no-conflict) variants.

Needs GPU. Run with --limit N for a quick smoke test before the full pass.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from phase0.activations.extract import (
    generate_short_answer,
    hidden_states_at_last_token,
    load_decomposer_model,
)
from phase0.utils.answer_match import answer_matches

def closed_book_messages(question: str) -> list[dict]:
    return [{
        "role": "user",
        "content": f"Answer the question with only the short answer itself, no explanation.\n\nQuestion: {question}",
    }]


def context_qa_messages(context: str, question: str) -> list[dict]:
    return [{
        "role": "user",
        "content": (
            f"Answer the question using only the given context. "
            f"Respond with only the short answer itself, no explanation.\n\n"
            f"Context: {context}\n\nQuestion: {question}"
        ),
    }]


def load_nq_swap_rows(limit: int | None):
    rows = []
    with open(config.NQ_SWAP_JSONL, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process first N examples (smoke test)")
    args = parser.parse_args()

    rows = load_nq_swap_rows(args.limit)
    print(f"Loaded {len(rows)} NQ-Swap examples")

    tokenizer, model = load_decomposer_model()
    print("Model loaded")

    activations = []
    labels = []  # 1 = conflict, 0 = no_conflict
    meta_records = []

    n_kept_examples = 0
    t0 = time.time()
    for i, row in enumerate(rows):
        closed_book_answer = generate_short_answer(tokenizer, model, closed_book_messages(row["question"]))
        is_correct_parametric = answer_matches(closed_book_answer, row["org_answer"])
        answers_differ = not answer_matches(row["sub_answer"][0], row["org_answer"])

        if not (is_correct_parametric and answers_differ):
            continue
        n_kept_examples += 1

        # conflict instance: model fed the substituted (wrong) context
        conflict_msgs = context_qa_messages(row["sub_context"], row["question"])
        conflict_act = hidden_states_at_last_token(tokenizer, model, conflict_msgs)
        activations.append(conflict_act)
        labels.append(1)
        meta_records.append(
            {
                "row_id": len(meta_records),
                "example_id": row["example_id"],
                "question": row["question"],
                "context_variant": "sub",
                "label": "conflict",
                "closed_book_answer": closed_book_answer,
            }
        )

        # no-conflict instance: model fed the original (correct) context
        no_conflict_msgs = context_qa_messages(row["org_context"], row["question"])
        no_conflict_act = hidden_states_at_last_token(tokenizer, model, no_conflict_msgs)
        activations.append(no_conflict_act)
        labels.append(0)
        meta_records.append(
            {
                "row_id": len(meta_records),
                "example_id": row["example_id"],
                "question": row["question"],
                "context_variant": "org",
                "label": "no_conflict",
                "closed_book_answer": closed_book_answer,
            }
        )

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  processed {i + 1}/{len(rows)} raw examples, kept {n_kept_examples}, {elapsed:.1f}s elapsed")

    print(f"Kept {n_kept_examples}/{len(rows)} examples with correct closed-book knowledge "
          f"({n_kept_examples / len(rows):.1%}), producing {len(labels)} probe rows")

    activations_arr = np.stack(activations, axis=0)  # (N, num_hidden_states, hidden_size)
    labels_arr = np.array(labels, dtype=np.int64)
    np.savez_compressed(config.NQ_SWAP_ACTIVATIONS, activations=activations_arr, labels=labels_arr)
    print(f"Saved activations {activations_arr.shape} to {config.NQ_SWAP_ACTIVATIONS}")

    meta_path = config.NQ_SWAP_ACTIVATIONS.with_suffix(".meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for rec in meta_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
