"""Stage 39 (SelfCheckGPT baseline, resampling): the paper
had no empirical baseline comparison against existing reference-free
hallucination detectors, most notably SelfCheckGPT-style self-consistency
sampling (Manakul et al., 2023). This generates the resampling data needed
for an NLI-based SelfCheckGPT-style detector on the SAME n=64 (H3 baseline)
DI-CC/legit/DI-UE-labeled dataset H0 was evaluated on, so the two detectors
can be compared on identical data with identical AUC methodology.

Method: for each (person, condition) in the perturbed_baseline/vague_baseline
H3 data (i.e. the exact source of the n=64 labels), keep the ORIGINAL bio
text fixed (SelfCheckGPT checks self-consistency of what the model says
about a fixed input, not variance in the input itself) and re-run
decomposition N=10 times with temperature sampling instead of the original
greedy decoding. Reuses the exact same decomposition prompts as the primary
pipeline (DECOMPOSE_PROMPT / DECOMPOSE_PROMPT_CORRECT by bio_variant).

Resumable (append+flush+skip-done) + sleep prevention + UTF-8 stdout,
following the same pattern as prior long-running stage scripts (25, 35).
Needs GPU.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from phase0.activations.extract import _build_chat_inputs, load_decomposer_model
from phase0.decomposition.decompose import DECOMPOSE_PROMPT, DECOMPOSE_PROMPT_CORRECT, _parse_numbered_subclaims
from phase0.utils.keep_awake import prevent_sleep

prevent_sleep()

N_SAMPLES = 10
TEMPERATURE = 1.0
_PROMPT_BY_VARIANT = {"vague": DECOMPOSE_PROMPT, "perturbed": DECOMPOSE_PROMPT_CORRECT}
_BASELINE_CONDITIONS = {"perturbed_baseline", "vague_baseline"}
OUT_PATH = config.DATA_PROCESSED / "h3_selfcheckgpt_resamples.jsonl"


@torch.no_grad()
def generate_sampled(tokenizer, model, messages, max_new_tokens=400, temperature=TEMPERATURE):
    """Same generation call as decompose_into_subclaims, but with sampling
    enabled instead of greedy decoding -- the one change SelfCheckGPT-style
    resampling requires. Not touching generate_short_answer() itself since
    every other stage in the pipeline depends on its greedy-decoding
    behavior staying fixed.
    """
    inputs = _build_chat_inputs(tokenizer, messages)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_done_keys() -> set[tuple[str, str, int]]:
    done = set()
    if not OUT_PATH.exists():
        return done
    with open(OUT_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            done.add((rec["person"], rec["condition"], rec["sample_idx"]))
    return done


def main():
    decompositions = {}
    with open(config.H3_DECOMPOSITIONS_JSONL, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["person"], rec["condition"])
            decompositions[key] = {"bio": rec["bio"], "bio_variant": rec["bio_variant"]}

    # Only the (person, condition) pairs that actually appear in the n=64
    # labeled dataset need resampling -- restricting to non-excluded spans'
    # source records, deduped, keeps this from wastefully resampling entities
    # whose spans were all excluded (not part of the labeled comparison).
    needed_keys = set()
    with open(config.H3_LABELED_JSONL, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["label"] != "excluded" and rec["condition"] in _BASELINE_CONDITIONS:
                needed_keys.add((rec["person"], rec["condition"]))
    print(f"{len(needed_keys)} (person, condition) pairs need resampling "
          f"(source of the n=64 labeled dataset)")

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process first N (person, condition) pairs (smoke test)")
    args = parser.parse_args()

    sorted_keys = sorted(needed_keys)
    if args.limit is not None:
        sorted_keys = sorted_keys[: args.limit]
        print(f"--limit={args.limit}: restricting to {len(sorted_keys)} pairs")

    work = []
    for key in sorted_keys:
        for sample_idx in range(N_SAMPLES):
            work.append((key, sample_idx))

    done_keys = load_done_keys()
    remaining = [w for w in work if (w[0][0], w[0][1], w[1]) not in done_keys]
    print(f"{len(work)} total (pair, sample) targets; {len(done_keys)} already done; "
          f"{len(remaining)} remaining")

    if not remaining:
        print("Nothing left to do.")
        return

    tokenizer, model = load_decomposer_model()
    print("Model loaded")

    t0 = time.time()
    with open(OUT_PATH, "a", encoding="utf-8") as out_f:
        for i, ((person, condition), sample_idx) in enumerate(remaining):
            entry = decompositions[(person, condition)]
            prompt_template = _PROMPT_BY_VARIANT[entry["bio_variant"]]
            messages = [{"role": "user", "content": prompt_template.format(bio=entry["bio"])}]
            raw_response = generate_sampled(tokenizer, model, messages)
            subclaims = _parse_numbered_subclaims(raw_response)

            out_rec = {
                "person": person,
                "condition": condition,
                "bio_variant": entry["bio_variant"],
                "sample_idx": sample_idx,
                "raw_response": raw_response,
                "subclaims": [{"text": c["text"]} for c in subclaims],
            }
            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            out_f.flush()

            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                rate = elapsed / (i + 1)
                remaining_est_min = rate * (len(remaining) - i - 1) / 60
                print(f"  [{i + 1}/{len(remaining)}] {person} ({condition}, sample {sample_idx}): "
                      f"{len(subclaims)} subclaims ({elapsed:.1f}s elapsed, "
                      f"~{remaining_est_min:.1f}min remaining)")

    print(f"Appended resamples to {OUT_PATH}")


if __name__ == "__main__":
    main()
