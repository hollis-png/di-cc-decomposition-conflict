"""Stage 40 (SelfCheckGPT baseline, evaluation): score an
NLI-based SelfCheckGPT-style inconsistency detector on the n=64 labeled
dataset, using the N=10 temperature-sampled resamples from stage 39, and
compute its AUC against the same DI-CC/legit/DI-UE labels and the same
permutation-test/bootstrap-CI methodology as the H0 probe (Section 4.2), so
the two detectors are directly comparable on identical data.

Per-claim inconsistency score (Manakul et al. 2023's NLI variant): for each
original (greedily-decoded) subclaim, check whether each of the N=10
temperature-sampled resamples' full text entails that claim. The
inconsistency score is the fraction of the N resamples that do NOT entail
it (contradiction or neutral) -- SelfCheckGPT's premise is that a claim the
model is confident about will recur consistently across resamples, while an
injected/hallucinated one will not.

Needs GPU (NLI model only).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from phase0.decomposition.entailment import load_nli_model, nli_label
from phase0.transfer_eval.evaluate import auc_bootstrap_ci, auc_with_permutation_test
from phase0.utils.keep_awake import prevent_sleep

prevent_sleep()

RESAMPLES_PATH = config.DATA_PROCESSED / "h3_selfcheckgpt_resamples.jsonl"
OUT_PATH = config.OUTPUTS_REPORTS / "h3_selfcheckgpt_baseline.json"


def load_resamples_by_key():
    by_key = {}
    with open(RESAMPLES_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["person"], rec["condition"])
            by_key.setdefault(key, []).append(rec["raw_response"])
    return by_key


def load_labeled_spans():
    spans = []
    baseline_conditions = {"perturbed_baseline", "vague_baseline"}
    with open(config.H3_LABELED_JSONL, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["label"] != "excluded" and rec["condition"] in baseline_conditions:
                spans.append(rec)
    return spans


def main():
    resamples_by_key = load_resamples_by_key()
    spans = load_labeled_spans()
    print(f"Loaded {len(spans)} labeled spans (n=64 dataset source) and "
          f"{len(resamples_by_key)} (person, condition) resample groups")

    nli_tokenizer, nli_model, id2label = load_nli_model()
    print(f"NLI model loaded, labels: {id2label}")

    scores_by_label = {"DI-CC": [], "legit": [], "DI-UE": []}
    n_skipped_no_resamples = 0

    for i, span in enumerate(spans):
        key = (span["person"], span["condition"])
        resamples = resamples_by_key.get(key)
        if not resamples:
            n_skipped_no_resamples += 1
            continue

        claim_text = span["claim_text"]
        n_not_entailed = 0
        for resample_text in resamples:
            label = nli_label(nli_tokenizer, nli_model, id2label, premise=resample_text, hypothesis=claim_text)
            if label != "entailment":
                n_not_entailed += 1
        inconsistency_score = n_not_entailed / len(resamples)

        if span["label"] in scores_by_label:
            scores_by_label[span["label"]].append(inconsistency_score)

        if (i + 1) % 100 == 0:
            print(f"  scored {i + 1}/{len(spans)}")

    if n_skipped_no_resamples:
        print(f"WARNING: {n_skipped_no_resamples} spans skipped, no matching resamples found "
              f"(should be 0 if stage 39 completed fully)")

    for label, scores in scores_by_label.items():
        print(f"  {label}: n={len(scores)}, mean inconsistency score={sum(scores) / len(scores) if scores else float('nan'):.3f}")

    report = {
        "method": "SelfCheckGPT-NLI (Manakul et al. 2023), N=10 temperature-1.0 resamples",
        "n_resamples_per_claim": 10,
        "group_sizes": {k: len(v) for k, v in scores_by_label.items()},
    }
    if len(scores_by_label["DI-CC"]) >= 2 and len(scores_by_label["legit"]) >= 2:
        report["DI-CC_vs_legit_permutation"] = auc_with_permutation_test(
            scores_by_label["DI-CC"], scores_by_label["legit"])
        report["DI-CC_vs_legit_bootstrap_ci"] = auc_bootstrap_ci(
            scores_by_label["DI-CC"], scores_by_label["legit"])
    if len(scores_by_label["DI-CC"]) >= 2 and len(scores_by_label["DI-UE"]) >= 2:
        report["DI-CC_vs_DI-UE_permutation"] = auc_with_permutation_test(
            scores_by_label["DI-CC"], scores_by_label["DI-UE"])
        report["DI-CC_vs_DI-UE_bootstrap_ci"] = auc_bootstrap_ci(
            scores_by_label["DI-CC"], scores_by_label["DI-UE"])

    print("\n=== SelfCheckGPT-NLI baseline results (n=64 dataset) ===")
    for key in ("DI-CC_vs_legit_permutation", "DI-CC_vs_DI-UE_permutation"):
        if key in report:
            r = report[key]
            verdict = "SIGNIFICANT" if r["observed_auc"] > 0.5 and r["permutation_p_value"] < 0.05 else "not significant"
            print(f"  {key}: AUC={r['observed_auc']:.3f}, p={r['permutation_p_value']:.4f} -> {verdict}")

    print("\nFor comparison, the H0 mechanistic probe (Section 4.2, same dataset): "
          "DI-CC vs legit AUC=0.881, DI-CC vs DI-UE AUC=0.863")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {OUT_PATH}")


if __name__ == "__main__":
    main()
