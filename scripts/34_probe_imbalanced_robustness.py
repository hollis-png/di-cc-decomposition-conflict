"""Stage 34 (class-imbalance robustness check): H0's frozen
probe is trained on NQ-Swap with a perfectly balanced 413/413 conflict
label split, but DI-CC occurs at only ~3.2% of atomic claims in real
decomposition (Phase 0 Sec 4.1). This reruns probe training after
downsampling NQ-Swap's conflict class to match that natural imbalance
(subject to a floor for training stability), keeping everything else
identical to the variance-ratio pipeline (stage 30/32), and re-evaluates
H0 transfer AUC on the n=64 pool to check whether the balanced-training
result was an artifact of that choice. Reuses existing activations; no
GPU/decomposer needed.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from phase0.probe.layer_sweep import sweep_layers_by_variance_ratio
from phase0.transfer_eval.evaluate import auc_bootstrap_ci, auc_with_permutation_test

TARGET_IMBALANCE = 0.032  # DI-CC's real rate among atomic claims, Phase 0 Sec 4.1
MIN_POS = 30  # floor so logistic regression + train/test split remain stable


def downsample_to_imbalance(activations, labels, target_rate, seed=config.RANDOM_SEED):
    rng = np.random.RandomState(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    # NQ-Swap's balanced 413/413 split has far more positives than a 3.2%
    # imbalance can use alongside a fixed negative pool (413 negatives would
    # need ~12.5k positives to hit 3.2% the other way around) -- so fix all
    # negatives and downsample positives instead, the direction that is
    # actually achievable given the available data.
    n_neg = len(neg_idx)
    n_pos_exact = int(round(n_neg * target_rate / (1 - target_rate)))
    # 14 exact positives is too few for a stable stratified train/test split
    # + logistic regression fit; MIN_POS is a floor on training stability,
    # not on how closely we hit the target rate. Using MIN_POS positives
    # against all 413 negatives gives an achieved rate (~6.8%) still far
    # closer to the ~3.2% real-world rate than the original 50/50 split,
    # while keeping the fit itself trustworthy -- this tradeoff (and the
    # gap between the requested and achieved rate) is reported explicitly
    # in main()'s output rather than silently substituted.
    n_pos_target = max(n_pos_exact, MIN_POS)
    n_pos_target = min(n_pos_target, len(pos_idx))
    pos_sample = rng.choice(pos_idx, size=n_pos_target, replace=False)

    keep = np.concatenate([pos_sample, neg_idx])
    rng.shuffle(keep)
    return activations[keep], labels[keep]


def main():
    data = np.load(config.NQ_SWAP_ACTIVATIONS)
    activations, labels = data["activations"], data["labels"]
    print(f"Original NQ-Swap: {len(labels)} total, {labels.sum()} conflict / {len(labels) - labels.sum()} no_conflict "
          f"({100 * labels.mean():.1f}% positive)")

    X_imb, y_imb = downsample_to_imbalance(activations, labels, TARGET_IMBALANCE)
    print(f"Downsampled to: {len(y_imb)} total, {int(y_imb.sum())} conflict / {int(len(y_imb) - y_imb.sum())} no_conflict "
          f"({100 * y_imb.mean():.2f}% positive, target {100 * TARGET_IMBALANCE:.1f}%)")

    results = sweep_layers_by_variance_ratio(X_imb, y_imb)
    best = results[0]
    print(f"\nBest layer under imbalanced training: {best['layer']} "
          f"(variance_ratio={best['variance_ratio']:.3f}, test_acc={best['test_acc']:.3f})")
    # Probe intentionally not persisted via save_probe(): that helper writes
    # to the shared config.PROBE_CHECKPOINT path keyed only by layer number,
    # which would silently overwrite the validated primary-pipeline probe if
    # this run selects the same layer. This is a one-off robustness check,
    # not a replacement for the primary probe, so it stays in-memory only.

    # Re-evaluate H0 transfer on the n=64 pool with this imbalance-trained probe.
    h3_data = np.load(config.H3_BASELINE_ACTIVATIONS, allow_pickle=True)
    h3_activations, h3_labels = h3_data["activations"], h3_data["labels"]
    X_transfer = h3_activations[:, best["layer"], :]
    scores = best["probe"].predict_proba(X_transfer)[:, 1]

    groups = {label: scores[h3_labels == label] for label in ["DI-CC", "legit", "DI-UE"]}
    for label, s in groups.items():
        print(f"  {label}: n={len(s)}, mean probe score={s.mean() if len(s) else float('nan'):.3f}")

    report = {
        "target_imbalance_rate": TARGET_IMBALANCE,
        "actual_imbalance_rate": float(y_imb.mean()),
        "n_train_pos": int(y_imb.sum()),
        "n_train_neg": int(len(y_imb) - y_imb.sum()),
        "selected_layer": best["layer"],
        "variance_ratio": best["variance_ratio"],
        "nq_swap_test_acc": best["test_acc"],
        "group_sizes": {k: int(len(v)) for k, v in groups.items()},
    }
    if len(groups["DI-CC"]) >= 2 and len(groups["legit"]) >= 2:
        report["DI-CC_vs_legit_permutation"] = auc_with_permutation_test(groups["DI-CC"], groups["legit"])
        report["DI-CC_vs_legit_bootstrap_ci"] = auc_bootstrap_ci(groups["DI-CC"], groups["legit"])
    if len(groups["DI-CC"]) >= 2 and len(groups["DI-UE"]) >= 2:
        report["DI-CC_vs_DI-UE_permutation"] = auc_with_permutation_test(groups["DI-CC"], groups["DI-UE"])
        report["DI-CC_vs_DI-UE_bootstrap_ci"] = auc_bootstrap_ci(groups["DI-CC"], groups["DI-UE"])

    print("\nH0 (imbalance-trained probe, n=64 transfer pool) results:")
    for key in ("DI-CC_vs_legit_permutation", "DI-CC_vs_DI-UE_permutation"):
        if key in report:
            r = report[key]
            verdict = "SIGNIFICANT" if r["observed_auc"] > 0.5 and r["permutation_p_value"] < 0.05 else "not significant"
            print(f"  {key}: AUC={r['observed_auc']:.3f}, p={r['permutation_p_value']:.4f} -> {verdict}")

    out_path = config.OUTPUTS_REPORTS / "imbalanced_probe_robustness.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
