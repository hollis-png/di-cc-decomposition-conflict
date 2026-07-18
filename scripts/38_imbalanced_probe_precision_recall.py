"""Stage 38 (class-imbalance robustness, precision/recall follow-up): stage 34's
imbalanced-training robustness check found AUC and NQ-Swap validation
accuracy both rose under the 6.77%-positive downsampled training set (0.881
-> 0.930 AUC; 0.802 -> 0.928 accuracy). Rising accuracy under class
imbalance is also consistent with the classifier learning a majority-class
shortcut rather than a cleaner conflict direction -- accuracy alone can't
distinguish the two. This computes positive-class (conflict) precision/
recall/F1 on the same held-out split for both the balanced and imbalanced
training conditions, side by side, to check whether recall on the minority
(conflict) class holds up or collapses under imbalance.

Reuses stage 34's exact downsampling and train_test_split logic (same seed)
so the compared splits are the ones stage 34 actually evaluated on -- not a
fresh resample. Pure CPU, reuses existing NQ-Swap activations, no GPU/
decomposer needed.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from phase0.probe.layer_sweep import variance_ratio

TARGET_IMBALANCE = 0.032
MIN_POS = 30


def downsample_to_imbalance(activations, labels, target_rate, seed=config.RANDOM_SEED):
    """Identical logic to stage 34's downsample_to_imbalance()."""
    rng = np.random.RandomState(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_neg = len(neg_idx)
    n_pos_exact = int(round(n_neg * target_rate / (1 - target_rate)))
    n_pos_target = max(n_pos_exact, MIN_POS)
    n_pos_target = min(n_pos_target, len(pos_idx))
    pos_sample = rng.choice(pos_idx, size=n_pos_target, replace=False)
    keep = np.concatenate([pos_sample, neg_idx])
    rng.shuffle(keep)
    return activations[keep], labels[keep]


def fit_and_report(activations, labels, layer, seed=config.RANDOM_SEED, label_name=""):
    X = activations[:, layer, :]
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.25, random_state=seed, stratify=labels
    )
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report_dict = classification_report(y_test, y_pred, target_names=["no_conflict", "conflict"],
                                         output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(f"\n=== {label_name} (layer {layer}, n_test={len(y_test)}, "
          f"test-set positive rate={y_test.mean():.3f}) ===")
    print(classification_report(y_test, y_pred, target_names=["no_conflict", "conflict"], zero_division=0))
    print(f"Confusion matrix [rows=true, cols=pred], order [no_conflict, conflict]:\n{cm}")

    return {
        "layer": layer,
        "n_test": int(len(y_test)),
        "test_set_positive_rate": float(y_test.mean()),
        "conflict_precision": report_dict["conflict"]["precision"],
        "conflict_recall": report_dict["conflict"]["recall"],
        "conflict_f1": report_dict["conflict"]["f1-score"],
        "overall_accuracy": report_dict["accuracy"],
        "confusion_matrix": cm.tolist(),
    }


def main():
    data = np.load(config.NQ_SWAP_ACTIVATIONS)
    activations, labels = data["activations"], data["labels"]
    print(f"Original NQ-Swap: {len(labels)} total, {int(labels.sum())} conflict / "
          f"{int(len(labels) - labels.sum())} no_conflict ({100 * labels.mean():.1f}% positive)")

    # Balanced condition: same layer (20) selected by stage 30/32's variance-ratio
    # sweep on the full balanced set, so this is an apples-to-apples comparison at
    # the SAME layer stage 34 compared against, not a fresh layer search.
    balanced_layer = 20
    balanced_report = fit_and_report(activations, labels, balanced_layer, label_name="Balanced (413/413)")

    # Imbalanced condition: identical downsampling to stage 34.
    X_imb, y_imb = downsample_to_imbalance(activations, labels, TARGET_IMBALANCE)
    print(f"\nDownsampled to: {len(y_imb)} total, {int(y_imb.sum())} conflict / "
          f"{int(len(y_imb) - y_imb.sum())} no_conflict ({100 * y_imb.mean():.2f}% positive)")

    # Re-derive which layer stage 34 selected under this imbalance (by variance
    # ratio, same criterion), so the imbalanced report is evaluated at the same
    # layer stage 34 actually used -- not assumed to also be 20.
    ratios = {layer: variance_ratio(X_imb, y_imb, layer) for layer in config.PROBE_LAYER_CANDIDATES}
    imbalanced_layer = max(ratios, key=ratios.get)
    print(f"Variance-ratio-selected layer under imbalance: {imbalanced_layer} "
          f"(vs. balanced-condition layer {balanced_layer})")

    imbalanced_report = fit_and_report(X_imb, y_imb, imbalanced_layer, label_name="Imbalanced (6.77% positive)")

    print("\n=== Side-by-side: conflict-class precision/recall/F1 ===")
    print(f"{'Condition':<28}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Accuracy':>12}")
    for name, r in [("Balanced", balanced_report), ("Imbalanced", imbalanced_report)]:
        print(f"{name:<28}{r['conflict_precision']:>12.3f}{r['conflict_recall']:>12.3f}"
              f"{r['conflict_f1']:>12.3f}{r['overall_accuracy']:>12.3f}")

    recall_drop = balanced_report["conflict_recall"] - imbalanced_report["conflict_recall"]
    print(f"\nConflict-class recall change (balanced - imbalanced): {recall_drop:+.3f}")
    if imbalanced_report["conflict_recall"] < balanced_report["conflict_recall"] - 0.05:
        print("WARNING: conflict-class recall dropped meaningfully under imbalance -- "
              "the earlier 'accuracy increased' finding is consistent with a majority-class "
              "shortcut, not purely a cleaner conflict direction.")
    else:
        print("Conflict-class recall held up under imbalance -- the accuracy increase does "
              "NOT appear to be explained by a majority-class shortcut alone.")

    out = {
        "target_imbalance_rate": TARGET_IMBALANCE,
        "balanced": balanced_report,
        "imbalanced": imbalanced_report,
        "conflict_recall_change_balanced_minus_imbalanced": recall_drop,
    }
    out_path = config.OUTPUTS_REPORTS / "imbalanced_probe_precision_recall.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
