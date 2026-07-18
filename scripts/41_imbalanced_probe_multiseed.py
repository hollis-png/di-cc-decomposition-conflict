"""Stage 41 (class-imbalance robustness, multi-seed follow-up): stage 38 found
conflict-class recall collapsing from 0.806 (balanced
training) to 0.250 (imbalanced training, ~6.8% positive) on a SINGLE random seed,
with only 8 true positives in the imbalanced test split -- too few for that one
number to be trustworthy on its own. This reruns the full stage-38 pipeline
(downsample -> variance-ratio layer selection -> train/test split -> fit -> score)
across N=50 different seeds, so the recall estimate can be reported as a
distribution (mean +/- std, range) rather than a single point, to establish
whether the recall collapse is a stable finding or single-split noise.

Each seed drives the downsampling, the layer reselection, and the train/test
split together (not independently reseeded), so seed i reproduces "what if the
whole stage-34/38 pipeline had been run with a different random seed" faithfully.
Pure CPU, reuses existing NQ-Swap activations, no GPU/decomposer needed.
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
N_SEEDS = 50
BALANCED_LAYER = 20  # fixed: the variance-ratio-selected layer on the full balanced set (stage 30/32)


def downsample_to_imbalance(activations, labels, target_rate, seed):
    """Identical logic to stage 34/38's downsample_to_imbalance()."""
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


def fit_and_score(X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report_dict = classification_report(y_test, y_pred, target_names=["no_conflict", "conflict"],
                                         output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return {
        "n_test": int(len(y_test)),
        "n_test_positive": int(y_test.sum()),
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
    print(f"Running {N_SEEDS} seeds for both balanced and imbalanced conditions...")

    balanced_results = []
    imbalanced_results = []
    imbalanced_layers = []

    for seed in range(N_SEEDS):
        # Balanced condition: full 826-example set, fixed layer 20, only the
        # train/test split varies by seed (no downsampling to vary).
        X_bal = activations[:, BALANCED_LAYER, :]
        balanced_results.append(fit_and_score(X_bal, labels, seed))

        # Imbalanced condition: downsampling AND layer reselection AND split
        # all driven by this seed.
        X_imb_full, y_imb = downsample_to_imbalance(activations, labels, TARGET_IMBALANCE, seed)
        ratios = {layer: variance_ratio(X_imb_full, y_imb, layer) for layer in config.PROBE_LAYER_CANDIDATES}
        imb_layer = max(ratios, key=ratios.get)
        imbalanced_layers.append(imb_layer)
        X_imb = X_imb_full[:, imb_layer, :]
        imbalanced_results.append(fit_and_score(X_imb, y_imb, seed))

        if (seed + 1) % 10 == 0:
            print(f"  [{seed + 1}/{N_SEEDS}] done")

    def summarize(results, label):
        recalls = np.array([r["conflict_recall"] for r in results])
        precisions = np.array([r["conflict_precision"] for r in results])
        f1s = np.array([r["conflict_f1"] for r in results])
        accs = np.array([r["overall_accuracy"] for r in results])
        n_pos_list = [r["n_test_positive"] for r in results]
        print(f"\n=== {label} (N={len(results)} seeds) ===")
        print(f"  test-set positive count per seed: min={min(n_pos_list)}, max={max(n_pos_list)}, "
              f"mean={np.mean(n_pos_list):.1f}")
        print(f"  conflict recall:    mean={recalls.mean():.3f}  std={recalls.std():.3f}  "
              f"min={recalls.min():.3f}  max={recalls.max():.3f}")
        print(f"  conflict precision: mean={precisions.mean():.3f}  std={precisions.std():.3f}")
        print(f"  conflict F1:        mean={f1s.mean():.3f}  std={f1s.std():.3f}")
        print(f"  overall accuracy:   mean={accs.mean():.3f}  std={accs.std():.3f}")
        return {
            "recall_mean": float(recalls.mean()), "recall_std": float(recalls.std()),
            "recall_min": float(recalls.min()), "recall_max": float(recalls.max()),
            "recall_values": recalls.tolist(),
            "precision_mean": float(precisions.mean()), "precision_std": float(precisions.std()),
            "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std()),
            "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std()),
            "n_test_positive_per_seed": n_pos_list,
        }

    balanced_summary = summarize(balanced_results, "Balanced (413/413)")
    imbalanced_summary = summarize(imbalanced_results, "Imbalanced (~6.8% positive)")

    from collections import Counter
    layer_counts = Counter(imbalanced_layers)
    print(f"\nImbalanced-condition layer selection across {N_SEEDS} seeds: {dict(sorted(layer_counts.items()))}")

    recall_gap = balanced_summary["recall_mean"] - imbalanced_summary["recall_mean"]
    print(f"\nMean conflict recall gap (balanced - imbalanced): {recall_gap:+.3f}")
    print(f"Imbalanced recall range across seeds: [{imbalanced_summary['recall_min']:.3f}, "
          f"{imbalanced_summary['recall_max']:.3f}] (std={imbalanced_summary['recall_std']:.3f})")
    if imbalanced_summary["recall_max"] < balanced_summary["recall_min"]:
        print("STABLE FINDING: imbalanced recall never reaches even the LOWEST balanced-condition "
              "recall across any of the 50 seeds -- the recall collapse is not single-split noise.")
    elif imbalanced_summary["recall_mean"] < balanced_summary["recall_mean"] - 2 * imbalanced_summary["recall_std"]:
        print("LIKELY STABLE: imbalanced mean recall is more than 2 std below balanced mean, "
              "though some overlap in range exists.")
    else:
        print("UNSTABLE / INCONCLUSIVE: imbalanced recall varies too much across seeds to "
              "confidently distinguish from balanced-condition performance.")

    out = {
        "n_seeds": N_SEEDS,
        "target_imbalance_rate": TARGET_IMBALANCE,
        "balanced": balanced_summary,
        "imbalanced": imbalanced_summary,
        "imbalanced_layer_selection_counts": {str(k): v for k, v in layer_counts.items()},
        "recall_gap_balanced_minus_imbalanced": recall_gap,
    }
    out_path = config.OUTPUTS_REPORTS / "imbalanced_probe_multiseed.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
