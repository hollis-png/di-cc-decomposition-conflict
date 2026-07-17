"""Stage 7: the H0 decision point. Apply the frozen NQ-Swap probe (never
trained on any decomposition data) to the three labeled decomposition
groups, and test whether it fires more on DI-CC than on legit / DI-UE
spans. No GPU needed (activations already extracted).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from phase0.probe.model import load_probe
from phase0.transfer_eval.evaluate import auc_with_permutation_test


def main():
    with open(config.PROBE_SWEEP_REPORT, encoding="utf-8") as f:
        sweep = json.load(f)
    best_layer = sweep[0]["layer"]
    print(f"Using frozen probe from layer {best_layer} (NQ-Swap test_acc={sweep[0]['test_acc']:.3f})")

    probe = load_probe(best_layer)

    data = np.load(config.DECOMP_ACTIVATIONS, allow_pickle=True)
    activations, labels = data["activations"], data["labels"]
    X = activations[:, best_layer, :]
    scores = probe.predict_proba(X)[:, 1]  # P(conflict)

    groups = {label: scores[labels == label] for label in ["DI-CC", "legit", "DI-UE"]}
    for label, s in groups.items():
        print(f"  {label}: n={len(s)}, mean probe score={s.mean() if len(s) else float('nan'):.3f}")

    report = {"best_layer": best_layer, "group_sizes": {k: len(v) for k, v in groups.items()}}

    if len(groups["DI-CC"]) >= 2 and len(groups["legit"]) >= 2:
        report["DI-CC_vs_legit"] = auc_with_permutation_test(groups["DI-CC"], groups["legit"])
    if len(groups["DI-CC"]) >= 2 and len(groups["DI-UE"]) >= 2:
        report["DI-CC_vs_DI-UE"] = auc_with_permutation_test(groups["DI-CC"], groups["DI-UE"])

    print("\nH0 test results:")
    for key in ("DI-CC_vs_legit", "DI-CC_vs_DI-UE"):
        if key in report:
            r = report[key]
            verdict = "SIGNIFICANT" if r["observed_auc"] > 0.5 and r["permutation_p_value"] < 0.05 else "not significant"
            print(f"  {key}: AUC={r['observed_auc']:.3f}, p={r['permutation_p_value']:.4f} -> {verdict}")

    with open(config.TRANSFER_EVAL_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {config.TRANSFER_EVAL_REPORT}")


if __name__ == "__main__":
    main()
