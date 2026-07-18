"""Stage 32 (H0 sample-size expansion x layer-selection reanalysis):
apply the variance-ratio-selected frozen probe (stage 30, 7B scale) to the
larger H3-baseline activation set (stage 28, n~64 DI-CC), combining the
two reanalyses that were so far only done separately -- n=64 used the
accuracy-selected layer (stage 29), and variance-ratio selection was only
applied at the original 40-entity pool (stage 31, per scale). No GPU
needed; reuses activations already on disk.
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
    with open(config.PROBE_SWEEP_REPORT_VARRATIO, encoding="utf-8") as f:
        sweep = json.load(f)
    best_layer = sweep[0]["layer"]
    print(f"Using variance-ratio-selected frozen probe from layer {best_layer} "
          f"(variance_ratio={sweep[0]['variance_ratio']:.3f}, NQ-Swap test_acc={sweep[0]['test_acc']:.3f})")

    probe = load_probe(best_layer)

    data = np.load(config.H3_BASELINE_ACTIVATIONS, allow_pickle=True)
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

    print("\nH0 (expanded-N, variance-ratio layer) test results:")
    for key in ("DI-CC_vs_legit", "DI-CC_vs_DI-UE"):
        if key in report:
            r = report[key]
            verdict = "SIGNIFICANT" if r["observed_auc"] > 0.5 and r["permutation_p_value"] < 0.05 else "not significant"
            print(f"  {key}: AUC={r['observed_auc']:.3f}, p={r['permutation_p_value']:.4f} -> {verdict}")

    out_path = config.OUTPUTS_REPORTS / "h3_baseline_transfer_eval_varratio.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
