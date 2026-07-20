"""Re-select the probe layer for mistral7b using the variance-ratio
criterion (sweep_layers_by_variance_ratio) instead of held-out accuracy,
then re-evaluate H0 transfer at that layer. Checks whether the accuracy-
selected layer 18's null result (AUC=0.397, CI=[0.34,0.45]) is specific to
that layer-selection method, mirroring the earlier Qwen scale-ablation
finding that accuracy-selected layers transfer unevenly across model scale.
No GPU needed -- reuses NQ-Swap and decomposition activations already on disk.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from phase0.probe.layer_sweep import sweep_layers_by_variance_ratio
from phase0.probe.model import save_probe, save_sweep_report_variance_ratio
from phase0.transfer_eval.evaluate import auc_bootstrap_ci, auc_with_permutation_test


def main():
    nq_data = np.load(config.NQ_SWAP_ACTIVATIONS)
    activations, labels = nq_data["activations"], nq_data["labels"]

    results = sweep_layers_by_variance_ratio(activations, labels)
    save_sweep_report_variance_ratio(results)

    print("Variance-ratio layer sweep (best first):")
    for r in results:
        print(f"  layer {r['layer']:2d}: variance_ratio={r['variance_ratio']:.4f}  "
              f"train_acc={r['train_acc']:.3f}  test_acc={r['test_acc']:.3f}")

    best = results[0]
    save_probe(best["layer"], best["probe"])
    print(f"\nVariance-ratio-best layer: {best['layer']} "
          f"(variance_ratio={best['variance_ratio']:.4f}, test_acc={best['test_acc']:.3f})")

    decomp_data = np.load(config.DECOMP_ACTIVATIONS, allow_pickle=True)
    decomp_activations, decomp_labels = decomp_data["activations"], decomp_data["labels"]
    X = decomp_activations[:, best["layer"], :]
    scores = best["probe"].predict_proba(X)[:, 1]
    groups = {label: scores[decomp_labels == label] for label in ["DI-CC", "legit", "DI-UE"]}

    print(f"\nGroup probe-score stats at layer {best['layer']}:")
    for k, v in groups.items():
        print(f"  {k}: n={len(v)}, mean={v.mean():.4f}, std={v.std():.4f}")

    result = {"layer": best["layer"], "variance_ratio": best["variance_ratio"],
              "group_sizes": {k: int(len(v)) for k, v in groups.items()}}

    for pair, (pos_key, neg_key) in [("DI-CC_vs_legit", ("DI-CC", "legit")), ("DI-CC_vs_DI-UE", ("DI-CC", "DI-UE"))]:
        ci = auc_bootstrap_ci(groups[pos_key], groups[neg_key])
        perm = auc_with_permutation_test(groups[pos_key], groups[neg_key])
        result[pair] = {**ci, "permutation_p_value": perm["permutation_p_value"]}
        print(f"\n{pair}:")
        print(f"  observed AUC={ci['observed_auc']:.4f}, 95% CI=[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        print(f"  permutation p={perm['permutation_p_value']:.4f}")

    out_path = config.OUTPUTS_REPORTS / "transfer_eval_varratio_mistral7b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
