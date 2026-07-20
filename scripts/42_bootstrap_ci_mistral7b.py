"""Bootstrap CI for the Mistral-7B H0 transfer-eval result, to check whether
the null AUC (0.397) reflects a true null/negative effect or just low power
from the small DI-CC group (n=74). Mirrors scripts/33_bootstrap_ci_all_h0_results.py
but points at the mistral7b-scale activations and probe instead of the
original Qwen n14/n64 pools.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from phase0.probe.model import load_probe
from phase0.transfer_eval.evaluate import auc_bootstrap_ci, auc_with_permutation_test


def main():
    data = np.load(config.DECOMP_ACTIVATIONS, allow_pickle=True)
    activations, labels = data["activations"], data["labels"]

    with open(config.PROBE_SWEEP_REPORT, encoding="utf-8") as f:
        sweep = json.load(f)
    layer = sweep[0]["layer"]

    probe = load_probe(layer)
    X = activations[:, layer, :]
    scores = probe.predict_proba(X)[:, 1]
    groups = {label: scores[labels == label] for label in ["DI-CC", "legit", "DI-UE"]}

    print(f"Layer {layer}")
    for k, v in groups.items():
        print(f"  {k}: n={len(v)}, mean={v.mean():.4f}, std={v.std():.4f}")

    result = {"layer": layer, "group_sizes": {k: int(len(v)) for k, v in groups.items()}}

    for pair, (pos_key, neg_key) in [("DI-CC_vs_legit", ("DI-CC", "legit")), ("DI-CC_vs_DI-UE", ("DI-CC", "DI-UE"))]:
        ci = auc_bootstrap_ci(groups[pos_key], groups[neg_key])
        perm = auc_with_permutation_test(groups[pos_key], groups[neg_key])
        result[pair] = {**ci, "permutation_p_value": perm["permutation_p_value"]}
        print(f"\n{pair}:")
        print(f"  observed AUC={ci['observed_auc']:.4f}, 95% CI=[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        print(f"  permutation p={perm['permutation_p_value']:.4f}")

    out_path = config.OUTPUTS_REPORTS / f"bootstrap_ci_{config.PHASE0_SCALE}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
