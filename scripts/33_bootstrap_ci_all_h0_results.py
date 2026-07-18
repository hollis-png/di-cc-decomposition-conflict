"""Stage 33 (uncertainty quantification for the H0 primary result): compute
bootstrap 95% CIs for every H0 transfer-eval AUC result produced so far
(original 40-entity pool at accuracy- and variance-ratio-selected layers,
and the n~64 H3-baseline pool at both layer-selection criteria). Reuses
activations + frozen probes already on disk; no GPU/decomposer needed.
Consolidates the n=64, variance-ratio-layer primary result alongside the
n=14 pilot for direct before/after comparison (see README.md).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from phase0.probe.model import load_probe
from phase0.transfer_eval.evaluate import auc_bootstrap_ci


def _groups_from_activations(npz_path, layer):
    data = np.load(npz_path, allow_pickle=True)
    activations, labels = data["activations"], data["labels"]
    X = activations[:, layer, :]
    return X, labels


def _run_one(name, probe_layer, npz_path):
    probe = load_probe(probe_layer)
    X, labels = _groups_from_activations(npz_path, probe_layer)
    scores = probe.predict_proba(X)[:, 1]
    groups = {label: scores[labels == label] for label in ["DI-CC", "legit", "DI-UE"]}

    result = {"config": name, "layer": probe_layer, "group_sizes": {k: int(len(v)) for k, v in groups.items()}}
    if len(groups["DI-CC"]) >= 2 and len(groups["legit"]) >= 2:
        result["DI-CC_vs_legit"] = auc_bootstrap_ci(groups["DI-CC"], groups["legit"])
    if len(groups["DI-CC"]) >= 2 and len(groups["DI-UE"]) >= 2:
        result["DI-CC_vs_DI-UE"] = auc_bootstrap_ci(groups["DI-CC"], groups["DI-UE"])
    return result


def _best_layer(sweep_report_path):
    with open(sweep_report_path, encoding="utf-8") as f:
        sweep = json.load(f)
    return sweep[0]["layer"]


def main():
    runs = [
        ("n14_accuracy_layer", _best_layer(config.PROBE_SWEEP_REPORT), config.DECOMP_ACTIVATIONS),
        ("n14_variance_ratio_layer", _best_layer(config.PROBE_SWEEP_REPORT_VARRATIO), config.DECOMP_ACTIVATIONS),
        ("n64_accuracy_layer", _best_layer(config.PROBE_SWEEP_REPORT), config.H3_BASELINE_ACTIVATIONS),
        ("n64_variance_ratio_layer", _best_layer(config.PROBE_SWEEP_REPORT_VARRATIO), config.H3_BASELINE_ACTIVATIONS),
    ]

    all_results = []
    for name, layer, npz_path in runs:
        print(f"\n=== {name} (layer {layer}) ===")
        result = _run_one(name, layer, npz_path)
        for key in ("DI-CC_vs_legit", "DI-CC_vs_DI-UE"):
            if key in result:
                r = result[key]
                print(f"  {key}: AUC={r['observed_auc']:.3f}  95% CI=[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]  (n_pos={r['n_pos']}, n_neg={r['n_neg']})")
        all_results.append(result)

    out_path = config.OUTPUTS_REPORTS / "h0_bootstrap_ci_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved consolidated bootstrap CI summary to {out_path}")


if __name__ == "__main__":
    main()
