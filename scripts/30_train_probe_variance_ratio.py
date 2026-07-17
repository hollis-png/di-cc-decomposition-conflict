"""Stage 30 (layer-selection reanalysis): layer-sweep + train the frozen
CM-conflict probe on NQ-Swap activations, selecting the layer by Burger et
al. (2024) variance-ratio criterion instead of held-out accuracy (stage 3's
method). Reuses the SAME NQ-Swap activations stage 2 already extracted for
this scale -- no GPU needed.

Drafted by gemma-2-9b-it (LM Studio), reviewed and finalized by Claude.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from phase0.probe.layer_sweep import sweep_layers_by_variance_ratio
from phase0.probe.model import save_probe, save_sweep_report_variance_ratio


def main():
    data = np.load(config.NQ_SWAP_ACTIVATIONS)
    activations, labels = data["activations"], data["labels"]
    print(f"Loaded activations {activations.shape}, labels {labels.shape} "
          f"({labels.sum()} conflict / {len(labels) - labels.sum()} no_conflict)")

    results = sweep_layers_by_variance_ratio(activations, labels)
    save_sweep_report_variance_ratio(results)

    print("Layer sweep results (best first, ranked by variance_ratio):")
    for r in results:
        print(f"  layer {r['layer']:2d}: variance_ratio={r['variance_ratio']:.3f}  train_acc={r['train_acc']:.3f}  test_acc={r['test_acc']:.3f}")

    best = results[0]
    save_probe(best["layer"], best["probe"])
    print(f"\nBest layer: {best['layer']} (variance_ratio={best['variance_ratio']:.3f}, test_acc={best['test_acc']:.3f}), probe saved.")


if __name__ == "__main__":
    main()
