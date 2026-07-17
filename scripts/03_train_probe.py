"""Stage 3: layer-sweep + train the frozen CM-conflict probe on NQ-Swap
activations. No GPU needed (activations are already extracted).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from phase0.probe.layer_sweep import sweep_layers
from phase0.probe.model import save_probe, save_sweep_report


def main():
    data = np.load(config.NQ_SWAP_ACTIVATIONS)
    activations, labels = data["activations"], data["labels"]
    print(f"Loaded activations {activations.shape}, labels {labels.shape} "
          f"({labels.sum()} conflict / {len(labels) - labels.sum()} no_conflict)")

    results = sweep_layers(activations, labels)
    save_sweep_report(results)

    print("Layer sweep results (best first):")
    for r in results:
        print(f"  layer {r['layer']:2d}: train_acc={r['train_acc']:.3f}  test_acc={r['test_acc']:.3f}")

    best = results[0]
    save_probe(best["layer"], best["probe"])
    print(f"\nBest layer: {best['layer']} (test_acc={best['test_acc']:.3f}), probe saved.")


if __name__ == "__main__":
    main()
