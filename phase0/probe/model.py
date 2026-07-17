"""Save/load the frozen probe selected by layer_sweep. Kept minimal: the
probe is a plain sklearn LogisticRegression, saved with joblib alongside its
selected layer index so downstream transfer-eval code applies it consistently.
"""
import json

import joblib

import config


def save_probe(layer: int, probe) -> None:
    path = str(config.PROBE_CHECKPOINT).format(layer=layer)
    joblib.dump(probe, path)


def load_probe(layer: int):
    path = str(config.PROBE_CHECKPOINT).format(layer=layer)
    return joblib.load(path)


def save_sweep_report(results: list[dict]) -> None:
    report = [{"layer": r["layer"], "train_acc": r["train_acc"], "test_acc": r["test_acc"]} for r in results]
    with open(config.PROBE_SWEEP_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def save_sweep_report_variance_ratio(results: list[dict]) -> None:
    """Same as save_sweep_report, but for sweep_layers_by_variance_ratio()
    results -- includes the variance_ratio field and writes to a separate
    report path so it never overwrites the accuracy-based sweep report."""
    report = [
        {"layer": r["layer"], "variance_ratio": r["variance_ratio"], "train_acc": r["train_acc"], "test_acc": r["test_acc"]}
        for r in results
    ]
    with open(config.PROBE_SWEEP_REPORT_VARRATIO, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
