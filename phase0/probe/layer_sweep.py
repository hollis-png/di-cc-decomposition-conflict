"""Layer sweep: train a logistic-regression probe per candidate layer on the
NQ-Swap conflict/no-conflict activations, pick the best layer.

sweep_layers() selects by held-out classification accuracy (original Phase 0
method). sweep_layers_by_variance_ratio() is a 2026-07-16 addition selecting
by the between/within-class variance ratio instead, following Burger et al.
(2024, "Truth is Universal: Robust Detection of Lies in LLMs", NeurIPS 2024)
-- added after Phase 2's scale ablation found the accuracy-selected layer's
transfer signal to be non-monotonic across model scale (3B/14B not
significant, 7B significant), a pattern the OOD-generalization literature
(Uselis & Oh, 2025, "Intermediate Layer Classifiers for OOD Generalization",
ICLR 2025) attributes partly to in-distribution validation accuracy being a
poor proxy for which layer's direction actually transfers.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import config


def sweep_layers(activations: np.ndarray, labels: np.ndarray, seed: int = config.RANDOM_SEED):
    """`activations`: (N, num_hidden_states, hidden_size). Returns a list of
    dicts (one per candidate layer) with train/test accuracy and the fitted
    probe, sorted best-first by test accuracy.
    """
    results = []
    for layer in config.PROBE_LAYER_CANDIDATES:
        X = activations[:, layer, :]
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.25, random_state=seed, stratify=labels
        )
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        results.append({"layer": layer, "train_acc": train_acc, "test_acc": test_acc, "probe": clf})
    results.sort(key=lambda r: r["test_acc"], reverse=True)
    return results


def variance_ratio(activations: np.ndarray, labels: np.ndarray, layer: int) -> float:
    """Burger et al. (2024) layer-selection criterion: ratio of between-class
    to within-class variance at a given layer. Layers that maximize this
    ratio encode a cleaner, more separable concept direction -- the
    conjecture (grounded in the OOD-generalization literature cited above)
    is that this is less prone to overfitting idiosyncrasies of the training
    distribution than raw held-out classification accuracy, and therefore
    transfers better to a different setting (here: decomposition-induced
    conflict, never seen during probe training).
    """
    X = activations[:, layer, :]
    X1 = X[labels == 1]
    X0 = X[labels == 0]
    mu1, mu0 = X1.mean(axis=0), X0.mean(axis=0)
    between = np.mean((mu1 - mu0) ** 2)
    within = X1.var(axis=0).mean() + X0.var(axis=0).mean()
    return float(between / within)


def sweep_layers_by_variance_ratio(activations: np.ndarray, labels: np.ndarray, seed: int = config.RANDOM_SEED):
    """Same interface and output shape as sweep_layers(), but ranks
    candidate layers by variance_ratio() instead of held-out accuracy. The
    probe itself is still fit as a standard logistic regression exactly like
    sweep_layers() -- only the layer-selection criterion differs -- so any
    change in downstream transfer AUC can be attributed to the selection
    criterion alone, not to a different probe-fitting method.
    """
    results = []
    for layer in config.PROBE_LAYER_CANDIDATES:
        ratio = variance_ratio(activations, labels, layer)
        X = activations[:, layer, :]
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.25, random_state=seed, stratify=labels
        )
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        results.append(
            {"layer": layer, "variance_ratio": ratio, "train_acc": train_acc, "test_acc": test_acc, "probe": clf}
        )
    results.sort(key=lambda r: r["variance_ratio"], reverse=True)
    return results
