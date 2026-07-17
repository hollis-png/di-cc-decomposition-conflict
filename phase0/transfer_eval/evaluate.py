"""H0 decision logic (design doc Phase 0, "?��??�決策�?"): does the frozen
NQ-Swap probe fire more on DI-CC decomposition spans than on legit-coref /
DI-UE spans? AUC + permutation test, no parametric assumptions.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def auc_with_permutation_test(pos_scores: np.ndarray, neg_scores: np.ndarray, n_permutations: int = 2000, seed: int = 42):
    """pos_scores: probe P(conflict) scores for the positive group (e.g. DI-CC).
    neg_scores: same for the comparison group (e.g. legit, or DI-UE).
    Returns dict with observed AUC, permutation p-value, and group sizes.
    """
    y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])
    observed_auc = roc_auc_score(y, scores)

    rng = np.random.RandomState(seed)
    perm_aucs = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_y = rng.permutation(y)
        perm_aucs[i] = roc_auc_score(perm_y, scores)
    p_value = float((perm_aucs >= observed_auc).mean())

    return {
        "observed_auc": float(observed_auc),
        "permutation_p_value": p_value,
        "n_pos": int(len(pos_scores)),
        "n_neg": int(len(neg_scores)),
    }


def auc_bootstrap_ci(pos_scores: np.ndarray, neg_scores: np.ndarray, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 42):
    """Percentile bootstrap CI for AUC: resample pos/neg groups independently
    (stratified, matching the paired-group AUC setup used throughout this
    project), recompute AUC each time. Returns observed AUC plus the
    [alpha/2, 1-alpha/2] percentile interval. Complements (does not
    replace) the permutation test above -- permutation tests the null of
    no separation, the CI quantifies uncertainty in the effect size itself.
    """
    pos_scores = np.asarray(pos_scores)
    neg_scores = np.asarray(neg_scores)
    y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])
    observed_auc = roc_auc_score(y, scores)

    rng = np.random.RandomState(seed)
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    boot_aucs = np.empty(n_resamples)
    for i in range(n_resamples):
        pos_sample = pos_scores[rng.randint(0, n_pos, n_pos)]
        neg_sample = neg_scores[rng.randint(0, n_neg, n_neg)]
        y_b = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
        scores_b = np.concatenate([pos_sample, neg_sample])
        boot_aucs[i] = roc_auc_score(y_b, scores_b)

    lower = float(np.percentile(boot_aucs, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_aucs, 100 * (1 - alpha / 2)))

    return {
        "observed_auc": float(observed_auc),
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_level": 1 - alpha,
        "n_resamples": n_resamples,
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
    }
