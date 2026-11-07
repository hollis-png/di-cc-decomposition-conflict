"""E10 (CPU): structured null. Conflict-free copy-vs-recall contrast inside the vague condition (same instruction, same text variant):
legit claims (restate the vague source) vs DI-UE claims (add recalled detail the source never stated). No source conflict exists here.
The NQ-Swap probe is scored at every layer; nulls: isotropic and claim-covariance Gaussian directions (K=2000), Holm over 21 layers;
reference: in-domain LR / DoM (5-fold grouped by entity). Run from the repo root."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, str(Path.cwd()))
import config
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
LAYERS = list(range(8, 29)); K = 2000
nq = np.load(config.NQ_SWAP_ACTIVATIONS); A, Y = nq["activations"], nq["labels"]
tr, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=config.RANDOM_SEED).split(A[:, 0, :], Y, np.arange(len(Y)) // 2))
d = np.load(config.OUTPUTS_ACTIVATIONS / "rerun_baseline_all.npz", allow_pickle=True)["activations"]
meta = [json.loads(l) for l in open(str(config.OUTPUTS_ACTIVATIONS / "rerun_baseline_all.npz") + ".meta.jsonl", encoding="utf-8")]
lab = np.array([m["label"] for m in meta]); cond = np.array([m["condition"] for m in meta]); per = np.array([m["person"] for m in meta])
m = (cond == "vague_baseline") & np.isin(lab, ["legit", "DI-UE"]); y = (lab[m] == "DI-UE").astype(int); g = per[m]
def aucs(S, yy):
    R = np.apply_along_axis(rankdata, 0, S); n1 = yy.sum(); n0 = len(yy) - n1
    return (R[yy == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
def p_two(obs, null): return (1 + (np.abs(null - .5) >= abs(obs - .5)).sum()) / (len(null) + 1)
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o): run = max(run, (len(p) - r) * p[i]); adj[i] = min(1, run)
    return adj
def one(L):
    rng = np.random.default_rng(config.RANDOM_SEED + 2000 + L)
    X = d[:, L, :].astype(np.float32)
    s = LogisticRegression(max_iter=2000, C=1.0).fit(A[tr, L, :], Y[tr]).predict_proba(X)[m, 1]
    obs = float(aucs(s[:, None], y)[0])
    Xc = (X - X.mean(0)); Wc = Xc.T @ rng.standard_normal((X.shape[0], K)).astype(np.float32) / np.sqrt(X.shape[0])
    nulls = {"isotropic": aucs(X[m] @ rng.standard_normal((X.shape[1], K)).astype(np.float32), y), "claim_cov": aucs(X[m] @ Wc, y)}
    Xm = X[m]; oof_lr = np.zeros(len(y)); oof_dom = np.zeros(len(y))
    for a, b in GroupKFold(5).split(Xm, y, g):
        sc = StandardScaler().fit(Xm[a]); oof_lr[b] = LogisticRegression(max_iter=2000, C=0.01).fit(sc.transform(Xm[a]), y[a]).decision_function(sc.transform(Xm[b]))
        oof_dom[b] = Xm[b] @ (Xm[a][y[a] == 1].mean(0) - Xm[a][y[a] == 0].mean(0))
    r = {"auc": obs, "indomain_lr": float(aucs(oof_lr[:, None], y)[0]), "indomain_dom": float(aucs(oof_dom[:, None], y)[0])}
    for k, nl in nulls.items(): r[k] = {"p_two": float(p_two(obs, nl)), "band95": np.percentile(nl, [2.5, 97.5]).round(3).tolist()}
    print("layer", L, round(obs, 3), flush=True); return r
out = Parallel(n_jobs=6)(delayed(one)(L) for L in LAYERS)
summ = {"n_recall_DI-UE": int(y.sum()), "n_copy_legit": int(len(y) - y.sum()), "entities": int(len(set(g))), "layers": LAYERS, "per_layer": out, "holm": {}}
for k in ("isotropic", "claim_cov"):
    adj = holm([o[k]["p_two"] for o in out]); summ["holm"][k] = {"adj": adj.round(4).tolist(), "sig_layers": [L for L, a in zip(LAYERS, adj) if a < .05]}
json.dump(summ, open(config.OUTPUTS_REPORTS / "e10_structured_null.json", "w"), indent=1)
print("E10_DONE", {k: v["sig_layers"] for k, v in summ["holm"].items()})
