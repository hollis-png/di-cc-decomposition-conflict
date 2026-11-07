"""E9 (CPU): covariance-matched random-direction nulls for the NQ-Swap probe (same contrasts as E7).
Two null families per layer, K=2000 each:
  claim-cov : w ~ N(0, Sigma_claims), Sigma estimated from the claim activations of the run (w = Xc^T z / sqrt(n))
  nq-cov    : w ~ N(0, Sigma_NQ), from the NQ-Swap training activations (the space the real probe is trained in)
Empirical two-sided p on |AUC-0.5| as in E7, Holm over the 21 layers. Usage (from the repo root): python rerun/e9_cov_null.py"""
import json, re, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, str(Path.cwd()))
import config
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

YEAR = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b"); LAYERS = list(range(8, 29)); K = 2000
nq = np.load(config.NQ_SWAP_ACTIVATIONS); A, Y = nq["activations"], nq["labels"]
tr, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=config.RANDOM_SEED).split(A[:, 0, :], Y, np.arange(len(Y)) // 2))
d = np.load(config.OUTPUTS_ACTIVATIONS / "rerun_baseline_all.npz", allow_pickle=True)["activations"]
meta = [json.loads(l) for l in open(str(config.OUTPUTS_ACTIVATIONS / "rerun_baseline_all.npz") + ".meta.jsonl", encoding="utf-8")]
lab = np.array([m["label"] for m in meta]); cond = np.array([m["condition"] for m in meta]); per = np.array([m["person"] for m in meta]); txt = [m["claim_text"] for m in meta]
bios = {json.loads(l)["person"]: json.loads(l) for l in open(config.H3_BIOS_JSONL, encoding="utf-8")}


def yl(p, c):
    b = bios[p]; mo, mp = YEAR.search(b["specific_bio"]), YEAR.search(b["perturbed_bio"])
    if not (mo and mp): return "na"
    ys = set(YEAR.findall(c)); yo, yp = mo.group(0), mp.group(0)
    return "contrast" if (yo in ys and yp in ys) else "copy" if yp in ys else "override" if yo in ys else "other"


sl = np.array([yl(p, t) if c == "perturbed_baseline" else "na" for p, t, c in zip(per, txt, cond)])
keep = lab != "excluded"
m1 = np.isin(lab, ["DI-CC", "legit"]) & keep; m2 = (sl == "override") | (sl == "copy")
c1 = keep & (lab == "legit") & (cond == "perturbed_baseline"); c2 = keep & (lab == "legit") & (cond == "vague_baseline"); m3 = c1 | c2
sets = {"dicc_vs_legit": (m1, (lab[m1] == "DI-CC").astype(int)), "override_vs_copy": (m2, (sl[m2] == "override").astype(int)),
        "cond_identity": (m3, (cond[m3] == "perturbed_baseline").astype(int))}


def aucs(S, y):
    R = np.apply_along_axis(rankdata, 0, S); n1 = y.sum(); n0 = len(y) - n1
    return (R[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def p_two(obs, null): return (1 + (np.abs(null - .5) >= abs(obs - .5)).sum()) / (len(null) + 1)


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o): run = max(run, (len(p) - r) * p[i]); adj[i] = min(1, run)
    return adj


def cov_dirs(M, rng):  # K draws from N(0, cov(M)) without forming the d x d matrix
    Mc = (M - M.mean(0)).astype(np.float32)
    Z = rng.standard_normal((Mc.shape[0], K)).astype(np.float32)
    return Mc.T @ Z / np.sqrt(Mc.shape[0])


def one_layer(L):
    rng = np.random.default_rng(config.RANDOM_SEED + 1000 + L)
    X = d[:, L, :].astype(np.float32)
    s_real = LogisticRegression(max_iter=2000, C=1.0).fit(A[tr, L, :], Y[tr]).predict_proba(X)[:, 1]
    P = {"claim_cov": X @ cov_dirs(X, rng), "nq_cov": X @ cov_dirs(A[:, L, :], rng)}
    res = {}
    for k, (m, y) in sets.items():
        obs = float(aucs(s_real[m][:, None], y)[0])
        res[k] = {"auc": obs}
        for nm, M in P.items():
            null = aucs(M[m], y)
            res[k][nm] = {"p_two": float(p_two(obs, null)), "band95": np.percentile(null, [2.5, 97.5]).round(3).tolist()}
    print("layer", L, {k: round(res[k]["auc"], 3) for k in sets}, flush=True)
    return res


out = Parallel(n_jobs=6)(delayed(one_layer)(L) for L in LAYERS)
summ = {"layers": LAYERS, "K": K, "per_layer": out, "holm": {}}
for k in sets:
    for nm in ("claim_cov", "nq_cov"):
        adj = holm([o[k][nm]["p_two"] for o in out])
        summ["holm"][f"{k}|{nm}"] = {"adj": adj.round(4).tolist(), "sig_layers": [L for L, a in zip(LAYERS, adj) if a < .05]}
json.dump(summ, open(config.OUTPUTS_REPORTS / "e9_cov_null.json", "w"), indent=1)
print("E9_DONE", {k: v["sig_layers"] for k, v in summ["holm"].items()})
