"""E7 (CPU): (a) empirical p-values + Holm for the NQ-Swap probe against random-direction (K=2000) and shuffled-label (K=100) nulls;
(b) in-domain probe and difference-of-means direction as comparison points (5-fold, grouped by entity);
(c) layer-28 token-type stratification; (d) per-layer activation norms (post-final-norm check)."""
import json, re, sys, os
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, str(Path.cwd()))
import config
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b"); LAYERS = list(range(8, 29)); KR, KS = 2000, 100
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
def pval(obs, null):  # null: (K,) ; returns (one-sided upper, two-sided on |auc-.5|)
    return (1 + (null >= obs).sum()) / (len(null) + 1), (1 + (np.abs(null - .5) >= abs(obs - .5)).sum()) / (len(null) + 1)
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o): run = max(run, (len(p) - r) * p[i]); adj[i] = min(1, run)
    return adj
def one_layer(j, L):
    rng = np.random.default_rng(config.RANDOM_SEED + L)
    X = d[:, L, :].astype(np.float32)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(A[tr, L, :], Y[tr]); s_real = clf.predict_proba(X)[:, 1]
    dom_nq = A[Y == 1, L, :].mean(0) - A[Y == 0, L, :].mean(0)   # NQ-Swap difference of means (label 1 = conflict)
    W = rng.standard_normal((X.shape[1], KR)).astype(np.float32); P = X @ W
    shuf = [LogisticRegression(max_iter=2000, C=1.0).fit(A[tr, L, :], rng.permutation(Y[tr])).predict_proba(X)[:, 1] for _ in range(KS)]
    res = {}
    for k, (m, y) in sets.items():
        obs = float(aucs(s_real[m][:, None], y)[0]); rnd = aucs(P[m], y); sh = np.array([aucs(s[m][:, None], y)[0] for s in shuf])
        # in-domain probe and DoM, 5-fold grouped by entity (out-of-fold scores)
        Xm, g = X[m], per[m]; oof_lr = np.zeros(m.sum()); oof_dom = np.zeros(m.sum())
        for a, b in GroupKFold(5).split(Xm, y, g):
            sc = StandardScaler().fit(Xm[a]); oof_lr[b] = LogisticRegression(max_iter=2000, C=0.01).fit(sc.transform(Xm[a]), y[a]).decision_function(sc.transform(Xm[b]))
            oof_dom[b] = Xm[b] @ (Xm[a][y[a] == 1].mean(0) - Xm[a][y[a] == 0].mean(0))
        res[k] = {"auc": obs, "p_rand": pval(obs, rnd), "p_shuf": pval(obs, sh), "auc_nq_dom_zero_shot": float(aucs((X[m] @ dom_nq)[:, None], y)[0]),
                  "auc_indomain_lr": float(aucs(oof_lr[:, None], y)[0]), "auc_indomain_dom": float(aucs(oof_dom[:, None], y)[0])}
    # token-type stratification (override vs copy): does the claim end in a year?
    m, y = sets["override_vs_copy"]; idx = np.where(m)[0]; endy = np.array([bool(YEAR.search(txt[i].rstrip(" .")[-6:])) for i in idx])
    res["strat_override_vs_copy"] = {nm: {"n_pos": int(y[sel].sum()), "n_neg": int((1 - y[sel]).sum()),
        "auc": float(aucs(s_real[idx][sel][:, None], y[sel])[0]) if y[sel].sum() > 1 and (1 - y[sel]).sum() > 1 else None}
        for nm, sel in {"ends_with_year": endy, "year_not_last": ~endy}.items()}
    res["norm_mean"] = float(np.linalg.norm(X, axis=1).mean())
    print("layer", L, {k: round(res[k]["auc"], 3) for k in sets}, flush=True)
    return res
out = Parallel(n_jobs=6)(delayed(one_layer)(j, L) for j, L in enumerate(LAYERS))
summ = {"layers": LAYERS, "K_random": KR, "K_shuffled": KS, "per_layer": out, "holm": {}}
for k in sets:
    for nm in ("p_rand", "p_shuf"):
        for side, si in (("upper", 0), ("two", 1)):
            adj = holm([o[k][nm][si] for o in out]); summ["holm"][f"{k}|{nm}|{side}"] = {"adj": adj.round(4).tolist(), "sig_layers": [L for L, a in zip(LAYERS, adj) if a < .05]}
json.dump(summ, open(config.OUTPUTS_REPORTS / "e7_cpu_checks.json", "w"), indent=1)
print("E7_DONE")
