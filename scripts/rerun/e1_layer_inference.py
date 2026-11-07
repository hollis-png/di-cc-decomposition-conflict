"""E1: per-layer AUC with entity-cluster bootstrap CIs, max-statistic permutation across layers, and
layer-sensitivity statistics. Probes trained with grouped split (pair-leakage fixed). CPU only."""
import json, re, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, str(Path.cwd()))
import config
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
LAYERS = list(range(8, 29)); B = 2000; rng = np.random.default_rng(config.RANDOM_SEED)
nq = np.load(config.NQ_SWAP_ACTIVATIONS); A, Y = nq["activations"], nq["labels"]
grp = np.arange(len(Y)) // 2
tr, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=config.RANDOM_SEED).split(A[:, 0, :], Y, grp))
d = np.load(config.OUTPUTS_ACTIVATIONS / "rerun_baseline_all.npz", allow_pickle=True)
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
sets = {}
m = np.isin(lab, ["DI-CC", "legit"]) & keep; sets["dicc_vs_legit"] = (np.where(m)[0], (lab[m] == "DI-CC").astype(int), cond[m])
m = (sl == "override") | (sl == "copy"); sets["override_vs_copy"] = (np.where(m)[0], (sl[m] == "override").astype(int), np.array(["x"] * m.sum()))
c1 = keep & (lab == "legit") & (cond == "perturbed_baseline"); c2 = keep & (lab == "legit") & (cond == "vague_baseline"); m = c1 | c2
sets["cond_identity"] = (np.where(m)[0], (cond[m] == "perturbed_baseline").astype(int), np.array(["x"] * m.sum()))
print("probes...", flush=True)
S = np.zeros((len(lab), len(LAYERS)))
for j, L in enumerate(LAYERS):
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(A[tr, L, :], Y[tr])
    S[:, j] = clf.predict_proba(d["activations"][:, L, :].astype(np.float32))[:, 1]
def auc_cols(R, y):  # R: ranks (n, k); y: 0/1
    n1 = y.sum(); n0 = len(y) - n1
    return (R[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
out = {"layers": LAYERS, "B": B}
for name, (idx, y, strata) in sets.items():
    s = S[idx]; R = np.apply_along_axis(rankdata, 0, s); obs = auc_cols(R, y); pr = per[idx]
    # cluster bootstrap
    up = np.unique(pr); byp = {p: np.where(pr == p)[0] for p in up}; boots = []
    for _ in range(B):
        ii = np.concatenate([byp[p] for p in rng.choice(up, len(up))]); yy = y[ii]
        if yy.sum() < 2 or (1 - yy).sum() < 2: continue
        boots.append(auc_cols(np.apply_along_axis(rankdata, 0, s[ii]), yy))
    boots = np.array(boots); lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    # max-statistic permutation (labels permuted within strata)
    mx1, mx2, sd0 = [], [], []
    for _ in range(B):
        yp = y.copy()
        for st in np.unique(strata):
            k = np.where(strata == st)[0]; yp[k] = rng.permutation(y[k])
        a = auc_cols(R, yp); mx1.append(a.max()); mx2.append(np.abs(a - .5).max()); sd0.append(a.std())
    mx1, mx2, sd0 = map(np.array, (mx1, mx2, sd0))
    res = {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "auc": obs.round(4).tolist(), "ci_lo": lo.round(4).tolist(), "ci_hi": hi.round(4).tolist(),
           "max_auc": float(obs.max()), "p_max_onesided": float((1 + (mx1 >= obs.max()).sum()) / (B + 1)),
           "max_dev": float(np.abs(obs - .5).max()), "p_max_twosided": float((1 + (mx2 >= np.abs(obs - .5).max()).sum()) / (B + 1)),
           "layer_sd": float(obs.std()), "null_sd_mean": float(sd0.mean()), "p_sd": float((1 + (sd0 >= obs.std()).sum()) / (B + 1)),
           "bootstrap_valid": int(len(boots)), "pairs": {}}
    for a_, b_ in ((28, 22), (28, 20), (20, 22)):
        ia, ib = LAYERS.index(a_), LAYERS.index(b_); dd = boots[:, ia] - boots[:, ib]
        res["pairs"][f"{a_}-{b_}"] = {"diff": float(obs[ia] - obs[ib]), "ci": np.percentile(dd, [2.5, 97.5]).round(3).tolist()}
    out[name] = res
    print(name, {k: v for k, v in res.items() if k not in ("auc", "ci_lo", "ci_hi")}, flush=True)
json.dump(out, open(config.OUTPUTS_REPORTS / "e1_layer_inference.json", "w"), indent=1)
