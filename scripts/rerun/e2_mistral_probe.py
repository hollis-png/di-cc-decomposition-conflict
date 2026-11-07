"""E2: original release Mistral probe vs retrained probe, on double-BOS (m-orig) and single-BOS (m-fix) activations."""
import json, sys, warnings
from pathlib import Path
import numpy as np, joblib
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
DS = sys.argv[1]; rng = np.random.default_rng(0); B = 2000; HOME = Path.home()
orig = joblib.load(HOME / "dicc-m-orig/outputs/probes/probe_layer18_mistral7b.joblib")
def load(d):
    a = np.load(HOME / d / f"outputs/activations/{DS}.npz", allow_pickle=True)["activations"]
    meta = [json.loads(l) for l in open(HOME / d / (f"outputs/activations/{DS}.npz.meta.jsonl" if DS.startswith("rerun") else f"outputs/activations/{DS}.meta.jsonl"), encoding="utf-8")]
    nq = np.load(HOME / d / "outputs/activations/nq_swap_activations_mistral7b.npz")
    return a, np.array([m["label"] for m in meta]), np.array([m["person"] for m in meta]), nq["activations"], nq["labels"]
def ci(y, s, per):
    up = np.unique(per); byp = {p: np.where(per == p)[0] for p in up}; o = []
    for _ in range(B):
        ii = np.concatenate([byp[p] for p in rng.choice(up, len(up))])
        if y[ii].sum() >= 2 and (1 - y[ii]).sum() >= 2: o.append(roc_auc_score(y[ii], s[ii]))
    return np.percentile(o, [2.5, 97.5]).round(3).tolist()
res = {}
for L in (18,):
    for dn, tag in (("dicc-m-orig", "doubleBOS"), ("dicc-m-fix", "singleBOS")):
        a, lab, per, nqa, nqy = load(dn)
        retr = LogisticRegression(max_iter=2000, C=1.0).fit(nqa[:, L, :], nqy)
        for pn, p in (("orig_release", orig), ("retrained_own_nq", retr)):
            s = p.predict_proba(a[:, L, :].astype(np.float32))[:, 1]
            for other in ("legit", "DI-UE"):
                m = np.isin(lab, ["DI-CC", other]); y = (lab[m] == "DI-CC").astype(int)
                r = {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "auc": round(roc_auc_score(y, s[m]), 3), "cluster_ci": ci(y, s[m], per[m])}
                res[f"L{L}|acts={tag}|probe={pn}|vs_{other}"] = r; print(f"L{L}", tag, pn, other, r, flush=True)
        res[f"L{L}|acts={tag}|cos_orig_vs_retrained"] = float(orig.coef_.ravel() @ retr.coef_.ravel() / np.linalg.norm(orig.coef_) / np.linalg.norm(retr.coef_))
        print("cos", tag, res[f"L{L}|acts={tag}|cos_orig_vs_retrained"])
json.dump(res, open(HOME / "dicc-m-orig/outputs/reports/e2_mistral_probe.json", "w"), indent=1)
