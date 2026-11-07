"""Analysis of a Phase-0-style run (stages 4-7, scale-suffixed paths): paper-style AUCs plus
entity-clustered CIs and a supervised ceiling, for both the accuracy- and variance-ratio-selected layer."""
import json, sys
from pathlib import Path
import joblib, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path.cwd()))
import config
rng = np.random.RandomState(42)
d = np.load(config.DECOMP_ACTIVATIONS, allow_pickle=True)
meta = [json.loads(l) for l in open(config.DECOMP_ACTIVATIONS.with_suffix(".meta.jsonl"), encoding="utf-8")]
lab = np.array([m["label"] for m in meta]); per = np.array([m["person"] for m in meta])
acc_layer = json.load(open(config.PROBE_SWEEP_REPORT))[0]["layer"]
vr_layer = json.load(open(config.PROBE_SWEEP_REPORT_VARRATIO))[0]["layer"]
def auc(y, s): return float(roc_auc_score(y, s))
def cl_ci(y, s, g, B=2000):
    up = np.unique(g); idx = {p: np.where(g == p)[0] for p in up}; out = []
    for _ in range(B):
        ii = np.concatenate([idx[p] for p in up[rng.randint(0, len(up), len(up))]])
        if y[ii].sum() >= 2 and (1 - y[ii]).sum() >= 2: out.append(roc_auc_score(y[ii], s[ii]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
rep = {"sizes": {k: int((lab == k).sum()) for k in ("DI-CC", "legit", "DI-UE")}, "acc_layer": int(acc_layer), "vr_layer": int(vr_layer)}
for name, L in (("accuracy", acc_layer), ("varratio", vr_layer)):
    probe = joblib.load(str(config.PROBE_CHECKPOINT).format(layer=L))
    s_all = probe.predict_proba(d["activations"][:, L, :].astype(np.float32))[:, 1]
    for other in ("legit", "DI-UE"):
        m = np.isin(lab, ["DI-CC", other]); y = (lab[m] == "DI-CC").astype(int)
        rep[f"{name}_vs_{other}"] = {"auc": auc(y, s_all[m]), "cluster_ci": cl_ci(y, s_all[m], per[m])}
m = np.isin(lab, ["DI-CC", "legit"]); y = (lab[m] == "DI-CC").astype(int); g = per[m]
rep["ceiling"] = {}
for L in (8, 16, acc_layer, 24):
    X = d["activations"][m][:, L, :].astype(np.float32); X = (X - X.mean(0)) / (X.std(0) + 1e-6); s = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        s[te] = LogisticRegression(C=0.01, max_iter=2000).fit(X[tr], y[tr]).decision_function(X[te])
    rep["ceiling"][int(L)] = auc(y, s)
json.dump(rep, open(config.OUTPUTS_REPORTS / "rerun_p0_analysis.json", "w"), indent=2)
print(json.dumps(rep))
