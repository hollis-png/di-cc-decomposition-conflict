"""Supervised ceiling: can DI-CC be separated from legit at all in this model's decomposition
activations (entity-grouped CV)? Compared with the NQ-Swap-transfer AUC from a_main.py."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path.cwd()))
import config
R = config.OUTPUTS_REPORTS
d = np.load(config.OUTPUTS_ACTIVATIONS / "rerun_baseline_all.npz", allow_pickle=True)
meta = [json.loads(l) for l in open(str(config.OUTPUTS_ACTIVATIONS / "rerun_baseline_all.npz") + ".meta.jsonl", encoding="utf-8")]
lab = np.array([m["label"] for m in meta]); per = np.array([m["person"] for m in meta])
m = np.isin(lab, ["DI-CC", "legit"]); y = (lab[m] == "DI-CC").astype(int); g = per[m]
out = {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "layers": {}}
for L in range(8, d["activations"].shape[1], 4):
    X = d["activations"][m][:, L, :].astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    s = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        clf = LogisticRegression(C=0.01, max_iter=2000).fit(X[tr], y[tr]); s[te] = clf.decision_function(X[te])
    out["layers"][L] = float(roc_auc_score(y, s)); print(L, round(out["layers"][L], 3), flush=True)
json.dump(out, open(R / "rerun_a5_ceiling.json", "w"), indent=2)
