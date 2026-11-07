"""Layer-wise transfer curve: probe trained on NQ-Swap at each candidate layer (same split/params as
stage 3), applied to the rerun decomposition activations. Reports DI-CC vs legit, the matched
override-vs-copy control, and the condition-identity check, per layer."""
import json, re, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path.cwd()))
import config
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
nq = np.load(config.NQ_SWAP_ACTIVATIONS); A, Y = nq["activations"], nq["labels"]
grp = np.arange(len(Y)) // 2  # rows are appended as (conflict, no_conflict) pairs of the same example
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=config.RANDOM_SEED).split(A[:, 0, :], Y, grp))
res = []
for L in config.PROBE_LAYER_CANDIDATES:
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(A[tr, L, :], Y[tr])
    res.append({"layer": L, "test_acc": clf.score(A[te, L, :], Y[te]), "probe": clf})
d = np.load(config.OUTPUTS_ACTIVATIONS / "rerun_baseline_all.npz", allow_pickle=True)
meta = [json.loads(l) for l in open(str(config.OUTPUTS_ACTIVATIONS / "rerun_baseline_all.npz") + ".meta.jsonl", encoding="utf-8")]
lab = np.array([m["label"] for m in meta]); cond = np.array([m["condition"] for m in meta]); per = [m["person"] for m in meta]; txt = [m["claim_text"] for m in meta]
bios = {json.loads(l)["person"]: json.loads(l) for l in open(config.H3_BIOS_JSONL, encoding="utf-8")}
def yl(p, c):
    b = bios[p]; mo, mp = YEAR.search(b["specific_bio"]), YEAR.search(b["perturbed_bio"])
    if not (mo and mp): return "na"
    ys = set(YEAR.findall(c)); yo, yp = mo.group(0), mp.group(0)
    return "contrast" if (yo in ys and yp in ys) else "copy" if yp in ys else "override" if yo in ys else "other"
sl = np.array([yl(p, t) if c == "perturbed_baseline" else "na" for p, t, c in zip(per, txt, cond)])
keep = lab != "excluded"; mm = np.isin(lab, ["DI-CC", "legit"]) & keep; y1 = (lab[mm] == "DI-CC").astype(int)
mo_ = (sl == "override") | (sl == "copy"); y2 = (sl[mo_] == "override").astype(int)
c1 = keep & (lab == "legit") & (cond == "perturbed_baseline"); c2 = keep & (lab == "legit") & (cond == "vague_baseline")
out = []
for r in res:
    L = r["layer"]; s = r["probe"].predict_proba(d["activations"][:, L, :].astype(np.float32))[:, 1]
    out.append({"layer": L, "nq_test_acc": round(r["test_acc"], 3), "dicc_vs_legit": round(roc_auc_score(y1, s[mm]), 3),
                "override_vs_copy": round(roc_auc_score(y2, s[mo_]), 3),
                "cond_identity": round(roc_auc_score(np.r_[np.ones(c1.sum()), np.zeros(c2.sum())], np.r_[s[c1], s[c2]]), 3)})
    print(out[-1], flush=True)
json.dump(out, open(config.OUTPUTS_REPORTS / "rerun_layer_curve_grouped.json", "w"), indent=1)
