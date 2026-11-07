"""E3: apply the fixed grouped-split NQ-Swap probes (layers 8-28, trained once) to one run's activations."""
import json, re, sys
from pathlib import Path
import numpy as np, joblib
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b"); LAYERS = list(range(8, 29)); HOME = Path.home()
cache = HOME / "dicc-e3" / "probes_grouped_seed42.joblib"
if cache.exists(): probes = joblib.load(cache)
else:
    nq = np.load(HOME / "dicc-rerun/outputs/activations/nq_swap_activations.npz"); A, Y = nq["activations"], nq["labels"]
    tr, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(A[:, 0, :], Y, np.arange(len(Y)) // 2))
    probes = {L: LogisticRegression(max_iter=2000, C=1.0).fit(A[tr, L, :], Y[tr]) for L in LAYERS}; joblib.dump(probes, cache)
d = Path.cwd(); z = np.load(d / "outputs/activations/e3_pert.npz", allow_pickle=True)
meta = [json.loads(l) for l in open(d / "outputs/activations/e3_pert.npz.meta.jsonl", encoding="utf-8")]
lab = np.array([m["label"] for m in meta]); per = [m["person"] for m in meta]; txt = [m["claim_text"] for m in meta]
bios = {json.loads(l)["person"]: json.loads(l) for l in open(d / "data/processed/h3_bios.jsonl", encoding="utf-8")}
def yl(p, c):
    b = bios[p]; mo, mp = YEAR.search(b["specific_bio"]), YEAR.search(b["perturbed_bio"])
    if not (mo and mp): return "na"
    ys = set(YEAR.findall(c)); yo, yp = mo.group(0), mp.group(0)
    return "contrast" if (yo in ys and yp in ys) else "copy" if yp in ys else "override" if yo in ys else "other"
sl = np.array([yl(p, t) for p, t in zip(per, txt)])
keep = lab != "excluded"; m1 = np.isin(lab, ["DI-CC", "legit"]) & keep; y1 = (lab[m1] == "DI-CC").astype(int)
m2 = (sl == "override") | (sl == "copy"); y2 = (sl[m2] == "override").astype(int)
out = {"n": {"DI-CC": int((lab == "DI-CC").sum()), "legit": int((lab == "legit").sum()), "excluded": int((lab == "excluded").sum()),
             "DI-CC_entities": len({p for p, l in zip(per, lab) if l == "DI-CC"}), "override": int(y2.sum()), "copy": int((1 - y2).sum())},
       "layers": LAYERS, "dicc_vs_legit": [], "override_vs_copy": []}
auc = lambda y, s: float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None
for L in LAYERS:
    s = probes[L].predict_proba(z["activations"][:, L, :].astype(np.float32))[:, 1]
    out["dicc_vs_legit"].append(auc(y1, s[m1])); out["override_vs_copy"].append(auc(y2, s[m2]))
json.dump(out, open(d / "outputs/reports/e3_run.json", "w"), indent=1); print(out["n"])
