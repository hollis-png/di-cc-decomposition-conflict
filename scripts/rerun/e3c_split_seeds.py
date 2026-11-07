"""E3c: variability of layer AUCs across 20 NQ-Swap grouped-split seeds (probe training variance). CPU."""
import json, re, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b"); HOME = Path.home(); R = HOME / "dicc-rerun"; LAYERS = [18, 19, 20, 22, 23, 25, 28]
nq = np.load(R / "outputs/activations/nq_swap_activations.npz"); A, Y = nq["activations"], nq["labels"]; grp = np.arange(len(Y)) // 2
z = np.load(R / "outputs/activations/rerun_baseline_all.npz", allow_pickle=True)
meta = [json.loads(l) for l in open(R / "outputs/activations/rerun_baseline_all.npz.meta.jsonl", encoding="utf-8")]
lab = np.array([m["label"] for m in meta]); cond = np.array([m["condition"] for m in meta]); per = [m["person"] for m in meta]; txt = [m["claim_text"] for m in meta]
bios = {json.loads(l)["person"]: json.loads(l) for l in open(R / "data/processed/h3_bios.jsonl", encoding="utf-8")}
def yl(p, c):
    b = bios[p]; mo, mp = YEAR.search(b["specific_bio"]), YEAR.search(b["perturbed_bio"])
    if not (mo and mp): return "na"
    ys = set(YEAR.findall(c)); yo, yp = mo.group(0), mp.group(0)
    return "contrast" if (yo in ys and yp in ys) else "copy" if yp in ys else "override" if yo in ys else "other"
sl = np.array([yl(p, t) if c == "perturbed_baseline" else "na" for p, t, c in zip(per, txt, cond)])
keep = lab != "excluded"; m1 = np.isin(lab, ["DI-CC", "legit"]) & keep; y1 = (lab[m1] == "DI-CC").astype(int); m2 = (sl == "override") | (sl == "copy"); y2 = (sl[m2] == "override").astype(int)
res = {L: {"dicc": [], "match": [], "acc": []} for L in LAYERS}
for seed in range(20):
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(A[:, 0, :], Y, grp))
    for L in LAYERS:
        c = LogisticRegression(max_iter=2000, C=1.0).fit(A[tr, L, :], Y[tr]); s = c.predict_proba(z["activations"][:, L, :].astype(np.float32))[:, 1]
        res[L]["dicc"].append(roc_auc_score(y1, s[m1])); res[L]["match"].append(roc_auc_score(y2, s[m2])); res[L]["acc"].append(c.score(A[te, L, :], Y[te]))
    print("seed", seed, flush=True)
summ = {L: {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "min": float(np.min(v)), "max": float(np.max(v))} for k, v in r.items()} for L, r in res.items()}
json.dump({"raw": {str(L): r for L, r in res.items()}, "summary": {str(L): s for L, s in summ.items()}}, open(R / "outputs/reports/e3c_split_seeds.json", "w"), indent=1)
for L, s in summ.items(): print(L, {k: (round(v["mean"], 3), round(v["sd"], 3)) for k, v in s.items()})
