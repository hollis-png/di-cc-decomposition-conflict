"""E4: null probes (shuffled-label NQ-Swap probes and random directions) applied to the same three contrasts as E1."""
import json, re, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, str(Path.cwd()))
import config
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b"); LAYERS = list(range(8, 29)); KS, KR = 30, 1000; rng = np.random.default_rng(config.RANDOM_SEED)
nq = np.load(config.NQ_SWAP_ACTIVATIONS); A, Y = nq["activations"], nq["labels"]
tr, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=config.RANDOM_SEED).split(A[:, 0, :], Y, np.arange(len(Y)) // 2))
d = np.load(config.OUTPUTS_ACTIVATIONS / "rerun_baseline_all.npz", allow_pickle=True)["activations"]
meta = [json.loads(l) for l in open(str(config.OUTPUTS_ACTIVATIONS / "rerun_baseline_all.npz") + ".meta.jsonl", encoding="utf-8")]
lab = np.array([m["label"] for m in meta]); cond = np.array([m["condition"] for m in meta]); per = [m["person"] for m in meta]; txt = [m["claim_text"] for m in meta]
bios = {json.loads(l)["person"]: json.loads(l) for l in open(config.H3_BIOS_JSONL, encoding="utf-8")}
def yl(p, c):
    b = bios[p]; mo, mp = YEAR.search(b["specific_bio"]), YEAR.search(b["perturbed_bio"])
    if not (mo and mp): return "na"
    ys = set(YEAR.findall(c)); yo, yp = mo.group(0), mp.group(0)
    return "contrast" if (yo in ys and yp in ys) else "copy" if yp in ys else "override" if yo in ys else "other"
sl = np.array([yl(p, t) if c == "perturbed_baseline" else "na" for p, t, c in zip(per, txt, cond)])
keep = lab != "excluded"
m1 = np.isin(lab, ["DI-CC", "legit"]) & keep; y1 = (lab[m1] == "DI-CC").astype(int)
m2 = (sl == "override") | (sl == "copy"); y2 = (sl[m2] == "override").astype(int)
c1 = keep & (lab == "legit") & (cond == "perturbed_baseline"); c2 = keep & (lab == "legit") & (cond == "vague_baseline"); m3 = c1 | c2; y3 = (cond[m3] == "perturbed_baseline").astype(int)
sets = {"dicc_vs_legit": (m1, y1), "override_vs_copy": (m2, y2), "cond_identity": (m3, y3)}
def aucs(S, y):  # S: (n, k) scores
    R = np.apply_along_axis(rankdata, 0, S); n1 = y.sum(); n0 = len(y) - n1
    return (R[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
out = {"layers": LAYERS, "K_shuffled": KS, "K_random": KR}
res = {k: {"shuf": [], "rand": []} for k in sets}
for L in LAYERS:
    X = d[:, L, :].astype(np.float32)
    # shuffled-label probes
    sh = {k: [] for k in sets}
    for _ in range(KS):
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(A[tr, L, :], rng.permutation(Y[tr])); s = clf.predict_proba(X)[:, 1]
        for k, (m, y) in sets.items(): sh[k].append(aucs(s[m][:, None], y)[0])
    W = rng.standard_normal((X.shape[1], KR)).astype(np.float32); P = X @ W
    for k, (m, y) in sets.items():
        res[k]["shuf"].append(np.percentile(sh[k], [2.5, 50, 97.5]).round(4).tolist())
        res[k]["rand"].append(np.percentile(aucs(P[m], y), [2.5, 50, 97.5]).round(4).tolist())
    print("layer", L, {k: (res[k]["shuf"][-1], res[k]["rand"][-1]) for k in sets}, flush=True)
out.update(res); json.dump(out, open(config.OUTPUTS_REPORTS / "e4_null_probes.json", "w"), indent=1)
