"""Self-review checks 1-3 on the rerun activations (CPU only).
 0  reproduction of the paper's H0 AUCs (iid bootstrap) from regenerated data
 1  entity-clustered bootstrap CIs
 2  NLI(R, claim) detector (circular for NLI-defined labels) and string-label version
 3  matched controls: override vs faithful-copy of the tampered year; condition-identity check
"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import joblib, numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

R = config.OUTPUTS_REPORTS
rng = np.random.RandomState(42)
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")

def auc(pos, neg):
    if len(pos) < 2 or len(neg) < 2: return float("nan")
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))

def iid_ci(pos, neg, B=2000):
    a = [auc(pos[rng.randint(0, len(pos), len(pos))], neg[rng.randint(0, len(neg), len(neg))]) for _ in range(B)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

def cluster_ci(scores, y, persons, B=2000):
    """Resample persons with replacement; keep all of a person's claims."""
    up = np.array(sorted(set(persons))); idx = defaultdict(list)
    for i, p in enumerate(persons): idx[p].append(i)
    out = []
    for _ in range(B):
        pick = up[rng.randint(0, len(up), len(up))]
        ii = np.concatenate([idx[p] for p in pick])
        yy, ss = y[ii], scores[ii]
        if yy.sum() >= 2 and (1 - yy).sum() >= 2: out.append(roc_auc_score(yy, ss))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))], len(out)

def main():
    data = np.load(R.parent / "activations" / "rerun_baseline_all.npz", allow_pickle=True)
    acts = data["activations"]
    meta = [json.loads(l) for l in open(R.parent / "activations" / "rerun_baseline_all.npz.meta.jsonl", encoding="utf-8")]
    lab = np.array([m["label"] for m in meta]); cond = np.array([m["condition"] for m in meta])
    per = np.array([m["person"] for m in meta]); txt = [m["claim_text"] for m in meta]
    sweep_v = json.load(open(config.PROBE_SWEEP_REPORT_VARRATIO)); lv = sweep_v[0]["layer"]
    sweep_a = json.load(open(config.PROBE_SWEEP_REPORT)); la = max(sweep_a, key=lambda r: r["test_acc"])["layer"]
    print("layers: var-ratio", lv, "accuracy", la)
    rep = {"layers": {"varratio": lv, "accuracy": la}, "label_counts": {c: dict(Counter(lab[cond == c])) for c in sorted(set(cond))}}
    print(rep["label_counts"])
    scores = {}
    for name, L in (("varratio", lv), ("accuracy", la)):
        probe = joblib.load(str(config.PROBE_CHECKPOINT).format(layer=L))
        scores[name] = probe.predict_proba(acts[:, L, :].astype(np.float32))[:, 1]

    # --- 0/1: paper's comparison on the non-excluded claims of both baseline conditions
    keep = lab != "excluded"
    rep["h0"] = {}
    for name, s in scores.items():
        for other in ("legit", "DI-UE"):
            m = keep & np.isin(lab, ["DI-CC", other]); y = (lab[m] == "DI-CC").astype(int)
            pos, neg = s[m][y == 1], s[m][y == 0]
            cci, nb = cluster_ci(s[m], y, per[m])
            rep["h0"][f"{name}_DI-CC_vs_{other}"] = {
                "n_pos": int(len(pos)), "n_neg": int(len(neg)), "n_persons_pos": int(len(set(per[m][y == 1]))),
                "auc": auc(pos, neg), "iid_ci": iid_ci(pos, neg), "cluster_ci": cci, "cluster_valid_resamples": nb}
            print(name, other, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep["h0"][f"{name}_DI-CC_vs_{other}"].items() if k not in ("iid_ci", "cluster_ci")},
                  "iid", np.round(rep["h0"][f"{name}_DI-CC_vs_{other}"]["iid_ci"], 3), "cluster", np.round(cci, 3))

    # --- string-based (NLI-independent) labels for the tampered year
    bios = {json.loads(l)["person"]: json.loads(l) for l in open(config.H3_BIOS_JSONL, encoding="utf-8")}
    def year_label(person, claim):
        b = bios[person]; mo, mp = YEAR.search(b["specific_bio"]), YEAR.search(b["perturbed_bio"])
        if not mo or not mp: return "na"
        yo, yp = mo.group(0), mp.group(0); ys = set(YEAR.findall(claim))
        if yp in ys and yo in ys: return "contrast"
        if yp in ys: return "copy"
        if yo in ys: return "override"
        return "other"
    sl = np.array([year_label(p, t) if c == "perturbed_baseline" else "na" for p, t, c in zip(per, txt, cond)])
    rep["string_label_counts"] = dict(Counter(sl))
    rep["string_vs_nli"] = {k: dict(Counter(lab[sl == k])) for k in ("copy", "override", "contrast", "other")}
    print("string labels", rep["string_label_counts"]); print("string vs nli", rep["string_vs_nli"])

    # --- 3: matched control: override vs faithful copy (same bio variant, same instruction, year-bearing claim)
    m = (sl == "override") | (sl == "copy"); y = (sl[m] == "override").astype(int)
    rep["matched_override_vs_copy"] = {"n_override": int(y.sum()), "n_copy": int((1 - y).sum())}
    for name, s in scores.items():
        if y.sum() >= 2 and (1 - y).sum() >= 2:
            cci, nb = cluster_ci(s[m], y, per[m])
            rep["matched_override_vs_copy"][name] = {"auc": auc(s[m][y == 1], s[m][y == 0]), "iid_ci": iid_ci(s[m][y == 1], s[m][y == 0]), "cluster_ci": cci}
    print("matched override vs copy", json.dumps(rep["matched_override_vs_copy"], default=float)[:600])
    # DI-CC vs legit restricted to the perturbed condition only
    rep["perturbed_only"] = {}
    for name, s in scores.items():
        mm = keep & (cond == "perturbed_baseline") & np.isin(lab, ["DI-CC", "legit"]); yy = (lab[mm] == "DI-CC").astype(int)
        cci, _ = cluster_ci(s[mm], yy, per[mm])
        rep["perturbed_only"][name] = {"n_pos": int(yy.sum()), "n_neg": int((1 - yy).sum()), "auc": auc(s[mm][yy == 1], s[mm][yy == 0]), "cluster_ci": cci}
    # condition-identity check: legit(perturbed) vs legit(vague)
    rep["condition_identity_legit"] = {}
    for name, s in scores.items():
        a_, b_ = s[keep & (lab == "legit") & (cond == "perturbed_baseline")], s[keep & (lab == "legit") & (cond == "vague_baseline")]
        rep["condition_identity_legit"][name] = {"n_perturbed": int(len(a_)), "n_vague": int(len(b_)), "auc_perturbed_vs_vague": auc(a_, b_)}
    print("perturbed only", rep["perturbed_only"]); print("cond identity", rep["condition_identity_legit"])

    # --- 2: NLI detector (needs rerun_nli_scores.jsonl)
    p = R / "rerun_nli_scores.jsonl"
    if p.exists():
        rows = {}
        for l in open(p, encoding="utf-8"):
            r = json.loads(l); rows.setdefault((r["person"], r["condition"], r["claim_text"]), r)
        nli_c = np.array([rows[(a, b, c)]["nli_src"]["contradiction"] if (a, b, c) in rows else np.nan for a, b, c in zip(per, cond, txt)])
        nli_e = np.array([rows[(a, b, c)]["nli_src"]["entailment"] if (a, b, c) in rows else np.nan for a, b, c in zip(per, cond, txt)])
        rep["nli_detector"] = {}
        for other in ("legit", "DI-UE"):
            mm = keep & np.isin(lab, ["DI-CC", other]); yy = (lab[mm] == "DI-CC").astype(int)
            rep["nli_detector"][f"label_defined_DI-CC_vs_{other}"] = {"auc_P_contradiction": auc(nli_c[mm][yy == 1], nli_c[mm][yy == 0]), "note": "circular: labels are argmax of this NLI"}
        mo_ = (sl == "override") | (sl == "copy"); yo = (sl[mo_] == "override").astype(int)
        rep["nli_detector"]["override_vs_copy"] = {"auc_P_contradiction": auc(nli_c[mo_][yo == 1], nli_c[mo_][yo == 0]),
                                                    "auc_1_minus_P_entailment": auc(1 - nli_e[mo_][yo == 1], 1 - nli_e[mo_][yo == 0]),
                                                    "probe_varratio": rep["matched_override_vs_copy"].get("varratio", {}).get("auc"),
                                                    "probe_accuracy": rep["matched_override_vs_copy"].get("accuracy", {}).get("auc")}
        print("NLI detector", json.dumps(rep["nli_detector"], default=float))
    json.dump(rep, open(R / "rerun_analysis_1_2_3.json", "w"), indent=2, default=float)
    print("saved")

if __name__ == "__main__":
    main()
