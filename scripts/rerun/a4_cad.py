"""Item 4: CAD vs instruction-matched 'faithful' prompt vs sham vs baseline, plus a
matched-output random-drop baseline. Uses decompositions + labels from the rerun."""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, str(Path.cwd()))
import config
from statsmodels.stats.contingency_tables import mcnemar
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
rng = np.random.RandomState(42)
bios = {json.loads(l)["person"]: json.loads(l) for l in open(config.H3_BIOS_JSONL, encoding="utf-8")}
recs = [json.loads(l) for l in open(config.H3_DECOMPOSITIONS_JSONL, encoding="utf-8")]
spans = [json.loads(l) for l in open(config.H3_LABELED_JSONL, encoding="utf-8")]
conds = ["perturbed_baseline", "perturbed_faithful", "perturbed_sham", "perturbed_cad"]
def ystr(p, claim):
    b = bios[p]; mo, mp = YEAR.search(b["specific_bio"]), YEAR.search(b["perturbed_bio"])
    if not (mo and mp): return "na"
    ys = set(YEAR.findall(claim)); yo, yp = mo.group(0), mp.group(0)
    return "contrast" if (yo in ys and yp in ys) else "copy" if yp in ys else "override" if yo in ys else "other"
persons = sorted({r["person"] for r in recs if r["condition"] == "perturbed_baseline"})
by = {c: defaultdict(lambda: Counter()) for c in conds}
nclaims = {c: defaultdict(int) for c in conds}; failed = {c: set() for c in conds}; ystat = {c: Counter() for c in conds}
for r in recs:
    if r["condition"] in conds and not r["subclaims"]: failed[r["condition"]].add(r["person"])
for s in spans:
    c = s["condition"]
    if c in conds:
        by[c][s["person"]][s["label"]] += 1; nclaims[c][s["person"]] += 1; ystat[c][ystr(s["person"], s["claim_text"])] += 1
rep = {}
for c in conds:
    n = sum(nclaims[c].values()); dicc = sum(by[c][p]["DI-CC"] for p in persons)
    ent = [by[c][p]["DI-CC"] > 0 for p in persons]
    rep[c] = {"entities": len(persons), "parse_fail_entities": len(failed[c]), "claims": n, "DI-CC": int(dicc),
              "DI-CC_per_claim": dicc / n if n else None, "entities_with_DICC": int(sum(ent)),
              "labels": dict(sum((by[c][p] for p in persons), Counter())), "year_string_labels": dict(ystat[c])}
base = np.array([by["perturbed_baseline"][p]["DI-CC"] for p in persons])
for c in conds[1:]:
    x = np.array([by[c][p]["DI-CC"] for p in persons])
    b01 = int(((base > 0) & (x == 0)).sum()); b10 = int(((base == 0) & (x > 0)).sum())
    mc = mcnemar([[int(((base == 0) & (x == 0)).sum()), b10], [b01, int(((base > 0) & (x > 0)).sum())]], exact=True)
    try: w = float(wilcoxon(base, x, zero_method="zsplit").pvalue)
    except Exception: w = None
    rep[c]["vs_baseline"] = {"entity_incidence_baseline_only": b01, "entity_incidence_cond_only": b10, "mcnemar_exact_p": float(mc.pvalue), "wilcoxon_counts_p": w}
# matched-output random drop: drop each baseline claim with prob = CAD's output loss
f = 1 - rep["perturbed_cad"]["claims"] / rep["perturbed_baseline"]["claims"]
claims_by_p = defaultdict(list)
for s in spans:
    if s["condition"] == "perturbed_baseline": claims_by_p[s["person"]].append(s["label"] == "DI-CC")
inc = []; rate = []
for _ in range(1000):
    kept = {p: [v for v in claims_by_p[p] if rng.rand() > f] for p in persons}
    inc.append(np.mean([any(kept[p]) for p in persons])); rate.append(sum(sum(v) for v in kept.values()) / max(1, sum(len(v) for v in kept.values())))
rep["random_drop_matched"] = {"drop_fraction": f, "entity_incidence_mean": float(np.mean(inc)), "entity_incidence_95": [float(np.percentile(inc, 2.5)), float(np.percentile(inc, 97.5))],
                              "per_claim_rate_mean": float(np.mean(rate)), "baseline_entity_incidence": rep["perturbed_baseline"]["entities_with_DICC"] / len(persons),
                              "cad_entity_incidence": rep["perturbed_cad"]["entities_with_DICC"] / len(persons)}
json.dump(rep, open(config.OUTPUTS_REPORTS / "rerun_analysis_4.json", "w"), indent=2, default=float)
for c in conds: print(c, {k: v for k, v in rep[c].items() if k not in ("labels", "year_string_labels")}, rep[c]["labels"], rep[c]["year_string_labels"])
print(rep["random_drop_matched"])
