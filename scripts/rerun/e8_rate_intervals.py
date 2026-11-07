"""E8: per-condition DI-CC rate with entity-clustered bootstrap CI and exact McNemar vs baseline
for ALL instruction conditions. Usage: e8_rate_intervals.py MAIN_SPANS.jsonl E6_SPANS.jsonl OUT.json
MAIN_SPANS has baseline/faithful/sham/CAD/knowledge-filling; E6_SPANS has perturbed_nopron/perturbed_factscore."""
import json, sys
from collections import Counter, defaultdict
import numpy as np
from math import comb


def binom_two_sided(k, n):  # exact McNemar: two-sided binomial(0.5) on discordant pairs, no scipy needed
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n)


main, e6, out = sys.argv[1:4]
spans = [json.loads(l) for f in (main, e6) for l in open(f, encoding="utf-8")]
for f in sys.argv[4:]:  # extra span files, e.g. the knowledge-filling run (code name perturbed_neutral in ~/dicc-e5)
    for l in open(f, encoding="utf-8"):
        s = json.loads(l)
        if s["condition"] == "perturbed_neutral":
            s["condition"] = "perturbed_knowfill"
            spans.append(s)
by = defaultdict(lambda: defaultdict(lambda: np.zeros(2)))  # cond -> person -> [n_claims, n_dicc]
for s in spans:
    c = s["condition"]
    if not c.startswith("perturbed"):
        continue
    by[c][s["person"]] += [1, s["label"] == "DI-CC"]
persons = sorted(by["perturbed_baseline"])
rng = np.random.RandomState(0)
rep = {}
base_inc = np.array([by["perturbed_baseline"][p][1] > 0 for p in persons])
for c, d in sorted(by.items()):
    arr = np.array([d[p] if p in d else np.zeros(2) for p in persons])
    n, k = arr.sum(0)
    boots = []
    for _ in range(5000):
        i = rng.randint(0, len(persons), len(persons))
        s = arr[i].sum(0)
        boots.append(s[1] / s[0] if s[0] else np.nan)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    inc = arr[:, 1] > 0
    r = {"claims": int(n), "dicc": int(k), "rate_pct": 100 * k / n, "ci95_pct": [100 * lo, 100 * hi], "entities_with": int(inc.sum())}
    if c != "perturbed_baseline":
        b01 = int((base_inc & ~inc).sum()); b10 = int((~base_inc & inc).sum())
        r["mcnemar_exact_p"] = binom_two_sided(min(b01, b10), b01 + b10) if b01 + b10 else 1.0
        r["baseline_only"], r["cond_only"] = b01, b10
    rep[c] = r
json.dump(rep, open(out, "w"), indent=1)
for c, r in rep.items():
    print(c, r)
