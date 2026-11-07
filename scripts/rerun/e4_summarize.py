import json, sys
from pathlib import Path
R = Path(__file__).parent.parent / "rerun_results"
e1 = json.load(open(R / "e1_layer_inference.json")); e4 = json.load(open(R / "e4_null_probes.json")); L = e1["layers"]
out = {}
for k in ("dicc_vs_legit", "override_vs_copy", "cond_identity"):
    rows = []
    for i, l in enumerate(L):
        a = e1[k]["auc"][i]; s = e4[k]["shuf"][i]; r = e4[k]["rand"][i]
        rows.append({"layer": l, "auc": a, "shuf": s, "rand": r,
                     "above_shuf": a > s[2], "below_shuf": a < s[0], "above_rand": a > r[2], "below_rand": a < r[0]})
    out[k] = rows
    print(k, "above shuf:", sum(x["above_shuf"] for x in rows), "below shuf:", sum(x["below_shuf"] for x in rows),
          "above rand:", sum(x["above_rand"] for x in rows), "below rand:", sum(x["below_rand"] for x in rows), "of", len(rows))
    print("  above-shuf layers:", [x["layer"] for x in rows if x["above_shuf"]], " above-rand layers:", [x["layer"] for x in rows if x["above_rand"]])
    print("  band widths (shuf 2.5-97.5) min/max:", min(x["shuf"][2] - x["shuf"][0] for x in rows), max(x["shuf"][2] - x["shuf"][0] for x in rows),
          " rand:", min(x["rand"][2] - x["rand"][0] for x in rows), max(x["rand"][2] - x["rand"][0] for x in rows))
for l in (19, 23, 25, 28):
    i = L.index(l); x = out["override_vs_copy"][i]; print("matched layer", l, round(x["auc"], 3), "shuf", x["shuf"], "rand", x["rand"])
json.dump(out, open(R / "e4_summary.json", "w"), indent=1)
