"""Generate LaTeX tables and the layer-curve figure for paper_arr.tex from rerun_results JSONs (no numbers typed by hand)."""
import json, statistics as st
from pathlib import Path
import numpy as np
R = Path(__file__).parent / "rerun_results"
e1 = json.load(open(R / "e1_layer_inference.json")); L = e1["layers"]
tags = ["s1", "s2", "s3", "s4", "s5"]; e3 = {t: json.load(open(R / "e3" / f"{t}.json")) for t in tags}
e4 = json.load(open(R / "e4_null_probes.json"))
f = lambda x: f"{x:.2f}"
def cell(k, i): d = e1[k]; return f"{f(d['auc'][i])} [{f(d['ci_lo'][i])},{f(d['ci_hi'][i])}]"
rows = []
for l in (18, 19, 20, 22, 23, 25, 28):
    i = L.index(l)
    sd = lambda k: st.stdev([e3[t][k][i] for t in tags])
    rows.append(f"{l} & {cell('dicc_vs_legit', i)} & {cell('override_vs_copy', i)} & {f(e1['cond_identity']['auc'][i])} & {f(e4['override_vs_copy']['rand'][i][2])} & {sd('dicc_vs_legit'):.3f} / {sd('override_vs_copy'):.3f} \\\\")
Path(__file__).with_name("tab_layers.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")
full = [f"{l} & {f(e1['dicc_vs_legit']['auc'][i])} & {f(e1['override_vs_copy']['auc'][i])} & {f(e1['cond_identity']['auc'][i])} \\\\" for i, l in enumerate(L)]
Path(__file__).with_name("tab_layers_full.tex").write_text("\n".join(full) + "\n\\bottomrule\n")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, axes = plt.subplots(2, 1, figsize=(3.2, 3.9), sharex=True)
for ax, k, lab, c in ((axes[0], "dicc_vs_legit", "DI-CC vs. legit", "#1b6ca8"), (axes[1], "override_vs_copy", "override vs. copy (matched)", "#c4501b")):
    lo = [b[0] for b in e4[k]["rand"]]; hi = [b[2] for b in e4[k]["rand"]]
    ax.fill_between(L, lo, hi, color="#bbbbbb", alpha=0.5, lw=0, label="random-direction 95% band")
    ax.fill_between(L, e1[k]["ci_lo"], e1[k]["ci_hi"], color=c, alpha=0.25, lw=0)
    ax.plot(L, e1[k]["auc"], marker="o", ms=3, lw=1, color=c, label=lab)
    ax.axhline(0.5, color="k", lw=0.6, ls="--"); ax.set_ylabel("AUC"); ax.set_ylim(0, 1.02); ax.legend(fontsize=6, loc="lower right", frameon=False)
axes[1].set_xlabel("layer"); fig.tight_layout(); fig.savefig(Path(__file__).parent / "figures" / "layer_curve.pdf")
print(open(Path(__file__).with_name("tab_layers.tex")).read())
