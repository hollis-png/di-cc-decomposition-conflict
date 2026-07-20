"""Recompute the mistral7b H0 transfer AUC after excluding the 3 DI-CC cases
flagged as likely mislabels during manual spot-check of all 74 DI-CC spans
(2 decomposition-artifact "born in his youth" claims for Genghis Khan and
Pablo Picasso, 1 unrelated-claim mislabel for John Lennon). Checks whether
removing these changes the null/negative H0 result -- run entirely locally
against the already-pulled meta.jsonl, activations npz, and probe, no SSH
round-trip needed for the compute itself.
"""
import json

import numpy as np
from joblib import load
from sklearn.metrics import roc_auc_score

MISLABELED = {
    ("Genghis Khan", "Genghis Khan was born in his youth."),
    ("Pablo Picasso", "Pablo was born in his youth."),
    ("John Lennon", "John Lennon is a British singer-songwriter and peace activist."),
}

meta = [json.loads(l) for l in open("decomp_activations_mistral7b.meta.jsonl", encoding="utf-8")]
data = np.load("decomp_activations_mistral7b.npz", allow_pickle=True)
activations, labels = data["activations"], data["labels"]

assert len(meta) == len(labels), (len(meta), len(labels))

layer = 18
X = activations[:, layer, :]

exclude_idx = {i for i, m in enumerate(meta) if (m["person"], m["claim_text"]) in MISLABELED}
print(f"Excluding {len(exclude_idx)} flagged rows (expected 3)")

keep_mask = np.array([i not in exclude_idx for i in range(len(meta))])
labels_kept = labels[keep_mask]

probe = load("probe_layer18_mistral7b.joblib")
scores = probe.predict_proba(X)[:, 1][keep_mask]

groups = {label: scores[labels_kept == label] for label in ["DI-CC", "legit", "DI-UE"]}
for k, v in groups.items():
    print(f"  {k}: n={len(v)}, mean={v.mean():.4f}")

for pair, (pos, neg) in [("DI-CC_vs_legit", ("DI-CC", "legit")), ("DI-CC_vs_DI-UE", ("DI-CC", "DI-UE"))]:
    y = np.concatenate([np.ones(len(groups[pos])), np.zeros(len(groups[neg]))])
    s = np.concatenate([groups[pos], groups[neg]])
    auc = roc_auc_score(y, s)
    print(f"{pair}: AUC={auc:.4f} (n_pos={len(groups[pos])})")
