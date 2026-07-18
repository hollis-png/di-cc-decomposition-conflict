"""Stage 37 (CAD alpha sensitivity, evaluation): compute the
same paired statistical comparison as stage 27 (McNemar's test for the
primary DI-CC-suppression effect, aggregate label-rate shift + leftover-
pronoun check for the vague-bio side effect), but for alpha in {0.5, 1.5}
instead of the default alpha=1.0.

The perturbed_baseline / vague_baseline conditions don't depend on alpha, so
they're reused unchanged from the existing h3_labeled_spans.jsonl (stage 26
output) and paired against each new alpha's _cad conditions from stage 36.
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from statsmodels.stats.contingency_tables import mcnemar

import config

H3_ALPHA_SENSITIVITY_LABELED_JSONL = config.DATA_PROCESSED / "h3_alpha_sensitivity_labeled.jsonl"
H3_ALPHA_SENSITIVITY_REPORT = config.OUTPUTS_REPORTS / "h3_alpha_sensitivity.json"

_PRONOUN_RE = re.compile(r"\b(he|she|it|his|her|him)\b", re.IGNORECASE)

ALPHAS = [0.5, 1.5]


def load_baseline_records():
    """perturbed_baseline / vague_baseline spans, alpha-independent, from the
    existing alpha=1.0 pipeline's labeled output."""
    with open(config.H3_LABELED_JSONL, encoding="utf-8") as f:
        return [json.loads(line) for line in f if json.loads(line)["condition"] in ("perturbed_baseline", "vague_baseline")]


def load_cad_records(alpha):
    with open(H3_ALPHA_SENSITIVITY_LABELED_JSONL, encoding="utf-8") as f:
        return [json.loads(line) for line in f if json.loads(line)["alpha"] == alpha]


def primary_effect(baseline_records, cad_records):
    by_person_condition: dict[tuple[str, str], list[dict]] = {}
    for rec in baseline_records:
        if rec["bio_variant"] != "perturbed":
            continue
        by_person_condition.setdefault((rec["person"], rec["condition"]), []).append(rec)
    for rec in cad_records:
        if rec["bio_variant"] != "perturbed":
            continue
        by_person_condition.setdefault((rec["person"], rec["condition"]), []).append(rec)

    people = sorted({p for (p, c) in by_person_condition if c == "perturbed_baseline"}
                     & {p for (p, c) in by_person_condition if c == "perturbed_cad"})

    both_yes = both_no = only_baseline = only_cad = 0
    for person in people:
        baseline_spans = by_person_condition.get((person, "perturbed_baseline"), [])
        cad_spans = by_person_condition.get((person, "perturbed_cad"), [])
        baseline_has_dicc = any(s["label"] == "DI-CC" for s in baseline_spans)
        cad_has_dicc = any(s["label"] == "DI-CC" for s in cad_spans)
        if baseline_has_dicc and cad_has_dicc:
            both_yes += 1
        elif baseline_has_dicc and not cad_has_dicc:
            only_baseline += 1
        elif cad_has_dicc and not baseline_has_dicc:
            only_cad += 1
        else:
            both_no += 1

    table = [[both_yes, only_baseline], [only_cad, both_no]]
    result = mcnemar(table, exact=True)

    def rate(condition):
        spans = [s for (p, c), lst in by_person_condition.items() if c == condition for s in lst]
        n_dicc = sum(1 for s in spans if s["label"] == "DI-CC")
        return n_dicc, len(spans), n_dicc / len(spans) if spans else 0.0

    baseline_dicc, baseline_total, baseline_rate = rate("perturbed_baseline")
    cad_dicc, cad_total, cad_rate = rate("perturbed_cad")

    return {
        "n_entities_paired": len(people),
        "entity_level_2x2": {
            "both_flagged_DI-CC": both_yes,
            "only_baseline_flagged": only_baseline,
            "only_cad_flagged": only_cad,
            "neither_flagged": both_no,
        },
        "mcnemar_statistic": float(result.statistic),
        "mcnemar_pvalue": float(result.pvalue),
        "baseline_DI-CC_rate": {"n_DI-CC": baseline_dicc, "n_total_subclaims": baseline_total, "rate": baseline_rate},
        "cad_DI-CC_rate": {"n_DI-CC": cad_dicc, "n_total_subclaims": cad_total, "rate": cad_rate},
    }


def side_effect(baseline_records, cad_records):
    by_condition: dict[str, list[dict]] = {}
    for rec in baseline_records:
        if rec["bio_variant"] == "vague":
            by_condition.setdefault(rec["condition"], []).append(rec)
    for rec in cad_records:
        if rec["bio_variant"] == "vague":
            by_condition.setdefault(rec["condition"], []).append(rec)

    def summarize(condition):
        spans = by_condition.get(condition, [])
        counts = {"legit": 0, "DI-CC": 0, "DI-UE": 0, "excluded": 0}
        for s in spans:
            counts[s["label"]] += 1
        total = len(spans)
        raw_texts = {s["raw_decomposition"] for s in spans}
        n_with_leftover_pronoun = sum(1 for t in raw_texts if _PRONOUN_RE.search(t))
        return {
            "label_counts": counts,
            "total_subclaims": total,
            "n_unique_decompositions": len(raw_texts),
            "n_decompositions_with_leftover_pronoun": n_with_leftover_pronoun,
            "leftover_pronoun_rate": n_with_leftover_pronoun / len(raw_texts) if raw_texts else 0.0,
        }

    return {
        "vague_baseline": summarize("vague_baseline"),
        "vague_cad": summarize("vague_cad"),
    }


def main():
    baseline_records = load_baseline_records()
    print(f"Loaded {len(baseline_records)} alpha-independent baseline spans")

    reports = {}
    for alpha in ALPHAS:
        cad_records = load_cad_records(alpha)
        print(f"\n### alpha={alpha}: loaded {len(cad_records)} cad-condition labeled spans")

        primary = primary_effect(baseline_records, cad_records)
        side = side_effect(baseline_records, cad_records)
        reports[str(alpha)] = {"alpha": alpha, "primary_effect": primary, "side_effect": side}

        print(f"=== Primary effect (perturbed bio, DI-CC rate), alpha={alpha} ===")
        print(f"Paired entities: {primary['n_entities_paired']}")
        print(f"2x2 table: {primary['entity_level_2x2']}")
        print(f"McNemar statistic={primary['mcnemar_statistic']:.4f}, p={primary['mcnemar_pvalue']:.6f}")
        print(f"Baseline DI-CC rate: {primary['baseline_DI-CC_rate']}")
        print(f"CAD DI-CC rate: {primary['cad_DI-CC_rate']}")

        print(f"=== Side effect (vague bio, coreference/elaboration damage), alpha={alpha} ===")
        print(f"Baseline: {side['vague_baseline']}")
        print(f"CAD: {side['vague_cad']}")

    with open(H3_ALPHA_SENSITIVITY_REPORT, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {H3_ALPHA_SENSITIVITY_REPORT}")


if __name__ == "__main__":
    main()
