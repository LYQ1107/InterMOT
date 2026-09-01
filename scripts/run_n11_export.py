#!/usr/bin/env python
"""Export N11 required CSVs / freeze record."""

import csv
import json
from pathlib import Path


ROOT = Path(".")
OUT = ROOT / "outputs/n11"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # official calibration metrics (N10 eval + N11 eval)
    n10 = list(csv.DictReader(open(ROOT / "outputs/n10/eval/train/combined_metrics.csv")))
    n11 = list(csv.DictReader(open(OUT / "eval/train/combined_metrics.csv")))
    n11_extra = list(csv.DictReader(open(OUT / "eval/train_extra/combined_metrics.csv")))
    wanted = {
        "p0": "p0",
        "reid_b0_post": "AUTO-RM b0",
        "pairwise_b0_post": "AUTO-P b0",
        "human_b1_post": "N10 Global B1",
        "human_b2_post": "N10 Global B2",
        "human_b4_post": "N10 Global B4",
        "human_b8_post": "N10 Global B8",
        "local_native0_decay_b1_post": "N11 Local B1",
        "local_native0_decay_b2_post": "N11 Local B2",
        "local_native0_decay_b4_post": "N11 Local B4",
        "local_native0_decay_b8_post": "N11 Local B8",
        "local_native0_evidence_b8_post": "N11 Local-Evidence B8",
        "local_perm_b8_post": "N11 Local-perm B8",
        "local_native0_b8_post": "N11 Local native0 B8",
        "local_decay_memfreeze_b8_post": "N11 Local-Decay memfreeze B8",
    }
    rows = []
    seen = set()
    for src in (n10, n11, n11_extra):
        for r in src:
            if r["method"] in wanted:
                label = wanted[r["method"]]
                if label in seen:
                    continue
                seen.add(label)
                rows.append(
                    {
                        "method": label,
                        "HOTA": r.get("HOTA"),
                        "DetA": r.get("DetA"),
                        "AssA": r.get("AssA"),
                        "MOTA": r.get("MOTA"),
                        "IDF1": r.get("IDF1"),
                        "IDSW": r.get("IDSW"),
                        "Frag": r.get("Frag"),
                        "FP": r.get("FP"),
                        "FN": r.get("FN"),
                    }
                )
    with (OUT / "calibration_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # mechanism summary from collateral analysis
    col = json.loads((OUT / "collateral_analysis_summary.json").read_text())
    with (OUT / "mechanism_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "variant",
                "n_events",
                "same_id_tte_median",
                "same_id_tte_mean",
                "retention_t1",
                "retention_t30",
                "self_changes_per_event",
                "other_changes_per_event",
                "target_err_30_mean",
                "unrelated_err_30_mean",
            ]
        )
        for r in col:
            w.writerow(
                [
                    r["variant"],
                    r["n"],
                    r["same_id_tte"]["median"],
                    r["same_id_tte"]["mean"],
                    r["same_id_retention"].get("t+1"),
                    r["same_id_retention"].get("t+30"),
                    r["self_changes_per_event"],
                    r["other_changes_per_event"],
                    r["target_err_30_mean"],
                    r["unrelated_err_30_mean"],
                ]
            )
    # event-type analysis (computed manually from runs)
    with (OUT / "event_type_analysis.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "event_type", "n", "t+1", "t+3", "t+5", "t+10", "t+30"])
        rows_et = [
            ("global_b8", "TEMPORAL_ID_BREAK", 4, 100.0, 100.0, 100.0, 100.0, 75.0),
            ("global_b8", "TRUE_MISS_NEW", 38, 28.9, 18.4, 18.4, 16.2, 13.9),
            ("global_b8", "RECOVERABLE_MISS short-gap", 38, 68.4, 50.0, 34.2, 33.3, 18.9),
            ("local_decay_b8", "TEMPORAL_ID_BREAK", 9, 55.6, 0.0, 0.0, 0.0, 0.0),
            ("local_decay_b8", "TRUE_MISS_NEW", 38, 28.9, 13.2, 10.5, 5.4, 5.6),
            ("local_decay_b8", "RECOVERABLE_MISS short-gap", 33, 66.7, 48.5, 33.3, 29.0, 18.8),
        ]
        w.writerows(rows_et)
    (OUT / "n11_frozen.json").write_text(
        json.dumps(
            {
                "stage": "N11-FREEZE-RECORD",
                "date": "2026-08-09",
                "final_status": "FAIL_LOCALITY_HYPOTHESIS / FAIL_INTERACTION_SPECIFIC_GAIN",
                "frozen": False,
                "gate_result": "FAIL: spatial scope-v0 has zero effect (global Hungarian already preserves unrelated assignments); temporal decay reduces assignment churn but also reduces same-ID persistence; official AssA/IDF1/IDSW never beat AUTO on calibration",
                "architecture": {
                    "auto_base": "N10 PairwiseMLP (AUTO-P); ReID+motion as AUTO-RM reference",
                    "local_scope": "deterministic Scope-v0: corrected identity + direct conflict identities; non-scope AUTO assignments frozen for scope_frames=10",
                    "temporal_scope": "native_constraint_frames=0, authority_mode decay/evidence (hard=1, decay=8), optional machine-memory freeze in scope",
                },
                "calibration_trackeval": "outputs/n11/eval/train/combined_metrics.csv + outputs/n11/eval/train_extra/combined_metrics.csv",
                "collateral_analysis": "outputs/n11/collateral_analysis.csv",
                "canonical_25": "NOT_RUN (calibration official-metric gate failed)",
                "three_sequence_sanity": "NOT_RUN (gate failed)",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"exports": "OK", "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
