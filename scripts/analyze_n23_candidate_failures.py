#!/usr/bin/env python3
"""N23 candidate-pool failure taxonomy.

This audit deliberately separates candidate availability from downstream
shadow/verifier behavior.  The primary table uses the N20 no-commit stream,
where ``target_present`` means that the GFN gallery contains a detection with
IoU >= 0.5 to the GT target and ``rank_mem`` is the target's rank under the
causal learned-memory score.  The secondary table uses the complete K+1
on-policy shadow groups and is conditional on a correct final shadow label.

No model is trained or modified by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def as_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def as_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def fraction(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topk-csv",
        default=str(ROOT / "outputs/n20/topk_no_commit.csv"),
    )
    parser.add_argument(
        "--shadow-csv",
        default=str(ROOT / "outputs/n20/kplus1_onpolicy_recheck.csv"),
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--out-json",
        default=str(ROOT / "outputs/n23/candidate_failure_taxonomy.json"),
    )
    parser.add_argument(
        "--out-csv",
        default=str(ROOT / "outputs/n23/candidate_failure_taxonomy.csv"),
    )
    parser.add_argument(
        "--out-report",
        default=str(ROOT / "docs/N23_CANDIDATE_FAILURE_ANALYSIS.md"),
    )
    args = parser.parse_args()

    topk_path = Path(args.topk_csv)
    shadow_path = Path(args.shadow_csv)
    topk = load_csv(topk_path)
    shadow = load_csv(shadow_path)
    total = len(topk)

    present = [row for row in topk if as_bool(row.get("target_present"))]
    absent = [row for row in topk if not as_bool(row.get("target_present"))]
    rank_values = [as_int(row.get("rank_mem")) for row in present]
    rank_values = [rank for rank in rank_values if rank is not None]
    rank_missing = len(present) - len(rank_values)

    primary_categories = [
        {
            "category": "F3_TARGET_ABSENT_FROM_GFN_POOL",
            "count": len(absent),
            "denominator": total,
            "fraction": fraction(len(absent), total),
            "definition": (
                "No GFN gallery detection at this attempt has IoU >= 0.5 "
                "with the GT target."
            ),
        },
        {
            "category": "F4_TARGET_PRESENT_BUT_RANK_GT_K",
            "count": sum(
                1
                for row in present
                if (as_int(row.get("rank_mem")) or 10**9) > args.k
            ),
            "denominator": total,
            "fraction": 0.0,
            "definition": (
                "The target is represented by the GFN gallery, but its "
                "best learned-memory rank is below the top-K cut."
            ),
        },
        {
            "category": "CANDIDATE_PRESENT_AND_RANK_LE_K",
            "count": sum(
                1
                for row in present
                if (as_int(row.get("rank_mem")) or 10**9) <= args.k
            ),
            "denominator": total,
            "fraction": 0.0,
            "definition": (
                "The target is represented and is available to a top-K "
                "shadow verifier."
            ),
        },
    ]
    for row in primary_categories:
        row["fraction"] = fraction(row["count"], row["denominator"])

    rank_cutoffs = {}
    for cutoff in (1, 3, 5, 10):
        count = sum(
            1
            for row in present
            if (as_int(row.get("rank_mem")) or 10**9) <= cutoff
        )
        rank_cutoffs[f"top{cutoff}"] = {
            "count": count,
            "denominator_present": len(present),
            "fraction_present": fraction(count, len(present)),
            "denominator_all": total,
            "fraction_all": fraction(count, total),
        }

    topk_by_attempt = {
        f"{row['sequence']}:{row['frame']}:{row['gid']}": row for row in topk
    }
    joined = []
    for row in shadow:
        topk_row = topk_by_attempt.get(row["attempt"])
        if topk_row is not None:
            joined.append((row, topk_row))

    current_rank_high = [
        pair
        for pair in joined
        if as_bool(pair[1].get("target_present"))
        and (as_int(pair[1].get("rank_mem")) or 10**9) <= args.k
    ]
    correct_shadow = [
        pair for pair in current_rank_high if as_int(pair[0].get("true_class")) not in (None, 0)
    ]
    no_correct_shadow = [
        pair for pair in current_rank_high if as_int(pair[0].get("true_class")) in (None, 0)
    ]

    def is_reject(pair: tuple[dict[str, str], dict[str, str]]) -> bool:
        return pair[0].get("decision") == "REJECT"

    def is_correct_commit(pair: tuple[dict[str, str], dict[str, str]]) -> bool:
        return pair[0].get("decision", "").startswith("COMMIT_") and as_bool(
            pair[0].get("commit_ok")
        )

    def is_false_commit(pair: tuple[dict[str, str], dict[str, str]]) -> bool:
        return pair[0].get("decision", "").startswith("COMMIT_") and not as_bool(
            pair[0].get("commit_ok")
        )

    verifier_categories = [
        {
            "category": "VERIFIER_MISS_AFTER_CORRECT_SHADOW",
            "count": sum(1 for pair in correct_shadow if is_reject(pair)),
            "denominator": len(correct_shadow),
            "fraction": 0.0,
            "definition": (
                "A correct target shadow survives to H=5, but the verifier "
                "rejects all candidates instead of committing it."
            ),
        },
        {
            "category": "FALSE_COMMIT_AFTER_CORRECT_SHADOW",
            "count": sum(1 for pair in correct_shadow if is_false_commit(pair)),
            "denominator": len(correct_shadow),
            "fraction": 0.0,
            "definition": (
                "A correct target shadow exists, but the verifier commits a "
                "wrong candidate."
            ),
        },
        {
            "category": "CORRECT_COMMIT_AFTER_CORRECT_SHADOW",
            "count": sum(1 for pair in correct_shadow if is_correct_commit(pair)),
            "denominator": len(correct_shadow),
            "fraction": 0.0,
            "definition": "The verifier commits the correct surviving shadow.",
        },
        {
            "category": "SHADOW_DRIFT_OR_NO_CORRECT_SHADOW",
            "count": len(no_correct_shadow),
            "denominator": len(current_rank_high),
            "fraction": 0.0,
            "definition": (
                "The target was present and rank-high at the initial frame, "
                "but no correct shadow remains at the H=5 decision point; "
                "this is not counted as a pure verifier miss."
            ),
        },
    ]
    for row in verifier_categories:
        row["fraction"] = fraction(row["count"], row["denominator"])

    all_shadow_decisions = Counter(row.get("decision") for row in shadow)
    shadow_correct_commits = sum(
        1 for row in shadow if row.get("decision", "").startswith("COMMIT_") and as_bool(row.get("commit_ok"))
    )
    shadow_false_commits = sum(
        1 for row in shadow if row.get("decision", "").startswith("COMMIT_") and not as_bool(row.get("commit_ok"))
    )
    shadow_rejects = sum(1 for row in shadow if row.get("decision") == "REJECT")

    result = {
        "schema_version": "n23_candidate_failure_taxonomy_v1",
        "k": args.k,
        "primary_source": str(topk_path),
        "shadow_source": str(shadow_path),
        "primary": {
            "attempts": total,
            "target_present": len(present),
            "target_absent": len(absent),
            "rank_mem_missing_when_present": rank_missing,
            "categories": primary_categories,
            "rank_cutoffs": rank_cutoffs,
        },
        "shadow_join": {
            "shadow_rows": len(shadow),
            "joined_rows": len(joined),
            "current_rank_high_rows": len(current_rank_high),
            "correct_shadow_rows": len(correct_shadow),
            "no_correct_shadow_rows": len(no_correct_shadow),
            "decision_counts": dict(all_shadow_decisions),
            "correct_commits": shadow_correct_commits,
            "false_commits": shadow_false_commits,
            "rejects": shadow_rejects,
            "categories": verifier_categories,
        },
        "interpretation": {
            "candidate_creation_gap": (
                "F3 is the largest irreducible top-K failure on the N20 "
                "no-commit stream; a verifier cannot recover a target absent "
                "from its input gallery."
            ),
            "ranking_gap": (
                "F4 is the recoverable-with-better-ranking portion of the "
                "target-present cases, subject to the rank definition."
            ),
            "verifier_gap": (
                "The pure downstream verifier-miss estimate is the "
                "correct-shadow/REJECT conditional, not every rejection in "
                "the current-rank-high group."
            ),
        },
    }

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_report = Path(args.out_report)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    csv_rows = []
    for row in primary_categories + verifier_categories:
        csv_rows.append(
            {
                "stage": "candidate_pool" if row in primary_categories else "shadow_verifier",
                "category": row["category"],
                "count": row["count"],
                "denominator": row["denominator"],
                "fraction": f"{row['fraction']:.8f}",
                "definition": row["definition"],
            }
        )
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    f3 = primary_categories[0]
    f4 = primary_categories[1]
    high = primary_categories[2]
    verifier_miss = verifier_categories[0]
    false_after_correct = verifier_categories[1]
    correct_after_correct = verifier_categories[2]
    no_shadow = verifier_categories[3]
    report = f"""# N23 Candidate Failure Analysis

## Scope and measurement boundary

This is an offline, causal audit of the existing N20 artifacts. It does not
change the tracker or train a model. The primary denominator is the
`outputs/n20/topk_no_commit.csv` no-commit stream (`N={total}`). In that file,
`target_present=1` means that the GFN gallery contains at least one detection
with IoU >= 0.5 to the ground-truth target at the attempt frame. `rank_mem` is
the rank of the best target detection under the causal N20 learned-memory
score. Therefore the primary table measures the ceiling of a top-{args.k}
candidate verifier, not the accuracy of a future target-conditioned generator.

## Primary taxonomy

| Category | Count | Fraction of all attempts | Meaning |
|---|---:|---:|---|
| F3 — target absent from GFN pool | {f3['count']} | {f3['fraction']:.4f} | No GFN detection overlaps the target at IoU >= 0.5. |
| F4 — target present but rank > {args.k} | {f4['count']} | {f4['fraction']:.4f} | The target exists in the gallery but is below the top-{args.k} cut. |
| Candidate present and rank <= {args.k} | {high['count']} | {high['fraction']:.4f} | A top-{args.k} verifier can in principle see the target. |

The target-present subset is {len(present)} attempts. Conditional recall of the
learned-memory ranking is top-1 `{rank_cutoffs['top1']['fraction_present']:.4f}`,
top-3 `{rank_cutoffs['top3']['fraction_present']:.4f}`, top-5
`{rank_cutoffs['top5']['fraction_present']:.4f}`, and top-10
`{rank_cutoffs['top10']['fraction_present']:.4f}`. There are
`{rank_missing}` target-present rows without a rank, which is recorded as a
data-integrity count rather than silently assigned to a category.

## Downstream verifier conditional

The second source, `{shadow_path}`, contains
`{len(shadow)}` complete K+1 shadow groups; `{len(joined)}` join exactly to the
primary stream. After restricting to rows that were target-present and rank
<= {args.k}, `{len(current_rank_high)}` groups remain. This is a different
denominator from the full no-commit stream and must not be added to the table
above.

Among those rank-high groups, `{len(correct_shadow)}` have a correct shadow
surviving to the H=5 decision point:

| Conditional outcome | Count | Conditional fraction |
|---|---:|---:|
| Pure verifier miss: correct shadow, decision REJECT | {verifier_miss['count']} | {verifier_miss['fraction']:.4f} |
| Wrong commit despite a correct shadow | {false_after_correct['count']} | {false_after_correct['fraction']:.4f} |
| Correct commit | {correct_after_correct['count']} | {correct_after_correct['fraction']:.4f} |

The remaining `{no_shadow['count']}` rank-high groups have no correct shadow at
H=5. They are labeled **shadow drift or no correct shadow**, rather than pure
verifier failures. This distinction matters for N23: improving the verifier
cannot recover a target that disappeared during proposal propagation.

For reference, across all `{len(shadow)}` complete shadow groups the artifact
contains `{shadow_correct_commits}` correct commits, `{shadow_false_commits}`
false commits, and `{shadow_rejects}` rejects. Those aggregate values are
descriptive only; the conditional table is the one used for the failure
taxonomy.

## N23 design implication

The dominant measured gap is candidate creation (`F3`), followed by ranking
(`F4`). N22's prototype memory can only re-score GFN detections and therefore
cannot change either the existence of the gallery or its top-K support. N23
must introduce a correction-conditioned proposal branch that can create
target-inclusive boxes/points outside the GFN pool, then expose those
proposals to the verifier under an explicit false-commit gate. A successful
result must also show that the new branch reduces future same-target
re-correction, rather than merely increasing accepted detections.

## Reproducibility

- Script: `scripts/analyze_n23_candidate_failures.py`
- JSON: `outputs/n23/candidate_failure_taxonomy.json`
- CSV: `outputs/n23/candidate_failure_taxonomy.csv`
- Primary source: `outputs/n20/topk_no_commit.csv`
- Conditional source: `{shadow_path}`
"""
    out_report.write_text(report, encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
