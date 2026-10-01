#!/usr/bin/env python3
"""Post-hoc P0/P1/P2/P3 taxonomy and R2 wrong-write reclassification."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.n72r20r3_common import (
    DATASET_ROOT,
    IOU_THRESHOLD,
    R2_ASSET_ROOT,
    SEQUENCES,
    best_target_candidate,
    gt_by_frame,
    load_sequence,
    public_assignment_uid,
    read_events,
    read_labels,
    target_boxes_for_frame,
    target_iou_for_uid,
)


def classify_frame(
    *,
    target_gt_present: bool,
    best_iou: float | None,
    best_uid: str | None,
    base_uid: str | None,
    base_iou: float | None,
) -> str:
    if not target_gt_present:
        return "P0_TARGET_ABSENT"
    if best_iou is None or best_iou < IOU_THRESHOLD:
        if base_uid is not None and best_uid is not None and base_uid == best_uid:
            return "P1a_BEST_AVAILABLE_LOCALIZATION"
        return "P1b_WRONG_AVAILABLE_OBJECT"
    if base_iou is not None and base_iou >= IOU_THRESHOLD:
        return "P3_TARGET_AVAILABLE_ASSOCIATION_CORRECT"
    return "P2_TARGET_AVAILABLE_ASSOCIATION_WRONG"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=R2_ASSET_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/N72R20R3"))
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    args = parser.parse_args()

    events = read_events(args.asset_root)
    labels = read_labels(args.asset_root)
    all_rows: list[dict[str, Any]] = []
    sequence_summaries: list[dict[str, Any]] = []
    wrong_rows: list[dict[str, Any]] = []
    for sequence in args.sequences:
        event = events[str(sequence)]
        label = labels[str(sequence)]
        frames, base_rows = load_sequence(args.asset_root, str(sequence))
        gt = gt_by_frame(args.dataset_root / "train" / str(sequence) / "gt" / "gt.txt")
        target_gt_id = int(label["target_gt_id"])
        counts = Counter()
        sequence_rows: list[dict[str, Any]] = []
        c0_correct = 0
        c0_wrong = 0
        for (frame_payload, candidates), base_row in zip(frames, base_rows):
            frame = int(frame_payload["frame"])
            if frame <= int(event["event_frame"]):
                continue
            target_boxes = target_boxes_for_frame(gt, frame, target_gt_id)
            target_present = bool(target_boxes)
            best_uid, best_iou = best_target_candidate(candidates, target_boxes)
            base_uid = public_assignment_uid(base_row["base_assignment"], int(event["target_public_id"]))
            base_iou = target_iou_for_uid(base_uid, candidates, target_boxes)
            taxonomy = classify_frame(
                target_gt_present=target_present,
                best_iou=best_iou,
                best_uid=best_uid,
                base_uid=base_uid,
                base_iou=base_iou,
            )
            row = {
                "stage": "N72R20R3",
                "sequence": str(sequence),
                "frame": frame,
                "target_gt_id": target_gt_id,
                "candidate_count": len(candidates),
                "target_gt_present": target_present,
                "candidate_set_present": bool(target_present and best_iou is not None and best_iou >= IOU_THRESHOLD),
                "best_target_candidate_uid": best_uid,
                "best_target_candidate_iou": best_iou,
                "base_candidate_uid": base_uid,
                "base_candidate_target_iou": base_iou,
                "taxonomy": taxonomy,
                "posthoc_gt_used": True,
                "runtime_future_gt_used": False,
            }
            sequence_rows.append(row)
            all_rows.append(row)
            counts[taxonomy] += 1
            if base_uid is not None and base_iou is not None and base_iou >= IOU_THRESHOLD:
                c0_correct += 1
            elif base_uid is not None:
                c0_wrong += 1
                wrong_rows.append(
                    {
                        **row,
                        "historical_r2_wrong_write": True,
                        "reclassification": {
                            "target_absent": taxonomy == "P0_TARGET_ABSENT",
                            "target_candidate_unavailable": taxonomy.startswith("P1"),
                            "true_candidate_association_error": taxonomy == "P2_TARGET_AVAILABLE_ASSOCIATION_WRONG",
                            "localization_gray_case": taxonomy == "P1a_BEST_AVAILABLE_LOCALIZATION",
                            "insufficient_evidence": False,
                        },
                    }
                )
        sequence_summaries.append(
            {
                "sequence": str(sequence),
                "future_frames": len(sequence_rows),
                "taxonomy_counts": dict(sorted(counts.items())),
                "candidate_set_present": sum(bool(row["candidate_set_present"]) for row in sequence_rows),
                "candidate_set_absent": sum(not bool(row["candidate_set_present"]) for row in sequence_rows),
                "r2_c0_correct_writes": c0_correct,
                "r2_c0_wrong_writes": c0_wrong,
            }
        )

    total_counts = Counter(row["taxonomy"] for row in all_rows)
    total_future = len(all_rows)
    wrong_counts = Counter(row["taxonomy"] for row in wrong_rows)
    total_wrong = len(wrong_rows)
    reclassification = {
        "stage": "N72R20R3",
        "source_stage": "N72R20R2",
        "historical_r2_wrong_write_definition": "C0 accepted base assignment with target candidate IoU < 0.50",
        "historical_r2_wrong_write_count": total_wrong,
        "historical_r2_c0_correct_write_count": sum(item["r2_c0_correct_writes"] for item in sequence_summaries),
        "classification_counts": {
            "P0_target_absent": wrong_counts["P0_TARGET_ABSENT"],
            "P1_target_candidate_unavailable": wrong_counts["P1a_BEST_AVAILABLE_LOCALIZATION"] + wrong_counts["P1b_WRONG_AVAILABLE_OBJECT"],
            "P1a_localization_gray": wrong_counts["P1a_BEST_AVAILABLE_LOCALIZATION"],
            "P1b_wrong_available_object": wrong_counts["P1b_WRONG_AVAILABLE_OBJECT"],
            "P2_true_candidate_association_error": wrong_counts["P2_TARGET_AVAILABLE_ASSOCIATION_WRONG"],
            "other_insufficient_evidence": 0,
        },
        "percent_of_historical_wrong_writes": {
            key: (None if total_wrong == 0 else value / total_wrong)
            for key, value in {
                "P0_target_absent": wrong_counts["P0_TARGET_ABSENT"],
                "P1_target_candidate_unavailable": wrong_counts["P1a_BEST_AVAILABLE_LOCALIZATION"] + wrong_counts["P1b_WRONG_AVAILABLE_OBJECT"],
                "P2_true_candidate_association_error": wrong_counts["P2_TARGET_AVAILABLE_ASSOCIATION_WRONG"],
                "other_insufficient_evidence": 0,
            }.items()
        },
        "r2_failure_unchanged": True,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "wrong_write_rows": wrong_rows,
    }
    taxonomy = {
        "stage": "N72R20R3",
        "source_stage": "N72R20R2",
        "status": "PASS_N72R20R3_FAILURE_TAXONOMY",
        "iou_threshold": IOU_THRESHOLD,
        "future_frames": total_future,
        "taxonomy_counts": dict(sorted(total_counts.items())),
        "p0_count": total_counts["P0_TARGET_ABSENT"],
        "p1_count": total_counts["P1a_BEST_AVAILABLE_LOCALIZATION"] + total_counts["P1b_WRONG_AVAILABLE_OBJECT"],
        "p1a_count": total_counts["P1a_BEST_AVAILABLE_LOCALIZATION"],
        "p1b_count": total_counts["P1b_WRONG_AVAILABLE_OBJECT"],
        "p2_count": total_counts["P2_TARGET_AVAILABLE_ASSOCIATION_WRONG"],
        "p3_count": total_counts["P3_TARGET_AVAILABLE_ASSOCIATION_CORRECT"],
        "candidate_set_present_count": sum(bool(row["candidate_set_present"]) for row in all_rows),
        "candidate_set_absent_count": sum(not bool(row["candidate_set_present"]) for row in all_rows),
        "sequence_summaries": sequence_summaries,
        "r2_wrong_write_reclassification_reference": "outputs/N72R20R3/r2_wrong_write_reclassification.json",
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "failure_taxonomy.json").write_text(json.dumps(taxonomy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "r2_wrong_write_reclassification.json").write_text(json.dumps(reclassification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# N72R20R3 failure taxonomy",
        "",
        "The historical R2 wrong-write definition is preserved: an accepted base assignment is wrong when target-candidate IoU is below 0.50.",
        "",
        f"Future frames: {total_future}; P0={taxonomy['p0_count']}, P1={taxonomy['p1_count']} (P1a={taxonomy['p1a_count']}, P1b={taxonomy['p1b_count']}), P2={taxonomy['p2_count']}, P3={taxonomy['p3_count']}.",
        "",
        "| Sequence | Future | P0 | P1a | P1b | P2 | P3 | Candidate-set present |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sequence_summaries:
        c = item["taxonomy_counts"]
        lines.append(
            f"| {item['sequence']} | {item['future_frames']} | {c.get('P0_TARGET_ABSENT', 0)} | {c.get('P1a_BEST_AVAILABLE_LOCALIZATION', 0)} | {c.get('P1b_WRONG_AVAILABLE_OBJECT', 0)} | {c.get('P2_TARGET_AVAILABLE_ASSOCIATION_WRONG', 0)} | {c.get('P3_TARGET_AVAILABLE_ASSOCIATION_CORRECT', 0)} | {item['candidate_set_present']} |"
        )
    lines.extend(
        [
            "",
            f"Historical R2 C0 wrong writes: {total_wrong} (the R2 decision remains FAIL_RUNTIME_MEMORY_COMMIT).",
            "",
            "| R2 wrong-write category | Count | Fraction |",
            "|---|---:|---:|",
        ]
    )
    for key, count in reclassification["classification_counts"].items():
        fraction = 0.0 if total_wrong == 0 else count / total_wrong
        lines.append(f"| {key} | {count} | {fraction:.6f} |")
    lines.extend(
        [
            "",
            "P1a is a geometry/localization diagnostic: the base-selected candidate is the best available candidate but remains below the 0.50 target-IoU threshold. P1b records that the base selection is not the best available candidate or is absent. These labels are post-hoc only and never enter runtime decisions.",
        ]
    )
    (args.output_dir / "failure_taxonomy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": taxonomy["status"], "future_frames": total_future, "taxonomy_counts": dict(sorted(total_counts.items())), "r2_wrong_writes": total_wrong}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
