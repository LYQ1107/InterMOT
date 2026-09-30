#!/usr/bin/env python3
"""Post-hoc contamination and learned-memory write audit for N72R20R1.

The runtime tape is already sealed.  This audit may read DanceTrack GT only
to classify the recorded machine writes after the fact; it never changes a
runtime assignment or memory state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from scripts.n72r20r1_build_base_score_tape import load_candidate_frames, read_json, read_zstd_jsonl


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R1_assets")
DATASET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R16_assets/dataset")
SEQUENCES = ("dancetrack0001", "dancetrack0002")
VARIANTS = (
    "FRESH_BASELINE_B0",
    "FRESH_TRUSTED_RELATIVE_STATE",
    "FRESH_LEARNED_IDENTITY_TRUSTED",
)
CHECKPOINT_SHA = "94ba9c44e177b254ee54984195e839f37c78a648c010ffc723dd3f1d553976a2"
ENCODER_SHA = "2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154"
IOU_THRESHOLD = 0.50
REQUIRED_WRITE_FIELDS = (
    "sequence",
    "frame",
    "public_id",
    "candidate_uid",
    "candidate_feature_hash",
    "previous_state_hash",
    "new_state_hash",
    "human_anchor_hash",
    "checkpoint_sha256",
    "encoder_sha256",
    "runtime_future_gt_used",
    "consensus",
    "base_assignment",
    "treatment_assignment",
)


def iou(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def feature_hash(value: Any) -> str:
    return hashlib.sha256(np.asarray(value, dtype=np.float32).tobytes()).hexdigest()


def gt_by_frame(path: Path) -> dict[int, list[tuple[int, list[float]]]]:
    output: dict[int, list[tuple[int, list[float]]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [item.strip() for item in line.split(",")]
        if len(values) < 6:
            continue
        frame = int(float(values[0])) - 1
        track_id = int(float(values[1]))
        x, y, width, height = (float(item) for item in values[2:6])
        output.setdefault(frame, []).append((track_id, [x, y, x + width, y + height]))
    return output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_runtime_replay(sequence: str, output_root: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    rows_by_variant = {
        variant: read_zstd_jsonl(output_root / sequence / variant / "runtime_frames.jsonl.zst")
        for variant in VARIANTS
    }
    count = len(rows_by_variant[VARIANTS[0]])
    checks = {
        "variant_frame_counts_equal": all(len(rows) == count for rows in rows_by_variant.values()),
        "candidate_axes_identical": True,
        "base_matrix_hashes_identical": True,
        "state_axes_immutable": True,
        "public_axes_immutable": True,
        "runtime_gt_flags_clean": True,
        "e2_assignment_changes_zero": True,
        "e2_none_changes_zero": True,
        "event_frame_memory_read_false": True,
        "event_frame_present": False,
    }
    for frame_index in range(count):
        reference = rows_by_variant[VARIANTS[0]][frame_index]
        for variant in VARIANTS[1:]:
            current = rows_by_variant[variant][frame_index]
            checks["candidate_axes_identical"] &= current.get("candidate_uid_axis") == reference.get("candidate_uid_axis")
            checks["base_matrix_hashes_identical"] &= current.get("base_score_matrix_sha256") == reference.get("base_score_matrix_sha256")
            checks["state_axes_immutable"] &= current.get("association_state_axis") == reference.get("association_state_axis")
            checks["public_axes_immutable"] &= current.get("public_id_axis") == reference.get("public_id_axis")
        for variant in VARIANTS:
            row = rows_by_variant[variant][frame_index]
            checks["runtime_gt_flags_clean"] &= (
                row.get("runtime_future_gt_used") is False
                and row.get("runtime_gt_read") is False
                and row.get("posthoc_gt_used") is False
            )
            if int(row.get("frame", -1)) == int(event["event_frame"]):
                checks["event_frame_present"] = True
                checks["event_frame_memory_read_false"] &= row.get("memory_read") is False
        e2 = rows_by_variant["FRESH_LEARNED_IDENTITY_TRUSTED"][frame_index]
        delta = e2.get("assignment_delta", {})
        checks["e2_assignment_changes_zero"] &= int(delta.get("changed_public_count", 0)) == 0
        checks["e2_none_changes_zero"] &= int(delta.get("candidate_none_change_count", 0)) == 0
    return {"sequence": sequence, "frame_count": count, "checks": checks, "all_checks_pass": all(checks.values())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=ASSET_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/N72R20R1/contamination_audit.json")
    parser.add_argument("--replay-root", type=Path, default=ROOT / "outputs/N72R20R1/causal_replay/train_smoke")
    args = parser.parse_args()
    asset_root = args.asset_root.resolve()
    event_payload = read_json(asset_root / "interaction_events.json")
    label_payload = read_json(asset_root / "posthoc_event_labels.json")
    fresh_manifest = read_json(ROOT / "outputs/N72R20R1/fresh_tape_manifest.json")
    replay_manifest = read_json(ROOT / "outputs/N72R20R1/causal_replay/train_smoke_replay_manifest.json")
    events = {str(item["sequence"]): dict(item) for item in event_payload["events"]}
    labels = {str(item["sequence"]): dict(item) for item in label_payload["labels"]}
    writes = read_jsonl(ROOT / "outputs/N72R20R1/memory_write_audit.jsonl")
    candidate_features_by_uid: dict[str, str] = {}
    candidate_frames_by_sequence: dict[str, list[tuple[int, list[dict[str, Any]]]]] = {}
    for sequence in args.sequences:
        frames = load_candidate_frames(asset_root, str(sequence))
        candidate_frames_by_sequence[str(sequence)] = frames
        for frame_payload, candidates in frames:
            for candidate in candidates:
                uid = str(candidate["candidate_uid"])
                loaded_feature_hash = feature_hash(candidate["feature"])
                if uid in candidate_features_by_uid and candidate_features_by_uid[uid] != loaded_feature_hash:
                    raise ValueError(f"candidate UID has conflicting loaded feature hashes: {uid}")
                candidate_features_by_uid[uid] = loaded_feature_hash
    write_checks = {
        "required_write_fields_complete": all(all(field in item for field in REQUIRED_WRITE_FIELDS) for item in writes),
        "all_variant_e2": all(item.get("variant") == "FRESH_LEARNED_IDENTITY_TRUSTED" for item in writes),
        "all_consensus": all(item.get("consensus") is True for item in writes),
        "all_non_none": all(item.get("base_assignment", {}).get("candidate_uid") not in (None, "", "None") and item.get("treatment_assignment", {}).get("candidate_uid") not in (None, "", "None") for item in writes),
        "base_treatment_uid_equal": all(item.get("base_assignment", {}).get("candidate_uid") == item.get("treatment_assignment", {}).get("candidate_uid") for item in writes),
        "base_treatment_public_equal": all(item.get("base_assignment", {}).get("public_id") == item.get("treatment_assignment", {}).get("public_id") for item in writes),
        "no_none_assignment_status": all(item.get("base_assignment", {}).get("status") == "ASSIGNED_CANDIDATE" and item.get("treatment_assignment", {}).get("status") == "ASSIGNED_CANDIDATE" for item in writes),
        "runtime_gt_flags_clean": all(item.get("runtime_future_gt_used") is False for item in writes),
        "checkpoint_lineage_fixed": all(item.get("checkpoint_sha256") == CHECKPOINT_SHA for item in writes),
        "encoder_lineage_fixed": all(item.get("encoder_sha256") == ENCODER_SHA for item in writes),
        "candidate_feature_lineage_fixed": all(candidate_features_by_uid.get(str(item.get("candidate_uid"))) == item.get("candidate_feature_hash") for item in writes),
        "unique_sequence_public_frame": len({(item.get("sequence"), item.get("public_id"), item.get("frame")) for item in writes}) == len(writes),
    }
    per_sequence: dict[str, Any] = {}
    wrong_writes: list[dict[str, Any]] = []
    for sequence in args.sequences:
        sequence = str(sequence)
        event = events[sequence]
        label = labels[sequence]
        candidate_frames = candidate_frames_by_sequence[sequence]
        boxes_by_uid = {
            str(candidate["candidate_uid"]): candidate["box_xyxy"]
            for _, candidates in candidate_frames
            for candidate in candidates
        }
        gt = gt_by_frame(args.dataset_root.resolve() / "train" / sequence / "gt" / "gt.txt")
        sequence_writes = [item for item in writes if str(item.get("sequence")) == sequence]
        anchor_hashes = sorted({str(item.get("human_anchor_hash")) for item in sequence_writes})
        expected_future_frames = {
            int(frame_payload["frame"])
            for frame_payload, _ in candidate_frames
            if int(frame_payload["frame"]) > int(event["event_frame"])
        }
        observed_write_frames = {int(item["frame"]) for item in sequence_writes}
        ious: list[float] = []
        for item in sequence_writes:
            uid = str(item.get("candidate_uid"))
            frame = int(item["frame"])
            candidate_box = boxes_by_uid.get(uid)
            target_boxes = [box for track_id, box in gt.get(frame, []) if int(track_id) == int(label["target_gt_id"])]
            value = max((iou(candidate_box, box) for box in target_boxes), default=0.0) if candidate_box is not None else 0.0
            ious.append(float(value))
            if value < IOU_THRESHOLD:
                wrong_writes.append(
                    {
                        "sequence": sequence,
                        "frame": frame,
                        "public_id": item.get("public_id"),
                        "candidate_uid": uid,
                        "target_gt_id": int(label["target_gt_id"]),
                        "target_iou": float(value),
                        "reason": "posthoc_target_iou_below_0.50",
                    }
                )
        expected_anchor_hash = str(event["human_anchor_sha256"])
        per_sequence[sequence] = {
            "write_count": len(sequence_writes),
            "expected_future_frames": len(expected_future_frames),
            "expected_future_frame_axis_matches_writes": observed_write_frames == expected_future_frames,
            "human_anchor_hashes": anchor_hashes,
            "human_anchor_hash_matches_event": anchor_hashes == [expected_anchor_hash],
            "target_iou_min": None if not ious else float(min(ious)),
            "target_iou_mean": None if not ious else float(np.mean(ious)),
            "wrong_memory_write_count": sum(1 for item in wrong_writes if item["sequence"] == sequence),
        }
    replay_checks = [check_runtime_replay(str(sequence), args.replay_root.resolve(), events[str(sequence)]) for sequence in args.sequences]
    write_checks["anchor_hash_immutable_per_sequence"] = all(item["human_anchor_hash_matches_event"] for item in per_sequence.values())
    write_checks["expected_write_count"] = len(writes) == sum(int(item["expected_future_frames"]) for item in per_sequence.values())
    write_checks["future_write_frame_axes_exact"] = all(item["expected_future_frame_axis_matches_writes"] for item in per_sequence.values())
    write_checks["wrong_memory_write_count_zero"] = len(wrong_writes) == 0
    runtime_checks_pass = all(item["all_checks_pass"] for item in replay_checks)
    manifest_checks = {
        "fresh_tape_manifest_sealed": fresh_manifest.get("fresh_tape_sealed") is True,
        "fresh_manifest_runtime_gt_clean": fresh_manifest.get("runtime_future_gt_used") is False,
        "historical_n72r15_not_used": fresh_manifest.get("historical_n72r15_used") is False and fresh_manifest.get("historical_n72r15_reproduction") is False,
        "replay_manifest_runtime_gt_clean": replay_manifest.get("runtime_future_gt_used") is False,
        "replay_manifest_historical_n72r15_not_used": replay_manifest.get("historical_n72r15_used") is False,
    }
    contamination_pass = all(write_checks.values()) and all(manifest_checks.values()) and runtime_checks_pass
    payload = {
        "stage": "N72R20R1",
        "status": "PASS_N72R20R1_CONTAMINATION_AUDIT" if contamination_pass else "FAIL_MEMORY_CONTAMINATION",
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "gt_role": "posthoc_classification_of_recorded_writes_only",
        "wrong_write_definition": "candidate box IoU with the labeled target GT box is below 0.50",
        "posthoc_target_iou_threshold": IOU_THRESHOLD,
        "write_count": len(writes),
        "write_checks": write_checks,
        "manifest_checks": manifest_checks,
        "per_sequence": per_sequence,
        "posthoc_wrong_memory_writes": wrong_writes,
        "replay_checks": replay_checks,
        "fresh_tape_manifest_sealed": bool(fresh_manifest.get("fresh_tape_sealed")),
        "historical_n72r15_used": bool(fresh_manifest.get("historical_n72r15_used")),
        "final_decision_compatible": "FAIL_LEARNED_MEMORY_DECISION_INACTIVE" if contamination_pass else "FAIL_MEMORY_CONTAMINATION",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "write_count": len(writes), "wrong_memory_write_count": len(wrong_writes)}, sort_keys=True))
    return 0 if contamination_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
