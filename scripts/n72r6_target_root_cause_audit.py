#!/usr/bin/env python3
"""Posthoc diagnosis for the first N72R6 C1 failure.

This is deliberately separate from the runtime replay.  It opens GT only to
classify target-session absence/drift and protected regressions after the
sealed C0/C1 artifacts have passed their runtime audit.
"""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n72r6_effect_scoring import (  # noqa: E402
    EVENT_MANIFEST,
    GT_ROOT,
    HORIZON,
    IOU_THRESHOLD,
    PRIVATE_ROOT,
    REPLAY_ROOT,
    STAGE08,
    box_iou,
    _effective_public,
    _gt,
    _target_frame,
)
from sam3_intermot.association.branch_public_replay import (  # noqa: E402
    atomic_json,
    now_utc,
    read_json,
    read_jsonl,
    sha256_file,
)


# Do not inherit the effect module's historical default (attempt_4).  This
# audit is intentionally tied to the replay root and effect artifact supplied
# on the command line, while retaining the shadow-dedup attempt as the
# default for reproducibility.
REPLAY_ROOT = ROOT / "outputs/N72R6/public_replay/attempt_5"
EFFECT = ROOT / "outputs/N72R6/ccam_paired_replay_results.json"
OUT = ROOT / "outputs/N72R6/target_root_cause_audit.json"
STATUS = ROOT / "outputs/N72R6/stage_07_status.json"


def _load_rows(path: str) -> dict[int, dict[str, Any]]:
    rows = read_jsonl(Path(path))
    return {int(row["frame"]): row for row in rows}


def main() -> int:
    global REPLAY_ROOT, EFFECT, OUT, STATUS
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-root", default=str(REPLAY_ROOT))
    parser.add_argument("--effect", default=str(EFFECT))
    parser.add_argument("--output", default=str(OUT))
    parser.add_argument("--status-output", default=str(STATUS))
    args = parser.parse_args()
    REPLAY_ROOT = Path(args.replay_root)
    if not REPLAY_ROOT.is_absolute():
        REPLAY_ROOT = ROOT / REPLAY_ROOT
    EFFECT = Path(args.effect)
    if not EFFECT.is_absolute():
        EFFECT = ROOT / EFFECT
    OUT = Path(args.output)
    if not OUT.is_absolute():
        OUT = ROOT / OUT
    STATUS = Path(args.status_output)
    if not STATUS.is_absolute():
        STATUS = ROOT / STATUS

    batch = read_json(REPLAY_ROOT / "replay_batch_status.json")
    if batch.get("status") != "PASS_N72R6_C0_C1_REPLAY" or int(batch.get("completed_event_count", -1)) != 32:
        raise RuntimeError("replay batch is not a complete PASS")
    effect = read_json(EFFECT)
    if effect.get("status") != "FAIL_FUTURE_EFFECT":
        raise RuntimeError(f"expected C1 effect to be a recorded FAIL: {effect.get('status')}")
    stage08 = read_json(STAGE08)
    eligible = {}
    for item in stage08.get("events", []):
        branches = {str(branch.get("branch")): branch for branch in item.get("branches", [])}
        b1 = branches.get("B1_SPATIAL_CORRECTION_ONLY")
        if b1 and b1.get("action_precondition_status") == "APPLIED":
            eligible[str(item["event_id"])] = item
    events = {str(item["event_id"]): item for item in read_json(EVENT_MANIFEST).get("events", [])}
    if len(eligible) != 32:
        raise RuntimeError(f"expected 32 eligible events, found {len(eligible)}")

    root_cause_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    totals = Counter()
    event_records: list[dict[str, Any]] = []
    protected_regression_records: list[dict[str, Any]] = []
    gt_hashes: dict[str, str] = {}

    for event_id in sorted(eligible):
        if event_id not in events:
            raise RuntimeError(f"missing frozen event policy: {event_id}")
        event = events[event_id]
        manifest = read_json(REPLAY_ROOT / event_id / "event_manifest.json")
        c0 = _load_rows(str(manifest["c0"]["path"]))
        c1 = _load_rows(str(manifest["c1"]["path"]))
        gt = _gt(str(event["sequence"]))
        gt_path = GT_ROOT / str(event["sequence"]) / "gt" / "gt.txt"
        gt_hashes[str(event["sequence"])] = sha256_file(gt_path)
        private = read_json(PRIVATE_ROOT / event_id / "oracle_private_mapping.json")
        gt_to_public = {int(key): int(value) for key, value in private["dataset_gt_to_public"].items()}
        target_gt = int(event["dataset_gt_id"])
        target_public = int(manifest["target_public_id"])
        if gt_to_public.get(target_gt) != target_public:
            raise RuntimeError(f"target posthoc mapping mismatch: {event_id}")

        visible = 0
        present = 0
        spatial_hit = 0
        drift = 0
        absent = 0
        target_assigned = 0
        target_identity_errors = 0
        shadow_rows = 0
        event_protected_regressions: list[dict[str, Any]] = []
        protected_ids = [
            gt_id for gt_id in sorted(gt_to_public)
            if int(gt_id) not in {target_gt, event.get("other_dataset_gt_id")}
        ]
        for frame in range(int(manifest["event_frame"]) + 1, int(manifest["event_frame"]) + HORIZON + 1):
            row = c1[frame]
            target_rows = [
                item for item in row.get("candidate_rows", [])
                if str(item.get("candidate_kind", "")) == "TARGET_CORRECTION_SESSION_CANDIDATE"
            ]
            shadow_rows += len((row.get("target_exclusive_constraint") or {}).get("shadowed_main_candidate_uids", []))
            totals["future_window_frames"] += 1
            totals["target_candidate_rows"] += len(target_rows)
            if target_rows:
                present += 1
                totals["target_candidate_present_frames"] += 1
                target_assigned += int(any(_effective_public(item) == target_public for item in target_rows))
                if target_gt in gt.get(frame, {}):
                    candidate_iou = max(
                        float(box_iou(item.get("box_xyxy"), gt[frame][target_gt]))
                        for item in target_rows
                    )
                    if candidate_iou >= IOU_THRESHOLD:
                        spatial_hit += 1
                        totals["target_candidate_spatial_hits"] += 1
                    else:
                        drift += 1
                        totals["target_candidate_drift_frames"] += 1
            elif target_gt in gt.get(frame, {}):
                absent += 1
                totals["target_candidate_absent_visible_frames"] += 1
            if target_gt in gt.get(frame, {}):
                visible += 1
                target_item = _target_frame(row, gt[frame][target_gt], target_public)
                target_identity_errors += int(target_item["identity_error"])
                totals["target_visible_frames"] += 1
            # N72R6's primary protected-regression gate is H20.  Target
            # candidate absence/drift above is audited over the full H100;
            # do not accidentally relabel H50/H100 regressions as the H20
            # gate count.
            if frame > int(manifest["event_frame"]) + 20:
                continue
            for gt_id in protected_ids:
                box = gt.get(frame, {}).get(int(gt_id))
                if box is None:
                    continue
                public = int(gt_to_public[gt_id])
                baseline_item = _target_frame(c0[frame], box, public)
                treatment_item = _target_frame(c1[frame], box, public)
                if baseline_item["correct"] and not treatment_item["correct"]:
                    detail = {
                        "frame": int(frame),
                        "gt_id": int(gt_id),
                        "public_id": public,
                        "baseline_assigned_public_id": baseline_item["assigned_public_id"],
                        "treatment_assigned_public_id": treatment_item["assigned_public_id"],
                        "baseline_geometry_iou": baseline_item["geometry_iou"],
                        "treatment_geometry_iou": treatment_item["geometry_iou"],
                    }
                    event_protected_regressions.append(detail)
                    protected_regression_records.append({"event_id": event_id, **detail})

        if absent and drift:
            root = "TARGET_SESSION_PROPAGATION_FAILURE_AND_TARGET_SESSION_IDENTITY_DRIFT"
        elif absent:
            root = "TARGET_SESSION_PROPAGATION_FAILURE"
        elif drift:
            root = "TARGET_SESSION_IDENTITY_DRIFT"
        else:
            root = "TARGET_CANDIDATE_SPATIAL_SUPPORT_NOT_OBSERVED"
        root_cause_counts[root] += 1
        action_counts[str(event["action_type"])] += 1
        event_records.append({
            "event_id": event_id,
            "sequence": str(event["sequence"]),
            "action_type": str(event["action_type"]),
            "event_frame": int(manifest["event_frame"]),
            "target_gt_id": target_gt,
            "target_public_id": target_public,
            "target_visible_future_frames": int(visible),
            "target_candidate_present_future_frames": int(present),
            "target_candidate_spatial_hit_frames": int(spatial_hit),
            "target_candidate_drift_frames": int(drift),
            "target_candidate_absent_visible_frames": int(absent),
            "target_candidate_assigned_target_rows": int(target_assigned),
            "target_identity_error_visible_frames": int(target_identity_errors),
            "shadowed_main_row_count": int(shadow_rows),
            "protected_regression_count_h20": len(event_protected_regressions),
            "protected_regressions_h20": event_protected_regressions,
            "root_cause": root,
            "runtime_future_gt_used": False,
            "posthoc_gt_used": True,
        })

    payload = {
        "schema_version": "N72R6_TARGET_ROOT_CAUSE_AUDIT_V1",
        "status": "PASS_TARGET_ROOT_CAUSE_AUDIT",
        "effect_status": effect.get("status"),
        "effect_artifact": str(EFFECT),
        "event_count": len(event_records),
        "sequence_count": len({item["sequence"] for item in event_records}),
        "action_counts": dict(sorted(action_counts.items())),
        "root_cause_event_counts": dict(sorted(root_cause_counts.items())),
        "totals": dict(sorted(totals.items())),
        "target_candidate_recall_over_all_future_frames": float(totals["target_candidate_rows"] / totals["future_window_frames"]),
        "target_candidate_spatial_recall_over_visible_present_frames": None if not (totals["target_candidate_spatial_hits"] + totals["target_candidate_drift_frames"]) else float(totals["target_candidate_spatial_hits"] / (totals["target_candidate_spatial_hits"] + totals["target_candidate_drift_frames"])),
        "protected_regression_count_h20": len(protected_regression_records),
        "protected_regression_scope": "event_frame+1..event_frame+20",
        "protected_regressions_h20": protected_regression_records,
        "events": event_records,
        "inputs": {
            "protocol": str(ROOT / "outputs/N72R6/protocol.json"),
            "protocol_sha256": sha256_file(ROOT / "outputs/N72R6/protocol.json"),
            "stage08": str(STAGE08),
            "stage08_sha256": sha256_file(STAGE08),
            "replay_batch": str(REPLAY_ROOT / "replay_batch_status.json"),
            "replay_batch_sha256": sha256_file(REPLAY_ROOT / "replay_batch_status.json"),
            "gt_sha256_by_sequence": gt_hashes,
        },
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "next_action": "N72R6-07_TARGET_SESSION_RECOVERY_AND_HUMAN_ANCHOR_GATE",
        "created_at_utc": now_utc(),
    }
    atomic_json(OUT, payload)
    atomic_json(STATUS, {
        "schema_version": "N72R6_STAGE_STATUS_V1",
        "stage": "N72R6-07_TARGET_ROOT_CAUSE_AUDIT",
        "status": payload["status"],
        "effect_status": payload["effect_status"],
        "event_count": payload["event_count"],
        "root_cause_event_counts": payload["root_cause_event_counts"],
        "target_candidate_recall_over_all_future_frames": payload["target_candidate_recall_over_all_future_frames"],
        "target_candidate_spatial_recall_over_visible_present_frames": payload["target_candidate_spatial_recall_over_visible_present_frames"],
        "protected_regression_count_h20": payload["protected_regression_count_h20"],
        "next_action": payload["next_action"],
        "runtime_future_gt_used": False,
        "created_at_utc": now_utc(),
    })
    print(json.dumps({
        "status": payload["status"],
        "root_causes": payload["root_cause_event_counts"],
        "target_candidate_recall": payload["target_candidate_recall_over_all_future_frames"],
        "target_candidate_spatial_recall": payload["target_candidate_spatial_recall_over_visible_present_frames"],
        "protected_regression_h20": payload["protected_regression_count_h20"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
