#!/usr/bin/env python3
"""C0-C3 causal trusted-commit replay on the sealed R2 dev tapes."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sam3_intermot.association.identity_memory_commit import evaluate_commit
from sam3_intermot.association.learned_identity_memory import LearnedIdentityMemoryBank
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames, read_json, read_zstd_jsonl, unit
from scripts.n72r20r2_forensics import gt_by_frame, iou


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R1_assets")
DATASET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R16_assets/dataset")
CHECKPOINT = ROOT / "outputs/N72R18/checkpoints/identity_memory_gru.pt"
CHECKPOINT_SHA = "94ba9c44e177b254ee54984195e839f37c78a648c010ffc723dd3f1d553976a2"
ENCODER_SHA = "2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154"
IOU_THRESHOLD = 0.50
C2_THRESHOLDS = (0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15)


def initialize_bank(event: Mapping[str, Any]) -> LearnedIdentityMemoryBank:
    bank = LearnedIdentityMemoryBank.from_checkpoint(
        CHECKPOINT,
        encoder_sha256=ENCODER_SHA,
        expected_encoder_sha256=ENCODER_SHA,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
    )
    bank.initialize_public_identity(
        public_id=int(event["target_public_id"]),
        association_state_id=int(event["target_association_state_id"]),
        frame=int(event["event_frame"]),
        human_anchor=unit(event["human_anchor"], "human anchor"),
        interaction_source="simulated_from_gt",
    )
    return bank


def target_assignment(solver: Mapping[str, Any], public_id: int) -> str | None:
    for item in solver.get("public_assignments", []):
        if int(item["public_id"]) == int(public_id):
            uid = item.get("candidate_uid")
            return None if uid in (None, "", "None") else str(uid)
    raise ValueError(f"public ID {public_id} absent from solver")


def solver_without_target(solver: Mapping[str, Any], public_id: int) -> dict[str, Any]:
    result = copy.deepcopy(dict(solver))
    for item in result.get("public_assignments", []):
        if int(item["public_id"]) == int(public_id):
            item["candidate_uid"] = None
            item["status"] = "NO_CANDIDATE_ASSIGNED"
    return result


def target_iou(uid: str | None, candidates: Sequence[Mapping[str, Any]], gt: Mapping[int, list[tuple[int, list[float]]]], frame: int, target_gt_id: int) -> float | None:
    if uid is None:
        return None
    candidate = next((row for row in candidates if str(row["candidate_uid"]) == str(uid)), None)
    if candidate is None:
        return None
    boxes = [box for track_id, box in gt.get(int(frame), []) if int(track_id) == int(target_gt_id)]
    return float(max((iou(candidate["box_xyxy"], box) for box in boxes), default=0.0))


def replay_policy(
    *,
    asset_root: Path,
    dataset_root: Path,
    sequence: str,
    event: Mapping[str, Any],
    label: Mapping[str, Any],
    policy: str,
    margin_threshold: float = 0.0,
) -> dict[str, Any]:
    candidate_frames = load_candidate_frames(asset_root, sequence)
    base_rows = read_zstd_jsonl(asset_root / "base_scores" / sequence / "base_scores.jsonl.zst")
    gt = gt_by_frame(dataset_root / "train" / sequence / "gt" / "gt.txt")
    target_public = int(event["target_public_id"])
    target_gt_id = int(label["target_gt_id"])
    bank = initialize_bank(event)
    rows: list[dict[str, Any]] = []
    previous_frame = None
    previous_compatible = False
    for (frame_payload, candidates), base_row in zip(candidate_frames, base_rows):
        frame = int(frame_payload["frame"])
        if frame == int(event["event_frame"]):
            continue
        base_solver = base_row["base_assignment"]
        assigned_uid = target_assignment(base_solver, target_public)
        state_before = bank.records[target_public].current_state
        state_hash_before = bank.records[target_public].state_hash()
        decision: dict[str, Any]
        if policy == "C0":
            decision = {
                "commit": assigned_uid is not None,
                "reason": "C0_BASE_ASSIGNMENT_ACCEPTED" if assigned_uid is not None else "NONE_ASSIGNMENT",
                "learned_top1_candidate_uid": None,
                "learned_margin": None,
                "learned_rank": None,
                "learned_score": None,
                "runtime_future_gt_used": False,
            }
        else:
            score_payload = bank.score_matrix(candidate_rows=candidates, public_id_axis=[target_public], frame=frame)
            scores = np.asarray(score_payload["state_candidate_scores"], dtype=np.float64)[0]
            decision = evaluate_commit(
                public_id=target_public,
                assigned_candidate_uid=assigned_uid,
                candidate_rows=candidates,
                learned_state=state_before,
                frame=frame,
                margin_threshold=0.0 if policy in {"C1", "C3"} else float(margin_threshold),
                require_learned_top1=True,
                temporal_confirmed=previous_compatible and previous_frame is not None and frame == previous_frame + 1,
                require_temporal_confirmation=policy == "C3",
            )
            score_by_uid = {
                str(row["candidate_uid"]): float(scores[index])
                for index, row in enumerate(candidates)
            }
            decision["learned_score_of_assigned_candidate"] = (
                None if assigned_uid is None else score_by_uid.get(str(assigned_uid))
            )
        update_solver = base_solver if decision["commit"] else solver_without_target(base_solver, target_public)
        update = bank.update_from_consensus(
            frame=frame,
            candidate_rows=candidates,
            base_solver=update_solver,
            treatment_solver=update_solver,
            source=f"N72R20R2_{policy}_TRUSTED_COMMIT",
        )
        state_after = bank.records[target_public].current_state
        target_box_iou = target_iou(assigned_uid, candidates, gt, frame, target_gt_id)
        row = {
            "stage": "N72R20R2",
            "sequence": sequence,
            "frame": frame,
            "policy": policy,
            "margin_threshold": float(margin_threshold),
            "base_target_candidate_uid": assigned_uid,
            "commit": bool(decision["commit"]),
            "commit_reason": decision.get("reason"),
            "learned_top1_candidate_uid": decision.get("learned_top1_candidate_uid"),
            "learned_margin": decision.get("learned_margin"),
            "learned_rank_of_assigned_candidate": decision.get("learned_rank"),
            "learned_score_of_assigned_candidate": decision.get("learned_score_of_assigned_candidate", decision.get("learned_score")),
            "target_iou_posthoc": target_box_iou,
            "correct_write": bool(decision["commit"] and target_box_iou is not None and target_box_iou >= IOU_THRESHOLD),
            "wrong_write": bool(decision["commit"] and (target_box_iou is None or target_box_iou < IOU_THRESHOLD)),
            "state_hash_before": state_hash_before,
            "state_hash_after": bank.records[target_public].state_hash(),
            "state_similarity_before_after": float(np.dot(state_before, state_after) / max(np.linalg.norm(state_before) * np.linalg.norm(state_after), 1.0e-8)),
            "memory_update_count": len(update.get("updated_public_ids", [])),
            "runtime_future_gt_used": False,
            "posthoc_gt_used": True,
        }
        rows.append(row)
        current_compatible = assigned_uid is not None and decision.get("learned_top1_candidate_uid") == assigned_uid
        previous_compatible = bool(current_compatible)
        previous_frame = frame
    writes = [row for row in rows if row["commit"]]
    correct = [row for row in writes if row["correct_write"]]
    wrong = [row for row in writes if row["wrong_write"]]
    return {
        "sequence": sequence,
        "policy": policy,
        "margin_threshold": float(margin_threshold),
        "frames": len(rows),
        "accepted_writes": len(writes),
        "correct_writes": len(correct),
        "wrong_writes": len(wrong),
        "wrong_write_rate": None if not writes else len(wrong) / len(writes),
        "rows": rows,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
    }


def summarize(results: Sequence[Mapping[str, Any]], c0_correct_total: int) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for result in results:
        key = f"{result['policy']}_delta_{result['margin_threshold']:.2f}"
        grouped.setdefault(key, {"policy": result["policy"], "margin_threshold": result["margin_threshold"], "sequences": []})["sequences"].append(
            {key: value for key, value in result.items() if key != "rows"}
        )
    output = []
    for item in grouped.values():
        sequences = item["sequences"]
        accepted = sum(int(x["accepted_writes"]) for x in sequences)
        correct = sum(int(x["correct_writes"]) for x in sequences)
        wrong = sum(int(x["wrong_writes"]) for x in sequences)
        output.append(
            {
                "policy": item["policy"],
                "margin_threshold": item["margin_threshold"],
                "sequences": sequences,
                "accepted_writes": accepted,
                "correct_writes": correct,
                "wrong_writes": wrong,
                "wrong_write_rate": None if accepted == 0 else wrong / accepted,
                "correct_write_retention_vs_c0": None if c0_correct_total == 0 else correct / c0_correct_total,
                "commit_safety_pass": accepted > 0 and wrong / accepted <= 0.02,
                "commit_usefulness_pass": c0_correct_total > 0 and correct / c0_correct_total >= 0.60,
            }
        )
    return {"policies": sorted(output, key=lambda x: (x["policy"], x["margin_threshold"]))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=ASSET_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--sequences", nargs="+", default=["dancetrack0001", "dancetrack0002"])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/N72R20R2/commit_policy_scan.json")
    args = parser.parse_args()
    events_payload = read_json(args.asset_root / "interaction_events.json")
    labels_payload = read_json(args.asset_root / "posthoc_event_labels.json")
    events = {str(item["sequence"]): dict(item) for item in events_payload["events"]}
    labels = {str(item["sequence"]): dict(item) for item in labels_payload["labels"]}
    raw: list[dict[str, Any]] = []
    for sequence in args.sequences:
        raw.append(replay_policy(asset_root=args.asset_root.resolve(), dataset_root=args.dataset_root.resolve(), sequence=str(sequence), event=events[str(sequence)], label=labels[str(sequence)], policy="C0"))
    c0_correct_total = sum(int(item["correct_writes"]) for item in raw)
    policies = [("C0", 0.0), ("C1", 0.0), ("C3", 0.0)]
    policies.extend(("C2", threshold) for threshold in C2_THRESHOLDS)
    for policy, threshold in policies:
        if policy == "C0":
            continue
        for sequence in args.sequences:
            raw.append(replay_policy(asset_root=args.asset_root.resolve(), dataset_root=args.dataset_root.resolve(), sequence=str(sequence), event=events[str(sequence)], label=labels[str(sequence)], policy=policy, margin_threshold=threshold))
    summary = summarize(raw, c0_correct_total)
    payload = {
        "stage": "N72R20R2",
        "source_stage": "N72R20R1",
        "status": "PASS_N72R20R2_COMMIT_POLICY_SCAN",
        "policies": ["C0", "C1", "C2", "C3"],
        "c2_thresholds": list(C2_THRESHOLDS),
        "c3_definition": "C1 learned-top1 agreement plus agreement on the immediately preceding consecutive frame; no margin threshold",
        "c0_correct_write_denominator": c0_correct_total,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "summary": summary,
        "raw_results": [
            {key: value for key, value in item.items() if key != "rows"}
            for item in raw
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "c0_correct": c0_correct_total, "policies": len(summary["policies"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
