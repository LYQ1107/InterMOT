#!/usr/bin/env python3
"""Generate runtime-only S0/S1 presence features from sealed R2 tapes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sam3_intermot.association.identity_presence import evaluate_identity_presence
from scripts.n72r20r3_common import (
    CHECKPOINT_SHA,
    DATASET_ROOT,
    ENCODER_SHA,
    R2_ASSET_ROOT,
    ROOT,
    SEQUENCES,
    candidate_by_uid,
    gt_by_frame,
    initialize_base_states,
    initialize_learned_bank,
    load_sequence,
    public_assignment_uid,
    read_events,
    read_labels,
    target_base_context,
    target_boxes_for_frame,
    target_iou_for_uid,
    unit,
    update_base_for_frame,
    write_zstd_jsonl,
    best_target_candidate,
)


def _hash_vector(value: Any) -> str:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    return hashlib.sha256(vector.tobytes()).hexdigest()


def _runtime_policy() -> dict[str, Any]:
    return {
        "method": "B0_ALWAYS_PRESENT",
        "policy_version": "N72R20R3_FEATURE_EXTRACTION_B0_V1",
        "encoder_sha256": ENCODER_SHA,
    }


def _presence_row(
    *,
    condition: str,
    event: Mapping[str, Any],
    frame: int,
    candidates: list[dict[str, Any]],
    base_row: Mapping[str, Any],
    base_states: Mapping[int, Any],
    state: np.ndarray,
    anchor: np.ndarray,
    memory_state_hash: str,
    last_memory_write_frame: int,
) -> dict[str, Any]:
    public_id = int(event["target_public_id"])
    association_state_id = int(event["target_association_state_id"])
    base_uid = public_assignment_uid(base_row["base_assignment"], public_id)
    context = target_base_context(
        states=base_states,
        candidates=candidates,
        base_row=base_row,
        public_id=public_id,
        association_state_id=association_state_id,
        frame=frame,
    )
    context.update(
        {
            "public_id": public_id,
            "frame": int(frame),
            "encoder_sha256": ENCODER_SHA,
            "frames_since_human_initialization": max(0, int(frame) - int(event["event_frame"])),
            "frames_since_last_memory_write": max(0, int(frame) - int(last_memory_write_frame)),
            "memory_state_hash": memory_state_hash,
            "human_anchor_hash": _hash_vector(anchor),
        }
    )
    row = evaluate_identity_presence(
        learned_state=state,
        human_anchor=anchor,
        candidate_rows=candidates,
        base_assignment=base_row["base_assignment"],
        runtime_context=context,
        frozen_policy=_runtime_policy(),
    )
    row.update(
        {
            "stage": "N72R20R3",
            "sequence": str(base_row["sequence"]),
            "state_condition": condition,
            "base_assignment_candidate_uid": base_uid,
            "association_state_id": association_state_id,
            "target_public_id": public_id,
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
        }
    )
    return row


def _oracle_clean_rows(
    *,
    event: Mapping[str, Any],
    label: Mapping[str, Any],
    frames: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    base_rows: list[dict[str, Any]],
    base_states: Mapping[int, Any],
    dataset_root: Path,
) -> list[dict[str, Any]]:
    """Optional S2 diagnostic; GT is used only to gate post-hoc state updates."""

    gt = gt_by_frame(dataset_root / "train" / str(event["sequence"]) / "gt" / "gt.txt")
    bank = initialize_learned_bank(event)
    target_public = int(event["target_public_id"])
    rows: list[dict[str, Any]] = []
    for (frame_payload, candidates), base_row in zip(frames, base_rows):
        frame = int(frame_payload["frame"])
        if frame == int(event["event_frame"]):
            continue
        state = bank.records[target_public].current_state
        anchor = bank.records[target_public].human_anchor
        context = target_base_context(
            states=base_states,
            candidates=candidates,
            base_row=base_row,
            public_id=target_public,
            association_state_id=int(event["target_association_state_id"]),
            frame=frame,
        )
        context.update(
            {
                "public_id": target_public,
                "frame": frame,
                "encoder_sha256": ENCODER_SHA,
                "frames_since_human_initialization": frame - int(event["event_frame"]),
                "frames_since_last_memory_write": frame - int(bank.records[target_public].last_update_frame),
                "memory_state_hash": bank.records[target_public].state_hash(),
                "human_anchor_hash": bank.records[target_public].anchor_hash(),
            }
        )
        row = evaluate_identity_presence(
            learned_state=state,
            human_anchor=anchor,
            candidate_rows=candidates,
            base_assignment=base_row["base_assignment"],
            runtime_context=context,
            frozen_policy=_runtime_policy(),
        )
        row.update(
            {
                "stage": "N72R20R3",
                "sequence": str(event["sequence"]),
                "frame": frame,
                "state_condition": "S2_ORACLE_CLEAN",
                "posthoc_oracle_only": True,
                "posthoc_gt_used": True,
                "runtime_future_gt_used": False,
            }
        )
        rows.append(row)
        target_boxes = target_boxes_for_frame(gt, frame, int(label["target_gt_id"]))
        base_uid = public_assignment_uid(base_row["base_assignment"], target_public)
        base_iou = target_iou_for_uid(base_uid, candidates, target_boxes)
        if base_uid is not None and base_iou is not None and base_iou >= 0.50:
            bank.update_from_consensus(
                frame=frame,
                candidate_rows=candidates,
                base_solver=base_row["base_assignment"],
                treatment_solver=base_row["base_assignment"],
                source="N72R20R3_S2_POSTHOC_ORACLE_CLEAN_ONLY",
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=R2_ASSET_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/N72R20R3/presence")
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--include-oracle-clean", action="store_true")
    args = parser.parse_args()

    events = read_events(args.asset_root)
    labels = read_labels(args.asset_root)
    runtime_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    sequence_summary: list[dict[str, Any]] = []
    for sequence in args.sequences:
        event = events[str(sequence)]
        frames, base_rows = load_sequence(args.asset_root, str(sequence))
        base_states = initialize_base_states(event, frames[0][1])
        target_public = int(event["target_public_id"])
        learned_bank = initialize_learned_bank(event)
        anchor = unit(event["human_anchor"], "human anchor")
        if int(frames[0][0]["frame"]) != int(event["event_frame"]):
            raise ValueError(f"{sequence}: event frame is not first tape frame")
        update_base_for_frame(
            states=base_states,
            event=event,
            candidates=frames[0][1],
            base_solver=base_rows[0]["base_assignment"],
            frame=int(event["event_frame"]),
        )
        for (frame_payload, candidates), base_row in zip(frames, base_rows):
            frame = int(frame_payload["frame"])
            if frame == int(event["event_frame"]):
                continue
            s1_record = learned_bank.records[target_public]
            runtime_rows.append(
                _presence_row(
                    condition="S0_HUMAN_ANCHOR_ONLY",
                    event=event,
                    frame=frame,
                    candidates=candidates,
                    base_row=base_row,
                    base_states=base_states,
                    state=anchor,
                    anchor=anchor,
                    memory_state_hash=_hash_vector(anchor),
                    last_memory_write_frame=int(event["event_frame"]),
                )
            )
            runtime_rows.append(
                _presence_row(
                    condition="S1_R2_STYLE_CAUSAL_LEARNED_STATE",
                    event=event,
                    frame=frame,
                    candidates=candidates,
                    base_row=base_row,
                    base_states=base_states,
                    state=s1_record.current_state,
                    anchor=s1_record.human_anchor,
                    memory_state_hash=s1_record.state_hash(),
                    last_memory_write_frame=int(s1_record.last_update_frame),
                )
            )
            learned_bank.update_from_consensus(
                frame=frame,
                candidate_rows=candidates,
                base_solver=base_row["base_assignment"],
                treatment_solver=base_row["base_assignment"],
                source="N72R20R3_S1_R2_STYLE_C0_BASE_ASSIGNMENT",
            )
            update_base_for_frame(
                states=base_states,
                event=event,
                candidates=candidates,
                base_solver=base_row["base_assignment"],
                frame=frame,
            )
        if args.include_oracle_clean:
            oracle_base_states = initialize_base_states(event, frames[0][1])
            update_base_for_frame(
                states=oracle_base_states,
                event=event,
                candidates=frames[0][1],
                base_solver=base_rows[0]["base_assignment"],
                frame=int(event["event_frame"]),
            )
            oracle_rows.extend(
                _oracle_clean_rows(
                    event=event,
                    label=labels[str(sequence)],
                    frames=frames,
                    base_rows=base_rows,
                    base_states=oracle_base_states,
                    dataset_root=args.dataset_root,
                )
            )
        sequence_summary.append(
            {
                "sequence": str(sequence),
                "future_frames": len(frames) - 1,
                "runtime_rows": 2 * (len(frames) - 1),
                "runtime_future_gt_used": False,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = args.output_dir / "frame_runtime_features.jsonl.zst"
    write_zstd_jsonl(runtime_path, runtime_rows)
    if oracle_rows:
        write_zstd_jsonl(args.output_dir / "oracle_clean_features.jsonl.zst", oracle_rows)
    manifest = {
        "stage": "N72R20R3",
        "status": "PASS_N72R20R3_RUNTIME_FEATURES",
        "source_stage": "N72R20R2",
        "asset_root": str(args.asset_root),
        "sequences": sequence_summary,
        "state_conditions": ["S0_HUMAN_ANCHOR_ONLY", "S1_R2_STYLE_CAUSAL_LEARNED_STATE"],
        "oracle_clean_included": bool(oracle_rows),
        "runtime_rows": len(runtime_rows),
        "runtime_future_gt_used": False,
        "posthoc_gt_used": False,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "encoder_sha256": ENCODER_SHA,
        "runtime_feature_file": str(runtime_path),
        "runtime_feature_file_sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
    }
    if oracle_rows:
        oracle_path = args.output_dir / "oracle_clean_features.jsonl.zst"
        manifest["oracle_rows"] = len(oracle_rows)
        manifest["oracle_file_sha256"] = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
        manifest["oracle_clean_posthoc_only"] = True
    (args.output_dir / "runtime_feature_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "runtime_rows": len(runtime_rows), "oracle_rows": len(oracle_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
