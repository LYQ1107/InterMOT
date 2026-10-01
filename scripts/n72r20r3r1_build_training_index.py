#!/usr/bin/env python3
"""Build a reference-only explicit-NONE training index from sealed R3 assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r3_common import (
    DATASET_ROOT,
    R2_ASSET_ROOT,
    ROOT,
    SEQUENCES,
    best_target_candidate,
    gt_by_frame,
    initialize_learned_bank,
    load_sequence,
    public_assignment_uid,
    read_events,
    read_labels,
    target_boxes_for_frame,
    target_iou_for_uid,
    unit,
)
from scripts.n72r20r3_failure_taxonomy import classify_frame


def _hash_vector(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype=np.float32).tobytes()).hexdigest()


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def _runtime_rows() -> dict[tuple[str, int, str], dict[str, Any]]:
    rows = read_zstd_jsonl(ROOT / "outputs/N72R20R3/presence/frame_runtime_features.jsonl.zst")
    return {(str(row["sequence"]), int(row["frame"]), str(row["state_condition"])): row for row in rows}


def _posthoc_rows() -> dict[tuple[str, int, str], dict[str, Any]]:
    rows = read_zstd_jsonl(ROOT / "outputs/N72R20R3/presence/frame_posthoc_labels.jsonl.zst")
    return {(str(row["sequence"]), int(row["frame"]), str(row["state_condition"])): row for row in rows}


def _candidate_reference(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return only frozen candidate-axis references and causal metadata."""

    return {
        "candidate_uid": str(candidate["candidate_uid"]),
        "candidate_index": int(candidate["candidate_index"]),
        "embedding_offset": int(candidate["embedding_offset"]),
        "candidate_source": candidate.get("candidate_source"),
        "presence": candidate.get("presence"),
        "iou_pred": candidate.get("iou_pred"),
        "native_tid": candidate.get("native_tid"),
        "runtime_future_gt_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=R2_ASSET_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/N72R20R3R1/training")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    events = read_events(args.asset_root)
    labels = read_labels(args.asset_root)
    runtime = _runtime_rows()
    posthoc = _posthoc_rows()
    state_vectors: list[np.ndarray] = []
    anchor_vectors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    taxonomy_counts: Counter[str] = Counter()
    state_hash_mismatches: list[dict[str, Any]] = []
    runtime_missing_state_hash: list[dict[str, Any]] = []
    anchor_index_by_sequence: dict[str, int] = {}

    for sequence in SEQUENCES:
        event = events[sequence]
        label = labels[sequence]
        frames, base_rows = load_sequence(args.asset_root, sequence)
        gt = gt_by_frame(args.dataset_root / "train" / sequence / "gt" / "gt.txt")
        anchor = unit(event["human_anchor"], f"{sequence} human anchor")
        anchor_index_by_sequence[sequence] = len(anchor_vectors)
        anchor_vectors.append(anchor)
        bank = initialize_learned_bank(event)
        target_public = int(event["target_public_id"])
        for (frame_payload, candidates), base_row in zip(frames, base_rows):
            frame = int(frame_payload["frame"])
            if frame == int(event["event_frame"]):
                continue
            target_boxes = target_boxes_for_frame(gt, frame, int(label["target_gt_id"]))
            # Keep the exact deterministic tie rule already used by the
            # sealed R3 posthoc taxonomy so P1a/P1b remain lineage-compatible.
            best_uid, best_iou = best_target_candidate(candidates, target_boxes)
            base_uid = public_assignment_uid(base_row["base_assignment"], target_public)
            base_iou = target_iou_for_uid(base_uid, candidates, target_boxes)
            taxonomy = classify_frame(
                target_gt_present=bool(target_boxes),
                best_iou=best_iou,
                best_uid=best_uid,
                base_uid=base_uid,
                base_iou=base_iou,
            )
            target_candidate_index = None
            target_label = "NONE"
            if best_uid is not None and best_iou is not None and best_iou >= 0.50:
                target_candidate_index = next(index for index, item in enumerate(candidates) if str(item["candidate_uid"]) == best_uid)
                target_label = "CANDIDATE"
            label_counts[target_label] += 1
            taxonomy_counts[taxonomy] += 1
            for condition in ("S0_HUMAN_ANCHOR_ONLY", "S1_R2_STYLE_CAUSAL_LEARNED_STATE"):
                runtime_row = runtime[(sequence, frame, condition)]
                posthoc_row = posthoc[(sequence, frame, condition)]
                if runtime_row.get("runtime_future_gt_used") is not False or runtime_row.get("posthoc_gt_used") is not False:
                    raise ValueError(f"{sequence}:{frame}:{condition}: runtime provenance is not clean")
                if posthoc_row.get("posthoc_gt_used") is not True:
                    raise ValueError(f"{sequence}:{frame}:{condition}: posthoc label is not marked")
                if str(posthoc_row["taxonomy"]) != taxonomy:
                    raise ValueError(f"{sequence}:{frame}:{condition}: taxonomy mismatch")
                if condition == "S0_HUMAN_ANCHOR_ONLY":
                    state = anchor
                else:
                    state = bank.records[target_public].current_state
                state_ref = len(state_vectors)
                state_vectors.append(np.asarray(state, dtype=np.float32).copy())
                state_hash = _hash_vector(state)
                runtime_state_hash = runtime_row.get("memory_state_hash")
                if runtime_state_hash is None:
                    # R3's NO_CANDIDATES early return predates the complete
                    # runtime schema.  Keep the sealed R3 row untouched and
                    # reconstruct only this provenance field from the frozen
                    # state already being replayed here.
                    runtime_missing_state_hash.append(
                        {"sequence": sequence, "frame": frame, "condition": condition, "reason": runtime_row.get("presence_reason")}
                    )
                elif state_hash != str(runtime_state_hash):
                    state_hash_mismatches.append(
                        {"sequence": sequence, "frame": frame, "condition": condition, "expected": runtime_state_hash, "actual": state_hash}
                    )
                if condition == "S0_HUMAN_ANCHOR_ONLY":
                    fallback_last_write = int(event["event_frame"])
                else:
                    fallback_last_write = int(bank.records[target_public].last_update_frame)
                rows.append(
                    {
                        "stage": "N72R20R3R1",
                        "sequence": sequence,
                        "frame": frame,
                        "state_condition": condition,
                        "target_public_id": target_public,
                        "association_state_id": int(event["target_association_state_id"]),
                        "anchor_ref": anchor_index_by_sequence[sequence],
                        "state_ref": state_ref,
                        "candidate_axis": [_candidate_reference(candidate) for candidate in candidates],
                        "runtime_context": {
                            "candidate_count": int(runtime_row.get("candidate_count", len(candidates))),
                            "frames_since_human_initialization": int(
                                runtime_row.get("frames_since_human_initialization", max(0, frame - int(event["event_frame"])))
                            ),
                            "frames_since_last_memory_write": int(
                                runtime_row.get("frames_since_last_memory_write", max(0, frame - fallback_last_write))
                            ),
                            "learned_state_human_anchor_cosine": float(
                                runtime_row.get("learned_state_human_anchor_cosine", float(np.dot(unit(state, "state"), anchor)))
                            ),
                            "runtime_future_gt_used": False,
                        },
                        "training_label": {
                            "class": target_label,
                            "candidate_index": target_candidate_index,
                            "candidate_uid": best_uid if target_candidate_index is not None else None,
                            "tie_policy": "max target IoU, then lexicographically largest candidate UID (frozen R3 rule)",
                        },
                        "taxonomy_posthoc": taxonomy,
                        "target_gt_present_posthoc": bool(target_boxes),
                        "best_target_iou_posthoc": best_iou,
                        "posthoc_gt_used": True,
                        "runtime_future_gt_used": False,
                    }
                )
            bank.update_from_consensus(
                frame=frame,
                candidate_rows=candidates,
                base_solver=base_row["base_assignment"],
                treatment_solver=base_row["base_assignment"],
                source="N72R20R3R1_INDEX_REPLAY_ONLY",
            )

    if state_hash_mismatches:
        raise ValueError(f"state hash mismatches: {state_hash_mismatches[:2]}")
    if len(rows) != 2 * 8414:
        raise ValueError(f"unexpected index row count: {len(rows)}")
    state_array = np.asarray(state_vectors, dtype=np.float32)
    anchor_array = np.asarray(anchor_vectors, dtype=np.float32)
    np.save(args.output_dir / "state_vectors.float32.npy", state_array)
    np.save(args.output_dir / "anchor_vectors.float32.npy", anchor_array)
    index_path = args.output_dir / "training_index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json_line(row))

    folds = []
    for heldout in SEQUENCES:
        training = [sequence for sequence in SEQUENCES if sequence != heldout]
        internal_validation = [training[-1]]
        internal_training = [sequence for sequence in training if sequence != internal_validation[0]]
        folds.append(
            {
                "heldout_sequence": heldout,
                "training_sequences": training,
                "internal_training_sequences": internal_training,
                "internal_validation_sequences": internal_validation,
                "frame_random_split": False,
            }
        )
    manifest = {
        "stage": "N72R20R3R1",
        "status": "PASS_R3R1_TRAINING_INDEX_COMPLETE",
        "source_stage": "N72R20R3",
        "asset_root": str(args.asset_root),
        "dataset_root": str(args.dataset_root),
        "sequences": list(SEQUENCES),
        "rows": len(rows),
        "future_frames": 8414,
        "state_conditions": ["S0_HUMAN_ANCHOR_ONLY", "S1_R2_STYLE_CAUSAL_LEARNED_STATE"],
        "label_counts": dict(sorted(label_counts.items())),
        "taxonomy_counts": dict(sorted(taxonomy_counts.items())),
        "training_index": str(index_path),
        "state_vectors": str(args.output_dir / "state_vectors.float32.npy"),
        "anchor_vectors": str(args.output_dir / "anchor_vectors.float32.npy"),
        "candidate_features_copied": False,
        "candidate_feature_reference": "sealed per-sequence embeddings.f16 via embedding_offset",
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "candidate_label_definition": "max target IoU >= 0.50; otherwise NONE",
        "tie_policy": "max target IoU, then lexicographically largest candidate UID (frozen R3 rule)",
        "state_hash_checks": len(state_vectors),
        "runtime_missing_state_hash_rows": len(runtime_missing_state_hash),
        "runtime_missing_state_hash_examples": runtime_missing_state_hash[:10],
        "fold_count": len(folds),
    }
    (args.output_dir / "training_index_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "split_manifest.json").write_text(json.dumps({"stage": "N72R20R3R1", "folds": folds, "formal_protocol": "LOSO sequence-held-out; no frame-IID split"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    whitelist = {
        "stage": "N72R20R3R1",
        "runtime_features": [
            "frozen_candidate_embedding_reference",
            "human_anchor_512d",
            "frozen_n72r18_state_512d",
            "candidate_count",
            "frames_since_human_initialization",
            "frames_since_last_memory_write",
            "learned_state_human_anchor_cosine",
        ],
        "candidate_metadata": ["candidate_uid", "candidate_index", "embedding_offset"],
        "forbidden_runtime_features": [
            "target_gt_id",
            "target_iou",
            "best_target_candidate_uid",
            "taxonomy_posthoc",
            "training_label",
            "sequence_id",
            "absolute_frame_number",
        ],
        "runtime_future_gt_used": False,
    }
    (args.output_dir / "runtime_feature_whitelist.json").write_text(json.dumps(whitelist, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "rows": len(rows), "label_counts": manifest["label_counts"], "taxonomy_counts": manifest["taxonomy_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
