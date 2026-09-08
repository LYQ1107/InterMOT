#!/usr/bin/env python3
"""CPU-only contract check for N72R11R3 shared state and edge features."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.target_edge_bridge import build_target_edge_feature, build_target_edge_feature_from_scalars
from sam3_intermot.reacquisition.target_candidate_pool import FUTURE_FRAME_REQUERY
from sam3_intermot.reacquisition.temporal_state_policy import (
    TEMPORAL_FEATURE_SCHEMA,
    build_temporal_features,
    initialize_temporal_state,
    state_audit,
    state_memory_arrays,
    update_temporal_state,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/N72R11R3/stage_09_contract_check.json")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    seed = 721103
    rng = np.random.default_rng(seed)
    anchor = rng.normal(size=512).astype(np.float32)
    candidates = [
        {
            "candidate_uid": "toy:target",
            "candidate_source": "MAIN_B0_CANDIDATE",
            "feature": rng.normal(size=512).astype(np.float32),
            "box_xyxy": [10.0, 10.0, 40.0, 70.0],
            "official_raw_sam_id": 17,
            "native_scope": "toy:scope",
            "confidence": 0.9,
            "presence_score": 0.9,
            "incumbent_public_id_if_any": 1007,
        },
        {
            "candidate_uid": "toy:distractor",
            "candidate_source": FUTURE_FRAME_REQUERY,
            "feature": rng.normal(size=512).astype(np.float32),
            "box_xyxy": [50.0, 10.0, 80.0, 70.0],
            "official_raw_sam_id": 18,
            "native_scope": "toy:scope",
            "confidence": 0.7,
            "presence_score": 0.7,
            "incumbent_public_id_if_any": 1008,
        },
    ]
    state_training = initialize_temporal_state(anchor_feature=anchor, anchor_box=[10.0, 10.0, 40.0, 70.0], previous_raw_sam_id=17, previous_native_scope="toy:scope")
    state_runtime = initialize_temporal_state(anchor_feature=anchor, anchor_box=[10.0, 10.0, 40.0, 70.0], previous_raw_sam_id=17, previous_native_scope="toy:scope")
    temporal_training = build_temporal_features(state_training, frame_horizon=1, causal_top_score=0.8, causal_second_score=0.2, causal_margin=0.6, has_future_requery=True)
    temporal_runtime = build_temporal_features(state_runtime, frame_horizon=1, causal_top_score=0.8, causal_second_score=0.2, causal_margin=0.6, has_future_requery=True)
    if not np.array_equal(temporal_training, temporal_runtime):
        raise AssertionError("training/runtime temporal feature mismatch")
    memories_training = state_memory_arrays(state_training)
    memories_runtime = state_memory_arrays(state_runtime)
    if not all(np.array_equal(left, right) for left, right in zip(memories_training, memories_runtime, strict=True)):
        raise AssertionError("training/runtime memory tensor mismatch")
    update_kwargs = {
        "candidates": candidates,
        "target_uid": "toy:target",
        "selected_uid": "toy:target",
        "selected_score": 0.8,
        "selected_margin": 0.6,
        "fused_target_scores": [0.8, 0.1],
        "frame_horizon": 1,
        "assigned_candidate": candidates[0],
        "base_top_score": 0.8,
        "base_second_score": 0.2,
    }
    update_temporal_state(state_training, **update_kwargs)
    update_temporal_state(state_runtime, **update_kwargs)
    if state_audit(state_training) != state_audit(state_runtime):
        raise AssertionError("training/runtime state update mismatch")
    scalar = build_target_edge_feature_from_scalars(
        candidate_logit=0.4, none_logit=0.1, legacy_target_score=0.2,
        legacy_best_other_score=0.3, incumbent_target=1.0, incumbent_other=0.0,
        confidence=0.9, presence=0.9, motion_iou=0.6, candidate_source="MAIN_B0_CANDIDATE",
    )
    wrapped = build_target_edge_feature(
        candidates[0], candidate_logit=0.4, none_logit=0.1,
        legacy_target_score=0.2, legacy_public_scores={1007: 0.2, 1008: 0.3},
        target_public_id=1007, motion_iou=0.6,
    )
    if not np.allclose(np.asarray(scalar), np.asarray(wrapped)):
        raise AssertionError("training/runtime target-edge scalar mismatch")
    result = {
        "schema_version": "N72R11R3_CPU_CONTRACT_CHECK_V1",
        "status": "PASS_N72R11R3_CPU_CONTRACT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "temporal_feature_schema": list(TEMPORAL_FEATURE_SCHEMA),
        "temporal_feature_equal": True,
        "memory_arrays_equal": True,
        "state_update_equal": True,
        "target_edge_feature_equal": True,
        "event_frame_memory_read": False,
        "first_memory_visible_frame": 1,
        "runtime_future_gt_used": False,
        "toy_fixture_only": True,
        "scientific_result": None,
    }
    _atomic_json(output, result)
    result["output"] = str(output)
    result["output_sha256"] = _sha256(output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
