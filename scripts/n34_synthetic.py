"""Deterministic, explicitly synthetic N34 tapes used only as code fallback."""

from __future__ import annotations

from typing import Any

import numpy as np


FEATURE_DIM = 512


def feature(index: int) -> np.ndarray:
    value = np.zeros(FEATURE_DIM, dtype=np.float32)
    value[int(index) % FEATURE_DIM] = 1.0
    return value


def box_for(native_tid: int, frame: int) -> list[float]:
    if int(native_tid) == 11:
        x = 10.0 + 0.25 * int(frame)
    elif int(native_tid) == 22:
        x = 60.0 - 0.15 * int(frame)
    else:
        x = 110.0 + 0.10 * int(frame)
    return [x, 10.0, x + 20.0, 50.0]


def observation(native_tid: int, frame: int, obs_id: int | None = None) -> dict[str, Any]:
    index = {11: 1, 22: 2, 33: 3}[int(native_tid)]
    return {
        "obs_id": int(index if obs_id is None else obs_id),
        "box": box_for(native_tid, frame),
        "embedding": feature(index).tolist(),
        "feature": feature(index).tolist(),
        "native_tid": int(native_tid),
        "native_age": float(frame),
        "confidence": 0.95,
        "geometry_score": 0.9,
        "motion_score": 0.9,
        "native_score": 0.9,
        "base_score": 0.9,
    }


def event_spec(action: str, frame: int = 1) -> dict[str, Any]:
    action = str(action)
    event: dict[str, Any] = {
        "event_id": f"n34-synthetic-{action.lower()}",
        "event_type": action,
        "action_type": action,
        "frame": int(frame),
        "interaction_source": "simulated_from_gt",
        "future_gt_used_runtime": False,
        "quality": 1.0,
    }
    if action == "ADD_NEW_IDENTITY":
        event.update(
            {
                "public_id": 103,
                "canonical_public_id": 103,
                "gt_box": box_for(33, frame),
                "human_embedding": feature(3).tolist(),
                "correction_embedding": feature(3).tolist(),
            }
        )
    elif action == "AUTHORITATIVE_REASSIGN":
        event.update(
            {
                "public_id": 101,
                "canonical_public_id": 101,
                "current_public_id": 102,
                "gt_box": box_for(11, frame),
                "human_embedding": feature(1).tolist(),
                "correction_embedding": feature(1).tolist(),
            }
        )
    elif action == "ATOMIC_ID_SWAP":
        event.update(
            {
                "public_id": 101,
                "canonical_public_id": 101,
                "current_public_id": 102,
                "other_canonical_public_id": 102,
                "other_auto_tid": 101,
                "gt_box": box_for(11, frame),
                "other_gt_box": box_for(22, frame),
                "human_embedding": feature(1).tolist(),
                "correction_embedding": feature(1).tolist(),
                "spatial_corrections": [
                    {"public_id": 101, "box": box_for(11, frame), "native_tid": 11, "embedding": feature(1).tolist()},
                    {"public_id": 102, "box": box_for(22, frame), "native_tid": 22, "embedding": feature(2).tolist()},
                ],
            }
        )
    elif action == "RECOVER_IDENTITY":
        event.update(
            {
                "public_id": 101,
                "canonical_public_id": 101,
                "gt_box": box_for(11, frame),
                "human_embedding": feature(1).tolist(),
                "correction_embedding": feature(1).tolist(),
            }
        )
    else:
        raise ValueError(f"unsupported synthetic action: {action}")
    if action != "ATOMIC_ID_SWAP":
        event.setdefault(
            "spatial_correction",
            {
                "public_id": event["canonical_public_id"],
                "box": event["gt_box"],
                "native_tid": 33 if action == "ADD_NEW_IDENTITY" else 11,
                "embedding": event["correction_embedding"],
            },
        )
    event["competing_embeddings"] = [feature(2).tolist()]
    return event


def build_tape(action: str, future_frames: int = 120) -> dict[str, Any]:
    event = event_spec(action, frame=0)
    prefix = [
        {"public_id": 101, "embedding": feature(1).tolist(), "box": box_for(11, 0), "native_tid": 11},
        {"public_id": 102, "embedding": feature(2).tolist(), "box": box_for(22, 0), "native_tid": 22},
    ]
    if action == "ADD_NEW_IDENTITY":
        # The event's spatial transaction creates the new identity; it is not
        # included in the pre-event prefix.
        pass
    frames = []
    for frame in range(1, int(future_frames) + 1):
        native_ids = [11, 22] + ([33] if action == "ADD_NEW_IDENTITY" else [])
        candidates = [observation(native_tid, frame, index) for index, native_tid in enumerate(native_ids)]
        public_ids = [101, 102] + ([103] if action == "ADD_NEW_IDENTITY" else [])
        score_matrix = []
        for candidate_index, _ in enumerate(candidates):
            score_matrix.append(
                [
                    float(1.0 if candidate_index == state_index else 0.1)
                    for state_index in range(len(public_ids))
                ]
            )
        frames.append(
            {
                "frame": frame,
                "candidates": candidates,
                "candidate_complete": True,
                "candidate_set_complete": True,
                "public_ids": public_ids,
                "base_score_matrix": score_matrix,
                "geometry_score_matrix": score_matrix,
                "motion_score_matrix": score_matrix,
                "native_score_matrix": score_matrix,
            }
        )
    return {
        "protocol": "N34_SYNTHETIC_CANDIDATE_COMPLETE_TAPE",
        "synthetic": True,
        "interaction_source": "simulated_from_gt",
        "future_gt_used_runtime": False,
        "candidate_complete": True,
        "candidate_set_complete": True,
        "prefix_state": prefix,
        "event": event,
        "frames": frames,
    }


class SyntheticHumanExtractor:
    feature_dim = FEATURE_DIM

    def extract(self, seq_dir, frame: int, box: np.ndarray) -> np.ndarray:
        del seq_dir, frame
        x = int(round(float(np.asarray(box).reshape(-1)[0])))
        index = 1 if x < 50 else (2 if x < 100 else 3)
        return feature(index)

    def extract_mask(self, seq_dir, frame: int, box: np.ndarray, mask: np.ndarray) -> np.ndarray:
        del mask
        return self.extract(seq_dir, frame, box)
