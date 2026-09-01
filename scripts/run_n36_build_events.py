#!/usr/bin/env python3
"""Build a small, auditable real-data event manifest after N36 tape completion.

The candidate tape is produced without annotations.  This script is the
offline boundary: it reads only ``train/train_fold`` GT to simulate the
current-frame human interaction, derives a prefix from observations strictly
before the event, and extracts the human feature from the supplied GT box
crop.  No GT field is copied into candidate rows or used by the replay
association code.

The manifest intentionally contains a bounded, deterministic event sample
from six independent sequences.  It is sufficient for the N36 real
full-loop and paired-replay gates without pretending that every frame is a
human interaction.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.ccam_replay import _manager_from_prefix
from sam3_intermot.association.human_intervention import HumanFeatureExtractor
from sam3_intermot.association.state_manager import StateManagerConfig
from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.interaction.n8_temporal_observer import N8Config, N8TemporalObserver
from sam3_intermot.interaction.simulator import GTFrame
from scripts.n36_tape_common import DATA_ROOT, atomic_json, box_iou, iter_jsonl


FEATURE_DIM = 512
DEFAULT_SEQUENCE_LIST = ROOT / "outputs/n34/selected_sequences.json"
DEFAULT_TAPE_ROOT = ROOT / "outputs/n36/real_tape/frames"
DEFAULT_OUTPUT = ROOT / "outputs/n36/real_event_manifest.json"
HUMAN_CHECKPOINT = ROOT / "outputs/n9/checkpoints/osnet_x1_0_market1501.pth"
PUBLIC_OFFSET = 100_000
ADD_PUBLIC_OFFSET = 900_000
REPLAY_HORIZON = 100

ACTION_TYPES = (
    "AUTHORITATIVE_REASSIGN",
    "ATOMIC_ID_SWAP",
    "RECOVER_IDENTITY",
    "ADD_NEW_IDENTITY",
)

# One deliberately diverse event per sequence.  If a preferred action is not
# available before the H100 boundary, the script chooses the earliest legal
# fallback in this order.
PREFERRED_ACTION = {
    "dancetrack0001": "AUTHORITATIVE_REASSIGN",
    "dancetrack0002": "ATOMIC_ID_SWAP",
    "dancetrack0006": "ADD_NEW_IDENTITY",
    "dancetrack0008": "RECOVER_IDENTITY",
    "dancetrack0012": "AUTHORITATIVE_REASSIGN",
    "dancetrack0015": "ATOMIC_ID_SWAP",
}


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def digest_feature(value: Any) -> str:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    return hashlib.sha256(vector.tobytes()).hexdigest()


def load_gt(dataset: DanceTrackDataset, sequence: str) -> dict[int, GTFrame]:
    return dataset.load_gt(sequence)


def load_backbone(path: Path) -> tuple[dict[int, list[tuple[int, np.ndarray]]], int]:
    """Load only boxes/native IDs for the offline N8 event detector."""
    backbone: dict[int, list[tuple[int, np.ndarray]]] = {}
    frame_count = 0
    for _line_no, row in iter_jsonl(path):
        frame = int(row["frame"])
        observations = []
        for candidate in row.get("candidates", []):
            native = int(candidate.get("sequence_global_native_id", candidate["native_tid"]))
            observations.append((native, np.asarray(candidate["box"], dtype=float)))
        backbone[frame] = observations
        frame_count = max(frame_count, frame + 1)
    return backbone, frame_count


def public_id(n8_public_id: int) -> int:
    """Keep observed/N8 canonical IDs separate from ADD allocations."""
    return PUBLIC_OFFSET + int(n8_public_id)


def convert_pre_rows(rows: list[tuple[int, np.ndarray]]) -> list[list[Any]]:
    return [[public_id(int(pid)), np.asarray(box, dtype=float).tolist()] for pid, box in rows]


def obs_from_candidate(candidate: dict[str, Any], index: int) -> dict[str, Any]:
    feature = np.asarray(candidate["machine_embedding"], dtype=np.float32).reshape(-1)
    if feature.size != FEATURE_DIM or not np.all(np.isfinite(feature)):
        raise ValueError(f"machine feature invalid at candidate {index}")
    norm = float(np.linalg.norm(feature))
    if norm <= 1e-6:
        raise ValueError(f"machine feature is zero at candidate {index}")
    return {
        "obs_id": int(candidate.get("candidate_index", index)),
        "box": np.asarray(candidate["box"], dtype=float),
        "feat": feature / norm,
        "has_feat": 1.0,
        "native_tid": int(candidate.get("sequence_global_native_id", candidate["native_tid"])),
        "native_age": float(candidate.get("native_age", 0.0)),
        "conf": float(candidate.get("confidence", 1.0)),
    }


def ranked_infos(infos: list[dict[str, Any]], sequence: str, frame_count: int) -> list[dict[str, Any]]:
    eligible = [
        info
        for info in infos
        if 20 <= int(info["frame"]) and int(info["frame"]) + REPLAY_HORIZON < frame_count
    ]
    preferred = PREFERRED_ACTION.get(sequence)
    order = [preferred] if preferred else []
    order += [action for action in ACTION_TYPES if action not in order]
    ranked = []
    for action in order:
        ranked.extend(
            sorted(
                [item for item in eligible if item["action_type"] == action],
                key=lambda item: (int(item["frame"]), str(item["event_id"])),
            )
        )
    return ranked


def event_box(gt: GTFrame, gid: int) -> np.ndarray:
    for current_id, box in zip(gt.gt_ids, gt.boxes):
        if int(current_id) == int(gid):
            return np.asarray(box, dtype=float).copy()
    raise KeyError(f"GT id {gid} is absent from the event frame")


def best_candidate(observations: list[dict[str, Any]], box: np.ndarray) -> tuple[int | None, float]:
    if not observations:
        return None, 0.0
    values = [box_iou(box, item["box"]) for item in observations]
    index = int(np.argmax(np.asarray(values, dtype=float)))
    return index, float(values[index])


def build_prefix(
    path: Path,
    event_frame: int,
    map_before_by_frame: dict[int, dict[int, int]],
) -> list[dict[str, Any]]:
    """Build public-ID state only from frames strictly before the event."""
    latest: dict[int, dict[str, Any]] = {}
    for _line_no, row in iter_jsonl(path):
        frame = int(row["frame"])
        if frame >= event_frame:
            break
        native_map = map_before_by_frame.get(frame, {})
        for index, candidate in enumerate(row.get("candidates", [])):
            native = int(candidate.get("sequence_global_native_id", candidate["native_tid"]))
            n8_pid = int(native_map.get(native, native))
            item = obs_from_candidate(candidate, index)
            latest[n8_pid] = {
                "public_id": public_id(n8_pid),
                "embedding": item["feat"].tolist(),
                "box": item["box"].tolist(),
                "native_tid": int(item["native_tid"]),
                "last_observed_frame": frame,
            }
    return [latest[key] for key in sorted(latest)]


def current_rows(path: Path, event_frame: int) -> list[dict[str, Any]]:
    for _line_no, row in iter_jsonl(path):
        if int(row["frame"]) == int(event_frame):
            return [obs_from_candidate(item, index) for index, item in enumerate(row.get("candidates", []))]
    raise KeyError(f"frame {event_frame} is absent from {path}")


def make_runtime_probe(prefix: list[dict[str, Any]], observations: list[dict[str, Any]], frame: int) -> list[tuple[int, np.ndarray]]:
    config = StateManagerConfig(
        variant="reid",
        score_threshold=-100.0,
        max_lost_gap=90,
        use_appearance_memory=False,
    )
    manager = _manager_from_prefix(prefix, frame, config, FEATURE_DIM)
    rows = manager.rollout_frame(frame, observations, model=None)
    return [(int(pid), np.asarray(box, dtype=float).copy()) for pid, box in rows]


def build_event(
    info: dict[str, Any],
    prefix: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    gt_frames: dict[int, GTFrame],
    sequence: str,
    sequence_dir: Path,
    extractor: HumanFeatureExtractor,
) -> dict[str, Any]:
    action = str(info["action_type"])
    frame = int(info["frame"])
    gid = int(info["dataset_gt_id"])
    gt_box = event_box(gt_frames[frame], gid)
    target_index, target_iou = best_candidate(observations, gt_box)
    target_obs = None if target_index is None else observations[target_index]
    n8_event = info["event"]
    canonical_n8 = n8_event.get("canonical_public_id")
    if action == "ADD_NEW_IDENTITY":
        planned_pid = max([int(item["public_id"]) for item in prefix] or [PUBLIC_OFFSET]) + 1
        canonical = int(planned_pid)
    else:
        if canonical_n8 is None:
            raise ValueError(f"{sequence}:{frame} has no canonical public id for {action}")
        canonical = public_id(int(canonical_n8))

    human_feature = np.asarray(extractor.extract(sequence_dir, frame, gt_box), dtype=np.float32).reshape(-1)
    if human_feature.size != FEATURE_DIM or not np.all(np.isfinite(human_feature)):
        raise ValueError(f"human feature invalid for {sequence}:{frame}")
    human_norm = float(np.linalg.norm(human_feature))
    if human_norm <= 1e-6:
        raise ValueError(f"human feature is zero for {sequence}:{frame}")
    human_feature = human_feature / human_norm

    # Probe the exact prefix/current-candidate pairing used by the full-loop
    # runner.  This makes the ATOMIC_ID_SWAP lookup use the actual public IDs
    # in the pre-correction rows rather than assuming raw native numbering.
    probe_rows = make_runtime_probe(prefix, observations, frame)
    probe_target_pid = None
    if target_obs is not None:
        if probe_rows:
            values = [box_iou(target_obs["box"], box) for _pid, box in probe_rows]
            probe_target_pid = int(probe_rows[int(np.argmax(values))][0]) if values else None

    current_public = None
    if action == "AUTHORITATIVE_REASSIGN":
        current_public = probe_target_pid
        if current_public is None:
            raise ValueError(f"{sequence}:{frame} reassign has no current public assignment")
    other_gt_box = None
    other_canonical = None
    other_auto = None
    spatial: list[dict[str, Any]] = []
    if action == "AUTHORITATIVE_REASSIGN":
        if target_obs is None or target_iou < 0.3:
            raise ValueError(f"{sequence}:{frame} reassign target candidate IoU={target_iou:.3f}")
        spatial.append(
            {
                "public_id": canonical,
                "box": target_obs["box"].tolist(),
                "native_tid": int(target_obs["native_tid"]),
                "embedding": target_obs["feat"].tolist(),
                "embedding_source": "machine_candidate_feature_for_spatial_state",
            }
        )
    elif action == "ATOMIC_ID_SWAP":
        other_gid = n8_event.get("other_dataset_gt_id")
        other_canonical_n8 = n8_event.get("other_canonical_public_id")
        if other_gid is None or other_canonical_n8 is None:
            raise ValueError(f"{sequence}:{frame} atomic event lacks second identity")
        other_gt_box = event_box(gt_frames[frame], int(other_gid))
        other_canonical = public_id(int(other_canonical_n8))
        other_index, other_iou = best_candidate(observations, other_gt_box)
        other_obs = None if other_index is None else observations[other_index]
        if target_obs is None or target_iou < 0.3 or other_obs is None or other_iou < 0.3:
            raise ValueError(f"{sequence}:{frame} atomic candidate IoUs={target_iou:.3f}/{other_iou:.3f}")
        if probe_rows:
            values = [box_iou(other_obs["box"], box) for _pid, box in probe_rows]
            other_auto = int(probe_rows[int(np.argmax(values))][0]) if values else None
        if other_auto is None:
            raise ValueError(f"{sequence}:{frame} atomic has no second pre-row assignment")
        probe_public_ids = [int(pid) for pid, _box in probe_rows]
        if len(probe_public_ids) != len(set(probe_public_ids)):
            raise ValueError(f"{sequence}:{frame} atomic probe has duplicate public IDs")
        if probe_target_pid == other_auto:
            raise ValueError(f"{sequence}:{frame} atomic target/other share current public ID")
        if probe_target_pid == canonical:
            raise ValueError(f"{sequence}:{frame} atomic target is already canonical")
        if other_auto == other_canonical:
            raise ValueError(f"{sequence}:{frame} atomic other identity is already canonical")
        spatial.extend(
            [
                {
                    "public_id": canonical,
                    "box": target_obs["box"].tolist(),
                    "native_tid": int(target_obs["native_tid"]),
                    "embedding": target_obs["feat"].tolist(),
                    "embedding_source": "machine_candidate_feature_for_spatial_state",
                },
                {
                    "public_id": other_canonical,
                    "box": other_obs["box"].tolist(),
                    "native_tid": int(other_obs["native_tid"]),
                    "embedding": other_obs["feat"].tolist(),
                    "embedding_source": "machine_candidate_feature_for_spatial_state",
                },
            ]
        )
        current_public = probe_target_pid
    elif action == "RECOVER_IDENTITY":
        spatial.append(
            {
                "public_id": canonical,
                "box": gt_box.tolist(),
                "native_tid": -1,
                "embedding": human_feature.tolist(),
                "embedding_source": "human_roi_box",
            }
        )
    elif action == "ADD_NEW_IDENTITY":
        spatial.append(
            {
                "public_id": canonical,
                "box": gt_box.tolist(),
                "native_tid": -1,
                "embedding": human_feature.tolist(),
                "embedding_source": "human_roi_box",
            }
        )

    competitors = [
        item["feat"].tolist()
        for index, item in enumerate(observations)
        if index != target_index
    ][:16]
    event_id = f"n36-{sequence}-{frame:04d}-{action.lower()}"
    event = {
        "event_id": event_id,
        "sequence": sequence,
        "frame": frame,
        "public_id": canonical,
        "canonical_public_id": canonical,
        "event_type": action,
        "action_type": action,
        "dataset_gt_id": gid,
        "interaction_source": "simulated_from_gt",
        "future_gt_used_runtime": False,
        "gt_box": gt_box.tolist(),
        "current_public_id": current_public,
        "other_gt_box": None if other_gt_box is None else other_gt_box.tolist(),
        "other_canonical_public_id": other_canonical,
        "other_auto_tid": other_auto,
        "target_native_tid": None if target_obs is None else int(target_obs["native_tid"]),
        "target_candidate_iou": float(target_iou),
        "quality": 1.0,
        "human_feature_source": "HumanFeatureExtractor.extract(train/train_fold/sequence/img1, frame, gt_box)",
        "human_feature_checkpoint": str(HUMAN_CHECKPOINT),
        "human_feature_digest": digest_feature(human_feature),
        "human_embedding": human_feature.tolist(),
        "competing_embeddings": competitors,
        "spatial_corrections": spatial,
        "n8_event_type": n8_event.get("event_type"),
        "n8_current_public_id": n8_event.get("current_public_id"),
        "n8_canonical_public_id": n8_event.get("canonical_public_id"),
    }
    return {
        "event": event,
        "prefix_state": prefix,
        "event_pre_rows_offline": convert_pre_rows(info["pre_rows"]),
        "event_pre_rows_probe": [[int(pid), box.tolist()] for pid, box in probe_rows],
        "event_current_candidate_count": len(observations),
        "source_tape": f"outputs/n36/real_tape/frames/{sequence}.jsonl",
        "sequence_frame_count": int(info["frame_count"]),
        "future_frame_start": frame + 1,
        "future_frame_end": int(info["frame_count"]) - 1,
        "candidate_complete_source": True,
        "interaction_source": "simulated_from_gt",
        "runtime_gt_used": False,
    }


def build_sequence(sequence: str, tape_root: Path, dataset: DanceTrackDataset, extractor: HumanFeatureExtractor) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = tape_root / f"{sequence}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    backbone, frame_count = load_backbone(path)
    gt_frames = load_gt(dataset, sequence)
    observer = N8TemporalObserver(
        backbone,
        gt_frames,
        frame_count,
        N8Config(budget=-1, match_iou_threshold=0.5, sequence=sequence),
        sequence=sequence,
    )
    map_before_by_frame: dict[int, dict[int, int]] = {}
    infos: list[dict[str, Any]] = []
    for frame in range(frame_count):
        raw = backbone.get(frame, [])
        map_before = dict(observer.canonical_map)
        map_before_by_frame[frame] = map_before
        pre_rows = observer._assemble_pre(raw)
        observer.pre_rows[frame] = pre_rows
        events = observer._detect_errors(frame, raw, gt_frames.get(frame, GTFrame()))
        selected = [
            event
            for event in events
            if event.get("interaction_required") and event.get("action_type") in ACTION_TYPES
        ]
        observer._apply_events(frame, events)
        for event in selected:
            event_copy = copy.deepcopy(event)
            infos.append(
                {
                    "event": event_copy,
                    "event_id": str(event_copy.get("event_id") or f"n8-{sequence}-{frame}"),
                    "action_type": str(event_copy["action_type"]),
                    "dataset_gt_id": int(event_copy["dataset_gt_id"]),
                    "frame": frame,
                    "frame_count": frame_count,
                    "pre_rows": copy.deepcopy(pre_rows),
                }
            )
        observer.post_rows[frame] = observer._assemble_post(frame, raw, gt_frames.get(frame, GTFrame()))
    item = None
    chosen = None
    selection_rejections: list[str] = []
    for candidate in ranked_infos(infos, sequence, frame_count):
        prefix = build_prefix(path, int(candidate["frame"]), map_before_by_frame)
        if not prefix and candidate["action_type"] != "ADD_NEW_IDENTITY":
            selection_rejections.append(f"{candidate['event_id']}:prefix_empty")
            continue
        canonical_n8 = candidate["event"].get("canonical_public_id")
        prefix_ids = {int(state["public_id"]) for state in prefix}
        if candidate["action_type"] != "ADD_NEW_IDENTITY" and (
            canonical_n8 is None or public_id(int(canonical_n8)) not in prefix_ids
        ):
            selection_rejections.append(f"{candidate['event_id']}:canonical_absent_from_prefix")
            continue
        if candidate["action_type"] == "ATOMIC_ID_SWAP":
            other = candidate["event"].get("other_canonical_public_id")
            if other is None or public_id(int(other)) not in prefix_ids:
                selection_rejections.append(f"{candidate['event_id']}:other_canonical_absent_from_prefix")
                continue
        observations = current_rows(path, int(candidate["frame"]))
        try:
            item = build_event(
                candidate,
                prefix,
                observations,
                gt_frames,
                sequence,
                DATA_ROOT / "train" / sequence / "img1",
                extractor,
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            selection_rejections.append(
                f"{candidate['event_id']}:{type(exc).__name__}:{exc}"
            )
            continue
        chosen = candidate
        break
    if item is None or chosen is None:
        actions = sorted(set(info["action_type"] for info in infos))
        raise RuntimeError(
            f"no legal eligible interaction event for {sequence}; actions={actions}; "
            f"rejections={selection_rejections[:8]}"
        )
    sequence_items = [item]
    # dancetrack0012 has a later, valid true-new identity after its initial
    # empty window.  Keep it as a second event so the real transaction loop
    # exercises ADD_NEW_IDENTITY without accepting the frame-0 empty-prefix
    # artifact rejected above.
    if sequence == "dancetrack0012":
        for candidate in ranked_infos(infos, sequence, frame_count):
            if candidate["action_type"] != "ADD_NEW_IDENTITY" or int(candidate["frame"]) == int(chosen["frame"]):
                continue
            extra_prefix = build_prefix(path, int(candidate["frame"]), map_before_by_frame)
            extra_observations = current_rows(path, int(candidate["frame"]))
            try:
                extra_item = build_event(
                    candidate,
                    extra_prefix,
                    extra_observations,
                    gt_frames,
                    sequence,
                    DATA_ROOT / "train" / sequence / "img1",
                    extractor,
                )
            except (KeyError, RuntimeError, ValueError) as exc:
                selection_rejections.append(
                    f"{candidate['event_id']}:extra:{type(exc).__name__}:{exc}"
                )
                continue
            sequence_items.append(extra_item)
            break
    summary = {
        "sequence": sequence,
        "frame_count": frame_count,
        "candidate_tape": str(path.relative_to(ROOT)),
        "n8_event_count": len(infos),
        "n8_action_counts": {
            action: sum(1 for info in infos if info["action_type"] == action)
            for action in ACTION_TYPES
        },
        "selected_event_id": item["event"]["event_id"],
        "selected_action_type": item["event"]["action_type"],
        "selected_frame": int(item["event"]["frame"]),
        "prefix_state_count": len(prefix),
        "current_candidate_count": len(observations),
        "selection_rejections_before_pass": selection_rejections,
    }
    summary["event_count"] = len(sequence_items)
    return sequence_items, summary


def run(sequence_list: list[str], tape_root: Path, output: Path) -> dict[str, Any]:
    dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sequence_list, split="train")
    extractor = HumanFeatureExtractor(HUMAN_CHECKPOINT)
    events = []
    summaries = []
    for sequence in sequence_list:
        sequence_items, summary = build_sequence(sequence, tape_root, dataset, extractor)
        events.extend(sequence_items)
        summaries.append(summary)
    action_counts = {action: sum(item["event"]["action_type"] == action for item in events) for action in ACTION_TYPES}
    payload = {
        "protocol": "N36_REAL_OFFLINE_HUMAN_EVENT_MANIFEST",
        "status": "PASS" if len(events) >= 6 and len({item["event"]["sequence"] for item in events}) >= 6 else "PARTIAL",
        "split": "train/train_fold",
        "candidate_tape_candidate_complete": True,
        "interaction_source": "simulated_from_gt",
        "gt_used_only_offline": True,
        "runtime_gt_used": False,
        "event_count": len(events),
        "independent_sequence_count": len({item["event"]["sequence"] for item in events}),
        "action_counts": action_counts,
        "replay_horizon": REPLAY_HORIZON,
        "human_feature": {
            "extractor": "HumanFeatureExtractor",
            "source": "explicit human GT box crop; no machine candidate feature used as human embedding",
            "checkpoint": str(HUMAN_CHECKPOINT),
            "feature_dim": FEATURE_DIM,
        },
        "events": events,
        "sequence_summaries": summaries,
        "builder": "scripts/run_n36_build_events.py",
    }
    atomic_json(output, jsonable(payload))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", default="dancetrack0001,dancetrack0002,dancetrack0006,dancetrack0008,dancetrack0012,dancetrack0015")
    parser.add_argument("--tape-root", type=Path, default=DEFAULT_TAPE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sequences = sorted({item.strip() for item in args.sequences.split(",") if item.strip()})
    payload = run(sequences, args.tape_root, args.output)
    try:
        output_display = str(args.output.resolve().relative_to(ROOT))
    except ValueError:
        output_display = str(args.output.resolve())
    print(json.dumps({"status": payload["status"], "event_count": payload["event_count"], "action_counts": payload["action_counts"], "output": output_display}, sort_keys=True))


if __name__ == "__main__":
    main()
