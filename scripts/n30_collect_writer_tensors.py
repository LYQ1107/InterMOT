#!/usr/bin/env python3
"""Collect real official decoder support/future tensors for N30-C/D."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SAM3_ROOT = ROOT / "third_party" / "sam3"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

from sam3_intermot.adaptation.corrected_mask_teacher import BOX_DERIVED_PSEUDO_MASK  # noqa: E402
from scripts.n29_lit_online_replay import (  # noqa: E402
    _clone_tree,
    _get_official_decoder,
    _image_files,
    _install_official_box_singleton,
    _make_backend,
    _read_gt,
    _session,
)
from scripts.n29r_paired_replay import _ensure_public_singleton_binding  # noqa: E402
from scripts.n29r_real_association import FrozenClipReID  # noqa: E402


CHECKPOINT = ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
TRAIN_ROOT = Path("/path/to/dancetrack/train")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


class FutureInputTape:
    """Record every real decoder kwargs mapping during one future stream."""

    def __init__(self, decoder: torch.nn.Module) -> None:
        self.records: list[dict[str, Any]] = []
        self.handle = decoder.register_forward_pre_hook(self._pre, with_kwargs=True)

    def reset(self) -> None:
        self.records = []

    def _pre(self, _module: torch.nn.Module, _args: tuple[Any, ...], kwargs: dict[str, Any]):
        self.records.append(_cpu_tree(kwargs))

    def close(self) -> None:
        self.handle.remove()


def _clean_singleton_cache(backend: Any, public_id: int) -> list[int]:
    state = backend._predictor._all_inference_states[backend._session_id]["state"]
    removed: set[int] = set()
    filtered: dict[int, dict[Any, Any]] = {}
    for frame, frame_cache in state.get("cached_frame_outputs", {}).items():
        kept = {}
        for object_id, mask in frame_cache.items():
            if int(object_id) == int(public_id):
                kept[object_id] = mask
            else:
                removed.add(int(object_id))
        if kept:
            filtered[int(frame)] = kept
    state["cached_frame_outputs"] = filtered
    state["action_history"] = [{"type": "add", "obj_ids": [int(public_id)], "frame_idx": int(state.get("last_prompt_frame", 0))}]
    return sorted(removed)


def _set_action_history(backend: Any, public_id: int, frame: int) -> None:
    state = backend._predictor._all_inference_states[backend._session_id]["state"]
    state["action_history"] = [{"type": "add", "obj_ids": [int(public_id)], "frame_idx": int(frame)}]


def _select_observation(outputs: Mapping[int, list[Any]], frame: int, public_id: int) -> Any | None:
    for observation in outputs.get(frame, []):
        if int(getattr(observation, "sam_object_id", -1)) == int(public_id):
            return observation
    return None


def _required_keys(kwargs: Mapping[str, Any]) -> tuple[str, ...]:
    required = ("image_embeddings", "image_pe", "high_res_features", "extra_per_object_embeddings")
    return tuple(key for key in required if key in kwargs)


def _run_episode(
    *,
    backend: Any,
    decoder: torch.nn.Module,
    tape: FutureInputTape,
    clip: FrozenClipReID,
    episode: Mapping[str, Any],
    train_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    sequence_dir = train_root / str(episode["parent_sequence"])
    images = _image_files(sequence_dir)
    gt = _read_gt(sequence_dir)
    init = int(episode["initialization_frame"])
    correction = int(episode["correction_frame"])
    query_frames = [int(value) for value in episode["query_frames"]]
    public_id = int(episode["public_id"])
    anchor_box = np.asarray(episode["anchor_box"], dtype=float)
    legal_start_box = np.asarray(gt[init][int(episode["dataset_identity"])], dtype=float)
    correction_box = np.asarray(episode["correction_box"], dtype=float)

    support_records: list[dict[str, Any]] = []
    pre_outputs: dict[int, list[Any]] = {}
    anchor_binding: dict[str, Any] = {}
    data_anchor_box = anchor_box
    data_anchor_override = False
    for attempt, candidate_anchor in enumerate((anchor_box, legal_start_box)):
        _session(backend, sequence_dir)
        backend.add_box(init, public_id, candidate_anchor)
        # Some official semantic box prompts create a transient raw tracker
        # slot (for example raw id 0) before the adapter-level public-id
        # binding is installed.  Reconcile that namespace before filtering
        # caches; otherwise the subsequent propagation has a tracker state
        # but no public-id support tape.
        anchor_binding = _ensure_public_singleton_binding(
            backend,
            frame=init,
            public_id=public_id,
            box=candidate_anchor,
        )
        _clean_singleton_cache(backend, public_id)
        _set_action_history(backend, public_id, init)
        tape.reset()
        pre_outputs = backend.propagate(init, correction, start_frame_index=init)
        support_records = list(tape.records)
        if support_records:
            data_anchor_box = candidate_anchor
            data_anchor_override = attempt > 0
            break
    if not support_records:
        raise RuntimeError("official decoder emitted no support input record for either declared or legal-start anchor")
    support_kwargs = support_records[min(max(correction - init, 0), len(support_records) - 1)]
    if "extra_per_object_embeddings" not in support_kwargs or "image_embeddings" not in support_kwargs:
        raise RuntimeError(f"support input is missing required official fields: {sorted(support_kwargs)}")
    predicted = _select_observation(pre_outputs, correction, public_id)
    predicted_box = anchor_box if predicted is None else np.asarray(predicted.box_xyxy, dtype=float)
    current_output_recorded = predicted is not None

    backend.correct_object(correction, public_id, box_xyxy=correction_box)
    correction_binding = _ensure_public_singleton_binding(
        backend,
        frame=correction,
        public_id=public_id,
        box=correction_box,
    )
    _clean_singleton_cache(backend, public_id)
    _set_action_history(backend, public_id, correction)
    tape.reset()
    future_outputs = backend.propagate(
        query_frames[0],
        query_frames[-1],
        start_frame_index=query_frames[0],
    )
    future_records = list(tape.records)
    if len(future_records) < len(query_frames):
        raise RuntimeError(
            f"future decoder tape is short: records={len(future_records)} expected={len(query_frames)}"
        )
    if len(future_records) > len(query_frames):
        future_records = future_records[: len(query_frames)]
    clip_feature = clip.encode(images[correction], [correction_box])[0]
    payload = {
        "episode_id": str(episode["episode_id"]),
        "video_id": str(episode["video_id"]),
        "public_id": public_id,
        "target_slot": 0,
        "parent_sequence": str(episode["parent_sequence"]),
        "role": str(episode["role"]),
        "initialization_frame": init,
        "correction_frame": correction,
        "future_frames": query_frames,
        "anchor_box": anchor_box,
        "data_collection_anchor_box": data_anchor_box,
        "data_collection_anchor_override": data_anchor_override,
        "predicted_box_at_correction": predicted_box,
        "correction_box": correction_box,
        "future_target_boxes": [np.asarray(episode["query_target_boxes"][str(frame)], dtype=float) for frame in query_frames],
        "clip_feature": np.asarray(clip_feature, dtype=np.float32),
        "image_size": tuple(int(value) for value in episode["image_size"]),
        "support_kwargs": _cpu_tree(support_kwargs),
        "future_kwargs": [_cpu_tree(record) for record in future_records],
        "support_required_keys": list(_required_keys(support_kwargs)),
        "future_required_keys": [_required_keys(record) for record in future_records],
        "support_decoder_record_count": len(support_records),
        "future_decoder_record_count": len(tape.records),
        "current_output_recorded": current_output_recorded,
        "anchor_binding": anchor_binding,
        "correction_binding": correction_binding,
        "correction_provenance": BOX_DERIVED_PSEUDO_MASK,
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_writer_input": False,
    }
    sample_path = output_dir / f"{episode['episode_id']}.pt"
    torch.save(payload, sample_path)
    return {
        "status": "PASS",
        "episode_id": str(episode["episode_id"]),
        "role": str(episode["role"]),
        "parent_sequence": str(episode["parent_sequence"]),
        "sample_path": str(sample_path),
        "support_decoder_record_count": len(support_records),
        "future_decoder_record_count": len(tape.records),
        "future_frame_count": len(future_records),
        "data_collection_anchor_override": data_anchor_override,
        "support_required_keys": list(_required_keys(support_kwargs)),
        "future_required_keys": [list(_required_keys(record)) for record in future_records],
        "current_output_recorded": current_output_recorded,
        "future_gt_used_for_writer_input": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-manifest", type=Path, default=ROOT / "outputs/n30/episode_manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--train-root", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/n30/writer_dataset")
    parser.add_argument("--index-output", type=Path, default=ROOT / "outputs/n30/writer_dataset_index.json")
    parser.add_argument("--roles", nargs="+", default=["meta_train", "selection", "calibration"])
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--episode-id", action="append", default=None)
    parser.add_argument(
        "--merge-index",
        type=Path,
        default=None,
        help="merge newly collected episode records with an existing index after a targeted retry",
    )
    args = parser.parse_args()
    manifest = json.loads(args.episode_manifest.read_text(encoding="utf-8"))
    episodes = [item for item in manifest["episodes"] if item.get("role") in set(args.roles)]
    if args.episode_id is not None:
        requested_ids = set(args.episode_id)
        episodes = [item for item in episodes if str(item["episode_id"]) in requested_ids]
    if args.episode_limit is not None:
        episodes = episodes[: args.episode_limit]
    if not episodes:
        raise RuntimeError("no requested episodes in frozen manifest")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backend = _make_backend(args.checkpoint)
    decoder = None
    tape = None
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    try:
        clip = FrozenClipReID(torch.device("cuda"))
        backend._ensure_model()
        decoder = _get_official_decoder(backend)
        tape = FutureInputTape(decoder)
        for index, episode in enumerate(episodes, start=1):
            print(f"N30_WRITER_DATA_START {index}/{len(episodes)} {episode['episode_id']}", flush=True)
            try:
                record = _run_episode(
                    backend=backend,
                    decoder=decoder,
                    tape=tape,
                    clip=clip,
                    episode=episode,
                    train_root=args.train_root,
                    output_dir=args.output_dir,
                )
            except Exception as exc:
                record = {
                    "status": "NOT_RUN",
                    "episode_id": str(episode["episode_id"]),
                    "role": str(episode["role"]),
                    "parent_sequence": str(episode["parent_sequence"]),
                    "failure": f"{type(exc).__name__}: {exc}",
                    "failure_traceback": traceback.format_exc(limit=16),
                    "future_gt_used_for_writer_input": False,
                }
            records.append(record)
            print(f"N30_WRITER_DATA_DONE {index}/{len(episodes)} {record['status']}", flush=True)
    finally:
        if tape is not None:
            tape.close()
        backend.close()
    result_records = records
    result_roles = list(args.roles)
    if args.merge_index is not None:
        existing = json.loads(args.merge_index.read_text(encoding="utf-8"))
        if existing.get("episode_manifest") != str(args.episode_manifest):
            raise ValueError("targeted retry manifest does not match the existing writer index")
        if existing.get("output_dir") != str(args.output_dir):
            raise ValueError("targeted retry output directory does not match the existing writer index")
        existing_records = {
            str(item["episode_id"]): item
            for item in existing.get("records", [])
            if "episode_id" in item
        }
        current_records = {str(item["episode_id"]): item for item in records}
        result_records = []
        for episode in manifest["episodes"]:
            episode_id = str(episode["episode_id"])
            item = current_records.get(episode_id, existing_records.get(episode_id))
            if item is None:
                raise ValueError(f"missing record for frozen episode {episode_id}")
            if item.get("status") == "PASS" and item.get("sample_path"):
                if not Path(item["sample_path"]).exists():
                    raise ValueError(f"missing PASS sample for frozen episode {episode_id}")
            result_records.append(item)
        result_roles = list(existing.get("roles", args.roles))
    result = {
        "protocol": "N30-C/D-REAL-OFFICIAL-DECODER-TAPE",
        "status": "PASS" if result_records and all(item["status"] == "PASS" for item in result_records) else "PARTIAL",
        "episode_manifest": str(args.episode_manifest),
        "output_dir": str(args.output_dir),
        "roles": result_roles,
        "episode_count_requested": len(result_records),
        "episode_count_pass": sum(item["status"] == "PASS" for item in result_records),
        "episode_count_failed": sum(item["status"] != "PASS" for item in result_records),
        "records": result_records,
        "frozen_split_sha256": manifest.get("manifest_sha256"),
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_writer_input": False,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _write(args.index_output, result)
    print(json.dumps({key: result[key] for key in ("status", "episode_count_requested", "episode_count_pass", "episode_count_failed", "elapsed_seconds")}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
