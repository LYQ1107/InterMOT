#!/usr/bin/env python3
"""Export one N36 frame-range shard in one independently exited process.

The child owns exactly one SAM3 session.  It writes a temporary JSONL while
the official stream is consumed, atomically renames that file only after the
inclusive range is complete, and then writes an atomic done manifest.  No
annotation file is opened by this runtime exporter.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n36_tape_common import (  # noqa: E402
    CHECKPOINT,
    DATA_ROOT,
    atomic_json,
    display_path,
    encode_mask,
    image_files,
)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def load_chunk(plan_path: Path, chunk_id: str) -> dict[str, Any]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    for item in payload.get("all_chunks", []):
        if str(item.get("chunk_id")) == str(chunk_id):
            return dict(item)
    raise KeyError(f"chunk_id not found in plan: {chunk_id}")


def candidate_to_json(candidate: dict[str, Any], native_age: float) -> dict[str, Any]:
    embedding = candidate.get("embedding")
    vector = None if embedding is None else np.asarray(embedding, dtype=np.float32).reshape(-1)
    vector_list = None if vector is None else vector.tolist()
    return {
        "candidate_index": int(candidate["candidate_index"]),
        "local_native_id": int(candidate["native_tid"]),
        "native_tid": int(candidate["native_tid"]),
        "native_id_source": str(candidate.get("native_id_source", "official_out_obj_ids")),
        "native_age": float(native_age),
        "box": np.asarray(candidate["box_xyxy"], dtype=float).reshape(-1).tolist(),
        "mask": encode_mask(candidate.get("mask")),
        "confidence": float(candidate.get("confidence", 0.0)),
        "presence_score": (
            None
            if candidate.get("presence_score") is None
            else float(candidate["presence_score"])
        ),
        "source": str(candidate.get("source", "automatic_propagation")),
        "machine_embedding": vector_list,
        "embedding_dim": None if vector is None else int(vector.size),
        "embedding_status": str(candidate.get("embedding_status", "NOT_EXPOSED")),
        "feature_source": str(candidate.get("feature_source", "official_response_no_embedding")),
        "is_human_verified": bool(candidate.get("is_human_verified", False)),
    }


def manager_observation(candidate: dict[str, Any], native_age: float) -> dict[str, Any]:
    return {
        "obs_id": int(candidate["candidate_index"]),
        "feat": np.asarray(candidate["embedding"], dtype=np.float32).reshape(-1),
        "has_feat": 1.0,
        "box": np.asarray(candidate["box_xyxy"], dtype=float).copy(),
        "native_tid": int(candidate["native_tid"]),
        "native_age": float(native_age),
        "conf": float(candidate.get("confidence", 0.0)),
    }


def _temporary_jsonl(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    return os.fdopen(fd, "w", encoding="utf-8"), temp_name


def export_chunk(
    chunk: dict[str, Any],
    output_root: Path,
    gpu_local_index: int,
    use_features: bool,
) -> dict[str, Any]:
    from sam3_intermot.association.state_manager import StateManager, StateManagerConfig
    from sam3_intermot.backend.sam3_backend import Sam3Backend

    sequence = str(chunk["sequence"])
    sequence_dir = DATA_ROOT / "train" / sequence
    paths = image_files(sequence_dir)
    frame_count = len(paths)
    expected_frame_count = int(chunk["frame_count_total"])
    if frame_count != expected_frame_count:
        raise RuntimeError(
            f"frame count changed for {sequence}: plan={expected_frame_count}, current={frame_count}"
        )
    frame_start = int(chunk["frame_start"])
    frame_end = int(chunk["frame_end"])
    core_start = int(chunk["core_frame_start"])
    core_end = int(chunk["core_frame_end"])
    if not (0 <= frame_start <= core_start <= core_end <= frame_end < frame_count):
        raise ValueError(f"invalid chunk range: {chunk}")

    chunk_id = str(chunk["chunk_id"])
    frame_path = output_root / "chunks" / sequence / f"{chunk_id}.jsonl"
    done_path = output_root / "chunk_done" / sequence / f"{chunk_id}.json"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    backend = None
    encoder = None
    manager = None
    handle = None
    temp_name: str | None = None
    seen_frames: set[int] = set()
    native_last_seen: dict[int, int] = {}
    written_candidate_count = 0

    try:
        backend = Sam3Backend(
            checkpoint_path=str(CHECKPOINT),
            max_num_objects=16,
            multiplex_count=16,
            use_fa3=False,
            use_rope_real=True,
            compile=False,
            warm_up=False,
            async_loading_frames=False,
            device="cuda",
        )
        # N35's failed nested CPU state repair is intentionally not enabled by
        # this exporter.  The adapter reports the official state defaults;
        # process exit, not mixed-device state, is the memory boundary.
        backend.start_video(str(sequence_dir / "img1"))
        if use_features:
            from scripts.run_n35_export_tape import FrozenMachineOSNet

            encoder = FrozenMachineOSNet(f"cuda:{gpu_local_index}")
        manager = StateManager(
            StateManagerConfig(
                variant="reid",
                score_threshold=0.0,
                max_lost_gap=90,
                use_appearance_memory=False,
            )
        )
        handle, temp_name = _temporary_jsonl(frame_path)

        def write_frame(frame: int, observations: list[Any]) -> None:
            nonlocal written_candidate_count
            frame = int(frame)
            if frame in seen_frames:
                return
            if frame < frame_start or frame > frame_end:
                return
            boxes = [np.asarray(item.box_xyxy, dtype=float).copy() for item in observations]
            features = (
                encoder.encode(paths[frame], boxes)
                if encoder is not None
                else np.zeros((len(observations), 0), dtype=np.float32)
            )
            # propagate(cache_outputs=False) deliberately does not retain the
            # whole range in the adapter.  This one-frame bridge is removed
            # immediately after export_frame_candidates returns.
            backend._output_cache[frame] = [item.copy() for item in observations]
            try:
                exported = backend.export_frame_candidates(
                    frame,
                    embeddings=(features if encoder is not None else None),
                    include_masks=True,
                )
            finally:
                backend._output_cache.pop(frame, None)
            manager_obs: list[dict[str, Any]] = []
            json_candidates: list[dict[str, Any]] = []
            for index, candidate in enumerate(exported):
                native_tid = int(candidate["native_tid"])
                age = (
                    float(frame - native_last_seen[native_tid])
                    if native_tid in native_last_seen
                    else 0.0
                )
                native_last_seen[native_tid] = frame
                json_candidates.append(candidate_to_json(candidate, age))
                if candidate.get("embedding") is None:
                    raise RuntimeError(
                        f"machine feature missing at {sequence}:{frame}:{index}"
                    )
                manager_obs.append(manager_observation(candidate, age))
            if len(manager_obs) != len(exported):
                raise RuntimeError("candidate export dropped a feature-missing candidate")
            rows = manager.rollout_frame(frame, manager_obs, model=None)
            if not manager.candidate_log:
                raise RuntimeError(f"StateManager produced no audit at {sequence}:{frame}")
            audit = jsonable(manager.candidate_log[-1])
            public_ids = audit.get("candidate_public_ids", [])
            if len(public_ids) != len(json_candidates) or any(pid is None for pid in public_ids):
                raise RuntimeError(
                    f"candidate/public mapping incomplete at {sequence}:{frame}: "
                    f"{len(public_ids)} vs {len(json_candidates)}"
                )
            for candidate, public_id in zip(json_candidates, public_ids):
                candidate["chunk_local_public_id"] = int(public_id)
                candidate["public_native_mapping_status"] = "CHUNK_LOCAL_EXPLICIT"
            row = {
                "record_type": "candidate_frame",
                "protocol": "N36_REAL_SHARDED_CANDIDATE_TAPE_CHUNK",
                "sequence": sequence,
                "split": "train/train_fold",
                "frame": frame,
                "chunk_id": chunk_id,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "core_frame_start": core_start,
                "core_frame_end": core_end,
                "overlap_start": chunk.get("next_overlap", {}).get("start")
                if isinstance(chunk.get("next_overlap"), dict)
                else None,
                "overlap_end": chunk.get("next_overlap", {}).get("end")
                if isinstance(chunk.get("next_overlap"), dict)
                else None,
                "overlap_with_previous": chunk.get("previous_overlap"),
                "overlap_with_next": chunk.get("next_overlap"),
                "is_core_frame": bool(core_start <= frame <= core_end),
                "candidate_complete": bool(use_features),
                "candidate_set_complete": bool(use_features),
                "candidate_set_source": "official_sam3_full_vg_post_nms_propagation",
                "runtime_future_gt_used": False,
                "runtime_gt_read": False,
                "public_id_namespace": "chunk_local_state_manager",
                "states_public_ids": audit.get("public_id_order", []),
                "candidates": json_candidates,
                "association_audit": audit,
            }
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            if len(seen_frames) % 8 == 0:
                handle.flush()
            seen_frames.add(frame)
            written_candidate_count += len(json_candidates)
            manager.candidate_log.clear()

        initial = backend.detect_concept(frame_start, "person")
        if initial:
            write_frame(frame_start, initial)
        backend.propagate(
            frame_start,
            frame_end,
            start_frame_index=frame_start,
            keep_masks=True,
            cache_outputs=False,
            output_callback=write_frame,
        )
        expected_frames = set(range(frame_start, frame_end + 1))
        if seen_frames != expected_frames:
            missing = sorted(expected_frames - seen_frames)
            extra = sorted(seen_frames - expected_frames)
            raise RuntimeError(
                f"chunk stream coverage mismatch missing={missing[:8]} extra={extra[:8]}"
            )
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        os.replace(temp_name, frame_path)
        temp_name = None
        result = {
            "protocol": "N36_REAL_SHARDED_CANDIDATE_TAPE_CHUNK_DONE",
            "sequence": sequence,
            "chunk_id": chunk_id,
            "status": "PASS",
            "candidate_complete": bool(use_features),
            "candidate_set_complete": bool(use_features),
            "frame_count_total": frame_count,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "core_frame_start": core_start,
            "core_frame_end": core_end,
            "written_frame_count": len(seen_frames),
            "candidate_count": written_candidate_count,
            "feature_dim": 512 if use_features else None,
            "feature_source": "machine_roi_fallback_osnet_market1501" if use_features else "not_exposed",
            "runtime_memory_policy": backend.runtime_memory_policy(),
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "output": display_path(frame_path),
            "elapsed_sec": time.time() - started,
            "process_isolation": "one_python_process_one_sam3_session_one_frame_range",
        }
        atomic_json(done_path, result)
        return result
    except Exception as exc:
        failure = {
            "protocol": "N36_REAL_SHARDED_CANDIDATE_TAPE_CHUNK_DONE",
            "sequence": sequence,
            "chunk_id": chunk_id,
            "status": "FAIL",
            "candidate_complete": False,
            "candidate_set_complete": False,
            "frame_count_total": frame_count,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "core_frame_start": core_start,
            "core_frame_end": core_end,
            "written_frame_count": len(seen_frames),
            "candidate_count": written_candidate_count,
            "failure_type": type(exc).__name__,
            "error": f"{type(exc).__name__}: {exc}",
            "is_oom": "outofmemory" in type(exc).__name__.lower()
            or "out of memory" in str(exc).lower(),
            "elapsed_sec": time.time() - started,
        }
        atomic_json(done_path, failure)
        raise
    finally:
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        del manager
        del encoder
        del backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--chunk-id", required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/n36/real_tape")
    parser.add_argument("--gpu", type=int, default=0, help="local CUDA index in CUDA_VISIBLE_DEVICES")
    parser.add_argument("--no-features", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    plan = args.plan.resolve()
    output_root = args.output_root.resolve()
    chunk = load_chunk(plan, args.chunk_id)
    done_path = output_root / "chunk_done" / str(chunk["sequence"]) / f"{args.chunk_id}.json"
    if args.skip_existing and done_path.is_file():
        existing = json.loads(done_path.read_text(encoding="utf-8"))
        if existing.get("status") == "PASS":
            print(json.dumps({"chunk_id": args.chunk_id, "status": "SKIP_EXISTING"}, sort_keys=True), flush=True)
            return
    if not torch.cuda.is_available():
        raise RuntimeError("N36 real SAM3 shard exporter requires CUDA")
    torch.cuda.set_device(int(args.gpu))
    print(
        json.dumps(
            {
                "protocol": "N36_REAL_SHARDED_CANDIDATE_TAPE_CHUNK",
                "sequence": chunk["sequence"],
                "chunk_id": chunk["chunk_id"],
                "frame_start": chunk["frame_start"],
                "frame_end": chunk["frame_end"],
                "core_frame_start": chunk["core_frame_start"],
                "core_frame_end": chunk["core_frame_end"],
                "gpu_local_index": int(args.gpu),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
                "features": not args.no_features,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    result = export_chunk(chunk, output_root, int(args.gpu), not args.no_features)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
