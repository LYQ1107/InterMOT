#!/usr/bin/env python3
"""Export one isolated real-SAM3 candidate window for N71.

The exporter is intentionally independent of N36/N70.  It does not import
annotations, events, public IDs, or replay labels; a window plan only selects
the frame range.  The new branch uses the official project backend with a
larger candidate capacity and preserves a window-local mapping because no
public-ID mapping is fabricated for a new candidate stream.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n36_tape_common import CHECKPOINT, DATA_ROOT, atomic_json, encode_mask, image_files


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    return value


def load_plan(path: Path, window_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload.get("windows", []):
        if str(item.get("window_id")) == window_id:
            return dict(item)
    raise KeyError(f"window_id not found: {window_id}")


def temp_jsonl(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    return os.fdopen(fd, "w", encoding="utf-8"), name


def export_window(item: dict[str, Any], output_root: Path, gpu_index: int) -> dict[str, Any]:
    from sam3_intermot.backend.sam3_backend import Sam3Backend
    from scripts.run_n35_export_tape import FrozenMachineOSNet

    sequence = str(item["sequence"])
    window_id = str(item["window_id"])
    sequence_dir = DATA_ROOT / "train" / sequence
    paths = image_files(sequence_dir)
    frame_count = len(paths)
    if frame_count != int(item["frame_count_total"]):
        raise RuntimeError(f"frame count changed for {sequence}: {frame_count} != {item['frame_count_total']}")
    frame_start, frame_end = int(item["frame_start"]), int(item["frame_end"])
    if not (0 <= frame_start <= frame_end < frame_count):
        raise ValueError(f"invalid range {item}")
    output = output_root / "windows" / f"{window_id}.jsonl"
    done = output_root / "done" / f"{window_id}.json"
    handle = None; temp_name = None; backend = None; encoder = None
    seen: set[int] = set(); native_last: dict[int, int] = {}; candidate_count = 0
    started = time.time()
    try:
        backend = Sam3Backend(
            checkpoint_path=str(CHECKPOINT), max_num_objects=16, multiplex_count=16,
            use_fa3=False, use_rope_real=True, compile=False, warm_up=False,
            output_prob_thresh=0.30, async_loading_frames=False, device="cuda",
        )
        backend.start_video(str(sequence_dir / "img1"))
        encoder = FrozenMachineOSNet(f"cuda:{gpu_index}")
        handle, temp_name = temp_jsonl(output)

        def write_frame(frame_idx: int, observations: list[Any]) -> None:
            nonlocal candidate_count
            frame_idx = int(frame_idx)
            if frame_idx in seen or not (frame_start <= frame_idx <= frame_end):
                return
            boxes = [np.asarray(obs.box_xyxy, dtype=float).copy() for obs in observations]
            features = encoder.encode(paths[frame_idx], boxes)
            if features.shape != (len(observations), 512) or not np.all(np.isfinite(features)):
                raise RuntimeError(f"nonfinite or malformed 512-D features at {sequence}:{frame_idx}")
            backend._output_cache[frame_idx] = [obs.copy() for obs in observations]
            try:
                exported = backend.export_frame_candidates(frame_idx, embeddings=features, include_masks=True)
            finally:
                backend._output_cache.pop(frame_idx, None)
            candidates = []
            native_seen: set[int] = set()
            for index, candidate in enumerate(exported):
                native = int(candidate["native_tid"])
                if native in native_seen: raise RuntimeError(f"duplicate native id {native} at {sequence}:{frame_idx}")
                native_seen.add(native)
                vector = np.asarray(candidate["embedding"], dtype=np.float32).reshape(-1)
                if vector.shape != (512,) or not np.all(np.isfinite(vector)) or float(np.linalg.norm(vector)) <= 1e-6:
                    raise RuntimeError(f"invalid feature at {sequence}:{frame_idx}:{index}")
                age = float(frame_idx - native_last[native]) if native in native_last else 0.0
                native_last[native] = frame_idx
                # Public identity is deliberately null: this new stream has no
                # proven native/local/global-to-public bridge and must not use GT.
                candidates.append({
                    "candidate_index": int(index), "native_tid": native, "local_id": native,
                    "global_id": f"{window_id}:g{native}", "global_id_scope": "window_local_until_overlap_audit",
                    "native_id_source": "official_out_obj_ids", "native_age": age,
                    "box": np.asarray(candidate["box_xyxy"], dtype=float).reshape(4).tolist(),
                    "mask": encode_mask(candidate.get("mask")), "confidence": float(candidate.get("confidence", 0.0)),
                    "presence_score": None if candidate.get("presence_score") is None else float(candidate["presence_score"]),
                    "machine_embedding": vector.tolist(), "embedding_dim": 512,
                    "embedding_source": "FrozenMachineOSNet", "mapping": {
                        "native_id": native, "local_id": native, "global_id": f"{window_id}:g{native}",
                        "public_id": None, "public_id_status": "EXPLICIT_NEW_BRANCH_PUBLIC_MAPPING_UNAVAILABLE",
                        "mapping_source": "N71_official_window_export_no_fabricated_public_mapping",
                        "runtime_future_gt_used": False,
                    },
                })
            frame_hash = sha256(paths[frame_idx])
            row = {
                "schema": "N71_OFFICIAL_SAM3_CANDIDATE_FRAME_V1", "record_type": "candidate_frame",
                "branch": "D_NEW_SAM3_CANDIDATE_BRANCH", "window_id": window_id, "sequence": sequence,
                "frame": frame_idx, "frame_hash_sha256": frame_hash, "frame_start": frame_start, "frame_end": frame_end,
                "core_start": int(item["core_start"]), "core_end": int(item["core_end"]),
                "is_core_frame": bool(int(item["core_start"]) <= frame_idx <= int(item["core_end"])),
                "candidate_count": len(candidates), "candidate_order": [int(c["candidate_index"]) for c in candidates],
                "candidate_set_complete": True, "candidates": candidates,
                "runtime_future_gt_used": False, "runtime_gt_read": False,
                "interaction_source": "simulated_from_gt", "not_real_human_evidence": True,
                "public_mapping_status": "EXPLICITLY_UNAVAILABLE_NOT_FABRICATED",
                "runtime_memory_policy": backend.runtime_memory_policy(),
            }
            handle.write(json.dumps(jsonable(row), sort_keys=True, allow_nan=False) + "\n")
            handle.flush(); seen.add(frame_idx); candidate_count += len(candidates)

        initial = backend.detect_concept(frame_start, "person")
        if initial: write_frame(frame_start, initial)
        backend.propagate(frame_start, frame_end, start_frame_index=frame_start, keep_masks=True, cache_outputs=False, output_callback=write_frame)
        expected = set(range(frame_start, frame_end + 1))
        if seen != expected:
            raise RuntimeError(f"window coverage mismatch missing={sorted(expected-seen)[:10]} extra={sorted(seen-expected)[:10]}")
        handle.flush(); os.fsync(handle.fileno()); handle.close(); handle = None
        os.replace(temp_name, output); temp_name = None
        result = {
            "schema": "N71_OFFICIAL_SAM3_CANDIDATE_WINDOW_DONE_V1", "status": "PASS", "window_id": window_id,
            "sequence": sequence, "frame_start": frame_start, "frame_end": frame_end,
            "written_frame_count": len(seen), "candidate_count": candidate_count, "feature_dim": 512,
            "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT),
            "candidate_generator_changed": True, "checkpoint_changed": False,
                    "settings": {"max_num_objects": 16, "multiplex_count": 16, "output_prob_thresh": 0.30, "async_loading_frames": False},
            "runtime_memory_policy": backend.runtime_memory_policy(), "runtime_future_gt_used": False,
            "public_mapping_status": "EXPLICITLY_UNAVAILABLE_NOT_FABRICATED", "output": str(output), "elapsed_sec": time.time()-started,
            "process_isolation": "one_python_process_one_sam3_session_one_frame_range",
        }
        atomic_json(done, result); return result
    except Exception as exc:
        failure = {"schema": "N71_OFFICIAL_SAM3_CANDIDATE_WINDOW_DONE_V1", "status": "FAIL", "window_id": window_id, "sequence": sequence, "frame_start": frame_start, "frame_end": frame_end, "written_frame_count": len(seen), "candidate_count": candidate_count, "failure_type": type(exc).__name__, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "is_oom": "outofmemory" in type(exc).__name__.lower() or "out of memory" in str(exc).lower(), "runtime_future_gt_used": False, "elapsed_sec": time.time()-started}
        atomic_json(done, failure); raise
    finally:
        if handle is not None:
            try: handle.close()
            except Exception: pass
        if temp_name and os.path.exists(temp_name): os.unlink(temp_name)
        if backend is not None:
            try: backend.close()
            except Exception: pass
        del encoder, backend
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/path/to/cache/SAM3_InterMOT_N71/candidate_branch"))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("N71 candidate export requires CUDA")
    torch.cuda.set_device(int(args.gpu))
    plan = load_plan(args.plan.resolve(), args.window_id)
    print(json.dumps({"schema": "N71_CANDIDATE_EXPORT_START_V1", "window": plan, "gpu_local_index": args.gpu, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")}, sort_keys=True), flush=True)
    print(json.dumps(export_window(plan, args.output_root.resolve(), int(args.gpu)), sort_keys=True), flush=True)


if __name__ == "__main__": main()
