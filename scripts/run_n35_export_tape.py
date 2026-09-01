#!/usr/bin/env python3
"""Export a real per-frame SAM3 candidate-complete train-fold tape.

The runtime branch never reads DanceTrack annotations.  SAM3 full-VG
propagation supplies the complete post-NMS candidate set; a separate frozen
OSNet box-crop encoder supplies machine features when the official response
does not expose decoder tokens.  Human-event generation is intentionally a
later offline step.
"""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import os
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/path/to/dancetrack")
CHECKPOINT = ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/n35/real_tape"
DEFAULT_SEQUENCES = ROOT / "outputs/n34/selected_sequences.json"


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


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


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def image_files(sequence_dir: Path) -> list[Path]:
    return sorted(
        [path for path in (sequence_dir / "img1").iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda path: int(path.stem),
    )


def encode_mask(mask: np.ndarray | None) -> dict[str, Any] | None:
    if mask is None:
        return None
    array = np.asarray(mask, dtype=bool)
    packed = np.packbits(array.reshape(-1), bitorder="little")
    compressed = zlib.compress(packed.tobytes(), level=1)
    return {
        "encoding": "packbits_zlib_base64",
        "shape": [int(v) for v in array.shape],
        "bitorder": "little",
        "data": base64.b64encode(compressed).decode("ascii"),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


class FrozenMachineOSNet:
    """Independent machine box-crop fallback; never used for human evidence."""

    feature_dim = 512

    def __init__(self, device: str) -> None:
        from torchreid.reid.utils.feature_extractor import FeatureExtractor

        self.device = device
        self.extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            model_path=str(ROOT / "outputs/n9/checkpoints/osnet_x1_0_market1501.pth"),
            image_size=(256, 128),
            device=device,
            verbose=False,
        )

    @staticmethod
    def _crop(image: Image.Image, box: Sequence[float]) -> np.ndarray:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((8, 8, 3), dtype=np.uint8)
        return np.asarray(image.crop((x1, y1, x2, y2)), dtype=np.uint8)

    def encode(self, image_path: Path, boxes: Sequence[Sequence[float]]) -> np.ndarray:
        if not boxes:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            crops = [self._crop(image, box) for box in boxes]
        with torch.no_grad():
            values = self.extractor(crops).detach().float().cpu().numpy()
        values = np.asarray(values, dtype=np.float32).reshape(len(boxes), -1)
        if values.shape[1] != self.feature_dim or not np.all(np.isfinite(values)):
            raise RuntimeError(f"machine ROI feature has invalid shape/values: {values.shape}")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms <= 1e-6):
            raise RuntimeError("machine ROI feature has zero norm")
        return values / norms


def load_sequences(path: Path, explicit: str) -> list[str]:
    if explicit:
        return sorted({item.strip() for item in explicit.split(",") if item.strip()})
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sorted(
        str(item["sequence"])
        for item in payload.get("sequences", [])
        if isinstance(item, dict) and item.get("sequence")
    )


def shard_sequences(sequences: list[str], shard: int, num_shards: int) -> list[str]:
    if not 0 <= shard < num_shards:
        raise ValueError("shard must satisfy 0 <= shard < num_shards")
    # Greedy duration-balanced assignment keeps four GPU processes close in
    # wall time without changing sequence-level data membership.
    weighted = []
    for sequence in sequences:
        count = len(image_files(DATA_ROOT / "train" / sequence))
        weighted.append((sequence, count))
    loads = [0] * num_shards
    buckets: list[list[str]] = [[] for _ in range(num_shards)]
    for sequence, weight in sorted(weighted, key=lambda item: (-item[1], item[0])):
        target = min(range(num_shards), key=lambda index: (loads[index], index))
        buckets[target].append(sequence)
        loads[target] += weight
    return sorted(buckets[shard])


def _candidate_to_json(candidate: dict[str, Any], native_age: float) -> dict[str, Any]:
    embedding = candidate.get("embedding")
    vector = None if embedding is None else np.asarray(embedding, dtype=np.float32).reshape(-1)
    return {
        "candidate_index": int(candidate["candidate_index"]),
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
        "embedding": None if vector is None else vector.tolist(),
        "embedding_status": str(candidate.get("embedding_status", "NOT_EXPOSED")),
        "feature_source": str(candidate.get("feature_source", "official_response_no_embedding")),
        "is_human_verified": bool(candidate.get("is_human_verified", False)),
    }


def _manager_observation(candidate: dict[str, Any], native_age: float) -> dict[str, Any]:
    return {
        "obs_id": int(candidate["candidate_index"]),
        "feat": np.asarray(candidate["embedding"], dtype=np.float32).reshape(-1),
        "has_feat": 1.0,
        "box": np.asarray(candidate["box_xyxy"], dtype=float).copy(),
        "native_tid": int(candidate["native_tid"]),
        "native_age": float(native_age),
        "conf": float(candidate.get("confidence", 0.0)),
    }


def export_sequence(
    sequence: str,
    output_root: Path,
    gpu_local_index: int,
    max_frames: int,
    use_features: bool,
) -> dict[str, Any]:
    from sam3_intermot.association.state_manager import StateManager, StateManagerConfig
    from sam3_intermot.backend.sam3_backend import Sam3Backend

    sequence_dir = DATA_ROOT / "train" / sequence
    paths = image_files(sequence_dir)
    if not paths:
        raise FileNotFoundError(f"no train-fold frames for {sequence}: {sequence_dir}")
    frame_count = len(paths) if max_frames <= 0 else min(len(paths), int(max_frames))
    output_path = output_root / "frames" / f"{sequence}.jsonl"
    done_path = output_root / "done" / f"{sequence}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    backend = None
    encoder = None
    manager = None
    handle = None
    temp_name: str | None = None
    seen_frames: set[int] = set()
    written_frame_count = 0
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
        backend.start_video(str(sequence_dir / "img1"))
        if use_features:
            encoder = FrozenMachineOSNet(f"cuda:{gpu_local_index}")
        manager = StateManager(
            StateManagerConfig(
                variant="reid",
                score_threshold=0.0,
                max_lost_gap=90,
                use_appearance_memory=False,
            )
        )
        native_last_seen: dict[int, int] = {}
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=str(output_path.parent)
        )
        handle = os.fdopen(fd, "w", encoding="utf-8")

        def write_frame(
            frame: int, observations: list[Any],
        ) -> None:
            nonlocal written_frame_count, written_candidate_count
            frame = int(frame)
            if frame in seen_frames:
                return
            if frame < 0 or frame >= frame_count:
                return
            boxes = [np.asarray(item.box_xyxy, dtype=float).copy() for item in observations]
            features = (
                encoder.encode(paths[frame], boxes)
                if encoder is not None
                else np.zeros((len(observations), 0), dtype=np.float32)
            )
            # export_frame_candidates reads the complete official frame output
            # and never drops a row.  The callback itself supplies the parsed
            # observations so the adapter does not need to retain them; the
            # temporary cache is populated for this one frame only.
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
                json_candidates.append(_candidate_to_json(candidate, age))
                if candidate.get("embedding") is not None:
                    manager_obs.append(_manager_observation(candidate, age))
                else:
                    raise RuntimeError(
                        f"candidate feature missing at {sequence}:{frame}:{index}; "
                        "real N35 tape requires finite machine ROI features"
                    )
            if len(manager_obs) != len(exported):
                raise RuntimeError("candidate export dropped a feature-missing candidate")
            manager.rollout_frame(frame, manager_obs, model=None)
            audit = jsonable(manager.candidate_log[-1])
            row = {
                "record_type": "candidate_frame",
                "protocol": "N35_REAL_CANDIDATE_COMPLETE_TAPE",
                "sequence": sequence,
                "split": "train/train_fold",
                "frame": frame,
                "candidate_complete": True,
                "candidate_set_complete": True,
                "candidate_set_source": "official_sam3_full_vg_post_nms_propagation",
                "runtime_future_gt_used": False,
                "runtime_gt_read": False,
                "states_public_ids": audit.get("public_id_order", []),
                "candidates": json_candidates,
                "association_audit": audit,
            }
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            if written_frame_count % 16 == 0:
                handle.flush()
            written_frame_count += 1
            written_candidate_count += len(json_candidates)
            seen_frames.add(frame)
            # The tape is the durable audit; StateManager only needs its
            # identity states for the next frame, not an unbounded audit log.
            manager.candidate_log.clear()

        # One text prompt seeds official full-VG; subsequent frames come from
        # the same full propagation stream, so no GT or per-frame prompt is
        # introduced by this exporter.
        backend.detect_concept(0, "person")
        initial = backend.get_frame_outputs(0)
        if initial:
            write_frame(0, initial)
        backend.propagate(
            0,
            frame_count - 1,
            start_frame_index=0,
            keep_masks=True,
            cache_outputs=False,
            output_callback=write_frame,
        )
        if seen_frames != set(range(frame_count)):
            missing = sorted(set(range(frame_count)) - seen_frames)
            raise RuntimeError(f"streaming propagation did not emit frames: {missing[:8]}")
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        os.replace(temp_name, output_path)
        temp_name = None
        result = {
            "sequence": sequence,
            "status": "PASS",
            "candidate_complete": True,
            "candidate_set_complete": True,
            "frame_count": frame_count,
            "candidate_count": int(written_candidate_count),
            "feature_dim": 512 if use_features else None,
            "feature_source": "machine_roi_fallback_osnet_market1501" if use_features else "not_exposed",
            "runtime_memory_policy": backend.runtime_memory_policy(),
            "output": display_path(output_path),
            "elapsed_sec": time.time() - started,
        }
        atomic_json(done_path, result)
        return result
    except Exception as exc:
        failure = {
            "sequence": sequence,
            "status": "FAIL",
            "candidate_complete": False,
            "candidate_set_complete": False,
            "frame_count": frame_count,
            "error": f"{type(exc).__name__}: {exc}",
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
        del encoder
        del backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def summarize(output_root: Path, sequences: Iterable[str], max_frames: int, use_features: bool) -> dict[str, Any]:
    results = []
    failures = []
    for sequence in sorted(sequences):
        path = output_root / "done" / f"{sequence}.json"
        if not path.is_file():
            failures.append({"sequence": sequence, "reason": "done_artifact_missing"})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        (results if payload.get("status") == "PASS" else failures).append(payload)
    status = "PASS" if len(results) == len(list(sequences)) and not failures else "PARTIAL"
    manifest = {
        "protocol": "N35_REAL_CANDIDATE_COMPLETE_TAPE",
        "status": status,
        "candidate_complete": bool(status == "PASS"),
        "candidate_set_complete": bool(status == "PASS"),
        "interaction_source": "event_tape_generated_offline_after_runtime_export",
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "split": "train/train_fold",
        "sequence_count_expected": len(list(sequences)),
        "sequence_count_pass": len(results),
        "frame_count": int(sum(int(item.get("frame_count", 0) or 0) for item in results)),
        "candidate_count": int(sum(int(item.get("candidate_count", 0) or 0) for item in results)),
        "feature_dim": 512 if use_features else None,
        "feature_source": "machine_roi_fallback_osnet_market1501" if use_features else "not_exposed",
        "runtime_memory_policy": (
            results[0].get("runtime_memory_policy", {}) if results else {}
        ),
        "max_frames_per_sequence": int(max_frames),
        "sequences": sorted(sequences),
        "completed": results,
        "failures": failures,
        "tape_files": [item.get("output") for item in results],
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "note": "Rows contain official full-VG candidates and association audit only; no GT or future label is read at runtime.",
    }
    atomic_json(output_root / "tape_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", default="", help="comma-separated train sequence names")
    parser.add_argument("--sequence-list", type=Path, default=DEFAULT_SEQUENCES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu", type=int, default=0, help="local CUDA index within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--max-frames", type=int, default=0, help="smoke limit; 0 means full sequence")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--no-features", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="do not race on a shared tape_manifest.json; aggregate after all workers finish",
    )
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("N35 real SAM3 exporter requires CUDA")
    torch.cuda.set_device(int(args.gpu))
    sequences = load_sequences(args.sequence_list, args.sequences)
    chosen = shard_sequences(sequences, int(args.shard), int(args.num_shards))
    print(
        json.dumps(
            {
                "protocol": "N35_REAL_CANDIDATE_COMPLETE_TAPE",
                "gpu_local_index": int(args.gpu),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "shard": int(args.shard),
                "num_shards": int(args.num_shards),
                "sequences": chosen,
                "max_frames": int(args.max_frames),
                "features": not args.no_features,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for sequence in chosen:
        done_path = args.output_root / "done" / f"{sequence}.json"
        if args.skip_existing and done_path.is_file():
            existing = json.loads(done_path.read_text(encoding="utf-8"))
            if existing.get("status") == "PASS":
                print(json.dumps({"sequence": sequence, "status": "SKIP_EXISTING"}, sort_keys=True), flush=True)
                continue
        try:
            result = export_sequence(
                sequence,
                args.output_root,
                int(args.gpu),
                int(args.max_frames),
                not args.no_features,
            )
            print(json.dumps(result, sort_keys=True), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {"sequence": sequence, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                    sort_keys=True,
                ),
                flush=True,
            )
            raise
    if args.no_manifest:
        print(
            json.dumps(
                {
                    "manifest": None,
                    "status": "SEQUENCE_WORKER_COMPLETE",
                    "sequences": sequences,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        manifest = summarize(args.output_root, sequences, int(args.max_frames), not args.no_features)
        print(json.dumps({"manifest": str(args.output_root / "tape_manifest.json"), **manifest}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
