#!/usr/bin/env python3
"""Generate one complete R2 train candidate tape with the frozen SAM3 path.

This worker deliberately shares the R1 SAM3/OSNet implementation and export
schema, but writes a new R2 asset root and permits a complete sequence.  It
never opens GT, never writes masks/crops/activations, and never assigns a
public identity.  One process owns one SAM3 session so GPU state is isolated.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.n72r20r1_candidate_stream_smoke import (
    MachineOSNet,
    crop_batch,
    feature_sha256,
    image_files,
    sha256,
    zstd_stream,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "outputs/N72R20R2/asset_manifest.json"
DEFAULT_OUTPUT_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R2_assets")


def read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "N72R20R2":
        raise ValueError(f"asset manifest is not N72R20R2: {path}")
    if payload.get("dataset", {}).get("test_present_or_used"):
        raise ValueError("R2 manifest claims an out-of-scope DanceTrack test asset")
    return payload


def resolve_path(cli_value: str | None, env_name: str, manifest_value: str | None = None) -> Path:
    value = cli_value or os.environ.get(env_name) or manifest_value
    if not value:
        raise ValueError(f"provide --{env_name.lower()} or set {env_name}")
    return Path(value).expanduser().resolve()


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.asset_manifest).expanduser().resolve()
    manifest = read_manifest(manifest_path)
    dataset_root = resolve_path(args.dataset_root, "DANCETRACK_ROOT", manifest.get("DANCETRACK_ROOT"))
    checkpoint = resolve_path(args.checkpoint, "SAM3_CHECKPOINT", manifest["sam3_checkpoint"]["path"])
    osnet = resolve_path(
        args.osnet_checkpoint,
        "OSNET_CHECKPOINT",
        manifest["frozen_identity_assets"]["osnet_x1_0_market1501"]["path"],
    )
    output_root = Path(args.output_root or os.environ.get("N72R20R2_ASSET_ROOT") or DEFAULT_OUTPUT_ROOT).expanduser().resolve()
    if args.split != "train":
        raise ValueError("R2 candidate development is train-only; val remains frozen and unopened")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing authenticated SAM3 checkpoint: {checkpoint}")
    if not osnet.is_file():
        raise FileNotFoundError(f"missing frozen OSNet checkpoint: {osnet}")
    if sha256(checkpoint) != str(manifest["sam3_checkpoint"]["sha256"]):
        raise ValueError("SAM3 checkpoint SHA256 does not match the R2 manifest")
    if sha256(osnet) != str(manifest["frozen_identity_assets"]["osnet_x1_0_market1501"]["sha256"]):
        raise ValueError("OSNet checkpoint SHA256 does not match the R2 manifest")

    sequence_dir = dataset_root / args.split / args.sequence
    paths = image_files(sequence_dir)
    if not paths:
        raise FileNotFoundError(f"no image frames under {sequence_dir / 'img1'}")
    requested = int(args.max_frames)
    frame_count = len(paths) if requested <= 0 else min(len(paths), requested)
    sequence_out = output_root / "candidates" / args.sequence
    final_meta = sequence_out / "metadata.jsonl.zst"
    final_embeddings = sequence_out / "embeddings.f16"
    final_index = sequence_out / "index.json"
    final_done = sequence_out / "done.json"
    if any(path.exists() for path in (final_meta, final_embeddings, final_index, final_done)):
        raise FileExistsError(f"refusing to overwrite existing R2 sequence output: {sequence_out}")
    sequence_out.mkdir(parents=True, exist_ok=True)
    meta_tmp = sequence_out / f".{final_meta.name}.{os.getpid()}.tmp"
    embedding_tmp = sequence_out / f".{final_embeddings.name}.{os.getpid()}.tmp"
    started = time.time()
    metadata_handle = None
    compressor = None
    embedding_handle = None
    backend = None
    encoder = None
    seen_frames: set[int] = set()
    candidate_count = 0
    embedding_count = 0
    frame_records: list[dict[str, Any]] = []
    last_observations: list[Any] = []
    session_records: list[dict[str, Any]] = []
    chunk_size = max(32, int(args.chunk_size))
    try:
        metadata_handle, compressor = zstd_stream(meta_tmp)
        embedding_handle = embedding_tmp.open("wb")
        from sam3_intermot.backend.sam3_backend import Sam3Backend

        def make_backend() -> Sam3Backend:
            return Sam3Backend(
                checkpoint_path=str(checkpoint),
                max_num_objects=16,
                multiplex_count=16,
                use_fa3=False,
                use_rope_real=True,
                compile=False,
                warm_up=False,
                output_prob_thresh=0.30,
                async_loading_frames=False,
                # The pinned SAM3 adapter exposes the official non-conditioning
                # memory trim for long eval videos.  It changes only the runtime
                # memory policy, not the model weights, candidate threshold, or
                # candidate export schema.
                trim_past_non_cond_mem_for_eval=True,
                device=str(args.device),
            )

        backend = make_backend()
        backend.start_video(str(sequence_dir / "img1"))
        encoder = MachineOSNet(osnet, str(args.device))

        def write_frame(frame: int, observations: list[Any]) -> None:
            nonlocal candidate_count, embedding_count, last_observations
            frame = int(frame)
            if frame in seen_frames or frame < 0 or frame >= frame_count:
                return
            last_observations = [item.copy() for item in observations]
            boxes = [np.asarray(item.box_xyxy, dtype=float).tolist() for item in observations]
            features = encoder.encode(paths[frame], boxes)
            backend._output_cache[frame] = [item.copy() for item in observations]
            try:
                exported = backend.export_frame_candidates(
                    frame,
                    embeddings=features,
                    include_masks=False,
                    include_raw_provenance=True,
                )
            finally:
                backend._output_cache.pop(frame, None)
            rows: list[dict[str, Any]] = []
            frame_embedding_start = embedding_count
            for index, candidate in enumerate(exported):
                vector = np.asarray(candidate.get("embedding"), dtype=np.float32).reshape(-1)
                if vector.shape != (512,) or not np.isfinite(vector).all():
                    raise RuntimeError(f"invalid candidate embedding at {args.sequence}:{frame}:{index}")
                raw_id = candidate.get("raw_native_id")
                adapter_id = int(candidate["native_tid"])
                embedding_handle.write(vector.astype(np.float16).tobytes(order="C"))
                rows.append(
                    {
                        "sequence": args.sequence,
                        "frame": frame,
                        "candidate_uid": f"{args.sequence}:{frame}:{index}:{raw_id if raw_id is not None else adapter_id}",
                        "box_xyxy": np.asarray(candidate["box_xyxy"], dtype=float).tolist(),
                        "presence": candidate.get("presence_score"),
                        "iou_pred": candidate.get("predicted_iou"),
                        "confidence": float(candidate.get("confidence", 0.0)),
                        "candidate_source": str(candidate.get("source", "automatic_propagation")),
                        "source": str(candidate.get("source", "automatic_propagation")),
                        "source_public_id": None,
                        "raw_native_id": raw_id,
                        "adapter_external_id": adapter_id,
                        "native_tid": adapter_id,
                        "valid": True,
                        "reject_reason": None,
                        "embedding_offset": embedding_count,
                        "embedding_dim": 512,
                        "embedding_dtype": "float16",
                        "feature_source": "frozen_osnet_x1_0_market1501_machine_box_crop",
                        "embedding_source": "frozen_osnet_x1_0_market1501_machine_box_crop",
                        "feature_sha256": feature_sha256(vector),
                        "runtime_future_gt_used": False,
                    }
                )
                embedding_count += 1
            frame_row = {
                "stage": "N72R20R2",
                "sequence": args.sequence,
                "split": args.split,
                "frame": frame,
                "candidate_count": len(rows),
                "candidate_set_complete": True,
                "candidate_set_source": "official_sam3_full_vg_post_nms_propagation_with_past_state_rebind",
                "candidates": rows,
                "runtime_future_gt_used": False,
                "runtime_gt_read": False,
            }
            assert compressor is not None and compressor.stdin is not None
            compressor.stdin.write((json.dumps(frame_row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode())
            candidate_count += len(rows)
            seen_frames.add(frame)
            frame_records.append({"frame": frame, "candidate_count": len(rows), "embedding_start": frame_embedding_start, "embedding_count": len(rows)})

        for chunk_start in range(0, frame_count, chunk_size):
            chunk_end = min(frame_count - 1, chunk_start + chunk_size - 1)
            seed_deduplicated = 0
            if chunk_start > 0:
                # Close only the official SAM3 session.  The predictor/model
                # stays frozen in this process, while the official session
                # tensors are released before the next bounded propagation.
                backend.close()
                backend.start_video(str(sequence_dir / "img1"))
                propagation_start = chunk_start
                if last_observations:
                    seeds: list[tuple[int, np.ndarray]] = []
                    seen_seed_ids: set[int] = set()
                    for observation in last_observations:
                        seed_id = int(observation.sam_object_id)
                        if seed_id in seen_seed_ids:
                            raw_id = getattr(observation, "raw_sam_object_id", None)
                            alternate = None if raw_id is None else int(raw_id)
                            if alternate is not None and alternate not in seen_seed_ids:
                                seed_id = alternate
                            else:
                                seed_deduplicated += 1
                                continue
                        seen_seed_ids.add(seed_id)
                        seeds.append((seed_id, np.asarray(observation.box_xyxy, dtype=np.float64)))
                    rebound = backend.rebind_past_state_boxes(chunk_start - 1, seeds)
                    if int(rebound.get("recovered_count", 0)) > 0:
                        propagation_start = chunk_start - 1
                    else:
                        fallback = backend.detect_concept(chunk_start, "person")
                        if fallback:
                            write_frame(chunk_start, fallback)
                else:
                    fallback = backend.detect_concept(chunk_start, "person")
                    if fallback:
                        write_frame(chunk_start, fallback)
            else:
                initial = backend.detect_concept(0, "person")
                if initial:
                    write_frame(0, initial)
                propagation_start = 0

            backend.propagate(
                propagation_start,
                chunk_end,
                start_frame_index=propagation_start,
                keep_masks=False,
                cache_outputs=False,
                output_callback=write_frame,
            )
            session_records.append(
                {
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                    "propagation_start": propagation_start,
                    "past_state_rebind": bool(chunk_start > 0 and propagation_start == chunk_start - 1),
                    "seed_count": len(last_observations),
                    "seed_deduplicated": seed_deduplicated,
                    "runtime_future_gt_used": False,
                }
            )
        expected_frames = set(range(frame_count))
        if seen_frames != expected_frames:
            raise RuntimeError(
                f"candidate stream frame mismatch: missing={sorted(expected_frames - seen_frames)[:8]} "
                f"extra={sorted(seen_frames - expected_frames)[:8]}"
            )
        assert compressor is not None and compressor.stdin is not None and metadata_handle is not None and embedding_handle is not None
        compressor.stdin.close()
        compressor.stdin = None
        stderr = compressor.stderr.read().decode(errors="replace") if compressor.stderr is not None else ""
        return_code = compressor.wait()
        metadata_handle.flush()
        os.fsync(metadata_handle.fileno())
        metadata_handle.close()
        metadata_handle = None
        if return_code != 0:
            raise RuntimeError(f"zstd failed with code {return_code}: {stderr[-2000:]}")
        embedding_handle.flush()
        os.fsync(embedding_handle.fileno())
        embedding_handle.close()
        embedding_handle = None
        os.replace(meta_tmp, final_meta)
        os.replace(embedding_tmp, final_embeddings)
        index = {
            "stage": "N72R20R2",
            "source_stage": "N72R20R1",
            "sequence": args.sequence,
            "split": args.split,
            "frame_count": frame_count,
            "candidate_count": candidate_count,
            "embedding_count": embedding_count,
            "embedding_dim": 512,
            "embedding_dtype": "float16",
            "metadata": str(final_meta),
            "metadata_sha256": sha256(final_meta),
            "embeddings": str(final_embeddings),
            "embeddings_sha256": sha256(final_embeddings),
            "frames": frame_records,
            "sam3_session_records": session_records,
            "sam3_chunk_size": chunk_size,
            "sam3_session_strategy": "bounded_official_sessions_with_previous_runtime_box_rebind",
            "sam3_checkpoint_sha256": sha256(checkpoint),
            "osnet_checkpoint_sha256": sha256(osnet),
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
        }
        final_index.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        profile = {
            "stage": "N72R20R2",
            "sequence": args.sequence,
            "split": args.split,
            "frames": frame_count,
            "candidates": candidate_count,
            "metadata_bytes": final_meta.stat().st_size,
            "embedding_bytes": final_embeddings.stat().st_size,
            "total_bytes": final_meta.stat().st_size + final_embeddings.stat().st_size,
            "bytes_per_frame": (final_meta.stat().st_size + final_embeddings.stat().st_size) / max(frame_count, 1),
            "bytes_per_candidate": (final_meta.stat().st_size + final_embeddings.stat().st_size) / max(candidate_count, 1),
            "sam3_chunk_size": chunk_size,
            "sam3_session_count": len(session_records),
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
        }
        final_done.write_text(
            json.dumps(
                {
                    "status": "PASS_N72R20R2_CANDIDATE_STREAM_SEQUENCE",
                    "sequence": args.sequence,
                    "index": str(final_index),
                    "profile": profile,
                    "runtime_memory_policy": backend.runtime_memory_policy(),
                    "process_isolation": "one_python_process_one_sequence_bounded_sam3_sessions",
                    "sam3_session_strategy": "bounded_official_sessions_with_previous_runtime_box_rebind",
                    "sam3_session_records": session_records,
                    "elapsed_sec": time.time() - started,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return profile
    finally:
        if compressor is not None:
            if compressor.stdin is not None:
                compressor.stdin.close()
            compressor.wait()
        if metadata_handle is not None:
            metadata_handle.close()
        if embedding_handle is not None:
            embedding_handle.close()
        for path in (meta_tmp, embedding_tmp):
            if path.exists():
                path.unlink()
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        if encoder is not None:
            del encoder
        if backend is not None:
            del backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means the complete sequence")
    parser.add_argument("--chunk-size", type=int, default=160, help="bounded official session length")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--osnet-checkpoint", type=Path)
    parser.add_argument("--asset-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        profile = run(args)
    except Exception as exc:
        print(f"N72R20R2 candidate generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(profile, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
