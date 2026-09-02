#!/usr/bin/env python3
"""Run one fixed official-SAM3 N72R1 candidate/assignment smoke window.

This process intentionally has one backend session, one frame range, and no
GT/evaluator import.  It emits Candidate V2 rows and a same-run assignment
sidecar.  Because the active candidate branch has no proven public authority
bridge, the sidecar records that gap instead of treating a StateManager PID as
a public MOT ID.
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
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.assignment_sidecar import build_assignment_sidecar, validate_assignment_sidecar
from sam3_intermot.association.state_manager import StateManager, StateManagerConfig
from sam3_intermot.backend.sam3_backend import Sam3Backend
from sam3_intermot.provenance.candidate_v2 import validate_candidate_v2_rows
from sam3_intermot.provenance.mapping import canonical_mask_digest
from sam3_intermot.provenance.mapping_v2 import HandoverLedger
from scripts.n36_tape_common import DATA_ROOT, CHECKPOINT, image_files


N72R1_ROOT = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R1")
PUBLIC_SOURCE_CHECKPOINT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
MACHINE_ENCODER = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/outputs/n9/checkpoints/osnet_x1_0_market1501.pth")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class FrozenMachineOSNet:
    """Exact N35/N71 OSNet box-crop fallback, with an explicit source path."""

    feature_dim = 512

    def __init__(self, device: str) -> None:
        from torchreid.reid.utils.feature_extractor import FeatureExtractor

        if not MACHINE_ENCODER.is_file():
            raise FileNotFoundError(f"frozen machine encoder unavailable: {MACHINE_ENCODER}")
        self.extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            model_path=str(MACHINE_ENCODER),
            image_size=(256, 128),
            device=device,
            verbose=False,
        )

    @staticmethod
    def _crop(image: Image.Image, box: Sequence[float]) -> np.ndarray:
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
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
        if values.shape != (len(boxes), self.feature_dim) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"machine ROI feature has invalid shape/values: {values.shape}")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms <= 1e-6):
            raise RuntimeError("machine ROI feature has zero norm")
        return values / norms


def load_plan(path: Path, window_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload.get("windows", []):
        if str(item.get("window_id")) == window_id:
            return dict(item)
    raise KeyError(f"window not found in frozen plan: {window_id}")


def run_window(plan: dict[str, Any], output_root: Path, gpu: int) -> dict[str, Any]:
    sequence = str(plan["sequence"])
    window_id = str(plan["window_id"])
    sequence_dir = DATA_ROOT / "train" / sequence
    frames = image_files(sequence_dir)
    if len(frames) != int(plan["frame_count_total"]):
        raise RuntimeError(f"frame count changed for {sequence}: {len(frames)} != {plan['frame_count_total']}")
    frame_start, frame_end = int(plan["frame_start"]), int(plan["frame_end"])
    if not 0 <= frame_start <= frame_end < len(frames):
        raise ValueError("frozen window range is invalid")

    run_id = f"n72r1-{window_id}-{hashlib.sha256(os.urandom(16)).hexdigest()[:12]}"
    session_id = None
    backend = None
    encoder = None
    candidate_rows: list[dict[str, Any]] = []
    legacy_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []
    sidecar_records: list[dict[str, Any]] = []
    seen_frames: set[int] = set()
    raw_keys: set[tuple[int, int]] = set()
    candidate_count = 0
    manager = StateManager(StateManagerConfig(variant="reid", score_threshold=-100.0))
    ledger = None
    started = time.time()
    try:
        backend = Sam3Backend(
            checkpoint_path=str(PUBLIC_SOURCE_CHECKPOINT),
            max_num_objects=16,
            multiplex_count=16,
            use_fa3=False,
            use_rope_real=True,
            compile=False,
            warm_up=False,
            session_expiration_sec=1200,
            output_prob_thresh=0.30,
            async_loading_frames=False,
            device="cuda",
        )
        session_id = backend.start_video(str(sequence_dir / "img1"))
        ledger = HandoverLedger(
            source_run_id=run_id,
            sequence=sequence,
            session_id=session_id,
            segment_id=f"segment-{frame_start}-{frame_end}",
            window_id=window_id,
            chunk_id=window_id,
        )
        encoder = FrozenMachineOSNet(f"cuda:{gpu}")
        metadata = {
            "source_run_id": run_id,
            "sequence": sequence,
            "video_id": sequence,
            "checkpoint_sha256": sha256(PUBLIC_SOURCE_CHECKPOINT),
            "runtime_config_sha256": hashlib.sha256(json.dumps({"max_num_objects": 16, "multiplex_count": 16, "output_prob_thresh": 0.30, "async_loading_frames": False, "offload_video_to_cpu": True}, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "session_id": session_id,
            "segment_id": f"segment-{frame_start}-{frame_end}",
            "window_id": window_id,
            "chunk_id": window_id,
        }

        def write_frame(frame_idx: int, observations: list[Any]) -> None:
            nonlocal candidate_count
            frame_idx = int(frame_idx)
            if frame_idx in seen_frames:
                return
            if not frame_start <= frame_idx <= frame_end:
                return
            observations = [obs.copy() for obs in observations]
            boxes = [np.asarray(obs.box_xyxy, dtype=float).copy() for obs in observations]
            features = encoder.encode(frames[frame_idx], boxes)
            backend._output_cache[frame_idx] = [obs.copy() for obs in observations]
            try:
                locals_: list[str] = []
                globals_: list[str] = []
                for obs in observations:
                    if obs.raw_sam_object_id is None:
                        raise RuntimeError(f"official raw ID missing at {sequence}:{frame_idx}")
                    local, global_id = ledger.axes_for(int(obs.sam_object_id))
                    if global_id is None:
                        raise RuntimeError(f"same-segment global ID missing at {sequence}:{frame_idx}:{obs.sam_object_id}")
                    locals_.append(local)
                    globals_.append(global_id)
                legacy = backend.export_frame_candidates(
                    frame_idx,
                    embeddings=features,
                    include_masks=True,
                )
                rows = backend.export_frame_candidates_v2(
                    frame_idx,
                    metadata=metadata,
                    segment_local_ids=locals_,
                    sequence_global_ids=globals_,
                    embeddings=features,
                )
            finally:
                backend._output_cache.pop(frame_idx, None)
            if len(rows) != len(observations):
                raise RuntimeError(f"candidate count changed during V2 export at frame {frame_idx}")
            if len(legacy) != len(rows):
                raise RuntimeError(f"legacy/V2 candidate count mismatch at frame {frame_idx}")
            frame_equivalence: list[dict[str, Any]] = []
            for old, new in zip(legacy, rows):
                old_mask = old.get("mask")
                new_mask_digest = new.get("mask_sha256")
                old_feature = old.get("embedding")
                new_feature = new.get("feature")
                fields = {
                    "frame_idx": int(old.get("frame_idx")) == int(new.get("frame_idx")),
                    "candidate_index": int(old.get("candidate_index")) == int(new.get("candidate_index")),
                    "candidate_order": True,
                    # V2's contract deliberately canonicalizes boxes as
                    # little-endian float32.  Compare the legacy projection
                    # at that same canonical boundary; comparing legacy
                    # float64 storage byte-for-byte would report a false
                    # incompatibility for ordinary non-exact coordinates.
                    "box": bool(np.array_equal(np.asarray(old.get("box_xyxy"), dtype="<f4"), np.asarray(new.get("box_xyxy"), dtype="<f4"))),
                    "mask": old_mask is not None and canonical_mask_digest(old_mask) == new_mask_digest,
                    "confidence": float(old.get("confidence")) == float(new.get("confidence")),
                    "presence_score": old.get("presence_score") == new.get("presence_score"),
                    "source": old.get("source") == new.get("source"),
                    "feature": old_feature is not None and new_feature is not None and bool(np.array_equal(np.asarray(old_feature, dtype=np.float32), np.asarray(new_feature, dtype=np.float32))),
                }
                frame_equivalence.append({"candidate_index": int(new["candidate_index"]), "fields": fields, "all_pass": all(fields.values())})
            equivalence_rows.append({"frame_idx": frame_idx, "candidate_count_legacy": len(legacy), "candidate_count_v2": len(rows), "rows": frame_equivalence, "all_pass": all(item["all_pass"] for item in frame_equivalence)})
            legacy_rows.append({
                "schema_version": "N72R1_LEGACY_CANDIDATE_FRAME_V1",
                "run_id": run_id,
                "session_id": session_id,
                "sequence": sequence,
                "window_id": window_id,
                "frame_idx": frame_idx,
                "candidates": [
                    {
                        "frame_idx": int(item["frame_idx"]),
                        "candidate_index": int(item["candidate_index"]),
                        "native_tid": int(item["native_tid"]),
                        "box_xyxy": np.asarray(item["box_xyxy"], dtype=float).tolist(),
                        "mask_sha256": None if item.get("mask") is None else canonical_mask_digest(item["mask"]),
                        "confidence": float(item["confidence"]),
                        "presence_score": item.get("presence_score"),
                        "embedding": None if item.get("embedding") is None else np.asarray(item["embedding"], dtype=np.float32).tolist(),
                        "candidate_order": int(item["candidate_index"]),
                    }
                    for item in legacy
                ],
                "runtime_future_gt_used": False,
            })
            frame_uids = set()
            for index, row in enumerate(rows):
                raw = row.get("official_raw_sam_id")
                if raw is None:
                    raise RuntimeError(f"official raw ID absent in V2 row at frame {frame_idx}:{index}")
                key = (frame_idx, int(raw))
                if key in raw_keys or row["candidate_uid"] in frame_uids:
                    raise RuntimeError(f"duplicate candidate key at {sequence}:{frame_idx}:{index}")
                raw_keys.add(key)
                frame_uids.add(row["candidate_uid"])
            validation = validate_candidate_v2_rows(rows)
            if validation["status"] != "PASS":
                raise RuntimeError(f"Candidate V2 validation failed at frame {frame_idx}: {validation['errors'][:3]}")
            candidate_rows.extend(rows)
            frame_records.append({
                "schema_version": "N72R1_CANDIDATE_FRAME_RECORD_V1",
                "run_id": run_id,
                "session_id": session_id,
                "sequence": sequence,
                "window_id": window_id,
                "frame_idx": frame_idx,
                "candidate_uids": [row["candidate_uid"] for row in rows],
                "candidate_count": len(rows),
                "candidate_set_complete": True,
                "runtime_future_gt_used": False,
                "interaction_source": "simulated_from_gt_metadata_inherited_from_frozen_plan_not_runtime_input",
                "public_mapping_status": "EXPLICITLY_UNAVAILABLE_NOT_FABRICATED",
                "runtime_memory_policy": backend.runtime_memory_policy(),
            })

            manager_obs = []
            for index, row in enumerate(rows):
                manager_obs.append({
                    "obs_id": index,
                    "candidate_uid": row["candidate_uid"],
                    "source_run_id": row["source_run_id"],
                    "session_id": row["session_id"],
                    "segment_id": row["segment_id"],
                    "window_id": row["window_id"],
                    "chunk_id": row["chunk_id"],
                    "official_raw_sam_id": row["official_raw_sam_id"],
                    "adapter_external_id": row["adapter_external_id"],
                    "segment_local_id": row["segment_local_id"],
                    "sequence_global_id": row["sequence_global_id"],
                    "feat": np.asarray(row["feature"], dtype=np.float32),
                    "has_feat": 1.0,
                    "box": np.asarray(row["box_xyxy"], dtype=float),
                    "native_tid": row["legacy_native_tid"],
                    "native_age": 0.0,
                    "conf": row["confidence"],
                })
            manager.rollout_frame(frame_idx, manager_obs)
            association_audit = manager.candidate_log[-1]
            sidecar = build_assignment_sidecar(
                rows,
                association_audit,
                resolver=None,
                source_run_id=run_id,
                session_id=session_id,
            )
            sidecar_errors = validate_assignment_sidecar(sidecar)
            if sidecar_errors:
                raise RuntimeError(f"same-run assignment sidecar invalid at frame {frame_idx}: {sidecar_errors}")
            sidecar_records.append({"frame_idx": frame_idx, "sidecar": sidecar, "runtime_future_gt_used": False})
            seen_frames.add(frame_idx)
            candidate_count += len(rows)

        initial = backend.detect_concept(frame_start, "person")
        write_frame(frame_start, initial)
        backend.propagate(
            frame_start,
            frame_end,
            start_frame_index=frame_start,
            keep_masks=True,
            cache_outputs=False,
            output_callback=write_frame,
        )
        expected = set(range(frame_start, frame_end + 1))
        if seen_frames != expected:
            raise RuntimeError(f"candidate window coverage mismatch missing={sorted(expected - seen_frames)[:10]}")
        candidate_validation = validate_candidate_v2_rows(candidate_rows)
        if candidate_validation["status"] != "PASS":
            raise RuntimeError(f"final Candidate V2 validation failed: {candidate_validation['errors'][:5]}")

        candidate_path = output_root / "candidate_v2.jsonl"
        legacy_path = output_root / "legacy_candidates.jsonl"
        frame_path = output_root / "candidate_frames.jsonl"
        sidecar_path = output_root / "assignment_sidecar.jsonl"
        mapping_path = output_root / "mapping_ledger.jsonl"
        equivalence_path = output_root / "equivalence_audit.json"
        public_authority_path = output_root / "public_authority_audit.json"
        equivalence_audit = {
            "schema_version": "N72R1_LEGACY_V2_EQUIVALENCE_AUDIT_V1",
            "frame_count": len(equivalence_rows),
            "candidate_count": sum(int(item["candidate_count_v2"]) for item in equivalence_rows),
            "failed_frame_count": sum(int(not item["all_pass"]) for item in equivalence_rows),
            "failed_candidate_count": sum(int(not row["all_pass"]) for item in equivalence_rows for row in item["rows"]),
            "rows": equivalence_rows,
            "all_pass": all(item["all_pass"] for item in equivalence_rows),
            "runtime_future_gt_used": False,
        }
        if not equivalence_audit["all_pass"]:
            raise RuntimeError(f"legacy/V2 common-field equivalence failed: {equivalence_audit['failed_candidate_count']} candidate rows")
        mapping_ledger_rows = [{"record_type": "ledger_audit", **ledger.audit(), "runtime_future_gt_used": False}]
        mapping_ledger_rows.extend({"record_type": "binding", **item, "runtime_future_gt_used": False} for item in ledger.transactions)
        public_authority_audit = {
            "schema_version": "N72R1_PUBLIC_AUTHORITY_AUDIT_V1",
            "status": "BLOCKED_PUBLIC_AUTHORITY_NOT_IN_ACTIVE_RUNTIME",
            "resolver_present": False,
            "association_state_ids_are_public": False,
            "numeric_fallback_used": False,
            "public_id_inferred": False,
            "candidate_row_count": len(candidate_rows),
            "assignment_sidecar_frame_count": len(sidecar_records),
            "public_assignment_artifact_absent_count": sum(len(item["sidecar"].get("public_assignment_rows", [])) for item in sidecar_records),
            "runtime_future_gt_used": False,
        }
        atomic_jsonl(candidate_path, candidate_rows)
        atomic_jsonl(legacy_path, legacy_rows)
        atomic_jsonl(frame_path, frame_records)
        atomic_jsonl(sidecar_path, sidecar_records)
        atomic_jsonl(mapping_path, mapping_ledger_rows)
        atomic_json(equivalence_path, equivalence_audit)
        atomic_json(public_authority_path, public_authority_audit)
        done = {
            "schema_version": "N72R1_STAGE15_GPU_SMOKE_DONE_V1",
            "status": "PASS_CANDIDATE_V2_AND_SAME_RUN_SIDECAR_NO_PUBLIC_AUTHORITY",
            "sequence": sequence,
            "window_id": window_id,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "frame_count": len(seen_frames),
            "candidate_row_count": candidate_count,
            "candidate_validation": candidate_validation,
            "legacy_candidate_row_count": sum(len(item["candidates"]) for item in legacy_rows),
            "legacy_v2_equivalence": equivalence_audit,
            "sidecar_frame_count": len(sidecar_records),
            "sidecar_public_authority_present": False,
            "public_mapping_status": "EXPLICITLY_UNAVAILABLE_NOT_FABRICATED",
            "local_global_audit": ledger.audit(),
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "runtime_config_sha256": metadata["runtime_config_sha256"],
            "runtime_memory_policy": backend.runtime_memory_policy(),
            "runtime_future_gt_used": False,
            "process_isolation": "one_python_process_one_sam3_session_one_frame_range",
            "outputs": {"legacy_candidates": str(legacy_path), "candidate_v2": str(candidate_path), "candidate_frames": str(frame_path), "assignment_sidecar": str(sidecar_path), "mapping_ledger": str(mapping_path), "equivalence_audit": str(equivalence_path), "public_authority_audit": str(public_authority_path)},
            "elapsed_sec": time.time() - started,
        }
        atomic_json(output_root / "done.json", done)
        return done
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        encoder = None
        backend = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=N72R1_ROOT / "smoke" / "stage_15")
    parser.add_argument("--status-path", type=Path, default=None, help="stage status destination; used by multi-window wrapper")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    status_path = (args.status_path.resolve() if args.status_path is not None else N72R1_ROOT / "status" / "stage_15_status.json")
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(int(args.gpu))
        plan = load_plan(args.plan.resolve(), args.window_id)
        result = run_window(plan, output_root, int(args.gpu))
        status = {
            "schema_version": "N72R1_STAGE_STATUS_V1",
            "stage": "N72R1-15",
            "status": "PARTIAL_PUBLIC_AUTHORITY_NOT_BRIDGED",
            "candidate_export": result,
            "research_efficacy": "NOT_RUN",
            "real_human_event_count": 0,
            "runtime_future_gt_used": False,
            "public_id_inferred": False,
            "next_minimum_action": "Add and audit an explicit same-run association_state_id-to-public_id runtime bridge before any public-ID efficacy interpretation.",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(status_path, status)
        atomic_json(output_root / "stage_15_status.json", status)
        print(json.dumps({"status": status["status"], "frames": result["frame_count"], "candidate_rows": result["candidate_row_count"]}, sort_keys=True))
    except Exception as exc:
        failure = {
            "schema_version": "N72R1_STAGE_STATUS_V1",
            "stage": "N72R1-15",
            "status": "BLOCKED_OFFICIAL_SAM3_GPU_SMOKE",
            "failure_type": type(exc).__name__,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "is_oom": "outofmemory" in type(exc).__name__.lower() or "out of memory" in str(exc).lower(),
            "runtime_future_gt_used": False,
            "real_human_event_count": 0,
            "public_id_inferred": False,
            "next_minimum_action": "Preserve traceback; perform at most the N72R1-approved bounded GPU repair before retrying the same frozen window.",
        }
        atomic_json(status_path, failure)
        atomic_json(output_root / "failure.json", failure)
        print(json.dumps({"status": failure["status"], "failure_type": failure["failure_type"], "is_oom": failure["is_oom"]}, sort_keys=True))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
