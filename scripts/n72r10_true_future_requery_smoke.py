#!/usr/bin/env python3
"""N72R10 Stage 03: one real current-frame future-requery smoke.

This is an engineering smoke, not a scientific effect result.  It uses one
frozen N72R9 B0 runtime row only to obtain the causal predicted box and the
already explicit raw binding.  It never loads GT or posthoc labels.  Every
query owns a fresh SAM3 backend/session; the selected query is rerun in one
fresh session before the event-local future suffix is propagated.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.backend.sam3_backend import Sam3Backend  # noqa: E402
from sam3_intermot.reacquisition.future_requery_session import (  # noqa: E402
    FUTURE_FRAME_REQUERY,
    FutureFrameRequerySession,
    QUERY_SPECS,
)
from scripts.n72r5_stage07_official_full_loop import (  # noqa: E402
    CHECKPOINT,
    DATA_ROOT,
    MACHINE_CHECKPOINT,
    FrozenMachineOSNetN72R5,
    image_files,
)


EVENT_ID = "n72r5-pool-n37-dancetrack0008-0071-atomic_id_swap-001"
SEQUENCE = "dancetrack0008"
EVENT_FRAME = 71
TRIGGER_FRAME = 72
END_FRAME = 171
RUNTIME_INPUT = (
    ROOT
    / "outputs/N72R9/replay/full"
    / EVENT_ID
    / "BASELINE_B0/runtime_frames.jsonl"
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_runtime_row(path: Path, frame: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    matches: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row.get("frame", -1)) == int(frame):
                matches.append(row)
    if len(matches) != 1:
        raise RuntimeError(f"expected one frozen runtime row at frame {frame}, found {len(matches)}")
    return matches[0]


def target_assignment_uid(row: Mapping[str, Any], target_public_id: int) -> str:
    assignment = row.get("assignment")
    if not isinstance(assignment, Mapping):
        pool = row.get("candidate_pool")
        assignment = pool.get("assignment") if isinstance(pool, Mapping) else None
    if isinstance(assignment, Mapping):
        direct = assignment.get("target_assigned_candidate_uid")
        if direct not in (None, ""):
            return str(direct)
        solver = assignment.get("solver")
        if isinstance(solver, Mapping):
            for item in solver.get("public_assignments", []):
                if isinstance(item, Mapping) and int(item.get("public_id", -1)) == int(target_public_id):
                    return str(item["candidate_uid"])
    raise RuntimeError("frozen B0 row has no explicit target public assignment UID")


def causal_target_from_runtime(row: Mapping[str, Any]) -> tuple[int, list[float], dict[str, Any]]:
    target_public_id = int(row["target_public_id"])
    pool = row.get("candidate_pool")
    if not isinstance(pool, Mapping):
        raise RuntimeError("frozen B0 row has no candidate_pool")
    candidates = pool.get("candidate_rows")
    if not isinstance(candidates, list):
        raise RuntimeError("frozen B0 row has no candidate_rows")
    target_uid = target_assignment_uid(row, target_public_id)
    target_rows = [item for item in candidates if isinstance(item, Mapping) and str(item.get("candidate_uid")) == target_uid]
    if len(target_rows) != 1:
        raise RuntimeError(f"target assignment UID is not unique in frozen candidate pool: {target_uid}")
    target = target_rows[0]
    box = np.asarray(target.get("box_xyxy"), dtype=np.float64).reshape(-1)
    if box.size != 4 or not np.all(np.isfinite(box)) or box[2] <= box[0] or box[3] <= box[1]:
        raise RuntimeError("frozen B0 target assignment has an invalid causal predicted box")
    raw = target.get("official_raw_sam_id")
    if raw is None:
        raw = target.get("adapter_external_id")
    if raw is None:
        raise RuntimeError("frozen B0 target assignment has no raw/native ID")
    causal_state = {
        "previous_raw_sam_id": int(raw),
        "current_raw_sam_id": int(raw),
        "source_runtime_record_kind": str(row.get("record_kind", "future_association_frame")),
        "source_candidate_uid": target_uid,
        "target_public_id_authority": "frozen_runtime_assignment_input_only",
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
    }
    return target_public_id, [float(value) for value in box], causal_state


def make_backend(device: str) -> Sam3Backend:
    return Sam3Backend(
        checkpoint_path=str(CHECKPOINT),
        max_num_objects=16,
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=True,
        compile=False,
        warm_up=False,
        session_expiration_sec=1200,
        output_prob_thresh=0.30,
        async_loading_frames=False,
        device=device,
        # This pinned-official runtime control was already smoke-validated in
        # N72R5 and bounds the detector/segmentation batch peak to one frame.
        official_batched_grounding_batch_size=1,
    )


def run_smoke(*, output_root: Path, attempt: int, device: str) -> dict[str, Any]:
    if not RUNTIME_INPUT.is_file():
        raise FileNotFoundError(RUNTIME_INPUT)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    if not MACHINE_CHECKPOINT.is_file():
        raise FileNotFoundError(MACHINE_CHECKPOINT)
    row = read_runtime_row(RUNTIME_INPUT, TRIGGER_FRAME)
    target_public_id, predicted_box, causal_state = causal_target_from_runtime(row)
    paths = image_files(DATA_ROOT / "train" / SEQUENCE)
    if END_FRAME >= len(paths):
        raise RuntimeError(f"frozen image coverage is incomplete: {SEQUENCE}:{TRIGGER_FRAME}:{END_FRAME}")
    encoder = FrozenMachineOSNetN72R5(device)

    def feature_fn(frame: int, box: list[float]) -> np.ndarray:
        return np.asarray(encoder.encode(paths[int(frame)], [box])[0], dtype=np.float32)

    backends: list[Sam3Backend] = []

    def backend_factory() -> Sam3Backend:
        backend = make_backend(device)
        backends.append(backend)
        return backend

    session = FutureFrameRequerySession(
        backend_factory=backend_factory,
        sequence=SEQUENCE,
        event_id=EVENT_ID,
        event_frame=EVENT_FRAME,
        target_public_id=target_public_id,
        frame_paths=paths,
        feature_fn=feature_fn,
    )
    started = time.time()
    start_info = session.start_from_frame(
        TRIGGER_FRAME,
        predicted_box,
        causal_state,
        end_frame=END_FRAME,
        main_y_pre_frozen=True,
    )
    probe_candidates = session.query_current_frame()
    query_failures = [item for item in session.audit()["query_audits"] if str(item.get("status", "")).startswith("FAIL")]
    if query_failures:
        raise RuntimeError(f"current-frame query failures: {len(query_failures)}")
    if len(probe_candidates) != len(QUERY_SPECS):
        raise RuntimeError(
            f"expected one current candidate per frozen query: {len(probe_candidates)} != {len(QUERY_SPECS)}"
        )
    if any(row.get("candidate_source") != FUTURE_FRAME_REQUERY for row in probe_candidates):
        raise RuntimeError("probe candidate source taxonomy is not FUTURE_FRAME_REQUERY")
    if any(row.get("public_id") is not None for row in probe_candidates):
        raise RuntimeError("future re-query candidate carried public-ID authority")
    selected_name = str(probe_candidates[0]["requery_name"])
    future_candidates = session.propagate_if_selected(
        selected_query_name=selected_name,
        selected_candidate_uid=str(probe_candidates[0]["candidate_uid"]),
        selection_audit={
            "selector": "SMOKE_FIXED_FIRST_QUERY",
            "selection_is_scientific": False,
            "runtime_future_gt_used": False,
        },
        margin=None,
    )
    audit_before_close = session.audit()
    if audit_before_close["event_frame_memory_read"] is not False:
        raise RuntimeError("event/trigger frame memory-read boundary failed")
    if int(audit_before_close["first_memory_visible_frame"]) != TRIGGER_FRAME + 1:
        raise RuntimeError("first memory-visible frame is not trigger+1")
    if audit_before_close["raw_rebinding"]["public_id_changed"] is not False:
        raise RuntimeError("future re-query changed public identity authority")
    if any(row.get("runtime_future_gt_used") is not False for row in future_candidates):
        raise RuntimeError("future candidate contains a runtime GT flag")
    if not any(
        int(row.get("frame", -1)) == TRIGGER_FRAME + 1
        for row in future_candidates
    ):
        raise RuntimeError(
            "targeted smoke requires an official target candidate at trigger+1; "
            "the prior artifact's trigger-only output is not a valid PASS"
        )
    session.close()
    audit_after_close = session.audit()
    all_probe_backends_closed = all(
        getattr(item, "_session_id", None) is None for item in backends[:-1]
    )
    active_backend_closed = (
        getattr(backends[-1], "_session_id", None) is None if backends else False
    )
    if not all_probe_backends_closed or not active_backend_closed:
        raise RuntimeError(
            "fresh future-requery backend/session lifecycle did not close every session"
        )
    result = {
        "schema_version": "N72R10_TRUE_FUTURE_REQUERY_SMOKE_RESULT_V1",
        "status": "PASS_TRUE_FUTURE_REQUERY_SMOKE",
        "event_id": EVENT_ID,
        "sequence": SEQUENCE,
        "event_frame": EVENT_FRAME,
        "trigger_frame": TRIGGER_FRAME,
        "end_frame": END_FRAME,
        "target_public_id": target_public_id,
        "causal_predicted_box_xyxy": predicted_box,
        "frozen_input": {
            "runtime_input": str(RUNTIME_INPUT),
            "runtime_input_sha256": sha256_file(RUNTIME_INPUT),
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "machine_checkpoint": str(MACHINE_CHECKPOINT),
            "machine_checkpoint_sha256": sha256_file(MACHINE_CHECKPOINT),
            "candidate_uid": causal_state["source_candidate_uid"],
            "candidate_source": "MAIN_B0_CANDIDATE",
        },
        "start": start_info,
        "probe_candidates": probe_candidates,
        "selected_query_name": selected_name,
        "future_candidate_count": len(future_candidates),
        "future_candidates": future_candidates,
        "audit_before_close": audit_before_close,
        "audit_after_close": audit_after_close,
        "backend_count": len(backends),
        "backend_runtime_memory_policies": [
            backend.runtime_memory_policy() for backend in backends
        ],
        "all_probe_backends_closed": all_probe_backends_closed,
        "active_backend_closed": active_backend_closed,
        "elapsed_sec": time.time() - started,
        "device": device,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "posthoc_gt_used": False,
        "scientific_effect_result": None,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/N72R10/stage_03_smoke",
    )
    args = parser.parse_args()
    attempt_root = args.output_root / f"attempt_{int(args.attempt):02d}"
    done_path = attempt_root / "done.json"
    if done_path.exists():
        raise RuntimeError(f"refusing to overwrite existing smoke attempt: {done_path}")
    try:
        result = run_smoke(output_root=args.output_root, attempt=int(args.attempt), device=str(args.device))
        atomic_json(attempt_root / "result.json", result)
        atomic_json(attempt_root / "audit.json", result["audit_after_close"])
        atomic_json(attempt_root / "candidates.json", {
            "probe_candidates": result["probe_candidates"],
            "future_candidates": result["future_candidates"],
            "runtime_future_gt_used": False,
        })
        atomic_json(attempt_root / "done.json", {
            "schema_version": "N72R10_TRUE_FUTURE_REQUERY_SMOKE_DONE_V1",
            "status": result["status"],
            "result": str(attempt_root / "result.json"),
            "result_sha256": sha256_file(attempt_root / "result.json"),
            "finished_at_utc": now_utc(),
        })
        print(json.dumps({"status": result["status"], "output": str(attempt_root)}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "N72R10_TRUE_FUTURE_REQUERY_SMOKE_FAILURE_V1",
            "status": "FAIL_TRUE_FUTURE_REQUERY_SMOKE",
            "attempt": int(args.attempt),
            "event_id": EVENT_ID,
            "sequence": SEQUENCE,
            "event_frame": EVENT_FRAME,
            "trigger_frame": TRIGGER_FRAME,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "command": [str(item) for item in sys.argv],
            "device": str(args.device),
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
            "finished_at_utc": now_utc(),
        }
        atomic_json(
            ROOT / "outputs/N72R10/attempts" / f"stage_03_smoke_attempt_{int(args.attempt):02d}.failure.json",
            failure,
        )
        print(json.dumps({"status": failure["status"], "failure": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
