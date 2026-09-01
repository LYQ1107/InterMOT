#!/usr/bin/env python3
"""Run one N32 policy retry in a fresh SAM3 backend/session.

The process handles exactly one ``(episode_id, policy)`` item and writes one
atomic JSON artifact.  The supervisor launches a new process for every item so
the model, official session, future outputs, snapshots, and allocator state
cannot accumulate across retries.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
try:
    import torch
except Exception:  # pragma: no cover - official backend imports torch itself
    torch = None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "third_party/sam3") not in sys.path:
    sys.path.insert(0, str(ROOT / "third_party/sam3"))

from sam3_intermot.adaptation.correction_application_policy import (  # noqa: E402
    CorrectionApplicationAction,
    CorrectionApplicationPolicy,
)
from sam3_intermot.adaptation.correction_selector_features import (  # noqa: E402
    FEATURE_NAMES,
    build_selector_features,
)
from sam3_intermot.backend.sam3_state_snapshot import snapshot_continuation_state  # noqa: E402
from scripts.n29_lit_online_replay import (  # noqa: E402
    _image_files,
    _install_official_box_singleton,
    _make_backend,
    _read_gt,
    _session,
)
from scripts.n29r_paired_replay import (  # noqa: E402
    _ensure_public_singleton_binding,
    _horizon_metrics,
)
from scripts.n32_policy_semantics import (  # noqa: E402
    VISIBLE_UNDEFINED,
    drift_status as _drift_status,
    policy_metric_issues,
    visible_h20_status,
)
from scripts.n32_build_policy_rollouts import (  # noqa: E402
    _observation_digest,
    _obs_summary,
)


CHECKPOINT = ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
RETRY_MANIFEST = ROOT / "outputs/n32/policy_rollouts/retry_manifest.json"
RETRY_DIR = ROOT / "outputs/n32/policy_rollouts/policy_retries"
POLICY_ACTIONS = {
    "K0_KEEP_OLD": CorrectionApplicationAction.KEEP_OLD_STATE,
    "K1_APPLY_ENSURE": CorrectionApplicationAction.APPLY_CURRENT_ENSURE,
    "K2_PROMPT_THEN_RESTORE": CorrectionApplicationAction.PROMPT_THEN_RESTORE,
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    raise TypeError(f"not JSON serializable: {type(value)}")


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _policy_reward(metrics: Mapping[str, Any]) -> float:
    iou = float(metrics.get("mean_box_iou_visible") or 0.0)
    missing = float(
        metrics.get("missing_prediction_rate_visible")
        if metrics.get("missing_prediction_rate_visible") is not None
        else 1.0
    )
    drift = float(metrics.get("mask_area_drift") or 0.0)
    return float(iou - 0.50 * missing - 0.05 * drift)


def _temporal_feature_sequence(
    *,
    backend: Any,
    correction_frame: int,
    corrected_box: np.ndarray,
    prefix_outputs: Mapping[int, Any],
    public_id: int,
) -> list[list[float]]:
    frames = sorted(int(frame) for frame in prefix_outputs if int(frame) <= int(correction_frame))
    frames = frames[-5:] if frames else [int(correction_frame)]
    sequence: list[list[float]] = []
    for frame in frames:
        causal_outputs = {
            key: value for key, value in prefix_outputs.items() if int(key) <= frame
        }
        payload = build_selector_features(
            backend=backend,
            correction_frame=frame,
            corrected_box=corrected_box,
            prefix_outputs=causal_outputs,
            public_id=public_id,
        )
        sequence.append(list(payload["features"]))
    return sequence


def _strict_complete(policy_row: Mapping[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    reasons = policy_metric_issues(policy_row, require_explicit_visible_status=True)
    h20 = (policy_row.get("metrics") or {}).get("20")
    drift = _drift_status(h20) if isinstance(h20, Mapping) else {}
    return not reasons, reasons, drift


def _artifact_path(output_dir: Path, episode_id: str, policy: str) -> Path:
    return output_dir / f"{_sha256(f'{episode_id}|{policy}')}.json"


def _load_item(manifest_path: Path, item_index: int | None, episode_id: str | None, policy: str | None) -> tuple[dict[str, Any], int]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("val25_read") is not False or payload.get("test_labels_used") is not False:
        raise ValueError("retry manifest is not a frozen blind PASS artifact")
    items = payload.get("items", [])
    if item_index is not None:
        if item_index < 0 or item_index >= len(items):
            raise IndexError(f"retry item index {item_index} outside [0, {len(items)})")
        item = items[item_index]
        return item, int(item_index)
    matches = [
        (index, item)
        for index, item in enumerate(items)
        if str(item.get("episode_id")) == str(episode_id) and str(item.get("policy")) == str(policy)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one retry item for {episode_id}|{policy}, got {len(matches)}")
    return matches[0][1], matches[0][0]


def _release_backend(backend: Any) -> None:
    if backend is not None:
        try:
            backend.close()
        except Exception:
            pass
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _offload_audit(backend: Any) -> dict[str, Any]:
    """Record only offload mechanisms observed on this live session.

    The pinned multiplex ``init_state`` has no state-offload argument.  Video
    offload is considered effective only when the loaded frame container is
    actually CPU-backed, and output offload is considered enabled only when
    the adapter set the official tracker flag on a live model object.
    """

    predictor = getattr(backend, "_predictor", None)
    session_id = getattr(backend, "_session_id", None)
    states = getattr(predictor, "_all_inference_states", {}) if predictor is not None else {}
    entry = states.get(session_id) if isinstance(states, Mapping) else None
    state = entry.get("state") if isinstance(entry, Mapping) else None
    input_batch = state.get("input_batch") if isinstance(state, Mapping) else None
    image_batch = getattr(input_batch, "img_batch", None)
    frame_container = getattr(image_batch, "tensors", image_batch)
    frame_device = getattr(frame_container, "device", None)
    video_effective = bool(
        getattr(frame_container, "offload_video_to_cpu", False)
        or (frame_device is not None and getattr(frame_device, "type", None) == "cpu")
    )

    model = getattr(predictor, "model", None)
    tracker = getattr(model, "tracker", None)
    tracker_models = [tracker]
    if tracker is not None and hasattr(tracker, "model"):
        tracker_models.append(tracker.model)
    output_flags = [
        bool(getattr(target, "offload_output_to_cpu_for_eval"))
        for target in tracker_models
        if target is not None and hasattr(target, "offload_output_to_cpu_for_eval")
    ]
    return {
        "offload_video_to_cpu_requested": True,
        "offload_video_to_cpu_effective": video_effective,
        "offload_video_frame_device": None if frame_device is None else str(frame_device),
        "offload_state_to_cpu": False,
        "offload_state_to_cpu_supported_by_pinned_multiplex_init": False,
        "async_loading_frames": bool(getattr(backend, "async_loading_frames", False)),
        "official_output_offload_flags_observed": output_flags,
        "official_output_offload_adapter_enabled": bool(output_flags) and all(output_flags),
    }


def run_one(*, item: Mapping[str, Any], checkpoint: Path, attempt: int) -> dict[str, Any]:
    episode_id = str(item["episode_id"])
    policy_name = str(item["policy"])
    if policy_name not in POLICY_ACTIONS:
        raise ValueError(f"unknown policy {policy_name}")
    sequence = Path(str(item["sequence_path"]))
    images = _image_files(sequence)
    init = int(item["initialization_frame"])
    correction = int(item["correction_frame"])
    query_end = min(int(item["query_end"]), len(images) - 1)
    public_id = int(item["public_id"])
    dataset_identity = int(item["dataset_identity"])
    if query_end < correction + 20:
        raise ValueError(f"episode lacks H20: {episode_id}")
    gt = _read_gt(sequence)
    init_box = np.asarray(gt[init][dataset_identity], dtype=float)
    corrected_box = np.asarray(item["correction_box"], dtype=float)
    backend = None
    prefix_outputs: Any = None
    prefix_snapshot: Any = None
    future: Any = None
    metrics: Any = None
    try:
        # One policy gets one model/backend/session.  No other policy shares
        # this process or official inference state.
        backend = _make_backend(checkpoint)
        started = time.perf_counter()
        _session(backend, sequence)
        backend.add_box(init, public_id, init_box)
        _install_official_box_singleton(
            backend,
            frame_idx=init,
            public_id=public_id,
            box_xyxy=init_box,
        )
        prefix_outputs = backend.propagate(init, correction, start_frame_index=init)
        raw_current = _obs_summary(prefix_outputs.get(correction, []))
        prefix_digest = _observation_digest(prefix_outputs.get(correction, []))
        raw_recorded = correction in prefix_outputs
        prefix_snapshot = snapshot_continuation_state(backend)
        feature_payload = build_selector_features(
            backend=backend,
            correction_frame=correction,
            corrected_box=corrected_box,
            prefix_outputs=prefix_outputs,
            public_id=public_id,
        )
        temporal_features = _temporal_feature_sequence(
            backend=backend,
            correction_frame=correction,
            corrected_box=corrected_box,
            prefix_outputs=prefix_outputs,
            public_id=public_id,
        )
        restore = __import__(
            "sam3_intermot.backend.sam3_state_snapshot",
            fromlist=["restore_continuation_state"],
        ).restore_continuation_state
        restore(backend, prefix_snapshot)
        ledger: list[dict[str, Any]] = []
        result = CorrectionApplicationPolicy(POLICY_ACTIONS[policy_name]).apply(
            backend,
            correction_frame=correction,
            public_id=public_id,
            corrected_box=corrected_box,
            pre_correction_snapshot=prefix_snapshot,
            ledger=ledger,
            raw_output_recorded=raw_recorded,
            ensure_binding=_ensure_public_singleton_binding,
        )
        failure = result.failure

        # The correction transaction is complete at this point.  The policy
        # result and ledger contain only small Python metadata; neither keeps
        # the continuation snapshot or prefix output graph alive.  Materialize
        # the values that will be written to the artifact before releasing the
        # large correction-time containers.  Do not touch the official
        # predictor/session state: future propagation still needs it.
        feature_vector = list(feature_payload["features"])
        feature_names = list(FEATURE_NAMES)
        temporal_feature_sequence = [list(values) for values in temporal_features]
        current_raw_output_digest = str(prefix_digest)
        del prefix_snapshot
        del prefix_outputs
        del raw_current
        del feature_payload
        del temporal_features
        # Keep the finally block idempotent while making the released
        # references explicit.  ``result`` and ``ledger`` are dataclass/dict
        # metadata only and do not contain the snapshot object.
        prefix_snapshot = None
        prefix_outputs = None
        raw_current = None
        feature_payload = None
        temporal_features = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        future = {}
        metrics = {}
        try:
            future = backend.propagate(
                correction + 1,
                query_end,
                start_frame_index=correction + 1,
            )
            metrics = _horizon_metrics(
                future,
                gt,
                {**item, "query_end": query_end},
            )
        except Exception as exc:
            failure = f"future_{type(exc).__name__}: {exc}"
        h20 = metrics.get("20", {}) if isinstance(metrics, Mapping) else {}
        offload_audit = _offload_audit(backend)
        available = bool(future) and (
            failure is None
            or (result.status == "ROLLBACK" and failure is not None)
        )
        policy_row = {
            "policy": policy_name,
            "action_value": int(POLICY_ACTIONS[policy_name].value),
            "status": "PASS" if available else "NOT_RUN",
            "available": available,
            "action_trace": result.to_dict(),
            "ledger": ledger,
            "current_raw_output_recorded": raw_recorded,
            "current_raw_output_digest": current_raw_output_digest,
            "current_delivered_box": corrected_box.tolist(),
            "current_delivered_source": "human_correction",
            "metrics": metrics,
            "reward": None if not metrics else _policy_reward(h20),
            "protected_identity_regression": False,
            "protected_identity_status": "NOT_APPLICABLE_SINGLE_ID",
            "future_frame_count": len(future),
            "prompt_success": bool(result.prompt_returned_target),
            "fallback_used": bool(result.fallback_used),
            "rollback_used": bool(result.rollback_used),
            "mapping_valid": bool(result.mapping_valid),
            "target_state_present": bool(result.target_state_present),
            "feature_vector": feature_vector,
            "feature_names": feature_names,
            "temporal_feature_sequence": temporal_feature_sequence,
            "future_gt_used_for_selection": False,
            "future_gt_used_for_posthoc_evaluation": True,
            "failure": failure,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        visible = visible_h20_status(h20) if isinstance(h20, Mapping) else None
        if visible is not None and visible["status"] == VISIBLE_UNDEFINED and visible["valid"]:
            # Record the legal zero-denominator state while retaining both
            # visible metrics as null.  No metric is imputed here.
            h20["visible_metric_status"] = VISIBLE_UNDEFINED
            h20["visible_metric_semantics"] = VISIBLE_UNDEFINED
        strict, strict_reasons, drift = _strict_complete(policy_row)
        return {
            "protocol": "N32-C-POLICY-LEVEL-RETRY-ITEM",
            "status": "PASS" if strict else "FAIL",
            "episode_id": episode_id,
            "policy": policy_name,
            "attempt": int(attempt),
            "policy_row": policy_row,
            "strict_complete": strict,
            "strict_failure_reasons": strict_reasons,
            "drift_status": drift,
            "fresh_backend_per_policy": True,
            "fresh_session_per_policy": True,
            "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "offload": offload_audit,
            "val25_read": False,
            "test_labels_used": False,
            "future_gt_used_for_selection": False,
        }
    finally:
        # Explicitly drop all large references before the child exits.  The
        # process boundary is the final guarantee that CUDA allocator state
        # cannot leak into the next episode-policy.
        del future
        del metrics
        del prefix_snapshot
        del prefix_outputs
        _release_backend(backend)
        del backend


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=RETRY_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=RETRY_DIR)
    parser.add_argument("--item-index", type=int, default=None)
    parser.add_argument("--episode-id", type=str, default=None)
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.item_index is None and (args.episode_id is None or args.policy is None):
        parser.error("provide --item-index or both --episode-id and --policy")
    item, item_index = _load_item(
        args.manifest,
        args.item_index,
        args.episode_id,
        args.policy,
    )
    output = _artifact_path(args.output_dir, str(item["episode_id"]), str(item["policy"]))
    try:
        payload = run_one(item=item, checkpoint=args.checkpoint, attempt=args.attempt)
    except Exception as exc:
        payload = {
            "protocol": "N32-C-POLICY-LEVEL-RETRY-ITEM",
            "status": "FAIL",
            "episode_id": str(item["episode_id"]),
            "policy": str(item["policy"]),
            "item_index": int(item_index),
            "attempt": int(args.attempt),
            "strict_complete": False,
            "strict_failure_reasons": [f"{type(exc).__name__}: {exc}"],
            "failure": f"{type(exc).__name__}: {exc}",
            "failure_traceback": traceback.format_exc(limit=40),
            "fresh_backend_per_policy": True,
            "fresh_session_per_policy": True,
            "allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "val25_read": False,
            "test_labels_used": False,
            "future_gt_used_for_selection": False,
        }
    payload["item_index"] = int(item_index)
    _write(output, payload)
    print(json.dumps({
        "item_index": item_index,
        "episode_id": item["episode_id"],
        "policy": item["policy"],
        "status": payload.get("status"),
        "strict_complete": payload.get("strict_complete"),
        "output": str(output),
    }, sort_keys=True), flush=True)
    return 0 if payload.get("status") == "PASS" and payload.get("strict_complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
