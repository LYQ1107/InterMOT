#!/usr/bin/env python3
"""N31-C: causal correction-state writer ablation on the frozen N29-R fold.

The prefix through the correction frame is materialized once per episode.
Every P0--P7 branch then restores the same lightweight continuation snapshot,
performs only its declared intervention, and propagates the same future
window.  This makes mapping-only, prompt failure, target-scoped masklet, and
LoRA effects separately auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SAM3_ROOT = ROOT / "third_party" / "sam3"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

from sam3_intermot.adaptation.corrected_mask_teacher import BOX_DERIVED_PSEUDO_MASK  # noqa: E402
from sam3_intermot.adaptation.correction_state_audit import (  # noqa: E402
    diff_snapshots,
    snapshot_backend_state,
)
from sam3_intermot.adaptation.correction_state_candidates import (  # noqa: E402
    BOX_PROMPTED_SAM_PSEUDO_MASK,
    BOX_RECTANGLE_MASKLET,
    candidate_mask_features,
    interactive_box_candidates,
    protected_state_signatures,
    rectangle_mask,
    tracker_ids,
    write_target_mask,
)
from sam3_intermot.backend.sam3_state_snapshot import (  # noqa: E402
    restore_continuation_state,
    snapshot_continuation_state,
    state_container_summary,
)
from sam3_intermot.adaptation.sam3_decoder_lit import DecoderLITConfig, SAM3DecoderLITAdapter  # noqa: E402
from scripts.n29_lit_online_replay import (  # noqa: E402
    DecoderCapture,
    _clone_tree,
    _get_official_decoder,
    _image_files,
    _install_official_box_singleton,
    _make_backend,
    _read_gt,
    _session,
)
from scripts.n29r_paired_replay import (  # noqa: E402
    _branch_metrics,
    _difference,
    _ensure_public_singleton_binding,
    _horizon_metrics,
    _load_manifest,
)
from scripts.n30_write_path_ablation import _lora_state_summary, _run_lora_update  # noqa: E402


CHECKPOINT = ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
HARD_MANIFEST = ROOT / "outputs/n29r/hard_episode_manifest.json"
OUT_DIR = ROOT / "outputs/n31"

BRANCHES = (
    "P0_no_correction_resume_control",
    "P1_mapping_rebind_only",
    "P2_official_correct_object_no_fallback",
    "P3_restore_old_state_after_prompt_failure",
    "P4_corrected_rectangle_masklet",
    "P5_current_ensure_path",
    "P6_online_lora_on_best_fixed_write",
    "P7_frozen_interactive_sam_masklet",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    raise TypeError(f"not JSON serializable: {type(value)}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_json_default).encode("utf-8")).hexdigest()


def _official_ids(backend: Any) -> list[int]:
    return tracker_ids(backend)


def _obs_summary(observations: Any) -> dict[str, Any]:
    items = list(observations or [])
    masklet_count = 0
    for item in items:
        mask = np.asarray(getattr(item, "mask", np.zeros((1, 1))), dtype=bool)
        if mask.ndim == 2 and mask.size > 1:
            masklet_count += 1
    return {
        "returned_count": len(items),
        "returned_sam_ids": [int(getattr(item, "sam_object_id", -1)) for item in items],
        "returned_masklet_count": masklet_count,
    }


@contextmanager
def _prompt_trace(backend: Any) -> Iterator[list[dict[str, Any]]]:
    original = backend._send_prompt
    calls: list[dict[str, Any]] = []

    def wrapped(frame_idx: int, *, boxes: Optional[list[np.ndarray]] = None, text: Optional[str] = None, source: str) -> Any:
        observations = original(frame_idx, boxes=boxes, text=text, source=source)
        calls.append({
            "frame": int(frame_idx),
            "source": str(source),
            "box_count": 0 if boxes is None else len(boxes),
            "text_present": text is not None,
            **_obs_summary(observations),
        })
        return observations

    backend._send_prompt = wrapped
    try:
        yield calls
    finally:
        backend._send_prompt = original


def _cache_signature(backend: Any, frame: int) -> Any:
    rows = []
    for observation in backend._output_cache.get(int(frame), []):
        mask = np.asarray(getattr(observation, "mask", np.zeros((1, 1))), dtype=bool)
        rows.append({
            "id": int(getattr(observation, "sam_object_id", -1)),
            "box": np.asarray(getattr(observation, "box_xyxy", []), dtype=float).round(5).tolist(),
            "mask_shape": list(mask.shape),
            "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
            "confidence": float(getattr(observation, "confidence", 0.0)),
            "source": str(getattr(observation, "source", "")),
        })
    return sorted(rows, key=lambda row: (row["id"], row["box"]))


def _restore_external_correction_cache(backend: Any, prefix_cache: Mapping[int, Any], correction: int) -> None:
    """Keep the externally reported correction frame out of branch deltas."""

    original = prefix_cache.get(int(correction), [])
    backend._output_cache[int(correction)] = [
        item.copy() if hasattr(item, "copy") else item for item in original
    ]


def _add_refine_history(backend: Any, frame: int, public_id: int) -> None:
    state = backend._predictor._all_inference_states[backend._session_id]["state"]
    raw_id = int(backend._ext_to_sam.get(int(public_id), int(public_id)))
    add_history = getattr(backend._predictor.model, "add_action_history", None)
    if add_history is not None:
        add_history(state, action_type="refine", frame_idx=int(frame), obj_ids=[raw_id])


def _future_signature(outputs: Mapping[int, Sequence[Any]]) -> list[dict[str, Any]]:
    result = []
    for frame in sorted(outputs):
        for observation in sorted(outputs[frame], key=lambda item: int(getattr(item, "sam_object_id", -1))):
            mask = np.asarray(getattr(observation, "mask", np.zeros((1, 1))), dtype=bool)
            result.append({
                "frame": int(frame),
                "box": np.asarray(getattr(observation, "box_xyxy", []), dtype=float).round(5).tolist(),
                "mask_shape": list(mask.shape),
                "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                "confidence": float(getattr(observation, "confidence", 0.0)),
                "presence_score": None if getattr(observation, "presence_score", None) is None else float(observation.presence_score),
                "source": str(getattr(observation, "source", "")),
            })
    return result


def _mean_metric(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
    return None if not values else float(np.mean(values))


def _bootstrap(values: Sequence[float], seed: int, draws: int = 2000) -> Optional[list[float]]:
    if not values:
        return None
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = data[rng.integers(0, len(data), size=(draws, len(data)))].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _cluster_bootstrap(rows: Sequence[Mapping[str, Any]], seed: int, draws: int = 2000) -> Optional[list[float]]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(str(row["sequence"]), []).append(float(row["value"]))
    if not groups:
        return None
    names = sorted(groups)
    if len(names) == 1:
        return [float(np.mean(groups[names[0]])), float(np.mean(groups[names[0]]))]
    group_values = np.asarray([np.mean(groups[name]) for name in names], dtype=float)
    rng = np.random.default_rng(seed)
    samples = group_values[rng.integers(0, len(group_values), size=(draws, len(group_values)))].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _conditional_iou_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> Optional[float]:
    left_rows = {int(row["frame"]): row for row in left.get("rows", [])}
    right_rows = {int(row["frame"]): row for row in right.get("rows", [])}
    values = []
    for frame in sorted(set(left_rows) & set(right_rows)):
        a, b = left_rows[frame], right_rows[frame]
        if a.get("target_present") and b.get("target_present") and a.get("prediction_present") and b.get("prediction_present"):
            values.append(float(a["box_iou"]) - float(b["box_iou"]))
    return None if not values else float(np.mean(values))


def _pair_summary(results: Sequence[Mapping[str, Any]], left_name: str, right_name: str, horizon: str) -> dict[str, Any]:
    metrics = ("mean_box_iou_visible", "success_at_0_5_visible", "missing_prediction_rate_visible", "mask_area_drift")
    out: dict[str, Any] = {}
    for metric in metrics:
        values = []
        cluster_rows = []
        for episode in results:
            left = episode.get("branches", {}).get(left_name, {})
            right = episode.get("branches", {}).get(right_name, {})
            lm = left.get("metrics", {}).get(horizon)
            rm = right.get("metrics", {}).get(horizon)
            if not isinstance(lm, Mapping) or not isinstance(rm, Mapping):
                continue
            a, b = lm.get(metric), rm.get(metric)
            if a is None or b is None:
                continue
            value = float(a) - float(b)
            values.append(value)
            cluster_rows.append({"value": value, "sequence": episode["sequence"]})
        out[metric] = {
            "mean": None if not values else float(np.mean(values)),
            "sample_count": len(values),
            "negative_rate": None if not values else float(np.mean(np.asarray(values) < 0.0)),
            "episode_bootstrap_ci95": _bootstrap(values, 31031 + int(horizon) + len(left_name)),
            "sequence_cluster_bootstrap_ci95": _cluster_bootstrap(cluster_rows, 41031 + int(horizon) + len(left_name)),
        }
    conditional = []
    for episode in results:
        left = episode.get("branches", {}).get(left_name, {})
        right = episode.get("branches", {}).get(right_name, {})
        lm = left.get("metrics", {}).get(horizon)
        rm = right.get("metrics", {}).get(horizon)
        if isinstance(lm, Mapping) and isinstance(rm, Mapping):
            value = _conditional_iou_delta(lm, rm)
            if value is not None:
                conditional.append({"value": value, "sequence": episode["sequence"]})
    values = [float(row["value"]) for row in conditional]
    out["conditional_iou_given_both"] = {
        "mean": None if not values else float(np.mean(values)),
        "sample_count": len(values),
        "episode_bootstrap_ci95": _bootstrap(values, 51031 + int(horizon)),
        "sequence_cluster_bootstrap_ci95": _cluster_bootstrap(conditional, 61031 + int(horizon)),
    }
    return out


def _branch_quality_summary(results: Sequence[Mapping[str, Any]], branch_name: str) -> dict[str, Any]:
    rows = [episode["branches"].get(branch_name, {}) for episode in results]
    out: dict[str, Any] = {
        "availability_rate": float(np.mean([bool(row.get("available", False)) for row in rows])) if rows else None,
        "target_state_present_rate": _mean_metric(rows, "target_state_present"),
        "protected_identity_namespace_unchanged_rate": _mean_metric(rows, "protected_identity_namespace_unchanged"),
        "prompt_success_rate": _mean_metric(rows, "prompt_success"),
        "quality_available_rate": float(np.mean([bool(row.get("quality_available", False)) for row in rows])) if rows else None,
        "horizons": {},
    }
    for horizon in ("5", "10", "20"):
        metrics = [row.get("metrics", {}).get(horizon, {}) for row in rows]
        out["horizons"][horizon] = {
            key: _mean_metric(metrics, key)
            for key in (
                "mean_box_iou_visible",
                "success_at_0_5_visible",
                "missing_prediction_rate_visible",
                "error_count_visible",
                "mask_area_drift",
                "box_area_drift_proxy",
            )
        }
    return out


def _episode_case(episode: Mapping[str, Any], query_end: int) -> dict[str, Any]:
    return {
        **episode,
        "episode_id": str(episode["episode_id"]),
        "query_end": int(query_end),
        "correction_box": list(episode["correction_box"]),
    }


def _run_branch(
    *,
    backend: Any,
    decoder: Optional[torch.nn.Module],
    adapter: Optional[SAM3DecoderLITAdapter],
    capture: DecoderCapture,
    episode: Mapping[str, Any],
    sequence: Path,
    prefix_snapshot: Any,
    prefix_cache: Mapping[int, Any],
    prefix_protected: Mapping[str, Any],
    support_kwargs: Optional[Mapping[str, Any]],
    branch_name: str,
    gt: Mapping[int, Mapping[int, np.ndarray]],
    query_end: int,
    correction: int,
    public_id: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    restore_continuation_state(backend, prefix_snapshot)
    before = snapshot_backend_state(backend, target_id=public_id, correction_frame=correction)
    before_container = state_container_summary(backend)
    prompt_calls: list[dict[str, Any]] = []
    writer: dict[str, Any] = {"status": "NOT_RUN", "reason": "branch_has_no_writer"}
    lora_state = None
    lora_update: dict[str, Any] = {"status": "NOT_RUN", "committed": False, "reason": "branch_has_no_lora_write"}
    restored_after_failure = False
    prompt_success: Optional[bool] = None
    available = True
    quality_available = False
    failure: Optional[str] = None
    future_adapter_active = False
    with _prompt_trace(backend) as prompt_calls_ref:
        try:
            if branch_name == "P0_no_correction_resume_control":
                prompt_success = False
            elif branch_name == "P1_mapping_rebind_only":
                raw_ids = _official_ids(backend)
                mapped_raw = backend._ext_to_sam.get(public_id)
                if mapped_raw is not None and int(mapped_raw) in raw_ids:
                    # An existing one-to-one external mapping is already
                    # enough even when the official namespace contains an
                    # additional unmatched raw object.  Mapping-only must not
                    # delete or guess that other state.
                    backend._bind_external_sam_id(public_id, int(mapped_raw))
                    selected_raw = int(mapped_raw)
                elif len(raw_ids) == 1:
                    backend._bind_external_sam_id(public_id, raw_ids[0])
                    selected_raw = int(raw_ids[0])
                else:
                    raise RuntimeError(f"mapping-only branch has no unambiguous existing mapping: raw_ids={raw_ids}, mapped={mapped_raw}")
                prompt_success = False
                writer = {"status": "NOT_WRITTEN", "method": "mapping_only", "raw_id": selected_raw, "raw_ids_present": raw_ids}
            elif branch_name in ("P2_official_correct_object_no_fallback", "P3_restore_old_state_after_prompt_failure"):
                backend.correct_object(
                    correction,
                    public_id,
                    box_xyxy=np.asarray(episode["correction_box"], dtype=float),
                    allow_prompt_fallback=False,
                )
                prompt_success = any(
                    int(call.get("frame", -1)) == correction and int(call.get("returned_masklet_count", 0)) > 0
                    for call in prompt_calls_ref
                )
                writer = {
                    "status": "OFFICIAL_PROMPT",
                    "allow_prompt_fallback": False,
                    "prompt_success": bool(prompt_success),
                }
                if branch_name == "P3_restore_old_state_after_prompt_failure" and not prompt_success:
                    restore_continuation_state(backend, prefix_snapshot)
                    restored_after_failure = True
                    writer["restore_status"] = "RESTORED_PREFIX_CONTINUATION"
            elif branch_name == "P4_corrected_rectangle_masklet":
                state = backend._predictor._all_inference_states[backend._session_id]["state"]
                mask = rectangle_mask(
                    episode["correction_box"],
                    int(state["orig_height"]),
                    int(state["orig_width"]),
                    device=state["device"],
                )
                writer = write_target_mask(
                    backend,
                    frame_idx=correction,
                    public_id=public_id,
                    mask=mask,
                    provenance=BOX_RECTANGLE_MASKLET,
                )
                _add_refine_history(backend, correction, public_id)
                prompt_success = True
            elif branch_name in ("P5_current_ensure_path", "P6_online_lora_on_best_fixed_write"):
                backend.correct_object(correction, public_id, box_xyxy=np.asarray(episode["correction_box"], dtype=float))
                prompt_success = any(
                    int(call.get("frame", -1)) == correction and int(call.get("returned_masklet_count", 0)) > 0
                    for call in prompt_calls_ref
                )
                binding = _ensure_public_singleton_binding(
                    backend,
                    frame=correction,
                    public_id=public_id,
                    box=np.asarray(episode["correction_box"], dtype=float),
                )
                writer = {
                    "status": "CURRENT_ENSURE_PATH",
                    "prompt_success": bool(prompt_success),
                    "binding": binding,
                }
                if branch_name == "P6_online_lora_on_best_fixed_write":
                    if decoder is None or adapter is None:
                        lora_update = {"status": "NOT_RUN", "committed": False, "reason": "decoder_adapter_unavailable"}
                    else:
                        lora_state, lora_update = _run_lora_update(
                            backend=backend,
                            adapter=adapter,
                            decoder=decoder,
                            episode=_episode_case(episode, query_end),
                            support_kwargs=None if support_kwargs is None else _clone_tree(support_kwargs),
                            current_output_recorded=True,
                        )
            elif branch_name == "P7_frozen_interactive_sam_masklet":
                candidates = interactive_box_candidates(
                    backend,
                    frame_idx=correction,
                    box_xyxy=episode["correction_box"],
                )
                top = int(candidates["rank_order"][0])
                state = backend._predictor._all_inference_states[backend._session_id]["state"]
                selected = {
                    "box_xyxy": list(episode["correction_box"]),
                    "token_index": top,
                    "predicted_iou": candidates["predicted_iou"],
                    "video_width": int(state["orig_width"]),
                    "video_height": int(state["orig_height"]),
                    "mask_area_ratio": float((candidates["masks"][top] > 0.5).float().mean().item()),
                }
                writer = write_target_mask(
                    backend,
                    frame_idx=correction,
                    public_id=public_id,
                    mask=candidates["masks"][top],
                    provenance=BOX_PROMPTED_SAM_PSEUDO_MASK,
                )
                writer["candidate_features"] = candidate_mask_features(selected)
                writer["token_index"] = top
                writer["predicted_iou"] = float(candidates["predicted_iou"][top])
                writer["token_count"] = int(candidates["token_count"])
                _add_refine_history(backend, correction, public_id)
                prompt_success = True
                quality_available = True
            else:  # pragma: no cover - guarded by BRANCHES
                raise ValueError(f"unknown N31 branch {branch_name}")
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            if branch_name == "P7_frozen_interactive_sam_masklet" and any(
                token in str(exc).lower() for token in ("unavailable", "feature cache", "interactive")
            ):
                available = False
                quality_available = False
                writer = {"status": "NOT_AVAILABLE", "reason": failure}
            elif branch_name in ("P2_official_correct_object_no_fallback", "P3_restore_old_state_after_prompt_failure"):
                # A failed unmodified prompt is a measured availability result;
                # P3 must still restore the prefix before its future attempt.
                available = False
                writer = {"status": "PROMPT_FAILED", "reason": failure, "allow_prompt_fallback": False}
                prompt_success = False
                if branch_name == "P3_restore_old_state_after_prompt_failure":
                    try:
                        restore_continuation_state(backend, prefix_snapshot)
                        restored_after_failure = True
                        failure = None
                        writer["restore_status"] = "RESTORED_PREFIX_CONTINUATION"
                    except Exception as restore_exc:
                        failure = f"{failure}; restore={type(restore_exc).__name__}: {restore_exc}"
            else:
                available = False

        prompt_calls = list(prompt_calls_ref)

    after = snapshot_backend_state(backend, target_id=public_id, correction_frame=correction)
    after_container = state_container_summary(backend)
    state_delta = diff_snapshots(before, after)
    protected_after = protected_state_signatures(backend, exclude_public_id=public_id)
    protected_namespace_unchanged = sorted(prefix_protected) == sorted(protected_after)
    _restore_external_correction_cache(backend, prefix_cache, correction)

    future_outputs: dict[int, list[Any]] = {}
    metrics: dict[str, Any] = {}
    if available and failure is None:
        try:
            if branch_name == "P6_online_lora_on_best_fixed_write" and bool(lora_update.get("committed", False)) and lora_state is not None and adapter is not None:
                with adapter.activate(lora_state):
                    future_outputs = backend.propagate(correction + 1, query_end, start_frame_index=correction + 1)
                future_adapter_active = True
            else:
                future_outputs = backend.propagate(correction + 1, query_end, start_frame_index=correction + 1)
            for frame, observations in future_outputs.items():
                backend._output_cache[int(frame)] = observations
            metrics = _horizon_metrics(future_outputs, gt, _episode_case(episode, query_end))
        except Exception as exc:
            failure = f"future_{type(exc).__name__}: {exc}"
            available = False

    return {
        "status": "PASS" if failure is None else "PARTIAL",
        "available": bool(available and failure is None),
        "quality_available": bool(quality_available),
        "prompt_success": None if prompt_success is None else bool(prompt_success),
        "target_state_present": float(int(public_id in _official_ids(backend) or int(backend._ext_to_sam.get(public_id, public_id)) in _official_ids(backend))),
        "protected_identity_namespace_unchanged": float(bool(protected_namespace_unchanged)),
        "branch": branch_name,
        "writer": writer,
        "prompt_trace": prompt_calls,
        "restored_after_prompt_failure": bool(restored_after_failure),
        "state_delta": state_delta,
        "prefix_container_summary": before_container,
        "post_intervention_container_summary": after_container,
        "mapping_after": {str(k): int(v) for k, v in backend._ext_to_sam.items()},
        "correction_frame_external_cache_unchanged": _cache_signature(backend, correction) == _cache_signature_from_cache(prefix_cache, correction),
        "future_adapter_active": bool(future_adapter_active),
        "future_frame_count": len(future_outputs),
        "future_raw_signature": _future_signature(future_outputs),
        "metrics": metrics,
        "lora_update": lora_update,
        "lora_state": _lora_state_summary(lora_state, update_called=branch_name == "P6_online_lora_on_best_fixed_write", update=lora_update),
        "failure": failure,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _cache_signature_from_cache(cache: Mapping[int, Any], frame: int) -> Any:
    rows = []
    for observation in cache.get(int(frame), []):
        mask = np.asarray(getattr(observation, "mask", np.zeros((1, 1))), dtype=bool)
        rows.append({
            "id": int(getattr(observation, "sam_object_id", -1)),
            "box": np.asarray(getattr(observation, "box_xyxy", []), dtype=float).round(5).tolist(),
            "mask_shape": list(mask.shape),
            "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
            "confidence": float(getattr(observation, "confidence", 0.0)),
            "source": str(getattr(observation, "source", "")),
        })
    return sorted(rows, key=lambda row: (row["id"], row["box"]))


def _run_episode(
    *,
    backend: Any,
    decoder: Optional[torch.nn.Module],
    adapter: Optional[SAM3DecoderLITAdapter],
    capture: DecoderCapture,
    episode: Mapping[str, Any],
    sequence: Path,
) -> dict[str, Any]:
    gt = _read_gt(sequence)
    images = _image_files(sequence)
    init = int(episode["initialization_frame"])
    correction = int(episode["correction_frame"])
    query_end = min(int(episode["query_end"]), len(images) - 1)
    public_id = int(episode["public_id"])
    dataset_identity = int(episode["dataset_identity"])
    if query_end < correction + 20:
        raise ValueError(f"N31 requires H20 but episode ends at {query_end}: {episode['episode_id']}")
    init_box = np.asarray(gt[init][dataset_identity], dtype=float)
    _session(backend, sequence)
    backend.add_box(init, public_id, init_box)
    anchor_binding = _install_official_box_singleton(backend, frame_idx=init, public_id=public_id, box_xyxy=init_box)
    capture.reset(target_call=max(0, correction - init))
    prefix_outputs = backend.propagate(init, correction, start_frame_index=init)
    support_kwargs = None if capture.target_inputs is None else _clone_tree(capture.target_inputs)
    prefix_snapshot = snapshot_continuation_state(backend)
    prefix_cache = {int(frame): [item.copy() if hasattr(item, "copy") else item for item in observations] for frame, observations in backend._output_cache.items()}
    prefix_protected = protected_state_signatures(backend, exclude_public_id=public_id)
    prefix_digest = _hash_json({
        "container": state_container_summary(backend),
        "correction_cache": _cache_signature(backend, correction),
        "prefix_frames": sorted(int(frame) for frame in prefix_outputs),
    })
    branches: dict[str, Any] = {}
    for branch_name in BRANCHES:
        branches[branch_name] = _run_branch(
            backend=backend,
            decoder=decoder,
            adapter=adapter,
            capture=capture,
            episode=episode,
            sequence=sequence,
            prefix_snapshot=prefix_snapshot,
            prefix_cache=prefix_cache,
            prefix_protected=prefix_protected,
            support_kwargs=support_kwargs,
            branch_name=branch_name,
            gt=gt,
            query_end=query_end,
            correction=correction,
            public_id=public_id,
        )
    p1 = branches["P1_mapping_rebind_only"]
    p0 = branches["P0_no_correction_resume_control"]
    p1_raw = p1.get("future_raw_signature", [])
    p0_raw = p0.get("future_raw_signature", [])
    return {
        "status": "PASS" if all(row.get("status") in {"PASS", "PARTIAL"} for row in branches.values()) else "PARTIAL",
        "episode_id": str(episode["episode_id"]),
        "sequence": str(episode["sequence"]),
        "split": str(episode["split"]),
        "dataset_identity": dataset_identity,
        "public_id": public_id,
        "initialization_frame": init,
        "correction_frame": correction,
        "query_start": correction + 1,
        "query_end": query_end,
        "anchor_binding": anchor_binding,
        "support_inputs_available": support_kwargs is not None,
        "prefix_frame_count": len(prefix_outputs),
        "common_prefix_digest": prefix_digest,
        "same_prefix_snapshot_protocol": True,
        "p1_p0_raw_signature_equal": bool(p1_raw == p0_raw) if p1.get("available") and p0.get("available") else False,
        "branches": branches,
        "paired_delta": {
            f"{left}_minus_{right}": {
                horizon: _difference(
                    branches[left].get("metrics", {}).get(horizon, {}),
                    branches[right].get("metrics", {}).get(horizon, {}),
                )
                for horizon in ("5", "10", "20")
            }
            for left, right in (
                ("P1_mapping_rebind_only", "P0_no_correction_resume_control"),
                ("P3_restore_old_state_after_prompt_failure", "P0_no_correction_resume_control"),
                ("P4_corrected_rectangle_masklet", "P3_restore_old_state_after_prompt_failure"),
                ("P5_current_ensure_path", "P4_corrected_rectangle_masklet"),
                ("P6_online_lora_on_best_fixed_write", "P5_current_ensure_path"),
                ("P7_frozen_interactive_sam_masklet", "P4_corrected_rectangle_masklet"),
            )
        },
        "gt_role": "current correction validation and post-hoc train-fold evaluation only",
    }


def _build_gate(results: Sequence[Mapping[str, Any]], protected_path: Path) -> dict[str, Any]:
    p1_equivalence = bool(results) and all(bool(row.get("p1_p0_raw_signature_equal")) for row in results)
    fixed_signals = []
    for name in ("P4_corrected_rectangle_masklet", "P5_current_ensure_path", "P7_frozen_interactive_sam_masklet"):
        values = []
        for row in results:
            left = row.get("branches", {}).get(name, {}).get("metrics", {}).get("20")
            base = row.get("branches", {}).get("P0_no_correction_resume_control", {}).get("metrics", {}).get("20")
            if isinstance(left, Mapping) and isinstance(base, Mapping):
                value = _conditional_iou_delta(left, base)
                if value is not None:
                    values.append(value)
        if values:
            fixed_signals.append({"branch": name, "conditional_h20_iou_gain": float(np.mean(values)), "sample_count": len(values)})
    legitimate = any(float(item["conditional_h20_iou_gain"]) > 0.0 for item in fixed_signals)
    protected = {"status": "NOT_RUN", "reason": "protected scope artifact not present"}
    if protected_path.is_file():
        try:
            protected = json.loads(protected_path.read_text(encoding="utf-8"))
        except Exception as exc:
            protected = {"status": "FAIL", "reason": f"cannot read protected artifact: {exc}"}
    protected_pass = protected.get("status") == "PASS" and bool(protected.get("unaffected_ids_preserved", False))
    return {
        "protocol": "N31-C-STATE-GATE",
        "status": "PASS" if p1_equivalence and legitimate and protected_pass else "FAIL",
        "p1_p0_raw_equivalence": p1_equivalence,
        "legitimate_fixed_write_signal": legitimate,
        "fixed_write_signals": fixed_signals,
        "mapping_only_not_quality": p1_equivalence,
        "protected_identity_regression": protected,
        "protected_identity_pass": protected_pass,
        "availability_is_reported_separately": True,
        "required_branches": list(BRANCHES),
    }


def run(
    *,
    manifest_path: Path,
    checkpoint: Path,
    output: Path,
    summary_output: Path,
    gate_output: Path,
    audit_output: Path,
    limit: Optional[int],
    resume: bool,
) -> dict[str, Any]:
    manifest, manifest_sha = _load_manifest(manifest_path)
    episodes = list(manifest["episodes"])
    if len(episodes) != 50 and limit is None:
        raise ValueError(f"N31-C requires exactly the frozen N29-R 50 episodes, got {len(episodes)}")
    if limit is not None:
        episodes = episodes[: int(limit)]
    partial = output.with_suffix(".partial.json")
    previous: dict[str, Any] = {}
    if resume and partial.is_file():
        try:
            old = json.loads(partial.read_text(encoding="utf-8"))
            previous = {str(row["episode_id"]): row for row in old.get("episode_results", [])}
        except Exception:
            previous = {}
    backend = _make_backend(checkpoint)
    decoder: Optional[torch.nn.Module] = None
    adapter: Optional[SAM3DecoderLITAdapter] = None
    capture: Optional[DecoderCapture] = None
    results: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for episode in episodes:
            episode_id = str(episode["episode_id"])
            if episode_id in previous and bool(previous[episode_id].get("p1_p0_raw_signature_equal", False)):
                results.append(previous[episode_id])
                continue
            sequence = Path(episode["sequence_path"])
            if any(token.lower() in {part.lower() for part in sequence.parts} for token in ("val", "test")):
                raise ValueError(f"N31-C refused non-train sequence {sequence}")
            try:
                _session(backend, sequence)
                if decoder is None:
                    decoder = _get_official_decoder(backend)
                    adapter = SAM3DecoderLITAdapter(decoder, DecoderLITConfig(rank=4, alpha=4.0, dropout=0.1))
                    capture = DecoderCapture(decoder)
                if capture is None:
                    raise RuntimeError("N31-C decoder capture was not initialized")
                row = _run_episode(
                    backend=backend,
                    decoder=decoder,
                    adapter=adapter,
                    capture=capture,
                    episode=episode,
                    sequence=sequence,
                )
                results.append(row)
                audits.append({"episode_id": episode_id, "common_prefix_digest": row["common_prefix_digest"], "same_prefix_snapshot_protocol": True})
            except Exception as exc:
                results.append({
                    "status": "NOT_RUN",
                    "episode_id": episode_id,
                    "sequence": str(episode["sequence"]),
                    "split": str(episode["split"]),
                    "failure": f"{type(exc).__name__}: {exc}",
                    "failure_traceback": traceback.format_exc(limit=20),
                })
            _write_json(partial, {
                "protocol": "N31-CORRECTION-STATE-ABLATION",
                "status": "PARTIAL",
                "episode_results": results,
                "audits": audits,
                "manifest_sha256": manifest_sha,
            })
    finally:
        if capture is not None:
            capture.close()
        backend.close()
    summary = {
        "protocol": "N31-CORRECTION-STATE-ABLATION-SUMMARY",
        "status": "PASS" if results and all(row.get("status") in {"PASS", "PARTIAL"} for row in results) else "PARTIAL",
        "episode_count": len(results),
        "branch_order": list(BRANCHES),
        "horizons": [5, 10, 20],
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "checkpoint": str(checkpoint),
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_selection": False,
        "same_prefix_snapshot_protocol": True,
        "branch_quality": {name: _branch_quality_summary(results, name) for name in BRANCHES},
        "comparisons": {
            f"{left}_minus_{right}": {
                horizon: _pair_summary(results, left, right, horizon)
                for horizon in ("5", "10", "20")
            }
            for left, right in (
                ("P1_mapping_rebind_only", "P0_no_correction_resume_control"),
                ("P3_restore_old_state_after_prompt_failure", "P0_no_correction_resume_control"),
                ("P4_corrected_rectangle_masklet", "P3_restore_old_state_after_prompt_failure"),
                ("P5_current_ensure_path", "P4_corrected_rectangle_masklet"),
                ("P6_online_lora_on_best_fixed_write", "P5_current_ensure_path"),
                ("P7_frozen_interactive_sam_masklet", "P4_corrected_rectangle_masklet"),
            )
        },
        "timing": {"elapsed_seconds": float(time.perf_counter() - started)},
    }
    gate = _build_gate(results, OUT_DIR / "protected_identity_scope.json")
    result = {
        "protocol": "N31-CORRECTION-STATE-ABLATION",
        "status": summary["status"],
        "episode_results": results,
        "audit": audits,
        "summary": str(summary_output),
        "gate": str(gate_output),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "branch_order": list(BRANCHES),
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_selection": False,
        "same_prefix_snapshot_protocol": True,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _write_json(output, result)
    _write_json(summary_output, summary)
    _write_json(gate_output, gate)
    _write_json(audit_output, {"protocol": "N31-C-STATE-AUDIT", "audits": audits, "episode_count": len(audits)})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HARD_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "correction_state_ablation.json")
    parser.add_argument("--summary-output", type=Path, default=OUT_DIR / "correction_state_ablation_summary.json")
    parser.add_argument("--gate-output", type=Path, default=OUT_DIR / "correction_state_gate.json")
    parser.add_argument("--audit-output", type=Path, default=OUT_DIR / "correction_state_ablation_audit.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(
        manifest_path=args.manifest,
        checkpoint=args.checkpoint,
        output=args.output,
        summary_output=args.summary_output,
        gate_output=args.gate_output,
        audit_output=args.audit_output,
        limit=args.limit,
        resume=args.resume,
    )
    print(json.dumps({key: result[key] for key in ("protocol", "status", "branch_order", "elapsed_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
