#!/usr/bin/env python3
"""N29-B: faithful SAM3 propagation-decoder A+B online replay.

The default ``--mode fixture`` is a deterministic engineering replay using
the official TwoWayTransformer shape.  ``--mode real`` loads the pinned local
SAM3.1 checkpoint, calls the official multiplex predictor/propagation path,
and applies the adapter update to captured official propagation-decoder
inputs.  Real runs use DanceTrack *train* sequences only; this script never
opens the 25-sequence validation split.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SAM3_ROOT = ROOT / "third_party" / "sam3"
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

from sam3_intermot.adaptation.corrected_mask_teacher import (  # noqa: E402
    BOX_DERIVED_PSEUDO_MASK,
)
from sam3_intermot.adaptation.decoder_update_transaction import (  # noqa: E402
    DecoderCorrectionEvent,
    DecoderUpdateConfig,
    DecoderUpdateTransaction,
)
from sam3_intermot.adaptation.sam3_decoder_lit import (  # noqa: E402
    DecoderLITConfig,
    IdentityAdapterState,
    SAM3DecoderLITAdapter,
)
from sam3_intermot.association.decoder_candidate_bridge import (  # noqa: E402
    DecoderCandidate,
    build_decoder_assignment,
    official_output_to_decoder_candidate,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)}")


def _iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(x) for x in a)
    bx1, by1, bx2, by2 = (float(x) for x in b)
    x1, y1, x2, y2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def _read_gt(sequence_dir: Path) -> dict[int, dict[int, np.ndarray]]:
    result: dict[int, dict[int, np.ndarray]] = {}
    gt_path = sequence_dir / "gt" / "gt.txt"
    if not gt_path.is_file():
        raise FileNotFoundError(f"DanceTrack GT is unavailable: {gt_path}")
    for line in gt_path.read_text().splitlines():
        fields = line.strip().split(",")
        if len(fields) < 6:
            continue
        frame = int(fields[0]) - 1
        identity = int(fields[1])
        x, y, w, h = (float(fields[i]) for i in range(2, 6))
        result.setdefault(frame, {})[identity] = np.asarray([x, y, x + w, y + h], dtype=float)
    return result


def _image_files(sequence_dir: Path) -> list[Path]:
    images = list((sequence_dir / "img1").glob("*.jpg"))
    return sorted(images, key=lambda path: int(path.stem))


def _first_gt_identity(gt: Mapping[int, Mapping[int, np.ndarray]]) -> tuple[int, int]:
    for frame in sorted(gt):
        if gt[frame]:
            identity = sorted(gt[frame])[0]
            return frame, identity
    raise ValueError("sequence has no GT identity")


@dataclass(frozen=True)
class ReplayIdentityBinding:
    """Keep dataset, public, and official tracker namespaces distinct."""

    dataset_identity: int
    public_id: int
    sam_object_id: int


def _select_observation(observations: list[Any], public_id: int, target: Optional[np.ndarray]) -> Optional[Any]:
    for observation in observations:
        if int(getattr(observation, "sam_object_id", -1)) == int(public_id):
            return observation
    if target is None or not observations:
        return None
    return max(observations, key=lambda obs: _iou(obs.box_xyxy, target))


def _box_candidate(observation: Any, frame_idx: int) -> DecoderCandidate:
    mask = np.asarray(observation.mask, dtype=bool)
    return DecoderCandidate(
        frame_idx=int(frame_idx),
        mask_logits=mask.astype(np.float32),
        mask=mask,
        box_xyxy=tuple(float(x) for x in observation.box_xyxy),
        presence=float(observation.presence_score or observation.confidence),
        iou_pred=float(observation.confidence),
        decoder_token=None,
        clip_feature=None,
        source="original_anchor",
        source_public_id=None,
    )


def _clone_tree(value: Any) -> Any:
    """Make normal (non-inference) tensors for a differentiable decoder call."""

    if isinstance(value, torch.Tensor):
        # ``torch.empty_like`` preserves the inference-tensor bit for inputs
        # captured inside the official ``@torch.inference_mode`` propagation
        # path.  Allocate by shape instead, then copy, so autograd can save
        # these support tensors during the decoder update.
        with torch.inference_mode(False):
            result = torch.empty(
                tuple(value.shape),
                dtype=value.dtype,
                device=value.device,
                layout=value.layout,
            )
            result.copy_(value)
            if torch.is_inference(result):
                result = torch.tensor(value, dtype=value.dtype, device=value.device)
        return result
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return value


def _materialize_normal_module_tensors(module: nn.Module) -> dict[str, list[str]]:
    """Copy any inference-mode params/buffers before entering autograd."""

    materialized = {"parameters": [], "buffers": []}

    def replace(path: str, value: torch.Tensor, *, parameter: bool) -> None:
        parent_path, _, child = path.rpartition(".")
        parent = module
        for part in parent_path.split(".") if parent_path else ():
            parent = getattr(parent, part)
        with torch.inference_mode(False):
            copied = torch.empty(
                tuple(value.shape),
                dtype=value.dtype,
                device=value.device,
                layout=value.layout,
            )
            copied.copy_(value)
        if parameter:
            setattr(parent, child, nn.Parameter(copied, requires_grad=value.requires_grad))
        else:
            parent._buffers[child] = copied

    for name, parameter in list(module.named_parameters()):
        if torch.is_inference(parameter):
            replace(name, parameter, parameter=True)
            materialized["parameters"].append(name)
    for name, buffer in list(module.named_buffers()):
        if torch.is_inference(buffer):
            replace(name, buffer, parameter=False)
            materialized["buffers"].append(name)
    return materialized


def _tensor_status_tree(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Summarize tensor provenance for a failed autograd binding."""

    if isinstance(value, torch.Tensor):
        return [
            {
                "name": prefix,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "is_inference": bool(torch.is_inference(value)),
                "requires_grad": bool(value.requires_grad),
            }
        ]
    if isinstance(value, Mapping):
        rows = []
        for key, item in value.items():
            rows.extend(_tensor_status_tree(item, f"{prefix}.{key}" if prefix else str(key)))
        return rows
    if isinstance(value, (list, tuple)):
        rows = []
        for index, item in enumerate(value):
            rows.extend(_tensor_status_tree(item, f"{prefix}[{index}]"))
        return rows
    return []


def _slot_tensor(value: Any, slot: int = 0, token: int = 0) -> Optional[torch.Tensor]:
    if value is None:
        return None
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim >= 5:  # [B, multiplex, mask_token, H, W]
        return tensor[0, slot, token : token + 1]
    if tensor.ndim == 4:  # [B, mask_token, H, W] or [B, H, W, C]
        return tensor[0, token : token + 1]
    if tensor.ndim == 3:
        return tensor[slot : slot + 1]
    return tensor.reshape(1)


def _slot_value(value: Any, slot: int = 0, token: int = 0) -> Any:
    """Select one multiplex slot from scalar/array official outputs."""

    if value is None:
        return None
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim >= 4:
        return array[0, slot, token]
    if array.ndim == 3:
        return array[0, slot, token] if array.shape[2] > token else array[0, slot]
    if array.ndim == 2:
        return array[0, slot]
    if array.ndim == 1:
        return array[slot]
    return array


def _slot_output(output: Mapping[str, Any], slot: int = 0) -> dict[str, Any]:
    masks = _slot_tensor(output.get("masks"), slot=slot)
    if masks is None:
        raise ValueError("official propagation decoder output has no masks")
    iou = _slot_value(output.get("iou_pred"), slot=slot, token=0)
    presence = _slot_value(output.get("object_score_logits"), slot=slot, token=0)
    tokens = _slot_value(
        output.get("sam_tokens_out", output.get("mask_tokens_out")),
        slot=slot,
        token=0,
    )
    return {
        "high_res_masks": masks,
        "ious": iou,
        "object_score_logits": presence,
        "sam_output_token": tokens,
    }


class DecoderCapture:
    """Capture only adapter-level propagation inputs/outputs, never images."""

    def __init__(self, decoder: nn.Module):
        self.decoder = decoder
        self.call_count = 0
        self.target_call: Optional[int] = None
        self.target_inputs: Optional[dict[str, Any]] = None
        self.last_raw: Optional[dict[str, np.ndarray]] = None
        self.total_call_count = 0
        self.reset_count = 0
        self.handle_pre = decoder.register_forward_pre_hook(self._pre, with_kwargs=True)
        self.handle_post = decoder.register_forward_hook(self._post)

    def reset(self, *, target_call: Optional[int] = None) -> None:
        self.call_count = 0
        self.target_call = target_call
        self.target_inputs = None
        self.last_raw = None
        self.reset_count += 1

    def _pre(self, _module, args, kwargs):
        if self.target_call is None or self.call_count == self.target_call:
            self.target_inputs = dict(kwargs)

    def _post(self, _module, _args, output):
        self.last_raw = {}
        if isinstance(output, Mapping):
            for key in ("masks", "iou_pred", "object_score_logits", "sam_tokens_out", "mask_tokens_out"):
                value = output.get(key)
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        self.last_raw[key] = value.detach().float().cpu().numpy().copy()
                    else:
                        self.last_raw[key] = np.asarray(value).copy()
        self.call_count += 1
        self.total_call_count += 1

    def close(self) -> None:
        self.handle_pre.remove()
        self.handle_post.remove()


def _decoder_candidate_from_capture(
    capture: DecoderCapture,
    *,
    frame_idx: int,
    public_id: int,
    adapter_version: int,
) -> Optional[DecoderCandidate]:
    if capture.last_raw is None:
        return None
    return official_output_to_decoder_candidate(
        _slot_output(capture.last_raw),
        frame_idx=frame_idx,
        source_public_id=public_id,
        adapter_version=adapter_version,
        min_presence=0.01,
        keep_rejected=True,
    )


def _get_official_decoder(backend: Any) -> nn.Module:
    model = backend._predictor.model
    tracker = getattr(model, "tracker", None)
    candidates = [tracker, getattr(tracker, "model", None), model]
    for candidate in candidates:
        decoder = getattr(candidate, "sam_mask_decoder", None)
        if decoder is not None:
            return decoder
    raise AttributeError("cannot locate official tracker sam_mask_decoder")


def _short_state_diagnostic(backend: Any, capture: DecoderCapture, decoder: nn.Module) -> dict[str, Any]:
    """Record binding evidence without serializing frames/features/tensors."""

    predictor = getattr(backend, "_predictor", None)
    model = getattr(predictor, "model", None)
    tracker_wrapper = getattr(model, "tracker", None)
    tracker_model = getattr(tracker_wrapper, "model", None)
    session_id = getattr(backend, "_session_id", None)
    state_entry = None
    if predictor is not None and session_id is not None:
        state_entry = getattr(predictor, "_all_inference_states", {}).get(session_id)
    state = state_entry.get("state") if isinstance(state_entry, dict) else None

    tracker_states = []
    metadata = {}
    if isinstance(state, dict):
        for tracker_state in state.get("sam2_inference_states", []):
            if isinstance(tracker_state, dict):
                tracker_states.append(
                    {
                        "obj_ids": np.asarray(tracker_state.get("obj_ids", [])).reshape(-1).tolist(),
                        "keys": sorted(str(key) for key in tracker_state.keys()),
                    }
                )
        raw_metadata = state.get("tracker_metadata", {})
        if isinstance(raw_metadata, dict):
            for key in ("obj_ids_all_gpu", "obj_ids_per_gpu", "num_obj_per_gpu", "max_obj_id"):
                value = raw_metadata.get(key)
                if value is not None:
                    metadata[key] = np.asarray(value).tolist() if not np.isscalar(value) else int(value)

    return {
        "model_class": None if model is None else type(model).__name__,
        "tracker_wrapper_class": None if tracker_wrapper is None else type(tracker_wrapper).__name__,
        "tracker_model_class": None if tracker_model is None else type(tracker_model).__name__,
        "decoder_class": type(decoder).__name__,
        "decoder_transformer_class": type(getattr(decoder, "transformer", None)).__name__,
        "decoder_hook_call_count_since_reset": capture.call_count,
        "decoder_hook_total_call_count": capture.total_call_count,
        "decoder_hook_reset_count": capture.reset_count,
        "support_inputs_exposed": capture.target_inputs is not None,
        "support_input_keys": [] if capture.target_inputs is None else sorted(str(key) for key in capture.target_inputs),
        "official_object_table": sorted(int(key) for key in getattr(backend, "_objects", {})),
        "external_to_sam": {str(key): int(value) for key, value in getattr(backend, "_ext_to_sam", {}).items()},
        "prompt_fallback_log": list(getattr(backend, "_prompt_fallback_log", [])),
        "inference_state_keys": [] if not isinstance(state, dict) else sorted(str(key) for key in state.keys()),
        "feature_cache_keys": [] if not isinstance(state, dict) else sorted(str(key) for key in state.get("feature_cache", {}).keys()),
        "sam2_state_count": len(tracker_states),
        "sam2_states": tracker_states,
        "tracker_metadata": metadata,
    }


def _make_backend(checkpoint: Path) -> Any:
    from sam3_intermot.backend.sam3_backend import Sam3Backend

    return Sam3Backend(
        checkpoint_path=str(checkpoint),
        max_num_objects=16,
        multiplex_count=16,
        use_fa3=False,
        use_rope_real=True,
        compile=False,
        warm_up=False,
        output_prob_thresh=0.0,
        async_loading_frames=False,
        device="cuda",
    )


def _session(backend: Any, sequence_dir: Path) -> None:
    if getattr(backend, "_session_id", None) is not None:
        backend.close()
    # The pinned SAM3 loader accepts an image directory, whereas DanceTrack
    # stores images below each sequence's ``img1`` directory.
    source = sequence_dir / "img1" if (sequence_dir / "img1").is_dir() else sequence_dir
    backend.start_video(str(source))


def _box_rectangle_mask(box_xyxy: np.ndarray, height: int, width: int) -> torch.Tensor:
    """Compile an explicit rectangle pseudo-mask from a legal box correction."""

    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    left = max(0, min(width - 1, int(np.floor(x1))))
    top = max(0, min(height - 1, int(np.floor(y1))))
    right = max(left + 1, min(width, int(np.ceil(x2))))
    bottom = max(top + 1, min(height, int(np.ceil(y2))))
    mask = torch.zeros((height, width), dtype=torch.float32, device="cuda")
    mask[top:bottom, left:right] = 1.0
    return mask


def _install_official_box_singleton(
    backend: Any,
    *,
    frame_idx: int,
    public_id: int,
    box_xyxy: np.ndarray,
) -> dict[str, Any]:
    """Bind a box-derived legal mask to the official singleton tracker state.

    The pinned multiplex demo exposes the propagation decoder only after a
    tracker masklet exists.  Its semantic box API can return no masklet on a
    valid box.  This adapter-level fallback uses the official ``add_new_masks``
    path with the explicit rectangle pseudo-mask; it never fabricates a click
    and never edits the third-party source.
    """

    predictor = backend._predictor
    model = predictor.model
    session_id = backend._session_id
    entry = predictor._all_inference_states[session_id]
    state = entry["state"]
    existing = state.get("sam2_inference_states", [])
    existing_ids = [
        int(obj_id)
        for tracker_state in existing
        for obj_id in np.asarray(tracker_state.get("obj_ids", [])).reshape(-1)
    ]
    if existing_ids:
        # A low-level write can leave the official singleton alive under a
        # raw SAM id that differs from the public id.  This is the only
        # unambiguous ALREADY_BOUND case; multi-object states are not guessed.
        if len(existing_ids) == 1:
            backend._bind_external_sam_id(int(public_id), int(existing_ids[0]))
        return {
            "status": "ALREADY_BOUND",
            "tracker_state_count": len(existing),
            "object_ids": existing_ids,
            "mapping_restored": len(existing_ids) == 1,
            "external_id": int(public_id) if len(existing_ids) == 1 else None,
            "sam_id": int(existing_ids[0]) if len(existing_ids) == 1 else None,
        }

    rectangle = _box_rectangle_mask(
        box_xyxy,
        int(state["orig_height"]),
        int(state["orig_width"]),
    ).unsqueeze(0)
    tracker_states = model._tracker_add_new_objects(
        frame_idx=int(frame_idx),
        num_frames=int(state["num_frames"]),
        new_obj_ids=[int(public_id)],
        new_obj_masks=rectangle,
        tracker_states_local=[],
        orig_vid_height=int(state["orig_height"]),
        orig_vid_width=int(state["orig_width"]),
        feature_cache=state["feature_cache"],
    )
    if len(tracker_states) != 1 or int(public_id) not in [
        int(value)
        for value in np.asarray(tracker_states[0].get("obj_ids", [])).reshape(-1)
    ]:
        raise RuntimeError("official singleton binding did not create the requested object")
    state["sam2_inference_states"] = tracker_states
    raw_ids = [
        int(value)
        for value in np.asarray(tracker_states[0].get("obj_ids", [])).reshape(-1)
    ]
    if len(raw_ids) != 1:
        raise RuntimeError("official singleton binding produced a non-singleton raw namespace")
    backend._bind_external_sam_id(int(public_id), raw_ids[0])

    # Mirror the official planner metadata for one object on rank 0.  This is
    # the same namespace used by the global Hungarian/demux path.
    metadata = model._initialize_metadata()
    rank = int(getattr(model, "rank", 0))
    metadata["obj_ids_per_gpu"][rank] = np.asarray([public_id], dtype=np.int64)
    metadata["num_obj_per_gpu"][rank] = 1
    metadata["obj_ids_all_gpu"] = np.asarray([public_id], dtype=np.int64)
    metadata["max_obj_id"] = int(public_id)
    metadata["obj_id_to_score"][public_id] = 1.0
    metadata["obj_id_to_sam2_score_frame_wise"][int(frame_idx)][public_id] = torch.tensor(
        1.0,
        dtype=torch.float32,
        device=state["device"],
    )
    rank0 = metadata["rank0_metadata"]
    rank0["obj_first_frame_idx"][public_id] = int(frame_idx)
    rank0["trk_keep_alive"][public_id] = int(model.init_trk_keep_alive)
    device = state["device"]
    metadata["gpu_metadata"] = {
        "N_obj": 1,
        "obj_first_frame": torch.tensor([frame_idx], dtype=torch.long, device=device),
        "consecutive_unmatch_count": torch.zeros(1, dtype=torch.long, device=device),
        "trk_keep_alive": torch.full(
            (1,), int(model.init_trk_keep_alive), dtype=torch.long, device=device
        ),
        "removed_mask": torch.zeros(1, dtype=torch.bool, device=device),
        "overlap_pair_counts": torch.zeros((1, 1), dtype=torch.long, device=device),
        "last_occluded_tensor": torch.full((1,), -1, dtype=torch.long, device=device),
    }
    metadata["num_buc_per_gpu"][rank] = model._count_buckets_in_states(tracker_states)
    state["tracker_metadata"] = metadata
    return {
        "status": "BOUND",
        "tracker_state_count": len(tracker_states),
        "object_ids": raw_ids,
        "mapping_restored": True,
        "external_id": int(public_id),
        "sam_id": raw_ids[0],
        "provenance": BOX_DERIVED_PSEUDO_MASK,
        "mask_source": "explicit_box_rectangle",
    }


def _trial_outputs(
    outputs: Mapping[int, list[Any]],
    gt: Mapping[int, Mapping[int, np.ndarray]],
    *,
    dataset_identity: int,
    public_id: int,
    start: int,
    end: int,
    require_visible: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for frame in range(start, end + 1):
        target = gt.get(frame, {}).get(dataset_identity)
        obs = _select_observation(outputs.get(frame, []), public_id, target)
        target_present = target is not None
        prediction_present = obs is not None
        value = (
            None
            if not target_present
            else 0.0
            if not prediction_present
            else _iou(obs.box_xyxy, target)
        )
        rows.append(
            {
                "frame": frame,
                "dataset_identity": int(dataset_identity),
                "public_id": int(public_id),
                "target_present": target_present,
                "prediction_present": prediction_present,
                "target_box": None if target is None else np.asarray(target).tolist(),
                "predicted_box": None if obs is None else np.asarray(obs.box_xyxy).tolist(),
                "box_iou": value,
                "present": prediction_present,
                "box": None if obs is None else np.asarray(obs.box_xyxy).tolist(),
            }
        )
    visible = [row for row in rows if row["target_present"]]
    visible_ious = [float(row["box_iou"]) for row in visible]
    absent = [row for row in rows if not row["target_present"]]
    if require_visible and absent:
        raise ValueError(
            "preselected visible episode contains missing GT target frames: "
            f"dataset_identity={dataset_identity}, frames={[row['frame'] for row in absent]}"
        )
    mean_iou = float(np.mean(visible_ious)) if visible_ious else None
    return {
        "rows": rows,
        "dataset_identity": int(dataset_identity),
        "public_id": int(public_id),
        "visible_frame_count": len(visible),
        "missing_prediction_on_visible_count": int(
            sum(not row["prediction_present"] for row in visible)
        ),
        "absent_gt_frame_count": len(absent),
        "mean_box_iou_visible": mean_iou,
        "success_at_0_5_visible": (
            float(np.mean([iou >= 0.5 for iou in visible_ious]))
            if visible_ious
            else None
        ),
        # Backward-compatible aliases used by the previous N29 evaluator.
        "mean_box_iou": mean_iou,
        "error_count_iou_lt_0_5": int(sum(iou < 0.5 for iou in visible_ious)),
    }


def _bridge_audit(
    capture: DecoderCapture,
    observations: list[Any],
    *,
    frame_idx: int,
    public_id: int,
    adapter_version: int,
) -> dict[str, Any]:
    decoder_candidate = _decoder_candidate_from_capture(
        capture,
        frame_idx=frame_idx,
        public_id=public_id,
        adapter_version=adapter_version,
    )
    original = [_box_candidate(observations[0], frame_idx)] if observations else []
    if decoder_candidate is None:
        return {"status": "NOT_AVAILABLE", "candidate_count": len(original)}
    candidates = [*original, decoder_candidate]
    anchor = np.asarray(
        [[
            original[0].presence if original else 0.0,
            decoder_candidate.presence * max(decoder_candidate.iou_pred, 0.0),
        ]],
        dtype=np.float64,
    )
    bridge = build_decoder_assignment(
        anchor,
        candidates,
        [public_id],
        none_scores=np.asarray([-1.0]),
    )
    return {
        "status": "PASS",
        "candidate_count": len(candidates),
        "matrix_shape": list(bridge.matrix.shape),
        "matrix": bridge.matrix.tolist(),
        "assignment": bridge.assignment.assignment.tolist(),
        "sources": [candidate.source for candidate in candidates],
        "source_public_ids": [candidate.source_public_id for candidate in candidates],
        "decoder_adapter_version": decoder_candidate.adapter_version,
    }


def run_real_sequence(
    backend: Any,
    adapter: SAM3DecoderLITAdapter,
    decoder: nn.Module,
    capture: DecoderCapture,
    sequence_dir: Path,
    *,
    sequence_index: int,
    max_frames: int,
    correction_frame: Optional[int],
    variant: str,
    inner_steps: int,
) -> dict[str, Any]:
    if "val" in sequence_dir.parts or "test" in sequence_dir.parts:
        raise ValueError(f"N29 real replay is train-only; refused {sequence_dir}")
    gt = _read_gt(sequence_dir)
    images = _image_files(sequence_dir)
    if not images:
        raise ValueError(f"sequence has no images: {sequence_dir}")
    prompt_frame, gt_identity = _first_gt_identity(gt)
    end = min(len(images) - 1, max_frames - 1)
    if prompt_frame > end:
        raise ValueError("prompt frame is outside max_frames")
    available_corr = [frame for frame in sorted(gt) if prompt_frame < frame <= end and gt[frame].get(gt_identity) is not None]
    if correction_frame is None:
        corr = available_corr[0] if available_corr else min(end, prompt_frame + 1)
    else:
        corr = int(correction_frame)
    if corr <= prompt_frame or corr > end:
        raise ValueError(f"correction frame {corr} is outside [{prompt_frame + 1}, {end}]")
    public_id = 100000 + sequence_index
    binding = ReplayIdentityBinding(
        dataset_identity=gt_identity,
        public_id=public_id,
        sam_object_id=public_id,
    )
    prompt_box = gt[prompt_frame][gt_identity]
    human_box = gt.get(corr, {}).get(gt_identity)
    if human_box is None:
        raise ValueError("correction frame has no legal GT box for the selected identity")

    # Frozen anchor trial.
    _session(backend, sequence_dir)
    backend.add_box(prompt_frame, public_id, prompt_box)
    anchor_binding = _install_official_box_singleton(
        backend,
        frame_idx=prompt_frame,
        public_id=public_id,
        box_xyxy=prompt_box,
    )
    capture.reset()
    anchor_outputs = backend.propagate(prompt_frame, end, start_frame_index=prompt_frame)
    anchor_eval = _trial_outputs(
        anchor_outputs,
        gt,
        dataset_identity=binding.dataset_identity,
        public_id=binding.public_id,
        start=corr + 1,
        end=end,
        require_visible=True,
    )

    state: Optional[IdentityAdapterState] = None
    update_result: Optional[Any] = None
    adapted_outputs: dict[int, list[Any]] = {}
    bridge = {"status": "NOT_AVAILABLE"}
    if variant != "update_disabled":
        state = adapter.new_state(str(sequence_dir), public_id, device=adapter.device)
        if variant == "b_only":
            for parameter in state.lora_a.values():
                parameter.requires_grad = False
        # Start a fresh official state and replay only through the correction.
        _session(backend, sequence_dir)
        backend.add_box(prompt_frame, public_id, prompt_box)
        adapted_binding = _install_official_box_singleton(
            backend,
            frame_idx=prompt_frame,
            public_id=public_id,
            box_xyxy=prompt_box,
        )
        capture.reset(target_call=max(0, corr - prompt_frame))
        pre_outputs = backend.propagate(prompt_frame, corr, start_frame_index=prompt_frame)
        support_kwargs = None if capture.target_inputs is None else _clone_tree(capture.target_inputs)
        event = DecoderCorrectionEvent(
            video_id=str(sequence_dir),
            public_id=public_id,
            frame_idx=corr,
            provenance=BOX_DERIVED_PSEUDO_MASK,
            box_xyxy=human_box,
            image_size=(int(backend._frame_h), int(backend._frame_w)),
            current_output_recorded=True,
            metadata={"teacher": "explicit_box_rectangle_pseudo_target", "click_count": "0"},
        )
        if support_kwargs is None:
            update_result = {
                "status": "NOT_RUN",
                "reason": "official propagation decoder hook did not expose support inputs",
            }
        else:
            transaction = DecoderUpdateTransaction(
                adapter,
                DecoderUpdateConfig(
                    inner_steps=inner_steps,
                    learning_rate=1.0e-4,
                    weight_decay=0.0,
                    require_loss_decrease=False,
                    require_observable_update=True,
                ),
            )
            forward_tensor_diagnostic: dict[str, Any] = {}
            first_state_parameter = next(iter(state.lora_a.values()))
            forward_tensor_diagnostic.update(
                {
                    "state_parameter_is_inference": bool(
                        torch.is_inference(first_state_parameter)
                    ),
                    "state_parameter_requires_grad": bool(
                        first_state_parameter.requires_grad
                    ),
                    "state_parameter_creation_grad_enabled": bool(
                        torch.is_grad_enabled()
                    ),
                    "state_parameter_creation_inference_mode_enabled": bool(
                        torch.is_inference_mode_enabled()
                    ),
                }
            )

            def forward_fn(_supervision, _step):
                # The official request handler runs under inference mode.  A
                # cloned support tensor is necessary but not sufficient if a
                # caller later wraps this update in the handler's context, so
                # explicitly restore ordinary autograd semantics here.
                if not forward_tensor_diagnostic:
                    forward_tensor_diagnostic.update(
                        {
                            "support_inputs": _tensor_status_tree(support_kwargs),
                            "decoder_inference_parameters": [
                                name
                                for name, parameter in decoder.named_parameters()
                                if torch.is_inference(parameter)
                            ],
                            "decoder_inference_buffers": [
                                name
                                for name, buffer in decoder.named_buffers()
                                if torch.is_inference(buffer)
                            ],
                        }
                    )
                with torch.inference_mode(False), torch.enable_grad():
                    raw = decoder(**support_kwargs)
                selected = _slot_tensor(raw["masks"], slot=0)
                forward_tensor_diagnostic.update(
                    {
                        "autograd_grad_enabled_after_forward": bool(
                            torch.is_grad_enabled()
                        ),
                        "inference_mode_enabled_after_forward": bool(
                            torch.is_inference_mode_enabled()
                        ),
                        "raw_masks_requires_grad": bool(
                            isinstance(raw.get("masks"), torch.Tensor)
                            and raw["masks"].requires_grad
                        ),
                        "raw_masks_grad_fn": (
                            None
                            if not isinstance(raw.get("masks"), torch.Tensor)
                            or raw["masks"].grad_fn is None
                            else type(raw["masks"].grad_fn).__name__
                        ),
                        "selected_logits_requires_grad": bool(
                            isinstance(selected, torch.Tensor)
                            and selected.requires_grad
                        ),
                        "selected_logits_grad_fn": (
                            None
                            if not isinstance(selected, torch.Tensor)
                            or selected.grad_fn is None
                            else type(selected.grad_fn).__name__
                        ),
                        "active_lora_wrapper_count": int(
                            sum(
                                wrapper._active_a is not None
                                and wrapper._active_b is not None
                                for wrapper in adapter.wrappers.values()
                            )
                        ),
                        "decoder_forward_type": type(decoder.forward).__name__,
                        "decoder_forward_repr": repr(decoder.forward)[:500],
                        "transformer_forward_type": type(
                            decoder.transformer.forward
                        ).__name__,
                        "first_wrapper_forward_diagnostic": getattr(
                            next(iter(adapter.wrappers.values())),
                            "_debug_last_forward",
                            None,
                        ),
                    }
                )
                return selected

            def deterministic_support_forward(_supervision):
                was_training = decoder.training
                decoder.eval()
                try:
                    with torch.inference_mode(False), torch.no_grad():
                        raw = decoder(**support_kwargs)
                    return _slot_tensor(raw["masks"], slot=0)
                finally:
                    decoder.train(was_training)

            update_result = transaction.apply(
                event,
                state,
                forward_fn=forward_fn,
                deterministic_forward_fn=deterministic_support_forward,
            )
            if isinstance(update_result, dict):
                update_result["forward_tensor_diagnostic"] = forward_tensor_diagnostic
            else:
                update_result = {
                    "status": update_result.status,
                    "committed": update_result.committed,
                    "adapter_version": update_result.adapter_version,
                    "loss_history": list(update_result.loss_history),
                    "gradient_parameter_count": update_result.gradient_parameter_count,
                    "rollback_reason": update_result.rollback_reason,
                    "exception_traceback": update_result.exception_traceback,
                    "optimization_diagnostic": dict(update_result.optimization_diagnostic),
                    "forward_tensor_diagnostic": forward_tensor_diagnostic,
                }
        update_committed = (
            bool(update_result.get("committed", False))
            if isinstance(update_result, dict)
            else bool(getattr(update_result, "committed", False))
        )
        if state is not None and update_committed:
            with adapter.activate(state):
                adapted_outputs = backend.propagate(
                    corr + 1,
                    end,
                    start_frame_index=corr + 1,
                )
        else:
            adapted_outputs = backend.propagate(
                corr + 1,
                end,
                start_frame_index=corr + 1,
            )
        # The current correction-frame output comes only from pre_outputs; it
        # is intentionally not replaced by a post-correction rerun.
        current_unchanged = corr in pre_outputs
    else:
        current_unchanged = True
        _session(backend, sequence_dir)
        backend.add_box(prompt_frame, public_id, prompt_box)
        capture.reset()
        adapted_outputs = backend.propagate(prompt_frame, end, start_frame_index=prompt_frame)

    adapted_eval = _trial_outputs(
        adapted_outputs,
        gt,
        dataset_identity=binding.dataset_identity,
        public_id=binding.public_id,
        start=corr + 1,
        end=end,
        require_visible=True,
    )
    last_frame = max(adapted_outputs) if adapted_outputs else end
    bridge_obs = adapted_outputs.get(last_frame, []) if adapted_outputs else []
    bridge = _bridge_audit(
        capture,
        bridge_obs,
        frame_idx=last_frame,
        public_id=public_id,
        adapter_version=0 if state is None else state.adapter_version,
    )
    anchor_errors = [row["box_iou"] < 0.5 for row in anchor_eval["rows"]]
    adapted_errors = [row["box_iou"] < 0.5 for row in adapted_eval["rows"]]
    update_status = (
        update_result.get("status")
        if isinstance(update_result, dict)
        else (None if update_result is None else update_result.status)
    )
    bridge_status = bridge.get("status")
    sequence_status = (
        "PASS"
        if update_status == "COMMIT" and bridge_status == "PASS"
        else "NOT_RUN"
    )
    return {
        "status": sequence_status,
        "sequence": sequence_dir.name,
        "split": "train",
        "identity": gt_identity,
        "public_id": public_id,
        "sam_object_id": binding.sam_object_id,
        "identity_binding": asdict(binding),
        "prompt_frame": prompt_frame,
        "correction_frame": corr,
        "end_frame": end,
        "variant": variant,
        "official_singleton_binding": {
            "anchor": anchor_binding,
            "adapted": adapted_binding if variant != "update_disabled" else None,
        },
        "supervision_provenance": BOX_DERIVED_PSEUDO_MASK,
        "box_actions": 1,
        "click_count": 0,
        "mask_corrections": 0,
        "current_frame_rewritten": not current_unchanged,
        "anchor_future": anchor_eval,
        "adapted_future": adapted_eval,
        "future_error_delta": float(np.mean(adapted_errors) - np.mean(anchor_errors)) if anchor_errors else 0.0,
        "update": (
            update_result
            if isinstance(update_result, dict)
            else {
                "status": update_result.status,
                "committed": update_result.committed,
                "adapter_version": update_result.adapter_version,
                "loss_history": list(update_result.loss_history),
                "gradient_parameter_count": update_result.gradient_parameter_count,
                "rollback_reason": update_result.rollback_reason,
            }
        ),
        "candidate_bridge": bridge,
        "official_path_diagnostic": _short_state_diagnostic(backend, capture, decoder),
    }


def run_fixture(*, seeds: int = 3, rank: int = 4) -> dict[str, Any]:
    """Run a multi-seed official-shaped causal replay without dataset claims."""

    from sam3.sam.transformer import TwoWayTransformer

    rows = []
    for seed in range(2901, 2901 + seeds):
        torch.manual_seed(seed)
        decoder = nn.Module()
        decoder.transformer = TwoWayTransformer(depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8)
        decoder.head = nn.Linear(256, 64)
        for parameter in decoder.head.parameters():
            parameter.requires_grad = False
        adapter = SAM3DecoderLITAdapter(decoder, DecoderLITConfig(rank=rank, alpha=float(rank), dropout=0.0))
        state = adapter.new_state("fixture", seed, device="cpu")
        image = torch.randn(1, 256, 2, 2)
        pe = torch.randn(1, 256, 2, 2)
        points = torch.randn(1, 1, 256)
        future = torch.randn(1, 256, 2, 2)
        future_pe = torch.randn(1, 256, 2, 2)

        def forward(_supervision, _step):
            token, _ = decoder.transformer(image, pe, points)
            return decoder.head(token[:, 0]).view(1, 1, 8, 8)

        event = DecoderCorrectionEvent(
            video_id="fixture",
            public_id=seed,
            frame_idx=0,
            provenance=BOX_DERIVED_PSEUDO_MASK,
            box_xyxy=(1.0, 1.0, 7.0, 7.0),
            image_size=(8, 8),
        )
        update = DecoderUpdateTransaction(
            adapter,
            DecoderUpdateConfig(inner_steps=5, learning_rate=0.01),
        ).apply(event, state, forward_fn=forward)
        with torch.no_grad():
            base_token, _ = decoder.transformer(future, future_pe, points)
        with adapter.activate(state), torch.no_grad():
            adapted_token, _ = decoder.transformer(future, future_pe, points)
        rows.append(
            {
                "seed": seed,
                "committed": update.committed,
                "future_token_delta": float((adapted_token - base_token).abs().max()),
                "parameter_count": state.parameter_count(),
            }
        )
    return {
        "status": "PASS",
        "mode": "official_shaped_fixture",
        "real_dataset_metrics_claimed": False,
        "seeds": rows,
        "rank": rank,
        "rank_parameter_count": rows[0]["parameter_count"] if rows else 0,
    }


def _default_sequences(manifest_path: Path, root: Path, limit: int) -> list[Path]:
    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text())
        names = [
            entry["video"]
            for entry in data.get("entries", [])
            if entry.get("dataset") == "DanceTrack" and entry.get("role") == "train_fold"
        ]
        names = list(dict.fromkeys(names))
    else:
        names = sorted(path.name for path in (root / "train").iterdir() if path.is_dir())
    return [root / "train" / name for name in names[:limit]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fixture", "real"), default="fixture")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
    parser.add_argument("--dataset-root", type=Path, default=Path("/path/to/dancetrack"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/n27/dataset_split_manifest.json")
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--max-sequences", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--correction-frame", type=int, default=None)
    parser.add_argument("--variant", choices=("formal_ab", "b_only", "update_disabled"), default="formal_ab")
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "n29")
    args = parser.parse_args()

    output_dir = args.output_dir
    result: dict[str, Any] = {
        "protocol": "N29-B",
        "mode": args.mode,
        "variant": args.variant,
        "val25_read": False,
        "checkpoint": str(args.checkpoint),
    }
    if args.mode == "fixture":
        result["fixture"] = run_fixture(seeds=args.seed_count, rank=4)
        _write_json(output_dir / "n29b_result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
        return 0

    sequences = (
        [args.dataset_root / "train" / name for name in args.sequence]
        if args.sequence
        else _default_sequences(args.manifest, args.dataset_root, args.max_sequences)
    )
    sequences = sequences[: args.max_sequences]
    if not args.checkpoint.is_file():
        result.update({"status": "NOT_RUN", "reason": "checkpoint_unavailable", "sequences": []})
        _write_json(output_dir / "n29b_result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
        return 0

    backend = None
    capture = None
    try:
        backend = _make_backend(args.checkpoint)
        # Start the first session so the official builder/model is available.
        _session(backend, sequences[0])
        decoder = _get_official_decoder(backend)
        adapter = SAM3DecoderLITAdapter(
            decoder,
            DecoderLITConfig(rank=4, alpha=4.0, dropout=0.1),
        )
        materialized_decoder_tensors = _materialize_normal_module_tensors(decoder)
        capture = DecoderCapture(decoder)
        sequence_results = []
        for index, sequence in enumerate(sequences):
            try:
                sequence_results.append(
                    run_real_sequence(
                        backend,
                        adapter,
                        decoder,
                        capture,
                        sequence,
                        sequence_index=index,
                        max_frames=args.max_frames,
                        correction_frame=args.correction_frame,
                        variant=args.variant,
                        inner_steps=args.inner_steps,
                    )
                )
            except Exception as exc:
                sequence_results.append(
                    {
                        "status": "NOT_RUN",
                        "sequence": sequence.name,
                        "reason": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=3),
                    }
                )
        adapter_states = adapter.bank.states()
        result.update(
            {
                "status": (
                    "PASS"
                    if sequence_results and all(item.get("status") == "PASS" for item in sequence_results)
                    else "PARTIAL"
                    if any(item.get("status") == "PASS" for item in sequence_results)
                    else "NOT_RUN"
                ),
                "sequence_results": sequence_results,
                "adapter_inventory": (
                    adapter.inventory(adapter_states[0])
                    if adapter_states
                    else adapter.inventory()
                ),
                "adapter_state_count": len(adapter_states),
                "materialized_decoder_tensors": materialized_decoder_tensors,
                "sequences": [str(path) for path in sequences],
                "data_split": "DanceTrack/train only; sequence-disjoint train_fold manifest",
                "official_path_diagnostic": _short_state_diagnostic(backend, capture, decoder),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "NOT_RUN",
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
                "sequence_results": [],
            }
        )
    finally:
        if capture is not None:
            capture.close()
        if backend is not None:
            backend.close()

    _write_json(output_dir / "n29b_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
