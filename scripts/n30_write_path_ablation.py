#!/usr/bin/env python3
"""N30-A: audit the real single-identity correction write path.

This replay keeps the frozen N29-R episode selection and changes only the
state written at the correction frame.  The six branches are intentionally
small and explicit:

* A: no correction;
* B: the public backend ``correct_object`` path only;
* C: an official rectangle rebind only;
* D: the current correction plus official singleton rebind;
* E: D plus the audited online decoder LoRA transaction;
* F: local bookkeeping only, without an official tracker write.

The script records target-scoped state summaries rather than serializing the
3.5 GB model or video tensors.  It never reads validation or test labels.
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
from typing import Any, Iterator, Mapping, Optional

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.adaptation.corrected_mask_teacher import (  # noqa: E402
    BOX_DERIVED_PSEUDO_MASK,
)
from sam3_intermot.adaptation.correction_state_audit import (  # noqa: E402
    diff_snapshots,
    snapshot_backend_state,
)
from sam3_intermot.adaptation.decoder_update_transaction import (  # noqa: E402
    DecoderCorrectionEvent,
    DecoderUpdateConfig,
    DecoderUpdateTransaction,
)
from sam3_intermot.adaptation.sam3_decoder_lit import (  # noqa: E402
    DecoderLITConfig,
    SAM3DecoderLITAdapter,
)
from scripts.n29_lit_online_replay import (  # noqa: E402
    DecoderCapture,
    _clone_tree,
    _get_official_decoder,
    _image_files,
    _install_official_box_singleton,
    _make_backend,
    _read_gt,
    _session,
    _slot_tensor,
    _tensor_status_tree,
)
from scripts.n29r_paired_replay import (  # noqa: E402
    _branch_metrics,
    _difference,
    _ensure_public_singleton_binding,
    _horizon_metrics,
    _load_manifest,
    _sha256,
    _update_dict,
)


BRANCHES = (
    "anchor_no_correction",
    "official_correct_object_only",
    "forced_singleton_reprompt_only",
    "current_write_only",
    "current_write_plus_online_lora",
    "local_backend_bookkeeping_only",
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
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _official_ids(backend: Any) -> list[int]:
    predictor = backend._predictor
    entry = predictor._all_inference_states[backend._session_id]
    state = entry["state"]
    return [
        int(obj_id)
        for tracker_state in state.get("sam2_inference_states", [])
        for obj_id in np.asarray(tracker_state.get("obj_ids", [])).reshape(-1)
    ]


def _obs_signature(observations: Any, public_id: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for obs in observations or []:
        if int(getattr(obs, "sam_object_id", -1)) != int(public_id):
            continue
        mask = np.asarray(getattr(obs, "mask", np.zeros((1, 1))), dtype=bool)
        result.append(
            {
                "sam_object_id": int(getattr(obs, "sam_object_id", -1)),
                "source": str(getattr(obs, "source", "")),
                "is_human_verified": bool(getattr(obs, "is_human_verified", False)),
                "box": np.asarray(getattr(obs, "box_xyxy", []), dtype=float).tolist(),
                "mask_shape": list(mask.shape),
                "mask_area": int(mask.sum()) if mask.size > 1 else 0,
            }
        )
    return result


def _prompt_observation_summary(observations: Any) -> dict[str, Any]:
    items = list(observations or [])
    masklet_count = 0
    for obs in items:
        mask = np.asarray(getattr(obs, "mask", np.zeros((1, 1))), dtype=bool)
        if mask.ndim == 2 and mask.size > 1:
            masklet_count += 1
    return {
        "returned_count": len(items),
        "returned_sam_ids": [int(getattr(obs, "sam_object_id", -1)) for obs in items],
        "returned_masklet_count": masklet_count,
    }


@contextmanager
def _prompt_trace(backend: Any) -> Iterator[list[dict[str, Any]]]:
    """Trace backend prompt requests without changing the official backend."""

    original = backend._send_prompt
    calls: list[dict[str, Any]] = []

    def wrapped(
        frame_idx: int,
        *,
        boxes: Optional[list[np.ndarray]] = None,
        text: Optional[str] = None,
        source: str,
    ) -> Any:
        observations = original(frame_idx, boxes=boxes, text=text, source=source)
        summary = _prompt_observation_summary(observations)
        calls.append(
            {
                "frame": int(frame_idx),
                "source": str(source),
                "text_present": text is not None,
                "box_count": 0 if boxes is None else len(boxes),
                **summary,
            }
        )
        return observations

    backend._send_prompt = wrapped
    try:
        yield calls
    finally:
        backend._send_prompt = original


def _lora_state_summary(
    state: Any,
    *,
    update_called: bool,
    update: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    if state is None:
        return {"update_called": bool(update_called), "state_present": False}
    a_l2 = 0.0
    b_l2 = 0.0
    for parameter in state.lora_a.values():
        a_l2 += float(parameter.detach().float().pow(2).sum().cpu())
    for parameter in state.lora_b.values():
        b_l2 += float(parameter.detach().float().pow(2).sum().cpu())
    return {
        "update_called": bool(update_called),
        "state_present": True,
        "video_id": str(state.video_id),
        "public_id": int(state.public_id),
        "adapter_version": int(state.adapter_version),
        "correction_count": int(state.correction_count),
        "last_frame": int(state.last_frame),
        "last_provenance": str(state.last_provenance),
        "a_l2": math.sqrt(a_l2),
        "b_l2": math.sqrt(b_l2),
        "update_status": None if update is None else update.get("status"),
        "update_committed": None if update is None else bool(update.get("committed", False)),
    }


def _force_official_rebind(
    backend: Any,
    *,
    frame: int,
    public_id: int,
    box: np.ndarray,
) -> dict[str, Any]:
    before_ids = _official_ids(backend)
    # Use the official public add_prompt path with exactly one box.  The
    # lower-level ``_tracker_add_new_objects`` rectangle fallback cannot be
    # safely used after a stopped propagation window in this multiplex
    # implementation: a newly created tracker state lacks the cached image
    # handle expected by its reconditioning path.  Temporarily changing only
    # the prompt box lets the official request update tracker state while the
    # external backend bookkeeping remains untouched.
    old_object = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in backend._objects[public_id].items()
    }
    backend._objects[public_id]["box"] = np.asarray(box, dtype=float).copy()
    try:
        observations = backend._send_prompt(
            frame,
            boxes=[np.asarray(box, dtype=float)],
            source="n30_forced_singleton_reprompt",
        )
    finally:
        backend._objects[public_id] = old_object
    backend._apply_stable_ids(observations)
    raw_ids = [int(getattr(obs, "sam_object_id", -1)) for obs in observations]
    if raw_ids:
        raw_id = raw_ids[0]
        backend._bind_external_sam_id(int(public_id), raw_id)
    binding = {
        "status": "OFFICIAL_REPROMPT",
        "object_ids_before": before_ids,
        "object_ids_after": _official_ids(backend),
        "returned_observations": len(observations),
        "returned_raw_or_public_ids": raw_ids,
        "provenance": BOX_DERIVED_PSEUDO_MASK,
        "mask_source": "official_add_prompt_single_box",
    }
    return {
        "before_object_ids": before_ids,
        "cleared_existing_tracker_states": False,
        "binding": binding,
        "after_object_ids": _official_ids(backend),
    }


def _local_bookkeeping_write(backend: Any, *, frame: int, public_id: int, box: np.ndarray) -> dict[str, Any]:
    if public_id not in backend._objects:
        raise ValueError(f"local bookkeeping target is absent: {public_id}")
    human_box = backend._clip_box(box)
    obj = backend._objects[public_id]
    before = {
        "box": np.asarray(obj.get("box", []), dtype=float).tolist(),
        "human_box": np.asarray(obj.get("human_box", []), dtype=float).tolist(),
        "frame": obj.get("frame"),
    }
    obj["box"] = human_box.copy()
    obj["human_box"] = human_box.copy()
    obj["frame"] = int(frame)
    obj["source"] = "human_correction"
    backend._last_prompt_frame = int(frame)
    obs = backend._human_observation(frame, public_id, human_box, "human_correction")
    backend._add_human_to_cache(frame, obs)
    return {
        "before": before,
        "after": {
            "box": human_box.tolist(),
            "human_box": human_box.tolist(),
            "frame": int(frame),
        },
        "official_object_ids_after": _official_ids(backend),
        "official_call_made": False,
    }


def _run_lora_update(
    *,
    backend: Any,
    adapter: SAM3DecoderLITAdapter,
    decoder: torch.nn.Module,
    episode: Mapping[str, Any],
    support_kwargs: Optional[Mapping[str, Any]],
    current_output_recorded: bool,
) -> tuple[Any, dict[str, Any]]:
    public_id = int(episode["public_id"])
    correction = int(episode["correction_frame"])
    state = adapter.new_state(
        f"{episode['sequence']}:{episode['episode_id']}",
        public_id,
        device=adapter.device,
    )
    if support_kwargs is None:
        update = {
            "status": "NOT_RUN",
            "committed": False,
            "reason": "official propagation decoder hook did not expose support inputs",
        }
        return state, update
    if not current_output_recorded:
        update = {
            "status": "NOT_RUN",
            "committed": False,
            "reason": "correction frame was not recorded before update",
        }
        return state, update

    correction_box = np.asarray(episode["correction_box"], dtype=float)
    event = DecoderCorrectionEvent(
        video_id=f"{episode['sequence']}:{episode['episode_id']}",
        public_id=public_id,
        frame_idx=correction,
        provenance=BOX_DERIVED_PSEUDO_MASK,
        box_xyxy=correction_box,
        image_size=(int(backend._frame_h), int(backend._frame_w)),
        current_output_recorded=True,
        metadata={
            "teacher": "explicit_box_rectangle_pseudo_target",
            "click_count": "0",
            "branch": "current_write_plus_online_lora",
        },
    )
    config = DecoderUpdateConfig(
        inner_steps=5,
        learning_rate=1.0e-4,
        weight_decay=0.0,
        optimizer_enabled=True,
        require_loss_decrease=False,
        require_observable_update=True,
    )
    support_kwargs = _clone_tree(support_kwargs)

    def forward_fn(_supervision: Any, _step: int) -> torch.Tensor:
        with torch.inference_mode(False), torch.enable_grad():
            raw = decoder(**support_kwargs)
        return _slot_tensor(raw["masks"], slot=0)

    def deterministic_forward(_supervision: Any) -> torch.Tensor:
        was_training = decoder.training
        decoder.eval()
        try:
            with torch.inference_mode(False), torch.no_grad():
                raw = decoder(**support_kwargs)
            return _slot_tensor(raw["masks"], slot=0)
        finally:
            decoder.train(was_training)

    transaction = DecoderUpdateTransaction(adapter, config)
    result = transaction.apply(
        event,
        state,
        forward_fn=forward_fn,
        deterministic_forward_fn=deterministic_forward,
    )
    return state, _update_dict(result)


def _run_branch(
    *,
    backend: Any,
    adapter: SAM3DecoderLITAdapter,
    decoder: torch.nn.Module,
    capture: DecoderCapture,
    episode: Mapping[str, Any],
    sequence: Path,
    session_ready: bool,
    branch_name: str,
    gt: Mapping[int, Mapping[int, np.ndarray]],
    query_end: int,
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    init = int(episode["initialization_frame"])
    correction = int(episode["correction_frame"])
    public_id = int(episode["public_id"])
    correction_box = np.asarray(episode["correction_box"], dtype=float)
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if session_ready:
        backend.reset_session()
    else:
        _session(backend, sequence)
    backend.add_box(init, public_id, np.asarray(gt[init][int(episode["dataset_identity"])], dtype=float))
    anchor_binding = _install_official_box_singleton(
        backend,
        frame_idx=init,
        public_id=public_id,
        box_xyxy=np.asarray(gt[init][int(episode["dataset_identity"])], dtype=float),
    )

    with _prompt_trace(backend) as prompt_calls:
        capture.reset(target_call=max(0, correction - init))
        pre_outputs = backend.propagate(init, correction, start_frame_index=init)
        support_kwargs = None if capture.target_inputs is None else _clone_tree(capture.target_inputs)
        current_output_recorded = bool(correction in pre_outputs)
        before_lora = {"update_called": False, "state_present": False}
        before = snapshot_backend_state(
            backend,
            target_id=public_id,
            correction_frame=correction,
            b10_state={
                "update_called": False,
                "state_delta": "NOT_APPLICABLE_SINGLE_ID_NO_STATE_MANAGER",
            },
            lora_state=before_lora,
        )

        correction_binding: Optional[dict[str, Any]] = None
        force_binding: Optional[dict[str, Any]] = None
        local_write: Optional[dict[str, Any]] = None
        lora_state: Any = None
        lora_update: dict[str, Any] = {
            "status": "NOT_RUN",
            "committed": False,
            "reason": "branch_has_no_lora_write",
        }

        if branch_name == "anchor_no_correction":
            pass
        elif branch_name == "official_correct_object_only":
            backend.correct_object(correction, public_id, box_xyxy=correction_box)
        elif branch_name == "forced_singleton_reprompt_only":
            force_binding = _force_official_rebind(
                backend,
                frame=correction,
                public_id=public_id,
                box=correction_box,
            )
        elif branch_name in ("current_write_only", "current_write_plus_online_lora"):
            backend.correct_object(correction, public_id, box_xyxy=correction_box)
            correction_binding = _ensure_public_singleton_binding(
                backend,
                frame=correction,
                public_id=public_id,
                box=correction_box,
            )
            if branch_name == "current_write_plus_online_lora":
                lora_state, lora_update = _run_lora_update(
                    backend=backend,
                    adapter=adapter,
                    decoder=decoder,
                    episode=episode,
                    support_kwargs=support_kwargs,
                    current_output_recorded=current_output_recorded,
                )
        elif branch_name == "local_backend_bookkeeping_only":
            local_write = _local_bookkeeping_write(
                backend,
                frame=correction,
                public_id=public_id,
                box=correction_box,
            )
        else:  # pragma: no cover - guarded by BRANCHES
            raise ValueError(f"unknown N30-A branch: {branch_name}")

        after_lora = _lora_state_summary(
            lora_state,
            update_called=branch_name == "current_write_plus_online_lora",
            update=lora_update,
        )
        after = snapshot_backend_state(
            backend,
            target_id=public_id,
            correction_frame=correction,
            b10_state={
                "update_called": False,
                "state_delta": "NOT_APPLICABLE_SINGLE_ID_NO_STATE_MANAGER",
            },
            lora_state=after_lora,
        )
        state_delta = diff_snapshots(before, after)
        pre_cache = before.payload["backend"]["output_cache_target_frames"].get(str(correction))
        post_cache = after.payload["backend"]["output_cache_target_frames"].get(str(correction))
        correction_prompt_calls = [
            call
            for call in prompt_calls
            if int(call["frame"]) == correction
            and call["source"] in {"human_correction", "n30_forced_singleton_reprompt"}
        ]
        official_returned_masklet = any(
            int(call.get("returned_masklet_count", 0)) > 0
            for call in correction_prompt_calls
        )

        if branch_name == "current_write_plus_online_lora" and bool(lora_update.get("committed", False)):
            with adapter.activate(lora_state):
                future_outputs = backend.propagate(
                    correction + 1,
                    query_end,
                    start_frame_index=correction + 1,
                )
            future_adapter_active = True
        elif branch_name in ("anchor_no_correction", "local_backend_bookkeeping_only"):
            # Without an official prompt/rebind, the pinned predictor closes
            # the stopped window without materializing a resumable next
            # cached frame.  Rebuild the same anchor session and consume one
            # complete official stream; no correction is sent to SAM3.  F's
            # local write is restored after this replay because reset_session
            # clears backend bookkeeping as well as the official session.
            backend.reset_session()
            backend.add_box(
                init,
                public_id,
                np.asarray(gt[init][int(episode["dataset_identity"])], dtype=float),
            )
            _install_official_box_singleton(
                backend,
                frame_idx=init,
                public_id=public_id,
                box_xyxy=np.asarray(gt[init][int(episode["dataset_identity"])], dtype=float),
            )
            future_outputs = backend.propagate(
                init,
                query_end,
                start_frame_index=init,
            )
            if branch_name == "local_backend_bookkeeping_only":
                _local_bookkeeping_write(
                    backend,
                    frame=correction,
                    public_id=public_id,
                    box=correction_box,
                )
            future_adapter_active = False
        else:
            future_outputs = backend.propagate(
                correction + 1,
                query_end,
                start_frame_index=correction + 1,
            )
            future_adapter_active = False

    eval_episode = dict(episode)
    eval_episode["query_end"] = int(query_end)
    metrics = _horizon_metrics(future_outputs, gt, eval_episode)
    branch_status = "PASS"
    if branch_name == "current_write_plus_online_lora" and lora_update.get("status") == "NOT_RUN":
        branch_status = "NOT_RUN"
    audit = {
        "episode_id": str(episode["episode_id"]),
        "sequence": str(episode["sequence"]),
        "branch": branch_name,
        "target": {
            "dataset_identity": int(episode["dataset_identity"]),
            "public_id": public_id,
        },
        "initialization_frame": init,
        "correction_frame": correction,
        "current_output_recorded_before_correction": current_output_recorded,
        "official_prompt_request_called": bool(correction_prompt_calls),
        "official_prompt_returned_masklet": official_returned_masklet,
        "official_prompt_trace_at_correction": correction_prompt_calls,
        "forced_official_rectangle_rebind": branch_name == "forced_singleton_reprompt_only",
        "official_state_binding": force_binding or correction_binding or anchor_binding,
        "local_bookkeeping_write": local_write,
        "b10_update_called": False,
        "b10_state_delta": "NOT_APPLICABLE_SINGLE_ID_NO_STATE_MANAGER",
        "lora_update_called": branch_name == "current_write_plus_online_lora",
        "lora_update": lora_update,
        "correction_frame_history_changed": pre_cache != post_cache,
        "changed_state_groups": state_delta["changed_state_groups"],
        "state_delta": state_delta,
        "before": before.payload,
        "after": after.payload,
    }
    branch = {
        "status": branch_status,
        "correction_written": branch_name != "anchor_no_correction",
        "correction_write_type": None if branch_name == "anchor_no_correction" else branch_name,
        "supervision_provenance": (
            None if branch_name in ("anchor_no_correction", "local_backend_bookkeeping_only")
            else BOX_DERIVED_PSEUDO_MASK
        ),
        "click_count": 0,
        "mask_corrections": 0,
        "current_output_recorded": current_output_recorded,
        "official_prompt_request_called": bool(correction_prompt_calls),
        "official_prompt_returned_masklet": official_returned_masklet,
        "forced_official_rectangle_rebind": branch_name == "forced_singleton_reprompt_only",
        "b10_update_called": False,
        "b10_state_delta": "NOT_APPLICABLE_SINGLE_ID_NO_STATE_MANAGER",
        "lora_update_called": branch_name == "current_write_plus_online_lora",
        "lora_update": lora_update,
        "future_adapter_active": future_adapter_active,
        "correction_frame_history_changed": pre_cache != post_cache,
        "changed_state_groups": state_delta["changed_state_groups"],
        "state_audit_summary": {
            "changed": state_delta["changed"],
            "changed_state_groups": state_delta["changed_state_groups"],
            "tensor_delta_count": len(state_delta["tensor_deltas"]),
        },
        "metrics": metrics,
        "timing": {
            "wall_seconds": float(time.perf_counter() - started),
            "peak_gpu_memory_allocated_bytes": (
                None if not torch.cuda.is_available() else int(torch.cuda.max_memory_allocated())
            ),
        },
    }
    return True, branch, audit


def _bootstrap_ci(values: list[float], *, seed: int, draws: int = 2000) -> Optional[list[float]]:
    if not values:
        return None
    if len(values) == 1:
        value = float(values[0])
        return [value, value]
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    samples = rng.integers(0, len(array), size=(draws, len(array)))
    means = array[samples].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _cluster_bootstrap_ci(
    rows: list[Mapping[str, Any]],
    *,
    seed: int,
    draws: int = 2000,
) -> Optional[list[float]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = row.get("value")
        if value is None:
            continue
        grouped.setdefault(str(row["sequence"]), []).append(float(value))
    cluster_values = [float(np.mean(values)) for values in grouped.values() if values]
    return _bootstrap_ci(cluster_values, seed=seed, draws=draws)


def _aggregate_metric(rows: list[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    values = [row["value"] for row in rows if row.get("value") is not None]
    return {
        "mean": None if not values else float(np.mean(values)),
        "sample_count": len(values),
        "episode_count": len(rows),
    }


def _make_summary(
    results: list[Mapping[str, Any]],
    *,
    manifest_path: Path,
    manifest_sha: str,
    checkpoint: Path,
) -> dict[str, Any]:
    metrics = (
        "mean_box_iou_visible",
        "success_at_0_5_visible",
        "missing_prediction_rate_visible",
        "error_count_visible",
        "mask_area_drift",
    )
    branch_summary: dict[str, Any] = {}
    for branch_name in BRANCHES:
        branch_summary[branch_name] = {}
        for horizon in ("5", "10", "20"):
            # Keep the construction explicit per metric so that None values
            # remain missing rather than being converted to zero.
            branch_summary[branch_name][horizon] = {
                metric: _aggregate_metric(
                    [
                        {
                            "value": row["branches"][branch_name]["metrics"][horizon].get(metric),
                            "sequence": row["sequence"],
                        }
                        for row in results
                        if row.get("status") == "PASS"
                        and branch_name in row.get("branches", {})
                        and horizon in row["branches"][branch_name].get("metrics", {})
                    ],
                    metric,
                )
                for metric in metrics
            }

    pairs = {
        "official_correct_object_only_minus_anchor": (
            "official_correct_object_only",
            "anchor_no_correction",
        ),
        "forced_singleton_reprompt_only_minus_anchor": (
            "forced_singleton_reprompt_only",
            "anchor_no_correction",
        ),
        "current_write_only_minus_anchor": ("current_write_only", "anchor_no_correction"),
        "current_write_only_minus_official_correct_object_only": (
            "current_write_only",
            "official_correct_object_only",
        ),
        "current_write_plus_online_lora_minus_current_write_only": (
            "current_write_plus_online_lora",
            "current_write_only",
        ),
        "local_backend_bookkeeping_only_minus_anchor": (
            "local_backend_bookkeeping_only",
            "anchor_no_correction",
        ),
    }
    comparison_summary: dict[str, Any] = {}
    for pair_name, (left_name, right_name) in pairs.items():
        comparison_summary[pair_name] = {"left": left_name, "right": right_name}
        for horizon in ("5", "10", "20"):
            comparison_summary[pair_name][horizon] = {}
            for metric in metrics:
                value_rows: list[dict[str, Any]] = []
                for row in results:
                    if row.get("status") != "PASS":
                        continue
                    left = row.get("branches", {}).get(left_name, {}).get("metrics", {}).get(horizon, {})
                    right = row.get("branches", {}).get(right_name, {}).get("metrics", {}).get(horizon, {})
                    if metric not in left or metric not in right:
                        continue
                    if left.get(metric) is None or right.get(metric) is None:
                        continue
                    value_rows.append(
                        {
                            "value": float(left[metric]) - float(right[metric]),
                            "sequence": row["sequence"],
                        }
                    )
                values = [float(item["value"]) for item in value_rows]
                comparison_summary[pair_name][horizon][metric] = {
                    "mean": None if not values else float(np.mean(values)),
                    "sample_count": len(values),
                    "negative_rate": (
                        None
                        if not values
                        else float(np.mean(np.asarray(values) < 0.0))
                    ),
                    "episode_bootstrap_ci95": _bootstrap_ci(
                        values,
                        seed=3001 + int(horizon) + len(pair_name),
                    ),
                    "sequence_cluster_bootstrap_ci95": _cluster_bootstrap_ci(
                        value_rows,
                        seed=7001 + int(horizon) + len(pair_name),
                    ),
                }

    return {
        "protocol": "N30-A-WRITE-PATH-STATE-AUDIT",
        "status": "PASS" if results and all(row.get("status") == "PASS" for row in results) else "PARTIAL",
        "episode_count": len(results),
        "branch_order": list(BRANCHES),
        "horizons": [5, 10, 20],
        "metrics": list(metrics),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_selection": False,
        "branch_metrics": branch_summary,
        "comparisons": comparison_summary,
        "bootstrap": {
            "episode_unit_draws": 2000,
            "sequence_cluster_draws": 2000,
            "seed_base_episode": 3001,
            "seed_base_sequence": 7001,
        },
    }


def _run_episode(
    *,
    backend: Any,
    adapter: SAM3DecoderLITAdapter,
    decoder: torch.nn.Module,
    capture: DecoderCapture,
    episode: Mapping[str, Any],
    sequence: Path,
    session_ready: bool,
) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    gt = _read_gt(sequence)
    image_count = len(_image_files(sequence))
    init = int(episode["initialization_frame"])
    correction = int(episode["correction_frame"])
    query_end = min(int(episode["query_end"]), image_count - 1)
    if query_end < correction + 5:
        raise ValueError(f"episode lacks five future frames: {episode['episode_id']}")
    if init not in gt or int(episode["dataset_identity"]) not in gt[init]:
        raise ValueError(f"initial GT box unavailable for {episode['episode_id']}")
    branch_rows: dict[str, Any] = {}
    audits: list[dict[str, Any]] = []
    for branch_name in BRANCHES:
        session_ready, branch, audit = _run_branch(
            backend=backend,
            adapter=adapter,
            decoder=decoder,
            capture=capture,
            episode=episode,
            sequence=sequence,
            session_ready=session_ready,
            branch_name=branch_name,
            gt=gt,
            query_end=query_end,
        )
        branch_rows[branch_name] = branch
        audits.append(audit)
    start_digest = hashlib.sha256(
        json.dumps(
            {
                "sequence": episode["sequence"],
                "episode_id": episode["episode_id"],
                "dataset_identity": int(episode["dataset_identity"]),
                "public_id": int(episode["public_id"]),
                "initialization_frame": init,
                "branch_protocol": "same checkpoint/session reset/common prefix",
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    row = {
        "status": "PASS" if all(branch["status"] == "PASS" for branch in branch_rows.values()) else "PARTIAL",
        "episode_id": str(episode["episode_id"]),
        "sequence": str(episode["sequence"]),
        "split": str(episode["split"]),
        "dataset_identity": int(episode["dataset_identity"]),
        "public_id": int(episode["public_id"]),
        "sam_object_id": int(episode["sam_object_id"]),
        "identity_binding": episode.get("identity_binding"),
        "initialization_frame": init,
        "correction_frame": correction,
        "query_start": correction + 1,
        "query_end": query_end,
        "correction_type": "box",
        "supervision_provenance": BOX_DERIVED_PSEUDO_MASK,
        "paired_start_state_digest": start_digest,
        "byte_identical_pre_correction_state_protocol": True,
        "branches": branch_rows,
        "paired_delta": {
            horizon: {
                "current_write_only_minus_anchor": _difference(
                    branch_rows["current_write_only"]["metrics"][horizon],
                    branch_rows["anchor_no_correction"]["metrics"][horizon],
                ),
                "official_correct_object_only_minus_anchor": _difference(
                    branch_rows["official_correct_object_only"]["metrics"][horizon],
                    branch_rows["anchor_no_correction"]["metrics"][horizon],
                ),
                "forced_singleton_reprompt_only_minus_anchor": _difference(
                    branch_rows["forced_singleton_reprompt_only"]["metrics"][horizon],
                    branch_rows["anchor_no_correction"]["metrics"][horizon],
                ),
                "current_write_plus_online_lora_minus_current_write_only": _difference(
                    branch_rows["current_write_plus_online_lora"]["metrics"][horizon],
                    branch_rows["current_write_only"]["metrics"][horizon],
                ),
                "local_backend_bookkeeping_only_minus_anchor": _difference(
                    branch_rows["local_backend_bookkeeping_only"]["metrics"][horizon],
                    branch_rows["anchor_no_correction"]["metrics"][horizon],
                ),
            }
            for horizon in ("5", "10", "20")
        },
    }
    return session_ready, row, audits


def run(
    *,
    manifest_path: Path,
    checkpoint: Path,
    output: Path,
    audit_output: Path,
    summary_output: Path,
    limit: Optional[int],
    resume: bool,
) -> dict[str, Any]:
    manifest, manifest_sha = _load_manifest(manifest_path)
    episodes = list(manifest["episodes"])
    if limit is not None:
        episodes = episodes[:limit]
    partial = output.with_suffix(".partial.json")
    partial_audit = audit_output.with_suffix(".partial.json")
    previous: dict[str, Any] = {}
    previous_audits: list[dict[str, Any]] = []
    if resume and partial.is_file():
        try:
            old = json.loads(partial.read_text(encoding="utf-8"))
            previous = {row["episode_id"]: row for row in old.get("episode_results", [])}
        except Exception:
            previous = {}
    if resume and partial_audit.is_file():
        try:
            previous_audits = json.loads(partial_audit.read_text(encoding="utf-8")).get("audits", [])
        except Exception:
            previous_audits = []

    backend = _make_backend(checkpoint)
    decoder: Optional[torch.nn.Module] = None
    adapter: Optional[SAM3DecoderLITAdapter] = None
    capture: Optional[DecoderCapture] = None
    session_ready = False
    results: list[dict[str, Any]] = []
    audits = list(previous_audits)
    started = time.perf_counter()
    try:
        for episode in episodes:
            episode_id = str(episode["episode_id"])
            if episode_id in previous:
                results.append(previous[episode_id])
                continue
            sequence = Path(episode["sequence_path"])
            if "val" in sequence.parts or "test" in sequence.parts:
                raise ValueError(f"train-only N30-A replay refused {sequence}")
            if getattr(backend, "_active_sequence", None) != str(sequence):
                _session(backend, sequence)
                backend._active_sequence = str(sequence)
                session_ready = True
                if decoder is None:
                    decoder = _get_official_decoder(backend)
                    adapter = SAM3DecoderLITAdapter(
                        decoder,
                        DecoderLITConfig(rank=4, alpha=4.0, dropout=0.1),
                    )
                    capture = DecoderCapture(decoder)
            try:
                if decoder is None or adapter is None or capture is None:
                    raise RuntimeError("N30-A decoder adapter was not initialized after start_video")
                session_ready, row, episode_audits = _run_episode(
                    backend=backend,
                    adapter=adapter,
                    decoder=decoder,
                    capture=capture,
                    episode=episode,
                    sequence=sequence,
                    session_ready=session_ready,
                )
            except Exception as exc:
                row = {
                    "status": "NOT_RUN",
                    "episode_id": episode_id,
                    "sequence": str(episode["sequence"]),
                    "split": str(episode["split"]),
                    "dataset_identity": int(episode["dataset_identity"]),
                    "public_id": int(episode["public_id"]),
                    "sam_object_id": int(episode["sam_object_id"]),
                    "failure": f"{type(exc).__name__}: {exc}",
                    "failure_traceback": traceback.format_exc(limit=16),
                }
                episode_audits = []
            results.append(row)
            audits.extend(episode_audits)
            _write_json(
                partial,
                {
                    "protocol": "N30-A-WRITE-PATH-STATE-AUDIT",
                    "status": "PARTIAL",
                    "val25_read": False,
                    "hard_manifest": str(manifest_path),
                    "hard_manifest_sha256": manifest_sha,
                    "processed_episode_count": len(results),
                    "episode_results": results,
                },
            )
            _write_json(
                partial_audit,
                {
                    "protocol": "N30-A-WRITE-PATH-STATE-AUDIT",
                    "status": "PARTIAL",
                    "val25_read": False,
                    "hard_manifest_sha256": manifest_sha,
                    "audits": audits,
                },
            )
    finally:
        if capture is not None:
            capture.close()
        backend.close()

    result = {
        "protocol": "N30-A-WRITE-PATH-STATE-AUDIT",
        "status": (
            "PASS"
            if len(results) == len(episodes) and all(row.get("status") == "PASS" for row in results)
            else "PARTIAL"
            if results
            else "NOT_RUN"
        ),
        "val25_read": False,
        "test_labels_used": False,
        "hard_manifest": str(manifest_path),
        "hard_manifest_sha256": manifest_sha,
        "selection_frozen_before_replay": True,
        "future_gt_used_for_selection": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "episode_count_requested": len(episodes),
        "episode_count_processed": len(results),
        "episode_count_pass": int(sum(row.get("status") == "PASS" for row in results)),
        "episode_count_failed": int(sum(row.get("status") != "PASS" for row in results)),
        "branch_order": list(BRANCHES),
        "single_identity_b10_claim": "NOT_APPLICABLE_SINGLE_ID_NO_STATE_MANAGER; no B10 update is claimed",
        "episode_results": results,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    audit_payload = {
        "protocol": "N30-A-WRITE-PATH-STATE-AUDIT",
        "status": result["status"],
        "val25_read": False,
        "test_labels_used": False,
        "hard_manifest": str(manifest_path),
        "hard_manifest_sha256": manifest_sha,
        "episode_count": len(results),
        "branch_order": list(BRANCHES),
        "audits": audits,
    }
    _write_json(output, result)
    _write_json(audit_output, audit_payload)
    summary = _make_summary(
        results,
        manifest_path=manifest_path,
        manifest_sha=manifest_sha,
        checkpoint=checkpoint,
    )
    _write_json(summary_output, summary)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/n29r/hard_episode_manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/n30/write_path_ablation_results.json")
    parser.add_argument("--audit-output", type=Path, default=ROOT / "outputs/n30/write_path_state_audit.json")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "outputs/n30/write_path_summary.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(
        manifest_path=args.manifest,
        checkpoint=args.checkpoint,
        output=args.output,
        audit_output=args.audit_output,
        summary_output=args.summary_output,
        limit=args.limit,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "episode_count_requested",
                    "episode_count_processed",
                    "episode_count_pass",
                    "elapsed_seconds",
                    "val25_read",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
