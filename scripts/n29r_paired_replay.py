#!/usr/bin/env python3
"""Run the frozen N29-R paired correction replay on train-fold episodes.

Every branch resets the official session, installs the same initial singleton
box, and observes the same pre-correction frame.  Only B/C/D then write the
same current box correction.  C additionally runs the real decoder LoRA
transaction; D calls the same transaction with its optimizer disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    SAM3DecoderLITAdapter,
)
from scripts.n29_lit_online_replay import (  # noqa: E402
    DecoderCapture,
    _clone_tree,
    _image_files,
    _install_official_box_singleton,
    _iou,
    _make_backend,
    _read_gt,
    _select_observation,
    _session,
    _slot_tensor,
    _tensor_status_tree,
    _trial_outputs,
    _get_official_decoder,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "PASS":
        raise ValueError("hard manifest is not a frozen PASS artifact")
    if payload.get("val25_read") is not False or payload.get("future_gt_used_for_selection") is not False:
        raise ValueError("hard manifest violates the blind/causal selection boundary")
    if payload.get("selection_frozen_before_paired") is not True:
        raise ValueError("hard manifest was not frozen before paired replay")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("hard manifest contains no episodes")
    for episode in episodes:
        sequence = str(episode.get("sequence", ""))
        split = str(episode.get("split", ""))
        if "val" in sequence.lower() or "test" in sequence.lower() or "val" in split.lower() or "test" in split.lower():
            raise ValueError(f"paired replay refused non-train episode {episode}")
    return payload, _sha256(path)


def _box_area(box: Any) -> float:
    x1, y1, x2, y2 = (float(value) for value in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _branch_metrics(
    outputs: Mapping[int, list[Any]],
    gt: Mapping[int, Mapping[int, np.ndarray]],
    *,
    episode: Mapping[str, Any],
    start: int,
    end: int,
) -> dict[str, Any]:
    dataset_identity = int(episode["dataset_identity"])
    public_id = int(episode["public_id"])
    evaluation = _trial_outputs(
        outputs,
        gt,
        dataset_identity=dataset_identity,
        public_id=public_id,
        start=start,
        end=end,
        require_visible=False,
    )
    rows = evaluation["rows"]
    visible = [row for row in rows if row["target_present"]]
    next_error = None
    for row in visible:
        if not row["prediction_present"] or float(row["box_iou"]) < 0.5:
            next_error = int(row["frame"])
            break
    masks: list[float] = []
    box_areas: list[float] = []
    for frame in range(start, end + 1):
        target = gt.get(frame, {}).get(dataset_identity)
        obs = _select_observation(outputs.get(frame, []), public_id, target)
        if obs is None:
            continue
        mask = np.asarray(getattr(obs, "mask", np.zeros((1, 1))), dtype=bool)
        if mask.ndim == 2 and mask.size > 1:
            masks.append(float(mask.sum()))
        box_areas.append(_box_area(obs.box_xyxy))
    area_values = masks if masks else box_areas
    area_drift = None
    if area_values and area_values[0] > 0:
        baseline = area_values[0]
        area_drift = float(np.mean([abs(value / baseline - 1.0) for value in area_values]))
    visible_count = len(visible)
    return {
        "start": int(start),
        "end": int(end),
        "evaluated_frame_count": len(rows),
        "visible_frame_count": evaluation["visible_frame_count"],
        "absent_gt_frame_count": evaluation["absent_gt_frame_count"],
        "mean_box_iou_visible": evaluation["mean_box_iou_visible"],
        "success_at_0_5_visible": evaluation["success_at_0_5_visible"],
        "missing_prediction_rate_visible": (
            float(evaluation["missing_prediction_on_visible_count"] / visible_count)
            if visible_count
            else None
        ),
        "missing_prediction_on_visible_count": evaluation["missing_prediction_on_visible_count"],
        "error_count_visible": int(sum(
            not row["prediction_present"] or float(row["box_iou"]) < 0.5
            for row in visible
        )),
        "time_to_next_correction_frames": (
            None if next_error is None else int(next_error - int(episode["correction_frame"]))
        ),
        "future_recorrection_rate": "NOT_RUN_FULL_LOOP",
        "mask_area_drift": area_drift,
        "mask_area_sample_count": len(masks),
        "box_area_drift_proxy": (
            None
            if not box_areas or box_areas[0] <= 0
            else float(np.mean([abs(value / box_areas[0] - 1.0) for value in box_areas]))
        ),
        "rows": rows,
    }


def _horizon_metrics(
    outputs: Mapping[int, list[Any]],
    gt: Mapping[int, Mapping[int, np.ndarray]],
    episode: Mapping[str, Any],
) -> dict[str, Any]:
    correction = int(episode["correction_frame"])
    end = int(episode["query_end"])
    result: dict[str, Any] = {}
    for name, horizon_end in (("5", correction + 5), ("10", correction + 10), ("20", correction + 20)):
        result[name] = _branch_metrics(
            outputs,
            gt,
            episode=episode,
            start=correction + 1,
            end=min(end, horizon_end),
        )
    return result


def _difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "mean_box_iou_visible",
        "success_at_0_5_visible",
        "missing_prediction_rate_visible",
        "error_count_visible",
        "mask_area_drift",
    )
    result: dict[str, Any] = {}
    for key in keys:
        a = left.get(key)
        b = right.get(key)
        result[key] = None if a is None or b is None else float(a - b)
    return result


def _update_dict(update: Any) -> dict[str, Any]:
    if update is None:
        return {"status": "NOT_RUN", "committed": False}
    if isinstance(update, dict):
        return update
    return {
        "transaction_id": update.transaction_id,
        "status": update.status,
        "committed": update.committed,
        "frame_idx": update.frame_idx,
        "public_id": update.public_id,
        "adapter_version": update.adapter_version,
        "loss_history": list(update.loss_history),
        "gradient_parameter_count": update.gradient_parameter_count,
        "rollback_reason": update.rollback_reason,
        "provenance": update.provenance,
        "exception_traceback": update.exception_traceback,
        "optimization_diagnostic": dict(update.optimization_diagnostic),
    }


def _prepare_singleton(
    backend: Any,
    *,
    sequence: Path,
    session_ready: bool,
    frame: int,
    public_id: int,
    box: np.ndarray,
) -> bool:
    if session_ready:
        backend.reset_session()
    else:
        _session(backend, sequence)
    backend.add_box(frame, public_id, box)
    _install_official_box_singleton(
        backend,
        frame_idx=frame,
        public_id=public_id,
        box_xyxy=box,
    )
    return True


def _ensure_public_singleton_binding(
    backend: Any,
    *,
    frame: int,
    public_id: int,
    box: np.ndarray,
) -> dict[str, Any]:
    """Keep the official raw/SAM object namespace bound after a box write."""

    predictor = backend._predictor
    entry = predictor._all_inference_states[backend._session_id]
    state = entry["state"]
    existing_ids = [
        int(obj_id)
        for tracker_state in state.get("sam2_inference_states", [])
        for obj_id in np.asarray(tracker_state.get("obj_ids", [])).reshape(-1)
    ]
    if existing_ids != [int(public_id)]:
        state["sam2_inference_states"] = []
    binding = _install_official_box_singleton(
        backend,
        frame_idx=frame,
        public_id=public_id,
        box_xyxy=box,
    )
    if binding.get("mapping_restored"):
        backend._bind_external_sam_id(
            int(public_id), int(binding.get("sam_id", public_id))
        )
    return binding


def _future_with_correction(
    backend: Any,
    *,
    sequence: Path,
    session_ready: bool,
    episode: Mapping[str, Any],
    capture: DecoderCapture,
    decoder: torch.nn.Module,
    adapter: SAM3DecoderLITAdapter,
    with_lora: bool,
    zero_update: bool,
) -> tuple[bool, dict[str, Any], dict[int, list[Any]], dict[str, Any]]:
    init = int(episode["initialization_frame"])
    correction = int(episode["correction_frame"])
    query_end = int(episode["query_end"])
    public_id = int(episode["public_id"])
    init_box = np.asarray(episode["anchor_current_error"].get("target_box"), dtype=float)
    correction_box = np.asarray(episode["correction_box"], dtype=float)
    _prepare_singleton(
        backend,
        sequence=sequence,
        session_ready=session_ready,
        frame=init,
        public_id=public_id,
        box=init_box,
    )
    capture.reset(target_call=max(0, correction - init))
    pre_outputs = backend.propagate(init, correction, start_frame_index=init)
    support_kwargs = None if capture.target_inputs is None else _clone_tree(capture.target_inputs)
    current_recorded = bool(correction in pre_outputs)
    backend.correct_object(correction, public_id, box_xyxy=correction_box)
    correction_binding = _ensure_public_singleton_binding(
        backend,
        frame=correction,
        public_id=public_id,
        box=correction_box,
    )
    event = DecoderCorrectionEvent(
        video_id=f"{episode['sequence']}:{episode['episode_id']}",
        public_id=public_id,
        frame_idx=correction,
        provenance=BOX_DERIVED_PSEUDO_MASK,
        box_xyxy=correction_box,
        image_size=(int(backend._frame_h), int(backend._frame_w)),
        current_output_recorded=current_recorded,
        metadata={
            "teacher": "explicit_box_rectangle_pseudo_target",
            "click_count": "0",
            "branch": "correction_plus_zero_update" if zero_update else "correction_plus_lora",
        },
    )
    state = None
    update: Any = None
    update_diagnostic: dict[str, Any] = {
        "status": "NOT_RUN",
        "reason": "write_only_branch" if not with_lora and not zero_update else None,
    }
    forward_diagnostic: dict[str, Any] = {
        "support_inputs": [] if support_kwargs is None else _tensor_status_tree(support_kwargs),
        "support_inputs_exposed": support_kwargs is not None,
        "current_output_recorded": current_recorded,
    }
    if with_lora or zero_update:
        state = adapter.new_state(
            f"{episode['sequence']}:{episode['episode_id']}:{'zero' if zero_update else 'lora'}",
            public_id,
            device=adapter.device,
        )
        if support_kwargs is None:
            update_diagnostic = {
                "status": "NOT_RUN",
                "reason": "official propagation decoder hook did not expose support inputs",
            }
        elif not current_recorded:
            update_diagnostic = {
                "status": "NOT_RUN",
                "reason": "correction frame was not recorded before update",
            }
        else:
            config = DecoderUpdateConfig(
                inner_steps=5,
                learning_rate=1.0e-4,
                weight_decay=0.0,
                optimizer_enabled=not zero_update,
                require_loss_decrease=False,
                require_observable_update=not zero_update,
            )

            def forward_fn(_supervision, _step):
                with torch.inference_mode(False), torch.enable_grad():
                    raw = decoder(**support_kwargs)
                return _slot_tensor(raw["masks"], slot=0)

            def deterministic_forward(_supervision):
                was_training = decoder.training
                decoder.eval()
                try:
                    with torch.inference_mode(False), torch.no_grad():
                        raw = decoder(**support_kwargs)
                    return _slot_tensor(raw["masks"], slot=0)
                finally:
                    decoder.train(was_training)

            transaction = DecoderUpdateTransaction(adapter, config)
            update = transaction.apply(
                event,
                state,
                forward_fn=forward_fn,
                deterministic_forward_fn=deterministic_forward,
            )
            update_diagnostic = _update_dict(update)
            forward_diagnostic.update(
                {
                    "decoder_class": type(decoder).__name__,
                    "decoder_transformer_class": type(decoder.transformer).__name__,
                    "state_parameter_is_inference": bool(
                        torch.is_inference(next(iter(state.lora_a.values())))
                    ),
                }
            )
    update_status = str(update_diagnostic.get("status", "NOT_RUN"))
    committed = bool(update_diagnostic.get("committed", False))
    future_outputs: dict[int, list[Any]]
    if with_lora and committed and state is not None:
        with adapter.activate(state):
            future_outputs = backend.propagate(
                correction + 1,
                query_end,
                start_frame_index=correction + 1,
            )
        future_adapter_active = True
    else:
        future_outputs = backend.propagate(
            correction + 1,
            query_end,
            start_frame_index=correction + 1,
        )
        future_adapter_active = False
    branch_status = "PASS" if (not with_lora or update_status in ("COMMIT", "ROLLBACK")) else "NOT_RUN"
    return (
        session_ready,
        {
            "status": branch_status,
            "correction_written": True,
            "correction_write_type": "box",
            "supervision_provenance": BOX_DERIVED_PSEUDO_MASK,
            "click_count": 0,
            "mask_corrections": 0,
            "update": update_diagnostic,
            "forward_diagnostic": forward_diagnostic,
            "future_adapter_active": future_adapter_active,
            "adapter_version_after": 0 if state is None else int(state.adapter_version),
            "official_correction_binding": correction_binding,
            "outputs": future_outputs,
        },
        future_outputs,
        {"update_status": update_status, "committed": committed},
    )


def _run_episode(
    backend: Any,
    capture: DecoderCapture,
    decoder: torch.nn.Module,
    adapter: SAM3DecoderLITAdapter,
    episode: Mapping[str, Any],
    *,
    sequence: Path,
    session_ready: bool,
) -> tuple[bool, dict[str, Any]]:
    gt = _read_gt(sequence)
    image_count = len(_image_files(sequence))
    init = int(episode["initialization_frame"])
    correction = int(episode["correction_frame"])
    query_end = min(int(episode["query_end"]), image_count - 1)
    if query_end < correction + 5:
        raise ValueError(f"episode lacks five future frames: {episode['episode_id']}")
    init_box = np.asarray(gt[init][int(episode["dataset_identity"])], dtype=float)
    public_id = int(episode["public_id"])
    branch_start_digest = hashlib.sha256(
        json.dumps(
            {
                "sequence": episode["sequence"],
                "episode_id": episode["episode_id"],
                "dataset_identity": episode["dataset_identity"],
                "public_id": public_id,
                "initialization_frame": init,
                "checkpoint_protocol": "same loaded frozen model; reset before each branch",
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    branches: dict[str, Any] = {}
    branch_outputs: dict[str, Mapping[int, list[Any]]] = {}
    branch_timings: dict[str, Any] = {}
    for name in ("anchor_no_correction", "correction_write_only"):
        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        session_ready = _prepare_singleton(
            backend,
            sequence=sequence,
            session_ready=session_ready,
            frame=init,
            public_id=public_id,
            box=init_box,
        )
        if name == "anchor_no_correction":
            outputs = backend.propagate(init, query_end, start_frame_index=init)
            branch = {
                "status": "PASS",
                "correction_written": False,
                "correction_write_type": None,
                "supervision_provenance": None,
                "click_count": 0,
                "mask_corrections": 0,
                "future_adapter_active": False,
                "outputs": outputs,
            }
        else:
            # The write-only branch is deliberately separate from A so that
            # the public correction/memory write is measured independently.
            capture.reset(target_call=max(0, correction - init))
            pre_outputs = backend.propagate(init, correction, start_frame_index=init)
            backend.correct_object(
                correction,
                public_id,
                box_xyxy=np.asarray(episode["correction_box"], dtype=float),
            )
            correction_binding = _ensure_public_singleton_binding(
                backend,
                frame=correction,
                public_id=public_id,
                box=np.asarray(episode["correction_box"], dtype=float),
            )
            outputs = backend.propagate(
                correction + 1,
                query_end,
                start_frame_index=correction + 1,
            )
            branch = {
                "status": "PASS",
                "correction_written": True,
                "correction_write_type": "box",
                "supervision_provenance": BOX_DERIVED_PSEUDO_MASK,
                "click_count": 0,
                "mask_corrections": 0,
                "current_output_recorded": bool(correction in pre_outputs),
                "future_adapter_active": False,
                "official_correction_binding": correction_binding,
                "outputs": outputs,
            }
        branch_outputs[name] = outputs
        branch["metrics"] = _horizon_metrics(outputs, gt, episode)
        branch["timing"] = {
            "wall_seconds": float(time.perf_counter() - started),
            "peak_gpu_memory_allocated_bytes": (
                None if not torch.cuda.is_available() else int(torch.cuda.max_memory_allocated())
            ),
        }
        branch.pop("outputs", None)
        branches[name] = branch
        branch_timings[name] = branch["timing"]

    for name, with_lora, zero_update in (
        ("correction_plus_lora", True, False),
        ("correction_plus_zero_update", False, True),
    ):
        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        session_ready, branch, outputs, _ = _future_with_correction(
            backend,
            sequence=sequence,
            session_ready=session_ready,
            episode=episode,
            capture=capture,
            decoder=decoder,
            adapter=adapter,
            with_lora=with_lora,
            zero_update=zero_update,
        )
        branch_outputs[name] = outputs
        branch["metrics"] = _horizon_metrics(outputs, gt, episode)
        branch["timing"] = {
            "wall_seconds": float(time.perf_counter() - started),
            "peak_gpu_memory_allocated_bytes": (
                None if not torch.cuda.is_available() else int(torch.cuda.max_memory_allocated())
            ),
        }
        branch.pop("outputs", None)
        branches[name] = branch
        branch_timings[name] = branch["timing"]

    # There is no prior correction in the frozen episode prefix.  Removing
    # the event therefore returns exactly to A; record this as a legal alias,
    # rather than pretending a non-existent historical memory entry was removed.
    branches["remove_latest_correction"] = {
        "status": "ALIAS_TO_ANCHOR",
        "semantics": "episode starts immediately before its first correction; removing that event is branch A",
        "reused_branch": "anchor_no_correction",
        "metrics": branches["anchor_no_correction"]["metrics"],
    }
    paired_delta: dict[str, Any] = {}
    for horizon in ("5", "10", "20"):
        paired_delta[horizon] = _difference(
            branches["correction_plus_lora"]["metrics"][horizon],
            branches["correction_write_only"]["metrics"][horizon],
        )
    paired_delta["lora_vs_write_only_definition"] = "correction_plus_lora - correction_write_only"
    paired_delta["same_future_frame_count"] = {
        horizon: branches["correction_plus_lora"]["metrics"][horizon]["evaluated_frame_count"]
        == branches["correction_write_only"]["metrics"][horizon]["evaluated_frame_count"]
        for horizon in ("5", "10", "20")
    }
    return session_ready, {
        "status": "PASS",
        "episode_id": episode["episode_id"],
        "sequence": episode["sequence"],
        "split": episode["split"],
        "dataset_identity": int(episode["dataset_identity"]),
        "public_id": public_id,
        "sam_object_id": int(episode["sam_object_id"]),
        "identity_binding": episode["identity_binding"],
        "initialization_frame": init,
        "correction_frame": correction,
        "query_start": correction + 1,
        "query_end": query_end,
        "correction_type": "box",
        "supervision_provenance": BOX_DERIVED_PSEUDO_MASK,
        "paired_start_state_digest": branch_start_digest,
        "byte_identical_pre_correction_state_protocol": True,
        "branches": branches,
        "paired_delta": paired_delta,
        "unaffected_identity_regression": "NOT_APPLICABLE_SINGLETON; measured in N29-R5 multi-ID association",
        "branch_timing": branch_timings,
    }


def run(
    *,
    manifest_path: Path,
    checkpoint: Path,
    output: Path,
    limit: Optional[int],
    resume: bool,
) -> dict[str, Any]:
    manifest, manifest_sha = _load_manifest(manifest_path)
    episodes = list(manifest["episodes"])
    if limit is not None:
        episodes = episodes[:limit]
    partial = output.with_suffix(".partial.json")
    previous: dict[str, Any] = {}
    if resume and partial.is_file():
        try:
            old = json.loads(partial.read_text(encoding="utf-8"))
            previous = {row["episode_id"]: row for row in old.get("episode_results", [])}
        except Exception:
            previous = {}
    backend = _make_backend(checkpoint)
    decoder = None
    adapter = None
    capture = None
    session_ready = False
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for index, episode in enumerate(episodes):
            if episode["episode_id"] in previous:
                results.append(previous[episode["episode_id"]])
                continue
            sequence = Path(episode["sequence_path"])
            if "val" in sequence.parts or "test" in sequence.parts:
                raise ValueError(f"train-only paired replay refused {sequence}")
            if decoder is None or str(getattr(backend, "_active_sequence", "")) != str(sequence):
                _session(backend, sequence)
                backend._active_sequence = str(sequence)
                # The official backend keeps one predictor/model alive across
                # train sequences.  The adapter wraps that decoder exactly
                # once; re-wrapping the already wrapped q/k/v projections on
                # every sequence makes a resumable multi-sequence replay fail
                # with "target is not Linear".
                if decoder is None:
                    decoder = _get_official_decoder(backend)
                    adapter = SAM3DecoderLITAdapter(
                        decoder,
                        DecoderLITConfig(rank=4, alpha=4.0, dropout=0.1),
                    )
                    capture = DecoderCapture(decoder)
                session_ready = True
            try:
                session_ready, row = _run_episode(
                    backend,
                    capture,
                    decoder,
                    adapter,
                    episode,
                    sequence=sequence,
                    session_ready=session_ready,
                )
            except Exception as exc:
                row = {
                    "episode_id": episode["episode_id"],
                    "sequence": episode["sequence"],
                    "split": episode["split"],
                    "dataset_identity": int(episode["dataset_identity"]),
                    "public_id": int(episode["public_id"]),
                    "sam_object_id": int(episode["sam_object_id"]),
                    "status": "NOT_RUN",
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            results.append(row)
            partial_payload = {
                "protocol": "N29-R3-PAIRED-REPLAY",
                "status": "PARTIAL",
                "val25_read": False,
                "hard_manifest": str(manifest_path),
                "hard_manifest_sha256": manifest_sha,
                "processed_episode_count": len(results),
                "episode_results": results,
            }
            _write_json(partial, partial_payload)
    finally:
        if capture is not None:
            capture.close()
        backend.close()
    return {
        "protocol": "N29-R3-PAIRED-REPLAY",
        "status": (
            "PASS"
            if len(results) == len(episodes)
            and all(row.get("status") == "PASS" for row in results)
            else "PARTIAL"
            if results
            else "NOT_RUN"
        ),
        "val25_read": False,
        "test_labels_used": False,
        "hard_manifest": str(manifest_path),
        "hard_manifest_sha256": manifest_sha,
        "selection_frozen_before_paired": True,
        "future_gt_used_for_selection": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "episode_count_requested": len(episodes),
        "episode_count_processed": len(results),
        "episode_count_pass": int(sum(row.get("status") == "PASS" for row in results)),
        "episode_count_failed": int(sum(row.get("status") != "PASS" for row in results)),
        "paired_branch_order": [
            "anchor_no_correction",
            "correction_write_only",
            "correction_plus_lora",
            "correction_plus_zero_update",
            "remove_latest_correction",
        ],
        "lora_gain_definition": "correction_plus_lora - correction_write_only",
        "episode_results": results,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/n29r/hard_episode_manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/n29r/paired_replay_results.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(
        manifest_path=args.manifest,
        checkpoint=args.checkpoint,
        output=args.output,
        limit=args.limit,
        resume=args.resume,
    )
    _write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "episode_count_requested", "episode_count_processed", "elapsed_seconds", "val25_read")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
