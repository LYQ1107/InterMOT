#!/usr/bin/env python3
"""Run the N30 strict-future correction-memory writer mechanism gate.

The script consumes only the frozen official-decoder tape produced by
``n30_collect_writer_tensors.py``.  The writer sees support-time evidence;
future decoder inputs are used only to compute offline query supervision.
The base SAM3 decoder is frozen and the only trainable parameters are those of
``CorrectionMemoryWriter``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
SAM3_ROOT = ROOT / "third_party" / "sam3"
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

from sam3_intermot.adaptation.correction_memory_transaction import (  # noqa: E402
    CorrectionMemoryTransaction,
)
from sam3_intermot.adaptation.correction_memory_writer import (  # noqa: E402
    CorrectionMemoryInputs,
    CorrectionMemoryWriter,
    action_id,
    roi_average_pool,
)
from scripts.n29_lit_online_replay import (  # noqa: E402
    _get_official_decoder,
    _make_backend,
)


CHECKPOINT = ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
DEFAULT_DATA_DIR = ROOT / "outputs/n30/writer_dataset"
DEFAULT_INDEX = ROOT / "outputs/n30/writer_dataset_index_final.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/n30"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _move_tree(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device=device, non_blocking=False)
    if isinstance(value, Mapping):
        return {key: _move_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device) for item in value)
    return value


def _writer_forward(writer: CorrectionMemoryWriter, inputs: CorrectionMemoryInputs) -> dict[str, Tensor]:
    """Keep the writer's low-rank state in fp32 beside official bf16 SAM3."""

    device = next(writer.parameters()).device
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", enabled=False):
            return writer(inputs)
    return writer(inputs)


def _load_samples(index_path: Path, data_dir: Path, limit: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("status") != "PASS":
        raise RuntimeError(f"writer dataset index is not PASS: {index.get('status')}")
    records = [item for item in index.get("records", []) if item.get("status") == "PASS" and item.get("role") == "meta_train"]
    if len(records) < 20 and limit is None:
        raise RuntimeError(f"strict N30-E requires 20 meta_train episodes, found {len(records)}")
    if limit is not None:
        records = records[: int(limit)]
    samples: list[dict[str, Any]] = []
    for record in records:
        sample_path = Path(record["sample_path"])
        if not sample_path.is_absolute():
            sample_path = ROOT / sample_path
        if not sample_path.exists():
            sample_path = data_dir / Path(record["sample_path"]).name
        sample = torch.load(sample_path, map_location="cpu", weights_only=False)
        if len(sample.get("future_kwargs", [])) != len(sample.get("future_frames", [])):
            raise RuntimeError(f"future tape/frame mismatch in {sample_path}")
        if len(sample.get("future_kwargs", [])) != 20:
            raise RuntimeError(f"N30-E requires H20 tape in {sample_path}")
        if any(int(frame) <= int(sample["correction_frame"]) for frame in sample["future_frames"]):
            raise RuntimeError(f"non-future frame found in {sample_path}")
        if sample.get("future_gt_used_for_writer_input") is not False:
            raise RuntimeError(f"future GT was marked as writer input in {sample_path}")
        samples.append(sample)
    return samples, index


def _normalize_box(box: Any, image_size: tuple[int, int], device: torch.device) -> Tensor:
    height, width = (float(value) for value in image_size)
    x1, y1, x2, y2 = (float(value) for value in box)
    return torch.tensor([x1 / width, y1 / height, x2 / width, y2 / height], dtype=torch.float32, device=device)


def _writer_inputs(sample: Mapping[str, Any], device: torch.device) -> CorrectionMemoryInputs:
    image_size = tuple(int(value) for value in sample["image_size"])
    correction_box = sample["correction_box"]
    predicted_box = sample["predicted_box_at_correction"]
    correction_norm = _normalize_box(correction_box, image_size, device).unsqueeze(0)
    predicted_norm = _normalize_box(predicted_box, image_size, device).unsqueeze(0)
    support = _move_tree(sample["support_kwargs"], device)
    image_embeddings = support["image_embeddings"].float()
    e_obj = support["extra_per_object_embeddings"].float()
    e_roi = roi_average_pool(image_embeddings, correction_norm)
    e_pred = roi_average_pool(image_embeddings, predicted_norm)
    height, width = (float(value) for value in image_size)
    x1, y1, x2, y2 = (float(value) for value in correction_box)
    px1, py1, px2, py2 = (float(value) for value in predicted_box)
    g_box = torch.tensor(
        [x1 / width, y1 / height, x2 / width, y2 / height,
         max(0.0, x2 - x1) / width, max(0.0, y2 - y1) / height,
         ((x1 + x2) / 2.0) / width, ((y1 + y2) / 2.0) / height],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    g_residual = torch.tensor(
        [(x2 - px2) / width, (y2 - py2) / height,
         (x1 - px1) / width, (y1 - py1) / height],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    clip_feature = torch.as_tensor(sample["clip_feature"], dtype=torch.float32, device=device).reshape(1, -1)
    missing_flag = torch.tensor(
        [[0.0 if bool(sample.get("current_output_recorded", False)) else 1.0]],
        dtype=torch.float32,
        device=device,
    )
    action = torch.tensor(action_id(str(sample.get("action", "BOX_CORRECTION"))), dtype=torch.long, device=device)
    return CorrectionMemoryInputs(
        e_obj=e_obj,
        e_roi=e_roi,
        e_pred=e_pred,
        f_clip=clip_feature,
        g_box=g_box,
        g_residual=g_residual,
        missing_flag=missing_flag,
        action=action,
    )


def _mask_logits(output: Mapping[str, Any], slot: int = 0, token: int = 0) -> Tensor:
    value = output.get("masks")
    if value is None:
        raise RuntimeError("official decoder output has no masks")
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    if tensor.ndim == 5:  # [B, multiplex, mask_token, H, W]
        return tensor[:, slot, token].float()
    if tensor.ndim == 4:  # [B, multiplex, H, W]
        return tensor[:, slot].float()
    if tensor.ndim == 3:  # [B, H, W]
        return tensor.float()
    raise RuntimeError(f"unsupported official mask shape {tuple(tensor.shape)}")


def _presence_logits(output: Mapping[str, Any], slot: int = 0) -> Tensor:
    value = output.get("object_score_logits")
    if value is None:
        return torch.zeros((1,), dtype=torch.float32, device=_mask_logits(output, slot).device)
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    if tensor.ndim >= 3:
        return tensor[:, slot, 0].float()
    if tensor.ndim == 2:
        return tensor[:, slot].float()
    if tensor.ndim == 1:
        return tensor[slot : slot + 1].float()
    return tensor.reshape(1).float()


def _target_mask(box: Any, image_size: tuple[int, int], output_size: tuple[int, int], device: torch.device) -> Tensor:
    image_height, image_width = (float(value) for value in image_size)
    output_height, output_width = output_size
    x1, y1, x2, y2 = (float(value) for value in box)
    left = max(0, min(output_width - 1, int(np.floor(x1 / image_width * output_width))))
    top = max(0, min(output_height - 1, int(np.floor(y1 / image_height * output_height))))
    right = max(left + 1, min(output_width, int(np.ceil(x2 / image_width * output_width))))
    bottom = max(top + 1, min(output_height, int(np.ceil(y2 / image_height * output_height))))
    target = torch.zeros((1, output_height, output_width), dtype=torch.float32, device=device)
    target[:, top:bottom, left:right] = 1.0
    return target


def _protected_loss(base_extra: Tensor, modified_extra: Tensor, target_slot: int) -> Tensor:
    if base_extra.shape[1] <= 1:
        return modified_extra.new_zeros(())
    protected = [index for index in range(base_extra.shape[1]) if index != int(target_slot)]
    return (modified_extra[:, protected, :].float() - base_extra[:, protected, :].float()).pow(2).mean()


def _query_loss(
    output: Mapping[str, Any],
    target_box: Any,
    image_size: tuple[int, int],
    residual: Tensor,
    base_extra: Tensor,
    modified_extra: Tensor,
    *,
    target_slot: int = 0,
) -> tuple[Tensor, dict[str, float], Tensor]:
    masks = _mask_logits(output, target_slot)
    target = _target_mask(target_box, image_size, (int(masks.shape[-2]), int(masks.shape[-1])), masks.device)
    box_loss = F.binary_cross_entropy_with_logits(masks, target)
    presence = _presence_logits(output, target_slot)
    presence_loss = F.binary_cross_entropy_with_logits(presence, torch.ones_like(presence))
    protect_loss = _protected_loss(base_extra, modified_extra, target_slot)
    residual_loss = residual.float().pow(2).mean()
    total = box_loss + 0.1 * presence_loss + 0.1 * protect_loss + 0.001 * residual_loss
    values = {
        "loss_total": float(total.detach().cpu()),
        "loss_box": float(box_loss.detach().cpu()),
        "loss_presence": float(presence_loss.detach().cpu()),
        "loss_protect": float(protect_loss.detach().cpu()),
        "loss_residual": float(residual_loss.detach().cpu()),
    }
    return total, values, target


def _decode(
    decoder: torch.nn.Module,
    kwargs: Mapping[str, Any],
    residual: Tensor | None = None,
    *,
    target_slot: int = 0,
    requires_grad: bool = False,
) -> tuple[Mapping[str, Any], Tensor | None]:
    call_kwargs = dict(kwargs)
    modified_extra = None
    if residual is not None:
        base_extra = call_kwargs["extra_per_object_embeddings"]
        modified_extra = base_extra.clone()
        modified_extra[:, target_slot, :] = modified_extra[:, target_slot, :] + residual[:, target_slot, :].to(
            device=base_extra.device, dtype=base_extra.dtype
        )
        call_kwargs["extra_per_object_embeddings"] = modified_extra
    with torch.inference_mode(False):
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            output = decoder(**call_kwargs)
    return output, modified_extra


def _mask_box(mask: Tensor) -> list[float] | None:
    coordinates = torch.nonzero(mask, as_tuple=False)
    if coordinates.numel() == 0:
        return None
    y1 = float(coordinates[:, -2].min())
    y2 = float(coordinates[:, -2].max() + 1)
    x1 = float(coordinates[:, -1].min())
    x2 = float(coordinates[:, -1].max() + 1)
    return [x1, y1, x2, y2]


def _box_iou(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return float(intersection / union) if union > 0 else 0.0


def _metric(output: Mapping[str, Any], target: Tensor, slot: int = 0) -> dict[str, float | None]:
    logits = _mask_logits(output, slot)
    predicted = torch.sigmoid(logits)[0] > 0.5
    target_bool = target[0] > 0.5
    intersection = float((predicted & target_bool).sum().detach().cpu())
    union = float((predicted | target_bool).sum().detach().cpu())
    mask_iou = intersection / union if union > 0 else 0.0
    predicted_box = _mask_box(predicted)
    target_box = _mask_box(target_bool)
    presence = float(torch.sigmoid(_presence_logits(output, slot).reshape(-1)[0]).detach().cpu())
    return {
        "mask_iou": float(mask_iou),
        "box_iou_proxy": _box_iou(predicted_box, target_box),
        "presence_probability": presence,
        "prediction_present": float(predicted_box is not None),
    }


def _bootstrap_ci(values: list[float], seed: int, count: int = 2000) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(count, len(array)), replace=True).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _evaluate(
    writer: CorrectionMemoryWriter,
    decoder: torch.nn.Module,
    samples: list[dict[str, Any]],
    writer_inputs: list[CorrectionMemoryInputs],
    device: torch.device,
) -> dict[str, Any]:
    writer.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        writer_outputs = [_writer_forward(writer, inputs) for inputs in writer_inputs]
    for sample, writer_output in zip(samples, writer_outputs):
        residual = writer_output["residual"].detach()
        for future_index, (future_kwargs_cpu, target_box, frame) in enumerate(
            zip(sample["future_kwargs"], sample["future_target_boxes"], sample["future_frames"])
        ):
            future_kwargs = _move_tree(future_kwargs_cpu, device)
            target_slot = int(sample.get("target_slot", 0))
            expected_extra_delta = residual[:, target_slot, :].to(
                device=future_kwargs["extra_per_object_embeddings"].device,
                dtype=future_kwargs["extra_per_object_embeddings"].dtype,
            )
            base_output, _ = _decode(decoder, future_kwargs, requires_grad=False)
            learned_output, modified_extra = _decode(
                decoder,
                future_kwargs,
                residual,
                target_slot=target_slot,
                requires_grad=False,
            )
            image_size = tuple(int(value) for value in sample["image_size"])
            target = _target_mask(
                target_box,
                image_size,
                (int(_mask_logits(base_output).shape[-2]), int(_mask_logits(base_output).shape[-1])),
                device,
            )
            base_metric = _metric(base_output, target)
            learned_metric = _metric(learned_output, target)
            rows.append(
                {
                    "episode_id": str(sample["episode_id"]),
                    "parent_sequence": str(sample["parent_sequence"]),
                    "future_index": int(future_index),
                    "frame": int(frame),
                    "base": base_metric,
                    "learned": learned_metric,
                    "delta_box_iou_proxy": float(learned_metric["box_iou_proxy"] - base_metric["box_iou_proxy"]),
                    "delta_mask_iou": float(learned_metric["mask_iou"] - base_metric["mask_iou"]),
                    "modified_extra_norm": float(expected_extra_delta.float().norm().cpu()),
                    "writer_target_slot_residual_norm": float(residual[:, target_slot, :].float().norm().cpu()),
                }
            )
    episode_gains: dict[str, list[float]] = {}
    for row in rows:
        episode_gains.setdefault(row["episode_id"], []).append(float(row["delta_box_iou_proxy"]))
    episode_means = [float(np.mean(values)) for values in episode_gains.values()]
    base_values = [float(row["base"]["box_iou_proxy"]) for row in rows]
    learned_values = [float(row["learned"]["box_iou_proxy"]) for row in rows]
    negative = [value < 0.0 for value in episode_means]
    return {
        "status": "PASS",
        "row_count": len(rows),
        "episode_count": len(episode_means),
        "base_mean_box_iou_proxy": float(np.mean(base_values)) if base_values else None,
        "learned_mean_box_iou_proxy": float(np.mean(learned_values)) if learned_values else None,
        "mean_box_iou_gain": float(np.mean(episode_means)) if episode_means else None,
        "episode_gain_ci95": _bootstrap_ci(episode_means, seed=3000),
        "negative_transfer_episode_rate": float(np.mean(negative)) if negative else None,
        "episode_gains": [
            {"episode_id": episode_id, "mean_box_iou_gain": float(np.mean(values))}
            for episode_id, values in sorted(episode_gains.items())
        ],
        "rows": rows,
    }


def _transaction_gate() -> dict[str, Any]:
    transaction = CorrectionMemoryTransaction()
    base = torch.arange(1 * 4 * 256, dtype=torch.float32).reshape(1, 4, 256)
    residual = torch.ones_like(base)
    state1 = transaction.write_latest(
        video_id="n30-protected",
        public_id=17,
        correction_frame=3,
        residual=residual,
        gate=torch.ones((1, 1)),
    )
    past = transaction.apply_to_extra(base, state=state1, target_slot=1, frame_idx=3)
    future = transaction.apply_to_extra(base, state=state1, target_slot=1, frame_idx=4)
    state2 = transaction.write_latest(
        video_id="n30-protected",
        public_id=17,
        correction_frame=5,
        residual=2.0 * residual,
        gate=torch.ones((1, 1)),
    )
    protected = torch.equal(future[:, [0, 2, 3], :], base[:, [0, 2, 3], :])
    target_changed = torch.equal(future[:, 1, :], base[:, 1, :] + residual[:, 1, :])
    past_unchanged = torch.equal(past, base)
    latest_replaced = state2.correction_version == 2 and transaction.get_latest("n30-protected", 17) is state2
    return {
        "status": "PASS" if protected and target_changed and past_unchanged and latest_replaced else "FAIL",
        "protected_slots_unchanged": protected,
        "target_slot_changed": target_changed,
        "past_frame_unchanged": past_unchanged,
        "latest_correction_replaced": latest_replaced,
        "ledger": transaction.ledger,
    }


def _save_load_gate(writer: CorrectionMemoryWriter, inputs: CorrectionMemoryInputs, path: Path) -> dict[str, Any]:
    writer.eval()
    with torch.no_grad():
        reference = _writer_forward(writer, inputs)
    checkpoint = {
        "model": {key: value.detach().cpu() for key, value in writer.state_dict().items()},
        "architecture": writer.architecture_summary(),
        "seed": 3000,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    restored = CorrectionMemoryWriter(
        clip_dim=writer.clip_dim,
        object_embedding_dim=writer.object_embedding_dim,
        num_object_tokens=writer.num_object_tokens,
        rank=writer.rank,
    )
    saved_model = torch.load(path, map_location="cpu", weights_only=False)["model"]
    restored.load_state_dict(saved_model)
    restored.to(next(writer.parameters()).device)
    restored.eval()
    with torch.no_grad():
        candidate = _writer_forward(restored, inputs)
    max_difference = max(
        float((reference[key] - candidate[key]).abs().max().detach().cpu())
        for key in ("residual", "gate", "delta_e")
    )
    state_difference = max(
        float((writer.state_dict()[key].detach().cpu() - saved_model[key]).abs().max())
        for key in writer.state_dict()
    )
    # The comparison is made after a real disk round-trip on the same device;
    # the strict tolerance catches missing or non-restored writer state.
    return {
        "status": "PASS" if max_difference <= 1e-6 and state_difference <= 1e-7 else "FAIL",
        "max_output_difference": max_difference,
        "state_difference_after_save": state_difference,
        "tolerance": 1e-6,
        "checkpoint": str(path),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    samples, index = _load_samples(args.index, args.data_dir, args.episode_limit)
    if len(samples) != 20 and args.episode_limit is None:
        raise RuntimeError(f"N30-E overfit requires exactly 20 meta_train samples, found {len(samples)}")
    first_extra = samples[0]["support_kwargs"]["extra_per_object_embeddings"]
    if not isinstance(first_extra, Tensor) or first_extra.ndim != 3:
        raise RuntimeError(f"unexpected support extra shape: {type(first_extra).__name__}")
    num_tokens = int(first_extra.shape[1])
    clip_dim = int(np.asarray(samples[0]["clip_feature"]).reshape(-1).shape[0])
    writer = CorrectionMemoryWriter(clip_dim=clip_dim, num_object_tokens=num_tokens).to(device)
    writer_inputs = [_writer_inputs(sample, device) for sample in samples]

    backend = _make_backend(args.checkpoint)
    decoder = None
    started = time.perf_counter()
    try:
        backend._ensure_model()
        decoder = _get_official_decoder(backend)
        backend_param_count = 0
        for parameter in backend._predictor.model.parameters():
            parameter.requires_grad_(False)
            backend_param_count += int(parameter.numel())
        decoder.eval()

        # Gate 2.1: strict zero-init equivalence on a real future tape.
        writer.eval()
        zero_inputs = writer_inputs[0]
        zero_output = _writer_forward(writer, zero_inputs)
        zero_kwargs = _move_tree(samples[0]["future_kwargs"][0], device)
        base_output, _ = _decode(decoder, zero_kwargs, requires_grad=False)
        zero_learned_output, _ = _decode(
            decoder,
            zero_kwargs,
            zero_output["residual"].detach(),
            target_slot=int(samples[0].get("target_slot", 0)),
            requires_grad=False,
        )
        zero_differences = {
            key: float((_mask_logits(base_output) - _mask_logits(zero_learned_output)).abs().max().detach().cpu())
            for key in ("masks",)
        }
        zero_differences["presence"] = float(
            (_presence_logits(base_output) - _presence_logits(zero_learned_output)).abs().max().detach().cpu()
        )
        zero_gate = {
            "status": "PASS" if max(zero_differences.values()) <= 1e-6 and float(zero_output["residual"].abs().max().cpu()) == 0.0 else "FAIL",
            "max_output_differences": zero_differences,
            "initial_residual_max_abs": float(zero_output["residual"].abs().max().detach().cpu()),
            "initial_gate_mean": float(zero_output["gate"].mean().detach().cpu()),
        }

        # Gate 2.2: a backward pass is allowed to touch writer parameters only.
        writer.train()
        writer.zero_grad(set_to_none=True)
        gradient_kwargs = _move_tree(samples[0]["future_kwargs"][0], device)
        gradient_output = _writer_forward(writer, writer_inputs[0])
        gradient_decoded, gradient_extra = _decode(
            decoder,
            gradient_kwargs,
            gradient_output["residual"],
            target_slot=int(samples[0].get("target_slot", 0)),
            requires_grad=True,
        )
        target = _target_mask(
            samples[0]["future_target_boxes"][0],
            tuple(int(value) for value in samples[0]["image_size"]),
            (int(_mask_logits(gradient_decoded).shape[-2]), int(_mask_logits(gradient_decoded).shape[-1])),
            device,
        )
        gradient_loss, gradient_values, _ = _query_loss(
            gradient_decoded,
            samples[0]["future_target_boxes"][0],
            tuple(int(value) for value in samples[0]["image_size"]),
            gradient_output["residual"],
            gradient_kwargs["extra_per_object_embeddings"],
            gradient_extra,
            target_slot=int(samples[0].get("target_slot", 0)),
        )
        gradient_loss.backward()
        writer_grad_norm = float(
            torch.sqrt(sum(parameter.grad.detach().float().pow(2).sum() for parameter in writer.parameters() if parameter.grad is not None)).cpu()
        )
        decoder_grad_names = [name for name, parameter in decoder.named_parameters() if parameter.grad is not None]
        writer_grad_names = [name for name, parameter in writer.named_parameters() if parameter.grad is not None]
        gradient_gate = {
            "status": "PASS" if writer_grad_names and not decoder_grad_names and writer_grad_norm > 0.0 else "FAIL",
            "writer_gradient_parameter_count": len(writer_grad_names),
            "decoder_gradient_parameter_count": len(decoder_grad_names),
            "decoder_gradient_names": decoder_grad_names[:20],
            "writer_gradient_norm": writer_grad_norm,
            "probe_loss": gradient_values,
            "decoder_trainable_parameter_count": sum(int(p.numel()) for p in decoder.parameters() if p.requires_grad),
            "full_backend_parameter_count_frozen": backend_param_count,
        }
        writer.zero_grad(set_to_none=True)

        # Train only on support-conditioned writer inputs and strict future query frames.
        optimizer = torch.optim.AdamW(writer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_metrics_path = output_dir / "overfit_metrics.jsonl"
        train_rows: list[dict[str, Any]] = []
        with train_metrics_path.open("w", encoding="utf-8") as handle:
            for epoch in range(args.epochs):
                writer.train()
                accum: dict[str, float] = {key: 0.0 for key in ("loss_total", "loss_box", "loss_presence", "loss_protect", "loss_residual")}
                # Keep the overfit probe at one fixed strict-future frame so
                # the epoch-to-epoch loss comparison is causal and valid;
                # the post-training gate still evaluates all H20 frames.
                future_index = 0
                for sample_index, (sample, inputs) in enumerate(zip(samples, writer_inputs)):
                    optimizer.zero_grad(set_to_none=True)
                    future_kwargs = _move_tree(sample["future_kwargs"][future_index], device)
                    writer_output = _writer_forward(writer, inputs)
                    decoded, modified_extra = _decode(
                        decoder,
                        future_kwargs,
                        writer_output["residual"],
                        target_slot=int(sample.get("target_slot", 0)),
                        requires_grad=True,
                    )
                    loss, values, _ = _query_loss(
                        decoded,
                        sample["future_target_boxes"][future_index],
                        tuple(int(value) for value in sample["image_size"]),
                        writer_output["residual"],
                        future_kwargs["extra_per_object_embeddings"],
                        modified_extra,
                        target_slot=int(sample.get("target_slot", 0)),
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(writer.parameters(), max_norm=1.0)
                    optimizer.step()
                    for key in accum:
                        accum[key] += values[key]
                row = {
                    "epoch": int(epoch),
                    "future_index": int(future_index),
                    "episode_count": len(samples),
                    **{key: value / len(samples) for key, value in accum.items()},
                }
                # Re-evaluate the same first future frame after every epoch.
                # This is the valid epoch-to-epoch loss probe; the training
                # loop itself also records the per-update average above.
                writer.eval()
                with torch.no_grad():
                    probe_writer_output = _writer_forward(writer, writer_inputs[0])
                    probe_kwargs = _move_tree(samples[0]["future_kwargs"][0], device)
                    probe_decoded, probe_modified = _decode(
                        decoder,
                        probe_kwargs,
                        probe_writer_output["residual"],
                        target_slot=int(samples[0].get("target_slot", 0)),
                        requires_grad=False,
                    )
                    _, probe_values, _ = _query_loss(
                        probe_decoded,
                        samples[0]["future_target_boxes"][0],
                        tuple(int(value) for value in samples[0]["image_size"]),
                        probe_writer_output["residual"],
                        probe_kwargs["extra_per_object_embeddings"],
                        probe_modified,
                        target_slot=int(samples[0].get("target_slot", 0)),
                    )
                row["probe_loss_total"] = probe_values["loss_total"]
                row["probe_loss_box"] = probe_values["loss_box"]
                train_rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()

        transaction_gate = _transaction_gate()
        checkpoint_path = output_dir / "checkpoints" / "n30_writer_overfit.pt"
        save_load_gate = _save_load_gate(writer, writer_inputs[0], checkpoint_path)
        future_evaluation = _evaluate(writer, decoder, samples, writer_inputs, device)
        future_activation_count = sum(1 for row in future_evaluation["rows"] if row["modified_extra_norm"] > 1e-8)
        total_future_count = len(future_evaluation["rows"])
        future_activation_gate = {
            "status": "PASS" if future_activation_count == total_future_count and future_activation_count > 0 else "FAIL",
            "modified_future_calls": future_activation_count,
            "expected_future_calls": total_future_count,
            "max_modified_extra_norm": max((float(row["modified_extra_norm"]) for row in future_evaluation["rows"]), default=0.0),
            "strict_future_only": all(int(row["frame"]) > int(next(sample["correction_frame"] for sample in samples if str(sample["episode_id"]) == row["episode_id"])) for row in future_evaluation["rows"]),
        }
        loss_gate = {
            "status": "PASS" if train_rows and train_rows[-1]["probe_loss_total"] < train_rows[0]["probe_loss_total"] else "FAIL",
            "first_epoch_loss": None if not train_rows else train_rows[0]["probe_loss_total"],
            "last_epoch_loss": None if not train_rows else train_rows[-1]["probe_loss_total"],
            "epochs": len(train_rows),
        }
        metric_gain = future_evaluation.get("mean_box_iou_gain")
        overfit_metric_gate = {
            "status": "PASS" if metric_gain is not None and metric_gain >= 0.005 else "FAIL",
            "metric": "decoder-mask-derived future box IoU proxy, learned writer minus zero/write-only",
            "mean_gain": metric_gain,
            "sequence_bootstrap_ci95": future_evaluation.get("episode_gain_ci95"),
            "threshold": 0.005,
            "negative_transfer_episode_rate": future_evaluation.get("negative_transfer_episode_rate"),
        }
        gate_checks = {
            "zero_init_equivalence": zero_gate,
            "only_writer_gradients": gradient_gate,
            "query_loss_descends": loss_gate,
            "future_activation": future_activation_gate,
            "c_minus_b_overfit": overfit_metric_gate,
            "protected_identity_isolation": transaction_gate,
            "save_load_consistency": save_load_gate,
        }
        gate_status = "PASS" if all(item.get("status") == "PASS" for item in gate_checks.values()) else "FAIL"
        result = {
            "protocol": "N30-E-STRICT-FUTURE-WRITER-OVERFIT-GATE",
            "status": gate_status,
            "seed": int(args.seed),
            "data_index": str(args.index),
            "output_dir": str(output_dir),
            "dataset_index_status": index.get("status"),
            "episode_count": len(samples),
            "future_horizon": 20,
            "device": str(device),
            "checkpoint": str(args.checkpoint),
            "writer_architecture": writer.architecture_summary(),
            "frozen_decoder": True,
            "future_gt_used_for_writer_input": False,
            "val25_read": False,
            "test_labels_used": False,
            "loss_weights_frozen_before_gate": {
                "lambda_box": 1.0,
                "lambda_presence": 0.1,
                "lambda_protect": 0.1,
                "lambda_residual": 0.001,
            },
            "optimizer": {"name": "AdamW", "lr": float(args.lr), "weight_decay": float(args.weight_decay), "gradient_clip": 1.0},
            "gate_checks": gate_checks,
            "future_evaluation_summary": {key: value for key, value in future_evaluation.items() if key != "rows"},
            "artifacts": {
                "overfit_metrics": str(train_metrics_path),
                "checkpoint": str(checkpoint_path),
            },
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        _write_json(output_dir / "overfit_gate.json", result)
        _write_json(output_dir / "overfit_future_evaluation.json", future_evaluation)
        _write_json(output_dir / "overfit_gradient_probe.json", {"zero_init": zero_gate, "gradient_gate": gradient_gate})
        return result
    finally:
        backend.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3000)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: result[key] for key in ("status", "episode_count", "future_horizon", "elapsed_seconds")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
