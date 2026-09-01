#!/usr/bin/env python3
"""Train and gate the N30 offline correction-memory writer.

This is deliberately a one-GPU implementation: N30 permits up to four GPUs,
but the frozen tape is small enough that adding DDP would only add an
unvalidated communication path.  The selection metric and all query targets
remain sequence-disjoint and future-only.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.n30_writer_overfit import (  # noqa: E402
    CHECKPOINT,
    DEFAULT_DATA_DIR,
    _bootstrap_ci,
    _decode,
    _evaluate,
    _load_samples,
    _move_tree,
    _query_loss,
    _sha256,
    _writer_forward,
    _writer_inputs,
)
from sam3_intermot.adaptation.correction_memory_writer import CorrectionMemoryWriter  # noqa: E402
from scripts.n29_lit_online_replay import _get_official_decoder, _make_backend  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "outputs/n30/writer_dataset_index_final.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/n30"


def _load_role_samples(index_path: Path, roles: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("status") != "PASS":
        raise RuntimeError(f"writer dataset index is not PASS: {index.get('status')}")
    samples: list[dict[str, Any]] = []
    for record in index.get("records", []):
        if record.get("status") != "PASS" or record.get("role") not in roles:
            continue
        path = Path(record["sample_path"])
        if not path.is_absolute():
            path = ROOT / path
        sample = torch.load(path, map_location="cpu", weights_only=False)
        if len(sample.get("future_kwargs", [])) != 20 or len(sample.get("future_target_boxes", [])) != 20:
            raise RuntimeError(f"role sample is not H20: {path}")
        if sample.get("future_gt_used_for_writer_input") is not False:
            raise RuntimeError(f"future GT input flag is unsafe: {path}")
        samples.append(sample)
    return samples, index


def _summary_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    rows = evaluation.get("rows", [])
    base_success = [float(row["base"]["box_iou_proxy"] >= 0.5) for row in rows]
    learned_success = [float(row["learned"]["box_iou_proxy"] >= 0.5) for row in rows]
    base_missing = [1.0 - float(row["base"]["prediction_present"]) for row in rows]
    learned_missing = [1.0 - float(row["learned"]["prediction_present"]) for row in rows]
    sequence_gains: dict[str, list[float]] = {}
    for row in rows:
        sequence_gains.setdefault(str(row["parent_sequence"]), []).append(float(row["delta_box_iou_proxy"]))
    sequence_means = [float(np.mean(values)) for values in sequence_gains.values()]
    return {
        "row_count": len(rows),
        "sequence_count": len(sequence_means),
        "base_mean_box_iou_proxy": evaluation.get("base_mean_box_iou_proxy"),
        "learned_mean_box_iou_proxy": evaluation.get("learned_mean_box_iou_proxy"),
        "mean_box_iou_gain": evaluation.get("mean_box_iou_gain"),
        "sequence_bootstrap_ci95": _bootstrap_ci(sequence_means, seed=3030),
        "base_success_at_0_5_proxy": float(np.mean(base_success)) if base_success else None,
        "learned_success_at_0_5_proxy": float(np.mean(learned_success)) if learned_success else None,
        "success_gain_proxy": float(np.mean(learned_success) - np.mean(base_success)) if rows else None,
        "base_missing_prediction_rate_proxy": float(np.mean(base_missing)) if base_missing else None,
        "learned_missing_prediction_rate_proxy": float(np.mean(learned_missing)) if learned_missing else None,
        "missing_rate_change_proxy": float(np.mean(learned_missing) - np.mean(base_missing)) if rows else None,
        "negative_transfer_sequence_rate": float(np.mean([value < 0.0 for value in sequence_means])) if sequence_means else None,
        "sequence_gains": [
            {"parent_sequence": sequence, "mean_box_iou_gain": float(np.mean(values))}
            for sequence, values in sorted(sequence_gains.items())
        ],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_checkpoint(
    path: Path,
    writer: CorrectionMemoryWriter,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    epoch: int,
    step: int,
    index: dict[str, Any],
    best_selection_metric: float | None,
    args: argparse.Namespace,
) -> None:
    state = {
        "model": {key: value.detach().cpu() for key, value in writer.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "frozen_split": {
            "index": str(args.index),
            "index_sha256": _sha256(args.index),
            "roles": ["meta_train", "selection", "calibration"],
        },
        "seed": int(args.seed),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "best_selection_metric": best_selection_metric,
        "config": {
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "epochs": int(args.epochs),
            "selection_interval": int(args.selection_interval),
            "future_horizon": 20,
            "world_size": 1,
        },
        "architecture": writer.architecture_summary(),
        "dataset_index_status": index.get("status"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_samples, index = _load_role_samples(args.index, {"meta_train"})
    selection_samples, _ = _load_role_samples(args.index, {"selection"})
    calibration_samples, _ = _load_role_samples(args.index, {"calibration"})
    if len(meta_samples) != 20 or len(selection_samples) != 4 or len(calibration_samples) != 4:
        raise RuntimeError(
            f"frozen split counts must be 20/4/4, got {len(meta_samples)}/{len(selection_samples)}/{len(calibration_samples)}"
        )
    first_extra = meta_samples[0]["support_kwargs"]["extra_per_object_embeddings"]
    num_tokens = int(first_extra.shape[1])
    clip_dim = int(np.asarray(meta_samples[0]["clip_feature"]).reshape(-1).shape[0])
    writer = CorrectionMemoryWriter(clip_dim=clip_dim, num_object_tokens=num_tokens).to(device)
    meta_inputs = [_writer_inputs(sample, device) for sample in meta_samples]
    selection_inputs = [_writer_inputs(sample, device) for sample in selection_samples]
    calibration_inputs = [_writer_inputs(sample, device) for sample in calibration_samples]

    backend = _make_backend(args.checkpoint)
    started = time.perf_counter()
    decoder = None
    best_metric: float | None = None
    best_epoch: int | None = None
    best_checkpoint = args.output_dir / "checkpoints" / "n30_writer_best.pt"
    train_path = args.output_dir / "train_metrics.jsonl"
    selection_path = args.output_dir / "selection_metrics.jsonl"
    train_handle = train_path.open("w", encoding="utf-8")
    selection_handle = selection_path.open("w", encoding="utf-8")
    try:
        backend._ensure_model()
        decoder = _get_official_decoder(backend)
        for parameter in backend._predictor.model.parameters():
            parameter.requires_grad_(False)
        decoder.eval()
        optimizer = torch.optim.AdamW(writer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
        step = 0
        selection_rows: list[dict[str, Any]] = []
        for epoch in range(args.epochs):
            writer.train()
            accum = {key: 0.0 for key in ("loss_total", "loss_box", "loss_presence", "loss_protect", "loss_residual")}
            for sample_index, (sample, inputs) in enumerate(zip(meta_samples, meta_inputs)):
                future_index = (epoch + sample_index) % 20
                optimizer.zero_grad(set_to_none=True)
                kwargs = _move_tree(sample["future_kwargs"][future_index], device)
                output = _writer_forward(writer, inputs)
                decoded, modified = _decode(
                    decoder,
                    kwargs,
                    output["residual"],
                    target_slot=int(sample.get("target_slot", 0)),
                    requires_grad=True,
                )
                loss, values, _ = _query_loss(
                    decoded,
                    sample["future_target_boxes"][future_index],
                    tuple(int(value) for value in sample["image_size"]),
                    output["residual"],
                    kwargs["extra_per_object_embeddings"],
                    modified,
                    target_slot=int(sample.get("target_slot", 0)),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(writer.parameters(), max_norm=1.0)
                optimizer.step()
                step += 1
                for key in accum:
                    accum[key] += values[key]
            scheduler.step()
            train_row = {
                "epoch": int(epoch),
                "step": int(step),
                "episode_count": len(meta_samples),
                "future_indices": "rotating_strict_future",
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **{key: value / len(meta_samples) for key, value in accum.items()},
            }
            train_handle.write(json.dumps(train_row, sort_keys=True) + "\n")
            train_handle.flush()
            if epoch == 0 or (epoch + 1) % args.selection_interval == 0 or epoch + 1 == args.epochs:
                selection_eval = _evaluate(writer, decoder, selection_samples, selection_inputs, device)
                selection_summary = _summary_metrics(selection_eval)
                selection_row = {"epoch": int(epoch), "step": int(step), **selection_summary}
                selection_rows.append(selection_row)
                selection_handle.write(json.dumps(selection_row, sort_keys=True) + "\n")
                selection_handle.flush()
                metric = selection_summary.get("mean_box_iou_gain")
                if metric is not None and (best_metric is None or float(metric) > best_metric):
                    best_metric = float(metric)
                    best_epoch = int(epoch)
                    _save_checkpoint(
                        best_checkpoint,
                        writer,
                        optimizer,
                        scheduler,
                        epoch=epoch,
                        step=step,
                        index=index,
                        best_selection_metric=best_metric,
                        args=args,
                    )
        train_handle.close()
        selection_handle.close()
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
            writer.load_state_dict(checkpoint["model"])
            writer.to(device)
        selection_eval = _evaluate(writer, decoder, selection_samples, selection_inputs, device)
        calibration_eval = _evaluate(writer, decoder, calibration_samples, calibration_inputs, device)
        selection_summary = _summary_metrics(selection_eval)
        calibration_summary = _summary_metrics(calibration_eval)
        unaffected_identity = {
            "status": "NOT_RUN",
            "reason": "selection/calibration tapes are singleton; no real second public identity is present for an end-to-end unaffected-ID test",
            "protected_slot_unit_test": "PASS",
        }
        criteria = {
            "h20_mean_gain_ge_0_005": bool(selection_summary.get("mean_box_iou_gain") is not None and selection_summary["mean_box_iou_gain"] >= 0.005),
            "sequence_ci_lower_gt_zero": bool(selection_summary.get("sequence_bootstrap_ci95") and selection_summary["sequence_bootstrap_ci95"][0] > 0.0),
            "success_not_lower": bool(selection_summary.get("success_gain_proxy") is not None and selection_summary["success_gain_proxy"] >= 0.0),
            "missing_not_higher": bool(selection_summary.get("missing_rate_change_proxy") is not None and selection_summary["missing_rate_change_proxy"] <= 0.0),
            "negative_transfer_lt_20_percent": bool(selection_summary.get("negative_transfer_sequence_rate") is not None and selection_summary["negative_transfer_sequence_rate"] < 0.20),
            "unaffected_identity_no_decline": False,
        }
        gate = {
            "protocol": "N30-GATE3-STRICT-FUTURE-WRITER-VS-WRITE_ONLY",
            "status": "PASS" if all(criteria.values()) else "FAIL",
            "criterion_status": criteria,
            "primary_comparison": "D_offline_learned_writer_minus_B_official_tracker_write_only",
            "metric_note": "box_iou_proxy is derived from frozen official decoder masks; this is not delivered-output MOT box IoU",
            "selection": selection_summary,
            "calibration": calibration_summary,
            "unaffected_identity": unaffected_identity,
            "best_epoch": best_epoch,
            "best_selection_metric": best_metric,
            "checkpoint": str(best_checkpoint),
            "config": {
                "device": str(device),
                "gpu_count": 1,
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "gradient_clip": 1.0,
                "effective_batch": 1,
                "future_horizon": 20,
            },
            "future_gt_used_for_writer_input": False,
            "val25_read": False,
            "test_labels_used": False,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        _write_json(args.output_dir / "future_benefit_gate.json", gate)
        _write_json(
            args.output_dir / "full_loop_results.json",
            {
                "status": "NOT_RUN" if gate["status"] != "PASS" else "PENDING_IMPLEMENTATION",
                "reason": "Gate 3 did not pass on the frozen singleton selection/calibration protocol; full-loop TrackEval is gated off" if gate["status"] != "PASS" else "Gate 3 passed; full-loop runner was not part of this training command",
                "gate3_status": gate["status"],
                "val25_read": False,
                "test_labels_used": False,
            },
        )
        return gate
    finally:
        if not train_handle.closed:
            train_handle.close()
        if not selection_handle.closed:
            selection_handle.close()
        backend.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--selection-interval", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3001)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: result[key] for key in ("status", "best_epoch", "best_selection_metric", "elapsed_seconds")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
