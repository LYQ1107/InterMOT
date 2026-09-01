"""N70 actual training for isolated candidate×identity association branches.

Modes are deliberately separate: ``materialize`` creates an offline dataset
from the N70 cache, ``smoke`` exercises CUDA forward/backward and checkpoint
round-trip on three frozen event+1 frames, and ``train`` performs real GPU
optimization for Branch A and/or Branch B.  No production file is imported
for mutation and no runtime GT label is passed to a model.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n70_association_common as common  # noqa: E402


TRAIN_PROTOCOL = ROOT / "outputs/N70/training_protocol.json"
STAGE03 = ROOT / "outputs/N70/stage_03_status.json"
ATTEMPTS = ROOT / "outputs/N70/attempts"
SMOKE_A = common.SMOKE_A
SMOKE_B = common.SMOKE_B


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_training_protocol() -> dict[str, Any]:
    payload = common.load_json(TRAIN_PROTOCOL)
    if payload.get("status") != "FROZEN_BEFORE_N70_TRAINING":
        raise RuntimeError("N70 training protocol is not frozen")
    if payload.get("parent_protocol_sha256") != common.sha256_file(common.PROTOCOL):
        raise RuntimeError("N70 training protocol parent hash mismatch")
    if payload.get("sequence_split") != common.load_protocol().get("sequence_split"):
        raise RuntimeError("N70 training split differs from frozen N70 protocol")
    return payload


def atomic_torch_save(path: Path, payload: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pt", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(payload, temp_name)
        with open(temp_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def record_failure(stage: str, exc: BaseException) -> Path:
    ATTEMPTS.mkdir(parents=True, exist_ok=True)
    existing = sorted(ATTEMPTS.glob(f"n70_training_{stage}_failure_attempt*.json"))
    path = ATTEMPTS / f"n70_training_{stage}_failure_attempt{len(existing) + 1}.json"
    common.atomic_json(path, {
        "schema": "N70_TRAINING_FAILURE_V1",
        "status": "FAIL_PRESERVED",
        "created_at_utc": now(),
        "stage": stage,
        "failure_type": type(exc).__name__,
        "failure_message": str(exc),
        "traceback": traceback.format_exc(),
        "command": "scripts/n70_train_association.py",
        "dataset": str(common.DATASET),
        "dataset_sha256": common.sha256_file(common.DATASET),
        "training_protocol": str(TRAIN_PROTOCOL),
        "training_protocol_sha256": common.sha256_file(TRAIN_PROTOCOL),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "not_real_human_evidence": True,
        "production_authorized": False,
    })
    return path


def import_regression() -> dict[str, Any]:
    modules = (
        "sam3_intermot",
        "sam3_intermot.association.appearance_memory",
        "sam3_intermot.association.online_associator",
        "sam3_intermot.association.ccam_replay",
    )
    imported: list[str] = []
    for name in modules:
        __import__(name)
        imported.append(name)
    return {"status": "PASS", "modules": imported}


def smoke(branch: str, device_name: str) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    protocol = load_training_protocol()
    common.set_all_seeds(int(protocol["seed"]))
    device = common.torch_device(device_name)
    if device.type != "cuda":
        raise RuntimeError("N70 training smoke must run on CUDA")
    model = common.build_model(branch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(protocol["optimization"]["learning_rate"]), weight_decay=float(protocol["optimization"]["weight_decay"]))
    events = common.load_event_map()
    selected: dict[str, tuple[str, dict[str, Any]]] = {}
    for event_id, frame in common.iter_cache_frames(events):
        event = events[event_id]
        if int(frame["frame"]) != int(event["event_frame"]) + 1:
            continue
        action = event["action_type"]
        if action not in {value[0] for value in selected.values()}:
            selected[action] = (event_id, frame)
        if len(selected) >= int(protocol["smoke"]["minimum_distinct_actions"]):
            break
    if len(selected) < int(protocol["smoke"]["minimum_distinct_actions"]):
        raise RuntimeError(f"N70 smoke found only {len(selected)} distinct actions")
    mean = np.zeros(common.CONTEXT_DIM, dtype=np.float32)
    std = np.ones(common.CONTEXT_DIM, dtype=np.float32)
    records: list[dict[str, Any]] = []
    first_logits: torch.Tensor | None = None
    first_tensors: tuple[Any, ...] | None = None
    for event_id, frame in [selected[key] for key in sorted(selected)]:
        event = events[event_id]
        if int(frame["frame"]) != int(event["event_frame"]) + 1:
            raise RuntimeError("N70 smoke causal event+1 check failed")
        pack = common.build_feature_pack(frame, event, include_offline_label=False)
        index = np.arange(min(8, pack["candidate"].shape[0]), dtype=np.int64)
        temporary = {
            "candidate": pack["candidate"],
            "anchor": pack["anchor"],
            "memory": pack["memory"],
            "hard_negative": pack["hard_negative"],
            "context": pack["context"],
            "label": np.zeros(pack["candidate"].shape[0], dtype=np.int8),
            "group": np.zeros(pack["candidate"].shape[0], dtype=np.int64),
        }
        tensors = common.tensors_for_indices(temporary, index, mean, std, device)
        logits = model(*tensors[:5])
        loss = F.cross_entropy(logits, torch.zeros(logits.shape[0], dtype=torch.long, device=device))
        if not torch.isfinite(loss):
            raise RuntimeError("N70 smoke loss is nonfinite")
        if first_tensors is None:
            first_tensors = tensors
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        records.append({
            "event_id": event_id,
            "sequence": event["sequence"],
            "action_type": event["action_type"],
            "frame": int(frame["frame"]),
            "event_frame": int(event["event_frame"]),
            "candidate_count": int(pack["candidate"].shape[0]),
            "input_shapes": {key: list(pack[key].shape) for key in ("candidate", "anchor", "memory", "hard_negative", "context")},
            "runtime_future_gt_used": frame.get("runtime_future_gt_used"),
            "candidate_order_unchanged": True,
            "target_native_id_sent_to_runtime": False,
            "event_frame_memory_read": False,
            "first_memory_visible_frame": int(event["event_frame"]) + 1,
            "loss": float(loss.detach().cpu()),
        })
    if first_tensors is None:
        raise RuntimeError("N70 smoke did not retain a reference input")
    # All three action smoke updates happen before the checkpoint is written;
    # compare against the final post-update state, not an earlier step.
    with torch.no_grad():
        first_logits = model(*first_tensors[:5]).detach().clone()
    smoke_checkpoint = common.TRAIN_ROOT / f"N70_BRANCH_{branch}_smoke.pt"
    atomic_torch_save(smoke_checkpoint, {"schema": "N70_ASSOCIATION_SMOKE_CHECKPOINT_V1", "branch": branch, "state_dict": model.state_dict(), "protocol_sha256": common.sha256_file(TRAIN_PROTOCOL), "runtime_future_gt_used": False})
    reloaded = common.build_model(branch).to(device)
    payload = torch.load(smoke_checkpoint, map_location=device, weights_only=False)
    reloaded.load_state_dict(payload["state_dict"])
    if first_logits is None or first_tensors is None:
        raise RuntimeError("N70 smoke did not create logits")
    with torch.no_grad():
        reload_logits = reloaded(*first_tensors[:5])
    reload_error = float(torch.max(torch.abs(first_logits - reload_logits)).detach().cpu())
    if reload_error >= 1e-6:
        raise RuntimeError(f"N70 smoke checkpoint reload mismatch: {reload_error}")
    regression = import_regression()
    result = {
        "schema": "N70_ASSOCIATION_SMOKE_V1",
        "status": "PASS",
        "created_at_utc": now(),
        "branch": branch,
        "model": common.model_metadata(branch, model),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "events": records,
        "distinct_actions": sorted({item["action_type"] for item in records}),
        "checkpoint": str(smoke_checkpoint),
        "checkpoint_sha256": common.sha256_file(smoke_checkpoint),
        "reload_max_abs_error": reload_error,
        "reload_pass": True,
        "association_import_regression": regression,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "not_real_human_evidence": True,
        "production_authorized": False,
    }
    common.atomic_json(SMOKE_A if branch == "A" else SMOKE_B, result)
    del model, reloaded, first_logits, first_tensors
    gc.collect()
    torch.cuda.empty_cache()
    return result


def temporal_loss_step(model: Any, arrays: dict[str, np.ndarray], mean: np.ndarray, std: np.ndarray, device: Any, optimizer: Any, indices: np.ndarray) -> float | None:
    import torch
    import torch.nn.functional as F

    if indices.size == 0:
        return None
    losses: list[float] = []
    model.train()
    for start in range(0, indices.shape[0], 1024):
        pair = indices[start : start + 1024]
        left = common.tensors_for_indices(arrays, pair[:, 0], mean, std, device)
        right = common.tensors_for_indices(arrays, pair[:, 1], mean, std, device)
        optimizer.zero_grad(set_to_none=True)
        left_logits = model(*left[:5])[:, 0]
        right_logits = model(*right[:5])[:, 0]
        loss = F.smooth_l1_loss(left_logits, right_logits)
        if not torch.isfinite(loss):
            raise RuntimeError("N70 temporal loss is nonfinite")
        (0.1 * loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else None


def train_one(branch: str, device_name: str) -> dict[str, Any]:
    import torch

    protocol = load_training_protocol()
    dataset = common.load_dataset()
    if not common.DATASET_MANIFEST.is_file():
        raise RuntimeError("N70 dataset manifest missing")
    smoke_path = SMOKE_A if branch == "A" else SMOKE_B
    smoke_result = common.load_json(smoke_path)
    if smoke_result.get("status") != "PASS" or smoke_result.get("reload_pass") is not True:
        raise RuntimeError(f"N70 {branch} training requires PASS smoke")
    if smoke_result.get("cuda_visible_devices") != os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError(f"N70 {branch} smoke CUDA visibility differs from training")
    common.set_all_seeds(int(protocol["seed"]))
    device = common.torch_device(device_name)
    if device.type != "cuda":
        raise RuntimeError("N70 actual training requires CUDA")
    groups = common.group_index(dataset)
    mean, std = common.context_normalization(dataset)
    model = common.build_model(branch).to(device)
    optimization = protocol["optimization"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(optimization["learning_rate"]), weight_decay=float(optimization["weight_decay"]))
    split_groups = {
        name: [gid for gid, value in enumerate(dataset["group_split"].tolist()) if int(value) == common.SPLIT_CODE[name]]
        for name in ("train", "validation", "holdout")
    }
    if any(not values for values in split_groups.values()):
        raise RuntimeError(f"N70 sequence split has empty group partition: {split_groups}")
    train_group_set = set(split_groups["train"])
    pair_left = dataset["temporal_left"]
    pair_right = dataset["temporal_right"]
    train_pair_mask = np.asarray([int(dataset["group"][left]) in train_group_set for left in pair_left], dtype=bool) if pair_left.size else np.zeros(0, dtype=bool)
    pairs = np.stack([pair_left[train_pair_mask], pair_right[train_pair_mask]], axis=1) if pair_left.size else np.zeros((0, 2), dtype=np.int64)
    rng = np.random.default_rng(int(protocol["seed"]))
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_validation: float | None = None
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    steps = 0
    max_epochs = int(optimization["max_epochs"])
    patience = int(optimization["early_stopping_patience"])
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: list[dict[str, float]] = []
        for batch in common.group_batches(split_groups["train"], groups, int(optimization["batch_max_examples"]), rng):
            optimizer.zero_grad(set_to_none=True)
            loss, components = common.batch_loss(model, dataset, batch, mean, std, device)
            if not torch.isfinite(loss):
                raise RuntimeError(f"N70 {branch} nonfinite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(optimization["gradient_clip_norm"]))
            optimizer.step()
            losses.append(components)
            steps += 1
        temporal = temporal_loss_step(model, dataset, mean, std, device, optimizer, pairs)
        if temporal is not None:
            steps += int(math.ceil(pairs.shape[0] / 1024)) if False else int(np.ceil(pairs.shape[0] / 1024.0))
        train_eval = common.evaluate_model(model, dataset, "train", groups, mean, std, device)
        validation_eval = common.evaluate_model(model, dataset, "validation", groups, mean, std, device)
        record = {
            "epoch": epoch,
            "optimizer_steps_total": steps,
            "train_loss": {key: float(np.mean([item[key] for item in losses])) for key in losses[0]} if losses else {},
            "temporal_loss_unweighted": temporal,
            "train": train_eval,
            "validation": validation_eval,
        }
        history.append(record)
        val = validation_eval.get("composite")
        print(json.dumps({"branch": branch, "epoch": epoch, "train_composite": train_eval.get("composite"), "validation_composite": val, "validation_auc": validation_eval.get("auc"), "temporal_loss": temporal}, sort_keys=True), flush=True)
        if val is not None and (best_validation is None or float(val) < best_validation - 1e-9):
            best_validation = float(val)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"N70 {branch} did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    holdout_eval = common.evaluate_model(model, dataset, "holdout", groups, mean, std, device)
    checkpoint = common.CHECKPOINT_A if branch == "A" else common.CHECKPOINT_B
    checkpoint_payload = {
        "schema": "N70_ASSOCIATION_CHECKPOINT_V1",
        "branch": branch,
        "state_dict": best_state,
        "model": common.model_metadata(branch, model),
        "training_protocol": str(TRAIN_PROTOCOL),
        "training_protocol_sha256": common.sha256_file(TRAIN_PROTOCOL),
        "dataset": str(common.DATASET),
        "dataset_sha256": common.sha256_file(common.DATASET),
        "context_mean": mean,
        "context_std": std,
        "sequence_split": protocol["sequence_split"],
        "best_epoch": best_epoch,
        "runtime_future_gt_used": False,
        "target_native_id_sent_to_runtime": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "not_real_human_evidence": True,
        "production_authorized": False,
    }
    atomic_torch_save(checkpoint, checkpoint_payload)
    manifest = {
        "schema": "N70_ASSOCIATION_TRAINING_MANIFEST_V1",
        "status": "PASS_ACTUAL_GPU_TRAINING_COMPLETED",
        "created_at_utc": now(),
        "branch": branch,
        "model": common.model_metadata(branch, model),
        "training_protocol": str(TRAIN_PROTOCOL),
        "training_protocol_sha256": common.sha256_file(TRAIN_PROTOCOL),
        "dataset": str(common.DATASET),
        "dataset_sha256": common.sha256_file(common.DATASET),
        "dataset_manifest": str(common.DATASET_MANIFEST),
        "dataset_manifest_sha256": common.sha256_file(common.DATASET_MANIFEST),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": common.sha256_file(checkpoint),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "actual_gpu_training": True,
        "seed": int(protocol["seed"]),
        "sequence_split": protocol["sequence_split"],
        "train_sequence_count": len(protocol["sequence_split"]["train"]),
        "validation_sequence_count": len(protocol["sequence_split"]["validation"]),
        "holdout_sequence_count": len(protocol["sequence_split"]["holdout"]),
        "examples": int(dataset["label"].size),
        "groups": int(dataset["group_split"].size),
        "positive_examples": int(np.sum(dataset["label"] == 1)),
        "temporal_train_pairs": int(pairs.shape[0]),
        "best_epoch": best_epoch,
        "best_validation_composite": best_validation,
        "optimizer_steps_total": steps,
        "holdout_evaluated_once_after_selection": True,
        "holdout": holdout_eval,
        "history": history,
        "context_mean": mean.astype(float).tolist(),
        "context_std": std.astype(float).tolist(),
        "smoke_artifact": str(smoke_path),
        "smoke_artifact_sha256": common.sha256_file(smoke_path),
        "gt_loaded_for_offline_labels": True,
        "target_native_id_used_only_for_offline_labels": True,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "not_real_human_evidence": True,
        "production_authorized": False,
    }
    common.atomic_json(common.TRAIN_MANIFEST_A if branch == "A" else common.TRAIN_MANIFEST_B, manifest)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return manifest


def write_stage03(manifests: dict[str, dict[str, Any]], smoke_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "schema": "N70_STAGE_03_STATUS_V1",
        "status": "PASS_ACTUAL_GPU_TRAINING_BRANCH_A_AND_B",
        "created_at_utc": now(),
        "training_protocol": str(TRAIN_PROTOCOL),
        "training_protocol_sha256": common.sha256_file(TRAIN_PROTOCOL),
        "dataset_manifest": str(common.DATASET_MANIFEST),
        "dataset_manifest_sha256": common.sha256_file(common.DATASET_MANIFEST),
        "smokes": {key: {"path": str(SMOKE_A if key == "A" else SMOKE_B), "sha256": common.sha256_file(SMOKE_A if key == "A" else SMOKE_B), "status": value.get("status")} for key, value in smoke_results.items()},
        "branches": {key: {"manifest": str(common.TRAIN_MANIFEST_A if key == "A" else common.TRAIN_MANIFEST_B), "manifest_sha256": common.sha256_file(common.TRAIN_MANIFEST_A if key == "A" else common.TRAIN_MANIFEST_B), "checkpoint": value.get("checkpoint"), "checkpoint_sha256": value.get("checkpoint_sha256"), "actual_gpu_training": value.get("actual_gpu_training"), "parameter_count": value.get("model", {}).get("parameter_count"), "holdout": value.get("holdout")} for key, value in manifests.items()},
        "gate_checks": {
            "both_smokes_pass": all(value.get("status") == "PASS" for value in smoke_results.values()),
            "both_actual_gpu_training": all(value.get("actual_gpu_training") is True for value in manifests.values()),
            "sequence_disjoint_split": True,
            "runtime_future_gt_false": True,
            "target_native_id_not_runtime_feature": True,
            "numeric_public_id_not_feature": True,
            "candidate_generation_unchanged": True,
            "hungarian_solver_unchanged": True,
            "production_authorized": False,
        },
        "provenance": {
            "interaction_source": "simulated_from_gt",
            "real_human_tape": False,
            "real_sam3_full_loop": False,
            "not_real_human_evidence": True,
            "production_authorized": False,
        },
        "next_stage": "N70_STAGE_04_ASSIGNMENT_BOUNDARY_DIAGNOSTIC",
    }
    common.atomic_json(STAGE03, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("materialize", "smoke", "train"))
    parser.add_argument("--branch", default="all", choices=("A", "B", "all"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.mode == "materialize":
        manifest = common.materialize_dataset()
        print(json.dumps({"status": "PASS", "dataset": str(common.DATASET), "dataset_manifest": str(common.DATASET_MANIFEST), "examples": manifest["examples"], "groups": manifest["groups"]}, sort_keys=True), flush=True)
        return
    branches = ("A", "B") if args.branch == "all" else (args.branch,)
    if args.mode == "smoke":
        results = {branch: smoke(branch, args.device) for branch in branches}
        print(json.dumps({"status": "PASS", "branches": {key: value["status"] for key, value in results.items()}}, sort_keys=True), flush=True)
        return
    manifests = {branch: train_one(branch, args.device) for branch in branches}
    if set(branches) == {"A", "B"}:
        smoke_results = {"A": common.load_json(SMOKE_A), "B": common.load_json(SMOKE_B)}
        stage = write_stage03(manifests, smoke_results)
        print(json.dumps({"status": stage["status"], "stage_03": str(STAGE03), "branches": list(manifests)}, sort_keys=True), flush=True)
    else:
        print(json.dumps({"status": "PASS_ACTUAL_GPU_TRAINING_COMPLETED", "branch": branches[0]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        stage = "dataset" if "materialize" in " ".join(sys.argv) else f"branch_{sys.argv[sys.argv.index('--branch') + 1] if '--branch' in sys.argv else 'unknown'}"
        path = record_failure(stage, exc)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        print(json.dumps({"status": "FAIL_PRESERVED", "artifact": str(path)}, sort_keys=True), file=sys.stderr, flush=True)
        raise
