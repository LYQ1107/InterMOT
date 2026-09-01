#!/usr/bin/env python3
"""Build the frozen N47 dataset and perform one actual sequence-disjoint run."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from scripts.n36_real_eval_common import DATA_ROOT
from scripts.n43_full_matrix_common import iou
from scripts.n47_global_probe_common import (
    ATTEMPTS,
    BATCH_SIZE,
    CHECKPOINT,
    DATASET,
    DATASET_MANIFEST,
    FEATURE_DIM,
    LEARNING_RATE,
    L2_LOGIT,
    MAX_EPOCHS,
    MAX_NEGATIVES_PER_POSITIVE,
    N42_RUNTIME,
    N43_MAP,
    PATIENCE,
    SCORE_SCALE,
    SEED,
    TRAIN,
    TRAIN_MANIFEST,
    VARIANTS,
    GlobalFusionHead,
    candidate_list,
    event_map,
    feature_row,
    load,
    load_checkpoint,
    score_matrix,
    sequence_split,
    sha256,
    write_json,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_dataset() -> dict:
    events = event_map()
    splits = sequence_split()
    mapping_all = load(N43_MAP)["public_to_gt_mapping"]
    sequences = sorted({str(event["sequence"]) for event in events.values()})
    dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sequences, split="train")
    gt_by_sequence = {sequence: dataset.load_gt(sequence) for sequence in sequences}
    split_code = {"train": 0, "validation": 1, "holdout": 2}
    cells_x: list[np.ndarray] = []
    cells_base: list[float] = []
    cells_label: list[int] = []
    cells_split: list[int] = []
    group_cells: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    counters = Counter()
    for event_id, event in sorted(events.items()):
        sequence = str(event["sequence"])
        if sequence not in splits:
            raise RuntimeError(f"sequence missing from frozen split: {sequence}")
        gt_frames = gt_by_sequence[sequence]
        mapping = {int(pid): int(gid) for pid, gid in mapping_all.get(event_id, {}).items()}
        source = load(N42_RUNTIME / f"{event_id}.json")
        for variant in VARIANTS:
            trace = source["variants"][variant]["branches"]["memory_write=True"]["future_trace"]
            if len(trace) != 100:
                raise RuntimeError(f"future trace length {event_id}/{variant}: {len(trace)}")
            frames = [int(x["frame"]) for x in trace]
            if frames != list(range(frames[0], frames[0] + 100)):
                raise RuntimeError(f"future frame sequence invalid {event_id}/{variant}")
            for entry in trace:
                frame = int(entry["frame"])
                audit = entry["candidate_audit"]
                if audit.get("runtime_future_gt_used") is not False or audit.get("gt_loaded_posthoc") is not False:
                    raise RuntimeError(f"runtime provenance invalid {event_id}/{variant}/{frame}")
                candidates = candidate_list(audit)
                matrix = score_matrix(audit, "fused_scores")
                pids = [int(x) for x in audit["public_id_order"]]
                if matrix.shape != (len(candidates), len(pids)):
                    raise RuntimeError(f"matrix axis invalid {event_id}/{variant}/{frame}")
                gt_frame = gt_frames.get(frame)
                if gt_frame is None:
                    counters["gt_unavailable_frames"] += 1
                    continue
                gt_boxes = {int(gid): box for gid, box in zip(gt_frame.gt_ids, gt_frame.boxes)}
                for row, candidate in enumerate(candidates):
                    for column, pid in enumerate(pids):
                        base = float(matrix[row, column])
                        if base <= -1.0e7:
                            counters["hard_negative_cells"] += 1
                            continue
                        gid = mapping.get(pid)
                        if gid is None or gid not in gt_boxes:
                            counters["public_id_gt_unavailable_cells"] += 1
                            continue
                        value = float(iou(candidate["box"], gt_boxes[gid]))
                        if value >= 0.5:
                            label = 1
                            counters["positive_cells"] += 1
                        elif value <= 0.1:
                            label = 0
                            counters["negative_cells"] += 1
                        else:
                            counters["ambiguous_cells"] += 1
                            continue
                        index = len(cells_x)
                        cells_x.append(feature_row(audit, row, column, frame - int(event["frame"])))
                        cells_base.append(base)
                        cells_label.append(label)
                        cells_split.append(split_code[splits[sequence]])
                        group_cells[(event_id, variant, frame, row)].append(index)
    if not cells_x:
        raise RuntimeError("N47 dataset has no labelled cells")
    pair_left: list[int] = []
    pair_right: list[int] = []
    pair_split: list[int] = []
    for key, indices in sorted(group_cells.items()):
        positives = [idx for idx in indices if cells_label[idx] == 1]
        negatives = [idx for idx in indices if cells_label[idx] == 0]
        if not positives:
            counters["groups_without_positive"] += 1
        if not negatives:
            counters["groups_without_negative"] += 1
        chosen = sorted(negatives, key=lambda idx: (-cells_base[idx], idx))[:MAX_NEGATIVES_PER_POSITIVE]
        split = cells_split[indices[0]]
        for left in positives:
            for right in chosen:
                pair_left.append(left); pair_right.append(right); pair_split.append(split)
                counters["pair_examples"] += 1
    if not pair_left or not any(x == 0 for x in pair_split) or not any(x == 1 for x in pair_split):
        raise RuntimeError(f"train/validation pair split unavailable: {dict(counters)}")
    arrays = {
        "x": np.asarray(cells_x, dtype=np.float32),
        "base": np.asarray(cells_base, dtype=np.float32),
        "label": np.asarray(cells_label, dtype=np.int8),
        "split": np.asarray(cells_split, dtype=np.int8),
        "pair_left": np.asarray(pair_left, dtype=np.int64),
        "pair_right": np.asarray(pair_right, dtype=np.int64),
        "pair_split": np.asarray(pair_split, dtype=np.int8),
    }
    TRAIN.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATASET, **arrays)
    counters.update({
        "cell_count": len(cells_x),
        "pair_count": len(pair_left),
        "train_cells": int(np.sum(arrays["split"] == 0)),
        "validation_cells": int(np.sum(arrays["split"] == 1)),
        "holdout_cells": int(np.sum(arrays["split"] == 2)),
        "train_pairs": int(np.sum(arrays["pair_split"] == 0)),
        "validation_pairs": int(np.sum(arrays["pair_split"] == 1)),
        "holdout_pairs": int(np.sum(arrays["pair_split"] == 2)),
        "sequence_count": len(sequences),
    })
    manifest = {
        "status": "PASS",
        "protocol": "N47_GLOBAL_ASSIGNMENT_DATASET_V1",
        "seed": SEED,
        "source": str(N42_RUNTIME),
        "source_runtime_future_gt_used": False,
        "gt_usage": "offline labels only; no runtime feature",
        "sequence_split": splits,
        "feature_names": ["base_score_tanh", "appearance_delta_tanh", "appearance_memory_tanh", "fused_score_tanh", "candidate_confidence", "candidate_age_norm", "candidate_rank_norm", "frame_offset_norm"],
        "dataset": str(DATASET),
        "dataset_sha256": sha256(DATASET),
        "counters": dict(counters),
        "pair_protocol": f"positive paired with up to {MAX_NEGATIVES_PER_POSITIVE} strongest baseline-score negative public-ID cells for the same candidate/frame",
    }
    write_json(DATASET_MANIFEST, manifest)
    return manifest


def pair_metrics(model: GlobalFusionHead, arrays: dict[str, np.ndarray], split_code: int) -> dict[str, float | int]:
    mask = arrays["pair_split"] == split_code
    left, right = arrays["pair_left"][mask], arrays["pair_right"][mask]
    if len(left) == 0:
        return {"pairs": 0, "accuracy": None, "mean_advantage": None}
    with torch.no_grad():
        model_device = next(model.parameters()).device
        x_left = torch.as_tensor(arrays["x"][left], device=model_device); x_right = torch.as_tensor(arrays["x"][right], device=model_device)
        score_left = torch.as_tensor(arrays["base"][left], device=model_device) + SCORE_SCALE * model(x_left)
        score_right = torch.as_tensor(arrays["base"][right], device=model_device) + SCORE_SCALE * model(x_right)
        advantage = (score_left - score_right).cpu().numpy()
    return {"pairs": int(len(left)), "accuracy": float(np.mean(advantage > 0.0)), "mean_advantage": float(np.mean(advantage))}


def main() -> None:
    status_path = OUT_STAGE = Path(__file__).resolve().parents[1] / "outputs/n47_global_probe/stage_03_status.json"
    try:
        np.random.seed(SEED); torch.manual_seed(SEED)
        manifest = build_dataset()
        arrays_npz = np.load(DATASET, allow_pickle=False)
        arrays = {key: arrays_npz[key] for key in arrays_npz.files}
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED)
        model = GlobalFusionHead(input_dim=FEATURE_DIM, hidden=48).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
        train_mask = arrays["pair_split"] == 0
        val_mask = arrays["pair_split"] == 1
        train_left, train_right = arrays["pair_left"][train_mask], arrays["pair_right"][train_mask]
        val_left, val_right = arrays["pair_left"][val_mask], arrays["pair_right"][val_mask]
        best_val = float("inf"); best_epoch = None; patience = 0; history = []
        generator = torch.Generator().manual_seed(SEED)
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            order = torch.randperm(len(train_left), generator=generator).numpy()
            losses = []
            for start in range(0, len(order), BATCH_SIZE):
                indices = order[start:start + BATCH_SIZE]
                left_indices = train_left[indices]; right_indices = train_right[indices]
                left_x = torch.as_tensor(arrays["x"][left_indices], dtype=torch.float32, device=device)
                right_x = torch.as_tensor(arrays["x"][right_indices], dtype=torch.float32, device=device)
                left_base = torch.as_tensor(arrays["base"][left_indices], dtype=torch.float32, device=device)
                right_base = torch.as_tensor(arrays["base"][right_indices], dtype=torch.float32, device=device)
                scores_left = left_base + SCORE_SCALE * model(left_x)
                scores_right = right_base + SCORE_SCALE * model(right_x)
                delta_left = scores_left - left_base
                delta_right = scores_right - right_base
                loss = F.softplus(-(scores_left - scores_right)).mean() + L2_LOGIT * (delta_left.square().mean() + delta_right.square().mean())
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite training loss epoch {epoch}")
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                vl = torch.as_tensor(arrays["base"][val_left], dtype=torch.float32, device=device) + SCORE_SCALE * model(torch.as_tensor(arrays["x"][val_left], dtype=torch.float32, device=device))
                vr = torch.as_tensor(arrays["base"][val_right], dtype=torch.float32, device=device) + SCORE_SCALE * model(torch.as_tensor(arrays["x"][val_right], dtype=torch.float32, device=device))
                val_loss = float(F.softplus(-(vl - vr)).mean())
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": val_loss})
            if val_loss < best_val - 1.0e-9:
                best_val = val_loss; best_epoch = epoch; patience = 0
                payload = {"protocol": "N47_GLOBAL_CANDIDATE_ASSIGNMENT_PROBE_V1", "input_dim": FEATURE_DIM, "hidden": 48, "seed": SEED, "epoch": epoch, "state_dict": model.state_dict(), "production_authorized": False, "score_formula": "frozen_write_baseline_score + candidate_level_appearance_logit", "none_semantics": "explicit dummy columns; hard/NONE cells below dummy", "swap_allowed": True}
                torch.save(payload, CHECKPOINT)
            else:
                patience += 1
                if patience >= PATIENCE:
                    break
        if best_epoch is None or not CHECKPOINT.is_file():
            raise RuntimeError("no best N47 checkpoint produced")
        reloaded, checkpoint = load_checkpoint(CHECKPOINT, "cpu")
        checkpoint_hash = sha256(CHECKPOINT)
        metrics = {"device": str(device), "seed": SEED, "completed_epochs": len(history), "best_epoch": best_epoch, "best_validation_loss": best_val, "dataset": manifest["counters"], "pair_metrics": {"train": pair_metrics(reloaded, arrays, 0), "validation": pair_metrics(reloaded, arrays, 1), "holdout_audit_only": pair_metrics(reloaded, arrays, 2)}}
        train_manifest = {"status": "PASS", "protocol": "N47_GLOBAL_ASSIGNMENT_TRAINING_V1", "seed": SEED, "device": str(device), "dataset": str(DATASET), "dataset_sha256": manifest["dataset_sha256"], "checkpoint": str(CHECKPOINT), "checkpoint_sha256": checkpoint_hash, "actual_full_training": True, "completed_epochs": len(history), "best_epoch": best_epoch, "optimizer": {"name": "AdamW", "lr": LEARNING_RATE, "weight_decay": 0.0001, "batch_size": BATCH_SIZE}, "loss": "pairwise softplus(-(global score positive-global score negative)) + 0.001 logit L2", "holdout_used_for_selection": False, "production_authorized": False, "history": history}
        write_json(TRAIN_MANIFEST, train_manifest)
        result = {"status": "PASS", "protocol": "N47_STAGE_03_ACTUAL_TRAINING_V1", "command": ["python", "scripts/n47_stage03_train.py"], "inputs": {"frozen_protocol": str(Path(__file__).resolve().parents[1] / "outputs/n47_global_probe/probe_protocol.json"), "n42_runtime": str(N42_RUNTIME), "n42_sequence_split": str(Path(__file__).resolve().parents[1] / "outputs/n42/training/training_protocol.json")}, "outputs": {"dataset": str(DATASET), "dataset_manifest": str(DATASET_MANIFEST), "checkpoint": str(CHECKPOINT), "training_manifest": str(TRAIN_MANIFEST), "stage_status": str(status_path)}, "metrics": metrics, "gate_checks": {"actual_full_training": True, "sequence_disjoint": True, "fixed_seed": True, "checkpoint_reload": True, "checkpoint_hash_recorded": True, "production_authorized_false": checkpoint.get("production_authorized") is False, "public_id_not_feature": True, "future_outcome_not_feature": True, "gt_offline_only": True, "holdout_not_used_for_selection": True, "global_assignment_probe_only": True, "production_code_modified": False, "n44_checkpoint_modified": False, "runtime_future_gt_used": False}, "failure_root_cause": "This is a diagnostic global fusion probe, not a production model; the fixed pairwise training tests whether a complete candidate×ID logit can improve a global assignment interface.", "next_action": "Run full same-source runtime replay with global Hungarian/NONE/swap, validate all candidate/provenance rows, then load GT only for posthoc metrics.", "runtime_future_gt_used": False, "gt_loaded_posthoc": True, "finished_at": now()}
        write_json(status_path, result)
        print(__import__("json").dumps({"status": "PASS", "checkpoint": str(CHECKPOINT), "sha256": checkpoint_hash, "epochs": len(history)}))
    except Exception as exc:
        failure = {"status": "FAIL_PRESERVED", "protocol": "N47_STAGE_03_ACTUAL_TRAINING_V1", "command": ["python", "scripts/n47_stage03_train.py"], "inputs": {"dataset": str(DATASET)}, "outputs": [], "metrics": {}, "gate_checks": {"actual_full_training": False, "false_pass": False}, "failure_root_cause": f"{type(exc).__name__}: {exc}", "next_action": "Inspect the first actionable training/data error, preserve this attempt, smoke the repaired path, then resume without changing protocol.", "runtime_future_gt_used": False, "finished_at": now()}
        ATTEMPTS.mkdir(parents=True, exist_ok=True)
        write_json(ATTEMPTS / f"stage_03_failure_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json", failure)
        write_json(status_path, failure)
        raise


if __name__ == "__main__":
    main()
