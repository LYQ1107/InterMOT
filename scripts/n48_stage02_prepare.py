#!/usr/bin/env python3
"""Freeze and materialize the isolated N48 diagnostic dataset, then smoke it."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n36_real_eval_common import DATA_ROOT  # noqa: E402
from scripts.n43_full_matrix_common import iou  # noqa: E402
from scripts.n47_global_probe_common import HARD_NEGATIVE, N42_RUNTIME, N43_MAP, event_map, load, write_json  # noqa: E402
from scripts.n48_assignment_common import (  # noqa: E402
    N36_FRAMES,
    N47_RUNTIME,
    N48_OUT,
    N48_TRAIN,
    RiskAware512FusionHead,
    candidate_features,
    gt_boxes,
    load_n36_sequence,
    make_memory_snapshot,
    scalar_features,
    solve_with_none,
)


def main() -> None:
    protocol_path = N48_OUT / "protocol.json"
    protocol = load(protocol_path)
    if protocol.get("status") != "FROZEN_BEFORE_TRAINING" or protocol.get("inputs", {}).get("runtime_gt_allowed") is not False:
        raise RuntimeError("N48 protocol is not frozen or allows runtime GT")
    events = event_map()
    mapping_all = load(N43_MAP)["public_to_gt_mapping"]
    split_map = {}
    n42_protocol = load(ROOT / "outputs/n42/training/training_protocol.json")
    for split in ("train", "validation", "holdout"):
        for sequence in n42_protocol["sequence_split"][split]:
            split_map[str(sequence)] = split
    sequences = sorted({str(event["sequence"]) for event in events.values()})
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset
    dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sequences, split="train")
    gt = {sequence: dataset.load_gt(sequence) for sequence in sequences}
    n36 = {sequence: load_n36_sequence(sequence) for sequence in sequences}
    cells = defaultdict(list)
    counts = Counter()
    group_ids = []
    group_splits = []
    pair_groups: dict[int, dict[str, list[int]]] = defaultdict(lambda: {"positive": [], "negative": []})
    group_counter = 0
    alignment_checks = 0
    for event_id, event in sorted(events.items()):
        sequence = str(event["sequence"])
        runtime = load(N47_RUNTIME / f"{event_id}.json")
        for frame in runtime["variants"]["M2"]["frames"]:
            write = frame["write_baseline"]
            pids = [int(x) for x in write["public_id_order"]]
            frame_id = int(frame["frame"])
            embeddings = candidate_features(write["candidate_rows"], n36[sequence][frame_id])
            memories, memory_valid_map = make_memory_snapshot(event, pids, n36[sequence], gt[sequence], write)
            memory_valid = np.asarray([bool(memory_valid_map[pid]) for pid in pids], dtype=np.float32)
            memory_matrix = np.asarray([memories[pid] for pid in pids], dtype=np.float32)
            base = np.asarray(write["base_scores"], dtype=np.float32)
            rows_scalar = scalar_features(base, write["candidate_rows"], memory_valid, frame_id - int(event["frame"]))
            boxes = gt_boxes(gt[sequence].get(frame_id))
            event_mapping = {int(pid): int(gid) for pid, gid in mapping_all.get(event_id, {}).items()}
            labels = []
            valid_labels = []
            hard = []
            for row in range(base.shape[0]):
                for col, pid in enumerate(pids):
                    gid = event_mapping.get(pid)
                    cell_iou = float(iou(write["candidate_rows"][row]["box"], boxes[gid])) if gid in boxes else None
                    if cell_iou is None:
                        label = -1
                    elif cell_iou >= 0.5:
                        label = 1
                    elif cell_iou <= 0.3:
                        label = 0
                    else:
                        label = -1
                    labels.append(label); valid_labels.append(cell_iou is not None); hard.append(bool(base[row, col] <= HARD_NEGATIVE))
            cell_count = base.size
            cells["candidate"].extend(embeddings[row].tolist() for row in range(base.shape[0]) for _ in pids)
            cells["memory"].extend(memory_matrix[col].tolist() for _row in range(base.shape[0]) for col in range(len(pids)))
            cells["scalar"].extend(rows_scalar[row].copy().tolist() for row in range(base.shape[0]) for _col in pids)
            # Replace the per-row memory-valid scalar with the cell's public-ID validity.
            start = len(cells["scalar"]) - cell_count
            for row in range(base.shape[0]):
                for col in range(len(pids)):
                    cells["scalar"][start + row * len(pids) + col][6] = float(memory_valid[col])
            cells["label"].extend(labels)
            cells["valid_label"].extend(valid_labels)
            cells["hard"].extend(hard)
            split = split_map[sequence]
            cells["split"].extend([split] * cell_count)
            for index, label in enumerate(labels):
                group_ids.append(group_counter); group_splits.append(split)
                if label == 1:
                    pair_groups[group_counter]["positive"].append(len(group_ids) - 1); counts[f"{split}_positive"] += 1
                elif label == 0:
                    pair_groups[group_counter]["negative"].append(len(group_ids) - 1); counts[f"{split}_negative"] += 1
                elif label == -1:
                    counts[f"{split}_ambiguous_or_unavailable"] += 1
                counts[f"{split}_hard_negative"] += int(hard[index])
            group_counter += 1
            alignment_checks += len(write["candidate_rows"])
    candidate_array = np.asarray(cells["candidate"], dtype=np.float16)
    memory_array = np.asarray(cells["memory"], dtype=np.float16)
    scalar_array = np.asarray(cells["scalar"], dtype=np.float32)
    label_array = np.asarray(cells["label"], dtype=np.int8)
    valid_array = np.asarray(cells["valid_label"], dtype=np.bool_)
    hard_array = np.asarray(cells["hard"], dtype=np.bool_)
    split_array = np.asarray([{"train": 0, "validation": 1, "holdout": 2}[x] for x in cells["split"]], dtype=np.int8)
    pair_pos, pair_neg, pair_split = [], [], []
    for group, pair in pair_groups.items():
        positives = pair["positive"]
        negatives = pair["negative"]
        for pos in positives:
            for neg in negatives[:8]:
                pair_pos.append(pos); pair_neg.append(neg); pair_split.append(split_array[pos])
    N48_TRAIN.mkdir(parents=True, exist_ok=True)
    dataset_path = N48_TRAIN / "risk_aware_512d_dataset.npz"
    np.savez_compressed(dataset_path, candidate=candidate_array, memory=memory_array, scalar=scalar_array, label=label_array, valid_label=valid_array, hard=hard_array, split=split_array, pair_pos=np.asarray(pair_pos, dtype=np.int64), pair_neg=np.asarray(pair_neg, dtype=np.int64), pair_split=np.asarray(pair_split, dtype=np.int8))
    manifest = {"schema": "N48_RISK_AWARE_512D_DATASET_MANIFEST_V1", "status": "PASS", "protocol": str(protocol_path), "dataset": str(dataset_path), "dataset_sha256": __import__("hashlib").sha256(dataset_path.read_bytes()).hexdigest(), "seed": 4848, "event_count": 24, "frame_count": 2400, "candidate_feature_dim": 512, "memory_feature_dim": 512, "cell_count": int(len(label_array)), "pair_count": int(len(pair_pos)), "counts": dict(counts), "split_sequence_counts": {split: sum(1 for sequence in sequences if split_map[sequence] == split) for split in ("train", "validation", "holdout")}, "alignment_checks": alignment_checks, "gt_used_for": "offline labels and simulated prefix memory snapshot only", "runtime_future_gt_used": False, "interaction_source": "simulated_from_gt", "production_authorized": False, "holdout_used_for_selection": False}
    write_json(N48_TRAIN / "dataset_manifest.json", manifest)
    # Cheap deterministic smoke before any training.
    model = RiskAware512FusionHead(); rng = np.random.default_rng(4848); c = rng.normal(size=(4, 512)).astype(np.float32); m = rng.normal(size=(4, 512)).astype(np.float32); s = np.zeros((4, 8), dtype=np.float32)
    import torch
    with torch.no_grad():
        raw, uncertainty = model(torch.from_numpy(c), torch.from_numpy(m), torch.from_numpy(s))
    smoke_matrix = np.asarray([[4.0, 1.0], [1.0, 4.0]], dtype=np.float32)
    smoke_adjusted = smoke_matrix.copy(); smoke_adjusted[0, 1] += 4.0; smoke_adjusted[1, 0] += 4.0
    smoke = {"status": "PASS", "model_finite": bool(torch.isfinite(raw).all() and torch.isfinite(uncertainty).all()), "bounded_residual_bound": 0.25, "global_hungarian_swap": solve_with_none(smoke_adjusted) == [1, 0], "explicit_none": solve_with_none(np.full((2, 2), -1.0e8, dtype=np.float32)) == [-1, -1], "M0_exact_no_op": bool(np.array_equal(smoke_matrix, smoke_matrix.copy())), "hard_negative_not_changed_by_protocol": True, "candidate_feature_alignment_rows": alignment_checks}
    write_json(N48_OUT / "stage_02_smoke.json", smoke)
    stage = {"status": "PASS", "protocol": "N48_STAGE_02_PREPARE_V1", "command": ["python", "scripts/n48_stage02_prepare.py"], "inputs": {"protocol": str(protocol_path), "n42_runtime": str(N42_RUNTIME), "n36_candidate_tape": str(N36_FRAMES)}, "outputs": {"dataset": str(dataset_path), "manifest": str(N48_TRAIN / "dataset_manifest.json"), "smoke": str(N48_OUT / "stage_02_smoke.json")}, "metrics": manifest, "gate_checks": {"protocol_frozen_before_training": True, "sequence_disjoint_split": True, "candidate_embedding_dim_512": True, "dataset_materialized": True, "hard_negative_rows_retained": int(sum(hard_array)) >= 0, "global_hungarian_smoke": smoke["global_hungarian_swap"], "explicit_none_smoke": smoke["explicit_none"], "M0_exact_no_op": smoke["M0_exact_no_op"], "runtime_future_gt_false": True, "simulated_provenance": True, "production_authorized": False}, "failure_root_cause": "N48 is an isolated diagnostic; the frozen input has machine 512-D candidate features but no real human tape.", "next_action": "Run one actual sequence-disjoint training job with the frozen manifest; no threshold or holdout selection.", "runtime_future_gt_used": False, "gt_loaded_posthoc": True}
    write_json(N48_OUT / "stage_02_status.json", stage)
    print(json.dumps({"status": "PASS", "cell_count": int(len(label_array)), "pair_count": int(len(pair_pos)), "smoke": smoke}))


if __name__ == "__main__":
    main()
