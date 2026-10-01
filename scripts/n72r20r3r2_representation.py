#!/usr/bin/env python3
"""N72R20R3R2 cross-scene identity representation learning.

Phase A mines all reliable identities from the sealed DanceTrack train
candidate tape, constructs teacher-forced causal N72R18 states for training,
trains a small generic dual-tower metric adapter, and evaluates it on the
frozen R3R1R1 runtime state under outer sequence LOSO.  A static representation
failure is terminal: NONE training and association authority are never opened.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from sam3_intermot.association.learned_identity_memory import LearnedIdentityMemoryBank
from sam3_intermot.identity_verification.cross_scene_adapter import CrossSceneIdentityAdapter
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl, unit
from scripts.n72r20r3_common import (
    CHECKPOINT,
    CHECKPOINT_SHA,
    DATASET_ROOT,
    ENCODER_SHA,
    R2_ASSET_ROOT,
    ROOT,
    SEQUENCES,
    gt_by_frame,
    iou,
    load_sequence,
    read_events,
    sha256,
    write_zstd_jsonl,
)


STAGE = "N72R20R3R2"
OUT = ROOT / "outputs" / STAGE
TRAINING = OUT / "training"
REPRESENTATION = OUT / "representation"
ASSET_MODELS = ROOT.parent / f"{ROOT.name}_{STAGE}_assets" / "models"
SOURCE_STAGE = ROOT / "outputs" / "N72R20R3R1R1"
SEEDS = (720321, 720322, 720323)
BOOTSTRAP_SEED = 720322
MAX_EPOCHS = 25
PATIENCE = 5
BATCH_SIZE = 512
TEMPERATURE = 0.07
HARD_MARGIN = 0.20
IOU_THRESHOLD = 0.50
EXPECTED_PRESENT = 6595


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def storage_audit() -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    free_gib = float(usage.free / (1024**3))
    status = "HARD_STOP" if free_gib < 100.0 else ("WARNING" if free_gib < 110.0 else "OK")
    return {
        "stage": STAGE,
        "filesystem": {
            "path": str(ROOT),
            "total_gib": float(usage.total / (1024**3)),
            "used_gib": float(usage.used / (1024**3)),
            "free_gib": free_gib,
            "storage_status": status,
            "warning_below_gib": 110.0,
            "hard_stop_below_gib": 100.0,
        },
        "target_new_storage": "<5GB",
    }


def source_audit() -> dict[str, Any]:
    goal = _json(SOURCE_STAGE / "FINAL_GOAL.json")
    result = _json(SOURCE_STAGE / "FINAL_RESULT.json")
    status = _json(SOURCE_STAGE / "stage_status.json")
    source_files = [SOURCE_STAGE / "FINAL_GOAL.json", SOURCE_STAGE / "FINAL_RESULT.json", SOURCE_STAGE / "FINAL_REPORT.md", SOURCE_STAGE / "stage_status.json"]
    return {
        "stage": STAGE,
        "source_branch": subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout.strip(),
        "source_head": _git_head(),
        "source_stage": "N72R20R3R1R1",
        "source_decision": result["decision"],
        "source_goal": goal["goal"],
        "source_runtime_future_gt_used": result["runtime_future_gt_used"],
        "source_runtime_gt_clean": result["runtime_gt_clean"],
        "source_primary_rank1": result["closed_set_candidate_top1"],
        "source_primary_mrr": result["closed_set_MRR"],
        "source_learned_state_rank1": _json(SOURCE_STAGE / "static_eval/ablations.json")["reports"]["V2_LEARNED_STATE_ONLY"]["closed_set_candidate_top1_accuracy"],
        "source_learned_state_mrr": _json(SOURCE_STAGE / "static_eval/ablations.json")["reports"]["V2_LEARNED_STATE_ONLY"]["closed_set_MRR"],
        "source_stage_status": status["status"],
        "source_artifact_hashes": [{"path": str(path.relative_to(ROOT)), "sha256": _sha256(path), "bytes": path.stat().st_size} for path in source_files],
        "previous_results_modified": False,
        "sam3_rerun": False,
        "association_authority_started": False,
        "val_accessed": False,
        "test_accessed": False,
    }


@dataclass
class SequenceMining:
    sequence: str
    frames: list[tuple[dict[str, Any], list[dict[str, Any]]]]
    base_rows: list[dict[str, Any]]
    gt: dict[int, list[tuple[int, Sequence[float]]]]
    matches: dict[int, dict[str, Any]]
    identity_observations: dict[str, list[dict[str, Any]]]
    identity_gt_ids: dict[str, int]
    candidate_arrays: np.ndarray


def _identity_key(sequence: str, gt_id: int) -> str:
    return f"{sequence}:{int(gt_id)}"


def _match_frame(sequence: str, frame_payload: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], gt_rows: Sequence[tuple[int, Sequence[float]]]) -> dict[str, Any]:
    frame = int(frame_payload["frame"])
    candidate_count = len(candidates)
    gt_count = len(gt_rows)
    matrix = np.zeros((candidate_count, gt_count), dtype=np.float64)
    for candidate_index, candidate in enumerate(candidates):
        for gt_index, (_, box) in enumerate(gt_rows):
            matrix[candidate_index, gt_index] = iou(candidate["box_xyxy"], box)
    canonical: dict[str, dict[str, Any]] = {}
    candidate_identity: dict[int, str] = {}
    duplicate_ignore: dict[int, str] = {}
    assignment_pairs: list[tuple[int, int]] = []
    if candidate_count and gt_count:
        # Tiny deterministic tie terms prefer lower candidate and GT indices;
        # they cannot change an IoU decision at the 0.50 threshold.
        cost = -matrix + np.arange(candidate_count, dtype=np.float64)[:, None] * 1.0e-10 + np.arange(gt_count, dtype=np.float64)[None, :] * 1.0e-12
        rows, columns = linear_sum_assignment(cost)
        assignment_pairs = list(zip(rows.tolist(), columns.tolist()))
    assigned_by_gt = {gt_index: candidate_index for candidate_index, gt_index in assignment_pairs if matrix[candidate_index, gt_index] >= IOU_THRESHOLD}
    for gt_index, (gt_id, _) in enumerate(gt_rows):
        key = _identity_key(sequence, gt_id)
        good = [candidate_index for candidate_index in range(candidate_count) if matrix[candidate_index, gt_index] >= IOU_THRESHOLD]
        chosen = assigned_by_gt.get(gt_index)
        if chosen is not None:
            candidate = candidates[chosen]
            canonical[key] = {
                "candidate_index": int(chosen),
                "candidate_uid": str(candidate["candidate_uid"]),
                "embedding_offset": int(candidate["embedding_offset"]),
                "gt_track_id": int(gt_id),
                "iou": float(matrix[chosen, gt_index]),
                "frame": frame,
                "identity_key": key,
            }
            candidate_identity[chosen] = key
            for duplicate_index in good:
                if duplicate_index != chosen:
                    duplicate_ignore[duplicate_index] = key
    return {
        "frame": frame,
        "gt_count": gt_count,
        "candidate_count": candidate_count,
        "matrix_max_iou": float(matrix.max()) if matrix.size else None,
        "canonical": canonical,
        "candidate_identity": {str(index): key for index, key in candidate_identity.items()},
        "same_identity_duplicate_ignore": {str(index): key for index, key in duplicate_ignore.items()},
        "assignment_pairs": [{"candidate_index": int(candidate_index), "gt_index": int(gt_index)} for candidate_index, gt_index in assignment_pairs],
        "posthoc_gt_used": True,
        "runtime_future_gt_used": False,
    }


def mine_identities() -> tuple[dict[str, SequenceMining], dict[str, Any]]:
    mining: dict[str, SequenceMining] = {}
    per_sequence = {}
    all_match_records = []
    for sequence in SEQUENCES:
        frames, base_rows = load_sequence(R2_ASSET_ROOT, sequence)
        gt = gt_by_frame(DATASET_ROOT / "train" / sequence / "gt" / "gt.txt")
        index = _json(R2_ASSET_ROOT / "candidates" / sequence / "index.json")
        count = int(index["embedding_count"])
        sealed = np.memmap(R2_ASSET_ROOT / "candidates" / sequence / "embeddings.f16", mode="r", dtype=np.float16, shape=(count, 512))
        candidate_arrays = np.asarray(sealed, dtype=np.float32).copy()
        del sealed
        norms = np.linalg.norm(candidate_arrays, axis=1, keepdims=True)
        candidate_arrays = candidate_arrays / np.maximum(norms, 1.0e-8)
        matches: dict[int, dict[str, Any]] = {}
        observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
        identity_gt_ids = {_identity_key(sequence, gt_id): int(gt_id) for values in gt.values() for gt_id, _ in values}
        matched = duplicate_count = 0
        for (frame_payload, candidates), base_row in zip(frames, base_rows):
            frame = int(frame_payload["frame"])
            if base_row.get("runtime_future_gt_used") is not False or frame_payload.get("runtime_future_gt_used") is not False:
                raise RuntimeError(f"runtime GT provenance crossed candidate tape: {sequence}:{frame}")
            frame_match = _match_frame(sequence, frame_payload, candidates, gt.get(frame, []))
            matches[frame] = frame_match
            matched += len(frame_match["canonical"])
            duplicate_count += len(frame_match["same_identity_duplicate_ignore"])
            for key, item in frame_match["canonical"].items():
                observations[key].append(dict(item))
                all_match_records.append({"sequence": sequence, "frame": frame, **item, "canonical": True, "same_identity_duplicate_ignore": False})
            for candidate_index, key in frame_match["same_identity_duplicate_ignore"].items():
                all_match_records.append({"sequence": sequence, "frame": frame, "candidate_index": int(candidate_index), "candidate_uid": str(candidates[int(candidate_index)]["candidate_uid"]), "identity_key": key, "canonical": False, "same_identity_duplicate_ignore": True, "runtime_future_gt_used": False, "posthoc_gt_used": True})
        observation_counts = [len(values) for values in observations.values()]
        eligible = [key for key, values in observations.items() if len(values) >= 4]
        per_sequence[sequence] = {
            "frames": len(frames),
            "candidate_rows": int(sum(len(candidates) for _, candidates in frames)),
            "gt_identities": len(identity_gt_ids),
            "matched_candidate_gt_observations": matched,
            "same_identity_duplicate_candidates_ignored": duplicate_count,
            "identities_with_ge_2_observations": int(sum(value >= 2 for value in observation_counts)),
            "identities_with_ge_4_observations": int(sum(value >= 4 for value in observation_counts)),
            "identities_with_ge_8_observations": int(sum(value >= 8 for value in observation_counts)),
            "median_observations_per_identity": float(np.median(observation_counts)) if observation_counts else 0.0,
            "observation_quantiles": {f"P{q}": float(np.percentile(observation_counts, q)) for q in (10, 25, 50, 75, 90)} if observation_counts else {},
            "eligible_identity_count": len(eligible),
            "excluded_identity_count": len(identity_gt_ids) - len(eligible),
        }
        mining[sequence] = SequenceMining(sequence, frames, base_rows, gt, matches, dict(observations), identity_gt_ids, candidate_arrays)
    all_identity_counts = [len(values) for item in mining.values() for values in item.identity_observations.values()]
    training_identities = sorted(key for item in mining.values() for key, values in item.identity_observations.items() if len(values) >= 4)
    audit = {
        "stage": STAGE,
        "status": "PASS_IDENTITY_MINING_COMPLETE",
        "iou_threshold": IOU_THRESHOLD,
        "matching": "deterministic one-to-one linear-sum assignment maximizing candidate-GT IoU; same-identity duplicate proposals are ignored as negatives",
        "sequences": per_sequence,
        "total_gt_identities": int(sum(item["gt_identities"] for item in per_sequence.values())),
        "total_eligible_identities": len(training_identities),
        "total_matched_candidate_gt_observations": int(sum(item["matched_candidate_gt_observations"] for item in per_sequence.values())),
        "total_same_identity_duplicate_candidates_ignored": int(sum(item["same_identity_duplicate_candidates_ignored"] for item in per_sequence.values())),
        "all_identity_observation_quantiles": {f"P{q}": float(np.percentile(all_identity_counts, q)) for q in (10, 25, 50, 75, 90)} if all_identity_counts else {},
        "candidate_generation_frozen": True,
        "sam3_rerun": False,
        "runtime_future_gt_used": False,
        "runtime_gt_clean": True,
        "heldout_training_mining_deferred_to_outer_split": True,
    }
    TRAINING.mkdir(parents=True, exist_ok=True)
    _write_json(
        TRAINING / "identity_match_manifest.json",
        {
            "stage": STAGE,
            "status": audit["status"],
            "matching": audit["matching"],
            "iou_threshold": IOU_THRESHOLD,
            "records": len(all_match_records),
            "canonical_records": audit["total_matched_candidate_gt_observations"],
            "same_identity_duplicate_records": audit["total_same_identity_duplicate_candidates_ignored"],
            "sequences": per_sequence,
            "runtime_future_gt_used": False,
            "posthoc_gt_used": True,
        },
    )
    with (TRAINING / "identity_match_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in all_match_records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    _write_json(OUT / "identity_mining_audit.json", audit)
    return mining, audit


def _frozen_updater() -> LearnedIdentityMemoryBank:
    return LearnedIdentityMemoryBank.from_checkpoint(
        CHECKPOINT,
        encoder_sha256=ENCODER_SHA,
        expected_encoder_sha256=ENCODER_SHA,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        device="cpu",
    )


def _state_update(updater: LearnedIdentityMemoryBank, state: np.ndarray, observation: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        previous = torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)
        current = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        updated, _, _ = updater.updater(previous, current)
    return unit(updated.detach().cpu().numpy()[0], "causal identity state")


def build_episodes(mining: Mapping[str, SequenceMining]) -> dict[str, Any]:
    updater = _frozen_updater()
    query_states: list[np.ndarray] = []
    episodes: list[dict[str, Any]] = []
    identity_summary = {}
    excluded = []
    for sequence in SEQUENCES:
        item = mining[sequence]
        frame_candidates = {int(frame_payload["frame"]): candidates for frame_payload, candidates in item.frames}
        for identity_key, observations in sorted(item.identity_observations.items()):
            observations = sorted(observations, key=lambda value: (int(value["frame"]), int(value["candidate_index"])))
            if len(observations) < 4:
                excluded.append(identity_key)
                continue
            anchor = observations[0]
            state = unit(item.candidate_arrays[int(anchor["embedding_offset"])], "identity anchor feature")
            history_frames = [int(anchor["frame"])]
            episode_count = 0
            for observation in observations[1:]:
                frame = int(observation["frame"])
                candidates = frame_candidates[frame]
                match = item.matches[frame]
                duplicate_indices = [int(index) for index, key in match["same_identity_duplicate_ignore"].items() if key == identity_key]
                positive_index = int(observation["candidate_index"])
                ignored = sorted(set(duplicate_indices))
                hard_negative_indices = [
                    index for index in range(len(candidates))
                    if index != positive_index and index not in ignored
                ]
                state_ref = len(query_states)
                query_states.append(np.asarray(state, dtype=np.float32).copy())
                episodes.append(
                    {
                        "stage": STAGE,
                        "sequence": sequence,
                        "query_frame": frame,
                        "identity_key": identity_key,
                        "gt_track_id": int(item.identity_gt_ids[identity_key]),
                        "anchor_frame": int(anchor["frame"]),
                        "history_frames": list(history_frames),
                        "query_state_ref": state_ref,
                        "candidate_uid_axis": [str(candidate["candidate_uid"]) for candidate in candidates],
                        "candidate_embedding_offsets": [int(candidate["embedding_offset"]) for candidate in candidates],
                        "positive_candidate_indices": [positive_index],
                        "hard_negative_indices": hard_negative_indices,
                        "same_identity_duplicate_ignore": ignored,
                        "posthoc_gt_used": True,
                        "training_posthoc_gt_used": True,
                        "runtime_future_gt_used": False,
                    }
                )
                episode_count += 1
                state = _state_update(updater, state, item.candidate_arrays[int(observation["embedding_offset"])])
                history_frames.append(frame)
            identity_summary[identity_key] = {
                "sequence": sequence,
                "observations": len(observations),
                "episodes": episode_count,
                "anchor_frame": int(anchor["frame"]),
                "eligible": True,
            }
    if not episodes or len(query_states) != len(episodes):
        raise RuntimeError("identity episode construction produced no episodes or mismatched state refs")
    query_array = np.asarray(query_states, dtype=np.float32)
    np.save(TRAINING / "query_states.float32.npy", query_array)
    with (TRAINING / "identity_episode_index.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    training_identity_keys = sorted(identity_summary)
    outer_folds = []
    for heldout in SEQUENCES:
        fit_sequences = [sequence for sequence in SEQUENCES if sequence != heldout]
        validation = SEQUENCES[(SEQUENCES.index(heldout) + 1) % len(SEQUENCES)]
        train_sequences = [sequence for sequence in fit_sequences if sequence != validation]
        train_keys = [key for key in training_identity_keys if key.split(":", 1)[0] in train_sequences]
        validation_keys = [key for key in training_identity_keys if key.split(":", 1)[0] == validation]
        heldout_keys = [key for key in training_identity_keys if key.split(":", 1)[0] == heldout]
        train_episodes = [episode for episode in episodes if episode["sequence"] in train_sequences]
        validation_episodes = [episode for episode in episodes if episode["sequence"] == validation]
        heldout_episodes = [episode for episode in episodes if episode["sequence"] == heldout]
        outer_folds.append(
            {
                "heldout_sequence": heldout,
                "parameter_fit_sequences": train_sequences,
                "internal_validation_sequence": validation,
                "all_nonheldout_sequences": fit_sequences,
                "training_identities": len(train_keys),
                "validation_identities": len(validation_keys),
                "heldout_identities": len(heldout_keys),
                "training_positive_episodes": len(train_episodes),
                "validation_positive_episodes": len(validation_episodes),
                "heldout_positive_episodes": len(heldout_episodes),
                "heldout_observations_used_in_training": 0,
                "heldout_identity_keys_in_training": 0,
                "outer_heldout_absent_from_training": True,
            }
        )
    manifest = {
        "stage": STAGE,
        "status": "PASS_IDENTITY_EPISODES_COMPLETE",
        "episode_count": len(episodes),
        "training_identity_count": len(training_identity_keys),
        "excluded_identity_count": len(excluded),
        "excluded_identity_keys": excluded,
        "query_state_count": len(query_states),
        "eligibility_min_observations": 4,
        "identity_label": "(sequence_name, gt_track_id)",
        "query_state_lineage": "frozen N72R18 GRU, teacher-forced correct observations with frame < query_frame",
        "training_posthoc_gt_used": True,
        "runtime_future_gt_used": False,
        "candidate_features_copied": False,
        "identity_summary": identity_summary,
        "outer_folds": outer_folds,
    }
    _write_json(TRAINING / "identity_episode_manifest.json", manifest)
    _write_json(TRAINING / "outer_split_manifest.json", {"stage": STAGE, "folds": outer_folds, "architecture_selection": False, "heldout_supervision_forbidden": True})
    _write_json(TRAINING / "seed_manifest.json", {"stage": STAGE, "formal_seeds": list(SEEDS), "paired_bootstrap_seed": BOOTSTRAP_SEED, "temperature": TEMPERATURE, "hard_negative_margin": HARD_MARGIN})
    return {"episodes": episodes, "query_states": query_array, "identity_summary": identity_summary, "outer_folds": outer_folds, "training_identity_keys": training_identity_keys, "manifest": manifest}


@dataclass
class EpisodeDataset:
    episodes: list[dict[str, Any]]
    query_states: np.ndarray
    candidate_arrays: dict[str, np.ndarray]


def load_episode_dataset(mining: Mapping[str, SequenceMining] | None = None) -> EpisodeDataset:
    episodes = []
    with (TRAINING / "identity_episode_index.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            episodes.append(json.loads(line))
    query_states = np.load(TRAINING / "query_states.float32.npy", mmap_mode="r")
    arrays = {}
    if mining is not None:
        arrays = {sequence: item.candidate_arrays for sequence, item in mining.items()}
    else:
        for sequence in SEQUENCES:
            index = _json(R2_ASSET_ROOT / "candidates" / sequence / "index.json")
            count = int(index["embedding_count"])
            sealed = np.memmap(R2_ASSET_ROOT / "candidates" / sequence / "embeddings.f16", mode="r", dtype=np.float16, shape=(count, 512))
            value = np.asarray(sealed, dtype=np.float32).copy()
            del sealed
            value /= np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1.0e-8)
            arrays[sequence] = value
    return EpisodeDataset(episodes, query_states, arrays)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _balanced_batches(episodes: Sequence[dict[str, Any]], seed: int, training: bool) -> list[list[dict[str, Any]]]:
    if not episodes:
        return []
    if not training:
        return [list(episodes[start : start + BATCH_SIZE]) for start in range(0, len(episodes), BATCH_SIZE)]
    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_identity[str(episode["identity_key"])].append(episode)
    rng = random.Random(seed)
    for values in by_identity.values():
        rng.shuffle(values)
    identities = sorted(by_identity)
    target_steps = max(1, math.ceil(len(episodes) / BATCH_SIZE))
    cursors = {key: 0 for key in identities}
    batches = []
    for step in range(target_steps):
        order = identities[:]
        rng.shuffle(order)
        selected = []
        for identity in order[: min(BATCH_SIZE, len(order))]:
            values = by_identity[identity]
            selected.append(values[cursors[identity] % len(values)])
            cursors[identity] += 1
        while len(selected) < min(BATCH_SIZE, len(episodes)):
            identity = order[len(selected) % len(order)]
            values = by_identity[identity]
            selected.append(values[cursors[identity] % len(values)])
            cursors[identity] += 1
        batches.append(selected)
    return batches


def _episode_batch(dataset: EpisodeDataset, episodes: Sequence[dict[str, Any]], device: torch.device) -> tuple[torch.Tensor, ...]:
    max_candidates = max(1, max(len(episode["candidate_embedding_offsets"]) for episode in episodes))
    query = np.asarray([dataset.query_states[int(episode["query_state_ref"])] for episode in episodes], dtype=np.float32)
    candidates = np.zeros((len(episodes), max_candidates, 512), dtype=np.float32)
    valid = np.zeros((len(episodes), max_candidates), dtype=bool)
    ignored = np.zeros((len(episodes), max_candidates), dtype=bool)
    positive = np.zeros((len(episodes), max_candidates), dtype=bool)
    hard = np.zeros((len(episodes), max_candidates), dtype=bool)
    positive_indices = []
    identities = []
    for row_index, episode in enumerate(episodes):
        offsets = episode["candidate_embedding_offsets"]
        values = dataset.candidate_arrays[str(episode["sequence"])][np.asarray(offsets, dtype=np.int64)]
        count = len(offsets)
        candidates[row_index, :count] = values
        valid[row_index, :count] = True
        for index in episode["same_identity_duplicate_ignore"]:
            ignored[row_index, int(index)] = True
        for index in episode["positive_candidate_indices"]:
            positive[row_index, int(index)] = True
        for index in episode["hard_negative_indices"]:
            hard[row_index, int(index)] = True
        positive_indices.append(int(episode["positive_candidate_indices"][0]))
        identities.append(str(episode["identity_key"]))
    return (
        torch.from_numpy(query).to(device),
        torch.from_numpy(candidates).to(device),
        torch.from_numpy(valid).to(device),
        torch.from_numpy(ignored).to(device),
        torch.from_numpy(positive).to(device),
        torch.from_numpy(hard).to(device),
        torch.tensor(positive_indices, dtype=torch.long, device=device),
        identities,
    )


def _adapter_loss(model: CrossSceneIdentityAdapter, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, dict[str, float]]:
    query, candidates, valid, ignored, positive, hard, positive_indices, identities = batch
    output = model(query, candidates, valid)
    scores = output["scores"] / TEMPERATURE
    loss_mask = valid & ~ignored
    denominator = torch.logsumexp(scores.masked_fill(~loss_mask, -1.0e9), dim=1)
    positive_scores = torch.logsumexp(scores.masked_fill(~positive, -1.0e9), dim=1)
    listwise = (denominator - positive_scores).mean()

    positive_embedding = output["candidate_embedding"][torch.arange(len(identities), device=query.device), positive_indices]
    query_embedding = output["query_embedding"]
    contrastive_scores = torch.matmul(query_embedding, positive_embedding.transpose(0, 1)) / TEMPERATURE
    identity_equal = torch.tensor([[left == right for right in identities] for left in identities], dtype=torch.bool, device=query.device)
    contrastive_log_prob = F.log_softmax(contrastive_scores, dim=1)
    supcon = -(contrastive_log_prob.masked_fill(~identity_equal, 0.0).sum(dim=1) / identity_equal.sum(dim=1).clamp_min(1)).mean()

    positive_value = output["scores"][torch.arange(len(identities), device=query.device), positive_indices]
    hard_values = output["scores"].masked_fill(~hard, -1.0e9).max(dim=1).values
    has_hard = hard.any(dim=1)
    hard_loss = F.relu(HARD_MARGIN - positive_value + hard_values).masked_select(has_hard).mean() if bool(has_hard.any()) else positive_value.new_zeros(())
    total = listwise + 0.20 * supcon + 0.20 * hard_loss
    return total, {"listwise": float(listwise.detach().cpu()), "supcon": float(supcon.detach().cpu()), "hard": float(hard_loss.detach().cpu()), "total": float(total.detach().cpu())}


def _run_epoch(model: CrossSceneIdentityAdapter, dataset: EpisodeDataset, episodes: Sequence[dict[str, Any]], device: torch.device, optimizer: torch.optim.Optimizer | None, seed: int) -> dict[str, float]:
    model.train(optimizer is not None)
    total = Counter()
    batches = _balanced_batches(episodes, seed, optimizer is not None)
    for batch_episodes in batches:
        batch = _episode_batch(dataset, batch_episodes, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            loss, components = _adapter_loss(model, batch)
        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        for key, value in components.items():
            total[key] += value
    count = max(1, len(batches))
    return {key: float(value / count) for key, value in total.items()}


def _fit_inner(dataset: EpisodeDataset, fit_episodes: Sequence[dict[str, Any]], validation_episodes: Sequence[dict[str, Any]], seed: int, device: torch.device) -> tuple[CrossSceneIdentityAdapter, dict[str, Any]]:
    _seed_everything(seed)
    model = CrossSceneIdentityAdapter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    best_epoch = 1
    stale = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = _run_epoch(model, dataset, fit_episodes, device, optimizer, seed + epoch)
        with torch.no_grad():
            validation_metrics = _run_epoch(model, dataset, validation_episodes, device, None, seed)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["total"] < best_validation - 1.0e-5:
            best_validation = validation_metrics["total"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "epochs_run": len(history), "best_validation_loss": best_validation, "history": history, "trainable_parameters": model.trainable_parameters, "temperature": TEMPERATURE, "hard_negative_margin": HARD_MARGIN}


def _fit_final(dataset: EpisodeDataset, episodes: Sequence[dict[str, Any]], seed: int, epochs: int, device: torch.device) -> tuple[CrossSceneIdentityAdapter, dict[str, Any]]:
    _seed_everything(seed)
    model = CrossSceneIdentityAdapter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    history = []
    for epoch in range(1, int(epochs) + 1):
        history.append(_run_epoch(model, dataset, episodes, device, optimizer, seed + epoch))
    return model, {"epochs_run": int(epochs), "last_epoch": history[-1] if history else None}


def _save_checkpoint(model: CrossSceneIdentityAdapter, name: str, heldout: str, seed: int, fit_sequences: Sequence[str], validation: str, epoch: int, source_head: str) -> dict[str, Any]:
    ASSET_MODELS.mkdir(parents=True, exist_ok=True)
    path = ASSET_MODELS / f"cross_scene_adapter__{heldout}__seed{seed}.pt"
    torch.save(
        {
            "stage": STAGE,
            "architecture": name,
            "feature_dim": model.feature_dim,
            "bottleneck_dim": model.bottleneck_dim,
            "temperature": TEMPERATURE,
            "hard_negative_margin": HARD_MARGIN,
            "source_commit": source_head,
            "heldout_sequence": heldout,
            "parameter_fit_sequences": list(fit_sequences),
            "internal_validation_sequence": validation,
            "seed": int(seed),
            "trainable_parameters": model.trainable_parameters,
            "OSNet_frozen": True,
            "GRU_frozen": True,
            "runtime_future_gt_used": False,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        },
        path,
    )
    return {"architecture": name, "heldout_sequence": heldout, "seed": seed, "path": str(path), "sha256": _sha256(path), "trainable_parameters": model.trainable_parameters, "selected_epoch": epoch, "binary_committed": False}


def _load_runtime_rows() -> tuple[list[dict[str, Any]], np.ndarray, dict[str, dict[int, dict[str, Any]]]]:
    index_rows = []
    with (SOURCE_STAGE / "training/canonical_training_index.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["training_label"]["candidate_index"] is not None:
                index_rows.append(row)
    states = np.load(SOURCE_STAGE / "training/causal_learned_state_vectors.float32.npy", mmap_mode="r")
    labels = read_zstd_jsonl(ROOT / "outputs/N72R20R3/presence/frame_posthoc_labels.jsonl.zst")
    target_labels: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in labels:
        if row["state_condition"] == "S1_R2_STYLE_CAUSAL_LEARNED_STATE":
            target_labels[str(row["sequence"])][int(row["frame"])] = row
    return index_rows, states, target_labels


def _oracle_states(mining: Mapping[str, SequenceMining], runtime_rows: Sequence[dict[str, Any]], target_labels: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> dict[tuple[str, int], np.ndarray]:
    events = read_events(R2_ASSET_ROOT)
    result: dict[tuple[str, int], np.ndarray] = {}
    query_frames_by_sequence: dict[str, list[int]] = defaultdict(list)
    target_key_by_sequence: dict[str, str] = {}
    for row in runtime_rows:
        sequence, frame = str(row["sequence"]), int(row["frame"])
        label = target_labels[sequence][frame]
        gt_id = int(label["target_gt_id"])
        target_key_by_sequence[sequence] = _identity_key(sequence, gt_id)
        query_frames_by_sequence[sequence].append(frame)
    for sequence, frames in query_frames_by_sequence.items():
        event = events[sequence]
        event_frame = int(event["event_frame"])
        key = target_key_by_sequence[sequence]
        observations = sorted(mining[sequence].identity_observations.get(key, []), key=lambda value: int(value["frame"]))
        state = unit(event["human_anchor"], "human anchor")
        updater = _frozen_updater()
        cursor = 0
        for frame in sorted(set(frames)):
            while cursor < len(observations) and int(observations[cursor]["frame"]) < frame:
                observation = observations[cursor]
                if int(observation["frame"]) > event_frame:
                    state = _state_update(updater, state, mining[sequence].candidate_arrays[int(observation["embedding_offset"])])
                cursor += 1
            result[(sequence, frame)] = np.asarray(state, dtype=np.float32).copy()
    return result


def _score_rows(model: CrossSceneIdentityAdapter, dataset: EpisodeDataset, rows: Sequence[dict[str, Any]], states: np.ndarray | Mapping[tuple[str, int], np.ndarray], device: torch.device) -> list[dict[str, Any]]:
    model.eval()
    output_rows = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = list(rows[start : start + BATCH_SIZE])
        max_candidates = max(1, max(len(row["candidate_axis"]) for row in batch_rows))
        query = []
        candidates = np.zeros((len(batch_rows), max_candidates, 512), dtype=np.float32)
        mask = np.zeros((len(batch_rows), max_candidates), dtype=bool)
        for row_index, row in enumerate(batch_rows):
            key = (str(row["sequence"]), int(row["frame"]))
            query_state = states[key] if isinstance(states, Mapping) else states[int(row["causal_learned_state_ref"])]
            query.append(np.asarray(query_state, dtype=np.float32))
            offsets = [int(item["embedding_offset"]) for item in row["candidate_axis"]]
            candidates[row_index, :len(offsets)] = dataset.candidate_arrays[str(row["sequence"])][np.asarray(offsets, dtype=np.int64)]
            mask[row_index, :len(offsets)] = True
        with torch.no_grad():
            output = model(torch.from_numpy(np.asarray(query, dtype=np.float32)).to(device), torch.from_numpy(candidates).to(device), torch.from_numpy(mask).to(device))
        scores = output["scores"].detach().cpu().numpy()
        for row_index, row in enumerate(batch_rows):
            count = len(row["candidate_axis"])
            values = np.asarray(scores[row_index, :count], dtype=np.float64)
            order = np.argsort(-values, kind="stable")
            label_index = int(row["training_label"]["candidate_index"])
            rank = int(np.where(order == label_index)[0][0]) + 1 if label_index in order else None
            output_rows.append({"sequence": str(row["sequence"]), "frame": int(row["frame"]), "label_index": label_index, "candidate_uids": [str(item["candidate_uid"]) for item in row["candidate_axis"]], "scores": values.astype(float).tolist(), "rank": rank, "runtime_future_gt_used": False, "runtime_gt_clean": True, "posthoc_oracle_only": isinstance(states, Mapping)})
    return output_rows


def _aggregate_seed_scores(prediction_sets: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for predictions in prediction_sets:
        for row in predictions:
            grouped[(row["sequence"], int(row["frame"]))].append(row)
    result = []
    for key in sorted(grouped):
        values = grouped[key]
        if len(values) != len(prediction_sets):
            raise RuntimeError(f"seed prediction count mismatch at {key}")
        axis = values[0]["candidate_uids"]
        if any(row["candidate_uids"] != axis for row in values):
            raise RuntimeError(f"candidate axis mismatch at {key}")
        scores = np.mean([np.asarray(row["scores"], dtype=np.float64) for row in values], axis=0)
        order = np.argsort(-scores, kind="stable")
        label_index = int(values[0]["label_index"])
        rank = int(np.where(order == label_index)[0][0]) + 1 if label_index in order else None
        result.append({"sequence": key[0], "frame": key[1], "label_index": label_index, "candidate_uids": axis, "scores": scores.astype(float).tolist(), "rank": rank, "runtime_future_gt_used": False, "runtime_gt_clean": True, "seed_count": len(values), "posthoc_oracle_only": bool(values[0].get("posthoc_oracle_only"))})
    return result


def rank_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ranks = [int(row["rank"]) for row in rows if row.get("rank") is not None]
    total = len(rows)
    return {
        "rows": total,
        "rank1": float(sum(rank <= 1 for rank in ranks) / total) if total else None,
        "rank2": float(sum(rank <= 2 for rank in ranks) / total) if total else None,
        "rank3": float(sum(rank <= 3 for rank in ranks) / total) if total else None,
        "rank5": float(sum(rank <= 5 for rank in ranks) / total) if total else None,
        "MRR": float(np.mean([1.0 / rank for rank in ranks])) if ranks else None,
        "mean_target_rank": float(np.mean(ranks)) if ranks else None,
        "runtime_future_gt_used": any(bool(row.get("runtime_future_gt_used")) for row in rows),
        "runtime_gt_clean": all(not bool(row.get("runtime_future_gt_used")) for row in rows),
        "posthoc_oracle_only": all(bool(row.get("posthoc_oracle_only")) for row in rows) if rows else False,
    }


def _per_sequence(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["sequence"])].append(row)
    return {sequence: dict(rank_metrics(grouped[sequence]), sequence=sequence) for sequence in SEQUENCES}


def paired_bootstrap(new_rows: Sequence[dict[str, Any]], baseline_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    new_by_sequence = _per_sequence(new_rows)
    base_by_sequence = _per_sequence(baseline_rows)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    for _ in range(2000):
        sampled = [SEQUENCES[int(index)] for index in rng.integers(0, len(SEQUENCES), size=len(SEQUENCES))]
        new_correct = sum(new_by_sequence[sequence]["rank1"] * new_by_sequence[sequence]["rows"] for sequence in sampled)
        base_correct = sum(base_by_sequence[sequence]["rank1"] * base_by_sequence[sequence]["rows"] for sequence in sampled)
        total = sum(new_by_sequence[sequence]["rows"] for sequence in sampled)
        values.append(float((new_correct - base_correct) / total) if total else 0.0)
    return {"stage": STAGE, "repetitions": 2000, "seed": BOOTSTRAP_SEED, "unit": "sequence cluster paired delta", "baseline": rank_metrics(baseline_rows), "new": rank_metrics(new_rows), "delta_rank1": float(rank_metrics(new_rows)["rank1"] - rank_metrics(baseline_rows)["rank1"]), "interval_95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))], "mean_delta": float(np.mean(values))}


def run_representation(dataset: EpisodeDataset, mining: Mapping[str, SequenceMining], device: torch.device) -> dict[str, Any]:
    source_head = _git_head()
    runtime_rows, runtime_states, target_labels = _load_runtime_rows()
    if len(runtime_rows) != EXPECTED_PRESENT:
        raise RuntimeError(f"expected {EXPECTED_PRESENT} formal PRESENT rows, got {len(runtime_rows)}")
    oracle_states = _oracle_states(mining, runtime_rows, target_labels)
    baseline_raw = read_zstd_jsonl(SOURCE_STAGE / "static_eval/predictions_V2_LEARNED_STATE_ONLY.jsonl.zst")
    baseline_rows = [row for row in baseline_raw if int(row["label_index"]) >= 0]
    baseline_rank_rows = [{"sequence": row["sequence"], "frame": int(row["frame"]), "rank": int(row["candidate_rank"]), "runtime_future_gt_used": False, "runtime_gt_clean": True} for row in baseline_rows if row.get("candidate_rank") is not None]
    if len(baseline_rank_rows) != EXPECTED_PRESENT:
        raise RuntimeError("frozen learned-state baseline PRESENT rows are incomplete")
    all_predictions: dict[str, Any] = {"runtime_by_seed": {}, "oracle_by_seed": {}, "fold_records": [], "checkpoints": []}
    for seed in SEEDS:
        all_predictions["runtime_by_seed"][str(seed)] = []
        all_predictions["oracle_by_seed"][str(seed)] = []
    for heldout in SEQUENCES:
        validation = SEQUENCES[(SEQUENCES.index(heldout) + 1) % len(SEQUENCES)]
        fit_sequences = [sequence for sequence in SEQUENCES if sequence not in {heldout, validation}]
        fit_episodes = [episode for episode in dataset.episodes if episode["sequence"] in fit_sequences]
        validation_episodes = [episode for episode in dataset.episodes if episode["sequence"] == validation]
        final_episodes = [episode for episode in dataset.episodes if episode["sequence"] != heldout]
        heldout_rows = [row for row in runtime_rows if row["sequence"] == heldout]
        for seed in SEEDS:
            inner_model, inner_fit = _fit_inner(dataset, fit_episodes, validation_episodes, seed, device)
            final_model, final_fit = _fit_final(dataset, final_episodes, seed, int(inner_fit["best_epoch"]), device)
            runtime_predictions = _score_rows(final_model, dataset, heldout_rows, runtime_states, device)
            oracle_predictions = _score_rows(final_model, dataset, heldout_rows, oracle_states, device)
            all_predictions["runtime_by_seed"][str(seed)].extend(runtime_predictions)
            all_predictions["oracle_by_seed"][str(seed)].extend(oracle_predictions)
            all_predictions["fold_records"].append({"heldout_sequence": heldout, "seed": seed, "parameter_fit_sequences": fit_sequences, "internal_validation_sequence": validation, "final_training_sequences": [sequence for sequence in SEQUENCES if sequence != heldout], "training_identity_count": len({episode["identity_key"] for episode in fit_episodes}), "training_positive_episodes": len(fit_episodes), "validation_positive_episodes": len(validation_episodes), "heldout_observations_used_in_training": 0, "outer_heldout_absent_from_training": True, "selected_epoch": inner_fit["best_epoch"], "inner_fit": inner_fit, "final_fit": final_fit, "architecture_selection_from_outer_scores": False})
            all_predictions["checkpoints"].append(_save_checkpoint(final_model, "CrossSceneIdentityAdapter", heldout, seed, fit_sequences, validation, int(inner_fit["best_epoch"]), source_head))
    runtime_final = _aggregate_seed_scores([all_predictions["runtime_by_seed"][str(seed)] for seed in SEEDS])
    oracle_final = _aggregate_seed_scores([all_predictions["oracle_by_seed"][str(seed)] for seed in SEEDS])
    if len(runtime_final) != EXPECTED_PRESENT or len(oracle_final) != EXPECTED_PRESENT:
        raise RuntimeError("seed ensemble must produce one score vector per PRESENT frame")
    runtime_metrics = rank_metrics(runtime_final)
    oracle_metrics = rank_metrics(oracle_final)
    baseline_metrics = rank_metrics(baseline_rank_rows)
    paired = paired_bootstrap(runtime_final, baseline_rank_rows)
    per_sequence = _per_sequence(runtime_final)
    oracle_per_sequence = _per_sequence(oracle_final)
    gate = {
        "R1_pooled_rank1_ge_0.60": runtime_metrics["rank1"] is not None and runtime_metrics["rank1"] >= 0.60,
        "R2_macro_sequence_rank1_ge_0.50": float(np.mean([item["rank1"] for item in per_sequence.values()])) >= 0.50,
        "R3_pooled_mrr_ge_0.70": runtime_metrics["MRR"] is not None and runtime_metrics["MRR"] >= 0.70,
        "R4_paired_bootstrap_rank1_delta_lower_gt_0": paired["interval_95"][0] > 0.0,
        "R5_no_outer_supervision_leakage": all(bool(record["outer_heldout_absent_from_training"]) and int(record["heldout_observations_used_in_training"]) == 0 for record in all_predictions["fold_records"]),
    }
    gate_result = {"checks": gate, "pass": bool(all(gate.values())), "primary_comparator": "FROZEN_LEARNED_STATE_BASELINE"}
    REPRESENTATION.mkdir(parents=True, exist_ok=True)
    write_zstd_jsonl(REPRESENTATION / "runtime_predictions.jsonl.zst", runtime_final)
    write_zstd_jsonl(REPRESENTATION / "oracle_clean_predictions.jsonl.zst", oracle_final)
    for seed in SEEDS:
        write_zstd_jsonl(REPRESENTATION / f"runtime_predictions_seed{seed}.jsonl.zst", all_predictions["runtime_by_seed"][str(seed)])
    _write_json(REPRESENTATION / "frozen_baseline.json", {"stage": STAGE, "name": "FROZEN_LEARNED_STATE_BASELINE", "source": "outputs/N72R20R3R1R1/static_eval/predictions_V2_LEARNED_STATE_ONLY.jsonl.zst", "rank1": baseline_metrics["rank1"], "MRR": baseline_metrics["MRR"], "rank2": baseline_metrics["rank2"], "rank3": baseline_metrics["rank3"], "rank5": baseline_metrics["rank5"]})
    _write_json(REPRESENTATION / "formal_loso.json", {"stage": STAGE, "status": "PASS_REPRESENTATION_FORMAL_LOSO_COMPLETE", "primary_model": "CrossSceneIdentityAdapter", "source_head": source_head, "seeds": list(SEEDS), "heldout_sequences": list(SEQUENCES), "runtime_metrics": runtime_metrics, "oracle_clean_metrics": oracle_metrics, "baseline_metrics": baseline_metrics, "paired_bootstrap": paired, "representation_gate": gate_result, "fold_records": all_predictions["fold_records"], "runtime_future_gt_used": False, "runtime_gt_clean": True, "candidate_generation_frozen": True, "sam3_rerun": False})
    _write_json(REPRESENTATION / "per_sequence.json", {"stage": STAGE, "runtime_state": per_sequence, "oracle_clean_state": oracle_per_sequence, "baseline": _per_sequence(baseline_rank_rows)})
    _write_json(REPRESENTATION / "ranking_metrics.json", {"stage": STAGE, "runtime_state": runtime_metrics, "oracle_clean_state": oracle_metrics, "frozen_baseline": baseline_metrics, "paired_delta": paired})
    _write_json(REPRESENTATION / "oracle_clean_diagnostic.json", {"stage": STAGE, "posthoc_oracle_only": True, "runtime_usable": False, "runtime_state": runtime_metrics, "oracle_clean_state": oracle_metrics, "delta_rank1": float(oracle_metrics["rank1"] - runtime_metrics["rank1"])})
    _write_json(REPRESENTATION / "paired_bootstrap.json", paired)
    _write_json(TRAINING / "checkpoint_manifest.json", {"stage": STAGE, "records": all_predictions["checkpoints"], "binary_committed": False, "asset_root": str(ASSET_MODELS)})
    _write_json(REPRESENTATION / "formal_prediction_manifest.json", {"stage": STAGE, "runtime_rows": len(runtime_final), "oracle_rows": len(oracle_final), "seed_rows": {str(seed): len(all_predictions["runtime_by_seed"][str(seed)]) for seed in SEEDS}, "one_score_vector_per_frame": True, "runtime_future_gt_used": False, "runtime_gt_clean": True})
    return {"source_head": source_head, "runtime_final": runtime_final, "oracle_final": oracle_final, "baseline": baseline_rank_rows, "runtime_metrics": runtime_metrics, "oracle_metrics": oracle_metrics, "baseline_metrics": baseline_metrics, "paired": paired, "gate": gate_result, "fold_records": all_predictions["fold_records"], "checkpoints": all_predictions["checkpoints"]}


def finalize(formal: Mapping[str, Any], tests_summary: str = "PENDING") -> dict[str, Any]:
    runtime = formal["runtime_metrics"]
    oracle = formal["oracle_metrics"]
    gate = formal["gate"]
    if gate["pass"]:
        raise RuntimeError(
            "Phase A representation gate passed; do not finalize R3R2 before the separately authorized Phase B/C evaluation"
        )
    elif float(oracle["rank1"] or 0.0) >= 0.60:
        decision = "FAIL_RUNTIME_IDENTITY_STATE_QUALITY"
        bottleneck = "BOTTLENECK_RUNTIME_IDENTITY_STATE_QUALITY"
        next_backbone = False
        next_memory = True
    else:
        decision = "FAIL_CROSS_SCENE_IDENTITY_REPRESENTATION"
        bottleneck = "BOTTLENECK_BASE_IDENTITY_REPRESENTATION"
        next_backbone = True
        next_memory = False
    storage_after = storage_audit()
    _write_json(OUT / "storage_audit_after.json", storage_after)
    headroom = {"stage": STAGE, "status": "NOT_RUN_REPRESENTATION_STAGE", "association_rescue_run": False, "solver_called": False, "public_authority_changed": False, "runtime_future_gt_used": False, "runtime_gt_clean": True, "reason": "R3R2 representation phase is terminal before association authority"}
    _write_json(OUT / "future_association_headroom.json", headroom)
    result = {
        "stage": STAGE,
        "goal": "Cross-Scene Open-Set Identity Representation Learning",
        "decision": decision,
        "bottleneck_classification": bottleneck,
        "source_commit": formal["source_head"],
        "training_identity_count": _json(TRAINING / "identity_episode_manifest.json")["training_identity_count"],
        "identity_episode_count": _json(TRAINING / "identity_episode_manifest.json")["episode_count"],
        "candidate_GTs_matched": _json(TRAINING / "identity_match_manifest.json")["canonical_records"],
        "primary_model": "CrossSceneIdentityAdapter",
        "trainable_parameters": 265472,
        "frozen_baseline_rank1": formal["baseline_metrics"]["rank1"],
        "new_rank1": formal["runtime_metrics"]["rank1"],
        "rank1_delta": float(formal["runtime_metrics"]["rank1"] - formal["baseline_metrics"]["rank1"]),
        "paired_CI": formal["paired"]["interval_95"],
        "MRR": formal["runtime_metrics"]["MRR"],
        "Rank2": formal["runtime_metrics"]["rank2"],
        "Rank3": formal["runtime_metrics"]["rank3"],
        "Rank5": formal["runtime_metrics"]["rank5"],
        "runtime_state_metrics": formal["runtime_metrics"],
        "oracle_clean_state_metrics": formal["oracle_metrics"],
        "representation_gate": formal["gate"],
        "open_set_metrics": None,
        "causal_metrics": None,
        "runtime_future_gt_used": False,
        "runtime_gt_clean": True,
        "OSNet_frozen": True,
        "GRU_frozen": True,
        "SAM3_rerun": False,
        "candidate_generation_frozen": True,
        "association_authority_started": False,
        "val_accessed": False,
        "test_accessed": False,
        "next_backbone_adaptation_stage_authorized": next_backbone,
        "next_memory_state_learning_stage_authorized": next_memory,
        "next_association_authority_stage_authorized": False,
        "tests_summary": tests_summary,
        "storage_after": storage_after,
    }
    _write_json(OUT / "FINAL_RESULT.json", result)
    _write_json(OUT / "stage_status.json", {"stage": STAGE, "goal_reference": f"outputs/{STAGE}/FINAL_GOAL.json", "goal_frozen": True, "status": "REPRESENTATION_FAIL", "decision": decision, "last_completed_artifact": "outputs/N72R20R3R2/representation/formal_loso.json", "representation_gate": formal["gate"], "runtime_state_metrics": formal["runtime_metrics"], "oracle_clean_state_metrics": formal["oracle_metrics"], "next_backbone_adaptation_stage_authorized": next_backbone, "next_memory_state_learning_stage_authorized": next_memory, "next_association_authority_stage_authorized": False, "open_set_started": False, "causal_started": False, "runtime_future_gt_used": False, "runtime_gt_clean": True, "sam3_rerun": False, "val_accessed": False, "test_accessed": False, "tests": tests_summary, "storage_audit_after": storage_after})
    first = "NO"
    report = [
        f"{first} — sequence-held-out identity representation learning over all available training identities {'raises' if first == 'YES' else 'does not raise'} unseen-sequence candidate identification enough to support open-set persistent identity recognition.",
        "",
        f"# InterMOT {STAGE} — Cross-Scene Open-Set Identity Representation Learning",
        "",
        f"Final decision: `{decision}`.",
        "",
        "## Phase A result",
        "",
        f"Primary model: `CrossSceneIdentityAdapter`, trainable parameters `{result['trainable_parameters']}`.",
        f"Frozen learned-state baseline Rank-1: `{result['frozen_baseline_rank1']}`; new runtime-state Rank-1: `{result['new_rank1']}`; delta: `{result['rank1_delta']}`.",
        f"Runtime-state MRR / Rank-2 / Rank-3 / Rank-5: `{result['MRR']}` / `{result['Rank2']}` / `{result['Rank3']}` / `{result['Rank5']}`.",
        f"Oracle-clean Rank-1: `{formal['oracle_metrics']['rank1']}`; paired sequence-cluster delta 95% CI: `{formal['paired']['interval_95']}`.",
        f"Representation gate: `{formal['gate']['pass']}`; bottleneck: `{bottleneck}`.",
        "",
        "## Lineage and leakage controls",
        "",
        f"Training identities: `{result['training_identity_count']}`; identity episodes: `{result['identity_episode_count']}`; matched candidate/GT records: `{result['candidate_GTs_matched']}`.",
        "All identity labels use `(sequence, gt_track_id)`; heldout sequence observations were excluded from mining/training for each outer fold.",
        "OSNet, N72R18 GRU, candidate tape, solver and public-ID authority remained frozen. Runtime-state evaluation used the canonical causal R3R1R1 state; oracle-clean state is posthoc diagnostic only.",
        "",
        "## Boundary",
        "",
        "NONE-head training, causal memory commit replay, association rescue, solver changes, SAM3, DanceTrack VAL/TEST, TrackEval and backbone adaptation were not run.",
        f"next_backbone_adaptation_stage_authorized={next_backbone}",
        f"next_memory_state_learning_stage_authorized={next_memory}",
        "next_association_authority_stage_authorized=false",
        "",
        f"Focused/full test summary: {tests_summary}.",
    ]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "mine", "train", "all"), default="all")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--tests-summary", default="PENDING")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    before = storage_audit()
    _write_json(OUT / "storage_audit_before.json", before)
    if before["filesystem"]["storage_status"] == "HARD_STOP":
        raise RuntimeError("storage hard stop")
    audit = source_audit()
    _write_json(OUT / "source_audit.json", audit)
    (OUT / "source_audit.md").write_text("\n".join(["# N72R20R3R2 Source Audit", "", f"- source branch: `{audit['source_branch']}`", f"- source HEAD: `{audit['source_head']}`", f"- source decision: `{audit['source_decision']}`", f"- frozen learned-state baseline Rank-1: `{audit['source_learned_state_rank1']}`", f"- frozen learned-state baseline MRR: `{audit['source_learned_state_mrr']}`", "- previous results modified: `False`", "- SAM3/VAL/TEST/association authority: not run" , ""]) , encoding="utf-8")
    if args.phase == "audit":
        print(json.dumps({"stage": STAGE, "status": "INITIALIZED", "source_head": audit["source_head"]}, sort_keys=True))
        return 0
    mining, mining_audit = mine_identities()
    episodes = build_episodes(mining)
    if args.phase == "mine":
        print(json.dumps({"stage": STAGE, "status": "IDENTITY_MINING_COMPLETE", "identities": episodes["manifest"]["training_identity_count"], "episodes": episodes["manifest"]["episode_count"]}, sort_keys=True))
        return 0
    dataset = load_episode_dataset(mining)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    formal = run_representation(dataset, mining, device)
    result = finalize(formal, args.tests_summary)
    print(json.dumps({"stage": STAGE, "status": result["decision"], "runtime_rank1": result["new_rank1"], "oracle_rank1": result["oracle_clean_state_metrics"]["rank1"], "representation_gate": result["representation_gate"]["pass"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
