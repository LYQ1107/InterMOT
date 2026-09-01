#!/usr/bin/env python3
"""Materialize DanceTrack train Round-0 B10 states for N27 P2.

N26 already contains the frozen SAM3/DanceTrack real candidate stream and its
causal Round-0 memory snapshots.  This adapter uses only canonical parent
states (one per independent parent), recomputes the B10 anchor from the saved
pre-event memory, and gives the N27 residual the same feature schema as the
external causal episodes.  Prefix states and N26 replay rounds are never
counted as new parents.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(".")
N26 = ROOT / "outputs/n26/dense_dataset"
OUT = ROOT / "outputs/n27"
DATA = OUT / "data"
MAX_CANDIDATES = 5
NONE_INDEX = 5
B10_LAMBDA = 0.8
B10_MARGIN = 0.02


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def max_sim(candidates: np.ndarray, memory: np.ndarray) -> np.ndarray:
    return np.max(candidates @ memory.T, axis=1).astype(np.float32) if len(memory) else np.zeros(len(candidates), dtype=np.float32)


def score(candidates: np.ndarray, mask: np.ndarray, memory_clip: np.ndarray, memory_kind: np.ndarray, memory_mask: np.ndarray) -> dict[str, Any]:
    candidates = candidates.astype(np.float32)
    valid_memory = memory_mask.astype(bool)
    root = memory_clip[0].astype(np.float32)
    positive = memory_clip[valid_memory & (memory_kind == 1)].astype(np.float32)
    negative = memory_clip[valid_memory & (memory_kind == 2)].astype(np.float32)
    hard = memory_clip[valid_memory & (memory_kind == 3)].astype(np.float32)
    root_sim = candidates @ root
    pos_sim = max_sim(candidates, positive)
    neg_sim = max_sim(candidates, negative)
    hard_sim = max_sim(candidates, hard)
    base = np.maximum(root_sim, pos_sim) if len(positive) else root_sim.copy()
    penalty = np.maximum(0.0, neg_sim - base + B10_MARGIN) if len(negative) else np.zeros_like(base)
    b10 = base - B10_LAMBDA * penalty
    b10[~mask] = -2.0
    valid_scores = b10[mask]
    if len(valid_scores):
        order = np.sort(valid_scores)
        margin = float(order[-1] - order[-2]) if len(order) > 1 else 1.0
        shifted = valid_scores - float(valid_scores.max())
        probabilities = np.exp(shifted)
        probabilities /= max(float(probabilities.sum()), 1e-12)
        entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))) / math.log(MAX_CANDIDATES)
    else:
        margin, entropy = 0.0, 0.0
    return {
        "root_similarity": root_sim, "positive_similarity": pos_sim, "negative_similarity": neg_sim,
        "hard_similarity": hard_sim, "positive_base": base, "b10_score": b10,
        "has_positive": bool(len(positive)), "has_negative": bool(len(negative)), "has_hard": bool(len(hard)),
        "positive_count": min(1.0, len(positive) / 4.0), "negative_count": min(1.0, len(negative) / 8.0),
        "hard_count": min(1.0, len(hard) / 4.0), "positive_age": 1.0 if not len(positive) else 0.0,
        "negative_age": 1.0 if not len(negative) else 0.0, "hard_age": 1.0 if not len(hard) else 0.0,
        "b10_margin": margin, "b10_entropy": entropy,
    }


def pad(values: np.ndarray, fill: float = 0.0) -> np.ndarray:
    result = np.full(MAX_CANDIDATES, fill, dtype=np.float32)
    result[: len(values)] = values
    return result


def append_features(rows: dict[str, list[Any]], event: dict[str, Any], snapshot: dict[str, Any], counterfactual: dict[str, Any], candidate_mask: np.ndarray, target: int, selected: int, pair_valid: bool, rejected: int, sequence: int, identity: int, fold: int) -> None:
    rows["candidate_mask"].append(candidate_mask.astype(bool))
    rows["target"].append(target)
    rows["target_present"].append(target < MAX_CANDIDATES)
    for key in ("b10_score", "root_similarity", "positive_similarity", "negative_similarity", "hard_similarity", "positive_base"):
        rows[key].append(pad(snapshot[key], -2.0 if key == "b10_score" else 0.0))
    for key in ("has_positive", "has_negative", "has_hard", "positive_count", "negative_count", "hard_count", "positive_age", "negative_age", "hard_age"):
        rows[key].append(snapshot[key])
    rows["b10_margin"].append(snapshot["b10_margin"])
    rows["b10_entropy"].append(snapshot["b10_entropy"])
    rows["candidate_count"].append(float(candidate_mask.sum()) / MAX_CANDIDATES)
    detector = np.zeros(MAX_CANDIDATES, dtype=np.float32)
    detector[:] = event["detector_score"]
    rows["detector_score"].append(detector)
    rows["selected"].append(selected)
    rows["selected_correct"].append(selected == target)
    rows["correction_event"].append(selected != target)
    rows["pair_valid"].append(pair_valid)
    rows["rejected_index"].append(rejected)
    rows["latest_correction_id"].append(1 if pair_valid else -1)
    for key in ("b10_score", "positive_similarity", "negative_similarity", "positive_base"):
        rows[f"cf_{key}"].append(pad(counterfactual[key], -2.0 if key == "b10_score" else 0.0))
    for key in ("has_positive", "has_negative", "positive_count", "negative_count", "positive_age", "negative_age"):
        rows[f"cf_{key}"].append(counterfactual[key])
    rows["dataset"].append(4)
    rows["sequence"].append(sequence)
    rows["fold"].append(fold)
    rows["frame"].append(int(event["frame"]))
    rows["identity"].append(identity)
    rows["candidate_source"].append(2)
    rows["parent_weight"].append(1.0)
    rows["event_hash"].append(hashlib.sha256(event["event_key"].encode()).hexdigest()[:16].encode("ascii"))


def make_rows() -> dict[str, list[Any]]:
    keys = [
        "candidate_mask", "target", "target_present", "b10_score", "root_similarity", "positive_similarity",
        "negative_similarity", "hard_similarity", "positive_base", "has_positive", "has_negative", "has_hard",
        "positive_count", "negative_count", "hard_count", "positive_age", "negative_age", "hard_age",
        "b10_margin", "b10_entropy", "candidate_count", "detector_score", "selected", "selected_correct",
        "correction_event", "pair_valid", "rejected_index", "latest_correction_id", "cf_b10_score",
        "cf_positive_similarity", "cf_negative_similarity", "cf_positive_base", "cf_has_positive",
        "cf_has_negative", "cf_positive_count", "cf_negative_count", "cf_positive_age", "cf_negative_age",
        "dataset", "sequence", "fold", "frame", "identity", "candidate_source", "parent_weight", "event_hash",
    ]
    return {key: [] for key in keys}


def cast(rows: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    bools = {"candidate_mask", "target_present", "has_positive", "has_negative", "has_hard", "selected_correct", "correction_event", "pair_valid", "cf_has_positive", "cf_has_negative"}
    ints8 = {"target", "selected", "rejected_index", "dataset", "fold", "candidate_source"}
    ints16 = {"sequence"}
    ints32 = {"frame", "identity", "latest_correction_id"}
    result = {}
    for key, value in rows.items():
        if key in bools:
            dtype = bool
        elif key in ints8:
            dtype = np.int8
        elif key in ints16:
            dtype = np.int16
        elif key in ints32:
            dtype = np.int32
        elif key == "parent_weight":
            dtype = np.float32
        elif key == "event_hash":
            dtype = "S16"
        else:
            dtype = np.float16
        result[key] = np.asarray(value, dtype=dtype)
    return result


def main() -> None:
    source = N26 / "round0_train30.npz"
    parent_path = N26 / "round0_train30_parents.jsonl"
    with np.load(source, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    parents = [json.loads(line) for line in parent_path.open(encoding="utf-8") if line.strip()]
    if len(parents) != 1500:
        raise RuntimeError(f"expected 1500 DanceTrack train parents, found {len(parents)}")
    split = json.loads((OUT / "dataset_split_manifest.json").read_text(encoding="utf-8"))
    folds = {entry["video"]: entry["fold"] for entry in split["entries"] if entry["dataset"] == "DanceTrack" and entry["role"] == "train_fold"}
    sequence_names = sorted({row["sequence"] for row in parents})
    sequence_to_index = {name: index for index, name in enumerate(sequence_names)}
    identities = sorted({(row["sequence"], int(row["public_identity_id"])) for row in parents})
    identity_to_index = {key: index for index, key in enumerate(identities)}
    rows = make_rows()
    metadata = []
    ledger = []
    correction_count = 0
    pair_count = 0
    b10_reproduction_max = 0.0
    for parent_index, parent in enumerate(parents):
        state = int(parent["canonical_state_index"])
        candidate_clip = arrays["candidate_clip"][state].astype(np.float32)
        candidate_mask = arrays["candidate_mask"][state].astype(bool)
        candidate_mask &= np.arange(MAX_CANDIDATES) < 5
        memory_clip = arrays["memory_clip"][parent_index].astype(np.float32)
        memory_kind = arrays["memory_kind"][parent_index].astype(np.int8)
        memory_mask = arrays["memory_mask"][parent_index].astype(bool)
        memory_pre = arrays["memory_pre_mask"][parent_index].astype(bool)
        snapshot = score(candidate_clip, candidate_mask, memory_clip, memory_kind, memory_mask)
        valid = np.flatnonzero(candidate_mask)
        selected = int(valid[np.argmax(snapshot["b10_score"][valid])]) if len(valid) else NONE_INDEX
        expected_selected = int(parent["round0_selected"])
        if expected_selected < 0:
            expected_selected = NONE_INDEX
        if selected != expected_selected:
            raise RuntimeError(f"B10 reproduction mismatch parent={parent_index}: recomputed={selected}, frozen={expected_selected}")
        target = int(arrays["target"][state])
        counterfactual = score(candidate_clip, candidate_mask, memory_clip, memory_kind, memory_pre)
        pair_valid = bool(arrays["pair_valid"][state]) and bool(np.any(memory_pre != memory_mask))
        rejected = int(arrays["rejected_index"][state]) if pair_valid else -1
        if pair_valid:
            pair_count += 1
        root_direct = candidate_clip @ memory_clip[0]
        direct_positive = memory_clip[memory_mask & (memory_kind == 1)]
        direct_negative = memory_clip[memory_mask & (memory_kind == 2)]
        direct_base = np.maximum(root_direct, max_sim(candidate_clip, direct_positive)) if len(direct_positive) else root_direct
        direct_negative_max = max_sim(candidate_clip, direct_negative) if len(direct_negative) else np.zeros(MAX_CANDIDATES, dtype=np.float32)
        direct_penalty = np.maximum(0.0, direct_negative_max - direct_base + B10_MARGIN) if len(direct_negative) else np.zeros(MAX_CANDIDATES, dtype=np.float32)
        direct_score = direct_base - B10_LAMBDA * direct_penalty
        direct_score[~candidate_mask] = -2.0
        b10_reproduction_max = max(b10_reproduction_max, float(np.max(np.abs(snapshot["b10_score"] - direct_score))))
        event_key = str(parent["event_key"])
        event = {
            "event_key": event_key, "frame": int(parent["frame"]),
            "detector_score": arrays["candidate_scalar"][state, :, 5].astype(np.float32).tolist(),
        }
        append_features(rows, event, snapshot, counterfactual, candidate_mask, target, selected, pair_valid, rejected,
                        sequence_to_index[parent["sequence"]], identity_to_index[(parent["sequence"], int(parent["public_identity_id"]))],
                        int(folds.get(parent["sequence"], -1)))
        correction = selected != target
        correction_count += int(correction)
        if correction:
            ledger.append({
                "source": "N26_FROZEN_ROUND0_B10_REAL_DANCETRACK_REPLAY", "event_key": event_key,
                "sequence": parent["sequence"], "frame": int(parent["frame"]),
                "public_identity_id": int(parent["public_identity_id"]), "selected": selected, "target": target,
                "human_negative": selected < MAX_CANDIDATES, "human_positive": target < MAX_CANDIDATES,
                "applies_from_next_parent_only": True, "same_parent_round_cluster": event_key,
            })
        metadata.append({
            "parent_index": parent_index, "event_key": event_key, "sequence": parent["sequence"],
            "frame": int(parent["frame"]), "public_identity_id": int(parent["public_identity_id"]),
            "fold": folds.get(parent["sequence"]), "candidate_source": "SAM3_DANCETRACK_REAL_CANDIDATE",
            "target": target, "round0_selected": selected, "round0_selected_correct": selected == target,
            "correction_event": correction, "pair_valid": pair_valid, "prefix_states_not_counted": int(parent["real_temporal_states"]),
            "policy_version": "N27_DANCETRACK_REAL_ROUND0_FROZEN_B10_V1",
        })

    output = cast(rows)
    output_path = DATA / "dance_train_real_b10_round0.npz"
    atomic_npz(output_path, output)
    metadata_path = OUT / "dance_train_real_metadata.jsonl"
    temporary = metadata_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    os.replace(temporary, metadata_path)
    ledger_path = OUT / "dance_train_real_correction_ledger.jsonl"
    temporary = ledger_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in ledger:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    os.replace(temporary, ledger_path)
    summary = {
        "role": "dance_train_real_p2", "parents": len(parents), "sequences": len(sequence_names),
        "identities": len(identities), "correction_events": correction_count, "counterfactual_pairs": pair_count,
        "parent_weight_sum": float(output["parent_weight"].sum()), "b10_reproduction_max_abs_error": b10_reproduction_max,
        "candidate_source": "SAM3_DANCETRACK_REAL_CANDIDATE", "round0_only": True,
        "prefix_states_not_counted_as_parents": True, "round1_will_share_parent_cluster": True,
        "npz": str(output_path.relative_to(ROOT)), "npz_sha256": sha256(output_path),
        "metadata": str(metadata_path.relative_to(ROOT)), "metadata_sha256": sha256(metadata_path),
        "ledger": str(ledger_path.relative_to(ROOT)), "ledger_sha256": sha256(ledger_path), "val25_read": False,
    }
    atomic_json(OUT / "dance_train_real_p2_summary.json", summary)
    manifest_path = OUT / "large_episode_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"status": "CAUSAL_ROLLOUT_COMPLETE_DANCE_REAL_P2_COMPLETE", "dance_train_real_p2": summary, "val25_read": False})
    atomic_json(manifest_path, manifest)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("N27_DANCE_REAL_P2_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
