#!/usr/bin/env python3
"""Roll out frozen B10 on N27 external episodes and materialize causal states.

The current event is scored before any simulated feedback is written.  Only a
wrong candidate actually selected by B10 becomes a human explicit negative;
the corrected target crop becomes a human explicit positive when available.
An ordinary model-induced hard negative is maintained in a separate channel.

For correction-response evaluation, every state with a prior correction also
contains a counterfactual snapshot in which exactly the latest correction
(its positive and negative tokens) is removed while the candidate observation
is held fixed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(".")
OUT = ROOT / "outputs/n27"
DATA = OUT / "data"
REQUESTS = DATA / "episode_requests.jsonl"
MAX_CANDIDATES = 5
NONE_INDEX = 5
POS_CAP = 4
NEG_CAP = 8
HARD_CAP = 4
B10_LAMBDA = 0.8
B10_MARGIN = 0.02

DATASET_TO_INDEX = {"BDD100K": 0, "KITTI": 1, "MOT17": 2, "MOT20": 3}
SOURCE_TO_INDEX = {"GT_BOX": 0, "PUBLIC_DETECTOR_BOX": 1, "SAM3_DANCETRACK_REAL_CANDIDATE": 2}


@dataclass(frozen=True)
class Token:
    embedding_index: int
    frame: int
    correction_id: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def open_atomic_text(path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    return temporary, temporary.open("w", encoding="utf-8")


def finish_atomic_text(temporary: Path, handle, path: Path) -> None:
    handle.flush()
    os.fsync(handle.fileno())
    handle.close()
    os.replace(temporary, path)


class FeatureStore:
    def __init__(self) -> None:
        identifiers: list[np.ndarray] = []
        embeddings: list[np.ndarray] = []
        self.shards: list[dict[str, Any]] = []
        for path in sorted(DATA.glob("clipreid_shard*.npz")):
            with np.load(path, allow_pickle=False) as payload:
                crop_ids = payload["crop_id"].copy()
                features = payload["embedding"].copy()
            if features.ndim != 2 or features.shape[1] != 1280 or len(crop_ids) != len(features):
                raise RuntimeError(f"invalid feature shard {path}")
            offset = sum(len(item) for item in embeddings)
            identifiers.append(crop_ids)
            embeddings.append(features)
            self.shards.append({"path": str(path.relative_to(ROOT)), "rows": len(crop_ids), "offset": offset, "sha256": sha256(path)})
        if len(embeddings) != 4:
            raise RuntimeError(f"expected four feature shards, found {len(embeddings)}")
        all_ids = np.concatenate(identifiers)
        self.embedding = np.concatenate(embeddings).astype(np.float16, copy=False)
        self.index = {value.decode("ascii"): index for index, value in enumerate(all_ids)}
        if len(self.index) != len(self.embedding):
            raise RuntimeError("duplicate crop IDs in feature store")
        norms = np.linalg.norm(self.embedding.astype(np.float32), axis=1)
        if not np.isfinite(norms).all() or float(np.max(np.abs(norms - 1.0))) > 0.001:
            raise RuntimeError("feature normalization/finite audit failed")

    def lookup(self, crop_id: str) -> int:
        try:
            return self.index[crop_id]
        except KeyError as error:
            raise RuntimeError(f"feature missing for crop {crop_id}") from error

    def vectors(self, indices: Iterable[int]) -> np.ndarray:
        return self.embedding[np.asarray(list(indices), dtype=np.int64)].astype(np.float32)


def age_log(frame: int, tokens: list[Token]) -> float:
    if not tokens:
        return 1.0
    return min(1.0, math.log1p(max(0, frame - tokens[-1].frame)) / 8.0)


def max_similarity(candidates: np.ndarray, tokens: list[Token], store: FeatureStore) -> np.ndarray:
    if not tokens:
        return np.zeros(len(candidates), dtype=np.float32)
    memory = store.vectors(token.embedding_index for token in tokens)
    return np.max(candidates @ memory.T, axis=1).astype(np.float32)


def score_snapshot(
    *,
    candidates: np.ndarray,
    root: np.ndarray,
    frame: int,
    positive: list[Token],
    negative: list[Token],
    hard: list[Token],
    store: FeatureStore,
) -> dict[str, Any]:
    root_similarity = (candidates @ root).astype(np.float32)
    positive_similarity = max_similarity(candidates, positive, store)
    negative_similarity = max_similarity(candidates, negative, store)
    hard_similarity = max_similarity(candidates, hard, store)
    positive_base = np.maximum(root_similarity, positive_similarity) if positive else root_similarity.copy()
    penalty = np.maximum(0.0, negative_similarity - positive_base + B10_MARGIN) if negative else np.zeros_like(positive_base)
    b10 = positive_base - B10_LAMBDA * penalty
    return {
        "root_similarity": root_similarity,
        "positive_similarity": positive_similarity,
        "negative_similarity": negative_similarity,
        "hard_similarity": hard_similarity,
        "positive_base": positive_base,
        "b10_score": b10,
        "has_positive": bool(positive),
        "has_negative": bool(negative),
        "has_hard": bool(hard),
        "positive_count": len(positive) / POS_CAP,
        "negative_count": len(negative) / NEG_CAP,
        "hard_count": len(hard) / HARD_CAP,
        "positive_age": age_log(frame, positive),
        "negative_age": age_log(frame, negative),
        "hard_age": age_log(frame, hard),
    }


def collection() -> dict[str, list[Any]]:
    keys = [
        "candidate_mask", "target", "target_present", "b10_score", "root_similarity",
        "positive_similarity", "negative_similarity", "hard_similarity", "positive_base",
        "has_positive", "has_negative", "has_hard", "positive_count", "negative_count",
        "hard_count", "positive_age", "negative_age", "hard_age", "b10_margin",
        "b10_entropy", "candidate_count", "detector_score", "selected", "selected_correct",
        "correction_event", "pair_valid", "rejected_index", "latest_correction_id",
        "cf_b10_score", "cf_positive_similarity", "cf_negative_similarity", "cf_positive_base",
        "cf_has_positive", "cf_has_negative", "cf_positive_count", "cf_negative_count",
        "cf_positive_age", "cf_negative_age", "dataset", "sequence", "fold", "frame",
        "identity", "candidate_source", "parent_weight", "event_hash",
    ]
    return {key: [] for key in keys}


def padded(values: np.ndarray, fill: float = 0.0) -> np.ndarray:
    output = np.full(MAX_CANDIDATES, fill, dtype=np.float32)
    output[: len(values)] = values
    return output


def summarize_scores(scores: np.ndarray) -> tuple[float, float]:
    if len(scores) == 1:
        margin = 1.0
    else:
        order = np.sort(scores)
        margin = float(order[-1] - order[-2])
    shifted = scores - float(np.max(scores))
    probabilities = np.exp(shifted)
    probabilities /= max(float(probabilities.sum()), 1e-12)
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))) / math.log(MAX_CANDIDATES)
    return margin, entropy


def append_state(
    arrays: dict[str, list[Any]],
    *,
    event: dict[str, Any],
    snapshot: dict[str, Any],
    counterfactual: dict[str, Any],
    candidate_count: int,
    target: int,
    selected: int,
    correction_event: bool,
    pair_valid: bool,
    rejected_index: int,
    latest_correction_id: int,
    sequence_index: int,
    identity_index: int,
) -> None:
    mask = np.zeros(MAX_CANDIDATES, dtype=bool)
    mask[:candidate_count] = True
    detector = np.zeros(MAX_CANDIDATES, dtype=np.float32)
    detector[:candidate_count] = [float(row.get("detector_score", 0.0)) for row in event["candidates"]]
    margin, entropy = summarize_scores(snapshot["b10_score"])
    arrays["candidate_mask"].append(mask)
    arrays["target"].append(target)
    arrays["target_present"].append(target < MAX_CANDIDATES)
    for key in ("b10_score", "root_similarity", "positive_similarity", "negative_similarity", "hard_similarity", "positive_base"):
        arrays[key].append(padded(snapshot[key], -2.0 if key == "b10_score" else 0.0))
    for key in ("has_positive", "has_negative", "has_hard", "positive_count", "negative_count", "hard_count", "positive_age", "negative_age", "hard_age"):
        arrays[key].append(snapshot[key])
    arrays["b10_margin"].append(margin)
    arrays["b10_entropy"].append(entropy)
    arrays["candidate_count"].append(candidate_count / MAX_CANDIDATES)
    arrays["detector_score"].append(detector)
    arrays["selected"].append(selected)
    arrays["selected_correct"].append(selected == target)
    arrays["correction_event"].append(correction_event)
    arrays["pair_valid"].append(pair_valid)
    arrays["rejected_index"].append(rejected_index)
    arrays["latest_correction_id"].append(latest_correction_id)
    for key in ("b10_score", "positive_similarity", "negative_similarity", "positive_base"):
        arrays[f"cf_{key}"].append(padded(counterfactual[key], -2.0 if key == "b10_score" else 0.0))
    for key in ("has_positive", "has_negative", "positive_count", "negative_count", "positive_age", "negative_age"):
        arrays[f"cf_{key}"].append(counterfactual[key])
    arrays["dataset"].append(DATASET_TO_INDEX[event["dataset"]])
    arrays["sequence"].append(sequence_index)
    arrays["fold"].append(-1 if event.get("fold") is None else int(event["fold"]))
    arrays["frame"].append(int(event["decision_frame"]))
    arrays["identity"].append(identity_index)
    arrays["candidate_source"].append(SOURCE_TO_INDEX[event["candidate_source"]])
    arrays["parent_weight"].append(float(event["parent_weight"]))
    arrays["event_hash"].append(hashlib.sha256(event["event_key"].encode()).hexdigest()[:16].encode("ascii"))


def cast_arrays(rows: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    bool_keys = {"candidate_mask", "target_present", "has_positive", "has_negative", "has_hard", "selected_correct", "correction_event", "pair_valid", "cf_has_positive", "cf_has_negative"}
    int8_keys = {"target", "selected", "rejected_index", "dataset", "fold", "candidate_source"}
    int32_keys = {"frame", "identity", "latest_correction_id"}
    int16_keys = {"sequence"}
    float32_keys = {"parent_weight"}
    output: dict[str, np.ndarray] = {}
    for key, values in rows.items():
        if key in bool_keys:
            dtype = bool
        elif key in int8_keys:
            dtype = np.int8
        elif key in int16_keys:
            dtype = np.int16
        elif key in int32_keys:
            dtype = np.int32
        elif key in float32_keys:
            dtype = np.float32
        elif key == "event_hash":
            dtype = "S16"
        else:
            dtype = np.float16
        output[key] = np.asarray(values, dtype=dtype)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, default=REQUESTS)
    args = parser.parse_args()
    started = time.monotonic()
    store = FeatureStore()
    print(f"FEATURES rows={len(store.embedding)} shards={len(store.shards)}", flush=True)

    sequence_names: list[str] = []
    sequence_to_index: dict[tuple[str, str], int] = {}
    identity_to_index: dict[tuple[str, str, str], int] = {}
    memories: dict[tuple[str, str, str], dict[str, list[Token]]] = defaultdict(lambda: {"positive": [], "negative": [], "hard": []})
    outputs = {"external_train": collection(), "external_heldout": collection()}
    state_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    dataset_counts: Counter[tuple[str, str]] = Counter()
    source_counts: Counter[tuple[str, str]] = Counter()
    correction_counts: Counter[tuple[str, str]] = Counter()
    ledger_counts: Counter[str] = Counter()
    per_group: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    seen_events: set[str] = set()
    previous_scope: tuple[str, str, str] | None = None
    previous_frame = -1
    correction_id = 0

    ledger_path = OUT / "correction_ledger.jsonl"
    provenance_path = OUT / "candidate_provenance.jsonl"
    metadata_paths = {role: OUT / f"event_metadata_{role}.jsonl" for role in outputs}
    ledger_tmp, ledger_handle = open_atomic_text(ledger_path)
    provenance_tmp, provenance_handle = open_atomic_text(provenance_path)
    metadata_files = {role: open_atomic_text(path) for role, path in metadata_paths.items()}

    try:
        with args.requests.open(encoding="utf-8") as request_handle:
            for event_index, line in enumerate(request_handle):
                if not line.strip():
                    continue
                event = json.loads(line)
                event_key = event["event_key"]
                if event_key in seen_events:
                    raise RuntimeError(f"duplicate event key {event_key}")
                seen_events.add(event_key)
                if event.get("feedback_visible_to_current_prediction") is not False:
                    raise RuntimeError(f"noncausal request flag {event_key}")
                role = event["role"]
                if role not in outputs:
                    raise RuntimeError(f"unexpected role {role}")
                scope = (event["dataset"], event["video"], event["track_id"])
                frame = int(event["decision_frame"])
                if scope == previous_scope and frame <= previous_frame:
                    raise RuntimeError(f"non-increasing frame order in scope {scope}: {frame} <= {previous_frame}")
                previous_scope, previous_frame = scope, frame
                memory = memories[scope]
                if (event["dataset"], event["video"]) not in sequence_to_index:
                    sequence_to_index[(event["dataset"], event["video"])] = len(sequence_names)
                    sequence_names.append(f"{event['dataset']}:{event['video']}")
                if scope not in identity_to_index:
                    identity_to_index[scope] = len(identity_to_index)

                candidate_indices = [store.lookup(row["crop_id"]) for row in event["candidates"]]
                candidates = store.vectors(candidate_indices)
                root = store.embedding[store.lookup(event["root_crop_id"])].astype(np.float32)
                snapshot = score_snapshot(
                    candidates=candidates, root=root, frame=frame,
                    positive=memory["positive"], negative=memory["negative"], hard=memory["hard"], store=store,
                )
                selected = int(np.argmax(snapshot["b10_score"]))
                positive_indices = [index for index, row in enumerate(event["candidates"]) if bool(row["correct"])]
                if len(positive_indices) > 1:
                    raise RuntimeError(f"multiple correct candidates {event_key}")
                target = positive_indices[0] if positive_indices else NONE_INDEX
                correction_event = selected != target

                human_tokens = [*memory["positive"], *memory["negative"]]
                latest = max((token.correction_id for token in human_tokens), default=-1)
                if latest >= 0:
                    cf_positive = [token for token in memory["positive"] if token.correction_id != latest]
                    cf_negative = [token for token in memory["negative"] if token.correction_id != latest]
                else:
                    cf_positive, cf_negative = memory["positive"], memory["negative"]
                counterfactual = score_snapshot(
                    candidates=candidates, root=root, frame=frame,
                    positive=cf_positive, negative=cf_negative, hard=memory["hard"], store=store,
                )
                latest_negative = [token for token in memory["negative"] if token.correction_id == latest]
                pair_valid = bool(latest_negative)
                rejected_index = -1
                if pair_valid:
                    rejected_vector = store.embedding[latest_negative[-1].embedding_index].astype(np.float32)
                    rejected_index = int(np.argmax(candidates @ rejected_vector))

                append_state(
                    outputs[role], event=event, snapshot=snapshot, counterfactual=counterfactual,
                    candidate_count=len(candidates), target=target, selected=selected,
                    correction_event=correction_event, pair_valid=pair_valid,
                    rejected_index=rejected_index, latest_correction_id=latest,
                    sequence_index=sequence_to_index[(event["dataset"], event["video"])],
                    identity_index=identity_to_index[scope],
                )

                correction_written = False
                positive_written = False
                negative_written = False
                hard_index = -1
                event_correction_id = -1
                if correction_event:
                    correction_id += 1
                    event_correction_id = correction_id
                    memory["negative"].append(Token(candidate_indices[selected], frame, correction_id))
                    memory["negative"] = memory["negative"][-NEG_CAP:]
                    negative_written = correction_written = True
                    ledger_counts["HUMAN_EXPLICIT_NEGATIVE"] += 1
                    ledger_handle.write(json.dumps({
                        "correction_id": correction_id, "role": role, "event_key": event_key,
                        "dataset": event["dataset"], "video": event["video"], "track_id": event["track_id"],
                        "decision_frame": frame, "candidate_index_storage_only": selected,
                        "crop_id": event["candidates"][selected]["crop_id"],
                        "memory_kind": "HUMAN_EXPLICIT_NEGATIVE",
                        "source": "FROZEN_B10_SELECTED_WRONG_THEN_SIMULATED_HUMAN_REJECTED",
                        "selected_wrong": True, "applies_from_next_parent_only": True,
                    }, separators=(",", ":"), sort_keys=True) + "\n")
                    if target < MAX_CANDIDATES:
                        feedback_index = store.lookup(event["feedback_positive_crop_id"])
                        memory["positive"].append(Token(feedback_index, frame, correction_id))
                        memory["positive"] = memory["positive"][-POS_CAP:]
                        positive_written = True
                        ledger_counts["HUMAN_EXPLICIT_POSITIVE"] += 1
                        ledger_handle.write(json.dumps({
                            "correction_id": correction_id, "role": role, "event_key": event_key,
                            "dataset": event["dataset"], "video": event["video"], "track_id": event["track_id"],
                            "decision_frame": frame, "candidate_index_storage_only": target,
                            "crop_id": event["feedback_positive_crop_id"],
                            "memory_kind": "HUMAN_EXPLICIT_POSITIVE",
                            "source": "SIMULATED_HUMAN_CORRECT_TARGET_GT_CROP_AFTER_B10_ERROR",
                            "correct_target_available": True, "applies_from_next_parent_only": True,
                        }, separators=(",", ":"), sort_keys=True) + "\n")

                ordinary_wrong = [index for index in range(len(candidates)) if index != target and index != selected]
                if ordinary_wrong:
                    hard_index = max(ordinary_wrong, key=lambda index: float(snapshot["b10_score"][index]))
                    memory["hard"].append(Token(candidate_indices[hard_index], frame, -1))
                    memory["hard"] = memory["hard"][-HARD_CAP:]
                    ledger_counts["MODEL_INDUCED_HARD_NEGATIVE"] += 1
                    ledger_handle.write(json.dumps({
                        "correction_id": None, "role": role, "event_key": event_key,
                        "dataset": event["dataset"], "video": event["video"], "track_id": event["track_id"],
                        "decision_frame": frame, "candidate_index_storage_only": hard_index,
                        "crop_id": event["candidates"][hard_index]["crop_id"],
                        "memory_kind": "MODEL_INDUCED_HARD_NEGATIVE",
                        "source": "HIGHEST_B10_UNSELECTED_WRONG_CANDIDATE_SEPARATE_CONTROL",
                        "human_label": False, "applies_from_next_parent_only": True,
                    }, separators=(",", ":"), sort_keys=True) + "\n")

                for candidate_index, candidate in enumerate(event["candidates"]):
                    provenance_handle.write(json.dumps({
                        "event_key": event_key, "role": role, "dataset": event["dataset"],
                        "video": event["video"], "decision_frame": frame,
                        "candidate_index_storage_only": candidate_index,
                        "candidate_source": event["candidate_source"], "crop_id": candidate["crop_id"],
                        "correct_or_none_label": bool(candidate["correct"]),
                        "human_explicit_negative": bool(negative_written and candidate_index == selected),
                        "model_induced_hard_negative": bool(candidate_index == hard_index),
                        "unselected_distractor": bool(not candidate["correct"] and candidate_index not in {selected, hard_index}),
                        "augmentation": candidate["augmentation"],
                    }, separators=(",", ":"), sort_keys=True) + "\n")

                state_label = "TARGET_CANDIDATE_PRESENT" if target < MAX_CANDIDATES else "TARGET_CANDIDATE_MISSING"
                state_counts[state_label] += 1
                role_counts[role] += 1
                dataset_counts[(role, event["dataset"])] += 1
                source_counts[(role, event["candidate_source"])] += 1
                correction_counts[(role, event["dataset"])] += int(correction_event)
                group_counter = per_group[(role, event["dataset"], event["video"])]
                group_counter["parents"] += 1
                group_counter["present"] += int(target < MAX_CANDIDATES)
                group_counter["corrections"] += int(correction_event)
                group_counter["selected_correct"] += int(selected == target)
                group_counter["positive_writes"] += int(positive_written)
                group_counter["negative_writes"] += int(negative_written)
                group_counter["hard_writes"] += int(hard_index >= 0)

                metadata_tmp, metadata_handle = metadata_files[role]
                metadata_handle.write(json.dumps({
                    "parent_index": len(outputs[role]["target"]) - 1,
                    "event_key": event_key, "dataset": event["dataset"], "video": event["video"],
                    "track_id": event["track_id"], "decision_frame": frame, "fold": event.get("fold"),
                    "candidate_source": event["candidate_source"], "state_label": state_label,
                    "target": target, "round0_selected": selected, "round0_selected_correct": selected == target,
                    "correction_event": correction_event, "event_correction_id": event_correction_id,
                    "explicit_negative_written": negative_written, "explicit_positive_written": positive_written,
                    "ordinary_hard_negative_index": hard_index, "latest_prior_correction_id": latest,
                    "counterfactual_pair_valid": pair_valid,
                    "current_feedback_used_by_current_prediction": False,
                    "parent_weight": event["parent_weight"], "prefix_cluster_size": event["prefix_cluster_size"],
                    "target_state_reliable": event["target_state_reliable"],
                    "policy_version": "N27_EXTERNAL_ROUND0_FROZEN_B10_LAMBDA0.8_MARGIN0.02_V1",
                }, separators=(",", ":"), sort_keys=True) + "\n")

                if (event_index + 1) % 5000 == 0:
                    print(f"ROLLOUT events={event_index + 1} corrections={correction_id} elapsed_s={time.monotonic() - started:.1f}", flush=True)
    except Exception:
        ledger_handle.close()
        provenance_handle.close()
        for _, handle in metadata_files.values():
            handle.close()
        raise

    finish_atomic_text(ledger_tmp, ledger_handle, ledger_path)
    finish_atomic_text(provenance_tmp, provenance_handle, provenance_path)
    for role, path in metadata_paths.items():
        temporary, handle = metadata_files[role]
        finish_atomic_text(temporary, handle, path)

    npz_summaries = {}
    for role, rows in outputs.items():
        arrays = cast_arrays(rows)
        path = DATA / f"{role}_b10_round0.npz"
        atomic_npz(path, arrays)
        npz_summaries[role] = {
            "path": str(path.relative_to(ROOT)), "sha256": sha256(path), "parents": len(arrays["target"]),
            "parent_weight_sum": float(arrays["parent_weight"].sum()),
            "candidate_present": int(arrays["target_present"].sum()),
            "correction_events": int(arrays["correction_event"].sum()),
            "counterfactual_pairs": int(arrays["pair_valid"].sum()),
            "b10_present_top1": float(arrays["selected_correct"][arrays["target_present"]].mean()),
        }

    statistics_path = OUT / "data_statistics.csv"
    temporary = statistics_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["role", "dataset", "sequence", "parents", "present", "corrections", "selected_correct", "positive_writes", "negative_writes", "hard_writes"])
        writer.writeheader()
        for (role, dataset, sequence), counter in sorted(per_group.items()):
            writer.writerow({"role": role, "dataset": dataset, "sequence": sequence, **counter})
    os.replace(temporary, statistics_path)

    audit = {
        "phase": "N27", "policy": "frozen B10 causal external rollout",
        "memory_scope": ["dataset", "video", "target_identity"],
        "prediction_before_current_feedback": True,
        "current_feedback_visible_to_current_prediction": False,
        "human_negative_rule": "only B10-selected wrong candidate after prediction",
        "human_positive_rule": "correct target GT crop after wrong prediction and only when candidate target is available",
        "ordinary_hard_negative_separate": True,
        "latest_correction_counterfactual": "remove all positive and explicit-negative tokens sharing exactly the latest correction_id",
        "caps": {"positive": POS_CAP, "negative": NEG_CAP, "ordinary_hard": HARD_CAP},
        "events": len(seen_events), "unique_events": len(seen_events),
        "roles": dict(role_counts), "states": dict(state_counts),
        "ledger_counts": dict(ledger_counts), "correction_ids": correction_id,
        "npz": npz_summaries,
        "dataset_counts": {f"{role}:{dataset}": value for (role, dataset), value in sorted(dataset_counts.items())},
        "source_counts": {f"{role}:{source}": value for (role, source), value in sorted(source_counts.items())},
        "correction_counts": {f"{role}:{dataset}": value for (role, dataset), value in sorted(correction_counts.items())},
        "candidate_provenance": str(provenance_path.relative_to(ROOT)),
        "candidate_provenance_sha256": sha256(provenance_path),
        "correction_ledger": str(ledger_path.relative_to(ROOT)),
        "correction_ledger_sha256": sha256(ledger_path),
        "metadata": {role: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for role, path in metadata_paths.items()},
        "data_statistics": str(statistics_path.relative_to(ROOT)),
        "data_statistics_sha256": sha256(statistics_path),
        "val25_read": False,
    }
    atomic_json(OUT / "correction_memory_audit.json", audit)
    state_audit = {
        "phase": "N27", "state_names": ["TARGET_CANDIDATE_PRESENT", "TARGET_CANDIDATE_MISSING"],
        "counts": dict(state_counts), "all_states_reliable": True,
        "none_supervision_sources": ["dense MOT17/MOT20/KITTI/BDD100K annotations under frozen simulated candidate-missing protocol"],
        "tao_sparse_none_used": False, "unknown_states": 0,
        "ground_truth_used_as_current_residual_input": False,
        "ground_truth_uses": ["post-prediction label", "post-error simulated human-positive feedback", "train-only candidate construction"],
        "val25_read": False,
    }
    atomic_json(OUT / "state_label_audit.json", state_audit)

    manifest_path = OUT / "large_episode_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    disk = shutil.disk_usage("/data1")
    manifest.update({
        "status": "EXTERNAL_CAUSAL_B10_ROLLOUT_COMPLETE_DANCE_REAL_P2_PENDING",
        "causal_rollout": audit, "data1_free_bytes_after_external_rollout": disk.free,
        "reserve_satisfied_after_external_rollout": disk.free >= 40 * 1024**3,
        "val25_read": False,
    })
    atomic_json(manifest_path, manifest)
    elapsed = time.monotonic() - started
    print(json.dumps({
        "events": len(seen_events), "identities": len(identity_to_index), "sequences": len(sequence_names),
        "corrections": correction_id, "ledger_counts": dict(ledger_counts), "npz": npz_summaries,
        "elapsed_seconds": elapsed, "free_gib": disk.free / 1024**3, "val25_read": False,
    }, indent=2, sort_keys=True), flush=True)
    print("N27_EXTERNAL_CAUSAL_ROLLOUT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
