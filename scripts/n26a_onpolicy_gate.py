#!/usr/bin/env python3
"""N26-A: same-policy causal B10 trajectories and sequence-OOF safety gate.

This script never reads val25.  The frozen N25-R B10 ranker is replayed with
the same simulated-human policy on train30 and cal10.  Current feedback is
applied only after the current score.  Small existence/commit heads are fit on
train30; each cal10 sequence receives a threshold selected from the other
cal10 sequences only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(".")
OUT = ROOT / "outputs/n26"
CHECKPOINTS = OUT / "checkpoints"
DATA_ROOT = Path("/path/to/dancetrack")
SEED = 26
HORIZON = 5
B10_LAMBDA = 0.8
B10_DELTA = 0.02
POS_CAP = 8
NEG_CAP = 16
MIN_PRECISION = 0.90
MIN_COVERAGE = 0.05
MAX_ABSENT_FA = 0.0726
CS = (0.001, 0.01, 0.1, 1.0)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from n25r_r5_gate import group_rank_summary, load_split, normalize, temporal_clip  # noqa: E402


FEATURE_NAMES = [
    "b10_top",
    "b10_margin",
    "b10_entropy",
    "b10_score_mean",
    "b10_score_std",
    "candidate_count",
    "valid_score_count",
    "query_age",
    "top_selected_rank_fraction",
    "top_query_similarity",
    "top_negative_penalty",
    "positive_memory_similarity_max",
    "positive_memory_similarity_mean",
    "negative_memory_similarity_max",
    "negative_memory_similarity_mean",
    "positive_memory_count",
    "negative_memory_count",
    "positive_memory_age",
    "negative_memory_age",
    "top_shadow_valid_ratio",
    "group_shadow_valid_ratio_mean",
    "group_shadow_valid_ratio_min",
    "top_temporal_similarity_mean",
    "top_temporal_similarity_std",
    "top_temporal_similarity_first",
    "top_temporal_similarity_last",
    "top_motion_speed_mean",
    "top_motion_acceleration_mean",
    "top_neighbor_crowd_mean",
    "top_neighbor_distance_mean",
    "top_neighbor_overlap_mean",
    "top_neighbor_density_mean",
    "gfn_top",
    "gfn_margin",
    "r0_top",
    "r0_margin",
    "b2_top",
    "b2_margin",
    "missing_top_clip",
    "missing_positive_memory",
    "missing_negative_memory",
    "missing_top_motion",
    "missing_top_neighbor",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class GroundTruth:
    def __init__(self, sequence: str):
        self.sequence = sequence
        self.by_frame: dict[int, dict[int, list[float]]] = defaultdict(dict)
        path = DATA_ROOT / "train" / sequence / "gt" / "gt.txt"
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split(",")
            if len(fields) < 6:
                continue
            frame = int(float(fields[0])) - 1
            gid = int(float(fields[1]))
            x, y, width, height = map(float, fields[2:6])
            if width > 0 and height > 0:
                self.by_frame[frame][gid] = [x, y, x + width, y + height]
        seqinfo = DATA_ROOT / "train" / sequence / "seqinfo.ini"
        self.length = 0
        for line in seqinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("seqLength="):
                self.length = int(line.split("=", 1)[1])

    def valid_frame(self, frame: int) -> bool:
        return 0 <= frame < self.length

    def visible(self, frame: int, gid: int) -> bool | None:
        if not self.valid_frame(frame):
            return None
        return gid in self.by_frame.get(frame, {})


def score_at(row: dict[str, Any], method: str) -> float:
    value = row["scores"][method].get(str(HORIZON))
    return float(value) if value is not None else math.nan


def top_margin(rows: list[dict[str, Any]], indices: list[int], method: str) -> tuple[float, float]:
    values = [score_at(rows[index], method) for index in indices]
    values = sorted((value for value in values if math.isfinite(value)), reverse=True)
    if not values:
        return math.nan, math.nan
    return values[0], values[0] - values[1] if len(values) > 1 else 0.0


def finite_stat(values: list[float], kind: str) -> float:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not len(array):
        return math.nan
    if kind == "max":
        return float(array.max())
    if kind == "mean":
        return float(array.mean())
    raise ValueError(kind)


def state_for_group(
    rows: list[dict[str, Any]], indices: list[int], ground_truth: GroundTruth
) -> tuple[str, dict[str, Any]]:
    first = rows[indices[0]]
    frame = int(first["decision_frame"])
    gid = int(first["gid"])
    query = first["legal_human_positive"]
    query_frame = int(query["frame"])
    query_gt = ground_truth.by_frame.get(query_frame, {}).get(gid)
    query_iou = None if query_gt is None else iou(query["box"], query_gt)
    identity_mapping_reliable = query_iou is not None and query_iou >= 0.5
    visible = ground_truth.visible(frame, gid)
    candidate_present = any(bool(rows[index]["positive"]) for index in indices)
    if not identity_mapping_reliable or visible is None:
        state = "UNKNOWN"
    elif visible and candidate_present:
        state = "VISIBLE_AND_CANDIDATE_PRESENT"
    elif visible and not candidate_present:
        state = "VISIBLE_BUT_CANDIDATE_MISSING"
    elif not visible and not candidate_present:
        state = "TARGET_NOT_VISIBLE_OR_ABSENT"
    else:
        state = "UNKNOWN"
    return state, {
        "raw_gt_frame": frame,
        "raw_gt_visible": visible,
        "candidate_set_positive_present": candidate_present,
        "query_to_gt_iou": query_iou,
        "identity_mapping_reliable": identity_mapping_reliable,
    }


def replay_split(data: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = data["rows"]
    clip = temporal_clip(data["features"], HORIZON)
    query = normalize(data["features"]["clip_query"].copy())
    embedding = clip["embedding"]
    ok = clip["ok"]
    scores = np.full(len(rows), np.nan, dtype=np.float64)
    events: list[dict[str, Any]] = []
    memory_ledger: list[dict[str, Any]] = []
    memories: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"positive": [], "negative": []}
    )
    event_groups = []
    for key, indices in data["groups"].items():
        first = rows[indices[0]]
        event_groups.append(((str(first["sequence"]), int(first["decision_frame"]), int(first["gid"])), indices))
    gt_cache: dict[str, GroundTruth] = {}
    writes = Counter()
    reads = Counter()
    correction_events = 0
    previous_sequence = None
    sequence_resets = 0

    for (sequence, frame, gid), indices in sorted(event_groups, key=lambda item: item[0]):
        if sequence != previous_sequence:
            sequence_resets += 1
            previous_sequence = sequence
        first = rows[indices[0]]
        public_id = first.get("public_identity_id")
        if public_id is None:
            public_id = gid
        scope = (sequence, int(public_id))
        memory = memories[scope]
        if memory["positive"]:
            reads["events_with_positive_memory"] += 1
        if memory["negative"]:
            reads["events_with_negative_memory"] += 1
        state, state_detail = state_for_group(
            rows, indices, gt_cache.setdefault(sequence, GroundTruth(sequence))
        )
        base_similarities: dict[int, float] = {}
        positive_sims_by_index: dict[int, list[float]] = {}
        negative_sims_by_index: dict[int, list[float]] = {}
        penalties: dict[int, float] = {}
        for index in indices:
            if not ok[index]:
                continue
            base = float(np.dot(query[index], embedding[index]))
            pos_sims = [float(np.dot(item["embedding"], embedding[index])) for item in memory["positive"]]
            neg_sims = [float(np.dot(item["embedding"], embedding[index])) for item in memory["negative"]]
            positive_similarity = max([base] + pos_sims)
            penalty = max(0.0, max(neg_sims) - positive_similarity + B10_DELTA) if neg_sims else 0.0
            scores[index] = positive_similarity - B10_LAMBDA * penalty
            base_similarities[index] = base
            positive_sims_by_index[index] = pos_sims
            negative_sims_by_index[index] = neg_sims
            penalties[index] = penalty

        valid = [index for index in indices if math.isfinite(scores[index])]
        ordered = sorted(valid, key=lambda index: (-float(scores[index]), int(rows[index]["candidate_rank"])))
        selected = ordered[0] if ordered else None
        selected_correct = bool(rows[selected]["positive"]) if selected is not None else False
        values = np.asarray([scores[index] for index in valid], dtype=np.float64)
        if len(values):
            shifted = values - values.max()
            probability = np.exp(shifted) / np.exp(shifted).sum()
            entropy = float(-(probability * np.log(probability + 1e-12)).sum())
            score_mean, score_std = float(values.mean()), float(values.std())
            margin = float(scores[ordered[0]] - scores[ordered[1]]) if len(ordered) > 1 else 0.0
        else:
            entropy = score_mean = score_std = margin = math.nan

        selected_pos_sims = positive_sims_by_index.get(selected, []) if selected is not None else []
        selected_neg_sims = negative_sims_by_index.get(selected, []) if selected is not None else []
        valid_ratio = data["features"]["valid"][indices, :HORIZON].mean(axis=1)
        if selected is not None:
            step_valid = data["features"]["valid"][selected, :HORIZON]
            step_embedding = data["features"]["clip_candidate"][selected, :HORIZON]
            step_similarity = np.einsum("d,hd->h", query[selected], step_embedding)
            temporal = step_similarity[step_valid]
            motion = data["features"]["motion"][selected, :HORIZON]
            neighbor = data["features"]["neighbor"][selected, :HORIZON]
            selected_valid_ratio = float(step_valid.mean())
        else:
            step_valid = np.zeros(HORIZON, dtype=bool)
            temporal = np.asarray([], dtype=np.float64)
            motion = np.full((HORIZON, 9), np.nan)
            neighbor = np.full((HORIZON, 4), np.nan)
            selected_valid_ratio = math.nan

        def age(kind: str) -> float:
            return float(frame - int(memory[kind][-1]["frame"])) if memory[kind] else math.nan

        gfn_top, gfn_margin = top_margin(rows, indices, "B0_GFN")
        r0_top, r0_margin = top_margin(rows, indices, "B1_R0")
        b2_top, b2_margin = top_margin(rows, indices, "B2_GFN_R0")
        speed = np.linalg.norm(motion[:, 4:6], axis=1)
        acceleration = np.linalg.norm(motion[:, 6:8], axis=1)
        causal_features = {
            "b10_top": float(scores[selected]) if selected is not None else math.nan,
            "b10_margin": margin,
            "b10_entropy": entropy,
            "b10_score_mean": score_mean,
            "b10_score_std": score_std,
            "candidate_count": float(len(indices)),
            "valid_score_count": float(len(valid)),
            "query_age": float(frame - int(first["correction_frame"])),
            "top_selected_rank_fraction": float(rows[selected]["candidate_rank"]) / max(1, len(indices)) if selected is not None else math.nan,
            "top_query_similarity": base_similarities.get(selected, math.nan),
            "top_negative_penalty": penalties.get(selected, math.nan),
            "positive_memory_similarity_max": finite_stat(selected_pos_sims, "max"),
            "positive_memory_similarity_mean": finite_stat(selected_pos_sims, "mean"),
            "negative_memory_similarity_max": finite_stat(selected_neg_sims, "max"),
            "negative_memory_similarity_mean": finite_stat(selected_neg_sims, "mean"),
            "positive_memory_count": float(len(memory["positive"])),
            "negative_memory_count": float(len(memory["negative"])),
            "positive_memory_age": age("positive"),
            "negative_memory_age": age("negative"),
            "top_shadow_valid_ratio": selected_valid_ratio,
            "group_shadow_valid_ratio_mean": float(valid_ratio.mean()),
            "group_shadow_valid_ratio_min": float(valid_ratio.min()),
            "top_temporal_similarity_mean": float(temporal.mean()) if len(temporal) else math.nan,
            "top_temporal_similarity_std": float(temporal.std()) if len(temporal) else math.nan,
            "top_temporal_similarity_first": float(temporal[0]) if len(temporal) else math.nan,
            "top_temporal_similarity_last": float(temporal[-1]) if len(temporal) else math.nan,
            "top_motion_speed_mean": float(np.nanmean(speed[step_valid])) if step_valid.any() else math.nan,
            "top_motion_acceleration_mean": float(np.nanmean(acceleration[step_valid])) if step_valid.any() else math.nan,
            "top_neighbor_crowd_mean": float(np.nanmean(neighbor[step_valid, 0])) if step_valid.any() else math.nan,
            "top_neighbor_distance_mean": float(np.nanmean(neighbor[step_valid, 1])) if step_valid.any() else math.nan,
            "top_neighbor_overlap_mean": float(np.nanmean(neighbor[step_valid, 2])) if step_valid.any() else math.nan,
            "top_neighbor_density_mean": float(np.nanmean(neighbor[step_valid, 3])) if step_valid.any() else math.nan,
            "gfn_top": gfn_top,
            "gfn_margin": gfn_margin,
            "r0_top": r0_top,
            "r0_margin": r0_margin,
            "b2_top": b2_top,
            "b2_margin": b2_margin,
            "missing_top_clip": float(selected is None),
            "missing_positive_memory": float(not memory["positive"]),
            "missing_negative_memory": float(not memory["negative"]),
            "missing_top_motion": float(selected is None or not step_valid.any()),
            "missing_top_neighbor": float(selected is None or not step_valid.any()),
        }
        if list(causal_features) != FEATURE_NAMES:
            raise RuntimeError("causal feature schema drift")
        event_key = f"{sequence}:{frame}:{gid}"
        event = {
            "split": data["name"],
            "event_key": event_key,
            "sequence": sequence,
            "decision_frame": frame,
            "gid": gid,
            "public_identity_id": int(public_id),
            "state_label": state,
            **state_detail,
            "selected_candidate_rank": None if selected is None else int(rows[selected]["candidate_rank"]),
            "selected_correct": selected_correct,
            "candidate_set_absent": not any(bool(rows[index]["positive"]) for index in indices),
            "feedback_visible_to_current_prediction": False,
            "causal_features": causal_features,
        }
        events.append(event)

        if selected is not None and not selected_correct and state != "UNKNOWN":
            correction_events += 1
            negative_record = {
                "split": data["name"], "event_key": event_key, "sequence": sequence,
                "frame": frame, "public_identity_id": int(public_id),
                "candidate_rank": int(rows[selected]["candidate_rank"]),
                "memory_kind": "HUMAN_EXPLICIT_NEGATIVE",
                "source": "FROZEN_B10_SELECTED_THEN_SIMULATED_HUMAN_REJECTED",
                "feature_valid": bool(ok[selected]), "applies_from_next_event_only": True,
            }
            memory_ledger.append(negative_record)
            if ok[selected]:
                memory["negative"].append({"embedding": embedding[selected].copy(), "frame": frame, **negative_record})
                memory["negative"] = memory["negative"][-NEG_CAP:]
                writes["negative"] += 1
            else:
                writes["negative_invalid"] += 1
            positives = [index for index in indices if bool(rows[index]["positive"]) and ok[index]]
            if positives:
                positive = min(positives, key=lambda index: int(rows[index]["candidate_rank"]))
                positive_record = {
                    "split": data["name"], "event_key": event_key, "sequence": sequence,
                    "frame": frame, "public_identity_id": int(public_id),
                    "candidate_rank": int(rows[positive]["candidate_rank"]),
                    "memory_kind": "HUMAN_EXPLICIT_POSITIVE",
                    "source": "SIMULATED_HUMAN_CORRECTED_TARGET_AFTER_CURRENT_ERROR",
                    "feature_valid": True, "applies_from_next_event_only": True,
                }
                memory_ledger.append(positive_record)
                memory["positive"].append({"embedding": embedding[positive].copy(), "frame": frame, **positive_record})
                memory["positive"] = memory["positive"][-POS_CAP:]
                writes["positive"] += 1
            else:
                writes["positive_unavailable"] += 1

    diagnostics = {
        "split": data["name"],
        "groups": len(events),
        "candidate_rows": len(rows),
        "sequences": len({event["sequence"] for event in events}),
        "sequence_resets": sequence_resets,
        "correction_events": correction_events,
        "memory_reads": dict(reads),
        "memory_writes": dict(writes),
        "state_counts": dict(Counter(event["state_label"] for event in events)),
        "current_feedback_used_for_current_prediction": False,
        "unselected_candidate_written_as_explicit_negative": False,
        "posthoc_hard_negative_written_as_explicit_negative": False,
        "state_isolation": "sequence+public_identity_id",
        "policy": "frozen_N25R_B10_H5_lambda_0.8",
        "simulator": "reject selected wrong candidate; provide matched positive only after current decision when available",
    }
    return scores, events, memory_ledger, diagnostics


def pipeline(c: float) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=SEED),
    )


def event_matrix(events: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[event["causal_features"][name] for name in FEATURE_NAMES] for event in events], dtype=np.float64)


def select_head(
    name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_sequence: np.ndarray,
    fit_mask: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    x, y, sequence = train_x[fit_mask], train_y[fit_mask], train_sequence[fit_mask]
    splitter = GroupKFold(n_splits=min(5, len(set(sequence))))
    trials = []
    best: tuple[float, float, np.ndarray] | None = None
    for c in CS:
        oof = np.full(len(y), np.nan, dtype=np.float64)
        for fit, held in splitter.split(x, y, groups=sequence):
            model = pipeline(c)
            model.fit(x[fit], y[fit])
            oof[held] = model.predict_proba(x[held])[:, 1]
        loss = float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6), labels=[0, 1]))
        brier = float(np.mean((oof - y) ** 2))
        trials.append({"C": c, "sequence_oof_log_loss": loss, "sequence_oof_brier": brier})
        objective = (loss, brier, c)
        if best is None or objective < (best[0], float(np.mean((best[2] - y) ** 2)), best[1]):
            best = (loss, c, oof)
    if best is None:
        raise RuntimeError(f"no {name} head selected")
    model = pipeline(best[1])
    model.fit(x, y)
    path = CHECKPOINTS / f"n26a_{name}.joblib"
    joblib.dump(model, path)
    manifest = {
        "name": name,
        "family": "impute+standardize+regularized_logistic",
        "selected_C": best[1],
        "selection": "minimum five-fold train30 sequence-OOF log loss; Brier and smaller C tie-break",
        "fit_samples": int(fit_mask.sum()),
        "fit_sequences": len(set(sequence)),
        "class_counts": dict(Counter(map(int, y))),
        "trials": trials,
        "feature_names": FEATURE_NAMES,
        "sequence_name_feature_used": False,
        "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(path),
        "cal10_gradient_used": False,
        "val25_read": False,
    }
    atomic_json(CHECKPOINTS / f"n26a_{name}.json", manifest)
    return model, manifest


def operating(events: list[dict[str, Any]], accepted: np.ndarray) -> dict[str, Any]:
    correct = np.asarray([event["selected_correct"] for event in events], dtype=bool)
    absent = np.asarray([event["candidate_set_absent"] for event in events], dtype=bool)
    accepted_count = int(accepted.sum())
    return {
        "events": len(events),
        "commits": accepted_count,
        "commit_precision": float(correct[accepted].mean()) if accepted_count else 1.0,
        "coverage": accepted_count / max(1, len(events)),
        "candidate_set_absent_events": int(absent.sum()),
        "candidate_set_absent_false_accept": float((accepted & absent).sum() / max(1, absent.sum())),
        "correct_commits": int((accepted & correct).sum()),
        "false_commits": int((accepted & ~correct).sum()),
    }


def select_threshold(events: list[dict[str, Any]], confidence: np.ndarray) -> dict[str, Any]:
    thresholds = np.unique(confidence[np.isfinite(confidence)])[::-1]
    best = None
    for threshold in thresholds:
        accepted = confidence >= threshold
        metrics = operating(events, accepted)
        if metrics["commit_precision"] >= MIN_PRECISION and metrics["candidate_set_absent_false_accept"] <= MAX_ABSENT_FA:
            candidate = {"threshold": float(threshold), **metrics}
            if best is None or (candidate["commits"], candidate["commit_precision"], candidate["threshold"]) > (
                best["commits"], best["commit_precision"], best["threshold"]
            ):
                best = candidate
    if best is None:
        return {"threshold": float("inf"), **operating(events, np.zeros(len(events), dtype=bool)), "status": "NO_FEASIBLE_THRESHOLD"}
    best["status"] = "FEASIBLE_ON_CALIBRATION_SEQUENCES"
    return best


def per_sequence_metrics(events: list[dict[str, Any]], accepted: np.ndarray, scores: np.ndarray, data: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for sequence in sorted({event["sequence"] for event in events}):
        event_indices = np.asarray([index for index, event in enumerate(events) if event["sequence"] == sequence])
        operation = operating([events[index] for index in event_indices], accepted[event_indices])
        keys = [event["event_key"].replace(":", ":", 1) for event in events if event["sequence"] == sequence]
        group_keys = [key for key, members in data["groups"].items() if str(data["rows"][members[0]]["sequence"]) == sequence]
        rank = group_rank_summary(data, scores, group_keys)
        output.append({"phase": "N26A", "split": data["name"], "sequence": sequence, **operation, **{f"rank_{key}": value for key, value in rank.items() if key in {"top1", "mrr", "pair_auc", "hardest_negative_margin"}}})
    return output


def bootstrap(events: list[dict[str, Any]], accepted: np.ndarray, n_boot: int = 2000) -> dict[str, Any]:
    sequences = sorted({event["sequence"] for event in events})
    indices = {sequence: np.asarray([i for i, event in enumerate(events) if event["sequence"] == sequence]) for sequence in sequences}
    rng = np.random.default_rng(SEED)
    values = defaultdict(list)
    for _ in range(n_boot):
        sampled = rng.choice(sequences, len(sequences), replace=True)
        sampled_events = []
        sampled_accepts = []
        for sequence in sampled:
            for index in indices[str(sequence)]:
                sampled_events.append(events[int(index)])
                sampled_accepts.append(bool(accepted[int(index)]))
        metrics = operating(sampled_events, np.asarray(sampled_accepts, dtype=bool))
        for key in ("commit_precision", "coverage", "candidate_set_absent_false_accept"):
            values[key].append(metrics[key])
    return {
        "resampling_unit": "sequence",
        "n_boot": n_boot,
        **{
            key: {
                "mean": float(np.mean(block)),
                "ci95_low": float(np.quantile(block, 0.025)),
                "ci95_high": float(np.quantile(block, 0.975)),
            }
            for key, block in values.items()
        },
    }


def full_attempt_state_audit(split: str, group_events: list[dict[str, Any]]) -> dict[str, Any]:
    attempts_path = ROOT / "outputs/n20" / f"dataset_attempts_{split}.csv"
    attempts = list(csv.DictReader(attempts_path.open(newline="", encoding="utf-8")))
    by_key = {event["event_key"]: event for event in group_events}
    gt_cache: dict[str, GroundTruth] = {}
    rows = []
    for attempt in attempts:
        sequence, frame, gid = attempt["sequence"], int(attempt["frame"]), int(attempt["gid"])
        key = f"{sequence}:{frame}:{gid}"
        visible = gt_cache.setdefault(sequence, GroundTruth(sequence)).visible(frame, gid)
        group = by_key.get(key)
        gallery_match = attempt.get("target_present") == "1"
        if visible is None:
            state = "UNKNOWN"
        elif group is not None:
            state = group["state_label"]
        elif visible and not gallery_match:
            state = "VISIBLE_BUT_CANDIDATE_MISSING"
        elif not visible and not gallery_match:
            state = "TARGET_NOT_VISIBLE_OR_ABSENT"
        else:
            state = "UNKNOWN"
        rows.append({"sequence": sequence, "frame": frame, "gid": gid, "raw_gt_visible": visible, "upstream_gallery_match": gallery_match, "top5_shadow_group_materialized": group is not None, "state_label": state})
    return {
        "split": split,
        "source": str(attempts_path.relative_to(ROOT)),
        "rows": len(rows),
        "state_counts": dict(Counter(row["state_label"] for row in rows)),
        "per_sequence": {
            sequence: dict(Counter(row["state_label"] for row in rows if row["sequence"] == sequence))
            for sequence in sorted({row["sequence"] for row in rows})
        },
        "materialized_model_group_state_counts": dict(Counter(event["state_label"] for event in group_events)),
        "existence_loss_eligible_model_groups": sum(event["state_label"] != "UNKNOWN" for event in group_events),
        "existence_loss_unknown_model_groups": sum(event["state_label"] == "UNKNOWN" for event in group_events),
        "target_present_field_semantics": "GFN gallery contains a GT-matched detection; audited source is analyze_n20_topk_availability.py, not scene visibility",
        "scene_visibility_source": "raw DanceTrack train GT at zero-based attempt frame",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    train = load_split("train30")
    cal = load_split("cal10")
    train_scores, train_events, train_ledger, train_diag = replay_split(train)
    cal_scores, cal_events, cal_ledger, cal_diag = replay_split(cal)

    for split, events in (("train30", train_events), ("cal10", cal_events)):
        path = OUT / f"on_policy_trajectory_{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, separators=(",", ":")) + "\n")
    with (OUT / "correction_memory_ledger.jsonl").open("w", encoding="utf-8") as handle:
        for row in train_ledger + cal_ledger:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    train_x, cal_x = event_matrix(train_events), event_matrix(cal_events)
    train_sequence = np.asarray([event["sequence"] for event in train_events])
    train_state = np.asarray([event["state_label"] for event in train_events])
    train_exist_y = (train_state == "VISIBLE_AND_CANDIDATE_PRESENT").astype(int)
    train_exist_mask = train_state != "UNKNOWN"
    train_commit_y = np.asarray([event["selected_correct"] for event in train_events], dtype=int)
    train_commit_mask = (train_state == "VISIBLE_AND_CANDIDATE_PRESENT") & np.asarray([event["selected_candidate_rank"] is not None for event in train_events])
    existence_model, existence_manifest = select_head("existence", train_x, train_exist_y, train_sequence, train_exist_mask)
    commit_model, commit_manifest = select_head("commit", train_x, train_commit_y, train_sequence, train_commit_mask)

    train_exist_p = existence_model.predict_proba(train_x)[:, 1]
    cal_exist_p = existence_model.predict_proba(cal_x)[:, 1]
    train_commit_p = commit_model.predict_proba(train_x)[:, 1]
    cal_commit_p = commit_model.predict_proba(cal_x)[:, 1]
    train_joint = train_exist_p * train_commit_p
    cal_joint = cal_exist_p * cal_commit_p

    cal_sequences = sorted({event["sequence"] for event in cal_events})
    cal_accepted = np.zeros(len(cal_events), dtype=bool)
    held_threshold = np.full(len(cal_events), np.inf, dtype=np.float64)
    fold_policies = {}
    for held_sequence in cal_sequences:
        fit_indices = np.asarray([index for index, event in enumerate(cal_events) if event["sequence"] != held_sequence])
        held_indices = np.asarray([index for index, event in enumerate(cal_events) if event["sequence"] == held_sequence])
        policy = select_threshold([cal_events[index] for index in fit_indices], cal_joint[fit_indices])
        threshold = float(policy["threshold"])
        held_threshold[held_indices] = threshold
        cal_accepted[held_indices] = cal_joint[held_indices] >= threshold
        fold_policies[held_sequence] = {"calibration_sequences": [sequence for sequence in cal_sequences if sequence != held_sequence], "held_sequence_labels_used": False, **policy}

    train_policy = select_threshold(train_events, train_joint)
    train_accepted = train_joint >= float(train_policy["threshold"])
    cal_operation = operating(cal_events, cal_accepted)
    train_operation = operating(train_events, train_accepted)
    descriptive_cal_policy = select_threshold(cal_events, cal_joint)

    prediction_rows = []
    for index, event in enumerate(cal_events):
        prediction_rows.append({
            "split": "cal10", "event_key": event["event_key"], "sequence": event["sequence"],
            "decision_frame": event["decision_frame"], "gid": event["gid"], "public_identity_id": event["public_identity_id"],
            "state_label": event["state_label"], "candidate_set_absent": int(event["candidate_set_absent"]),
            "selected_candidate_rank": event["selected_candidate_rank"], "selected_correct": int(event["selected_correct"]),
            "existence_probability_train30_fit": float(cal_exist_p[index]), "commit_probability_train30_fit": float(cal_commit_p[index]),
            "joint_confidence": float(cal_joint[index]), "loo_threshold": float(held_threshold[index]), "oof_commit": int(cal_accepted[index]),
            "held_sequence_label_used_for_threshold": 0,
        })
    write_csv(OUT / "n26a_oof_predictions.csv", prediction_rows)

    rank_train = group_rank_summary(train, train_scores)
    rank_cal = group_rank_summary(cal, cal_scores)
    old_b10_top1 = 0.545817
    rank_drop_pp = 100.0 * (old_b10_top1 - float(rank_cal["top1"]))
    commit_sequences = Counter(event["sequence"] for event, accepted in zip(cal_events, cal_accepted) if accepted)
    commit_count = int(cal_accepted.sum())
    max_sequence_fraction = max(commit_sequences.values(), default=0) / max(1, commit_count)
    criteria = {
        "commit_precision_at_least_90": cal_operation["commit_precision"] >= MIN_PRECISION,
        "coverage_at_least_5": cal_operation["coverage"] >= MIN_COVERAGE,
        "candidate_set_absent_false_accept_at_most_7_26": cal_operation["candidate_set_absent_false_accept"] <= MAX_ABSENT_FA,
        "not_reject_all": commit_count > 0,
        "commits_from_at_least_5_sequences": len(commit_sequences) >= 5,
        "no_sequence_above_50_percent_commits": max_sequence_fraction <= 0.5,
        "b10_h5_rank_drop_at_most_1pp": rank_drop_pp <= 1.0,
        "held_sequence_labels_not_used": True,
        "val25_not_read": True,
    }
    passed = all(criteria.values())
    per_sequence = per_sequence_metrics(cal_events, cal_accepted, cal_scores, cal)
    write_csv(OUT / "n26a_per_sequence.csv", per_sequence)
    write_csv(OUT / "per_sequence.csv", per_sequence)
    boot = bootstrap(cal_events, cal_accepted)
    atomic_json(OUT / "bootstrap_results.json", {"N26A": boot})

    risk_rows = []
    order = np.argsort(-cal_joint)
    for requested_coverage in np.linspace(0.01, 1.0, 100):
        count = max(1, int(math.ceil(requested_coverage * len(cal_events))))
        accepted = np.zeros(len(cal_events), dtype=bool)
        accepted[order[:count]] = True
        metrics = operating(cal_events, accepted)
        risk_rows.append({"phase": "N26A_DIAGNOSTIC_POSTHOC", "requested_coverage": float(requested_coverage), **metrics})
    write_csv(OUT / "risk_coverage.csv", risk_rows)

    state_audit = {
        "four_state_definitions": {
            "VISIBLE_AND_CANDIDATE_PRESENT": "raw GT target visible at decision and a correct identity is in frozen top-5",
            "VISIBLE_BUT_CANDIDATE_MISSING": "raw GT target visible but no correct identity is in frozen top-5",
            "TARGET_NOT_VISIBLE_OR_ABSENT": "raw GT target identity is not annotated at the valid decision frame and no positive candidate exists",
            "UNKNOWN": "identity/root mapping, frame validity, or GT/candidate evidence is contradictory or unreliable",
        },
        "candidate_missing_is_scene_absent": False,
        "unknown_silently_deleted": False,
        "splits": {
            "train30": full_attempt_state_audit("train30", train_events),
            "cal10": full_attempt_state_audit("cal10", cal_events),
        },
        "loss_mask_policy": "UNKNOWN retained in trajectory and excluded only from existence supervision; commit supervision is conditional on visible+candidate-present",
        "val25_read": False,
    }
    atomic_json(OUT / "state_label_audit.json", state_audit)
    atomic_json(OUT / "correction_memory_audit.json", {
        "train30": train_diag, "cal10": cal_diag,
        "ledger_rows": len(train_ledger) + len(cal_ledger),
        "negative_provenance": "only frozen-B10-selected then simulated-human-rejected candidates",
        "positive_provenance": "matched correction supplied after a current error",
        "current_feedback_changes_current_score": False,
        "sequence_memory_reset": True,
        "state_scope": "sequence+public_identity_id",
        "val25_read": False,
    })
    atomic_json(OUT / "on_policy_dataset_manifest.json", {
        "name": "N26-A frozen-B10 same-policy causal trajectories",
        "policy_version": "N26A_B10_H5_lambda0.8_sim_v1",
        "same_policy_train_cal": True,
        "candidate_stream": "N25-R repaired frozen GFN top-5",
        "train30": train_diag,
        "cal10": cal_diag,
        "feature_names": FEATURE_NAMES,
        "heads": {"existence": existence_manifest, "commit": commit_manifest},
        "artifacts": [
            "outputs/n26/on_policy_trajectory_train30.jsonl", "outputs/n26/on_policy_trajectory_cal10.jsonl",
            "outputs/n26/correction_memory_ledger.jsonl", "outputs/n26/state_label_audit.json",
        ],
        "gt_policy": "raw GT only for post-decision state labels/simulated human; never in causal feature vector",
        "cal10_gradient_used": False,
        "val25_read": False,
    })
    gate = {
        "phase": "N26A",
        "status": "PASS" if passed else "SCIENTIFIC_GATE_FAIL",
        "pass": passed,
        "criteria": criteria,
        "cal10_sequence_oof": cal_operation,
        "train30_descriptive": train_operation,
        "cal10_posthoc_diagnostic_not_deployable": descriptive_cal_policy,
        "fold_policies": fold_policies,
        "commit_sequences": dict(commit_sequences),
        "max_single_sequence_commit_fraction": max_sequence_fraction,
        "same_policy_b10_rank": {"train30": rank_train, "cal10": rank_cal, "historical_n25r_cal_top1": old_b10_top1, "rank_drop_pp": rank_drop_pp},
        "bootstrap": boot,
        "repair_rerun_used": False,
        "failure_class": None if passed else "SCIENTIFIC_FAILURE_UNLESS_INTEGRITY_VALIDATION_FINDS_IMPLEMENTATION_ERROR",
        "cal10_held_sequence_threshold_leakage": False,
        "val25_read": False,
        "next_route": "N26B_ALWAYS",
    }
    atomic_json(OUT / "n26a_gate.json", gate)
    print(json.dumps({"status": gate["status"], "cal10_oof": cal_operation, "criteria": criteria, "rank_cal_top1": rank_cal["top1"]}, indent=2), flush=True)
    print("N26A_DONE", flush=True)


if __name__ == "__main__":
    main()
