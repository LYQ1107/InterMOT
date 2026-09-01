#!/usr/bin/env python3
"""Run the repaired N25-R B0--B11 information and safety gate.

All learned parameters, B10 penalty weights, and commit rules are selected on
train30 only.  cal10 is evaluated once with those frozen choices.  val25 is
never opened.  Scene-level NONE is not identifiable in the inherited shadow
stream, so the report consistently uses CANDIDATE_SET_POSITIVE_ABSENT.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(".")
DATASET = ROOT / "outputs/n25r/repaired_dataset"
FEATURES = ROOT / "outputs/n25r/candidate_aligned_features"
OUT = ROOT / "outputs/n25r"
MODELS = OUT / "models/lightweight_r5"
SEED = 25
HORIZONS = (1, 5, 10)
PRIMARY_HORIZON = 5
MIN_PRECISION = 0.90
MIN_COVERAGE = 0.05
OLD_B2_GAP_PP = 46.02
MEMORY_POS_CAP = 8
MEMORY_NEG_CAP = 16
B10_DELTA = 0.02
B10_LAMBDAS = (0.05, 0.10, 0.20, 0.40, 0.80, 1.60)
B11_CS = (0.001, 0.01, 0.1)

METHODS = (
    "B0_GFN",
    "B1_R0",
    "B2_GFN_R0",
    "B3_RGB8_STATIC",
    "B4_DEEP_CLIPREID",
    "B5A_SAM3_BOX_ROI",
    "B6_MOTION",
    "B8_NEIGHBOR",
    "B10_EXPLICIT_NEGATIVE",
    "B11_ALL_LEGAL",
)
NOT_COMPUTABLE = {
    "B5B_SAM3_MASK_POOL": "alignment smoke failed mask-pool coverage/missingness gate after three grounded repairs",
    "B5C_SAM3_OBJ_PTR": "object pointer passed local coverage but full extraction was frozen because candidate binding/reset gate failed",
    "B5D_SAM3_MASK_MEMORY": "only candidate-pooled consolidated memory was observed and its alignment gate failed",
    "B5E_SAM3_LEGAL_FUSION": "only F1 survived the object-conditioned alignment gate; no multi-feature SAM3 fusion is claimed",
    "B7_LEGACY_RGB_MOTION": "legacy N25 fixed z-score combination is retained in repaired baseline artifacts, not rebranded as learned relation reasoning",
    "B9_LEGACY_RGB_MOTION_NEIGHBOR": "legacy N25 fixed z-score combination is retained in repaired baseline artifacts, not rebranded as B11",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def normalize(array: np.ndarray) -> np.ndarray:
    return array / (np.linalg.norm(array, axis=-1, keepdims=True) + 1e-12)


def group_indices(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["group_key"])].append(index)
    return dict(groups)


def load_feature_arrays(split: str, rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    n = len(rows)
    arrays = {
        "clip_query": np.full((n, 1280), np.nan, dtype=np.float32),
        "clip_candidate": np.full((n, 10, 1280), np.nan, dtype=np.float32),
        "f1_query_mean": np.full((n, 256), np.nan, dtype=np.float32),
        "f1_query_max": np.full((n, 256), np.nan, dtype=np.float32),
        "f1_candidate_mean": np.full((n, 10, 256), np.nan, dtype=np.float32),
        "f1_candidate_max": np.full((n, 10, 256), np.nan, dtype=np.float32),
        "valid": np.zeros((n, 10), dtype=bool),
    }
    seen = {"clipreid": np.zeros(n, dtype=bool), "sam3_f1": np.zeros(n, dtype=bool)}
    for backbone in ("clipreid", "sam3_f1"):
        for path in sorted((FEATURES / backbone / split).glob("*.npz")):
            done = path.with_suffix(".done")
            if not done.is_file():
                continue
            with np.load(path, allow_pickle=False) as cache:
                index = cache["row_indices"].astype(int)
                if seen[backbone][index].any():
                    raise RuntimeError(f"duplicate feature row in {backbone}/{split}")
                seen[backbone][index] = True
                if backbone == "clipreid":
                    arrays["clip_query"][index] = cache["query"]
                    arrays["clip_candidate"][index] = cache["candidate"]
                    arrays["valid"][index] = cache["candidate_valid"]
                else:
                    arrays["f1_query_mean"][index] = cache["query_mean"]
                    arrays["f1_query_max"][index] = cache["query_max"]
                    arrays["f1_candidate_mean"][index] = cache["candidate_mean"]
                    arrays["f1_candidate_max"][index] = cache["candidate_max"]
                    if not np.array_equal(arrays["valid"][index], cache["candidate_valid"]):
                        raise RuntimeError(f"cross-backbone valid mismatch {path}")
    if not all(value.all() for value in seen.values()):
        raise RuntimeError(f"incomplete feature merge {split}")
    raw = np.load(DATASET / f"raw_features_{split}.npz")
    arrays["motion"] = raw["motion"].astype(np.float32)
    arrays["neighbor"] = raw["neighbor"].astype(np.float32)
    raw.close()
    return arrays


def load_split(split: str) -> dict[str, Any]:
    path = DATASET / f"episodes_{split}.jsonl"
    rows = json_rows(path)
    return {
        "name": split,
        "path": path,
        "rows": rows,
        "groups": group_indices(rows),
        "features": load_feature_arrays(split, rows),
    }


def aggregate(candidate: np.ndarray, valid: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    prefix_valid = valid[:, :horizon]
    values = np.where(prefix_valid[:, :, None], candidate[:, :horizon], 0.0).sum(axis=1)
    values /= np.maximum(1, prefix_valid.sum(axis=1))[:, None]
    values = normalize(values)
    ok = prefix_valid.any(axis=1) & np.isfinite(values).all(axis=1)
    values[~ok] = np.nan
    return values, ok


def temporal_clip(features: dict[str, np.ndarray], horizon: int) -> dict[str, np.ndarray]:
    query = normalize(features["clip_query"].copy())
    candidate, ok = aggregate(features["clip_candidate"], features["valid"], horizon)
    mean_score = np.einsum("nd,nd->n", query, candidate)
    step_score = np.einsum("nd,nhd->nh", query, features["clip_candidate"][:, :horizon])
    step_valid = features["valid"][:, :horizon]
    step_score[~step_valid] = np.nan
    first = np.full(len(query), np.nan, dtype=np.float32)
    last = np.full(len(query), np.nan, dtype=np.float32)
    maximum = np.full(len(query), np.nan, dtype=np.float32)
    spread = np.full(len(query), np.nan, dtype=np.float32)
    for index in np.flatnonzero(ok):
        values = step_score[index, step_valid[index]]
        first[index], last[index] = values[0], values[-1]
        maximum[index], spread[index] = float(values.max()), float(values.std())
    return {
        "embedding": candidate,
        "ok": ok,
        "mean": mean_score,
        "first": first,
        "last": last,
        "maximum": maximum,
        "spread": spread,
    }


def temporal_f1(features: dict[str, np.ndarray], horizon: int) -> dict[str, np.ndarray]:
    query_mean = normalize(features["f1_query_mean"].copy())
    query_max = normalize(features["f1_query_max"].copy())
    candidate_mean, mean_ok = aggregate(features["f1_candidate_mean"], features["valid"], horizon)
    candidate_max, max_ok = aggregate(features["f1_candidate_max"], features["valid"], horizon)
    mean_score = np.einsum("nd,nd->n", query_mean, candidate_mean)
    max_score = np.einsum("nd,nd->n", query_max, candidate_max)
    ok = mean_ok & max_ok
    fused = (mean_score + max_score) / 2.0
    fused[~ok] = np.nan
    return {"mean": mean_score, "max": max_score, "fused": fused, "ok": ok}


def row_score(rows: list[dict[str, Any]], method: str, horizon: int) -> np.ndarray:
    output = np.full(len(rows), np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        value = row["scores"][method].get(str(horizon))
        if value is not None:
            output[index] = float(value)
    return output


def build_base_scores(data: dict[str, Any], horizon: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rows = data["rows"]
    features = data["features"]
    clip = temporal_clip(features, horizon)
    f1 = temporal_f1(features, horizon)
    neighbor = features["neighbor"][:, :horizon]
    valid = features["valid"][:, :horizon]
    conflict = np.where(valid[:, :, None], neighbor, np.nan)
    with np.errstate(invalid="ignore"):
        neighbor_score = -np.nanmean(conflict[:, :, [0, 2, 3]], axis=(1, 2))
    scores = {
        "B0_GFN": row_score(rows, "B0_GFN", horizon),
        "B1_R0": row_score(rows, "B1_R0", horizon),
        "B2_GFN_R0": row_score(rows, "B2_GFN_R0", horizon),
        "B3_RGB8_STATIC": row_score(rows, "B3_RAW_STATIC", horizon),
        "B4_DEEP_CLIPREID": clip["mean"].astype(np.float64),
        "B5A_SAM3_BOX_ROI": f1["fused"].astype(np.float64),
        "B6_MOTION": row_score(rows, "B6_MOTION", horizon),
        "B8_NEIGHBOR": neighbor_score.astype(np.float64),
    }
    return scores, {"clip": clip, "f1": f1}


def group_rank_summary(data: dict[str, Any], scores: np.ndarray, selected_groups: list[str] | None = None) -> dict[str, Any]:
    rows = data["rows"]
    groups = data["groups"]
    keys = list(groups) if selected_groups is None else selected_groups
    ranks = []
    margins = []
    wins = 0.0
    pairs = 0
    present = 0
    positive_evaluable = 0
    valid_groups = 0
    for key in keys:
        positive_all = [index for index in groups[key] if bool(rows[index]["positive"])]
        if positive_all:
            present += 1
        members = [index for index in groups[key] if np.isfinite(scores[index])]
        if not members:
            continue
        valid_groups += 1
        if not positive_all:
            continue
        positive = [index for index in members if bool(rows[index]["positive"])]
        negative = [index for index in members if not bool(rows[index]["positive"])]
        if not positive:
            continue
        positive_evaluable += 1
        ordered = sorted(members, key=lambda index: (-float(scores[index]), int(rows[index]["candidate_rank"])))
        ranks.append(next(rank + 1 for rank, index in enumerate(ordered) if bool(rows[index]["positive"])))
        if negative:
            margins.append(float(max(scores[index] for index in positive) - max(scores[index] for index in negative)))
        for positive_index in positive:
            for negative_index in negative:
                pairs += 1
                wins += 1.0 if scores[positive_index] > scores[negative_index] else 0.5 if scores[positive_index] == scores[negative_index] else 0.0
    rank_array = np.asarray(ranks, dtype=int)
    return {
        "valid_groups": valid_groups,
        "candidate_positive_present_groups": present,
        "positive_evaluable_groups": positive_evaluable,
        "candidate_positive_evaluable_recall": positive_evaluable / max(1, present),
        "top1": float(np.mean(rank_array == 1)) if len(rank_array) else None,
        "top3": float(np.mean(rank_array <= 3)) if len(rank_array) else None,
        "top5": float(np.mean(rank_array <= 5)) if len(rank_array) else None,
        "mrr": float(np.mean(1.0 / rank_array)) if len(rank_array) else None,
        "hardest_negative_margin": float(np.mean(margins)) if margins else None,
        "pair_auc": float(wins / pairs) if pairs else None,
        "pair_count": pairs,
    }


def correction_map(split: str) -> dict[tuple[str, int, int], list[dict[str, Any]]]:
    path = OUT / "human_negative_ledger" / (
        "simulated_train_human_explicit_negatives.jsonl"
        if split == "train30"
        else "canonical_human_explicit_negatives.jsonl"
    )
    output: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in json_rows(path):
        if split == "cal10" and not bool(row.get("admissible", False)):
            continue
        key = (str(row["sequence"]), int(row["decision_frame"]), int(row["gid"]))
        output[key].append(row)
    return dict(output)


def run_b10(
    data: dict[str, Any],
    clip: dict[str, np.ndarray],
    penalty_lambda: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    rows = data["rows"]
    groups = data["groups"]
    corrections = correction_map(data["name"])
    query = normalize(data["features"]["clip_query"].copy())
    embedding = clip["embedding"]
    ok = clip["ok"]
    scores = np.full(len(rows), np.nan, dtype=np.float64)
    negative_available = np.zeros(len(rows), dtype=np.float32)
    negative_penalty = np.zeros(len(rows), dtype=np.float32)
    positive_memory_available = np.zeros(len(rows), dtype=np.float32)
    memories: dict[tuple[str, int], dict[str, list[np.ndarray]]] = defaultdict(lambda: {"positive": [], "negative": []})
    event_groups: dict[tuple[str, int, int], list[int]] = {}
    for indices in groups.values():
        first = rows[indices[0]]
        event_groups[(str(first["sequence"]), int(first["decision_frame"]), int(first["gid"]))] = indices
    writes = Counter()
    correction_events_seen = 0
    for event, indices in sorted(event_groups.items(), key=lambda item: item[0]):
        first = rows[indices[0]]
        scope = (event[0], int(first["public_identity_id"]))
        memory = memories[scope]
        for index in indices:
            if not ok[index]:
                continue
            positive_similarity = float(np.dot(query[index], embedding[index]))
            if memory["positive"]:
                positive_memory_available[index] = 1.0
                positive_similarity = max(
                    positive_similarity,
                    max(float(np.dot(value, embedding[index])) for value in memory["positive"]),
                )
            penalty = 0.0
            if memory["negative"]:
                negative_available[index] = 1.0
                negative_similarity = max(float(np.dot(value, embedding[index])) for value in memory["negative"])
                penalty = max(0.0, negative_similarity - positive_similarity + B10_DELTA)
            negative_penalty[index] = penalty
            scores[index] = positive_similarity - penalty_lambda * penalty
        event_corrections = corrections.get(event, [])
        if event_corrections:
            correction_events_seen += 1
        rank_to_index = {int(rows[index]["candidate_rank"]): index for index in indices}
        for correction in event_corrections:
            negative_rank = int(correction["candidate_rank"])
            negative_index = rank_to_index.get(negative_rank)
            if negative_index is not None and ok[negative_index]:
                memory["negative"].append(embedding[negative_index].copy())
                memory["negative"] = memory["negative"][-MEMORY_NEG_CAP:]
                writes["negative"] += 1
            else:
                writes["negative_feature_unavailable"] += 1
            positive_rank = correction.get("positive_candidate_rank")
            if positive_rank is not None:
                positive_index = rank_to_index.get(int(positive_rank))
                if positive_index is not None and ok[positive_index]:
                    memory["positive"].append(embedding[positive_index].copy())
                    memory["positive"] = memory["positive"][-MEMORY_POS_CAP:]
                    writes["positive"] += 1
                else:
                    writes["positive_feature_unavailable"] += 1
    diagnostics = {
        "penalty_lambda": penalty_lambda,
        "delta": B10_DELTA,
        "correction_events_available": len(corrections),
        "correction_events_seen": correction_events_seen,
        "negative_writes": int(writes["negative"]),
        "positive_writes": int(writes["positive"]),
        "negative_feature_unavailable": int(writes["negative_feature_unavailable"]),
        "positive_feature_unavailable": int(writes["positive_feature_unavailable"]),
        "candidate_rows_with_prior_negative": int(negative_available.sum()),
        "candidate_rows_with_active_negative_penalty": int((negative_penalty > 0).sum()),
        "candidate_rows_with_prior_positive": int(positive_memory_available.sum()),
        "current_correction_used_for_current_score": False,
        "memory_caps": {"positive": MEMORY_POS_CAP, "negative": MEMORY_NEG_CAP},
    }
    return scores, {
        "negative_available": negative_available,
        "negative_penalty": negative_penalty,
        "positive_memory_available": positive_memory_available,
    }, diagnostics


def group_zscores(values: np.ndarray, groups: dict[str, list[int]]) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float64)
    for indices in groups.values():
        block = values[indices]
        for column in range(values.shape[1]):
            valid = np.isfinite(block[:, column])
            if not valid.any():
                continue
            mean = float(block[valid, column].mean())
            std = float(block[valid, column].std())
            if std < 1e-6:
                std = 1.0
            output[np.asarray(indices)[valid], column] = (block[valid, column] - mean) / std
    return output


def b11_features(
    data: dict[str, Any],
    scores: dict[str, np.ndarray],
    visual: dict[str, Any],
    b10_aux: dict[str, np.ndarray],
    horizon: int,
) -> tuple[np.ndarray, list[str]]:
    rows = data["rows"]
    features = data["features"]
    core_names = [
        "gfn", "r0", "gfn_r0", "rgb8", "clip_mean", "sam3_f1",
        "motion_baseline", "neighbor_baseline", "explicit_negative_score",
        "clip_first", "clip_last", "clip_maximum", "clip_temporal_spread",
        "sam3_roi_mean", "sam3_roi_max",
    ]
    core = np.column_stack(
        [
            scores["B0_GFN"], scores["B1_R0"], scores["B2_GFN_R0"],
            scores["B3_RGB8_STATIC"], scores["B4_DEEP_CLIPREID"], scores["B5A_SAM3_BOX_ROI"],
            scores["B6_MOTION"], scores["B8_NEIGHBOR"], scores["B10_EXPLICIT_NEGATIVE"],
            visual["clip"]["first"], visual["clip"]["last"], visual["clip"]["maximum"], visual["clip"]["spread"],
            visual["f1"]["mean"], visual["f1"]["max"],
        ]
    ).astype(np.float64)
    zscores = group_zscores(core, data["groups"])
    valid = features["valid"][:, :horizon]
    valid_ratio = valid.mean(axis=1)
    motion = features["motion"][:, :horizon]
    motion_speed = np.linalg.norm(motion[:, :, 4:6], axis=2)
    motion_acceleration = np.linalg.norm(motion[:, :, 6:8], axis=2)
    motion_speed = np.where(valid, motion_speed, np.nan)
    motion_acceleration = np.where(valid, motion_acceleration, np.nan)
    neighbor = np.where(valid[:, :, None], features["neighbor"][:, :horizon], np.nan)
    candidate_count = np.zeros(len(rows), dtype=np.float64)
    rank_fraction = np.zeros(len(rows), dtype=np.float64)
    for indices in data["groups"].values():
        count = len(indices)
        candidate_count[indices] = count
        for index in indices:
            rank_fraction[index] = int(rows[index]["candidate_rank"]) / max(1, count)
    extras = np.column_stack(
        [
            valid_ratio,
            candidate_count,
            rank_fraction,
            np.nanmean(motion_speed, axis=1),
            np.nanmean(motion_acceleration, axis=1),
            *[np.nanmean(neighbor[:, :, channel], axis=1) for channel in range(4)],
            b10_aux["negative_available"],
            b10_aux["negative_penalty"],
            b10_aux["positive_memory_available"],
        ]
    )
    extra_names = [
        "valid_ratio", "candidate_count", "candidate_rank_fraction", "mean_speed", "mean_acceleration",
        "neighbor_crowd", "neighbor_distance", "neighbor_overlap", "neighbor_density",
        "prior_explicit_negative_available", "negative_penalty", "prior_explicit_positive_available",
    ]
    validity = np.isfinite(core).astype(np.float64)
    names = core_names + [f"group_z_{name}" for name in core_names] + extra_names + [f"valid_{name}" for name in core_names]
    return np.column_stack([core, zscores, extras, validity]), names


def pipeline(c: float) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=SEED),
    )


def fit_b11(
    train: dict[str, Any],
    cal: dict[str, Any],
    train_x: np.ndarray,
    cal_x: np.ndarray,
    feature_names: list[str],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y = np.asarray([bool(row["positive"]) for row in train["rows"]], dtype=int)
    sequence = np.asarray([str(row["sequence"]) for row in train["rows"]])
    splitter = GroupKFold(n_splits=5)
    trials = []
    best = None
    for c in B11_CS:
        oof = np.full(len(y), np.nan, dtype=np.float64)
        for train_index, valid_index in splitter.split(train_x, y, groups=sequence):
            model = pipeline(c)
            model.fit(train_x[train_index], y[train_index])
            oof[valid_index] = model.predict_proba(train_x[valid_index])[:, 1]
        rank = group_rank_summary(train, oof)
        objective = (
            -1.0 if rank["top1"] is None else rank["top1"],
            -1.0 if rank["pair_auc"] is None else rank["pair_auc"],
            -1e9 if rank["hardest_negative_margin"] is None else rank["hardest_negative_margin"],
            -c,
        )
        trials.append({"C": c, **rank})
        if best is None or objective > best[0]:
            best = (objective, c, oof)
    if best is None:
        raise RuntimeError("no B11 model")
    selected_c = best[1]
    oof = best[2]
    model = pipeline(selected_c)
    model.fit(train_x, y)
    cal_score = model.predict_proba(cal_x)[:, 1]
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / f"b11_H{horizon}.joblib"
    joblib.dump(model, path)
    manifest = {
        "model": "regularized_logistic_all_legal_probe",
        "history": horizon,
        "parameter_upper_bound": len(feature_names) + 1,
        "selected_C": selected_c,
        "selection": "five-fold sequence-disjoint train30 OOF lexicographic top1/pair-AUC/margin; smaller C tie-break",
        "trials": trials,
        "feature_names": feature_names,
        "cal10_used_for_model_selection": False,
        "val25_read": False,
        "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(path),
    }
    (MODELS / f"b11_H{horizon}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return oof, cal_score, manifest


def group_records(data: dict[str, Any], scores: np.ndarray) -> list[dict[str, Any]]:
    rows = data["rows"]
    records = []
    for key, all_indices in data["groups"].items():
        indices = [index for index in all_indices if np.isfinite(scores[index])]
        first = rows[all_indices[0]]
        absent = not any(bool(rows[index]["positive"]) for index in all_indices)
        if not indices:
            records.append(
                {
                    "key": key,
                    "sequence": str(first["sequence"]),
                    "valid": False,
                    "absent": absent,
                    "selected_correct": False,
                    "top": np.nan,
                    "margin": np.nan,
                    "zmargin": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                    "count": 0,
                    "candidate_count": len(all_indices),
                    "selected_rank": None,
                }
            )
            continue
        ordered = sorted(indices, key=lambda index: (-float(scores[index]), int(rows[index]["candidate_rank"])))
        top_index = ordered[0]
        values = scores[indices]
        second = scores[ordered[1]] if len(ordered) > 1 else scores[top_index]
        std = float(np.std(values))
        records.append(
            {
                "key": key,
                "sequence": str(first["sequence"]),
                "valid": True,
                "absent": absent,
                "selected_correct": bool(rows[top_index]["positive"]),
                "top": float(scores[top_index]),
                "margin": float(scores[top_index] - second),
                "zmargin": float((scores[top_index] - np.mean(values)) / (std + 1e-6)),
                "mean": float(np.mean(values)),
                "std": std,
                "count": len(indices),
                "candidate_count": len(all_indices),
                "selected_rank": int(rows[top_index]["candidate_rank"]),
            }
        )
    return records


def commit_features(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = [record for record in records if record["valid"]]
    x = np.asarray(
        [
            [
                record["top"], record["margin"], record["zmargin"], record["mean"], record["std"],
                record["count"], record["candidate_count"], record["selected_rank"] / max(1, record["candidate_count"]),
            ]
            for record in valid
        ],
        dtype=np.float64,
    )
    y = np.asarray([record["selected_correct"] for record in valid], dtype=int)
    sequence = np.asarray([record["sequence"] for record in valid])
    return x, y, sequence


def best_threshold(values: np.ndarray, labels: np.ndarray) -> dict[str, Any] | None:
    best = None
    for threshold in np.unique(values)[::-1]:
        accepted = values >= threshold
        count = int(accepted.sum())
        if count == 0:
            continue
        precision = float(labels[accepted].mean())
        if precision >= MIN_PRECISION and (best is None or count > best["accepted"]):
            best = {"threshold": float(threshold), "accepted": count, "precision": precision}
    return best


def best_conjunction(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [record for record in records if record["valid"]]
    top = np.asarray([record["top"] for record in valid])
    margin = np.asarray([record["margin"] for record in valid])
    labels = np.asarray([record["selected_correct"] for record in valid])
    top_thresholds = np.unique(np.quantile(top, np.linspace(0, 1, 41)))
    margin_thresholds = np.unique(np.quantile(margin, np.linspace(0, 1, 41)))
    best = None
    for top_threshold in top_thresholds:
        for margin_threshold in margin_thresholds:
            accepted = (top >= top_threshold) & (margin >= margin_threshold)
            count = int(accepted.sum())
            if count == 0:
                continue
            precision = float(labels[accepted].mean())
            if precision >= MIN_PRECISION and (best is None or count > best["accepted"]):
                best = {
                    "top_threshold": float(top_threshold),
                    "margin_threshold": float(margin_threshold),
                    "accepted": count,
                    "precision": precision,
                }
    return best


def apply_rule(record: dict[str, Any], rule: dict[str, Any], probability: float | None) -> bool:
    if not record["valid"]:
        return False
    if rule["kind"] == "top":
        return record["top"] >= rule["threshold"]
    if rule["kind"] == "margin":
        return record["margin"] >= rule["threshold"]
    if rule["kind"] == "conjunction":
        return record["top"] >= rule["top_threshold"] and record["margin"] >= rule["margin_threshold"]
    if rule["kind"] == "calibrated_probability":
        return probability is not None and probability >= rule["threshold"]
    raise ValueError(rule["kind"])


def ece_brier(probability: np.ndarray, labels: np.ndarray, bins: int = 10) -> tuple[float, float]:
    ece = 0.0
    for lower in np.linspace(0, 1, bins + 1)[:-1]:
        upper = lower + 1.0 / bins
        members = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if members.any():
            ece += members.mean() * abs(float(probability[members].mean()) - float(labels[members].mean()))
    return float(ece), float(np.mean((probability - labels) ** 2))


def operating_metrics(records: list[dict[str, Any]], accepted: np.ndarray) -> dict[str, Any]:
    selected_correct = np.asarray([record["selected_correct"] for record in records], dtype=bool)
    absent = np.asarray([record["absent"] for record in records], dtype=bool)
    correct_accepts = int((accepted & selected_correct).sum())
    accepted_count = int(accepted.sum())
    correct_rejects = int((~accepted & absent).sum())
    return {
        "commit_accepts": accepted_count,
        "commit_precision": correct_accepts / max(1, accepted_count),
        "commit_coverage": accepted_count / max(1, len(records)),
        "candidate_or_none_exact_accuracy": (correct_accepts + correct_rejects) / max(1, len(records)),
        "candidate_set_positive_absent_groups": int(absent.sum()),
        "target_absent_false_acceptance": int((accepted & absent).sum()) / max(1, int(absent.sum())),
        "true_scene_none_status": "NOT_IDENTIFIABLE_IN_INHERITED_SHADOW_STREAM",
    }


def fit_commit_policy(
    train_records: list[dict[str, Any]],
    cal_records: list[dict[str, Any]],
    method: str,
    horizon: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    train_x, train_y, train_sequence = commit_features(train_records)
    cal_x, cal_y, _ = commit_features(cal_records)
    valid_train_indices = [index for index, record in enumerate(train_records) if record["valid"]]
    valid_cal_indices = [index for index, record in enumerate(cal_records) if record["valid"]]
    splitter = GroupKFold(n_splits=min(5, len(set(train_sequence))))
    oof = np.full(len(train_y), np.nan, dtype=np.float64)
    for fit_index, held_index in splitter.split(train_x, train_y, groups=train_sequence):
        model = pipeline(0.01)
        model.fit(train_x[fit_index], train_y[fit_index])
        oof[held_index] = model.predict_proba(train_x[held_index])[:, 1]
    calibrator = pipeline(0.01)
    calibrator.fit(train_x, train_y)
    train_probability_valid = calibrator.predict_proba(train_x)[:, 1]
    cal_probability_valid = calibrator.predict_proba(cal_x)[:, 1]
    train_probability = np.zeros(len(train_records), dtype=np.float64)
    cal_probability = np.zeros(len(cal_records), dtype=np.float64)
    train_probability[valid_train_indices] = train_probability_valid
    cal_probability[valid_cal_indices] = cal_probability_valid

    candidates = []
    for kind, values in (("top", np.asarray([record["top"] for record in train_records if record["valid"]])), ("margin", np.asarray([record["margin"] for record in train_records if record["valid"]])), ("calibrated_probability", oof)):
        selected = best_threshold(values, train_y)
        if selected is not None:
            candidates.append({"kind": kind, **selected})
    conjunction = best_conjunction(train_records)
    if conjunction is not None:
        candidates.append({"kind": "conjunction", **conjunction})
    if candidates:
        preference = {"top": 3, "margin": 2, "conjunction": 1, "calibrated_probability": 0}
        rule = max(candidates, key=lambda item: (item["accepted"], preference[item["kind"]]))
        rule_status = "TRAIN30_90P_FEASIBLE"
    else:
        rule = {"kind": "top", "threshold": float("inf"), "accepted": 0, "precision": 1.0}
        rule_status = "TRAIN30_90P_NOT_FEASIBLE_REJECT_ALL"
    train_accepted = np.asarray(
        [apply_rule(record, rule, train_probability[index]) for index, record in enumerate(train_records)], dtype=bool
    )
    cal_accepted = np.asarray(
        [apply_rule(record, rule, cal_probability[index]) for index, record in enumerate(cal_records)], dtype=bool
    )
    train_ece, train_brier = ece_brier(oof, train_y)
    cal_ece, cal_brier = ece_brier(cal_probability_valid, cal_y)
    train_operation = operating_metrics(train_records, train_accepted)
    cal_operation = operating_metrics(cal_records, cal_accepted)
    train_operation.update({"ece": train_ece, "brier": train_brier})
    cal_operation.update({"ece": cal_ece, "brier": cal_brier})

    curve = []
    order = np.argsort(-cal_probability_valid)
    for coverage in np.linspace(0.01, 1.0, 100):
        count = max(1, int(math.ceil(coverage * len(cal_records))))
        selected_valid = order[: min(count, len(order))]
        accepted = np.zeros(len(cal_records), dtype=bool)
        accepted[np.asarray(valid_cal_indices)[selected_valid]] = True
        operation = operating_metrics(cal_records, accepted)
        curve.append(
            {
                "method": method,
                "history": horizon,
                "requested_coverage": float(coverage),
                "actual_coverage": operation["commit_coverage"],
                "precision": operation["commit_precision"],
                "risk": 1.0 - operation["commit_precision"],
                "target_absent_false_acceptance": operation["target_absent_false_acceptance"],
            }
        )
    policy = {
        "status": rule_status,
        "selection": "maximum train30 accepted groups at >=90% precision among top, margin, conjunction, and sequence-OOF calibrated rules",
        "rule": rule,
        "candidate_rules": candidates,
        "calibrator": "regularized logistic C=0.01; five-fold sequence-OOF train threshold; full-train fit for frozen cal probability",
        "cal10_used_for_rule_or_threshold": False,
    }
    evaluations = {
        "train30": {record["key"]: {"accepted": bool(train_accepted[index]), "confidence": float(train_probability[index])} for index, record in enumerate(train_records)},
        "cal10": {record["key"]: {"accepted": bool(cal_accepted[index]), "confidence": float(cal_probability[index])} for index, record in enumerate(cal_records)},
    }
    return {"policy": policy, "train30": train_operation, "cal10": cal_operation}, curve, evaluations


def feature_coverage(data: dict[str, Any], scores: np.ndarray) -> dict[str, Any]:
    labels = np.asarray([bool(row["positive"]) for row in data["rows"]])
    valid = np.isfinite(scores)
    positive = float(valid[labels].mean()) if labels.any() else None
    negative = float(valid[~labels].mean()) if (~labels).any() else None
    return {
        "candidate_row_score_coverage": float(valid.mean()),
        "positive_candidate_row_score_coverage": positive,
        "negative_candidate_row_score_coverage": negative,
        "positive_negative_missingness_gap_pp": None if positive is None or negative is None else 100.0 * abs(positive - negative),
    }


def sequence_bootstrap_gain(
    data: dict[str, Any],
    method_scores: np.ndarray,
    baseline_scores: np.ndarray,
    n_boot: int = 2000,
) -> dict[str, Any]:
    sequences = sorted({str(row["sequence"]) for row in data["rows"]})
    keys_by_sequence = {
        sequence: [key for key, indices in data["groups"].items() if str(data["rows"][indices[0]]["sequence"]) == sequence]
        for sequence in sequences
    }
    per_sequence = {}
    for sequence in sequences:
        method = group_rank_summary(data, method_scores, keys_by_sequence[sequence])
        baseline = group_rank_summary(data, baseline_scores, keys_by_sequence[sequence])
        per_sequence[sequence] = None if method["top1"] is None or baseline["top1"] is None else method["top1"] - baseline["top1"]
    rng = np.random.default_rng(SEED)
    gains = []
    for _ in range(n_boot):
        sampled = rng.choice(sequences, size=len(sequences), replace=True)
        values = [per_sequence[sequence] for sequence in sampled if per_sequence[sequence] is not None]
        if values:
            gains.append(float(np.mean(values)))
    leave_one_out = {}
    for held in sequences:
        values = [value for sequence, value in per_sequence.items() if sequence != held and value is not None]
        leave_one_out[held] = float(np.mean(values)) if values else None
    positive = sum(value is not None and value > 0 for value in per_sequence.values())
    nonnegative = sum(value is not None and value >= 0 for value in per_sequence.values())
    return {
        "resampling_unit": "sequence",
        "n_boot": len(gains),
        "gain_mean": float(np.mean(gains)),
        "gain_ci95_low": float(np.quantile(gains, 0.025)),
        "gain_ci95_high": float(np.quantile(gains, 0.975)),
        "probability_gain_positive": float(np.mean(np.asarray(gains) > 0)),
        "per_sequence_top1_gain": per_sequence,
        "positive_sequences": positive,
        "nonnegative_sequences": nonnegative,
        "sequence_count": len(sequences),
        "leave_one_sequence_out_mean_gain": leave_one_out,
        "all_leave_one_out_gains_positive": all(value is not None and value > 0 for value in leave_one_out.values()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    train = load_split("train30")
    cal = load_split("cal10")
    all_scores: dict[str, dict[int, dict[str, np.ndarray]]] = {"train30": {}, "cal10": {}}
    visual_by_split: dict[str, dict[int, dict[str, Any]]] = {"train30": {}, "cal10": {}}
    b10_diagnostics = {"selection_policy": "train30 only", "histories": {}}
    b11_manifests = {}

    for horizon in HORIZONS:
        train_scores, train_visual = build_base_scores(train, horizon)
        cal_scores, cal_visual = build_base_scores(cal, horizon)
        lambda_trials = []
        best_lambda = None
        best_objective = None
        best_train_b10 = best_train_aux = best_train_diag = None
        for penalty_lambda in B10_LAMBDAS:
            candidate_scores, candidate_aux, candidate_diag = run_b10(train, train_visual["clip"], penalty_lambda)
            rank = group_rank_summary(train, candidate_scores)
            objective = (
                -1.0 if rank["top1"] is None else rank["top1"],
                -1.0 if rank["pair_auc"] is None else rank["pair_auc"],
                -1e9 if rank["hardest_negative_margin"] is None else rank["hardest_negative_margin"],
                -penalty_lambda,
            )
            lambda_trials.append({"lambda": penalty_lambda, **rank})
            if best_objective is None or objective > best_objective:
                best_objective = objective
                best_lambda = penalty_lambda
                best_train_b10, best_train_aux, best_train_diag = candidate_scores, candidate_aux, candidate_diag
        if best_lambda is None or best_train_b10 is None or best_train_aux is None:
            raise RuntimeError("B10 selection failed")
        cal_b10, cal_b10_aux, cal_b10_diag = run_b10(cal, cal_visual["clip"], best_lambda)
        train_scores["B10_EXPLICIT_NEGATIVE"] = best_train_b10
        cal_scores["B10_EXPLICIT_NEGATIVE"] = cal_b10
        b10_diagnostics["histories"][f"H{horizon}"] = {
            "selected_lambda": best_lambda,
            "selection_objective": "train30 top1, then pair AUC, then margin, then smaller lambda",
            "lambda_trials": lambda_trials,
            "train30": best_train_diag,
            "cal10": cal_b10_diag,
        }

        train_x, feature_names = b11_features(train, train_scores, train_visual, best_train_aux, horizon)
        cal_x, cal_feature_names = b11_features(cal, cal_scores, cal_visual, cal_b10_aux, horizon)
        if feature_names != cal_feature_names:
            raise RuntimeError("B11 feature mismatch")
        train_b11, cal_b11, b11_manifest = fit_b11(train, cal, train_x, cal_x, feature_names, horizon)
        train_scores["B11_ALL_LEGAL"] = train_b11
        cal_scores["B11_ALL_LEGAL"] = cal_b11
        b11_manifests[f"H{horizon}"] = b11_manifest
        all_scores["train30"][horizon] = train_scores
        all_scores["cal10"][horizon] = cal_scores
        visual_by_split["train30"][horizon] = train_visual
        visual_by_split["cal10"][horizon] = cal_visual

    information_rows = []
    per_sequence_rows = []
    calibration_rows = []
    curve_rows = []
    operation_evaluations: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for horizon in HORIZONS:
        for method in METHODS:
            train_score = all_scores["train30"][horizon][method]
            cal_score = all_scores["cal10"][horizon][method]
            train_rank = group_rank_summary(train, train_score)
            cal_rank = group_rank_summary(cal, cal_score)
            commit, curve, evaluations = fit_commit_policy(
                group_records(train, train_score), group_records(cal, cal_score), method, horizon
            )
            operation_evaluations[(method, horizon)] = evaluations
            curve_rows.extend(curve)
            for split_name, data, rank, operation, score in (
                ("train30", train, train_rank, commit["train30"], train_score),
                ("cal10", cal, cal_rank, commit["cal10"], cal_score),
            ):
                row = {
                    "status": "COMPUTED",
                    "split": split_name,
                    "history": horizon,
                    "method": method,
                    "groups": len(data["groups"]),
                    **rank,
                    **feature_coverage(data, score),
                    **operation,
                    "commit_policy_status": commit["policy"]["status"],
                    "commit_rule": json.dumps(commit["policy"]["rule"], sort_keys=True),
                    "threshold_selected_on": "train30_sequence_OOF_or_direct_train_scores",
                    "cal10_used_for_threshold": False,
                    "val25_read": False,
                }
                information_rows.append(row)
                calibration_rows.append(
                    {
                        "split": split_name,
                        "history": horizon,
                        "method": method,
                        "ece": operation["ece"],
                        "brier": operation["brier"],
                        "commit_precision": operation["commit_precision"],
                        "commit_coverage": operation["commit_coverage"],
                        "target_absent_false_acceptance": operation["target_absent_false_acceptance"],
                        "policy": json.dumps(commit["policy"], sort_keys=True),
                    }
                )
            for split_name, data, score in (("train30", train, train_score), ("cal10", cal, cal_score)):
                evaluations_for_split = evaluations[split_name]
                sequences = sorted({str(row["sequence"]) for row in data["rows"]})
                for sequence in sequences:
                    keys = [key for key, indices in data["groups"].items() if str(data["rows"][indices[0]]["sequence"]) == sequence]
                    rank = group_rank_summary(data, score, keys)
                    records = [record for record in group_records(data, score) if record["sequence"] == sequence]
                    accepted = np.asarray([evaluations_for_split[record["key"]]["accepted"] for record in records], dtype=bool)
                    operation = operating_metrics(records, accepted)
                    per_sequence_rows.append(
                        {
                            "split": split_name,
                            "sequence": sequence,
                            "history": horizon,
                            "method": method,
                            **rank,
                            **operation,
                        }
                    )
        for method, reason in NOT_COMPUTABLE.items():
            for split_name, data in (("train30", train), ("cal10", cal)):
                information_rows.append(
                    {
                        "status": "NOT_COMPUTABLE",
                        "split": split_name,
                        "history": horizon,
                        "method": method,
                        "groups": len(data["groups"]),
                        "reason": reason,
                        "val25_read": False,
                    }
                )

    lookup = {(row["split"], int(row["history"]), row["method"]): row for row in information_rows if row["status"] == "COMPUTED"}
    bootstrap = {}
    method_gate = {}
    baseline = lookup[("cal10", PRIMARY_HORIZON, "B2_GFN_R0")]
    baseline_train = lookup[("train30", PRIMARY_HORIZON, "B2_GFN_R0")]
    for method in ("B5A_SAM3_BOX_ROI", "B10_EXPLICIT_NEGATIVE", "B11_ALL_LEGAL"):
        current = lookup[("cal10", PRIMARY_HORIZON, method)]
        current_train = lookup[("train30", PRIMARY_HORIZON, method)]
        boot = sequence_bootstrap_gain(
            cal,
            all_scores["cal10"][PRIMARY_HORIZON][method],
            all_scores["cal10"][PRIMARY_HORIZON]["B2_GFN_R0"],
        )
        bootstrap[method] = boot
        top1_gain = current["top1"] - baseline["top1"]
        train_cal_gap_pp = 100.0 * abs(current_train["top1"] - current["top1"])
        criteria = {
            "cal_top1_gain_at_least_5pp": top1_gain >= 0.05,
            "hardest_negative_margin_improved": current["hardest_negative_margin"] > baseline["hardest_negative_margin"],
            "pair_auc_improved": current["pair_auc"] > baseline["pair_auc"],
            "target_absent_false_acceptance_not_worse": current["target_absent_false_acceptance"] <= baseline["target_absent_false_acceptance"] + 1e-12,
            "frozen_commit_precision_90_at_coverage_5": current["commit_precision"] >= MIN_PRECISION and current["commit_coverage"] >= MIN_COVERAGE,
            "majority_sequences_nonnegative": boot["nonnegative_sequences"] >= math.ceil(boot["sequence_count"] / 2),
            "sequence_bootstrap_not_single_sequence": boot["gain_ci95_low"] > 0 and boot["all_leave_one_out_gains_positive"],
            "feature_missingness_not_exploited": current["positive_negative_missingness_gap_pp"] <= 5.0 and current["candidate_row_score_coverage"] >= 0.90,
            "train_cal_gap_smaller_than_old_B2_collapse": train_cal_gap_pp < OLD_B2_GAP_PP,
            "no_val25_model_or_threshold_selection": True,
        }
        method_gate[method] = {
            "primary_history": PRIMARY_HORIZON,
            "cal_top1_gain_pp": 100.0 * top1_gain,
            "train_cal_top1_gap_pp": train_cal_gap_pp,
            "baseline_train_cal_gap_pp": 100.0 * abs(baseline_train["top1"] - baseline["top1"]),
            "criteria": criteria,
            "pass": all(criteria.values()),
        }
    passed_methods = [method for method, result in method_gate.items() if result["pass"]]
    rank_signal_methods = [
        method
        for method, result in method_gate.items()
        if result["criteria"]["cal_top1_gain_at_least_5pp"]
        and result["criteria"]["hardest_negative_margin_improved"]
        and result["criteria"]["pair_auc_improved"]
    ]
    if passed_methods:
        status = "PASS_N25R_INFORMATION_GATE"
    elif rank_signal_methods:
        status = "PARTIAL_N25R_FEATURE_SIGNAL"
    else:
        status = "FAIL_N25R_CANDIDATE_IDENTITY_SIGNAL"

    # Compact H5 stratification using thresholds frozen from train30.
    train_group_meta = []
    for key, indices in train["groups"].items():
        gap = float(np.mean([int(train["rows"][i]["decision_frame"]) - int(train["rows"][i]["correction_frame"]) for i in indices]))
        crowd = float(np.nanmean(train["features"]["neighbor"][indices, :PRIMARY_HORIZON, 3]))
        crossing = float(np.nanmean(train["features"]["neighbor"][indices, :PRIMARY_HORIZON, 2]))
        valid_ratio = float(train["features"]["valid"][indices, :PRIMARY_HORIZON].mean())
        train_group_meta.append((gap, crowd, crossing, valid_ratio))
    meta = np.asarray(train_group_meta)
    strata_thresholds = {
        "gap_q33": float(np.quantile(meta[:, 0], 1 / 3)),
        "gap_q67": float(np.quantile(meta[:, 0], 2 / 3)),
        "crowd_median": float(np.median(meta[:, 1])),
        "crossing_median": float(np.median(meta[:, 2])),
        "shadow_valid_median": float(np.median(meta[:, 3])),
    }
    stratified_rows = []
    for split_name, data in (("train30", train), ("cal10", cal)):
        categories: dict[str, list[str]] = defaultdict(list)
        for key, indices in data["groups"].items():
            gap = float(np.mean([int(data["rows"][i]["decision_frame"]) - int(data["rows"][i]["correction_frame"]) for i in indices]))
            crowd = float(np.nanmean(data["features"]["neighbor"][indices, :PRIMARY_HORIZON, 3]))
            crossing = float(np.nanmean(data["features"]["neighbor"][indices, :PRIMARY_HORIZON, 2]))
            valid_ratio = float(data["features"]["valid"][indices, :PRIMARY_HORIZON].mean())
            gap_name = "gap_short" if gap <= strata_thresholds["gap_q33"] else "gap_medium" if gap <= strata_thresholds["gap_q67"] else "gap_long"
            categories[gap_name].append(key)
            categories["crowd_high" if crowd >= strata_thresholds["crowd_median"] else "crowd_low"].append(key)
            categories["crossing_high" if crossing >= strata_thresholds["crossing_median"] else "crossing_low"].append(key)
            categories["shadow_loss_high" if valid_ratio < strata_thresholds["shadow_valid_median"] else "shadow_loss_low"].append(key)
        for method in ("B2_GFN_R0", "B4_DEEP_CLIPREID", "B5A_SAM3_BOX_ROI", "B10_EXPLICIT_NEGATIVE", "B11_ALL_LEGAL"):
            score = all_scores[split_name][PRIMARY_HORIZON][method]
            for category, keys in sorted(categories.items()):
                stratified_rows.append({"split": split_name, "history": PRIMARY_HORIZON, "method": method, "stratum": category, **group_rank_summary(data, score, keys)})

    write_csv(OUT / "information_gate.csv", information_rows)
    write_csv(OUT / "per_sequence.csv", per_sequence_rows)
    write_csv(OUT / "calibration.csv", calibration_rows)
    write_csv(OUT / "precision_risk_coverage.csv", curve_rows)
    write_csv(OUT / "stratified_metrics.csv", stratified_rows)
    (OUT / "bootstrap_sequence.json").write_text(json.dumps(bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "b10_memory_audit.json").write_text(json.dumps(b10_diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    frozen_config = {
        "seed": SEED,
        "histories": HORIZONS,
        "primary_gate_history": PRIMARY_HORIZON,
        "minimum_commit_precision": MIN_PRECISION,
        "minimum_commit_coverage": MIN_COVERAGE,
        "B10": {
            "candidate_lambdas": B10_LAMBDAS,
            "delta": B10_DELTA,
            "positive_memory_cap": MEMORY_POS_CAP,
            "negative_memory_cap": MEMORY_NEG_CAP,
            "selection_split": "train30",
        },
        "B11": {"candidate_C": B11_CS, "selection": "five-fold sequence-disjoint train30 OOF"},
        "strata_thresholds_from_train30": strata_thresholds,
        "cal10_gradient_or_threshold_selection": False,
        "val25_read": False,
    }
    (OUT / "frozen_config.json").write_text(json.dumps(frozen_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "status": status,
        "passed_methods": passed_methods,
        "rank_signal_methods": rank_signal_methods,
        "gate": method_gate,
        "primary_history": PRIMARY_HORIZON,
        "baseline_B2": {"train30": baseline_train, "cal10": baseline},
        "b10_selection": b10_diagnostics["histories"][f"H{PRIMARY_HORIZON}"],
        "B11": b11_manifests,
        "downstream_authorization": "CCRIM_CANDIDATE_UNION_FULL_LOOP_AUTHORIZED" if passed_methods else "STOP_BEFORE_CCRIM_UNION_FULL_LOOP",
        "scene_none_caveat": "candidate-set positive absence cannot be separated into visible-missing versus true/uncertain scene NONE in inherited data",
        "val25_read": False,
        "artifacts": [
            "outputs/n25r/information_gate.csv",
            "outputs/n25r/per_sequence.csv",
            "outputs/n25r/calibration.csv",
            "outputs/n25r/precision_risk_coverage.csv",
            "outputs/n25r/stratified_metrics.csv",
            "outputs/n25r/bootstrap_sequence.json",
            "outputs/n25r/b10_memory_audit.json",
            "outputs/n25r/frozen_config.json",
        ],
    }
    (OUT / "r5_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "passed_methods": passed_methods, "rank_signal_methods": rank_signal_methods, "gate": method_gate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
