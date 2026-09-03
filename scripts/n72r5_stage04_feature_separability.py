#!/usr/bin/env python3
"""N72R5 Stage 04: posthoc feature-separability audit for TVC_V0.

The audit asks whether the frozen human-anchor appearance feature ranks a
target candidate above competing candidates at all.  It does not alter
runtime association, select events, tune a residual, or train a model.  GT
is opened only after the Stage 03 runtime artifact schema and causal flags
have been checked.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n72r5_stage01_decision_boundary import (  # noqa: E402
    atomic_json,
    atomic_jsonl,
    box_iou,
    load_gt,
    load_inputs,
    read_json,
    read_jsonl,
    sha256,
)
from scripts.n72r5_stage03_tvc_v0 import (  # noqa: E402
    ROUND_ROOT as DEFAULT_STAGE03_ROOT,
    TVC_NAME,
)


OUT = ROOT / "outputs" / "N72R5"
STAGE03_ROOT = Path(
    os.environ.get("N72R5_STAGE03_INPUT_ROOT", str(OUT / "mechanism_rounds" / "round_03_tvc_v0"))
)
ROUND_ROOT = Path(
    os.environ.get(
        "N72R5_STAGE04_ROOT",
        str(OUT / "mechanism_rounds" / "round_04_feature_separability"),
    )
)
ARTIFACT_ROOT = STAGE03_ROOT / "artifacts"
RUNTIME_VALIDATION = STAGE03_ROOT / "runtime_validation.json"
STAGE03_PROTOCOL = STAGE03_ROOT / "tvc_v0_protocol.json"
PAIR_TABLE = ROUND_ROOT / "feature_pair_table.jsonl"
SUMMARY_PATH = ROUND_ROOT / "feature_separability_summary.json"
GATE_PATH = ROUND_ROOT / "gate.json"
STAGE_STATUS = OUT / "stage_status" / "stage_04_status.json"

IOU_THRESHOLD = 0.5
MIN_PAIRWISE_ACCURACY = 0.65
MIN_AUC = 0.65
MIN_PAIRS = 20
HORIZON_BUCKETS = ((1, 20, "H20"), (21, 50, "H50_ONLY"), (51, 100, "H100_ONLY"))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_gt_with_visibility(sequence: str) -> dict[int, dict[int, dict[str, Any]]]:
    path = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack/train") / str(sequence) / "gt" / "gt.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            values = [item.strip() for item in line.split(",")]
            if len(values) < 6:
                raise ValueError(f"malformed GT row: {path}:{line_number}")
            frame = int(values[0]) - 1
            identity = int(values[1])
            x, y, width, height = [float(value) for value in values[2:6]]
            visibility = None if len(values) <= 8 else float(values[8])
            result[frame][identity] = {
                "box": [x, y, x + width, y + height],
                "visibility": visibility,
            }
    return result


def validate_stage03_inputs(events: list[dict[str, Any]]) -> dict[str, Any]:
    validation = read_json(RUNTIME_VALIDATION)
    if validation.get("status") != "PASS_STAGE03_RUNTIME_ARTIFACT_VALIDATION":
        raise RuntimeError(f"Stage03 runtime validation is not PASS: {validation.get('status')}")
    if validation.get("runtime_future_gt_used") is not False or validation.get("posthoc_gt_used") is not False:
        raise RuntimeError("Stage03 runtime validation has invalid GT boundary")
    protocol = read_json(STAGE03_PROTOCOL)
    if protocol.get("name") != TVC_NAME or protocol.get("runtime_future_gt_used") is not False:
        raise RuntimeError("Stage03 protocol identity/boundary mismatch")
    checked = 0
    forbidden = {"dataset_gt_id", "gt_box", "future_gt", "future_identity_error", "reward"}
    for event in events:
        path = ARTIFACT_ROOT / f"{event['event_id']}.jsonl"
        rows = read_jsonl(path)
        expected = {(variant, frame) for variant in ("M0_CURRENT_FRAME_CORRECTION_ONLY", TVC_NAME) for frame in range(int(event["event_frame"]), int(event["event_frame"]) + 101)}
        actual = {(str(row.get("variant")), int(row.get("frame", -1))) for row in rows}
        if actual != expected:
            raise RuntimeError(f"Stage03 artifact key set invalid: {event['event_id']}")
        for row in rows:
            checked += 1
            if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False or row.get("posthoc_gt_used") is not False:
                raise RuntimeError(f"Stage03 runtime GT boundary invalid: {event['event_id']}/{row.get('frame')}")
            if forbidden.intersection(row):
                raise RuntimeError(f"posthoc field leaked into Stage03 runtime row: {event['event_id']}/{row.get('frame')}")
            candidates = row.get("candidate_rows")
            if not isinstance(candidates, list) or len({str(item.get("candidate_uid")) for item in candidates}) != len(candidates):
                raise RuntimeError(f"Stage03 candidate axis invalid: {event['event_id']}/{row.get('frame')}")
    return {
        "stage03_runtime_validation_sha256": sha256(RUNTIME_VALIDATION),
        "stage03_protocol_sha256": sha256(STAGE03_PROTOCOL),
        "artifact_count": len(events),
        "runtime_row_count": checked,
        "runtime_future_gt_used": False,
    }


def _auc_from_scores(labels: list[int], scores: list[float]) -> float | None:
    if not labels or len(labels) != len(scores):
        return None
    positives = [float(score) for label, score in zip(labels, scores) if label == 1]
    negatives = [float(score) for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif math.isclose(positive, negative, rel_tol=0.0, abs_tol=1.0e-12):
                wins += 0.5
    return float(wins / (len(positives) * len(negatives)))


def _pairwise_accuracy(pairs: list[dict[str, Any]]) -> float | None:
    if not pairs:
        return None
    return float(sum(int(float(item["correct_score"]) > float(item["competitor_score"])) for item in pairs) / len(pairs))


def _bucket(frame_horizon: int) -> str:
    for low, high, name in HORIZON_BUCKETS:
        if low <= frame_horizon <= high:
            return name
    return "OUT_OF_RANGE"


def _summary_for_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        return {
            "pair_count": 0,
            "pairwise_ranking_accuracy": None,
            "auc_human_similarity": None,
            "auc_relative_margin": None,
            "mean_human_margin": None,
            "median_human_margin": None,
            "p10_human_margin": None,
            "p90_human_margin": None,
            "correct_human_similarity_mean": None,
            "competitor_human_similarity_mean": None,
        }
    correct = [float(item["correct_score"]) for item in pairs]
    competitor = [float(item["competitor_score"]) for item in pairs]
    margins = [a - b for a, b in zip(correct, competitor)]
    labels = [1] * len(correct) + [0] * len(competitor)
    scores = correct + competitor
    rel_correct = [float(item["correct_relative_margin"]) for item in pairs]
    rel_comp = [float(item["competitor_relative_margin"]) for item in pairs]
    return {
        "pair_count": len(pairs),
        "pairwise_ranking_accuracy": _pairwise_accuracy(pairs),
        "auc_human_similarity": _auc_from_scores(labels, scores),
        "auc_relative_margin": _auc_from_scores(
            [1] * len(rel_correct) + [0] * len(rel_comp),
            rel_correct + rel_comp,
        ),
        "mean_human_margin": float(np.mean(margins)),
        "median_human_margin": float(np.median(margins)),
        "p10_human_margin": float(np.quantile(margins, 0.10)),
        "p90_human_margin": float(np.quantile(margins, 0.90)),
        "correct_human_similarity_mean": float(np.mean(correct)),
        "competitor_human_similarity_mean": float(np.mean(competitor)),
    }


def run(args: argparse.Namespace) -> int:
    if ROUND_ROOT.exists() and any(ROUND_ROOT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty Stage04 root: {ROUND_ROOT}")
    ROUND_ROOT.mkdir(parents=True, exist_ok=True)
    events, _, _ = load_inputs()
    if args.event_limit:
        events = events[: int(args.event_limit)]
    input_audit = validate_stage03_inputs(events)
    protocol = {
        "schema_version": "N72R5_STAGE04_FEATURE_SEPARABILITY_PROTOCOL_V1",
        "status": "FROZEN_BEFORE_POSTHOC_SCORING",
        "stage": "04_FEATURE_SEPARABILITY_AUDIT",
        "source": "N72R5_STAGE03_TVC_V0_RUNTIME_ARTIFACTS",
        "correct_candidate_rule": "highest target-GT box IoU, retained only if IoU>=0.5",
        "competitor_rule": "every other candidate in the same runtime frame; no future outcome selection",
        "score_fields": ["human_target_similarity", "relative_margin"],
        "min_pairwise_accuracy": MIN_PAIRWISE_ACCURACY,
        "min_auc": MIN_AUC,
        "min_pairs": MIN_PAIRS,
        "occlusion_proxy": "DanceTrack GT visibility field: >=0.7 visible, >=0.3 partial, <0.3 occluded, missing unknown",
        "posthoc_only": True,
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "created_at_utc": now_utc(),
    }
    protocol_path = ROUND_ROOT / "feature_separability_protocol.json"
    atomic_json(protocol_path, protocol)
    pair_rows: list[dict[str, Any]] = []
    for event in events:
        gt = read_gt_with_visibility(str(event["sequence"]))
        event_id = str(event["event_id"])
        rows = read_jsonl(ARTIFACT_ROOT / f"{event_id}.jsonl")
        by_key = {(str(row["variant"]), int(row["frame"])): row for row in rows}
        for frame_horizon in range(1, 101):
            frame = int(event["event_frame"]) + frame_horizon
            target_gt = gt.get(frame, {}).get(int(event["dataset_gt_id"]))
            if target_gt is None:
                continue
            tvc = by_key[(TVC_NAME, frame)]
            candidates = tvc["candidate_rows"]
            details = {str(item["candidate_uid"]): item for item in tvc.get("tvc_components_by_candidate", [])}
            if len(details) != len(candidates):
                raise RuntimeError(f"TVC component/candidate mismatch: {event_id}/{frame}")
            ranked = sorted(
                ((box_iou(candidate["box_xyxy"], target_gt["box"]), str(candidate["candidate_uid"]), candidate) for candidate in candidates),
                key=lambda item: (-item[0], item[1]),
            )
            if not ranked or ranked[0][0] < IOU_THRESHOLD:
                continue
            correct_iou, correct_uid, correct_candidate = ranked[0]
            competitors = [item for item in ranked[1:] if str(item[1]) in details]
            if not competitors:
                continue
            baseline = by_key[("M0_CURRENT_FRAME_CORRECTION_ONLY", frame)]
            baseline_map = {
                str(candidate["candidate_uid"]): candidate.get("public_id")
                for candidate in baseline["candidate_rows"]
            }
            baseline_target_uid = next(
                (uid for uid, public in baseline_map.items() if public is not None and int(public) == int(event["target_public_id"])),
                None,
            )
            if baseline_target_uid is not None and baseline_target_uid != correct_uid:
                ordered_competitors = [item for item in competitors if str(item[1]) == baseline_target_uid] or competitors
            else:
                ordered_competitors = competitors
            competitor_iou, competitor_uid, competitor_candidate = max(
                ordered_competitors,
                key=lambda item: (float(details[str(item[1])]["human_target_similarity"]), -float(item[0]), str(item[1])),
            )
            correct_detail = details[correct_uid]
            competitor_detail = details[str(competitor_uid)]
            visibility = target_gt.get("visibility")
            if visibility is None:
                occlusion_bucket = "unknown"
            elif float(visibility) >= 0.7:
                occlusion_bucket = "visible"
            elif float(visibility) >= 0.3:
                occlusion_bucket = "partial"
            else:
                occlusion_bucket = "occluded"
            pair_rows.append(
                {
                    "schema_version": "N72R5_STAGE04_FEATURE_PAIR_V1",
                    "event_id": event_id,
                    "sequence": str(event["sequence"]),
                    "action_type": str(event["action_type"]),
                    "event_frame": int(event["event_frame"]),
                    "frame": frame,
                    "frame_horizon": frame_horizon,
                    "horizon_bucket": _bucket(frame_horizon),
                    "occlusion_bucket_posthoc": occlusion_bucket,
                    "target_public_id": int(event["target_public_id"]),
                    "correct_candidate_uid": correct_uid,
                    "correct_candidate_iou_posthoc": float(correct_iou),
                    "competitor_candidate_uid": str(competitor_uid),
                    "competitor_candidate_iou_posthoc": float(competitor_iou),
                    "baseline_target_candidate_uid": baseline_target_uid,
                    "correct_score": float(correct_detail["human_target_similarity"]),
                    "competitor_score": float(competitor_detail["human_target_similarity"]),
                    "correct_relative_margin": float(correct_detail["relative_margin"]),
                    "competitor_relative_margin": float(competitor_detail["relative_margin"]),
                    "human_margin_correct_minus_competitor": float(correct_detail["human_target_similarity"] - competitor_detail["human_target_similarity"]),
                    "relative_margin_correct_minus_competitor": float(correct_detail["relative_margin"] - competitor_detail["relative_margin"]),
                    "baseline_target_was_correct_candidate": bool(baseline_target_uid == correct_uid),
                    "runtime_future_gt_used": False,
                    "posthoc_gt_used": True,
                    "interaction_source": "simulated_from_gt",
                    "not_real_human_evidence": True,
                }
            )
    atomic_jsonl(PAIR_TABLE, pair_rows)
    groupings: dict[str, list[dict[str, Any]]] = {"all": pair_rows}
    for key_fn, prefix in (
        (lambda item: str(item["action_type"]), "action="),
        (lambda item: str(item["sequence"]), "sequence="),
        (lambda item: str(item["horizon_bucket"]), "horizon="),
        (lambda item: str(item["occlusion_bucket_posthoc"]), "occlusion="),
    ):
        for item in pair_rows:
            groupings.setdefault(prefix + key_fn(item), []).append(item)
    grouped_summary = {key: _summary_for_pairs(value) for key, value in sorted(groupings.items())}
    all_summary = grouped_summary["all"]
    informative = bool(
        all_summary["pair_count"] >= MIN_PAIRS
        and all_summary["pairwise_ranking_accuracy"] is not None
        and float(all_summary["pairwise_ranking_accuracy"]) >= MIN_PAIRWISE_ACCURACY
        and all_summary["auc_human_similarity"] is not None
        and float(all_summary["auc_human_similarity"]) >= MIN_AUC
    )
    summary = {
        "schema_version": "N72R5_STAGE04_FEATURE_SEPARABILITY_SUMMARY_V1",
        "status": "PASS_STAGE04_POSTHOC_FEATURE_AUDIT",
        "event_count": len(events),
        "independent_sequence_count": len({str(event["sequence"]) for event in events}),
        "pair_count": len(pair_rows),
        "grouped": grouped_summary,
        "feature_informative_under_preregistered_gate": informative,
        "thresholds": {
            "min_pairwise_accuracy": MIN_PAIRWISE_ACCURACY,
            "min_auc": MIN_AUC,
            "min_pairs": MIN_PAIRS,
        },
        "input_audit": input_audit,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }
    atomic_json(SUMMARY_PATH, summary)
    if informative:
        status = "PASS_STAGE04_FEATURE_INFORMATIVE_ROUTE_TO_LEARNED_TVC"
        route = "LEARNED_TVC_ALLOWED_WITH_SEQUENCE_SPLIT"
    else:
        status = "FAIL_STAGE04_FEATURE_NOT_INFORMATIVE_ROUTE_TO_TEMPORAL_CONTEXT_DIAGNOSIS"
        route = "TEMPORAL_CONTEXT_FEATURE_BRANCH"
    gate = {
        "schema_version": "N72R5_STAGE04_FEATURE_SEPARABILITY_GATE_V1",
        "status": status,
        "route": route,
        "feature_informative": informative,
        "pair_count": len(pair_rows),
        "pairwise_ranking_accuracy": all_summary["pairwise_ranking_accuracy"],
        "auc_human_similarity": all_summary["auc_human_similarity"],
        "auc_relative_margin": all_summary["auc_relative_margin"],
        "training_authorized": informative,
        "production_authorized": False,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "next_step": "pre-registered sequence-split TVC_V1 only if feature_informative; otherwise temporal/context representation diagnosis",
    }
    atomic_json(GATE_PATH, gate)
    atomic_json(
        STAGE_STATUS,
        {
            "schema_version": "N72R5_STAGE_STATUS_V1",
            "stage": "04_FEATURE_SEPARABILITY_AUDIT",
            "status": status,
            "protocol": str(protocol_path),
            "pair_table": str(PAIR_TABLE),
            "summary": str(SUMMARY_PATH),
            "gate": str(GATE_PATH),
            "event_count": len(events),
            "pair_count": len(pair_rows),
            "feature_informative": informative,
            "runtime_future_gt_used": False,
            "posthoc_gt_used": True,
            "training_authorized": informative,
            "production_authorized": False,
        },
    )
    print(json.dumps({"status": status, "pair_count": len(pair_rows), "summary": str(SUMMARY_PATH), "gate": str(GATE_PATH)}, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-limit", type=int, default=0)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
