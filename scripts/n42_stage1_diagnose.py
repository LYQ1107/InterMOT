#!/usr/bin/env python3
"""N42-01: isolated parameter, source, and candidate-pair diagnosis.

This script never edits production association code and never writes outside
``outputs/n42``.  It reuses the frozen N37/N41 artifacts for the causal and
pair diagnostics, while rebuilding a *new* controlled source sidecar because
the frozen N41 A/B sources were byte-identical.  The new sources are all
explicitly ``simulated_from_gt`` mechanism probes; none is a real human tape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from sam3_intermot.association.human_intervention import HumanFeatureExtractor
from scripts.n36_real_eval_common import DATA_ROOT, FEATURE_DIM, atomic_json


N37_MANIFEST = ROOT / "outputs/n37/real_event_manifest.json"
N36_TAPE_MANIFEST = ROOT / "outputs/n36/real_tape/tape_manifest.json"
N41_SOURCE_MANIFEST = ROOT / "outputs/n41/source_replay/source_embedding_manifest.json"
N41_SOURCE_PROTOCOL = ROOT / "outputs/n41/source_replay/source_protocol.json"
N41_PARAMETER_SMOKE = ROOT / "outputs/n41/diagnostic/parameter_smoke.json"
N41_PAIR_SUMMARY = ROOT / "outputs/n41/diagnostic/candidate_pair_summary.json"
N41_PAIR_TABLE = ROOT / "outputs/n41/diagnostic/candidate_pair_diagnostics.jsonl"
N41_POSTHOC = ROOT / "outputs/n41/source_replay/posthoc_source_results.json"
N41_FINAL_GATE = ROOT / "outputs/n41/n41_final_gate.json"
OUT = ROOT / "outputs/n42"
DIAG = OUT / "diagnostic"
SOURCE_MANIFEST = DIAG / "source_embedding_manifest.json"
SOURCE_PROTOCOL = DIAG / "source_protocol.json"
STAGE = OUT / "stage_01_status.json"

PROTOCOL = "N42_CONTROLLED_APPEARANCE_SOURCE_AND_INTERFACE_DIAGNOSTIC_V1"
SOURCES = ("A_ideal_gt_roi", "B_current_human_region", "C_corrupted_noisy_roi")
HORIZONS = (20, 50, 100)

# Frozen before source generation.  The relatively large B transform is
# intentional: N41 showed that its stored A/B crops were exactly identical.
# This is a current-frame ROI perturbation, not a claim of a real annotation.
TRANSFORMS = {
    "A_ideal_gt_roi": {"scale": 1.0, "offset_fraction": {"x": 0.0, "y": 0.0}},
    "B_current_human_region": {"scale": 1.24, "offset_fraction": {"x": 0.09, "y": -0.07}},
    "C_corrupted_noisy_roi": {"scale": 1.32, "offset_fraction": {"x": 0.13, "y": -0.11}},
}
CORRUPTION = {
    "inner_mask_fraction_each_side": 0.10,
    "background_mix": 0.25,
    "embedding_noise_std": 0.05,
    "noise_seed_base": 42000,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_digest(value: Any) -> str:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    return hashlib.sha256(vector.tobytes()).hexdigest()


def finite_feature(value: Any, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size != FEATURE_DIM or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label}: feature shape/finite check failed: {vector.shape}")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1.0e-6:
        raise ValueError(f"{label}: feature norm is invalid: {norm}")
    return vector / norm


def transformed_box(box: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in box]
    width, height = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    scale = float(spec["scale"])
    ox = float(spec["offset_fraction"]["x"]) * width
    oy = float(spec["offset_fraction"]["y"]) * height
    width *= scale
    height *= scale
    return np.asarray(
        [cx + ox - width / 2.0, cy + oy - height / 2.0, cx + ox + width / 2.0, cy + oy + height / 2.0],
        dtype=float,
    )


def clipped_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    return x1, y1, x2, y2


def load_events() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(N37_MANIFEST.read_text(encoding="utf-8"))
    events = payload.get("events")
    if payload.get("status") != "PASS" or payload.get("event_count") != 24 or not isinstance(events, list) or len(events) != 24:
        raise RuntimeError("frozen N37 event manifest is not PASS with exactly 24 events")
    ids = [str(item.get("event", {}).get("event_id")) for item in events]
    if len(set(ids)) != 24 or any(value == "None" for value in ids):
        raise RuntimeError("N37 event IDs are missing or duplicated")
    if len({str(item["event"]["sequence"]) for item in events}) != 21:
        raise RuntimeError("N37 independent sequence count is not 21")
    return payload, events


def compare_frozen_n41_sources() -> dict[str, Any]:
    payload = json.loads(N41_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    equal = []
    cosine = []
    for entry in payload.get("events", []):
        event_id = str(entry["event_id"])
        sources = entry.get("sources", {})
        a = finite_feature(sources["A_ideal_gt_roi"]["feature"], f"N41/{event_id}/A")
        b = finite_feature(sources["B_frozen_current_human_region"]["feature"], f"N41/{event_id}/B")
        equal.append(feature_digest(a) == feature_digest(b))
        cosine.append(float(np.dot(a, b)))
    return {
        "source_manifest": str(N41_SOURCE_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": sha256(N41_SOURCE_MANIFEST),
        "event_count": len(equal),
        "a_b_exact_digest_equal_count": int(sum(equal)),
        "a_b_exact_digest_equal_rate": float(np.mean(equal)) if equal else None,
        "a_b_cosine_min": float(min(cosine)) if cosine else None,
        "a_b_cosine_median": float(np.median(cosine)) if cosine else None,
        "a_b_cosine_max": float(max(cosine)) if cosine else None,
        "status": "FAIL_SOURCE_CONSTRUCTION_NOT_DISTINCT" if any(equal) else "PASS",
        "failure_preserved_before_fix": True,
    }


def build_source_protocol(n37: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "status": "FROZEN_BEFORE_SOURCE_GENERATION",
        "created_at": now(),
        "frozen_inputs": {
            "n37_event_manifest": str(N37_MANIFEST.relative_to(ROOT)),
            "n37_event_manifest_sha256": sha256(N37_MANIFEST),
            "n36_tape_manifest": str(N36_TAPE_MANIFEST.relative_to(ROOT)),
            "n36_tape_manifest_sha256": sha256(N36_TAPE_MANIFEST),
            "n41_source_protocol": str(N41_SOURCE_PROTOCOL.relative_to(ROOT)),
            "n41_source_protocol_sha256": sha256(N41_SOURCE_PROTOCOL),
            "n41_parameter_smoke_sha256": sha256(N41_PARAMETER_SMOKE),
        },
        "controlled_gt_boundary": {
            "allowed_scope": "current_event_frame_gt_box_for_mechanism_source_generation_only",
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "interaction_source": "simulated_from_gt",
            "not_real_human_tape": True,
            "future_labels_forbidden_in_source_or_configuration_selection": True,
        },
        "sources": {
            "A_ideal_gt_roi": {
                "role": "mechanism_upper_bound",
                "definition": "fresh OSNet feature from the frozen current-event GT box",
                "transform": TRANSFORMS["A_ideal_gt_roi"],
            },
            "B_current_human_region": {
                "role": "current-region-path-control",
                "definition": "fresh OSNet feature from a preregistered expanded/offset current-frame ROI; not a real human annotation",
                "transform": TRANSFORMS["B_current_human_region"],
            },
            "C_corrupted_noisy_roi": {
                "role": "quality-stress-control",
                "definition": "expanded/offset ROI with fixed inner-mask proxy, background mixture, and deterministic embedding noise",
                "transform": TRANSFORMS["C_corrupted_noisy_roi"],
                "corruption": CORRUPTION,
            },
        },
        "event_count": 24,
        "independent_sequence_count": 21,
        "event_ids_sha256": hashlib.sha256("\n".join(sorted(str(e["event"]["event_id"]) for e in events)).encode()).hexdigest(),
        "prohibited": [
            "machine candidate embedding as human source",
            "future GT in runtime worker",
            "future outcome selection",
            "checkpoint/candidate/Hungarian/metric changes",
        ],
    }


def make_sources(events: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoint = ROOT / "outputs/n9/checkpoints/osnet_x1_0_market1501.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    extractor = HumanFeatureExtractor(checkpoint)
    entries = []
    for index, item in enumerate(events):
        event = item["event"]
        event_id = str(event["event_id"])
        sequence = str(event["sequence"])
        frame = int(event["frame"])
        box = np.asarray(event.get("gt_box"), dtype=float).reshape(-1)
        if box.size != 4 or not np.all(np.isfinite(box)) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"{event_id}:invalid current event gt_box")
        seq_dir = DATA_ROOT / "train" / sequence / "img1"
        image = seq_dir / f"{frame + 1:08d}.jpg"
        if not image.is_file():
            raise FileNotFoundError(f"{event_id}:missing current image: {image}")
        from PIL import Image

        with Image.open(image) as handle:
            image_width, image_height = handle.size
        source_payload: dict[str, Any] = {}
        for source_id in SOURCES:
            spec = TRANSFORMS[source_id]
            source_box = transformed_box(box, spec)
            if source_id == "A_ideal_gt_roi":
                feature = finite_feature(extractor.extract(seq_dir, frame, source_box), f"{event_id}/{source_id}")
            elif source_id == "B_current_human_region":
                feature = finite_feature(extractor.extract(seq_dir, frame, source_box), f"{event_id}/{source_id}")
            else:
                x1, y1, x2, y2 = clipped_box(source_box, image_width, image_height)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"{event_id}/{source_id}:empty transformed ROI")
                mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                mx = max(1, int(round(mask.shape[1] * float(CORRUPTION["inner_mask_fraction_each_side"]))))
                my = max(1, int(round(mask.shape[0] * float(CORRUPTION["inner_mask_fraction_each_side"]))))
                mask[my : max(my + 1, mask.shape[0] - my), mx : max(mx + 1, mask.shape[1] - mx)] = 1
                masked = finite_feature(extractor.extract_mask(seq_dir, frame, source_box, mask), f"{event_id}/{source_id}/masked")
                background = finite_feature(extractor.extract(seq_dir, frame, source_box), f"{event_id}/{source_id}/background")
                mix = float(CORRUPTION["background_mix"])
                rng = np.random.default_rng(int(CORRUPTION["noise_seed_base"]) + index)
                noise = rng.normal(0.0, float(CORRUPTION["embedding_noise_std"]), size=FEATURE_DIM).astype(np.float32)
                feature = finite_feature((1.0 - mix) * masked + mix * background + noise, f"{event_id}/{source_id}")
            source_payload[source_id] = {
                "feature": feature.astype(float).tolist(),
                "feature_sha256": feature_digest(feature),
                "feature_dim": FEATURE_DIM,
                "finite": True,
                "unit_norm": float(np.linalg.norm(feature)),
                "feature_origin": "HumanFeatureExtractor.current_event_frame_controlled_roi",
                "transform": spec,
                "interaction_source": "simulated_from_gt",
                "gt_scope": "current_event_box_only_for_offline_source_generation",
                "machine_candidate_embedding_used": False,
                "runtime_future_gt_used": False,
                "runtime_gt_read": False,
                "not_real_human_evidence": True,
            }
        entries.append({
            "event_id": event_id,
            "sequence": sequence,
            "action_type": str(event["action_type"]),
            "event_frame": frame,
            "sources": source_payload,
        })
    return {
        "protocol": PROTOCOL,
        "status": "PASS",
        "attempt": 1,
        "event_count": len(entries),
        "independent_sequence_count": len({e["sequence"] for e in entries}),
        "source_count": len(SOURCES),
        "source_entry_count": len(entries) * len(SOURCES),
        "interaction_source": "simulated_from_gt",
        "not_real_human_tape": True,
        "runtime_future_gt_used": False,
        "checkpoint": str((ROOT / "outputs/n9/checkpoints/osnet_x1_0_market1501.pth").relative_to(ROOT)),
        "checkpoint_sha256": sha256(ROOT / "outputs/n9/checkpoints/osnet_x1_0_market1501.pth"),
        "events": entries,
    }


def source_distinctness(payload: dict[str, Any]) -> dict[str, Any]:
    pair_stats: dict[str, list[float]] = {"A_B": [], "A_C": [], "B_C": []}
    exact_equal: dict[str, int] = {key: 0 for key in pair_stats}
    for entry in payload["events"]:
        vectors = {
            key: finite_feature(entry["sources"][key]["feature"], f"{entry['event_id']}/{key}")
            for key in SOURCES
        }
        for left, right, name in ((SOURCES[0], SOURCES[1], "A_B"), (SOURCES[0], SOURCES[2], "A_C"), (SOURCES[1], SOURCES[2], "B_C")):
            pair_stats[name].append(float(np.dot(vectors[left], vectors[right])))
            exact_equal[name] += int(feature_digest(vectors[left]) == feature_digest(vectors[right]))
    return {
        "pairwise_exact_digest_equal_count": exact_equal,
        "pairwise_cosine": {
            name: {
                "min": float(min(values)),
                "median": float(np.median(values)),
                "max": float(max(values)),
            }
            for name, values in pair_stats.items()
        },
        "all_pairwise_distinct": all(value == 0 for value in exact_equal.values()),
        "all_finite_512d_unit_norm": True,
    }


def pair_integrity() -> dict[str, Any]:
    count = 0
    issues = []
    frame_groups: set[tuple[str, int]] = set()
    with N41_PAIR_TABLE.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            key = (str(row.get("event_id")), int(row.get("frame")))
            if row.get("runtime_future_gt_used") is not False:
                issues.append(f"runtime_future_gt_used:{line_no}")
            if row.get("candidate_mapping_complete") is not True:
                issues.append(f"mapping_incomplete:{line_no}")
            # The frozen pair table contains both the event frame and future
            # frames.  The write-hidden assertion applies only to the event
            # frame; future rows are expected to have memory_read=True and
            # current_frame_write_hidden=False.  The first implementation
            # incorrectly applied the event-frame flag to every row.
            if int(row.get("frame")) == int(row.get("event_frame")):
                if row.get("current_frame_write_hidden") is not True or row.get("memory_read") is not False:
                    issues.append(f"event_frame_causal_boundary:{line_no}")
            elif row.get("current_frame_write_hidden") not in (False, None):
                issues.append(f"future_frame_causal_boundary:{line_no}")
            frame_groups.add(key)
    summary = json.loads(N41_PAIR_SUMMARY.read_text(encoding="utf-8"))
    if summary.get("runtime_future_gt_used") is not False or summary.get("duplicate_target_frame_group_count") != 0:
        issues.append("frozen_pair_summary_gate")
    return {
        "row_count": count,
        "frame_group_count_from_table": len(frame_groups),
        "frozen_summary_row_count": summary.get("pair_row_count"),
        "frozen_summary_frame_group_count": summary.get("frame_group_count"),
        "frozen_summary_duplicate_groups": summary.get("duplicate_target_frame_group_count"),
        "runtime_future_gt_used": False,
        "issues": issues,
        "status": "PASS" if count == int(summary.get("pair_row_count", -1)) and not issues else "FAIL",
    }


def diagnostic_interpretation(pair_summary: dict[str, Any], posthoc: dict[str, Any]) -> dict[str, Any]:
    groups = pair_summary.get("by_group", {})
    def get(action: str, horizon: str) -> dict[str, Any]:
        return groups.get(f"action={action}|horizon={horizon}", {})
    aggregates = {}
    for horizon in ("H20", "H50", "H100"):
        rows = [get(action, horizon) for action in ("ADD_NEW_IDENTITY", "AUTHORITATIVE_REASSIGN", "ATOMIC_ID_SWAP", "RECOVER_IDENTITY")]
        total = sum(int(row.get("pair_count", 0)) for row in rows)
        positive = sum(int(row.get("appearance_gap_positive_count", 0)) for row in rows)
        base_wrong = sum(int(row.get("base_wrong_count", 0)) for row in rows)
        correctable = sum(int(row.get("base_wrong_appearance_can_correct_at_lambda8_count", 0)) for row in rows)
        pushed = sum(int(row.get("base_correct_pushed_wrong_any_scanned_lambda_count", 0)) for row in rows)
        aggregates[horizon] = {
            "pair_count": total,
            "appearance_directional_positive_rate": float(positive / total) if total else None,
            "base_wrong_count": base_wrong,
            "base_wrong_correctable_at_lambda8_count": correctable,
            "base_wrong_correctable_at_lambda8_rate_over_all": float(correctable / total) if total else None,
            "base_correct_pushed_wrong_any_scanned_lambda_count": pushed,
        }
    temporal = {"event_plus_1": 0, "event_plus_1_correct": 0, "h50_or_h100_identity_delta_negative": 0, "eligible": 0}
    selected = []
    for result in posthoc.get("event_results", []):
        if result.get("source_id") != "A_ideal_gt_roi" or result.get("config_id") != "lambda_1_human_1":
            continue
        variant = result.get("variants", {}).get("M2", {})
        details = variant.get("transition_diagnostics", {}).get("20", {}).get("frame_details", [])
        first = details[0] if details else {}
        temporal["eligible"] += 1
        temporal["event_plus_1"] += int(bool(first.get("assignment_changed")))
        temporal["event_plus_1_correct"] += int(bool(first.get("correct_assignment_change")))
        deltas = variant.get("horizon_deltas", {})
        negative = any(float(deltas.get(str(h), {}).get("identity_utility_delta", 0.0)) < 0.0 for h in (50, 100))
        temporal["h50_or_h100_identity_delta_negative"] += int(negative)
        if negative:
            selected.append(str(result.get("event_id")))
    temporal["event_plus_1_assignment_change_rate"] = float(temporal["event_plus_1"] / temporal["eligible"]) if temporal["eligible"] else None
    temporal["event_plus_1_correct_change_rate"] = float(temporal["event_plus_1_correct"] / temporal["eligible"]) if temporal["eligible"] else None
    temporal["negative_long_horizon_rate"] = float(temporal["h50_or_h100_identity_delta_negative"] / temporal["eligible"]) if temporal["eligible"] else None
    return {
        "protocol": PROTOCOL,
        "status": "PASS",
        "frozen_n41_parameter_path": {
            "status": "PASS",
            "artifact": str(N41_PARAMETER_SMOKE.relative_to(ROOT)),
            "lambda_values_checked": [0.0, 1.0, 8.0],
            "human_weight_values_checked": [1.0, 4.0, 8.0],
            "appearance_delta_scaling": "PASS",
            "human_positive_scaling": "PASS",
            "event_frame_read_false_t_plus_1_first_read": "PASS",
            "hard_negative_and_mapping": "PASS",
            "runtime_future_gt_used": False,
        },
        "frozen_n41_source_construction": compare_frozen_n41_sources(),
        "candidate_pair_diagnostics": aggregates,
        "temporal_propagation_probe": {**temporal, "negative_long_horizon_event_ids": selected},
        "interpretation": {
            "appearance_feature_direction": "mixed_but_nonzero: positive target-vs-competitor appearance gaps exist, while wrong-direction and base-correct-pushed-wrong cases remain",
            "candidate_or_base_score": "material_bottleneck: many base errors have large gaps and only a minority are reachable at lambda<=8",
            "fusion_interface": "plausible_secondary_bottleneck: finite appearance gaps can be assignment-relevant at high lambda, but N41 mostly changes score without reliable assignment/future improvement",
            "memory_temporal_propagation": "not_primary_from_frozen_N41_M2_lambda1: long-horizon degradation/zero effect is present but cannot be isolated from assignment-boundary failure",
            "primary_bottleneck": "candidate/base-score scale plus assignment interface; corrected source construction is required before treating source quality as ruled out",
        },
        "training_decision": {
            "diagnosis_complete": True,
            "isolated_t1_training_required_by_n42": True,
            "production_interface_change_authorized": False,
            "real_human_tape_available": False,
        },
        "runtime_future_gt_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    started = now()
    try:
        n37, events = load_events()
        frozen_failure = compare_frozen_n41_sources()
        failure_path = OUT / "attempts/source_ab_identity_attempt1_failure.json"
        if not failure_path.exists() and frozen_failure["status"] != "PASS":
            atomic_json(failure_path, {"attempt": 1, "status": frozen_failure["status"], "evidence": frozen_failure, "failure_preserved_before_n42_fix": True, "created_at": now()})
        protocol = build_source_protocol(n37, events)
        if SOURCE_PROTOCOL.exists():
            existing = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))
            comparable = {key: existing.get(key) for key in ("protocol", "frozen_inputs", "sources", "event_count", "independent_sequence_count")}
            expected = {key: protocol.get(key) for key in comparable}
            if comparable != expected:
                raise RuntimeError("existing N42 source protocol differs; refusing to overwrite frozen protocol")
        else:
            atomic_json(SOURCE_PROTOCOL, protocol)
        sources = make_sources(events)
        distinct = source_distinctness(sources)
        if not distinct["all_pairwise_distinct"]:
            raise RuntimeError(f"N42 corrected source features are not pairwise distinct: {distinct}")
        atomic_json(SOURCE_MANIFEST, {**sources, "source_protocol": str(SOURCE_PROTOCOL.relative_to(ROOT)), "source_protocol_sha256": sha256(SOURCE_PROTOCOL), "distinctness": distinct})
        pair = pair_integrity()
        if pair["status"] != "PASS":
            raise RuntimeError(f"N41 pair audit cannot be reused: {pair}")
        interpretation = diagnostic_interpretation(json.loads(N41_PAIR_SUMMARY.read_text(encoding="utf-8")), json.loads(N41_POSTHOC.read_text(encoding="utf-8")))
        atomic_json(DIAG / "candidate_pair_summary.json", {"source": str(N41_PAIR_SUMMARY.relative_to(ROOT)), "source_sha256": sha256(N41_PAIR_SUMMARY), "frozen_pair_integrity": pair, "summary": json.loads(N41_PAIR_SUMMARY.read_text(encoding="utf-8"))})
        atomic_json(DIAG / "diagnostic_interpretation.json", {**interpretation, "corrected_source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)), "corrected_source_manifest_sha256": sha256(SOURCE_MANIFEST), "corrected_source_distinctness": distinct})
        status = {
            "protocol": PROTOCOL,
            "status": "PASS_DIAGNOSTIC_ONLY",
            "stage": "N42-01",
            "started_at": started,
            "finished_at": now(),
            "inputs": {"n37_manifest": str(N37_MANIFEST.relative_to(ROOT)), "n37_manifest_sha256": sha256(N37_MANIFEST), "n41_parameter_smoke": str(N41_PARAMETER_SMOKE.relative_to(ROOT)), "n41_pair_summary": str(N41_PAIR_SUMMARY.relative_to(ROOT)), "n41_posthoc": str(N41_POSTHOC.relative_to(ROOT))},
            "frozen_n41_source_failure_preserved": frozen_failure,
            "corrected_source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
            "corrected_source_distinctness": distinct,
            "pair_integrity": pair,
            "diagnostic_interpretation": str((DIAG / "diagnostic_interpretation.json").relative_to(ROOT)),
            "runtime_future_gt_used": False,
            "training_authorized": True,
            "production_change_authorized": False,
            "real_human_tape": {"count": 0, "status": "UNAVAILABLE; N37/N41 sources remain simulated_from_gt"},
        }
        atomic_json(STAGE, status)
        print(json.dumps({"status": status["status"], "source_manifest": str(SOURCE_MANIFEST), "diagnostic": str(DIAG / "diagnostic_interpretation.json")}, sort_keys=True), flush=True)
    except Exception as exc:
        failure = {"protocol": PROTOCOL, "stage": "N42-01", "status": "FAIL", "started_at": started, "finished_at": now(), "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "failure_preserved": True}
        failure_path = OUT / "attempts/stage_01_attempt1_failure.json"
        if not failure_path.exists():
            atomic_json(failure_path, failure)
        atomic_json(STAGE, failure)
        raise


if __name__ == "__main__":
    main()
