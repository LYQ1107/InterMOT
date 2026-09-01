#!/usr/bin/env python3
"""Freeze the N41-02 appearance-source protocol and build its source sidecar.

This is a controlled mechanism experiment over the already frozen N37 event
instances.  It is deliberately not a human-tape collector and it never
creates a real-human event.  A and B are two attestations of the same frozen
GT-box ROI pathway (A is an independently recomputed upper-bound feature; B
is the feature stored by N37).  C is a preregistered image-space corruption:
offset/expanded-box background, a geometric erosion *proxy* (N37 has no
human-confirmed masks), and deterministic feature noise.  No machine
candidate embedding is used to construct any source.

The sidecar is consumed by runtime workers which load only the selected
feature vector and source provenance.  Future GT is neither loaded nor sent
to those workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
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


N37_MANIFEST = ROOT / "outputs" / "n37" / "real_event_manifest.json"
N36_TAPE_MANIFEST = ROOT / "outputs" / "n36" / "real_tape" / "tape_manifest.json"
CHECKPOINT = ROOT / "outputs" / "n9" / "checkpoints" / "osnet_x1_0_market1501.pth"
OUT = ROOT / "outputs" / "n41"
SOURCE_DIR = OUT / "source_replay"
PROTOCOL_PATH = SOURCE_DIR / "source_protocol.json"
SOURCE_MANIFEST = SOURCE_DIR / "source_embedding_manifest.json"
STAGE_PATH = OUT / "stage_02_status.json"
PROTOCOL = "N41_GT_CONTROLLED_APPEARANCE_SOURCE_ABLATION_V1"
SOURCES = ("A_ideal_gt_roi", "B_frozen_current_human_region", "C_fixed_corrupted_roi")
LAMBDA_GRID = (1.0, 8.0)
HUMAN_WEIGHT_GRID = (1.0,)
SMOKE_ACTIONS = (
    "AUTHORITATIVE_REASSIGN",
    "ATOMIC_ID_SWAP",
    "RECOVER_IDENTITY",
)

# These values are frozen before source generation and are deliberately
# modest.  They are not selected from any future metric or replay result.
CORRUPTION = {
    "box_center_offset_fraction": {"x": 0.08, "y": -0.06},
    "background_box_scale": 1.20,
    "geometric_erosion_fraction_each_side": 0.075,
    "background_mix": 0.25,
    "embedding_noise_std": 0.05,
    "noise_seed_base": 41000,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_digest(feature: Any) -> str:
    vector = np.asarray(feature, dtype=np.float32).reshape(-1)
    return hashlib.sha256(vector.tobytes()).hexdigest()


def normalize(feature: Any, label: str) -> np.ndarray:
    vector = np.asarray(feature, dtype=np.float32).reshape(-1)
    if vector.size != FEATURE_DIM or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label}:invalid_feature_shape_or_values:{vector.shape}")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        raise ValueError(f"{label}:zero_feature")
    return vector / norm


def load_events() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(N37_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("event_count") != 24:
        raise RuntimeError("frozen N37 manifest is not PASS/24")
    events = payload.get("events")
    if not isinstance(events, list) or len(events) != 24:
        raise RuntimeError("frozen N37 events are not exactly 24")
    ids = [str(item.get("event", {}).get("event_id")) for item in events]
    if any(item == "None" for item in ids) or len(set(ids)) != 24:
        raise RuntimeError("N37 event IDs are missing or duplicated")
    sequences = {str(item["event"]["sequence"]) for item in events}
    if len(sequences) != 21:
        raise RuntimeError(f"N37 independent sequence count changed: {len(sequences)}")
    return payload, events


def choose_smoke(events: list[dict[str, Any]]) -> list[str]:
    selected: list[str] = []
    seen_actions: set[str] = set()
    seen_sequences: set[str] = set()
    for item in events:
        event = item["event"]
        action = str(event["action_type"])
        sequence = str(event["sequence"])
        if action not in SMOKE_ACTIONS or action in seen_actions or sequence in seen_sequences:
            continue
        selected.append(str(event["event_id"]))
        seen_actions.add(action)
        seen_sequences.add(sequence)
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise RuntimeError(f"deterministic three-event smoke selection failed: {selected}")
    return selected


def configuration_grid() -> list[dict[str, float | str]]:
    return [
        {
            "config_id": f"lambda_{float(lambda_value):g}_human_{float(human_value):g}".replace(".", "p"),
            "lambda_assoc": float(lambda_value),
            "human_weight": float(human_value),
        }
        for lambda_value in LAMBDA_GRID
        for human_value in HUMAN_WEIGHT_GRID
    ]


def build_protocol(n37_payload: dict[str, Any], events: list[dict[str, Any]], smoke_ids: list[str]) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "status": "FROZEN_BEFORE_SOURCE_GENERATION_AND_REPLAY",
        "frozen_inputs": {
            "n37_event_manifest": str(N37_MANIFEST.relative_to(ROOT)),
            "n37_event_manifest_sha256": sha256(N37_MANIFEST),
            "n36_candidate_tape": str(N36_TAPE_MANIFEST.relative_to(ROOT)),
            "n36_candidate_tape_sha256": sha256(N36_TAPE_MANIFEST),
            "human_feature_checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "human_feature_checkpoint_sha256": sha256(CHECKPOINT),
            "n37_protocol": n37_payload.get("protocol"),
        },
        "controlled_gt_boundary": {
            "gt_allowed_for": ["current_event_frame_target_box_for_source_generation_only"],
            "gt_forbidden_for": ["runtime_workers", "future_candidate_stream", "runtime_assignment", "source_or_config_selection", "post_treatment_selection"],
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "not_historical_human_click_or_annotation": True,
        },
        "sources": {
            "A_ideal_gt_roi": {
                "role": "mechanism_upper_bound",
                "definition": "fresh HumanFeatureExtractor OSNet feature from the frozen current-event gt_box crop",
                "human_evidence_claim": "none; controlled GT ROI upper bound only",
            },
            "B_frozen_current_human_region": {
                "role": "frozen_current_path",
                "definition": "human_embedding stored in the frozen N37 manifest, generated by the same explicit ROI extractor path",
                "human_evidence_claim": "none; N37 interaction_source=simulated_from_gt",
            },
            "C_fixed_corrupted_roi": {
                "role": "fixed_quality_stress_test",
                "definition": "raw-image ROI with frozen offset and expanded background crop, geometric erosion proxy, fixed background mixture and deterministic feature noise",
                "human_evidence_claim": "none; not a confirmed mask and not a real human event",
                "mask_semantics": "geometric_inner_box_proxy_only; no N37 human-confirmed mask exists",
                "parameters": CORRUPTION,
            },
        },
        "weight_grid": configuration_grid(),
        "variant_definitions": {
            "M0": "K1 only; CCAM disabled",
            "M1": "human EMA prototype",
            "M2": "human EMA prototype + positive human anchors",
            "M3": "M2 + negative competitor bank",
            "M4": "M3 + reliability/age gate",
        },
        "smoke": {
            "event_ids": smoke_ids,
            "require_distinct_action_and_sequence": True,
            "source_count": 3,
            "configuration_count": len(configuration_grid()),
            "variants_per_worker": 5,
            "checks": [
                "score_matrix_dimensions",
                "candidate_order_and_mapping",
                "same_future_candidate_stream_across_sources_and_weights",
                "runtime_future_gt_false",
                "event_frame_memory_read_false_and_t_plus_1_first_read",
                "assignment_reproducible_for_same_source_and_config",
            ],
        },
        "full_replay": {
            "event_count": 24,
            "independent_sequence_count": 21,
            "source_count": 3,
            "configuration_count": len(configuration_grid()),
            "runtime_workers": 24 * 3 * len(configuration_grid()),
            "variants_per_worker": 5,
            "future_horizons": [20, 50, 100],
            "same_frozen_prefix_and_future_tape": True,
        },
        "forbidden_adaptations": [
            "future-GT event or source selection",
            "weight/configuration selection after replay",
            "threshold or score normalization changes",
            "checkpoint/candidate/future-window/metric changes",
            "calibration head, selector, decoder LoRA",
            "calling any source real human evidence",
        ],
        "events": [
            {
                "event_id": str(item["event"]["event_id"]),
                "sequence": str(item["event"]["sequence"]),
                "action_type": str(item["event"]["action_type"]),
                "event_frame": int(item["event"]["frame"]),
                "future_frame_start": int(item["event"]["frame"]) + 1,
                "future_frame_end": int(item["future_frame_end"]),
            }
            for item in events
        ],
    }


def image_box_for_mask(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    return x1, y1, x2, y2


def transformed_box(box: np.ndarray) -> dict[str, Any]:
    x1, y1, x2, y2 = [float(value) for value in box]
    width, height = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2.0 + CORRUPTION["box_center_offset_fraction"]["x"] * width
    cy = (y1 + y2) / 2.0 + CORRUPTION["box_center_offset_fraction"]["y"] * height
    expanded_width = width * float(CORRUPTION["background_box_scale"])
    expanded_height = height * float(CORRUPTION["background_box_scale"])
    raw = np.asarray(
        [cx - expanded_width / 2.0, cy - expanded_height / 2.0, cx + expanded_width / 2.0, cy + expanded_height / 2.0],
        dtype=float,
    )
    return {
        "original_box": [x1, y1, x2, y2],
        "offset_fraction": CORRUPTION["box_center_offset_fraction"],
        "expanded_scale": float(CORRUPTION["background_box_scale"]),
        "raw_transformed_box": raw.tolist(),
    }


def generate_sources(item: dict[str, Any], event_index: int, extractor: HumanFeatureExtractor) -> dict[str, Any]:
    event = item["event"]
    event_id = str(event["event_id"])
    sequence = str(event["sequence"])
    frame = int(event["frame"])
    box = np.asarray(event.get("gt_box"), dtype=float).reshape(-1)
    if box.size != 4 or not np.all(np.isfinite(box)) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{event_id}:invalid_current_event_box")
    sequence_dir = DATA_ROOT / "train" / sequence / "img1"
    image_path = sequence_dir / f"{frame + 1:08d}.jpg"
    if not image_path.is_file():
        raise FileNotFoundError(f"{event_id}:current_event_image_missing:{image_path}")

    # A is deliberately recomputed from the frozen current-event box.  It is
    # an upper-bound mechanism probe, not an annotation provenance claim.
    ideal = normalize(extractor.extract(sequence_dir, frame, box), f"{event_id}:A")
    frozen = normalize(event.get("human_embedding"), f"{event_id}:B")

    from PIL import Image

    with Image.open(image_path) as image:
        image_width, image_height = image.size
    transform = transformed_box(box)
    transformed = np.asarray(transform["raw_transformed_box"], dtype=float)
    background = normalize(extractor.extract(sequence_dir, frame, transformed), f"{event_id}:C_background")
    x1, y1, x2, y2 = image_box_for_mask(transformed, image_width, image_height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{event_id}:C_transformed_box_empty")
    mask_height, mask_width = y2 - y1, x2 - x1
    erosion = float(CORRUPTION["geometric_erosion_fraction_each_side"])
    mx, my = int(round(mask_width * erosion)), int(round(mask_height * erosion))
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    mask[my : max(my + 1, mask_height - my), mx : max(mx + 1, mask_width - mx)] = 1
    eroded = normalize(
        extractor.extract_mask(sequence_dir, frame, transformed, mask),
        f"{event_id}:C_eroded",
    )
    mix = float(CORRUPTION["background_mix"])
    mixed = normalize((1.0 - mix) * eroded + mix * background, f"{event_id}:C_mixed")
    rng = np.random.default_rng(int(CORRUPTION["noise_seed_base"]) + int(event_index))
    noise = rng.normal(0.0, float(CORRUPTION["embedding_noise_std"]), size=FEATURE_DIM).astype(np.float32)
    corrupted = normalize(mixed + noise, f"{event_id}:C")

    sources = {
        "A_ideal_gt_roi": {
            "feature": ideal.tolist(),
            "feature_sha256": feature_digest(ideal),
            "feature_origin": "fresh HumanFeatureExtractor.extract(current_event_frame, frozen_gt_box)",
            "role": "mechanism_upper_bound_only",
            "interaction_source": "simulated_from_gt",
            "gt_scope": "current_event_box_only",
        },
        "B_frozen_current_human_region": {
            "feature": frozen.tolist(),
            "feature_sha256": feature_digest(frozen),
            "feature_origin": str(event.get("human_embedding_source", event.get("human_feature_source", "frozen_N37_human_embedding"))),
            "frozen_n37_feature_sha256": event.get("human_feature_digest"),
            "role": "frozen_current_N37_path",
            "interaction_source": "simulated_from_gt",
            "gt_scope": "current_event_box_only_in_original_N37_generation",
        },
        "C_fixed_corrupted_roi": {
            "feature": corrupted.tolist(),
            "feature_sha256": feature_digest(corrupted),
            "feature_origin": "raw_image_transformed_roi_with_controlled_corruption",
            "role": "fixed_quality_stress_test_not_human_evidence",
            "interaction_source": "simulated_from_gt",
            "gt_scope": "current_event_box_only",
            "transform": {
                **transform,
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "clipped_crop_box": [x1, y1, x2, y2],
                "background_component_sha256": feature_digest(background),
                "geometric_erosion_proxy": {
                    "fraction_each_side": erosion,
                    "mask_shape": [mask_height, mask_width],
                    "positive_pixel_count": int(mask.sum()),
                    "mask_source": "controlled_rectangular_corruption_proxy; not confirmed human mask",
                    "eroded_component_sha256": feature_digest(eroded),
                },
                "background_mix": mix,
                "embedding_noise_std": float(CORRUPTION["embedding_noise_std"]),
                "noise_seed": int(CORRUPTION["noise_seed_base"]) + int(event_index),
            },
        },
    }
    for source in sources.values():
        source.update({
            "feature_dim": FEATURE_DIM,
            "finite": True,
            "unit_norm": abs(float(np.linalg.norm(np.asarray(source["feature"], dtype=np.float32))) - 1.0) < 1e-4,
            "machine_candidate_embedding_used": False,
            "runtime_future_gt_used": False,
        })
    return {
        "event_id": event_id,
        "sequence": sequence,
        "event_frame": frame,
        "action_type": str(event["action_type"]),
        "event_box_sha256": hashlib.sha256(np.asarray(box, dtype=np.float32).tobytes()).hexdigest(),
        "sources": sources,
        "runtime_worker_contract": {
            "worker_receives_only_selected_source_feature": True,
            "current_event_gt_box_sent_to_worker": False,
            "future_gt_sent_to_worker": False,
            "runtime_future_gt_used": False,
        },
    }


def run(attempt: int) -> dict[str, Any]:
    started = now()
    n37_payload, events = load_events()
    smoke_ids = choose_smoke(events)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"human feature checkpoint missing: {CHECKPOINT}")
    protocol = build_protocol(n37_payload, events, smoke_ids)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError("existing N41-02 source protocol differs; refusing to change frozen protocol")
    else:
        atomic_json(PROTOCOL_PATH, protocol)

    extractor = HumanFeatureExtractor(CHECKPOINT)
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(events):
        entries.append(generate_sources(item, index, extractor))
        print(json.dumps({"event_id": entries[-1]["event_id"], "source_status": "PASS"}, sort_keys=True), flush=True)
    payload = {
        "protocol": PROTOCOL,
        "status": "PASS",
        "attempt": int(attempt),
        "started_at": started,
        "finished_at": now(),
        "event_count": len(entries),
        "independent_sequence_count": len({entry["sequence"] for entry in entries}),
        "source_count": len(SOURCES),
        "source_entry_count": len(entries) * len(SOURCES),
        "sources": list(SOURCES),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_tape": True,
        "protocol_artifact": str(PROTOCOL_PATH.relative_to(ROOT)),
        "n37_manifest_sha256": sha256(N37_MANIFEST),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "events": entries,
    }
    atomic_json(SOURCE_MANIFEST, payload)
    prior_failures = []
    if STAGE_PATH.exists():
        old = json.loads(STAGE_PATH.read_text(encoding="utf-8"))
        prior_failures = list(old.get("failure_artifacts", []))
        if old.get("status") == "FAIL" and old.get("failure_artifact") and old["failure_artifact"] not in prior_failures:
            prior_failures.append(old["failure_artifact"])
    stage = {
        "stage": "N41-02",
        "status": "SOURCE_SIDEcar_PASS_READY_FOR_SMOKE",
        "protocol": PROTOCOL,
        "attempt": int(attempt),
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "protocol_artifact": str(PROTOCOL_PATH.relative_to(ROOT)),
        "event_count": len(entries),
        "independent_sequence_count": len({entry["sequence"] for entry in entries}),
        "source_count": len(SOURCES),
        "source_entry_count": len(entries) * len(SOURCES),
        "runtime_future_gt_used": False,
        "real_human_tape_created": False,
        "downstream_replay_started": False,
        "failure_artifacts": prior_failures,
        "next_action": "Run the preregistered 3-event A/B/C smoke in independent workers; only then run the remaining frozen 24-event source replay.",
    }
    atomic_json(STAGE_PATH, stage)
    return stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    try:
        result = run(args.attempt)
        print(json.dumps({"status": result["status"], "source_manifest": result["source_manifest"], "event_count": result["event_count"]}, sort_keys=True), flush=True)
    except Exception as exc:
        failure_path = OUT / "attempts" / f"stage_02_source_prepare_attempt{int(args.attempt)}_failure.json"
        atomic_json(
            failure_path,
            {
                "protocol": PROTOCOL,
                "status": "FAIL",
                "attempt": int(args.attempt),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "artifact_is_failure_evidence": True,
                "runtime_future_gt_used": False,
            },
        )
        atomic_json(
            STAGE_PATH,
            {
                "stage": "N41-02",
                "status": "FAIL_SOURCE_PREPARATION",
                "protocol": PROTOCOL,
                "attempt": int(args.attempt),
                "failure_artifact": str(failure_path.relative_to(ROOT)),
                "real_human_tape_created": False,
                "downstream_replay_started": False,
                "downstream_authorized": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
