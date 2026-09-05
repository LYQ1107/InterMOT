#!/usr/bin/env python3
"""Freeze the N72R9 source-aware training and replay protocol.

This preparation step is deliberately metadata-only with respect to outcomes.
It consumes the already sealed N72R9 reservation and the fixed N72R7 D2 input
manifests, but never reads post-treatment metrics to choose events or values.
The protocol is written to a new N72R9 directory and does not mutate any
historical N36--N72R8 artifact.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/N72R9"
RESERVATION_ROOT = OUTPUT_ROOT / "confirmation_reservation"
REPLAY_ROOT = ROOT / "outputs/N72R7/dev_replay/d2_full_attempt1"
REQUERY_ROOT = ROOT / "outputs/N72R7/candidate_generator/r5_full_attempt1/attempt_1"
EVENT_POLICY = ROOT / "outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json"
RESERVATION_STATUS = OUTPUT_ROOT / "stage_01_confirmation_reservation_status.json"
RESERVATION_MANIFEST = RESERVATION_ROOT / "reservation_manifest.json"
PROTOCOL_PATH = OUTPUT_ROOT / "protocol.json"
STAGE_PATH = OUTPUT_ROOT / "stage_02_protocol_status.json"

TRAIN_SEQUENCES = (
    "dancetrack0001", "dancetrack0002", "dancetrack0006", "dancetrack0008",
    "dancetrack0012", "dancetrack0015", "dancetrack0016", "dancetrack0023",
    "dancetrack0024", "dancetrack0027", "dancetrack0029", "dancetrack0032",
    "dancetrack0033", "dancetrack0037", "dancetrack0055", "dancetrack0062",
)
VALIDATION_SEQUENCES = ("dancetrack0051", "dancetrack0052")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def main() -> int:
    started = now_utc()
    status: dict[str, Any] = {
        "schema_version": "N72R9_PROTOCOL_STATUS_V1",
        "stage": "N72R9_SOURCE_AWARE_PROTOCOL_FREEZE",
        "started_at_utc": started,
        "runtime_future_gt_used": False,
        "posthoc_metrics_used_for_selection": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }
    try:
        reservation = read_json(RESERVATION_STATUS)
        reservation_manifest = read_json(RESERVATION_MANIFEST)
        if reservation.get("status") != "BLOCKED_NO_FRESH_UNTOUCHED_CONFIRMATION_SEQUENCES":
            raise RuntimeError(f"unexpected reservation status: {reservation.get('status')}")
        if int(reservation.get("eligible_sequence_count", -1)) != 0:
            raise RuntimeError("protocol freeze expects the sealed zero-eligible reservation")
        if reservation_manifest.get("selection_sealed") is not True:
            raise RuntimeError("reservation manifest is not sealed")
        policy = read_json(EVENT_POLICY)
        policy_events = {str(item["event_id"]): item for item in policy.get("events", [])}
        manifests = sorted(REPLAY_ROOT.glob("*/event_manifest.json"), key=lambda path: path.parent.name)
        if len(manifests) != 32:
            raise RuntimeError(f"expected the frozen 32-event N72R7 D2 input set, found {len(manifests)}")
        events: list[dict[str, Any]] = []
        missing_requery: list[str] = []
        for path in manifests:
            item = read_json(path)
            event_id = str(item.get("event_id"))
            if event_id not in policy_events:
                raise RuntimeError(f"replay input event is absent from frozen event policy: {event_id}")
            if item.get("status") != "PASS_N72R7_CLOSED_LOOP_EVENT_REPLAY":
                raise RuntimeError(f"replay input event is not PASS: {event_id}")
            if item.get("runtime_future_gt_used") is not False or item.get("posthoc_gt_used") is not False:
                raise RuntimeError(f"runtime/posthoc flag violation in frozen input manifest: {event_id}")
            sequence = str(item["sequence"])
            split = "validation" if sequence in VALIDATION_SEQUENCES else "train" if sequence in TRAIN_SEQUENCES else "development_only"
            if split == "development_only":
                raise RuntimeError(f"frozen event sequence has no pre-registered split: {sequence}")
            requery_path = REQUERY_ROOT / event_id / "frames.jsonl"
            if not requery_path.is_file():
                missing_requery.append(event_id)
            event = {
                "event_id": event_id,
                "sequence": sequence,
                "event_frame": int(item["event_frame"]),
                "future_window": [int(item["future_window"][0]), int(item["future_window"][1])],
                "action_type": str(policy_events[event_id]["action_type"]),
                "dataset_gt_id": int(policy_events[event_id]["dataset_gt_id"]),
                "current_gt_box": [float(value) for value in policy_events[event_id]["current_gt_box"]],
                "split": split,
                "source_event_manifest": str(path),
                "source_event_manifest_sha256": sha256_file(path),
                "c0_source": str(item["n72r6_c0_source"]),
                "c0_source_sha256": sha256_file(Path(str(item["n72r6_c0_source"]))),
                "c1_source": str(item["n72r6_c1_source"]),
                "c1_source_sha256": sha256_file(Path(str(item["n72r6_c1_source"]))),
                "target_stream_source": str(item["n72r6_target_stream_source"]),
                "target_stream_source_sha256": sha256_file(Path(str(item["n72r6_target_stream_source"]))),
                "requery_source": str(requery_path),
                "requery_source_sha256": sha256_file(requery_path) if requery_path.is_file() else None,
                "interaction_source": "simulated_from_gt",
                "not_real_human_evidence": True,
                "runtime_future_gt_used": False,
            }
            events.append(event)
        if missing_requery:
            raise RuntimeError(f"missing frozen requery source for {len(missing_requery)} events: {missing_requery[:3]}")
        events.sort(key=lambda item: str(item["event_id"]))
        sequence_map = {
            sequence: "validation" if sequence in VALIDATION_SEQUENCES else "train"
            for sequence in sorted({str(item["sequence"]) for item in events})
        }
        body: dict[str, Any] = {
            "schema_version": "N72R9_SOURCE_AWARE_TEMPORAL_PROTOCOL_V1",
            "created_at_utc": now_utc(),
            "reservation": {
                "status": reservation["status"],
                "registry_sha256": reservation.get("registry_sha256"),
                "protocol_sha256": reservation.get("protocol_sha256"),
                "reserved_sequence_count": int(reservation.get("reserved_sequence_count", 0)),
                "eligible_sequence_count": int(reservation.get("eligible_sequence_count", 0)),
                "fresh_confirmation_authorized": False,
            },
            "source_event_selection": {
                "source": "frozen_N72R7_D2_event_manifests_after_reservation",
                "event_count": len(events),
                "sequence_count": len(sequence_map),
                "events": events,
                "sequence_map": sequence_map,
                "selection_uses_post_treatment_metrics": False,
                "old_confirmation_sequences_excluded": ["dancetrack0020", "dancetrack0049"],
            },
            "candidate_stream": {
                "base": "frozen_N72R6_C0",
                "current_target": "frozen_N72R6_target_correction_stream",
                "future_requery": "frozen_N72R7_R5_candidate_generator_requery_stream_used_only_after_runtime_uncertainty_trigger",
                "candidate_order": "MAIN_B0_CANDIDATE_then_TARGET_SESSION_CURRENT_RAW_then_TARGET_SESSION_REQUERY",
                "checkpoint_changed": False,
                "candidate_definition_changed": False,
                "hungarian_solver_changed": False,
            },
            "model": {
                "name": "N72R9SourceAwareTemporalIdentityModel",
                "candidate_feature_dim": 530,
                "explicit_source_feature_dim": 4,
                "trusted_memory_slots": 4,
                "distractor_memory_slots": 4,
                "temporal_feature_dim": 8,
                "hidden_dim": 96,
                "layers": 1,
                "heads": 4,
                "candidate_sources": ["MAIN_B0_CANDIDATE", "TARGET_SESSION_CURRENT_RAW", "TARGET_SESSION_REQUERY", "UNKNOWN"],
                "memory_update": "causal_base_only_selection_with_fixed_admission_rule",
                "public_id_inference": False,
            },
            "training": {
                "seed": 7290,
                "sequence_split": {"train": list(TRAIN_SEQUENCES), "validation": list(VALIDATION_SEQUENCES)},
                "label": "highest_IoU_candidate_at_least_0.50_else_NONE_from_offline_train_GT",
                "future_metrics_used": False,
                "epochs_max": 40,
                "patience": 8,
                "batch_size": 128,
                "learning_rate": 0.0005,
                "weight_decay": 0.0001,
                "loss": "cross_entropy_plus_fixed_hard_negative_pairwise_margin",
                "checkpoint_selection": "minimum_validation_loss_only",
            },
            "replay": {
                "variants": {
                    "BASELINE_B0": "frozen B0 candidates and exact public solver",
                    "TEMPORAL_CURRENT": "B0 plus current target-session candidates plus trained source-aware temporal model",
                    "TEMPORAL_REQUERY": "TEMPORAL_CURRENT plus frozen requery candidates only after uncertainty trigger",
                },
                "horizon_frames": [20, 50, 100],
                "future_window": "event_frame+1_through_event_frame+100",
                "uncertainty_trigger": {
                    "rule": "assigned_target_none_or_base_target_top1_minus_top2_margin_below_threshold",
                    "margin_threshold": 0.25,
                    "max_requery_per_frame": 1,
                    "no_future_gt_input": True,
                },
                "model_admission": {"score_threshold": 0.50, "margin_threshold": 0.20, "injection_scale": 1.0},
                "bootstrap": {"seed": 7290, "repetitions": 2000, "cluster_unit": "independent_sequence"},
                "posthoc_only_fields": ["GT", "identity_error", "missing", "IoU", "IDSW", "H20", "H50", "H100"],
            },
            "resource_plan": {
                "max_gpu_count": 4,
                "one_process_per_gpu_or_sequence": True,
                "current_plan": "CPU_replay_first; GPU only if an evidence-backed large run is required",
                "oom_sharding": [160, 100, 50],
            },
            "iclr_2027": {
                "abstract_deadline_aoe": "2026-09-18",
                "full_paper_deadline_aoe": "2026-09-25",
                "timezone": "AoE",
            },
            "interaction_source": "simulated_from_gt",
            "real_human_evidence": False,
            "runtime_future_gt_used": False,
            "posthoc_metrics_used_for_selection": False,
            "historical_outputs_read_only": True,
        }
        body["protocol_sha256"] = canonical_hash(body)
        atomic_json(PROTOCOL_PATH, body)
        result = {
            **status,
            "status": "PASS_N72R9_PROTOCOL_FROZEN_NO_FRESH_CONFIRMATION",
            "finished_at_utc": now_utc(),
            "protocol": str(PROTOCOL_PATH),
            "protocol_sha256": body["protocol_sha256"],
            "event_count": len(events),
            "sequence_count": len(sequence_map),
            "train_event_count": sum(item["split"] == "train" for item in events),
            "validation_event_count": sum(item["split"] == "validation" for item in events),
            "fresh_confirmation_authorized": False,
            "runtime_future_gt_used": False,
            "posthoc_metrics_used_for_selection": False,
            "production_authorized": False,
        }
        atomic_json(STAGE_PATH, result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        result = {
            **status,
            "status": "FAIL_N72R9_PROTOCOL_FREEZE",
            "finished_at_utc": now_utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "runtime_future_gt_used": False,
            "production_authorized": False,
        }
        atomic_json(STAGE_PATH, result)
        print(json.dumps(result, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
