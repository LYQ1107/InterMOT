#!/usr/bin/env python3
"""Finalize the N72R15 integrity and research gate from immutable artifacts.

This is deliberately post-hoc and CPU-only.  It does not rerun SAM3, change a
metric, choose an event, or alter any historical artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "N72R15"
FORMAL = OUT / "formal_attempt_04" / "formal_manifest.json"
CAUSAL = OUT / "diagnostics_attempt_04" / "causal_metrics_seed7215.json"
TRACKEVAL = OUT / "trackeval_attempt_01" / "trackeval_run_manifest.json"
AUDITS = {
    "state_propagation": OUT / "diagnostics_attempt_04" / "state_propagation_audit.json",
    "candidate_coverage": OUT / "diagnostics_attempt_04" / "candidate_coverage.json",
    "component_diagnosis": OUT / "diagnostics_attempt_04" / "state_edge_component_diagnosis.json",
    "self_reinforcement": OUT / "diagnostics_attempt_04" / "self_reinforcement_audit.json",
    "assignment_cardinality": OUT / "diagnostics_attempt_04" / "assignment_cardinality_audit.json",
}
CODE_FILES = [
    "sam3_intermot/association/trusted_persistent_public_state.py",
    "sam3_intermot/association/relative_persistent_state_edge.py",
    "scripts/n72r15_run_repaired_persistent_state_formal.py",
    "scripts/n72r15_interface_audit.py",
    "scripts/n72r15_aggregate_metrics.py",
    "scripts/n72r15_candidate_coverage_posthoc.py",
    "scripts/n72r15_state_edge_component_diagnosis.py",
    "scripts/n72r15_state_propagation_audit.py",
    "scripts/n72r15_integrity_audits.py",
    "scripts/n72r15_export_window_trackeval.py",
    "scripts/n72r15_run_window_trackeval.py",
    "scripts/n72r15_finalize_gate.py",
]


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def atomic_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def all_scalar_flags(node: Any, key: str, path: str = "") -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for name, value in node.items():
            current = f"{path}.{name}" if path else name
            if name == key:
                found.append((current, value))
            found.extend(all_scalar_flags(value, key, current))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(all_scalar_flags(value, key, f"{path}[{index}]"))
    return found


def main() -> int:
    previous_gate_path = OUT / "n72r15_final_gate.json"
    if previous_gate_path.is_file():
        try:
            previous_gate = load(previous_gate_path)
            if previous_gate.get("status") == "BLOCKED_INTEGRITY":
                for attempt in range(1, 10):
                    preserved_failure_path = OUT / f"finalizer_failure_attempt_{attempt:02d}.json"
                    if preserved_failure_path.exists():
                        continue
                    atomic_dump(preserved_failure_path, {
                        "schema_version": "N72R15_FINALIZER_FAILURE_PRESERVED_V1",
                        "preserved_at_utc": datetime.now(timezone.utc).isoformat(),
                        "attempt": attempt,
                        "original_gate": previous_gate,
                    })
                    break
        except (OSError, json.JSONDecodeError, AttributeError):
            # A fresh finalization must still report the current actionable error.
            pass
    formal = load(FORMAL)
    causal = load(CAUSAL)
    trackeval = load(TRACKEVAL)
    audit_data = {name: load(path) for name, path in AUDITS.items()}
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    expected_variants = [
        "E0_BASELINE_B0",
        "E1I_HUMAN_RELATIVE_STATE",
        "E1J_TRUSTED_GLOBAL_RELATIVE_STATE",
    ]
    events = formal.get("events", [])
    event_ids = [event.get("event_id") for event in events]
    sequences = {event.get("sequence") for event in events}
    check(formal.get("status") == "PASS_N72R15_FORMAL_REPLAY", "formal manifest is not PASS")
    check(formal.get("event_count") == 32, f"formal event_count={formal.get('event_count')}, expected 32")
    check(formal.get("expected_runtime_rows") == 32 * 3 * 101, "formal expected_runtime_rows is not 9696")
    check(len(event_ids) == len(set(event_ids)) == 32, "event_id uniqueness/count failed")
    check(len(sequences) == 18, f"independent sequence count={len(sequences)}, expected 18")
    check(formal.get("variants") == expected_variants, "formal variant axis changed")
    check(formal.get("candidate_pool_policy") == "MAIN_B0_ONLY", "candidate pool policy is not MAIN_B0_ONLY")
    check(formal.get("target_session_candidate_in_solver") is False, "target-session candidate is enabled in solver")
    check(formal.get("runtime_future_gt_used") is False, "formal manifest reports runtime GT use")
    check(formal.get("runtime_gt_read") is False, "formal manifest reports runtime GT read")
    check(formal.get("interaction_source") == "simulated_from_gt", "interaction source was relabeled")
    check(formal.get("not_real_human_evidence") is True, "simulated events are not marked non-human")

    runtime_rows = 0
    runtime_key_set: set[tuple[str, str, int]] = set()
    event_frame_rows = 0
    future_rows = 0
    candidate_uids_checked = 0
    row_max_checks = 0
    for event in events:
        event_id = event.get("event_id")
        event_frame = event.get("event_frame")
        check(event.get("status") == "PASS_N72R15_FORMAL_EVENT", f"event {event_id} not PASS")
        check(event.get("failures") == [], f"event {event_id} retains failures")
        variants = event.get("variants", [])
        check([variant.get("variant") for variant in variants] == expected_variants, f"variant axis failed for {event_id}")
        for variant in variants:
            variant_name = variant.get("variant")
            check(variant.get("status") == "PASS_N72R15_VARIANT", f"{event_id}/{variant_name} not PASS")
            check(variant.get("frame_count") == 101, f"{event_id}/{variant_name} frame_count is not 101")
            frame_path = Path(variant.get("frames", ""))
            check(frame_path.is_file(), f"missing runtime frame file {event_id}/{variant_name}")
            if not frame_path.is_file():
                continue
            rows = []
            with frame_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        failures.append(f"invalid JSON {frame_path}:{line_number}: {exc}")
            check(len(rows) == 101, f"{event_id}/{variant_name} runtime row count={len(rows)}, expected 101")
            frames = [row.get("frame") for row in rows]
            check(len(set(frames)) == len(frames), f"duplicate frames in {event_id}/{variant_name}")
            expected_frames = list(range(int(event_frame), int(event_frame) + 101))
            check(frames == expected_frames, f"frame range/order failed for {event_id}/{variant_name}")
            for row in rows:
                runtime_rows += 1
                key = (str(event_id), str(variant_name), int(row.get("frame")))
                check(key not in runtime_key_set, f"duplicate runtime key {key}")
                runtime_key_set.add(key)
                for flag_key in ("runtime_future_gt_used", "runtime_gt_read", "public_id_inference", "posthoc_gt_used"):
                    check(row.get(flag_key) is False, f"{key} {flag_key} is not false")
                check(row.get("event_frame") == event_frame, f"{key} event_frame mismatch")
                check(row.get("first_memory_visible_frame") == int(event_frame) + 1, f"{key} causal boundary mismatch")
                if int(row.get("frame")) == int(event_frame):
                    event_frame_rows += 1
                    check(row.get("record_kind") == "event_frame_correction", f"{key} record kind mismatch")
                    check(row.get("memory_read") is False and row.get("event_frame_memory_read") is False, f"{key} event frame read new memory")
                    check(row.get("candidate_count") == 0 and row.get("candidate_pool") is None, f"{key} event frame entered solver")
                    check(row.get("score_audit") is None, f"{key} event frame has solver audit")
                    check(row.get("memory_write") is (variant_name != "E0_BASELINE_B0"), f"{key} event memory-write semantics failed")
                else:
                    future_rows += 1
                    pool = row.get("candidate_pool")
                    state = row.get("persistent_state_association") or {}
                    score = row.get("score_audit") or {}
                    assignment = row.get("assignment") or {}
                    check(row.get("record_kind") == "future_association_frame", f"{key} record kind mismatch")
                    expected_memory_read = variant_name != "E0_BASELINE_B0"
                    check(row.get("memory_read") is expected_memory_read, f"{key} runtime memory-read causal flag failed")
                    check(pool.get("candidate_pool_policy") == "MAIN_B0_ONLY", f"{key} candidate policy changed")
                    check(pool.get("target_session_candidate_in_solver") is False, f"{key} target-session candidate entered solver")
                    check(state.get("target_session_candidate_in_solver") is False, f"{key} state target-session candidate enabled")
                    check(score.get("runtime_future_gt_used") is False, f"{key} score audit runtime GT flag failed")
                    candidate_rows = pool.get("candidate_rows") or []
                    uids = [candidate.get("candidate_uid") for candidate in candidate_rows]
                    check(len(uids) == len(set(uids)), f"{key} duplicate candidate UID")
                    check(all(candidate.get("source_kind") == "MAIN_B0_CANDIDATE" for candidate in candidate_rows), f"{key} non-MAIN candidate source")
                    check(all(candidate.get("candidate_source") == "MAIN_B0_CANDIDATE" for candidate in candidate_rows), f"{key} non-MAIN candidate label")
                    check(not any("TARGET_SESSION_CURRENT_RAW" in str(candidate) for candidate in candidate_rows), f"{key} raw target-session candidate present")
                    check(all(candidate.get("runtime_future_gt_used") is False and candidate.get("runtime_gt_read") is False for candidate in candidate_rows), f"{key} candidate GT flag failed")
                    candidate_uids_checked += len(uids)
                    edge = row.get("persistent_state_association") or {}
                    check(edge.get("state_edge_scale") == 1.0, f"{key} state-edge scale changed")
                    check(edge.get("row_max_preserved") is True, f"{key} row-max preservation failed")
                    row_max_checks += 1
                    check(edge.get("candidate_uids") == uids, f"{key} state/candidate axis mismatch")
                    check(assignment.get("candidate_axis") == uids, f"{key} assignment/candidate axis mismatch")

    check(runtime_rows == 9696, f"runtime row count={runtime_rows}, expected 9696")
    check(len(runtime_key_set) == runtime_rows, "runtime key uniqueness failed")
    check(event_frame_rows == 96, f"event-frame row count={event_frame_rows}, expected 96")
    check(future_rows == 9600, f"future row count={future_rows}, expected 9600")
    check(row_max_checks == 9600, f"row-max check count={row_max_checks}, expected 9600")

    # All nested runtime flags are checked once more to catch a newly added field.
    for flag_key in ("runtime_future_gt_used", "runtime_gt_read"):
        for location, value in all_scalar_flags(formal, flag_key):
            check(value is False, f"formal nested {flag_key}={value!r} at {location}")

    pooled_by_horizon: dict[str, dict[str, dict[str, float]]] = {}
    for result in trackeval.get("horizon_results", []):
        pooled_by_horizon[str(result.get("horizon"))] = result.get("pooled", {})
        check(result.get("status") == "PASS", f"TrackEval horizon {result.get('horizon')} not PASS")
        check(result.get("record_count") == 96, f"TrackEval horizon {result.get('horizon')} record count failed")
    check(trackeval.get("status") == "PASS_TRACKEVAL_N72R15", "TrackEval manifest not PASS")
    check(trackeval.get("record_count") == 288 and trackeval.get("expected_record_count") == 288, "TrackEval 288-row gate failed")
    check(trackeval.get("duplicate_keys") == [], "TrackEval duplicate keys present")
    check(trackeval.get("missing_keys") == [], "TrackEval missing keys present")
    check(trackeval.get("runtime_future_gt_used") is False, "TrackEval runtime GT flag failed")
    check(trackeval.get("trackeval_commit") == "12c8791b303e0a0b50f753af204249e622d0281a", "TrackEval commit is not pinned")

    metric_names = ["HOTA", "AssA", "IDF1", "IDSW", "DetA", "MOTA"]
    first_screen: dict[str, Any] = {"horizon": 100, "pooled": {}, "deltas": {}}
    for variant, metrics in pooled_by_horizon.get("100", {}).items():
        first_screen["pooled"][variant] = {name: metrics.get(name) for name in metric_names}
    b0 = first_screen["pooled"].get("E0_BASELINE_B0", {})
    for variant in ("E1I_HUMAN_RELATIVE_STATE", "E1J_TRUSTED_GLOBAL_RELATIVE_STATE"):
        values = first_screen["pooled"].get(variant, {})
        first_screen["deltas"][f"{variant}_MINUS_B0"] = {
            name: (values.get(name) - b0.get(name)) if finite(values.get(name)) and finite(b0.get(name)) else None
            for name in metric_names
        }

    g2_h100 = causal["comparisons"]["G2_MINUS_B0"]["100"]
    g2_h20 = causal["comparisons"]["G2_MINUS_B0"]["20"]
    strict_criteria = {
        "h100_delta_hota_ge_0.003": first_screen["deltas"].get("E1J_TRUSTED_GLOBAL_RELATIVE_STATE_MINUS_B0", {}).get("HOTA", -math.inf) >= 0.003,
        "h100_delta_assa_ge_0.005": first_screen["deltas"].get("E1J_TRUSTED_GLOBAL_RELATIVE_STATE_MINUS_B0", {}).get("AssA", -math.inf) >= 0.005,
        "h100_delta_idf1_ge_0.005": first_screen["deltas"].get("E1J_TRUSTED_GLOBAL_RELATIVE_STATE_MINUS_B0", {}).get("IDF1", -math.inf) >= 0.005,
        "h100_idsw_not_worse": first_screen["pooled"].get("E1J_TRUSTED_GLOBAL_RELATIVE_STATE", {}).get("IDSW", math.inf) <= b0.get("IDSW", -math.inf),
        "h20_target_identity_error_reduction_gt_0": g2_h20.get("mean_target_identity_error_reduction", -math.inf) > 0,
        "h100_sequence_cluster_ci_lower_gt_0": g2_h100.get("sequence_cluster_bootstrap_95ci", {}).get("lower", -math.inf) > 0,
        "protected_accuracy_not_worse": g2_h100.get("mean_protected_accuracy_delta", -math.inf) >= 0,
    }
    audit_statuses = {name: data.get("status") for name, data in audit_data.items()}
    all_audits_pass = all(str(status).startswith("PASS") for status in audit_statuses.values())
    strict_criteria["all_posthoc_audits_pass"] = all_audits_pass
    strict_criteria["formal_integrity_pass"] = not failures
    strong_positive = not failures and all(strict_criteria.values())

    coverage_h100 = audit_data["candidate_coverage"].get("aggregate", {}).get("100", {}).get("all", {})
    component_h100 = audit_data["component_diagnosis"].get("by_horizon", {}).get("100", {})
    component_auc = {}
    for key, data in component_h100.items():
        component_auc[key] = {
            component: data.get(component, {}).get("roc_auc_positive_vs_negative")
            for component in ("appearance", "motion", "native_continuity", "gap", "raw", "relative_delta")
        }

    source_hashes = {}
    for relative in CODE_FILES:
        path = ROOT / relative
        if path.is_file():
            source_hashes[relative] = sha256_file(path)

    audit_hashes = {name: sha256_file(path) for name, path in AUDITS.items() if path.is_file()}
    input_hashes = {
        "formal_manifest": sha256_file(FORMAL),
        "causal_metrics_seed7215": sha256_file(CAUSAL),
        "trackeval_run_manifest": sha256_file(TRACKEVAL),
        "posthoc_audits": audit_hashes,
    }
    causal_summary = {}
    for comparison in ("H1_MINUS_B0", "G2_MINUS_B0", "G2_MINUS_H1"):
        causal_summary[comparison] = {}
        for horizon in ("20", "50", "100"):
            item = causal["comparisons"][comparison][horizon]
            ci = item.get("sequence_cluster_bootstrap_95ci", {})
            causal_summary[comparison][horizon] = {
                "mean_global_identity_error_reduction": item.get("mean_global_identity_error_reduction"),
                "mean_target_identity_error_reduction": item.get("mean_target_identity_error_reduction"),
                "mean_target_missing_reduction": item.get("mean_target_missing_reduction"),
                "mean_target_iou_delta": item.get("mean_target_iou_delta"),
                "mean_id_switch_improvement": item.get("mean_id_switch_improvement"),
                "mean_protected_accuracy_delta": item.get("mean_protected_accuracy_delta"),
                "sequence_cluster_ci": {key: ci.get(key) for key in ("lower", "mean", "upper", "clusters", "seed", "repetitions")},
                "counts": item.get("counts", {}),
            }

    gate_status = "FAIL_FUTURE_EFFECT" if not failures else "BLOCKED_INTEGRITY"
    common = {
        "schema_version": "N72R15_FINAL_GATE_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": gate_status,
        "strong_positive": strong_positive,
        "ablations_authorized": False,
        "downstream_training_authorized": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "formal": {
            "events": len(events),
            "independent_sequences": len(sequences),
            "variants": expected_variants,
            "runtime_rows": runtime_rows,
            "event_frame_rows": event_frame_rows,
            "future_rows": future_rows,
            "candidate_uids_checked": candidate_uids_checked,
            "protocol_sha256": formal.get("protocol_sha256"),
            "manifest_sha256": sha256_file(FORMAL),
        },
        "trackeval": {
            "records": trackeval.get("record_count"),
            "expected_records": trackeval.get("expected_record_count"),
            "duplicate_keys": len(trackeval.get("duplicate_keys", [])),
            "missing_keys": len(trackeval.get("missing_keys", [])),
            "commit": trackeval.get("trackeval_commit"),
            "horizons": [20, 50, 100],
        },
        "first_screen_h100": first_screen,
        "causal_metrics": causal_summary,
        "strict_positive_criteria": strict_criteria,
        "audit_statuses": audit_statuses,
        "candidate_coverage_h100": {key: coverage_h100.get(key) for key in ("target_visible_frames", "main_covered_frames", "main_coverage", "target_session_covered_frames", "target_session_coverage", "target_session_only_rescue_frames", "target_session_only_rescue", "neither_source_coverage_failure_frames", "neither_source_failure")},
        "component_auc_h100": component_auc,
        "input_hashes": input_hashes,
        "source_code_hashes": source_hashes,
        "integrity_failures": failures,
        "historical_failure_evidence": [
            "outputs/N72R15/smoke_controller_failure_attempt_01.json",
            "outputs/N72R15/formal_controller_failure_attempt_01.json",
            "outputs/N72R15/formal/formal_manifest.json",
            "outputs/N72R15/formal_attempt_02/formal_controller_failure_attempt_01.json",
            "outputs/N72R15/causal_metrics_failure_attempt_01.json",
            "outputs/N72R15/state_propagation_audit_failure.json",
            "outputs/N72R15/diagnostics_attempt_04/state_edge_component_diagnosis_failure.json",
            "outputs/N72R15/diagnostics_attempt_04/state_edge_component_diagnosis_failure_attempt_02.json",
            "outputs/N72R15/diagnostics_attempt_04/integrity_audits_failure.json",
            "outputs/N72R15/diagnostics_attempt_04/assignment_cardinality_failure_attempt_01.json",
            "outputs/N72R15/diagnostics_attempt_04/assignment_cardinality_failure_attempt_02.json",
            "outputs/N72R15/trackeval_attempt_01/export_failure_attempt_01.json",
            "outputs/N72R15/finalizer_failure_attempt_01.json",
            "outputs/N72R15/finalizer_failure_attempt_02.json",
        ],
    }
    atomic_dump(OUT / "STRONG_POSITIVE_STATUS.json", {
        "schema_version": "N72R15_STRONG_POSITIVE_STATUS_V1",
        "status": "PASS_STRONG_POSITIVE" if strong_positive else "FAIL_STRONG_POSITIVE",
        "strong_positive": strong_positive,
        "gate_status": gate_status,
        "criteria": strict_criteria,
        "ablations_authorized": False,
        "downstream_training_authorized": False,
        "reason": "H100 G2-vs-B0 and H20 target future-effect criteria are not met; no ablation or downstream training is authorized." if not strong_positive else "All preregistered strong-positive criteria passed.",
        "input_hashes": input_hashes,
    })
    atomic_dump(OUT / "n72r15_final_gate.json", common)
    atomic_dump(OUT / "stage_11_status.json", {
        "schema_version": "N72R15_STAGE_11_STATUS_V1",
        "status": gate_status,
        "stage": "N72R15_FINAL_GATE",
        "strong_positive": strong_positive,
        "ablations_authorized": False,
        "downstream_training_authorized": False,
        "report": "docs/N72R15_FINAL_REPORT.md",
        "summary": "Persistent association integrity and TrackEval completed; future-effect gate failed without integrity violations." if not failures else "Final gate blocked by integrity failures; see integrity_failures.",
        "integrity_failures": failures,
        "strict_positive_criteria": strict_criteria,
        "input_hashes": input_hashes,
    })
    print(json.dumps({
        "status": gate_status,
        "strong_positive": strong_positive,
        "integrity_failures": len(failures),
        "events": len(events),
        "independent_sequences": len(sequences),
        "runtime_rows": runtime_rows,
        "trackeval_records": trackeval.get("record_count"),
        "strict_positive_criteria": strict_criteria,
        "output_files": [
            str(OUT / "STRONG_POSITIVE_STATUS.json"),
            str(OUT / "n72r15_final_gate.json"),
            str(OUT / "stage_11_status.json"),
        ],
    }, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
