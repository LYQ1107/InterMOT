#!/usr/bin/env python3
"""Build the reproducible N72R14 gate from the corrected attempt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
PINNED_TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"
VARIANTS = (
    "E0_BASELINE_B0",
    "E1F_TARGET_POOL_ONLY",
    "E1G_TARGET_STATE_ONLY",
    "E1H_PERSISTENT_GLOBAL_STATE",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            import os

            os.fsync(handle.fileno())
        import os

        os.replace(temporary, path)
    finally:
        import os

        if os.path.exists(temporary):
            os.unlink(temporary)


def _runtime_flags(value: Any, location: str = "root") -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in {"runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used"} and nested is not False:
                violations.append(f"{location}/{key}={nested!r}")
            violations.extend(_runtime_flags(nested, f"{location}/{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            violations.extend(_runtime_flags(nested, f"{location}/{index}"))
    return violations


def _formal_summary(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    failures: list[str] = []
    if manifest.get("status") != "PASS_N72R14_FORMAL_REPLAY":
        failures.append(f"formal_status={manifest.get('status')}")
    if int(manifest.get("event_count", -1)) != 32 or int(manifest.get("completed_event_count", -1)) != 32:
        failures.append("formal_event_count_not_32")
    events = manifest.get("events", [])
    event_ids = [str(item.get("event_id")) for item in events if isinstance(item, Mapping)]
    if len(event_ids) != 32 or len(set(event_ids)) != 32:
        failures.append("formal_event_keys_not_unique")
    variant_count = 0
    runtime_rows = 0
    invalid_geometry_assigned = 0
    runtime_flag_violations: list[str] = []
    for event in events:
        if not isinstance(event, Mapping) or event.get("status") != "PASS_N72R14_FORMAL_EVENT":
            failures.append(f"event_not_pass:{event.get('event_id') if isinstance(event, Mapping) else event!r}")
            continue
        variants = event.get("variants", [])
        if {str(item.get("variant")) for item in variants if isinstance(item, Mapping)} != set(VARIANTS):
            failures.append(f"variant_set:{event.get('event_id')}")
        for variant in variants:
            if not isinstance(variant, Mapping):
                failures.append("non_object_variant")
                continue
            variant_count += 1
            frames = Path(str(variant.get("frames")))
            if not frames.is_file():
                failures.append(f"missing_frames:{event.get('event_id')}/{variant.get('variant')}")
                continue
            rows = [json.loads(line) for line in frames.read_text(encoding="utf-8").splitlines() if line.strip()]
            runtime_rows += len(rows)
            if len(rows) != 101:
                failures.append(f"frame_count:{event.get('event_id')}/{variant.get('variant')}={len(rows)}")
            runtime_flag_violations.extend(
                _runtime_flags(rows, f"{event.get('event_id')}/{variant.get('variant')}")
            )
            for row in rows[1:]:
                for candidate in row.get("candidate_rows", []):
                    if candidate.get("solver_status") == "ASSIGNED_TO_PUBLIC_ID":
                        box = candidate.get("box_xyxy")
                        if not isinstance(box, list) or len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                            invalid_geometry_assigned += 1
    if variant_count != 32 * len(VARIANTS):
        failures.append(f"formal_variant_count={variant_count}")
    if runtime_rows != 32 * len(VARIANTS) * 101:
        failures.append(f"formal_runtime_rows={runtime_rows}")
    if runtime_flag_violations:
        failures.append("formal_runtime_gt_flag_violation")
    if invalid_geometry_assigned:
        failures.append(f"formal_invalid_assigned_geometry={invalid_geometry_assigned}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "event_count": len(event_ids),
        "variant_artifact_count": variant_count,
        "runtime_row_count": runtime_rows,
        "invalid_assigned_geometry_count": invalid_geometry_assigned,
        "runtime_flag_violation_count": len(runtime_flag_violations),
        "failures": failures,
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
    }


def _trackeval_summary(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    failures: list[str] = []
    if manifest.get("status") != "PASS_TRACKEVAL_N72R14":
        failures.append(f"trackeval_status={manifest.get('status')}")
    if int(manifest.get("record_count", -1)) != 384 or int(manifest.get("expected_record_count", -1)) != 384:
        failures.append("trackeval_record_count_not_384")
    if manifest.get("trackeval_commit") != PINNED_TRACKEVAL_COMMIT:
        failures.append("trackeval_commit_mismatch")
    if manifest.get("runtime_future_gt_used") is not False:
        failures.append("trackeval_runtime_future_gt")
    pooled: dict[str, dict[str, float]] = {}
    for result in manifest.get("horizon_results", []):
        if int(result.get("horizon", -1)) == 100:
            pooled = {str(name): {str(key): float(value) for key, value in metrics.items()} for name, metrics in result.get("pooled", {}).items()}
    if set(pooled) != set(VARIANTS):
        failures.append("trackeval_h100_variant_pool_incomplete")
    return {
        "status": "PASS" if not failures else "FAIL",
        "record_count": int(manifest.get("record_count", -1)),
        "expected_record_count": int(manifest.get("expected_record_count", -1)),
        "duplicate_key_count": len(manifest.get("duplicate_keys", [])),
        "trackeval_commit": manifest.get("trackeval_commit"),
        "h100_pooled": pooled,
        "failures": failures,
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
    }


def finalize(*, attempt_root: Path, output_root: Path) -> dict[str, Any]:
    formal_path = attempt_root / "formal_manifest.json"
    metrics_path = attempt_root / "causal_metrics.json"
    propagation_path = attempt_root / "state_causal_propagation_audit.json"
    trackeval_path = attempt_root / "trackeval" / "trackeval_run_manifest.json"
    interface_path = output_root / "interface_audit.json"
    formal = _formal_summary(formal_path)
    metrics = read_json(metrics_path)
    propagation = read_json(propagation_path)
    trackeval = _trackeval_summary(trackeval_path)
    interface = read_json(interface_path)
    structural_failures: list[str] = []
    if interface.get("status") != "PASS_N72R14_INTERFACE_AUDIT":
        structural_failures.append("interface_audit_not_pass")
    if int(interface.get("event_count", -1)) != 32 or int(interface.get("failure_count", -1)) != 0:
        structural_failures.append("interface_event_audit_incomplete")
    if metrics.get("status") != "PASS_N72R14_POSTHOC_METRICS" or int(metrics.get("event_count", -1)) != 32 or int(metrics.get("independent_sequence_count", -1)) != 18:
        structural_failures.append("posthoc_metrics_incomplete")
    if propagation.get("status") != "PASS_N72R14_STATE_CAUSAL_PROPAGATION":
        structural_failures.append("state_propagation_not_pass")
    structural_failures.extend(formal["failures"])
    structural_failures.extend(trackeval["failures"])

    causal = metrics["comparisons"]
    h20_reduction = float(causal["G1_MINUS_B0"]["20"]["mean_global_identity_error_reduction"])
    h100_b0 = trackeval["h100_pooled"]["E0_BASELINE_B0"]
    h100_g1 = trackeval["h100_pooled"]["E1H_PERSISTENT_GLOBAL_STATE"]
    trackeval_delta = {
        key: float(h100_g1[key] - h100_b0[key])
        for key in ("HOTA", "AssA", "IDF1", "IDSW", "DetA", "MOTA")
    }
    strong_checks = {
        "h100_delta_hota_at_least_0.0030": trackeval_delta["HOTA"] >= 0.0030,
        "h100_delta_assa_at_least_0.0050": trackeval_delta["AssA"] >= 0.0050,
        "h100_delta_idf1_at_least_0.0050": trackeval_delta["IDF1"] >= 0.0050,
        "h100_idsw_not_increased": h100_g1["IDSW"] <= h100_b0["IDSW"],
        "h20_identity_error_reduction_positive": h20_reduction > 0.0,
    }
    strong_positive = all(strong_checks.values())
    g1_minus_p0 = {
        str(horizon): float(causal["G1_MINUS_P0"][str(horizon)]["mean_global_identity_error_reduction"])
        for horizon in (20, 50, 100)
    }
    persistent_edge_status = (
        "PERSISTENT_STATE_EDGE_NOT_YET_EFFECTIVE"
        if all(value <= 0.0 for value in g1_minus_p0.values())
        else "PERSISTENT_STATE_EDGE_HAS_POSITIVE_DESCRIPTIVE_DELTA"
    )
    gate = {
        "schema_version": "N72R14_FINAL_GATE_V1",
        "status": persistent_edge_status if not strong_positive else "PASS_STRONG_POSITIVE_PSCA",
        "research_gate": "PASS_STRONG_POSITIVE_PSCA" if strong_positive else "FAIL_FUTURE_EFFECT",
        "disposition": "RETAIN_ISOLATED_RESEARCH_ARTIFACT_ONLY" if not strong_positive else "REQUIRES_SEPARATE_PRODUCTION_REVIEW",
        "created_at_utc": now_utc(),
        "attempt": "attempt_02_positive_geometry",
        "inputs": {
            "formal_manifest": str(formal_path),
            "causal_metrics": str(metrics_path),
            "state_propagation_audit": str(propagation_path),
            "trackeval_run_manifest": str(trackeval_path),
            "interface_audit": str(interface_path),
        },
        "structural_gate": {
            "status": "PASS" if not structural_failures else "FAIL",
            "failures": structural_failures,
            "interface_audit": interface["status"],
            "formal_replay": formal["status"],
            "posthoc_metrics": metrics.get("status"),
            "state_causal_propagation": propagation.get("status"),
            "trackeval": trackeval["status"],
            "event_count": formal["event_count"],
            "independent_sequence_count": int(metrics.get("independent_sequence_count", -1)),
            "runtime_artifact_count": formal["variant_artifact_count"],
            "runtime_row_count": formal["runtime_row_count"],
            "trackeval_record_count": trackeval["record_count"],
            "duplicate_key_count": trackeval["duplicate_key_count"],
            "runtime_future_gt_used": False,
        },
        "persistent_edge_diagnosis": {
            "status": persistent_edge_status,
            "g1_minus_p0_mean_global_identity_error_reduction": g1_minus_p0,
            "g1_minus_b0_h20_mean_global_identity_error_reduction": h20_reduction,
            "g1_minus_b0_h100_mean_global_identity_error_reduction": float(causal["G1_MINUS_B0"]["100"]["mean_global_identity_error_reduction"]),
            "g1_minus_p0_h100_sequence_cluster_ci": causal["G1_MINUS_P0"]["100"]["sequence_cluster_bootstrap_95ci"],
            "g1_minus_b0_h100_sequence_cluster_ci": causal["G1_MINUS_B0"]["100"]["sequence_cluster_bootstrap_95ci"],
        },
        "trackeval_h100": {
            "evaluation_scope": "INTERACTION_WINDOW_DIAGNOSTIC",
            "official_dancetrack_benchmark_score": False,
            "pooled": trackeval["h100_pooled"],
            "g1_minus_b0": trackeval_delta,
        },
        "strong_positive_psca": {
            "status": "PASS" if strong_positive else "FAIL",
            "checks": strong_checks,
            "h100_g1_minus_b0": trackeval_delta,
            "h20_identity_error_reduction": h20_reduction,
        },
        "authorization": {
            "calibration_head": False,
            "selector": False,
            "decoder_lora": False,
            "human_count_ablation": False,
            "reason": "strong_positive_gate_false" if not strong_positive else "requires_separate_gate",
        },
        "failure_history": {
            "attempt_01_trackeval_export": str(output_root / "trackeval" / "export_failure_attempt_01.json"),
            "attempt_01_stage_status": str(output_root / "stage_05_attempt_01_failure.json"),
            "attempt_01_root_cause": "N72R14 runner retained degenerate assigned boxes because require_positive_geometry was false; corrected runner applies the existing pre-solver positive-geometry policy.",
            "attempt_02_repair": "require_positive_geometry=true for event-frame initialization and every B0/P0/T1/G1 pool; reran smoke, formal replay, export, TrackEval and posthoc metrics in an isolated attempt directory.",
        },
        "ablation_status": "NOT_RUN_STRONG_POSITIVE_FALSE" if not strong_positive else "AUTHORIZED_ONLY_AFTER_SEPARATE_PROTOCOL",
        "historical_outputs_modified": False,
        "interaction_source": "simulated_from_gt",
        "real_human_evidence_count": 0,
    }
    strong_path = output_root / "STRONG_POSITIVE_STATUS.json"
    gate_path = output_root / "n72r14_final_gate.json"
    atomic_json(strong_path, gate["strong_positive_psca"])
    atomic_json(gate_path, gate)
    stage04 = {
        "schema_version": "N72R14_STAGE_STATUS_V1",
        "stage": "N72R14-04-POSTHOC-AND-PROPAGATION",
        "status": "PASS_N72R14_POSTHOC_AND_PROPAGATION" if not structural_failures else "FAIL_N72R14_POSTHOC_AND_PROPAGATION",
        "causal_metrics": str(metrics_path),
        "state_propagation_audit": str(propagation_path),
        "event_count": 32,
        "independent_sequence_count": 18,
        "runtime_future_gt_used": False,
        "historical_outputs_modified": False,
        "created_at_utc": now_utc(),
    }
    atomic_json(output_root / "stage_04_status.json", stage04)
    stage06 = {
        "schema_version": "N72R14_STAGE_STATUS_V1",
        "stage": "N72R14-06-OFFICIAL-TRACKEVAL",
        "status": "PASS_TRACKEVAL_N72R14" if trackeval["status"] == "PASS" else "BLOCKED_INCOMPLETE_TRACKEVAL_N72R14",
        "manifest": str(trackeval_path),
        "record_count": trackeval["record_count"],
        "expected_record_count": trackeval["expected_record_count"],
        "trackeval_commit": trackeval["trackeval_commit"],
        "runtime_future_gt_used": False,
        "historical_outputs_modified": False,
        "created_at_utc": now_utc(),
    }
    atomic_json(output_root / "stage_06_status.json", stage06)
    stage07 = {
        "schema_version": "N72R14_STAGE_STATUS_V1",
        "stage": "N72R14-07-FINAL-GATE",
        "status": gate["status"],
        "research_gate": gate["research_gate"],
        "strong_positive_status": gate["strong_positive_psca"]["status"],
        "gate": str(gate_path),
        "ablation_status": gate["ablation_status"],
        "downstream_training_authorized": False,
        "runtime_future_gt_used": False,
        "historical_outputs_modified": False,
        "created_at_utc": now_utc(),
    }
    atomic_json(output_root / "stage_07_status.json", stage07)
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-root", type=Path, default=ROOT / "outputs/N72R14/attempt_02")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/N72R14")
    args = parser.parse_args()
    try:
        gate = finalize(attempt_root=args.attempt_root.resolve(), output_root=args.output_root.resolve())
        print(json.dumps({
            "status": gate["status"],
            "research_gate": gate["research_gate"],
            "strong_positive": gate["strong_positive_psca"]["status"],
            "output": str(args.output_root.resolve() / "n72r14_final_gate.json"),
        }, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "N72R14_FAILURE_V1",
            "status": "FAIL_N72R14_FINALIZER",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "created_at_utc": now_utc(),
            "historical_outputs_modified": False,
        }
        atomic_json(args.output_root.resolve() / "finalizer_failure.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
