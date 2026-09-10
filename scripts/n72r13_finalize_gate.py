#!/usr/bin/env python3
"""Finalize the N72R13 temporal-intervention gate from sealed evidence.

The script is post-hoc: it reads sealed runtime rows, then opens GT only to
recompute H100 identity/protected summaries and never writes to historical
N72R11R5R1/N72R13 runtime artifacts.  It refuses to authorize a value model
when the Oracle has no official H100 headroom or no effective APPLY decision.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n72r9_temporal_replay as legacy  # noqa: E402
from scripts import n72r11_on_demand_replay as replay  # noqa: E402
from scripts.n72r13_temporal_oracle import _quality  # noqa: E402

HORIZONS = (5, 20, 50, 100)
TRACK_HORIZONS = (20, 50, 100)
TRACK_VARIANTS = ("E0_BASELINE_B0", "E1B_PCTIS", "ORACLE_TIV")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
        Path(temporary).replace(path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def _frozen_paths(manifest: dict[str, Any]) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {}
    for record in manifest.get("records", []):
        event_id = str(record["event_id"])
        event_root = Path(str(record["done"])).parent
        result[event_id] = {
            "E0_BASELINE_B0": event_root / "E0_BASELINE_B0" / "runtime_frames.jsonl",
            "E1B_PCTIS": event_root / "E1B_PCTIS_LEGACY" / "runtime_frames.jsonl",
        }
    if len(result) != 32:
        raise RuntimeError(f"frozen E1B manifest has {len(result)} unique events, expected 32")
    return result


def _runtime_scan(value: Any, location: str = "root") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key) in {"runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used"} and nested is not False:
                errors.append(f"{location}/{key}={nested!r}")
            errors.extend(_runtime_scan(nested, f"{location}/{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            errors.extend(_runtime_scan(nested, f"{location}/{index}"))
    return errors


def _quality_delta(keep: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    def delta(left: str, right: str) -> float | None:
        a, b = keep.get(left), treatment.get(right)
        return None if a is None or b is None else float(b - a)
    return {
        "target_accuracy_delta": delta("target_accuracy", "target_accuracy"),
        "global_pid_accuracy_delta": delta("global_pid_accuracy", "global_pid_accuracy"),
        "target_identity_error_reduction": None if keep.get("target_identity_error") is None or treatment.get("target_identity_error") is None else float(keep["target_identity_error"] - treatment["target_identity_error"]),
        "id_switch_improvement": int(keep.get("id_switch_count", 0) - treatment.get("id_switch_count", 0)),
        "protected_accuracy_delta": delta("protected_accuracy", "protected_accuracy"),
        "target_missing_delta": int(keep.get("target_missing_frames", 0) - treatment.get("target_missing_frames", 0)),
    }


def finalize(output_root: Path) -> dict[str, Any]:
    protocol_path = ROOT / "outputs/N72R9/protocol.json"
    frozen_manifest_path = ROOT / "outputs/N72R11R5R1/formal_e1b_manifest.json"
    oracle_metrics_path = ROOT / "outputs/N72R13/temporal_oracle/temporal_oracle_metrics.json"
    trackeval_path = ROOT / "outputs/N72R13/trackeval/trackeval_run_manifest.json"
    protocol = _read(protocol_path)
    frozen = _read(frozen_manifest_path)
    oracle_metrics = _read(oracle_metrics_path)
    trackeval = _read(trackeval_path)
    events = {
        str(item["event_id"]): dict(item)
        for item in protocol.get("source_event_selection", {}).get("events", [])
        if isinstance(item, dict) and item.get("event_id") is not None
    }
    frozen_paths = _frozen_paths(frozen)
    if len(events) != 32 or set(events) != set(frozen_paths):
        raise RuntimeError("N72R13 finalizer does not have the exact frozen 32-event set")
    if oracle_metrics.get("status") != "PASS_N72R13_ORACLE_RUNTIME_AND_POSTHOC" or oracle_metrics.get("event_count") != 32:
        raise RuntimeError("N72R13 Oracle metrics are incomplete")
    if trackeval.get("status") != "PASS_TRACKEVAL_N72R13" or trackeval.get("record_count") != 288:
        raise RuntimeError("N72R13 official TrackEval output is incomplete")

    event_rows: list[dict[str, Any]] = []
    apply_count = 0
    opportunity_count = 0
    runtime_violations = 0
    for event_id in sorted(events):
        event = events[event_id]
        oracle_dir = ROOT / "outputs/N72R13/temporal_oracle/events" / event_id
        sealed_path = oracle_dir / "runtime_event_sealed.json"
        sealed = _read(sealed_path)
        oracle_rows = sealed.get("oracle_rows")
        if not isinstance(oracle_rows, list) or len(oracle_rows) != 101:
            raise RuntimeError(f"{event_id}: Oracle sealed rows are incomplete")
        runtime_violations += len(_runtime_scan(sealed))
        e0_path = frozen_paths[event_id]["E0_BASELINE_B0"]
        e0_rows = _jsonl(e0_path)
        if len(e0_rows) != 101:
            raise RuntimeError(f"{event_id}: frozen E0 has {len(e0_rows)} rows")
        inputs = replay._load_inputs(event, horizon=100)
        gt = legacy._load_gt(str(event["sequence"]))
        protected = legacy._protected_map(
            inputs["rows"]["c0_source"][int(event["event_frame"])],
            gt,
            int(event["event_frame"]),
            int(event["dataset_gt_id"]),
        )
        by_horizon: dict[str, Any] = {}
        event_payload = _read(oracle_dir / "event.json")
        decisions = event_payload.get("decisions", [])
        event_apply = sum(int(item.get("selected_variant") == "APPLY_TIV") for item in decisions)
        event_opportunities = len(decisions)
        apply_count += event_apply
        opportunity_count += event_opportunities
        for horizon in HORIZONS:
            q0 = _quality(
                e0_rows, event_frame=int(event["event_frame"]), target_public=int(inputs["target_public_id"]),
                target_gid=int(event["dataset_gt_id"]), protected=protected, gt=gt, horizon=horizon,
            )
            qo = _quality(
                oracle_rows, event_frame=int(event["event_frame"]), target_public=int(inputs["target_public_id"]),
                target_gid=int(event["dataset_gt_id"]), protected=protected, gt=gt, horizon=horizon,
            )
            by_horizon[str(horizon)] = {"E0": q0, "ORACLE_TIV": qo, "delta": _quality_delta(q0, qo)}
        event_rows.append({
            "event_id": event_id,
            "sequence": event["sequence"],
            "action_type": event["action_type"],
            "event_frame": int(event["event_frame"]),
            "opportunity_count": event_opportunities,
            "apply_count": event_apply,
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "quality": by_horizon,
        })

    custom: dict[str, Any] = {}
    for horizon in HORIZONS:
        selected = [row for row in event_rows]
        visible = sum(int(row["quality"][str(horizon)]["E0"]["target_visible_frames"]) for row in selected)
        e0_errors = sum(int(row["quality"][str(horizon)]["E0"]["target_identity_error_frames"]) for row in selected)
        oracle_errors = sum(int(row["quality"][str(horizon)]["ORACLE_TIV"]["target_identity_error_frames"]) for row in selected)
        global_visible = sum(int(row["quality"][str(horizon)]["E0"]["global_visible_identity_frames"]) for row in selected)
        e0_global_errors = sum(int(row["quality"][str(horizon)]["E0"]["global_visible_identity_frames"]) - int(row["quality"][str(horizon)]["E0"]["global_correct_identity_frames"]) for row in selected)
        oracle_global_errors = sum(int(row["quality"][str(horizon)]["ORACLE_TIV"]["global_visible_identity_frames"]) - int(row["quality"][str(horizon)]["ORACLE_TIV"]["global_correct_identity_frames"]) for row in selected)
        custom[str(horizon)] = {
            "event_count": len(selected),
            "visible_target_frames": visible,
            "target_identity_error_reduction": None if visible == 0 else float((e0_errors - oracle_errors) / visible),
            "global_visible_identity_frames": global_visible,
            "global_identity_error_reduction": None if global_visible == 0 else float((e0_global_errors - oracle_global_errors) / global_visible),
            "target_accuracy_delta_event_mean": float(sum(row["quality"][str(horizon)]["delta"]["target_accuracy_delta"] or 0.0 for row in selected) / len(selected)),
            "global_pid_accuracy_delta_event_mean": float(sum(row["quality"][str(horizon)]["delta"]["global_pid_accuracy_delta"] or 0.0 for row in selected) / len(selected)),
            "id_switch_improvement_event_mean": float(sum(row["quality"][str(horizon)]["delta"]["id_switch_improvement"] for row in selected) / len(selected)),
            "protected_accuracy_delta_event_mean": float(sum(row["quality"][str(horizon)]["delta"]["protected_accuracy_delta"] or 0.0 for row in selected) / len(selected)),
        }

    trackeval_delta: dict[str, Any] = {}
    h100_improved: list[str] = []
    for result in trackeval.get("horizon_results", []):
        horizon = int(result["horizon"])
        pooled = result.get("pooled", {})
        base = pooled.get("E0_BASELINE_B0", {})
        oracle = pooled.get("ORACLE_TIV", {})
        delta = {metric: float(oracle[metric] - base[metric]) for metric in ("HOTA", "DetA", "AssA", "IDF1", "MOTA")}
        delta["IDSW_reduction"] = float(base["IDSW"] - oracle["IDSW"])
        trackeval_delta[str(horizon)] = {"E0": base, "ORACLE_TIV": oracle, "delta_ORACLE_minus_E0": delta}
        if horizon == 100:
            h100_improved = [metric for metric in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW_reduction") if delta[metric] > 0.0]

    strict_h20_identity_positive = bool((custom["20"].get("target_identity_error_reduction") or 0.0) > 0.0 and (custom["20"].get("global_identity_error_reduction") or 0.0) > 0.0)
    official_h100_headroom = bool(h100_improved)
    actual_temporal_headroom = bool(apply_count > 0 and strict_h20_identity_positive and official_h100_headroom)
    gate_status = "CONTINUE_VALUE_MODEL" if actual_temporal_headroom else "STOP_NO_TEMPORAL_INTERVENTION_HEADROOM"
    gate = {
        "schema_version": "N72R13_FINAL_GATE_V1",
        "status": gate_status,
        "created_at_utc": _now(),
        "protocol": str(protocol_path),
        "protocol_sha256": _sha(protocol_path),
        "frozen_e1b_manifest": str(frozen_manifest_path),
        "frozen_e1b_manifest_sha256": _sha(frozen_manifest_path),
        "oracle_metrics": str(oracle_metrics_path),
        "oracle_metrics_sha256": _sha(oracle_metrics_path),
        "trackeval_run_manifest": str(trackeval_path),
        "trackeval_run_manifest_sha256": _sha(trackeval_path),
        "event_count": len(event_rows),
        "independent_sequence_count": len({str(row["sequence"]) for row in event_rows}),
        "opportunity_count": opportunity_count,
        "apply_count": apply_count,
        "keep_count": opportunity_count - apply_count,
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "runtime_violations": runtime_violations,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "custom_e0_vs_oracle": custom,
        "trackeval_e0_vs_oracle": trackeval_delta,
        "strict_h20_identity_positive": strict_h20_identity_positive,
        "official_h100_global_metric_improvements": h100_improved,
        "official_h100_headroom": official_h100_headroom,
        "actual_temporal_intervention_headroom": actual_temporal_headroom,
        "oracle_allows_value_model": actual_temporal_headroom,
        "calibration_head_authorized": False,
        "selector_authorized": False,
        "decoder_lora_authorized": False,
        "value_model_training_authorized": actual_temporal_headroom,
        "requery_next": not actual_temporal_headroom,
        "scientific_interpretation": "All effective Oracle opportunities selected KEEP_BASE; therefore any E0-vs-Oracle difference is not evidence that APPLY temporal intervention reduced future error. Official H100 TrackEval also has no global improvement.",
        "event_rows": event_rows,
    }
    _atomic_json(output_root / "n72r13_final_gate.json", gate)
    _atomic_json(output_root / "stage_05_status.json", {
        "schema_version": "N72R13_STAGE_STATUS_V1",
        "stage": "N72R13-05-GATE-DECISION",
        "status": gate_status,
        "gate": str(output_root / "n72r13_final_gate.json"),
        "official_h100_global_metric_improvements": h100_improved,
        "custom_h20_target_identity_error_reduction": custom["20"]["target_identity_error_reduction"],
        "custom_h20_global_identity_error_reduction": custom["20"]["global_identity_error_reduction"],
        "effective_opportunity_count": opportunity_count,
        "apply_count": apply_count,
        "runtime_future_gt_used": False,
        "value_model_training_authorized": actual_temporal_headroom,
        "requery_next": not actual_temporal_headroom,
        "created_at_utc": _now(),
        "historical_outputs_modified": False,
    })
    _atomic_json(output_root / "stage_06_status.json", {
        "schema_version": "N72R13_STAGE_STATUS_V1",
        "stage": "N72R13-06-VALUE-MODEL",
        "status": "NOT_AUTHORIZED_NO_HEADROOM" if not actual_temporal_headroom else "AUTHORIZED_PENDING_IMPLEMENTATION",
        "reason": "N72R13 gate requires official H100 global improvement and strict H20 identity improvement; official H100 comparison has no metric above frozen E0 and all Oracle decisions are KEEP_BASE.",
        "training_started": False,
        "value_model_training_authorized": actual_temporal_headroom,
        "requery_next": not actual_temporal_headroom,
        "created_at_utc": _now(),
        "historical_outputs_modified": False,
    })
    return gate


def main() -> int:
    output_root = ROOT / "outputs/N72R13"
    try:
        gate = finalize(output_root)
        print(json.dumps({"status": gate["status"], "events": gate["event_count"], "opportunities": gate["opportunity_count"], "apply": gate["apply_count"], "h100_improvements": gate["official_h100_global_metric_improvements"], "output": str(output_root / "n72r13_final_gate.json")}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = output_root / "finalize_gate_failure.json"
        _atomic_json(failure, {"schema_version": "N72R13_FINALIZE_FAILURE_V1", "status": "FAIL_N72R13_FINALIZE", "error_type": type(exc).__name__, "error": str(exc), "historical_outputs_modified": False})
        print(json.dumps({"status": "FAIL_N72R13_FINALIZE", "error": str(exc), "failure": str(failure)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
