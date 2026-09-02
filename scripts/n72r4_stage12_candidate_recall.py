#!/usr/bin/env python3
"""Freeze and audit the N72R4 NO versus M0 candidate-recall decomposition.

Stage 10 already computed this posthoc quantity from the two official SAM3
streams.  This small reproducible stage does not rerun SAM3 and does not
change the recall threshold.  It only checks the frozen artifact and records
the recovery decision evidence separately from the identity-effect result.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n72r4_stage10_cpu_analysis import OUT, atomic_json, now_utc, read_json, sha256_file  # noqa: E402


RECALL_PATH = OUT / "candidate_recall" / "no_vs_m0_candidate_recall.json"
STATUS_PATH = OUT / "stage_status" / "stage_12_status.json"
HORIZONS = (20, 50, 100)
BRANCHES = ("B0_NO_INTERVENTION", "B1_CURRENT_FRAME_CORRECTION")
RECOVER_ACTION = "RECOVER_IDENTITY"


def _finite(value: object) -> bool:
    return value is not None and isinstance(value, (int, float)) and value == value and abs(float(value)) != float("inf")


def _check_recall(artifact: dict) -> dict:
    if artifact.get("status") != "PASS_STAGE10_NO_VS_M0_POSTHOC_RECALL":
        raise RuntimeError(f"candidate recall artifact is not a passing Stage10 artifact: {artifact.get('status')}")
    if artifact.get("runtime_future_gt_used") is not False:
        raise RuntimeError("candidate recall artifact violates runtime GT boundary")
    events = artifact.get("events")
    if not isinstance(events, list) or len(events) != 6:
        raise RuntimeError(f"expected six recall events, found {len(events) if isinstance(events, list) else None}")
    if len({str(event.get("event_id")) for event in events}) != len(events):
        raise RuntimeError("candidate recall event IDs are not unique")
    if len({str(event.get("sequence")) for event in events}) != 6:
        raise RuntimeError("candidate recall sequence clusters are not six independent sequences")
    for event in events:
        if event.get("runtime_future_gt_used") is not False or event.get("interaction_source") != "simulated_from_gt":
            raise RuntimeError(f"recall event provenance invalid: {event.get('event_id')}")
        horizons = event.get("horizons")
        if not isinstance(horizons, dict) or set(horizons) != set(BRANCHES):
            raise RuntimeError(f"recall branch set invalid: {event.get('event_id')}")
        for branch in BRANCHES:
            if set(horizons[branch]) != {str(horizon) for horizon in HORIZONS}:
                raise RuntimeError(f"recall horizons invalid: {event.get('event_id')}/{branch}")
            for horizon in HORIZONS:
                value = horizons[branch][str(horizon)]
                evaluated = int(value.get("evaluated_frames", -1))
                hits = int(value.get("candidate_present_frames", -1))
                recall = value.get("candidate_recall")
                if evaluated < 0 or hits < 0 or hits > evaluated:
                    raise RuntimeError(f"recall count invalid: {event.get('event_id')}/{branch}/H{horizon}")
                if evaluated and (not _finite(recall) or not 0.0 <= float(recall) <= 1.0):
                    raise RuntimeError(f"recall value invalid: {event.get('event_id')}/{branch}/H{horizon}")
    aggregate = artifact.get("aggregate")
    if not isinstance(aggregate, dict) or set(aggregate) != set(BRANCHES):
        raise RuntimeError("recall aggregate branch set invalid")
    summary: dict[str, object] = {"aggregate": {}, "recover_action": {}, "delta_m0_minus_no": {}}
    for horizon in HORIZONS:
        values = {}
        for branch in BRANCHES:
            value = aggregate[branch].get(str(horizon))
            if not isinstance(value, dict):
                raise RuntimeError(f"aggregate recall missing: {branch}/H{horizon}")
            evaluated = int(value.get("evaluated_frames", -1))
            hits = int(value.get("candidate_present_frames", -1))
            recall = value.get("candidate_recall")
            if evaluated < 0 or hits < 0 or hits > evaluated or (evaluated and not _finite(recall)):
                raise RuntimeError(f"aggregate recall invalid: {branch}/H{horizon}")
            values[branch] = {
                "evaluated_frames": evaluated,
                "candidate_present_frames": hits,
                "candidate_recall": None if recall is None else float(recall),
                "independent_sequence_count": int(value.get("independent_sequence_count", -1)),
            }
        summary["aggregate"][str(horizon)] = values
        no = values[BRANCHES[0]]["candidate_recall"]
        m0 = values[BRANCHES[1]]["candidate_recall"]
        summary["delta_m0_minus_no"][str(horizon)] = None if no is None or m0 is None else float(m0 - no)
    by_action = artifact.get("by_action")
    if not isinstance(by_action, dict) or RECOVER_ACTION not in by_action:
        raise RuntimeError("recover action recall breakdown is missing")
    for horizon in HORIZONS:
        recover_values = {}
        for branch in BRANCHES:
            value = by_action[RECOVER_ACTION][branch].get(str(horizon))
            if not isinstance(value, dict):
                raise RuntimeError(f"recover action recall missing: {branch}/H{horizon}")
            recover_values[branch] = {
                "evaluated_frames": int(value.get("evaluated_frames", -1)),
                "candidate_present_frames": int(value.get("candidate_present_frames", -1)),
                "candidate_recall": value.get("candidate_recall"),
                "event_count": int(value.get("event_count", -1)),
                "independent_sequence_count": int(value.get("independent_sequence_count", -1)),
            }
        summary["recover_action"][str(horizon)] = recover_values
    return summary


def main() -> int:
    started = now_utc()
    try:
        if STATUS_PATH.exists():
            raise RuntimeError(f"refusing to overwrite existing Stage12 status: {STATUS_PATH}")
        artifact = read_json(RECALL_PATH)
        summary = _check_recall(artifact)
        recover_h20 = summary["recover_action"]["20"]["B1_CURRENT_FRAME_CORRECTION"]
        recover_h50 = summary["recover_action"]["50"]["B1_CURRENT_FRAME_CORRECTION"]
        recover_h100 = summary["recover_action"]["100"]["B1_CURRENT_FRAME_CORRECTION"]
        # The plan does not freeze a numeric recovery threshold.  Therefore
        # this is an evidence classification, not a newly tuned gate: H50 and
        # H100 recovery availability are materially below complete coverage,
        # while the Stage 11 identity effect remains a separate FAIL result.
        recovery_needed = bool(
            float(recover_h50["candidate_recall"]) < 1.0
            or float(recover_h100["candidate_recall"]) < 1.0
        )
        status = {
            "schema_version": "N72R4_STAGE_STATUS_V1",
            "stage": "12_CANDIDATE_RECALL_REEVALUATION",
            "status": "PASS_STAGE12_CANDIDATE_RECALL_REEVALUATION",
            "started_at_utc": started,
            "finished_at_utc": now_utc(),
            "source_recall": str(RECALL_PATH),
            "source_recall_sha256": sha256_file(RECALL_PATH),
            "event_count": 6,
            "independent_sequence_count": 6,
            "thresholds": {"candidate_iou": 0.5, "horizons": list(HORIZONS)},
            "summary": summary,
            "recover_identity_m0_candidate_recall_incomplete_at_long_horizons": recovery_needed,
            "candidate_recovery_decision": "PROCEED_STAGE13_DIAGNOSTIC" if recovery_needed else "NOT_NEEDED",
            "decision_basis": "fixed completeness evidence only; no post-treatment identity metric, threshold scan, or event selection was used",
            "stage11_identity_effect_is_separate": True,
            "runtime_future_gt_used": False,
            "gt_usage": "posthoc_only_in_source_stage10_artifact",
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "scientific_result": "NO_VS_M0_CANDIDATE_AVAILABILITY_DECOMPOSITION_ONLY",
        }
        atomic_json(STATUS_PATH, status)
        print(json.dumps({"status": status["status"], "recovery_decision": status["candidate_recovery_decision"], "status_path": str(STATUS_PATH)}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        failure_root = OUT / "attempts" / "stage12"
        failure_root.mkdir(parents=True, exist_ok=True)
        failure_path = failure_root / f"failure_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        atomic_json(
            failure_path,
            {
                "schema_version": "N72R4_FAILURE_V1",
                "stage": "12_CANDIDATE_RECALL_REEVALUATION",
                "status": "FAIL",
                "started_at_utc": started,
                "finished_at_utc": now_utc(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "runtime_future_gt_used": False,
            },
        )
        print(json.dumps({"status": "FAIL", "failure_artifact": str(failure_path), "error": str(exc)}, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
