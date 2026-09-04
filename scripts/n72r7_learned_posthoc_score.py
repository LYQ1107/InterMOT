#!/usr/bin/env python3
"""Posthoc-score the sealed N72R7 learned D1/D2 replay.

The scorer reuses the frozen N72R7 metric implementation.  It only redirects
the two runtime roots and writes a versioned learned-result directory; GT is
still opened after runtime validation and artifact sealing.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n72r7_posthoc_score as scorer  # noqa: E402


def resolve_root_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


LEARNED_ROOT = resolve_root_path(
    os.environ.get("N72R7_LEARNED_POSTHOC_ROOT"),
    ROOT / "outputs/N72R7/learned_posthoc_attempt1",
)
D1_ROOT = resolve_root_path(
    os.environ.get("N72R7_LEARNED_D1_ROOT"),
    ROOT / "outputs/N72R7/learned_replay/d1_v1_attempt1",
)
D2_ROOT = resolve_root_path(
    os.environ.get("N72R7_LEARNED_D2_ROOT"),
    ROOT / "outputs/N72R7/learned_replay/d2_v1_attempt1",
)
CHECKPOINT = resolve_root_path(
    os.environ.get("N72R7_LEARNED_CHECKPOINT"),
    ROOT / "outputs/N72R7/training/HumanConditionedTargetIDDecoder_v1.pt",
)
REPLAY_PROTOCOL = resolve_root_path(
    os.environ.get("N72R7_LEARNED_REPLAY_PROTOCOL"),
    ROOT / "outputs/N72R7/training/learned_replay_protocol.json",
)
STAGE = resolve_root_path(
    os.environ.get("N72R7_LEARNED_POSTHOC_STATUS"),
    ROOT / "outputs/N72R7/stage_07_learned_posthoc_status.json",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    try:
        LEARNED_ROOT.mkdir(parents=True, exist_ok=True)
        scorer.D1_ROOT = D1_ROOT
        scorer.D2_ROOT = D2_ROOT
        scorer.POSTHOC_ROOT = LEARNED_ROOT
        scorer.RUNTIME_VALIDATION_PATH = LEARNED_ROOT / "runtime_validation.json"
        scorer.RESULT_PATH = LEARNED_ROOT / "n72r7_learned_d1_d2_posthoc_results.json"
        scorer.EVENT_METRICS_PATH = LEARNED_ROOT / "event_metrics.jsonl"
        scenarios = scorer._load_runtime_scenarios()
        runtime = scorer.validate_runtime(scenarios)
        result = scorer.posthoc_score(runtime)
        d1 = result["aggregate"]["D1_vs_D0"]["20"]
        d2 = result["aggregate"]["D2_vs_D0"]["20"]
        stage = {
            "schema_version": "N72R7_STAGE_07_LEARNED_POSTHOC_STATUS_V1",
            "status": "PASS_LEARNED_EXECUTION_FAIL_FUTURE_EFFECT" if result["gate"]["research_gate"] != "PASS_GT_SIMULATED_CLOSED_LOOP_REACQUISITION_CONFIRMED" else "PASS_LEARNED_DEVELOPMENT_EFFECT",
            "created_at_utc": now_utc(),
            "runtime_validation": str(scorer.RUNTIME_VALIDATION_PATH),
            "result_artifact": str(scorer.RESULT_PATH),
            "event_metrics": str(scorer.EVENT_METRICS_PATH),
            "event_count": result["event_count"],
            "independent_sequence_count": result["independent_sequence_count"],
            "learned_checkpoint": str(CHECKPOINT),
            "learned_checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
            "learned_replay_protocol": str(REPLAY_PROTOCOL),
            "learned_replay_protocol_sha256": hashlib.sha256(REPLAY_PROTOCOL.read_bytes()).hexdigest(),
            "research_gate": result["gate"]["research_gate"],
            "D1_vs_D0_H20": {"identity_error_reduction": d1.get("identity_error_reduction"), "ci": d1.get("sequence_cluster_bootstrap_95ci"), "correct_crossings": d1.get("true_correct_crossing_count"), "incorrect_crossings": d1.get("true_incorrect_crossing_count"), "protected_regression": d1.get("protected_regression_count")},
            "D2_vs_D0_H20": {"identity_error_reduction": d2.get("identity_error_reduction"), "ci": d2.get("sequence_cluster_bootstrap_95ci"), "correct_crossings": d2.get("true_correct_crossing_count"), "incorrect_crossings": d2.get("true_incorrect_crossing_count"), "protected_regression": d2.get("protected_regression_count")},
            "training_split_disjoint_from_validation": True,
            "confirmation_run": False,
            "production_authorized": False,
            "runtime_future_gt_used": False,
            "posthoc_gt_used": True,
            "interaction_source": "simulated_from_gt",
            "real_human_evidence": False,
        }
        atomic_json(STAGE, stage)
        print(json.dumps({"status": stage["status"], "research_gate": stage["research_gate"], "result": str(scorer.RESULT_PATH)}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = ROOT / "outputs/N72R7/attempts" / f"n72r7_learned_posthoc_failure_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        atomic_json(failure, {"schema_version": "N72R7_LEARNED_POSTHOC_FAILURE_V1", "status": "FAIL_PRESERVED", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "runtime_future_gt_used": False, "posthoc_gt_used": False, "created_at_utc": now_utc()})
        print(json.dumps({"status": "FAIL_LEARNED_POSTHOC", "failure": str(failure)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
