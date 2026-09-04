#!/usr/bin/env python3
"""Posthoc-score beam versus greedy on one frozen D1 or D2 candidate source.

The metric implementation is the frozen N72R7 scorer.  This adapter changes
only which already-sealed runtime roots are assigned to its baseline and
treatment slots; it performs no runtime selection and opens GT only inside
the scorer after the sidecar/runtime audit has passed.
"""

from __future__ import annotations

import argparse
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


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("D1", "D2"), required=True)
    parser.add_argument("--greedy-root", required=True)
    parser.add_argument("--beam-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--replay-protocol", required=True)
    parser.add_argument("--mechanism", choices=("beam", "concept"), default="beam")
    args = parser.parse_args()
    output_root = resolve(args.output_root)
    greedy_root = resolve(args.greedy_root)
    beam_root = resolve(args.beam_root)
    checkpoint = resolve(args.checkpoint)
    replay_protocol = resolve(args.replay_protocol)
    try:
        scenarios = scorer._load_runtime_scenarios()
        original_validate = scorer._validate_d1_d2_batch
        # Validate both roots under their actual manifest variant.  The
        # scorer's normal D1/D2 slots are comparison labels, not source labels.
        greedy_rows = original_validate(greedy_root, args.variant, scenarios)
        beam_rows = original_validate(beam_root, args.variant, scenarios)
        for event_id in sorted(scenarios):
            if len(greedy_rows[event_id]) != len(beam_rows[event_id]):
                raise RuntimeError(f"beam/greedy frame count mismatch: {event_id}")
            for greedy, beam in zip(greedy_rows[event_id], beam_rows[event_id]):
                if [str(item["candidate_uid"]) for item in greedy.get("candidate_rows", [])] != [str(item["candidate_uid"]) for item in beam.get("candidate_rows", [])]:
                    raise RuntimeError(f"beam changed the frozen candidate stream: {event_id}/{greedy.get('frame')}")
        runtime_audit = {
            "schema_version": "N72R7_BEAM_GREEDY_RUNTIME_VALIDATION_V1",
            "status": "PASS_N72R7_BEAM_GREEDY_RUNTIME_VALIDATION",
            "created_at_utc": now_utc(),
            "source_variant": args.variant,
            "greedy_root": str(greedy_root),
            "beam_root": str(beam_root),
            "event_count": len(scenarios),
            "independent_sequence_count": len({item["sequence"] for item in scenarios.values()}),
            "candidate_stream_equal": True,
            "exact_global_solver": True,
            "event_frame_memory_read": False,
            "first_memory_visible_frame": "event_frame+1",
            "public_id_inference": False,
            "runtime_future_gt_used": False,
            "gt_loaded_in_worker": False,
            "posthoc_gt_not_loaded_during_validation": True,
            "interaction_source": "simulated_from_gt",
            "real_human_evidence": False,
        }
        output_root.mkdir(parents=True, exist_ok=True)
        scorer.D1_ROOT = greedy_root
        scorer.D2_ROOT = beam_root
        scorer.POSTHOC_ROOT = output_root
        scorer.RUNTIME_VALIDATION_PATH = output_root / "runtime_validation.json"
        scorer.RESULT_PATH = output_root / "n72r7_beam_greedy_posthoc_results.json"
        scorer.EVENT_METRICS_PATH = output_root / "event_metrics.jsonl"
        atomic_json(scorer.RUNTIME_VALIDATION_PATH, runtime_audit)
        runtime = {
            "audit": runtime_audit,
            "rows": {
                "D0": {key: value["c0_rows"] for key, value in scenarios.items()},
                "D1": greedy_rows,
                "D2": beam_rows,
            },
        }
        result = scorer.posthoc_score(runtime)
        beam_greedy = result["aggregate"]["D2_vs_D1"]
        stage = {
            "schema_version": "N72R7_STAGE_SELECTOR_POSTHOC_STATUS_V1",
            "mechanism": args.mechanism,
            "status": "PASS_BEAM_DEVELOPMENT_EFFECT" if result["gate"]["research_gate"] == "PASS_GT_SIMULATED_CLOSED_LOOP_REACQUISITION_CONFIRMED" else "PASS_BEAM_EXECUTION_FAIL_FUTURE_EFFECT",
            "created_at_utc": now_utc(),
            "source_variant": args.variant,
            "greedy_root": str(greedy_root),
            "beam_root": str(beam_root),
            "runtime_validation": str(scorer.RUNTIME_VALIDATION_PATH),
            "result_artifact": str(scorer.RESULT_PATH),
            "event_metrics": str(scorer.EVENT_METRICS_PATH),
            "event_count": result["event_count"],
            "independent_sequence_count": result["independent_sequence_count"],
            "beam_size": 3,
            "beam_expansion_k": 3,
            "beam_diversity_box_iou": 0.70,
            "greedy_vs_d0_h20": result["aggregate"]["D1_vs_D0"]["20"],
            "beam_vs_d0_h20": result["aggregate"]["D2_vs_D0"]["20"],
            "beam_minus_greedy": {
                str(horizon): {
                    "identity_error_reduction": beam_greedy[str(horizon)]["identity_error_reduction"],
                    "ci": beam_greedy[str(horizon)]["sequence_cluster_bootstrap_95ci"],
                    "correct_crossings": beam_greedy[str(horizon)]["true_correct_crossing_count"],
                    "incorrect_crossings": beam_greedy[str(horizon)]["true_incorrect_crossing_count"],
                    "protected_regression": beam_greedy[str(horizon)]["protected_regression_count"],
                    "assignment_change_count": beam_greedy[str(horizon)]["assignment_change_count"],
                }
                for horizon in (20, 50, 100)
            },
            "by_action_beam_minus_greedy_h20": {
                action: values["D2_vs_D1"]["20"]
                for action, values in sorted(result["action_aggregate"].items())
            },
            "research_gate": result["gate"]["research_gate"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "replay_protocol": str(replay_protocol),
            "replay_protocol_sha256": hashlib.sha256(replay_protocol.read_bytes()).hexdigest(),
            "runtime_future_gt_used": False,
            "posthoc_gt_used": True,
            "interaction_source": "simulated_from_gt",
            "real_human_evidence": False,
            "production_authorized": False,
        }
        atomic_json(output_root / f"stage_r3_{args.mechanism}_posthoc_status.json", stage)
        print(json.dumps({"status": stage["status"], "source_variant": args.variant, "research_gate": stage["research_gate"], "beam_minus_greedy_h20": stage["beam_minus_greedy"]["20"]}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = ROOT / "outputs/N72R7/attempts" / f"n72r7_beam_posthoc_failure_{datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        atomic_json(
            failure,
            {
                "schema_version": "N72R7_BEAM_POSTHOC_FAILURE_V1",
                "status": "FAIL_PRESERVED",
                "source_variant": args.variant,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "runtime_future_gt_used": False,
                "posthoc_gt_used": False,
                "created_at_utc": now_utc(),
            },
        )
        print(json.dumps({"status": "FAIL_BEAM_POSTHOC", "failure": str(failure)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
