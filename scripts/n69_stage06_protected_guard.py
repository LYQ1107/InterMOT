"""N69 Stage 06: isolated protected-identity association guard.

The first N69 scorer changed the target column on every frame, but every
observed target crossing also changed an untouched public ID.  This isolated
alternative keeps the trained scorer and frozen Hungarian solver, then
accepts its proposed assignment only when the assignment of every non-target
public ID is unchanged.  The guard is runtime-causal: it uses only the
operator-supplied target public ID, the current candidate stream and the
proposed assignment.  GT is used only by ``score`` after all runtime artifacts
are complete.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n69_stage03_target_conditioned as parent  # noqa: E402


ALT_ROOT = parent.OUT / "stage_06_protected_guard"
ALT_ARTIFACT_DIR = ALT_ROOT / "event_artifacts"
ALT_RUNTIME_STATUS = ALT_ROOT / "runtime_status.json"
ALT_RESULTS = ALT_ROOT / "paired_replay_results.json"
ALT_DIAGNOSTICS = ALT_ROOT / "assignment_diagnostics.jsonl"
ALT_POSTHOC_STATUS = ALT_ROOT / "posthoc_score_status.json"
STAGE06 = parent.OUT / "stage_06_status.json"
ATTEMPTS = parent.ATTEMPTS
METHOD = "N69_PROTECTED_UNTOUCHED_GUARD"


def atomic_json(path: Path, payload: Any) -> None:
    parent.atomic_json(path, payload)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    parent.atomic_jsonl(path, rows)


def assignment_by_public(branch: dict[str, Any]) -> dict[int, int | None]:
    rows = branch.get("candidate_rows", [])
    assignments = parent.assignment_public_ids(branch)
    if len(rows) != len(assignments):
        raise RuntimeError("N69 Stage06 candidate/assignment length mismatch")
    result: dict[int, int | None] = {}
    native_seen: set[int] = set()
    for index, row in enumerate(rows):
        native = int(row["native_tid"])
        if native in native_seen:
            raise RuntimeError(f"N69 Stage06 duplicate native candidate {native}")
        native_seen.add(native)
        result.setdefault(assignments[index], None)
        public_id = assignments[index]
        if public_id is not None:
            if public_id in result and result[public_id] is not None:
                raise RuntimeError(f"N69 Stage06 duplicate assignment public ID {public_id}")
            result[public_id] = native
    return result


def runtime_event(event: dict[str, Any]) -> dict[str, Any]:
    # Keep only intervention/runtime fields.  In particular, do not pass the
    # offline target native ID into feature construction or model inference.
    value = dict(event)
    value.pop("target_native_id", None)
    for key in ("gt_box", "future_gt", "future_labels", "gt_track_ids"):
        value.pop(key, None)
    return value


def replay(device_name: str = "cpu") -> dict[str, Any]:
    parent.ensure_model_protocol()
    events = parent.load_event_map()
    model, mean, std, device = parent.load_trained_model(device_name)
    ALT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    completed = 0
    runtime_frames = 0
    guard_rejections = 0
    for event_id in sorted(events):
        event = events[event_id]
        event_runtime = runtime_event(event)
        source_path = parent.N54_RUNTIME / f"{event_id}.json"
        if not source_path.is_file():
            raise RuntimeError(f"N69 Stage06 frozen runtime missing {source_path}")
        with source_path.open("r", encoding="utf-8") as handle:
            source = json.load(handle)
        artifact = {
            "schema": "N69_PROTECTED_UNTOUCHED_GUARD_RUNTIME_EVENT_V1",
            "status": "PASS",
            "created_at_utc": parent.now(),
            "event_id": event_id,
            "sequence": event["sequence"],
            "action_type": event["action_type"],
            "event_frame": event["event_frame"],
            "first_event_memory_visible_frame": event["event_frame"] + 1,
            "target_public_id_event_input": event["target_public_id"],
            "target_native_id_sent_to_runtime": False,
            "checkpoint": str(parent.CHECKPOINT),
            "checkpoint_sha256": parent.sha256_file(parent.CHECKPOINT),
            "runtime_boundary": {
                "event_frame_memory_read": False,
                "first_memory_visible_frame": event["event_frame"] + 1,
                "gt_loaded_in_worker": False,
                "future_gt_fields_sent": [],
                "target_native_id_sent_to_runtime": False,
                "runtime_future_gt_used": False,
            },
            "alternative": {
                "name": METHOD,
                "parent_model": "N69_TARGET_CONDITIONED",
                "guard_rule": "accept proposed full assignment only when every currently assigned native candidate retains its public assignment; otherwise use frozen baseline assignment and scores",
                "hungarian_solver_changed": False,
                "candidate_generation_changed": False,
                "numeric_public_id_feature": False,
            },
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "real_human_tape": False,
            "real_sam3_full_loop": False,
            "production_authorized": False,
            "variants": {},
        }
        for variant in parent.VARIANTS:
            frames = source.get("variants", {}).get(variant, {}).get("frames", [])
            expected = list(range(event["event_frame"] + 1, event["event_frame"] + 1 + parent.FRAMES_PER_EVENT))
            if len(frames) != parent.FRAMES_PER_EVENT or [int(item.get("frame", -1)) for item in frames] != expected:
                raise RuntimeError(f"N69 Stage06 future range mismatch {event_id}/{variant}")
            frame_outputs: list[dict[str, Any]] = []
            for raw in frames:
                frame = dict(raw)
                frame["variant"] = variant
                features = parent.build_feature_arrays(frame, event_runtime, include_offline_label=False)
                baseline_assignment = features["source_assignment"]
                baseline_branch = parent.runtime_branch_payload(
                    frame["write_baseline"], baseline_assignment, features["base"], "CURRENT_CCAM_BASELINE", features["pids"]
                )
                proposal = parent.apply_model_sidecar(model, features, mean, std, device)
                adjusted = np.asarray(proposal["adjusted_scores"], dtype=np.float32)
                proposed_assignment = parent.assignment_from_scores(adjusted) if features["target_column"] is not None else baseline_assignment.copy()
                proposed_branch = parent.runtime_branch_payload(
                    frame["write_baseline"], proposed_assignment, adjusted, "N69_TARGET_CONDITIONED_PROPOSAL", features["pids"]
                )
                target_public = int(event["target_public_id"])
                baseline_native = parent.native_assignment_map(baseline_branch)
                proposed_native = parent.native_assignment_map(proposed_branch)
                # The strict N69 untouched audit is candidate/native scoped,
                # not merely public-column scoped.  At runtime the true
                # target native ID is unavailable, so this alternative is
                # deliberately conservative: every currently assigned native
                # candidate is protected.  A proposal that displaces or
                # reassigns one is rejected without consulting posthoc GT.
                protected_native = sorted(
                    int(native)
                    for native, baseline_public_id in baseline_native.items()
                    if baseline_public_id is not None and proposed_native.get(native) != baseline_public_id
                )
                protected_changed = sorted(
                    int(baseline_native[native])
                    for native in protected_native
                    if baseline_native.get(native) is not None
                )
                rejected = bool(protected_native)
                if rejected:
                    guard_rejections += 1
                final_assignment = baseline_assignment.copy() if rejected else proposed_assignment
                final_scores = features["base"].copy() if rejected else adjusted
                guarded_branch = parent.runtime_branch_payload(
                    frame["write_baseline"], final_assignment, final_scores, METHOD, features["pids"]
                )
                final_delta = final_scores - features["base"]
                guarded_sidecar = {
                    "guard_applied": True,
                    "proposal_method": "N69_TARGET_CONDITIONED",
                    "proposal": proposal,
                    "accepted": not rejected,
                    "rejected": rejected,
                    "rejected_reason": "PROTECTED_UNTOUCHED_ID_ASSIGNMENT_CHANGE" if rejected else None,
                    "protected_public_ids_changed": protected_changed,
                    "protected_native_ids_changed": protected_native,
                    "adjusted_scores": final_scores.astype(float).tolist(),
                    "score_cells_changed": int(np.sum(np.abs(final_delta) > 1.0e-12)),
                    "max_abs_score_delta": float(np.max(np.abs(final_delta))) if final_delta.size else 0.0,
                    "target_column": features["target_column"],
                    "target_column_only": True,
                    "runtime_future_gt_used": False,
                }
                frame_outputs.append({
                    "frame": int(frame["frame"]),
                    "upstream_variant": variant,
                    "feature_audit": {
                        "candidate_count": int(features["candidate"].shape[0]),
                        "public_id_order": features["pids"],
                        "target_public_id": target_public,
                        "target_column": features["target_column"],
                        "candidate_feature_sha256": features["candidate_feature_digests"],
                        "memory_feature_sha256": features["memory_feature_digests"],
                        "human_feature_sha256": features["human_feature_digest"],
                        "runtime_future_gt_used": False,
                    },
                    "methods": {
                        "CURRENT_CCAM_BASELINE": {
                            "assignment": baseline_branch,
                            "sidecar": {"reason": "frozen_write_baseline", "target_column": features["target_column"], "score_cells_changed": 0, "runtime_future_gt_used": False},
                            "assignment_recomputed_from_adjusted_scores": False,
                            "runtime_future_gt_used": False,
                        },
                        METHOD: {
                            "assignment": guarded_branch,
                            "sidecar": guarded_sidecar,
                            "proposal_assignment": proposed_branch,
                            "assignment_recomputed_from_adjusted_scores": not rejected and features["target_column"] is not None,
                            "runtime_future_gt_used": False,
                        },
                    },
                    "candidate_stream_same_across_methods": True,
                    "public_id_axis_same_across_methods": True,
                    "event_frame_memory_read": False,
                    "first_event_memory_visible_frame": event["event_frame"] + 1,
                    "is_future_frame": True,
                    "runtime_future_gt_used": False,
                })
                runtime_frames += 1
            artifact["variants"][variant] = {"frame_count": len(frame_outputs), "frames": frame_outputs}
        atomic_json(ALT_ARTIFACT_DIR / f"{event_id}.json", artifact)
        completed += 1
        print(json.dumps({"event_id": event_id, "replayed_events": completed, "runtime_frames": runtime_frames, "guard_rejections": guard_rejections}, sort_keys=True), flush=True)
        del source
    status = {
        "schema": "N69_PROTECTED_UNTOUCHED_GUARD_RUNTIME_STATUS_V1",
        "status": "PASS_ALTERNATIVE_RUNTIME_REPLAY",
        "created_at_utc": parent.now(),
        "protocol": str(parent.MODEL_PROTOCOL),
        "protocol_sha256": parent.sha256_file(parent.MODEL_PROTOCOL),
        "checkpoint": str(parent.CHECKPOINT),
        "checkpoint_sha256": parent.sha256_file(parent.CHECKPOINT),
        "outputs": {"event_artifacts": str(ALT_ARTIFACT_DIR)},
        "metrics": {"event_count": completed, "frames": runtime_frames, "expected_events": parent.EVENT_COUNT, "expected_frames": parent.EVENT_COUNT * len(parent.VARIANTS) * parent.FRAMES_PER_EVENT, "guard_rejections": guard_rejections},
        "gate_checks": {
            "all_24_events": completed == parent.EVENT_COUNT,
            "all_5_variants": True,
            "all_100_frames": runtime_frames == parent.EVENT_COUNT * len(parent.VARIANTS) * parent.FRAMES_PER_EVENT,
            "same_candidate_stream": True,
            "same_public_id_axis": True,
            "hungarian_solver_changed": False,
            "candidate_generation_changed": False,
            "target_native_id_sent_to_runtime": False,
            "event_frame_memory_read_false": True,
            "first_memory_visible_at_event_plus_one": True,
            "runtime_future_gt_false": True,
            "production_authorized": False,
        },
        "alternative": METHOD,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "runtime_future_gt_used": False,
        "production_authorized": False,
    }
    atomic_json(ALT_RUNTIME_STATUS, status)
    return status


def score() -> dict[str, Any]:
    runtime = parent.load_json(ALT_RUNTIME_STATUS)
    if runtime.get("status") != "PASS_ALTERNATIVE_RUNTIME_REPLAY":
        raise RuntimeError("N69 Stage06 alternative runtime replay is incomplete")
    events = parent.load_event_map()
    mapping_index = parent.load_mapping_index()
    artifacts = sorted(ALT_ARTIFACT_DIR.glob("*.json"))
    if len(artifacts) != parent.EVENT_COUNT:
        raise RuntimeError(f"N69 Stage06 expected {parent.EVENT_COUNT} artifacts, found {len(artifacts)}")
    outcomes: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for artifact_path in artifacts:
        artifact = parent.load_json(artifact_path)
        event_id = str(artifact["event_id"])
        if event_id not in events:
            raise RuntimeError(f"unknown N69 Stage06 event {event_id}")
        event = events[event_id]
        if artifact.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"Stage06 runtime future GT boundary failed {event_id}")
        for variant in parent.VARIANTS:
            frames = artifact.get("variants", {}).get(variant, {}).get("frames", [])
            if len(frames) != parent.FRAMES_PER_EVENT:
                raise RuntimeError(f"Stage06 frame denominator failed {event_id}/{variant}")
            for frame in frames:
                key = (event_id, variant, int(frame["frame"]))
                if key not in mapping_index:
                    raise RuntimeError(f"Stage06 mapping row missing {key}")
                for method in ("CURRENT_CCAM_BASELINE", METHOD):
                    item = parent.frame_outcome(frame, event, mapping_index[key], method)
                    outcomes.append(item)
                    if method == METHOD:
                        diagnostics.append(item)
    methods = {method: parent.summarize_outcomes(outcomes, method) for method in ("CURRENT_CCAM_BASELINE", METHOD)}
    actions = sorted({event["action_type"] for event in events.values()})
    sequences = sorted({event["sequence"] for event in events.values()})
    by_action = {action: {method: parent.summarize_outcomes([item for item in outcomes if item["action_type"] == action], method) for method in methods} for action in actions}
    by_sequence = {sequence: {method: parent.summarize_outcomes([item for item in outcomes if item["sequence"] == sequence], method) for method in methods} for sequence in sequences}
    by_variant = {variant: {method: parent.summarize_outcomes([item for item in outcomes if item["variant"] == variant], method) for method in methods} for variant in parent.VARIANTS}
    alt_summary = methods[METHOD]
    lower = {str(horizon): alt_summary["horizons"][str(horizon)]["sequence_cluster_bootstrap"]["ci95"][0] for horizon in parent.HORIZONS}
    mapping_summary = parent.load_json(parent.MAPPING_SUMMARY)
    runtime_gate = {
        "runtime_complete": True,
        "all_events_24": runtime.get("metrics", {}).get("event_count") == parent.EVENT_COUNT,
        "all_frames_12000": runtime.get("metrics", {}).get("frames") == parent.EVENT_COUNT * len(parent.VARIANTS) * parent.FRAMES_PER_EVENT,
        "runtime_future_gt_false": runtime.get("runtime_future_gt_used") is False,
        "candidate_frame_integrity_100": mapping_summary.get("candidate_frame_integrity_100") is True,
        "target_scope_mapping_100_on_available_candidates": mapping_summary.get("target_scope_mapping_100_on_available_candidates") is True,
        "mapping_formal_provenance_100": mapping_summary.get("full_native_local_global_public_provenance") is True,
        "protected_guard_runtime": True,
    }
    synthetic_gate = {
        "status": "PASS" if all(value is True for value in runtime_gate.values()) and all(value > 0.0 for value in lower.values()) and alt_summary["correct_changes"] > alt_summary["incorrect_changes"] and alt_summary["untouched_regression_frame_rate"] == 0.0 else "FAIL_FUTURE_EFFECT",
        "strict_lower_ci_by_horizon": lower,
        "correct_changes_gt_incorrect_changes": alt_summary["correct_changes"] > alt_summary["incorrect_changes"],
        "untouched_regression_safe": alt_summary["untouched_regression_frame_rate"] == 0.0,
        "formal_mapping_provenance_100": runtime_gate["mapping_formal_provenance_100"],
        "candidate_frame_integrity_100": runtime_gate["candidate_frame_integrity_100"],
        "production_authorized": False,
    }
    result = {
        "schema": "N69_PROTECTED_UNTOUCHED_GUARD_PAIRED_RESULTS_V1",
        "status": "N69_STAGE06_ALTERNATIVE_EVALUATED",
        "created_at_utc": parent.now(),
        "protocol": str(parent.MODEL_PROTOCOL),
        "protocol_sha256": parent.sha256_file(parent.MODEL_PROTOCOL),
        "parent_n69_results": {"path": str(parent.RESULTS), "sha256": parent.sha256_file(parent.RESULTS)},
        "runtime_status": str(ALT_RUNTIME_STATUS),
        "event_count": parent.EVENT_COUNT,
        "variant_count": len(parent.VARIANTS),
        "frame_count": len(outcomes),
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "runtime_future_gt_used": False,
        "gt_loaded_only_posthoc": True,
        "evaluation_boundary": {"candidate_generation_changed": False, "hungarian_solver_changed": False, "same_candidate_stream": True, "same_public_id_axis": True, "mapping_version": parent.MAPPING_VERSION, "protected_guard": True},
        "alternative": {"name": METHOD, "guard_rejections": runtime["metrics"]["guard_rejections"], "guard_acceptances": parent.EVENT_COUNT * len(parent.VARIANTS) * parent.FRAMES_PER_EVENT - runtime["metrics"]["guard_rejections"], "rule": "reject any proposed assignment that displaces or changes a currently assigned native candidate"},
        "methods": methods,
        "by_action_type": by_action,
        "by_upstream_variant": by_variant,
        "by_sequence": by_sequence,
        "runtime_gate": runtime_gate,
        "synthetic_science_gate": synthetic_gate,
        "production_evidence_gate": {"status": "BLOCKED_NO_REAL_HUMAN_TAPE_OR_REAL_SAM3_FULL_LOOP", "real_human_tape": False, "real_sam3_full_loop": False, "production_authorized": False},
        "failure_root_cause": "The guard directly tests whether target-conditioned score proposals can improve the target without collateral changes to untouched IDs. A zero-effect result is evidence of an assignment-interface tradeoff, not a positive production result.",
        "outputs": {"event_artifacts": str(ALT_ARTIFACT_DIR), "paired_results": str(ALT_RESULTS), "assignment_diagnostics": str(ALT_DIAGNOSTICS)},
    }
    atomic_jsonl(ALT_DIAGNOSTICS, diagnostics)
    atomic_json(ALT_RESULTS, result)
    atomic_json(ALT_POSTHOC_STATUS, {
        "schema": "N69_STAGE06_POSTHOC_STATUS_V1",
        "status": "PASS_ALTERNATIVE_POSTHOC_SCORED",
        "created_at_utc": parent.now(),
        "paired_results": str(ALT_RESULTS),
        "assignment_diagnostics": str(ALT_DIAGNOSTICS),
        "event_count": parent.EVENT_COUNT,
        "runtime_frames_per_method": parent.EVENT_COUNT * len(parent.VARIANTS) * parent.FRAMES_PER_EVENT,
        "runtime_future_gt_used": False,
        "gt_loaded_only_posthoc": True,
        "production_authorized": False,
    })
    atomic_json(STAGE06, {
        "schema": "N69_STAGE_06_STATUS_V1",
        "status": "FAIL_ALTERNATIVE_STRICT_FUTURE_EFFECT",
        "created_at_utc": parent.now(),
        "alternative": METHOD,
        "runtime_status": str(ALT_RUNTIME_STATUS),
        "paired_results": str(ALT_RESULTS),
        "assignment_diagnostics": str(ALT_DIAGNOSTICS),
        "gate_checks": runtime_gate,
        "synthetic_science_gate": synthetic_gate,
        "guard_rejections": runtime["metrics"]["guard_rejections"],
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "runtime_future_gt_used": False,
        "production_authorized": False,
        "next_action": "Do not run TACT, calibration, selector, or decoder LoRA. Complete isolation audit and final N69 blocked report; production requires real human tape and real SAM3 full-loop evidence.",
    })
    return result


def record_failure(stage: str, exc: BaseException) -> None:
    ATTEMPTS.mkdir(parents=True, exist_ok=True)
    existing = sorted(ATTEMPTS.glob(f"stage_06_{stage}_failure_attempt*.json"))
    atomic_json(ATTEMPTS / f"stage_06_{stage}_failure_attempt{len(existing) + 1}.json", {
        "schema": "N69_FAILURE_ARTIFACT_V1",
        "status": "FAIL_PRESERVED",
        "stage": f"stage_06_{stage}",
        "created_at_utc": parent.now(),
        "failure_root_cause": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "production_authorized": False,
        "next_action": "Preserve this alternative failure and do not run downstream learning.",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("replay", "score"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.mode == "replay":
        print(json.dumps(replay(args.device), sort_keys=True))
    else:
        result = score()
        print(json.dumps({"status": result["synthetic_science_gate"]["status"], "paired_results": str(ALT_RESULTS), "method": METHOD, "correct": result["methods"][METHOD]["correct_changes"], "incorrect": result["methods"][METHOD]["incorrect_changes"], "h20": result["methods"][METHOD]["horizons"]["20"]["mean_utility_delta_raw_event_variant"]}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "unknown"
        record_failure(mode, exc)
        raise
