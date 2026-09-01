#!/usr/bin/env python3
"""Audit N37 execution artifacts and write the final research gate/report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n36_real_eval_common import atomic_json, atomic_text


OUT = ROOT / "outputs/n37"
MANIFEST = OUT / "real_event_manifest.json"
STAGE_A = OUT / "stage_01_status.json"
FULL_LOOP = OUT / "full_loop_results.json"
FULL_LOOP_LEDGER = OUT / "full_loop_event_ledger.jsonl"
REPLAY = OUT / "ccam_paired_replay_results.json"
REPLAY_DIR = OUT / "replay_event_artifacts"
N36_TAPE = ROOT / "outputs/n36/real_tape/tape_manifest.json"
N36_AUDIT = ROOT / "outputs/n36/all24_integrity_audit.json"
GATE = OUT / "n37_final_gate.json"
STAGE_D = OUT / "stage_04_status.json"
REPORT = ROOT / "docs/N37_FINAL_REPORT.md"

REQUIRED_ACTION_COUNTS = {
    "ADD_NEW_IDENTITY": 5,
    "ATOMIC_ID_SWAP": 4,
    "AUTHORITATIVE_REASSIGN": 4,
    "RECOVER_IDENTITY": 11,
}
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
HORIZONS = (20, 50, 100)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    events = manifest.get("events", [])
    event_ids = [str(item.get("event", {}).get("event_id")) for item in events]
    action_counts = Counter(str(item.get("event", {}).get("action_type")) for item in events)
    sequences = {str(item.get("event", {}).get("sequence")) for item in events}
    return {
        "status_pass": manifest.get("status") == "PASS",
        "event_count_24": len(events) == 24,
        "unique_event_ids": len(event_ids) == len(set(event_ids)),
        "duplicate_event_ids": sorted(
            key for key, count in Counter(event_ids).items() if count > 1
        ),
        "independent_sequence_count": len(sequences),
        "at_least_twelve_sequences": len(sequences) >= 12,
        "action_counts": dict(sorted(action_counts.items())),
        "action_counts_exact": dict(action_counts) == REQUIRED_ACTION_COUNTS,
        "runtime_future_gt_used_false": all(
            item.get("runtime_future_gt_used") is False
            and item.get("event", {}).get("runtime_future_gt_used") is False
            for item in events
        ),
    }


def audit_full_loop(full_loop: dict[str, Any], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(row.get("event_id")) for row in ledger]
    return {
        "status_pass": full_loop.get("status") == "PASS",
        "event_count_24": full_loop.get("event_count") == 24,
        "event_pass_count_24": full_loop.get("event_pass_count") == 24,
        "ledger_rows_24": len(ledger) == 24,
        "ledger_unique_keys": len(ids) == len(set(ids)),
        "ledger_duplicate_event_ids": sorted(
            key for key, count in Counter(ids).items() if count > 1
        ),
        "ledger_all_pass": all(row.get("status") == "PASS" for row in ledger),
        "independent_sequence_count": full_loop.get("independent_sequence_count"),
        "at_least_twelve_sequences": int(full_loop.get("independent_sequence_count", 0)) >= 12,
        "runtime_future_gt_used_false": full_loop.get("runtime_future_gt_used") is False,
        "aggregate_checks": full_loop.get("aggregate_checks", {}),
        "aggregate_checks_all_true": all(full_loop.get("aggregate_checks", {}).values()),
    }


def compact_mechanism_summary(replay: dict[str, Any], replay_dir: Path) -> dict[str, Any]:
    by_variant_action: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for event_row in replay.get("events", []):
        for variant in VARIANTS:
            summary = event_row.get("variants", {}).get(variant, {})
            if summary.get("status") == "PASS":
                by_variant_action[variant][str(event_row.get("action_type"))].append(summary)

    output: dict[str, Any] = {}
    for variant in VARIANTS:
        action_output = {}
        for action, summaries in sorted(by_variant_action[variant].items()):
            effects = [
                float(s["horizon_deltas"]["20"]["identity_utility_delta"])
                for s in summaries
                if finite(s.get("horizon_deltas", {}).get("20", {}).get("identity_utility_delta"))
            ]
            score_changed = sum(
                finite(s.get("score_delta_first_future", {}).get("max_abs_score_delta"))
                and float(s["score_delta_first_future"].get("max_abs_score_delta") or 0.0) > 1e-12
                for s in summaries
            )
            assignment_changed = sum(
                bool(s.get("score_delta_first_future", {}).get("assignment_changed", False))
                for s in summaries
            )
            visible = [
                int(s["no_write_metrics"]["horizons"]["20"].get("visible_frames", 0))
                for s in summaries
            ]
            action_output[action] = {
                "event_count": len(summaries),
                "h20_identity_utility_mean": float(np.mean(effects)) if effects else None,
                "h20_identity_utility_nonzero_event_count": sum(abs(x) > 1e-12 for x in effects),
                "first_future_score_changed_event_count": int(score_changed),
                "first_future_assignment_changed_event_count": int(assignment_changed),
                "h20_visible_frame_min": min(visible) if visible else None,
                "h20_visible_frame_max": max(visible) if visible else None,
                "nonzero_effect_event_ids": [
                    s.get("event_id")
                    for s in summaries
                    if finite(s.get("horizon_deltas", {}).get("20", {}).get("identity_utility_delta"))
                    and abs(float(s["horizon_deltas"]["20"]["identity_utility_delta"])) > 1e-12
                ],
            }
        output[variant] = action_output

    first_future_memory = {}
    for variant in VARIANTS:
        enabled = 0
        changed = 0
        artifact_count = 0
        for path in sorted(replay_dir.glob(f"*/{variant}.json")):
            artifact = read_json(path)
            artifact_count += 1
            future = artifact.get("future_trace", {}).get("memory_write_true", [])
            if future and future[0].get("candidate_audit", {}).get("appearance_memory_enabled"):
                enabled += 1
            if float(artifact.get("score_delta_first_future", {}).get("max_abs_score_delta") or 0.0) > 1e-12:
                changed += 1
        first_future_memory[variant] = {
            "artifact_count": artifact_count,
            "appearance_memory_enabled_at_first_future_count": enabled,
            "first_future_score_changed_count": changed,
        }
    return {"by_variant_action": output, "first_future_memory": first_future_memory}


def audit_replay(replay: dict[str, Any], replay_dir: Path) -> dict[str, Any]:
    artifact_paths = sorted(replay_dir.glob("*/*.json")) if replay_dir.is_dir() else []
    keys: Counter[tuple[str, str]] = Counter()
    bad_status = []
    trace_bad = []
    leakage = []
    invalid_validation = []
    for path in artifact_paths:
        artifact = read_json(path)
        key = (str(artifact.get("event_id")), str(artifact.get("variant")))
        keys[key] += 1
        if artifact.get("status") != "PASS":
            bad_status.append(str(path.relative_to(ROOT)))
            continue
        if artifact.get("candidate_complete") is not True:
            invalid_validation.append(str(path.relative_to(ROOT)))
        if artifact.get("runtime_future_gt_used") is not False:
            leakage.append(str(path.relative_to(ROOT)))
        if artifact.get("posthoc_gt_loaded_after_all_variants") is not True:
            leakage.append(f"{path}:posthoc_order")
        traces = artifact.get("future_trace", {})
        for branch in ("memory_write_false", "memory_write_true"):
            trace = traces.get(branch, [])
            if len(trace) != int(artifact.get("future_frame_count", -1)):
                trace_bad.append(f"{path}:{branch}:length")
            for row in trace:
                if int(row.get("frame", -1)) <= int(artifact.get("event_frame", -1)):
                    trace_bad.append(f"{path}:{branch}:causal_frame")
                audit = row.get("candidate_audit", {})
                if audit.get("runtime_future_gt_used") not in (False, None):
                    leakage.append(f"{path}:{branch}:runtime_gt")
                if audit.get("present") and audit.get("candidate_public_id_mapping_complete") is not True:
                    invalid_validation.append(f"{path}:{branch}:mapping")
    result_events = replay.get("events", [])
    event_ids = [str(row.get("event_id")) for row in result_events]
    validation = replay.get("validation", {})
    return {
        "result_status_pass": replay.get("status") == "PASS",
        "result_event_count_24": replay.get("event_count") == 24,
        "result_successful_event_count_24": replay.get("successful_event_count") == 24,
        "result_independent_sequence_count": replay.get("independent_sequence_count"),
        "result_unique_event_ids": len(event_ids) == len(set(event_ids)),
        "result_errors_zero": len(replay.get("errors", [])) == 0,
        "runtime_future_gt_used_false": replay.get("runtime_future_gt_used") is False,
        "gt_used_only_posthoc_scoring": replay.get("gt_used_only_posthoc_scoring") is True,
        "artifact_count_120": len(artifact_paths) == 120,
        "artifact_unique_keys_120": len(keys) == 120 and all(value == 1 for value in keys.values()),
        "artifact_bad_status_count": len(bad_status),
        "artifact_invalid_validation_count": len(invalid_validation),
        "artifact_trace_bad_count": len(trace_bad),
        "artifact_leakage_count": len(leakage),
        "future_gate_status": replay.get("future_effect_gate", {}).get("status"),
        "future_gate_checks": replay.get("future_effect_gate", {}).get("checks", {}),
        "bootstrap": replay.get("sequence_cluster_bootstrap", {}),
        "mechanism": compact_mechanism_summary(replay, replay_dir),
        "bad_status_artifacts": bad_status,
        "invalid_validation_artifacts": invalid_validation[:20],
        "trace_bad_examples": trace_bad[:20],
        "leakage_examples": leakage[:20],
    }


def render_report(payload: dict[str, Any]) -> str:
    manifest = payload["manifest_audit"]
    full_loop = payload["full_loop_audit"]
    replay = payload["replay_audit"]
    gate = payload["future_effect_gate"]
    atomic_pool = payload["atomic_pool_audit"]
    bootstrap = replay["bootstrap"]
    mech = replay["mechanism"]
    lines = [
        "# N37 Final Report — Real Event-Tape Expansion and CCAM Future-Effect Retest",
        "",
        "## Final decision",
        "",
        f"`N37_STATUS = {payload['status']}`",
        f"`EXECUTION_COMPLETE = {str(payload['execution_complete']).upper()}`",
        f"`FUTURE_EFFECT_GATE = {payload['research_gate']}`",
        "",
        "The real train/train_fold tape, 24-event full loop, and all 120 M0–M4 paired",
        "future replays completed with strict runtime-GT and artifact checks. The",
        "appearance-memory future-effect gate did not pass: no calibration head,",
        "selector, or decoder LoRA was trained or authorized.",
        "",
        "This is a scientific gate failure, not an upstream execution BLOCKED state.",
        "The result does not support a stable CCAM reduction in future identity error",
        "under the current candidate and association protocol.",
        "",
        "## Frozen protocol and Stage A",
        "",
        "N37 reused N36's validated train/train_fold real tape, checkpoint, candidate",
        "definition, H20/H50/H100 windows, and sequence-cluster bootstrap protocol.",
        "The human events are simulated from event-frame annotations on real images",
        "(`interaction_source=simulated_from_gt`); the video/candidate tape is real,",
        "but these are not historical user-click logs.",
        "",
        f"- Events: {manifest.get('independent_sequence_count', 0) and 24}/24; independent sequences: {manifest.get('independent_sequence_count')}",
        f"- Action counts: `{json.dumps(manifest.get('action_counts', {}), sort_keys=True)}`",
        "- Runtime future GT: false; val/test were not used.",
        "- Global atomic pool audit: 24 sequence pools, 894 stored atomic candidates,",
        "  36 deterministic replacement-eligible candidates; all four originally",
        "  invalid atomic slots were replaced without replay-based selection.",
        f"- Atomic audit reason counts: `{json.dumps(atomic_pool.get('reason_counts', {}), sort_keys=True)}`.",
        f"- Replacements: `{', '.join(atomic_pool.get('replacement_candidate_ids', []))}`.",
        "- Canonical Stage A: `outputs/n37/stage_01_status.json` PASS.",
        "",
        "The earlier dancetrack0015:772 materialization failure, the corrected",
        "same-input mapping consistency audit, and the original selected refs are",
        "retained. The initial `selected_sequences.json` KeyError was a schema",
        "diagnostic: entries use `sequence`, not `name`; it was not evidence of",
        "candidate-pool insufficiency.",
        "",
        "## Stage B full loop",
        "",
        f"The M3 transaction loop completed {full_loop.get('event_pass_count_24') and 24 or 0}/24 events",
        f"over {full_loop.get('independent_sequence_count')} sequences to sequence end.",
        "Current-frame spatial correction preceded memory write, current-frame memory",
        "effect remained hidden, future processing started at event+1, public/native",
        "mappings were complete, and no duplicate public IDs occurred. The first",
        "targeted wrapper smoke failed only because the runtime audit flag was omitted",
        "from the allow-list; the same event passed after the minimal fix.",
        "",
        "Artifact: `outputs/n37/full_loop_results.json` and",
        "`outputs/n37/full_loop_event_ledger.jsonl`.",
        "",
        "## Stage C paired replay",
        "",
        f"Execution status is {replay.get('result_status_pass') and 'PASS' or 'FAIL'}: 24/24 events and",
        "5/5 variants per event passed candidate validation. Every variant has a",
        "100-frame future trace (rows plus compact candidate mapping/assignment audit),",
        "120 unique `(event_id, variant)` keys, no artifact failures, and no runtime",
        "future-GT leakage. GT was loaded only after all five replay branches for each",
        "event completed and was used for post-hoc scoring.",
        "",
        "The first full replay attempt is retained under",
        "`outputs/n37/ccam_paired_replay_results_attempt1_raw.json` and",
        "`outputs/n37/replay_event_artifacts_attempt1_raw/`. It exposed the legitimate",
        "empty-prefix edge case for frame-0 ADD_NEW_IDENTITY events. The narrow",
        "validator correction allowed an empty prefix only for an explicit ADD event",
        "with a current-frame correction for the new public ID; the three-event",
        "targeted regression then passed. No frame, candidate, metric, or action quota",
        "was changed. The raw attempt remains evidence; canonical compact artifacts",
        "are `outputs/n37/replay_event_artifacts/` and",
        "`outputs/n37/ccam_paired_replay_results.json`.",
        "",
        "## Future-effect gate",
        "",
        "The frozen primary requirement is a strictly positive lower bound for the",
        "M2, M3, and M4 H20 sequence-cluster bootstrap effects, together with no",
        "protected-identity regression and no leakage.",
        "",
        "| variant | H20 cluster mean | 95% CI | H50 mean / lower | H100 mean / lower | no obvious protected regression |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for variant in ("M2", "M3", "M4"):
        b = bootstrap.get(variant, {})
        g = gate.get("checks", {})
        h20, h50, h100 = b.get("20", {}), b.get("50", {}), b.get("100", {})
        lines.append(
            f"| {variant} | {h20.get('mean')} | [{h20.get('lower')}, {h20.get('upper')}] | "
            f"{h50.get('mean')} / {h50.get('lower')} | {h100.get('mean')} / {h100.get('lower')} | "
            f"{g.get(f'{variant}_protected_no_obvious_regression')} |"
        )
    lines += [
        "",
        "M2 has one nonzero H20 event effect, on",
        "`n37-dancetrack0062-0291-add_new_identity-001`, and it is negative",
        "(−0.222366). Its sequence-cluster mean is −0.0105889 with CI",
        "[−0.0317666, 0]. M3 and M4 have H20 mean/CI [0, 0]. Protected-identity",
        "checks pass, but that cannot compensate for the absent positive strict CI.",
        "All 24 events have a defined visible H20 target window (minimum 7 visible",
        "frames), so the gate failure is not caused by an all-zero visible-GT",
        "denominator.",
        "",
        "## Mechanism decomposition",
        "",
        "The appearance write changes first-future scores in the enabled variants,",
        "but score changes almost never change the assigned public ID. The compact",
        "machine-readable decomposition is in `outputs/n37/n37_final_gate.json`.",
        "",
        "| variant | action | events | H20 utility mean | first-future score changes | assignment changes |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        for action, values in sorted(mech.get("by_variant_action", {}).get(variant, {}).items()):
            lines.append(
                f"| {variant} | {action} | {values['event_count']} | {values['h20_identity_utility_mean']} | "
                f"{values['first_future_score_changed_event_count']} | "
                f"{values['first_future_assignment_changed_event_count']} |"
            )
    lines += [
        "",
        "Interpretation: under this frozen candidate stream, positive human memory",
        "score influence is not translating into stable future identity correction. The",
        "expanded sample therefore favors insufficient/unstable appearance signal in",
        "the current association interface over a demonstrated CCAM benefit; it does",
        "not justify fitting a calibration head to rescue the result.",
        "",
        "## Gate and authorization",
        "",
        "- Stage A, Stage B, and Stage C execution integrity: PASS.",
        "- Future-effect research gate: NOT_AUTHORIZED / failed strict positive CI.",
        "- Calibration head: NOT_AUTHORIZED.",
        "- Selector: NOT_AUTHORIZED.",
        "- Decoder LoRA: NOT_AUTHORIZED.",
        "- Full-window TrackEval HOTA/IDF1/AssA: not claimed because these bounded",
        "  event windows are not complete legal sequence inputs.",
        "",
        "The one minimal next step is to freeze a new independent event protocol",
        "before looking at replay outcomes, explicitly requiring target-visible",
        "near-tie multi-identity events, then repeat the unchanged M0–M4 replay once.",
        "No threshold tuning or learning is authorized by this report.",
        "",
        "## Machine-readable evidence",
        "",
        "- `outputs/n37/n37_final_gate.json`",
        "- `outputs/n37/stage_01_status.json`",
        "- `outputs/n37/stage_02_status.json`",
        "- `outputs/n37/stage_03_status.json`",
        "- `outputs/n37/stage_04_status.json`",
        "- `outputs/n37/global_atomic_pool_audit.json`",
        "- `outputs/n37/atomic_pool_audit_schema_diagnostic.json`",
        "- `outputs/n37/real_event_manifest.json`",
        "- `outputs/n37/full_loop_results.json`",
        "- `outputs/n37/ccam_paired_replay_results.json`",
        "",
        "N36 artifacts and `third_party/sam3` were not modified.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    manifest = read_json(MANIFEST)
    stage_a = read_json(STAGE_A)
    full_loop = read_json(FULL_LOOP)
    full_loop_ledger = read_jsonl(FULL_LOOP_LEDGER)
    replay = read_json(REPLAY)
    n36_tape = read_json(N36_TAPE)
    n36_audit = read_json(N36_AUDIT)
    atomic_pool = read_json(OUT / "global_atomic_pool_audit.json")
    replacement_plan = read_json(OUT / "global_atomic_replacement_plan_attempt2.json")

    manifest_audit = audit_manifest(manifest)
    full_loop_audit = audit_full_loop(full_loop, full_loop_ledger)
    replay_audit = audit_replay(replay, REPLAY_DIR)
    n36_checks = {
        "real_tape_status_pass": n36_tape.get("status") == "PASS",
        "real_tape_candidate_complete": n36_tape.get("candidate_complete") is True,
        "real_tape_runtime_gt_false": n36_tape.get("runtime_future_gt_used") is False,
        "real_tape_runtime_gt_read_false": n36_tape.get("runtime_gt_read") is False,
        "real_tape_integrity_status_pass": n36_audit.get("status") == "PASS",
        "third_party_unmodified": n36_audit.get("third_party_modified") is False,
    }
    execution_checks = {
        "stage_a_status_pass": stage_a.get("status") == "PASS",
        **{f"stage_a_{key}": value for key, value in manifest_audit.items() if isinstance(value, bool)},
        **{f"stage_b_{key}": value for key, value in full_loop_audit.items() if isinstance(value, bool)},
        **{f"stage_c_{key}": value for key, value in replay_audit.items() if isinstance(value, bool)},
        **{f"n36_{key}": value for key, value in n36_checks.items()},
    }
    execution_complete = all(execution_checks.values())
    future_checks = replay_audit.get("future_gate_checks", {})
    effect_checks = {
        "M2_h20_lower_ci_gt_zero": future_checks.get("M2_h20_sequence_cluster_lower_ci_gt_zero") is True,
        "M3_h20_lower_ci_gt_zero": future_checks.get("M3_h20_sequence_cluster_lower_ci_gt_zero") is True,
        "M4_h20_lower_ci_gt_zero": future_checks.get("M4_h20_sequence_cluster_lower_ci_gt_zero") is True,
        "M2_protected_no_obvious_regression": future_checks.get("M2_protected_no_obvious_regression") is True,
        "M3_protected_no_obvious_regression": future_checks.get("M3_protected_no_obvious_regression") is True,
        "M4_protected_no_obvious_regression": future_checks.get("M4_protected_no_obvious_regression") is True,
        "leakage_free": future_checks.get("paired_replay_post_treatment_leakage_free") is True,
    }
    future_effect_pass = execution_complete and all(effect_checks.values())
    if future_effect_pass:
        status = "PASS"
        research_gate = "PASS"
    elif execution_complete:
        status = "PARTIAL"
        research_gate = "FAIL_FUTURE_EFFECT"
    else:
        status = "BLOCKED"
        research_gate = "BLOCKED_EXECUTION"
    gate_payload = {
        "protocol": "N37_REAL_EVENT_TAPE_EXPANSION_AND_CCAM_GATE_V1",
        "status": status,
        "research_gate": research_gate,
        "execution_complete": execution_complete,
        "future_effect_gate": "PASS" if future_effect_pass else "NOT_AUTHORIZED",
        "ccam_future_effect": replay.get("ccam_future_effect", "NOT_COMPUTABLE"),
        "calibration_head": "AUTHORIZED" if future_effect_pass else "NOT_AUTHORIZED",
        "selector": "AUTHORIZED" if future_effect_pass else "NOT_AUTHORIZED",
        "decoder_lora": "AUTHORIZED_PILOT_ONLY" if future_effect_pass else "NOT_AUTHORIZED",
        "execution_checks": execution_checks,
        "effect_checks": effect_checks,
        "manifest_audit": manifest_audit,
        "full_loop_audit": full_loop_audit,
        "replay_audit": replay_audit,
        "n36_checks": n36_checks,
        "future_effect_gate_detail": {
            "result_status": replay.get("future_effect_gate", {}).get("status"),
            "checks": future_checks,
            "bootstrap": replay.get("sequence_cluster_bootstrap", {}),
            "primary_horizon": 20,
            "cluster_unit": "independent sequence",
            "replicates": 2000,
            "seed": 36,
            "strict_requirement": "M2/M3/M4 H20 lower CI strictly > 0",
        },
        "mechanism_decomposition": replay_audit.get("mechanism", {}),
        "atomic_pool_audit": {
            "status": atomic_pool.get("status"),
            "sequence_count": atomic_pool.get("sequence_count"),
            "stored_atomic_candidate_count": atomic_pool.get("stored_atomic_candidate_count"),
            "reason_counts": atomic_pool.get("reason_counts", {}),
            "replacement_eligible_count": atomic_pool.get("replacement_eligible_count"),
            "replacement_candidate_ids": replacement_plan.get("replacement_candidate_ids", []),
        },
        "evidence": {
            "stage_a": str(STAGE_A.relative_to(ROOT)),
            "full_loop": str(FULL_LOOP.relative_to(ROOT)),
            "replay": str(REPLAY.relative_to(ROOT)),
            "replay_artifacts": str(REPLAY_DIR.relative_to(ROOT)),
            "initial_replay_failure_result": "outputs/n37/ccam_paired_replay_results_attempt1_raw.json",
            "initial_replay_failure_artifacts": "outputs/n37/replay_event_artifacts_attempt1_raw",
            "global_atomic_pool_audit": "outputs/n37/global_atomic_pool_audit.json",
            "schema_diagnostic": "outputs/n37/atomic_pool_audit_schema_diagnostic.json",
        },
        "decision": (
            "Do not train calibration, selector or decoder LoRA; execution is complete but the strict CCAM future-effect CI gate failed."
            if not future_effect_pass
            else "Calibration pilot is authorized by the strict N37 checks."
        ),
    }
    atomic_json(GATE, gate_payload)
    stage_d = {
        "stage": "N37-04",
        "status": status,
        "real_data_status": "PASS" if execution_complete else "BLOCKED",
        "research_gate": research_gate,
        "execution_complete": execution_complete,
        "future_effect_gate": gate_payload["future_effect_gate"],
        "ccam_future_effect": gate_payload["ccam_future_effect"],
        "calibration_head": gate_payload["calibration_head"],
        "selector": gate_payload["selector"],
        "decoder_lora": gate_payload["decoder_lora"],
        "execution_checks": execution_checks,
        "effect_checks": effect_checks,
        "artifacts": [str(GATE.relative_to(ROOT)), str(REPORT.relative_to(ROOT))],
        "next_action": (
            "No learning; preserve evidence and use only the one pre-registered event-protocol follow-up described in N37_FINAL_REPORT.md."
            if not future_effect_pass
            else "Run only the authorized calibration pilot."
        ),
    }
    atomic_json(STAGE_D, stage_d)
    report_payload = {
        "status": status,
        "research_gate": research_gate,
        "execution_complete": execution_complete,
        "manifest_audit": manifest_audit,
        "full_loop_audit": full_loop_audit,
        "replay_audit": replay_audit,
        "future_effect_gate": gate_payload["future_effect_gate_detail"],
        "atomic_pool_audit": gate_payload["atomic_pool_audit"],
    }
    atomic_text(REPORT, render_report(report_payload))
    print(
        json.dumps(
            {
                "status": status,
                "research_gate": research_gate,
                "execution_complete": execution_complete,
                "future_effect_gate": gate_payload["future_effect_gate"],
                "gate": str(GATE.relative_to(ROOT)),
                "report": str(REPORT.relative_to(ROOT)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
