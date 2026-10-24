"""Seal the N72R21R2 negative/unfinished scientific closure.

This is a read-only evidence assembler.  It never launches a replay, fit,
TrackEval job, confirmation run, or memory write.  Missing branches are
represented explicitly as NOT_RUN; presence of a named file is not treated as
scientific success.
"""
import argparse
import json
from pathlib import Path

from scripts.n72r21r2_common import (
    GOAL,
    OUT,
    ROOT,
    append_log,
    read_json,
    sha256,
    storage,
    update_status,
    utcnow,
    write_json,
)


SELF = "scripts/n72r21r2_finalize_result.py"
DECISION = "FAIL_GLOBAL_MOT_TRANSFER"
MISSING_REPORTS = (
    "on_policy/CORRECTION_TRAINING.json",
    "mot/INNER_RESULTS.json",
    "mot/PER_SEQUENCE.json",
    "mot/HOTA_ASSA_IDF1.json",
    "mot/N01_N10.json",
    "mot/NON_TARGET_DAMAGE.json",
    "mot/LATENCY.json",
    "confirmation/FROZEN_POLICY.json",
    "confirmation/RESULTS.json",
    "confirmation/PAIRED_DELTAS.json",
)


def ref(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha256(path)}


def latest_versioned(directory, prefix):
    candidates = list(Path(directory).glob(f"{prefix}_V*.json"))
    if not candidates:
        raise FileNotFoundError(f"No versioned {prefix} receipt in {directory}")
    return max(candidates, key=lambda path: int(path.stem.rsplit("_V", 1)[1]))


def compact_group(group):
    if not group["status"].startswith("COMPLETE_ACTUAL"):
        return {
            "status": group["status"],
            "actual_group_driver_status": group.get("actual_group_driver_status"),
            "retained_attempt_failures": group.get("retained_attempt_failures", []),
            "scientific_success": False,
        }
    roles = {}
    for role, role_record in group.get("roles", {}).items():
        summary = role_record["full_MOT_and_density"]
        overall = summary["actual_full_video"]["overall"]
        decisions = summary["seed_mean_one_shot_decisions_NOT_independent_roots"]
        gates = role_record["original_development_gates"]
        roles[role] = {
            "usable_video_clusters": overall["usable_video_clusters"],
            "macro_metrics": overall["macro_metrics"],
            "macro_deltas_vs_C0": overall["macro_deltas"],
            "paired_video_bootstrap95_delta": overall["paired_video_bootstrap95_delta"],
            "improved_HOTA_videos": overall["improved_HOTA_videos"],
            "degraded_HOTA_videos": overall["degraded_HOTA_videos"],
            "effective_decisions": decisions["effective_decisions"],
            "beneficial_complete_H100_decisions": decisions["beneficial_complete_H100_decisions"],
            "N01_frames": decisions.get("N01_frames"),
            "N10_frames": decisions.get("N10_frames"),
            "G1": gates["G1"]["status"],
            "G2": gates["G2"]["status"],
            "confirmation_authorized": gates["confirmation_authorized"],
            "next_association_stage_authorized": gates["next_association_stage_authorized"],
        }
    return {"status": group["status"], "selected_point": group["selected_point"], "roles": roles,
            "scientific_success": False}


def load_evidence():
    latest = read_json(OUT / "mot/development_closure_v1/latest.json")
    closure = read_json(latest["path"])
    disposition = read_json(read_json(OUT / "mot/development_dispositions_v2/latest.json")["path"])
    main_driver = read_json(OUT / "mot/main_policy_v1/driver.json")
    state_driver = read_json(OUT / "mot/state_policy_v1/driver.json")
    goal = read_json(OUT / "FINAL_GOAL.json")
    prereg = read_json(OUT / "PREREGISTRATION.json")
    assert goal["goal"] == prereg["goal"] == GOAL and goal["goal_frozen"] and prereg["frozen"]
    assert closure["registered_groups"] == 17 and closure["actual_complete_groups"] == 16
    assert disposition["measured_no_fit_terminal_groups"] == 1
    assert main_driver["status"] == "AVAILABLE_MAIN_FULL_VIDEO_DEVELOPMENT_COMPLETE_SEPARATE_CLOSURE_REQUIRED"
    assert len(main_driver["completed"]) == 1344 and not main_driver["failed_retained"]
    assert state_driver["status"] == "AVAILABLE_STATE_SOURCE_FULL_MOT_FINISHED_SEPARATE_ON_POLICY_AND_GATE_CLOSURE_REQUIRED"
    assert len(state_driver["completed"]) == 192 and not state_driver["failed_retained"]
    compact = {key: compact_group(value) for key, value in closure["all17_group_census"].items()}
    complete = [key for key, value in compact.items() if value["status"].startswith("COMPLETE_ACTUAL")]
    terminal = [key for key, value in compact.items() if not value["status"].startswith("COMPLETE_ACTUAL")]
    assert len(complete) == 16 and terminal == ["STATE/BASELINE_STATE__L3_HARM_AWARE"]
    assert all(
        role["G1"].startswith("FAIL") and role["G2"].startswith("FAIL")
        for value in compact.values() if value["status"].startswith("COMPLETE_ACTUAL")
        for role in value["roles"].values()
    )
    return {
        "goal": goal,
        "preregistration": prereg,
        "closure_ref": ref(latest["path"]),
        "closure_latest_ref": ref(OUT / "mot/development_closure_v1/latest.json"),
        "disposition_ref": ref(OUT / "mot/development_dispositions_v2/latest.json"),
        "main_driver_ref": ref(OUT / "mot/main_policy_v1/driver.json"),
        "state_driver_ref": ref(OUT / "mot/state_policy_v1/driver.json"),
        "groups": compact,
        "complete_groups": complete,
        "terminal_groups": terminal,
    }


def common(evidence, status):
    return {
        "stage": "N72R21R2",
        "goal": GOAL,
        "goal_file": "outputs/N72R21R2/FINAL_GOAL.json",
        "status": status,
        "scientific_success": False,
        "scientific_decision": DECISION,
        "confirmation_authorized": False,
        "next_association_stage_authorized": False,
        "CONFIRM_VAL_TEST_SOT_accessed": False,
        "runtime_future_GT_used": False,
        "evidence": {
            "development_closure": evidence["closure_ref"],
            "development_dispositions": evidence["disposition_ref"],
        },
    }


def write_missing(evidence):
    reports = {
        "on_policy/CORRECTION_TRAINING.json": {
            "status": "NOT_RUN_ON_POLICY_NOT_AUTHORIZED_AFTER_DEVELOPMENT_G1_G2_FAILURE",
            "reason": "No development group satisfied both nonvacuous-correction and MOT gates; controlled state-source replay is not model-generated on-policy learning.",
        },
        "mot/INNER_RESULTS.json": {
            "status": "COMPLETE_ACTUAL_DEVELOPMENT_INNER_CENSUS_NO_QUALIFIED_STRATEGY",
            "reason": "All 16 complete groups retain both frozen INNER points and shared selection readback; every role failed G1 and G2.",
            "groups": evidence["groups"],
        },
        "mot/PER_SEQUENCE.json": {
            "status": "COMPLETE_ACTUAL_DEVELOPMENT_PER_SEQUENCE_SUMMARY_NO_IMPROVEMENT",
            "group_role_summary": evidence["groups"],
            "per_sequence_cluster_CI_source": evidence["closure_ref"],
        },
        "mot/HOTA_ASSA_IDF1.json": {
            "status": "COMPLETE_ACTUAL_DEVELOPMENT_MOT_METRICS_NO_DELTA",
            "reported_metrics": ["HOTA", "AssA", "IDF1", "DetA", "IDSW", "MOTA"],
            "group_role_summary": evidence["groups"],
            "all_complete_group_macro_deltas_are_zero": True,
            "all_complete_group_paired_video_intervals_are_zero": True,
        },
        "mot/N01_N10.json": {
            "status": "COMPLETE_ACTUAL_DEVELOPMENT_DECISION_CENSUS_ZERO_EFFECTIVE_ACTIONS",
            "group_role_summary": evidence["groups"],
            "independent_event_roots_proven": False,
        },
        "mot/NON_TARGET_DAMAGE.json": {
            "status": "COMPLETE_ACTUAL_DEVELOPMENT_HARM_AUDIT_NO_QUALIFICATION",
            "group_role_summary": evidence["groups"],
            "UNKNOWN_is_not_verified_OTHER": True,
            "memory_or_population_safety_not_established": True,
        },
        "mot/LATENCY.json": {
            "status": "NOT_RUN_NO_QUALIFIED_DEPLOYED_POLICY",
            "reason": "No final intervention policy was authorized; model cost/timing evidence remains a separate diagnostic.",
        },
        "confirmation/FROZEN_POLICY.json": {
            "status": "NOT_RUN_CONFIRMATION_UNAUTHORIZED_G1_G2_FAILED",
            "reason": "No development strategy passed the preregistered gates; CONFIRMATION8 remained unopened.",
        },
        "confirmation/RESULTS.json": {
            "status": "NOT_RUN_CONFIRMATION_UNAUTHORIZED",
            "registered_confirmation_sequences": evidence["preregistration"]["split"]["confirmation"],
            "GT_labels_parsed": False,
        },
        "confirmation/PAIRED_DELTAS.json": {
            "status": "NOT_RUN_CONFIRMATION_UNAUTHORIZED",
            "paired_deltas": None,
            "reason": "No confirmation policy was frozen.",
        },
    }
    refs = {}
    for relative, payload in reports.items():
        path = write_json(relative, {**common(evidence, payload.pop("status")), **payload})
        refs[relative] = ref(path)
    return refs


def report_text(evidence, report_refs, test_receipt):
    return f"""# InterMOT N72R21R2 Final Report

FINAL GOAL:
Event-Level Causal Identity Association: Safe One-Click Intervention with Fresh-Sequence Generalization for Online MOT

## Scientific decision

`{DECISION}`

The frozen question was whether a single user click can produce a safe,
globally causal identity intervention that improves online MOT on fresh
development sequences.  The answer in this registered development evidence is
no: all completed intervention groups failed the nonvacuous-correction and
MOT-improvement gates.  The result is a negative transfer conclusion, not a
claim that the underlying visual representation contains no information.

## Executed scope

- Frozen TRAIN lineage: 40 roles; FIT16 and INNER8 were used for development;
  CONFIRMATION8 remained unopened.
- MAIN: 42 registered seed jobs, 1344 full-MOT cells, 14 groups, zero retained
  cell failures.
- Controlled STATE: 192 full-MOT cells, two complete groups; BASELINE_STATE
  retained as three measured no-optimizer supervision failures, not as a
  fabricated trained model.
- Development closure: 16 complete full-MOT groups plus one terminal no-fit
  disposition out of 17 registered groups.
- Every complete group/role failed `G1` and `G2`; effective actions and
  beneficial complete-H100 decisions were zero, all paired nine-metric MOT
  deltas were zero, and paired video intervals were `[0, 0]`.

Evidence: `{evidence['closure_ref']['path']}` and
`{evidence['disposition_ref']['path']}`.

## Memory and open-set boundaries

The executed M-A memory frontier did not satisfy the necessary empirical
memory conditions; M-B was therefore not authorized.  Open-set and memory
reports remain diagnostics, not permission to override the full global MOT
association.  Independent beneficial causal roots were not established.

## Confirmation and next stage

`CONFIRMATION8`, VAL, TEST and SOT were not accessed.  No confirmation policy
was frozen, `confirmation_authorized=false`, and
`next_association_stage_authorized=false`.  Runtime future GT was not used.

## Regression evidence

The latest recorded full regression is **1582 passed / 5 unchanged historical
failures**, with process exit 1; it is not reported as all-pass.  The five
retained failures are listed in `{test_receipt['path']}`.  Focused R2 tests
remain separately recorded as passing in `outputs/N72R21R2/tests/FOCUSED.json`.

## Reproducibility and delivery

All source, protocol, input and result references are SHA-sealed in the JSON
artifacts.  Large datasets, checkpoints and traces remain outside GitHub.
Only code, tests and necessary documentation are eligible for publication.
The final result is `{OUT / 'FINAL_RESULT.json'}`.

This report deliberately leaves the scientific result negative and the next
stage closed; no threshold, split, candidate matcher, association solver or
confirmation outcome was changed after observing the result.
"""


def run(dry_run=False):
    storage(8 << 20)
    evidence = load_evidence()
    existing = [relative for relative in (*MISSING_REPORTS, "FINAL_RESULT.json", "FINAL_REPORT.md")
                if (OUT / relative).exists()]
    if existing:
        raise FileExistsError(f"Preserve existing final evidence; already present: {existing}")
    regression_path = latest_versioned(OUT / "tests", "REGRESSION")
    test_receipt = read_json(regression_path)
    if not dry_run:
        report_refs = write_missing(evidence)
        report_path = OUT / "FINAL_REPORT.md"
        report_path.write_text(report_text(evidence, report_refs, ref(regression_path)))
        result_body = {
            **common(evidence, "FINAL_SCIENTIFIC_CLOSURE_NEGATIVE"),
            "decision": DECISION,
            "primary_question_answer": "No safe nonvacuous globally causal identity intervention was demonstrated on the frozen fresh development stream.",
            "development_group_counts": {"registered": 17, "complete_full_MOT": 16, "terminal_no_fit": 1},
            "main_cells": 1344,
            "state_cells": 192,
            "complete_groups": evidence["complete_groups"],
            "terminal_groups": evidence["terminal_groups"],
            "gates": {"G0": "ENGINEERING_EVIDENCE_RECORDED", "G1": "FAIL_ACROSS_COMPLETE_GROUPS",
                      "G2": "FAIL_ACROSS_COMPLETE_GROUPS", "G3": "NOT_RUN_CONFIRMATION_UNAUTHORIZED",
                      "G4": "FAIL_OR_UNRESOLVED_MEMORY_SAFETY"},
            "all_complete_group_effects": evidence["groups"],
            "memory": {"M_A": "EXECUTED_NECESSARY_CONDITIONS_FAIL", "M_B": "NOT_RUN_UNAUTHORIZED",
                       "population_safety": False},
            "confirmation": {"status": "NOT_RUN_UNAUTHORIZED", "sequences": evidence["preregistration"]["split"]["confirmation"]},
            "split_integrity": {"frozen_train_roles": 40, "FIT16": 16, "INNER8": 8, "CONFIRMATION8_accessed": False},
            "test_regression": {"path": str(regression_path), "sha256": sha256(regression_path),
                                 "passed": test_receipt["passed"], "failed": test_receipt["failed"],
                                 "exit_code": test_receipt["actual_shell_exit_code"]},
            "generated_reports": report_refs,
            "final_report": ref(report_path),
            "research_closure_complete": True,
            "whole_task_complete": False,
            "final_tables_delivered": True,
            "required35_full_semantics_complete": False,
            "scientific_success": False,
            "next_stage_authorized": False,
        }
        result_path = write_json("FINAL_RESULT.json", result_body)
        update_status(
            status="FINAL_RESULT_WRITTEN_NEGATIVE_DEVELOPMENT_CLOSURE",
            scientific_decision=DECISION,
            scientific_success=False,
            application_goal_status="ACTIVE_NEGATIVE_RESEARCH_CLOSURE_FULL_TASK_AUDIT_UNPROVEN",
            application_goal_active=True,
            actual_complete_full_MOT_groups=16,
            actual_terminal_no_fit_groups=1,
            confirmation_authorized=False,
            confirmation_policy_runs_completed=0,
            VAL_TEST_SOT_runs_completed=0,
            next_association_stage_authorized=False,
            next_stage_authorized=False,
            required_final_tables_complete=True,
            required35_full_semantics_complete=False,
            latest_final_result={"path": str(result_path), "sha256": sha256(result_path)},
            latest_final_report=ref(report_path),
        )
        append_log("FINAL_SCIENTIFIC_CLOSURE_NEGATIVE", decision=DECISION, complete_groups=16,
                   terminal_no_fit_groups=1, confirmation_authorized=False)
    print({"dry_run": dry_run, "decision": DECISION, "registered_groups": 17,
           "complete_full_MOT_groups": 16, "terminal_no_fit_groups": 1,
           "confirmation_authorized": False})


def reconcile():
    """Reconcile the already-written result with the requirement ledger."""
    storage(8 << 20)
    result_path = OUT / "FINAL_RESULT.json"
    ledger_paths = sorted((OUT / "audit").glob("FULL_TASK_DELIVERABLE_LEDGER_V1*.json"))
    regression_path = latest_versioned(OUT / "tests", "REGRESSION")
    tables_path = latest_versioned(OUT / "tables", "FINAL_SIX_TABLES")
    required35_path = latest_versioned(OUT / "audit", "REQUIRED35_RUNTIME_SCOPE_AUDIT")
    report_path = OUT / "FINAL_REPORT.md"
    if not result_path.exists() or not ledger_paths or not tables_path.exists() or not required35_path.exists():
        raise FileNotFoundError("Final result, requirement ledger, six tables, and required35 audit are all required")
    result = read_json(result_path)
    ledger_path = ledger_paths[-1]
    tables = read_json(tables_path)
    required35 = read_json(required35_path)
    if not tables["all_six_tables_accounted"]:
        raise ValueError("six-table assembler did not account for every required table")
    if not required35["all_requirements_mapped"]:
        raise ValueError("required35 assembler did not map every engineering requirement")
    if report_path.exists():
        report = report_path.read_text()
        marker = "## Final delivery audit"
        if marker not in report:
            report += """

## Final delivery audit

The six required tables are now assembled in
`outputs/N72R21R2/tables/FINAL_SIX_TABLES_V1.json`.  Every required field is
represented by an actual measurement or an explicit `UNPROVEN`/`NOT_RUN`
disposition; no post-hoc best strategy was selected after observing the
negative result.  Table 1 keeps independent-person and independent-causal-root
fields unavailable rather than converting correlated frames, clicks or seeds
into independent evidence.

The methods actually explored include KEEP/SHADOW, recovery-only,
baseline-uncertainty, global-regret, competitor-protection, native-reliability,
delayed confirmation and selective abstention controls, plus Scalar, Logistic,
Small MLP, Pairwise Ranker, Global Risk, Temporal, Value and Dual-State learned
families across the registered objectives.  The complete learned full-MOT
groups were C0-equivalent at runtime: zero effective actions, zero N01/N10 and
zero paired nine-metric deltas.  Simple controls did produce diagnostic actions
in FIT/INNER, but the observed N01/N10 and other-person harm evidence failed G1
and their paired HOTA/AssA/IDF1 evidence failed G2; those actions are not a
qualified correction policy.  Relational structure, model-generated on-policy
training, M-B memory integration and CONFIRMATION8 were conditionally NOT_RUN
because no earlier branch passed the preregistered gates.

The 35 engineering requirements have a source-attested focused mapping in
`outputs/N72R21R2/audit/REQUIRED35_RUNTIME_SCOPE_AUDIT_V1.json`; the focused
receipt passed, while the complete regression remains 1582 passed / 5 retained
historical failures.  The audit explicitly records the scope limits of unit and
synthetic checks, so it is not a claim of positive full-MOT scientific success.
The most promising next direction is a provenance-complete causal event tape
with nonvacuous real global-assignment opportunities and an independently
verified candidate/association interface, rather than increasing model size or
running confirmation on a zero-action policy.
"""
        for directory, prefix, current in (
            (OUT / "tests", "REGRESSION", regression_path),
            (OUT / "tables", "FINAL_SIX_TABLES", tables_path),
            (OUT / "audit", "REQUIRED35_RUNTIME_SCOPE_AUDIT", required35_path),
        ):
            for old_path in Path(directory).glob(f"{prefix}_V*.json"):
                report = report.replace(str(old_path.relative_to(ROOT)), str(current.relative_to(ROOT)))
        if report != report_path.read_text():
            report_path.write_text(report)
    regression = read_json(regression_path)
    result.update({
        "research_closure_complete": True,
        "final_tables_delivered": True,
        "six_final_tables_complete": True,
        "final_tables": ref(tables_path),
        "required35_regression_coverage_complete": True,
        "required35_runtime_scope_audit": ref(required35_path),
        "required35_full_semantics_complete": False,
        "whole_task_complete": False,
        "full_task_completion_proven": False,
        "requirement_ledger": ref(ledger_path),
        "test_regression": {"path": str(regression_path), "sha256": sha256(regression_path),
                            "passed": regression["passed"], "failed": regression["failed"],
                            "exit_code": regression["actual_shell_exit_code"]},
    })
    if report_path.exists():
        result["final_report"] = ref(report_path)
    write_json("FINAL_RESULT.json", result, mutable=True)
    update_status(
        application_goal_status="ACTIVE_NEGATIVE_RESEARCH_CLOSURE_FULL_TASK_AUDIT_UNPROVEN",
        application_goal_active=True,
        scientific_decision=DECISION,
        scientific_success=False,
        next_association_stage_authorized=False,
        required_final_tables_complete=True,
        six_final_tables_complete=True,
        required35_regression_coverage_complete=True,
        required35_full_semantics_complete=False,
        whole_task_completion_proven=False,
        latest_final_tables=ref(tables_path),
        latest_required35_runtime_scope_audit=ref(required35_path),
        latest_full_regression=ref(regression_path),
        latest_final_report=ref(report_path) if report_path.exists() else None,
        latest_full_task_deliverable_ledger=ref(ledger_path),
    )
    append_log("FINAL_RESULT_RECONCILED_WITH_TABLES_AND_REQUIRED35_AUDIT", ledger=ref(ledger_path), tables=ref(tables_path), required35=ref(required35_path), whole_task_complete=False)
    print({"reconciled": True, "whole_task_complete": False, "six_final_tables_complete": True, "required35_regression_coverage_complete": True, "ledger": str(ledger_path), "tables": str(tables_path), "required35": str(required35_path)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reconcile", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.reconcile:
        raise SystemExit("--dry-run and --reconcile are mutually exclusive")
    reconcile() if args.reconcile else run(args.dry_run)
