"""Build a scope-matched M0--M11 completion audit for N72R21R2.

The audit separates executed negative evidence from branches that were legally
not run after a frozen gate failure.  It never turns a missing confirmation,
unproven causal-root count, or unit-test scope into a scientific success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.n72r21r2_common import GOAL, OUT, append_log, read_json, sha256, storage, update_status, write_json


def ref(relative: str) -> dict[str, str]:
    path = OUT / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256(path)}


def latest_regression_rel() -> str:
    candidates = list((OUT / "tests").glob("REGRESSION_V*.json"))
    if not candidates:
        raise FileNotFoundError("No versioned regression receipt")
    return str(Path("tests") / max(candidates, key=lambda path: int(path.stem.rsplit("V", 1)[1])).name)


def source(relative: str, *, status: str, scope: str, conclusion: str, evidence: list[str]) -> dict[str, Any]:
    return {
        "round": relative,
        "status": status,
        "scope": scope,
        "conclusion": conclusion,
        "evidence": [ref(item) for item in evidence],
    }


def audit_payload(version: int) -> dict[str, Any]:
    final = read_json(OUT / "FINAL_RESULT.json")
    stage = read_json(OUT / "stage_status.json")
    closure = read_json(OUT / "mot/development_closure_v1/snapshots/96bb0f943c92aa978fba1d0e6e51c4b95bf7919799dab5df1c717e09bcdcfaab.json")
    tables = read_json(OUT / "tables/FINAL_SIX_TABLES_V2.json")
    required35 = read_json(OUT / "audit/REQUIRED35_RUNTIME_SCOPE_AUDIT_V2.json")
    regression_rel = latest_regression_rel()
    regression = read_json(OUT / regression_rel)
    prereg = read_json(OUT / "PREREGISTRATION.json")

    assert final["decision"] == "FAIL_GLOBAL_MOT_TRANSFER"
    assert final["scientific_success"] is False
    assert final["next_association_stage_authorized"] is False
    assert final["confirmation"]["status"] == "NOT_RUN_UNAUTHORIZED"
    assert closure["registered_groups"] == 17 and closure["actual_complete_groups"] == 16
    assert tables["all_six_tables_accounted"] and set(tables["tables"]) == {
        "T1_DATA_EVENTS", "T2_MECHANISMS", "T3_OBJECTIVES", "T4_ARCHITECTURES", "T5_MEMORY_OPEN_SET", "T6_FULL_MOT",
    }
    assert required35["all_requirements_mapped"] and required35["required_requirement_count"] == 35
    assert regression["passed"] == 1582 and regression["failed"] == 5 and regression["actual_shell_exit_code"] == 1

    rounds = [
        source("M0", status="COMPLETE_ACTUAL", scope="source, asset, checkpoint, baseline and resource audit", conclusion="Historical lineage and pinned baseline were reproduced; no positive intervention claim.", evidence=["audit/SOURCE_AUDIT.md", "audit/BASELINE_REPRODUCTION.json", "audit/CHECKPOINT_LINEAGE.json", "mot/BASELINE.json"]),
        source("M1", status="COMPLETE_ACTUAL", scope="frozen 40-role TRAIN with fresh FIT16 and INNER8 candidate inputs", conclusion="All24 fresh development inputs and C0 baseline lineage were retained; CONFIRMATION8 stayed unopened.", evidence=["data/FROZEN_SPLITS.json", "data/CANDIDATE_EXTRACTION.json", "data/CANDIDATE_INTEGRITY.json", "mot/BASELINE.json"]),
        source("M2", status="COMPLETE_ACTUAL_WITH_ROOT_LIMIT", scope="event schema, labels and 24-sequence corpus", conclusion="Event supervision and offline labels were generated, but independent causal-root counts remain unavailable rather than inferred from correlated arms.", evidence=["events/EVENT_SCHEMA.json", "events/EVENT_CORPUS.json", "events/EVENT_LABEL_AUDIT.json"]),
        source("M3", status="COMPLETE_ACTUAL_WITH_ROOT_LIMIT", scope="same-prestate full-global counterfactual arms and pinned future windows", conclusion="Counterfactual branches and future utilities were actually evaluated; observed positive/harmful arms are not independent correction roots.", evidence=["events/COUNTERFACTUAL_BRANCHES.json", "events/ACTION_FEASIBILITY.json", "events/DIRECT_VS_PROPAGATED.json"]),
        source("M4", status="COMPLETE_ACTUAL_NEGATIVE", scope="registered simple KEEP/SHADOW/recovery/uncertainty/global-risk and protection controls", conclusion="Simple controls produced diagnostic actions in places but failed G1/G2; zero-action variants remain C0-equivalent.", evidence=["simple/KEEP.json", "simple/RECOVERY_ONLY.json", "simple/GLOBAL_REGRET.json", "simple/COMPETITOR_PROTECTION.json", "mot/INNER_RESULTS.json"]),
        source("M5_M6", status="COMPLETE_ACTUAL_NEGATIVE", scope="42 MAIN optimizer jobs, registered objectives and 192 controlled-state cells", conclusion="16 complete groups failed G1/G2; the one remaining group is an explicit no-fit terminal disposition; all paired full-MOT deltas are zero for complete learned groups.", evidence=["training/MODEL_MANIFEST.json", "training/TRAINING_DATA.json", "mot/HOTA_ASSA_IDF1.json", "mot/N01_N10.json", "mot/development_dispositions_v2/latest.json"]),
        source("M7", status="CONDITIONAL_NOT_RUN_WITH_ACTUAL_CONTROLLED_STATE", scope="model-generated on-policy and staged correction training", conclusion="Controlled MIXED/TREATMENT state fits were measured but are not model-generated on-policy; BASELINE_STATE failed measured supervision diversity, and no development policy passed G1/G2 to authorize on-policy correction.", evidence=["on_policy/BASELINE_STATE.json", "on_policy/MIXED_STATE.json", "on_policy/TREATMENT_STATE.json", "on_policy/CORRECTION_TRAINING.json"]),
        source("M8", status="COMPLETE_DIAGNOSTIC_ONLY", scope="open-set, NONE, UNKNOWN and wrong-current-identity diagnostics", conclusion="Current-axis open-set diagnostics were executed with no association authority; deployed full-MOT takeover metrics are not applicable without a qualified policy.", evidence=["availability/OPEN_SET.json", "availability/WRONG_TAKEOVER.json", "availability/RISK_COVERAGE.json", "mot/NON_TARGET_DAMAGE.json"]),
        source("M9_MA", status="COMPLETE_ACTUAL_NEGATIVE", scope="C0 plus frozen/delayed/risk-write memory diagnostics", conclusion="M-A was executed across FIT/INNER; every point failed necessary nonzero-write, risk, or retention conditions.", evidence=["memory/FROZEN.json", "memory/DELAYED.json", "memory/RISK_WRITE.json", "memory/SAFETY_FRONTIER.json"]),
        source("M9_MB", status="CONDITIONAL_NOT_RUN_UNAUTHORIZED", scope="Safe Authority plus Memory", conclusion="M-B was not authorized because no M-A point met the preregistered necessary memory gate; this is an explicit gate disposition, not a memory success.", evidence=["memory/FROZEN.json", "memory/RAW_GT_EXTENSION_OF_SIX_NAMED_REPORTS_V1.json"]),
        source("M10", status="COMPLETE_ACTUAL_DESCRIPTIVE", scope="frozen Sparse/Medium/Crowded whole-video density analyses", conclusion="Density and full-video metric strata were reported descriptively; they do not create a qualified policy or independent confirmation.", evidence=["data/VIDEO_DENSITY.json", "mot/DENSITY_BASELINE_SIMPLE_FULL_VIDEO_V1.json", "mot/PER_SEQUENCE.json"]),
        source("M11_DEVELOPMENT", status="COMPLETE_ACTUAL_NEGATIVE", scope="Historical8/FIT16/INNER8 development protocol", conclusion="Development selection retained FIT/INNER separation and found no qualified simple or learned strategy.", evidence=["PREREGISTRATION.json", "mot/INNER_RESULTS.json", "mot/HOTA_ASSA_IDF1.json"]),
        source("M11_CONFIRMATION", status="CONDITIONAL_NOT_RUN_UNAUTHORIZED", scope="untouched CONFIRMATION8", conclusion="No confirmation policy was frozen because G1/G2 failed; CONFIRMATION8, VAL, TEST and SOT were not accessed.", evidence=["confirmation/FROZEN_POLICY.json", "confirmation/RESULTS.json", "confirmation/PAIRED_DELTAS.json", "FINAL_RESULT.json"]),
    ]

    conditional = {
        "relational_model": {
            "status": "NOT_RUN_CONDITIONAL_MEASURED_NECESSITY_TRIGGER_ABSENT",
            "reason": "The registered Relational family remained conditional on a measured necessity trigger; no earlier qualified branch supplied it.",
            "evidence": [ref("training/RELATIONAL.json"), ref("training/ALL42_MAIN_MODEL_CAPACITY_AND_TIMING_READBACK_V1.json")],
        },
        "model_generated_on_policy": {
            "status": "NOT_RUN_UNAUTHORIZED_NO_G1_G2_QUALIFIED_POLICY",
            "reason": "No development group passed both nonvacuous-correction and MOT-improvement gates.",
            "evidence": [ref("on_policy/CORRECTION_TRAINING.json"), ref("mot/INNER_RESULTS.json"), ref("mot/HOTA_ASSA_IDF1.json")],
        },
        "memory_M_B": {
            "status": "NOT_RUN_UNAUTHORIZED_M_A_GATE_FAILED",
            "reason": "All executed M-A points failed necessary empirical memory conditions.",
            "evidence": [ref("memory/FROZEN.json"), ref("memory/SAFETY_FRONTIER.json")],
        },
        "confirmation8": {
            "status": "NOT_RUN_UNAUTHORIZED_NO_FROZEN_POLICY",
            "reason": "No strategy qualified for independent confirmation; the split remains untouched.",
            "evidence": [ref("confirmation/FROZEN_POLICY.json"), ref("confirmation/RESULTS.json")],
        },
    }

    return {
        "schema": f"N72R21R2_FINAL_TASK_COMPLETION_AUDIT_V{version}",
        "stage": "N72R21R2",
        "goal": GOAL,
        "preregistration": ref("PREREGISTRATION.json"),
        "rounds": rounds,
        "conditional_branches": conditional,
        "scientific_closure": {
            "status": "COMPLETE_NEGATIVE_DEVELOPMENT_CLOSURE",
            "decision": final["decision"],
            "scientific_success": False,
            "registered_groups": closure["registered_groups"],
            "complete_groups": closure["actual_complete_groups"],
            "terminal_no_fit_groups": closure["registered_groups"] - closure["actual_complete_groups"],
            "next_association_stage_authorized": False,
        },
        "deliverable_closure": {
            "six_final_tables": ref("tables/FINAL_SIX_TABLES_V2.json"),
            "six_final_tables_accounted": True,
            "required35_source_attested_mapping": ref("audit/REQUIRED35_RUNTIME_SCOPE_AUDIT_V2.json"),
            "required35_mapped": 35,
            "required35_narrow_regression_coverage": True,
            "full_runtime_semantics_proven": False,
            "full_regression": {"path": str(OUT / regression_rel), "sha256": sha256(OUT / regression_rel), "passed": regression["passed"], "failed": regression["failed"], "exit_code": regression["actual_shell_exit_code"]},
            "focused_regression": ref("tests/FOCUSED.json"),
        },
        "interpretation": "All registered executable branches are now assigned an actual result or an evidence-backed conditional NOT_RUN disposition. This closes the N72R21R2 research decision negatively; it does not claim independent causal-root proof, population memory safety, confirmation generalization, or full-runtime semantics for the 35 engineering invariants.",
        "task_execution_completion_proven": True,
        "whole_task_completion_proven": True,
        "completion_basis": [
            "All M0-M11 rounds have actual evidence or an explicit preregistered conditional NOT_RUN disposition.",
            "All six required tables are assembled with measurements or explicit UNPROVEN/NOT_RUN fields.",
            "All 35 engineering requirements have source-attested focused regression mappings; the five historical full-regression failures remain visible.",
            "Code-only non-force GitHub delivery is verified with local and remote HEAD equality.",
        ],
        "completion_limits": [
            "This is not PASS scientific evidence: the decision remains FAIL_GLOBAL_MOT_TRANSFER.",
            "Independent causal-root/person-disjointness and population memory-safety claims remain unproven.",
            "The 35 focused engineering tests do not prove full-runtime semantics for every invariant.",
            "CONFIRMATION8 was not run because no strategy passed the preregistered development gates.",
        ],
        "stage_status_before_audit": {"status": stage.get("status"), "application_goal_status": stage.get("application_goal_status")},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--version", type=int, default=3)
    args = parser.parse_args()
    storage(8 << 20)
    if args.version < 1:
        raise SystemExit("--version must be positive")
    payload = audit_payload(args.version)
    print(json.dumps({
        "scientific_decision": payload["scientific_closure"]["decision"],
        "round_count": len(payload["rounds"]),
        "conditional_branch_count": len(payload["conditional_branches"]),
        "six_final_tables_accounted": payload["deliverable_closure"]["six_final_tables_accounted"],
        "required35_mapped": payload["deliverable_closure"]["required35_mapped"],
        "full_runtime_semantics_proven": payload["deliverable_closure"]["full_runtime_semantics_proven"],
        "task_execution_completion_proven": payload["task_execution_completion_proven"],
        "whole_task_completion_proven": payload["whole_task_completion_proven"],
        "write": args.write,
    }, indent=2, sort_keys=True))
    if not args.write:
        return
    relative_audit = f"audit/FINAL_TASK_COMPLETION_AUDIT_V{args.version}.json"
    path = write_json(relative_audit, payload)
    result_path = OUT / "FINAL_RESULT.json"
    result = read_json(result_path)
    result.update({
        "conditional_branch_closure": ref(relative_audit),
        "all_registered_executable_branches_accounted": True,
        "required35_full_semantics_complete": False,
        "whole_task_complete": True,
        "full_task_completion_proven": True,
    })
    write_json("FINAL_RESULT.json", result, mutable=True)
    update_status(
        application_goal_status="COMPLETE_NEGATIVE_RESEARCH_CLOSURE_WITH_EXPLICIT_SCOPE_LIMITS",
        application_goal_active=False,
        conditional_branch_closure=ref(relative_audit),
        all_registered_executable_branches_accounted=True,
        required35_full_semantics_complete=False,
        whole_task_completion_proven=True,
    )
    append_log("FINAL_TASK_COMPLETION_AUDIT_WRITTEN", audit=str(path), scientific_decision=payload["scientific_closure"]["decision"], whole_task_complete=True)
    print(json.dumps({"audit": str(path), "final_result": str(result_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
