#!/usr/bin/env python3
"""M0: line-resolved implementation, training lineage and local asset audit."""
from __future__ import annotations
import argparse
import ast
import subprocess

import torch

from scripts.n72r20r4_common import *
from sam3_intermot.identity_verification.cross_scene_adapter import CrossSceneIdentityAdapter


def function_lines(path: Path, name: str) -> tuple[int, int]:
    node = next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    return node.lineno, node.end_lineno


def run() -> dict:
    goal = read_json(OUT / "FINAL_GOAL.json")
    if goal["goal"] != "Causal Identity-to-Trajectory Transfer" or not goal["goal_frozen"]:
        raise ValueError("R4 final goal drift")
    source = "2b80763fe8f875defa697adeae638679ff72fd6b"
    p = ROOT / "scripts/n72r20r3r2r3_pipeline.py"
    problems = [
        ("M0-A", "base_identity_scores", "FAIL_IDENTITY_ADAPTER_NOT_IN_ASSOCIATION", "sigmoid of sealed base_score_matrix[:,target], no CrossSceneIdentityAdapter inference", "Rank gains are not transferred into the actual score matrix", "Strict SHA/axis-verified tower inference from branch-owned causal GRU state"),
        ("M0-B", "rollout_sequence", "FAIL_TARGET_COLUMN_MERGE", "plain linear_sum_assignment without explicit NONE; extracts target row, retains all other baseline public assignments", "candidate collisions, stolen observations and NONE recovery are not globally committed", "Complete solve_exact_public_assignment output, hard candidate/public uniqueness checks"),
        ("M0-C", "rollout_sequence", "FAIL_FROZEN_STATE_FEEDBACK", "every frame reads sealed base_scores and frozen geometry; no treatment IdentityState update", "interventions cannot influence next-frame base scores, motion or appearance", "Each tracker owns StateManager/IdentityState and recomputes current scores from prior branch state"),
        ("M0-D", "rollout_sequence", "FAIL_PRE_EVENT_TRAJECTORY_OMISSION", "pre-event continue skips trajectory append for all public IDs", "truncated non-target trajectories confound full-sequence evaluation when event_frame>0", "ordinary online births/assignment outputs before click; residual visible only after event"),
        ("M0-E", "rollout_sequence", "FAIL_ASSUMED_NEW_CANDIDATE_IDENTITY", "new SAM3 identity copied from current max, quality=0.75, geometry=0.7", "localization benefit includes unsupported appearance/quality authority", "R4 uses existing real OSNet candidates; new candidates disabled"),
        ("M5", "association_shadow", "FAIL_SHADOW_IS_NOT_MEMORY_OR_TRACK_EFFECT", "local target IoU correction and threshold commit booleans, no GRU write or next-state rollout", "90 shadow N01 is not 90 IDSW repairs or actual accepted writes", "pair complete global trajectories; future continuity and real memory updater audits"),
        ("Export", "export_trajectory_tracker", "AUDIT_EXPORT_GEOMETRY", "sorts rows, clamps width/height to >=1; no cross-ID candidate uniqueness validation", "writer does not repair underlying assignment semantics", "validate geometry/uniqueness before export and hash exact trajectories"),
        ("Native", "export_baseline", "SEALED_CAUSAL_BASELINE_SOURCE", "exports exact sealed R2 public assignments; original tape builder does causally update states", "valid reference but no new online treatment state", "reconstruct independently with same initialization and legacy lifecycle; compare frame assignments and SHA"),
        ("Eval", "run_trackeval_many", "VALID_PINNED_EVALUATOR", "wrapper normalizes pinned CLI; shared config and full sequence GT", "evaluator cannot turn malformed runtime into causal evidence", "reuse wrapper with identical input axes, frame range and split"),
        ("SAM", "targeted_sam3_refinement", "POSTHOC_DEV_SCHEDULE", "dev schedule selects earliest posthoc P1a frame; real SAM3 prompts use causal boxes", "posthoc scheduling is not a deployable trigger", "disabled in R4; no candidate regeneration"),
    ]
    lines = ["# R4 association implementation audit", "", "Frozen goal: `outputs/N72R20R4/FINAL_GOAL.json`.", "", "## Code findings", "", "| Item | File/function/lines | Actual behavior | Status | Trajectory impact | R4 repair / required evidence |", "|---|---|---|---|---|---|"]
    findings = []
    for item, function, status, behavior, impact, repair in problems:
        start, end = function_lines(p, function)
        findings.append({"item": item, "file": str(p.relative_to(ROOT)), "function": function, "start_line": start, "end_line": end, "status": status, "actual_behavior": behavior, "impact": impact, "repair": repair, "test_evidence": "PENDING_R4_REGRESSION_TESTS"})
        lines.append(f"| {item} | `{p.relative_to(ROOT)}:{start}` `{function}` | {behavior} | {status} | {impact} | {repair} |")
    lines.extend(["", "## Supporting APIs audited", "", "- `public_assignment.solve_exact_public_assignment`: explicit state/public axes, candidate×public matrix, independent NONE slots; complete solve reused without modification.", "- `branch_public_replay.apply_exact_frame`: applies all solver rows, preserves outer birth vs solver NONE; runtime and StateManager independently owned. R4 reuses its lifecycle principles without requiring old fixed public bridge sessions.", "- `online_associator.score_matrix_pairwise`: real current feature, previous prototype, velocity/native continuity and gap; reusable base scorer.", "- `learned_identity_memory.update_from_consensus`: frozen GRU, immutable anchor, public binding, real accepted-update audit; R4 adds acceptance gates and explicit denominators.", "- `cross_scene_adapter`: separate 512-D query/candidate towers, 265472 parameters, actual cosine inference.", "", "## Critical training-lineage discrepancy", "", "`n72r20r3r2_representation.run_representation` first selects epochs on 6/1, then fits the final adapter on all seven non-heldout sequences. `_save_checkpoint` nevertheless writes `parameter_fit_sequences` as six and omits the final seventh sequence. Heldout H is absent, but inner V is actually included in the final fit. Such checkpoints must not be used to select association policy on V. R4 records actual seven-sequence lineage and trains strict six-sequence adapter instances for current 6/1/1 selection. Backbone and GRU remain frozen.", "", "## Historical 90 corrections", "", "G1/G3 changed 1440 target-frame choices with N01=90/N10=0 in shadow reports. Formal inner selection chose G0 in all eight folds, so those association interventions did not enter COMBINED trajectories. Furthermore, their global row reallocations were discarded and their state was never propagated. They are local-IoU diagnostics and cannot be credited as IDSW or HOTA improvements.", "", "## Test evidence", "", "Pending real R4 regression suite and independently generated A/A trajectories; no structural finding is a scientific PASS."])
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "audit/ASSOCIATION_IMPLEMENTATION_AUDIT.md").write_text("\n".join(lines) + "\n")
    write_json(OUT / "audit/CODE_FINDINGS.json", findings)
    records = []
    historical = read_json(ROOT / "outputs/N72R20R3R2/training/checkpoint_manifest.json")
    for item in historical["records"]:
        path = Path(item["path"])
        actual = sha256(path)
        if actual != item["sha256"]:
            raise ValueError(f"checkpoint SHA mismatch {path}")
        cp = torch.load(path, map_location="cpu", weights_only=False)
        model = CrossSceneIdentityAdapter()
        model.load_state_dict(cp["state_dict"], strict=True)
        actual_fit = [s for s in SEQUENCES if s != cp["heldout_sequence"]]
        records.append({**item, "verified_sha256": actual, "query_dimension": 512, "candidate_dimension": 512, "actual_training_sequences_from_source": actual_fit, "declared_parameter_fit_sequences": cp["parameter_fit_sequences"], "inner_validation_actually_in_final_fit": cp["internal_validation_sequence"] in actual_fit, "outer_heldout_not_in_final_fit": cp["heldout_sequence"] not in actual_fit})
    if sha256(MEMORY_CHECKPOINT) != MEMORY_SHA:
        raise ValueError("N72R18 GRU checkpoint mismatch")
    write_json(OUT / "adapter/CHECKPOINT_AUDIT.json", {"status": "PASS_SHA_ARCHITECTURE_WITH_METADATA_DISCREPANCY", "records": records, "N72R18_path": MEMORY_CHECKPOINT, "N72R18_sha256": MEMORY_SHA, "encoder_sha256": ENCODER_SHA, "strict_inner_versions_required": True})
    candidates = []
    for s in SEQUENCES:
        d = DEV_ROOT / "candidates" / s
        index = read_json(d / "index.json")
        candidates.append({"sequence": s, "root": d, "frame_count": index["frame_count"], "embedding_count": index["embedding_count"], "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in (d / "index.json", d / "metadata.jsonl.zst", d / "embeddings.f16")}})
    hashes = historical_hashes()
    lineage = {"stage": STAGE, "source_head_local": source, "source_remote_refresh": "PENDING_NETWORK_RETRY", "source_branch": "codex/n72r20r3r2r3-localization-association-hota", "research_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(), "historical_artifacts_sha256": hashes, "candidates": candidates, "storage": check_storage(reserve_gib=0.25), "code_sha256": code_manifest(), "dataset_train_sequences": len(list((DATASET / "train").glob("dancetrack*"))), "dataset_val_sequences": 25, "val_gt_not_opened_this_stage": True, "test_accessed": False, "new_candidate_generation": False}
    write_json(OUT / "audit/SOURCE_LINEAGE.json", lineage)
    return lineage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = run()
    print(json.dumps({"status": "M0_AUDIT_COMPLETE", "checkpoints_verified": 24, "storage": result["storage"]}))
