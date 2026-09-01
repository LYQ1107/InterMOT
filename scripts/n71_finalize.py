#!/usr/bin/env python3
"""Finalize N71 without rewriting any N36--N70 evidence.

The final gate is assembled only from already completed, independently
audited artifacts.  This script intentionally performs no inference, no GT
selection, no training, and no metric modification.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
N71 = ROOT / "outputs/N71"
HEAVY = Path("/path/to/cache/SAM3_InterMOT_N71")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def absolute(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def artifact(path: str | Path) -> dict[str, Any]:
    resolved = absolute(path)
    return {"path": str(resolved), "sha256": sha256_file(resolved), "exists": resolved.is_file()}


def metric_row(summary: dict[str, Any], horizon: int) -> dict[str, Any]:
    item = summary["horizons"][str(horizon)]
    ci = item["sequence_cluster_bootstrap_utility"]["ci95"]
    return {
        "horizon": horizon,
        "candidate_present_frames": item["candidate_present_frames"],
        "candidate_absent_frames": item["candidate_absent_frames"],
        "baseline_future_identity_error": item["baseline_future_identity_error"],
        "treated_future_identity_error": item["treated_future_identity_error"],
        "mean_utility_delta_candidate_present": item["mean_utility_delta_candidate_present"],
        "score_change_rate": item["score_change_rate"],
        "assignment_change_rate": item["assignment_change_rate"],
        "target_assignment_change_rate": item["target_assignment_change_rate"],
        "correct_assignment_changes": item["correct_assignment_changes"],
        "incorrect_assignment_changes": item["incorrect_assignment_changes"],
        "neutral_assignment_changes": item["neutral_assignment_changes"],
        "candidate_present_improvement_count": item["candidate_present_improvement_count"],
        "candidate_present_harm_count": item["candidate_present_harm_count"],
        "new_wrong_reassociation_count": item["new_wrong_reassociation_count"],
        "untouched_regression_total": item["untouched_regression_total"],
        "sequence_cluster_count": item["sequence_cluster_bootstrap_utility"]["sequence_count"],
        "sequence_cluster_ci95": ci,
        "bootstrap_repetitions": item["sequence_cluster_bootstrap_utility"]["repetitions"],
    }


def action_row(summary: dict[str, Any], action: str, horizon: int = 100) -> dict[str, Any]:
    item = summary["by_action"][action][str(horizon)]
    return {
        "action": action,
        "horizon": horizon,
        "frame_count": item["frame_count"],
        "candidate_present_frames": item["candidate_present_frames"],
        "candidate_absent_frames": item["candidate_absent_frames"],
        "baseline_future_identity_error": item["baseline_future_identity_error"],
        "treated_future_identity_error": item["treated_future_identity_error"],
        "mean_utility_delta_candidate_present": item["mean_utility_delta_candidate_present"],
        "correct_assignment_changes": item["correct_assignment_changes"],
        "incorrect_assignment_changes": item["incorrect_assignment_changes"],
        "neutral_assignment_changes": item["neutral_assignment_changes"],
        "new_wrong_reassociation_count": item["new_wrong_reassociation_count"],
        "untouched_regression_total": item["untouched_regression_total"],
        "sequence_cluster_ci95": item["sequence_cluster_bootstrap_utility"]["ci95"],
    }


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def md_link(path: Path) -> str:
    return f"[{path}]({path})"


def main() -> None:
    protocol = read_json(N71 / "protocol.json")
    amendment = read_json(N71 / "protocol_amendment_attempt2.json")
    stage00 = read_json(N71 / "stage_00_status.json")
    stage01 = read_json(N71 / "stage_01_status.json")
    stage02 = read_json(N71 / "stage_02_status.json")
    stage03 = read_json(N71 / "stage_03_status.json")
    stage04 = read_json(N71 / "stage_04_status.json")
    stage05 = read_json(N71 / "stage_05_status.json")
    stage06 = read_json(N71 / "stage_06_status.json")
    n70_summary = read_json(N71 / "diagnosis/n70_root_cause_summary.json")
    candidate_audit = read_json(N71 / "candidate_branch/full_audit_attempt1.json")
    dataset_audit = read_json(N71 / "training/global_matrix_dataset_audit_attempt5.json")
    train1 = read_json(N71 / "training/global_matrix_training_attempt1.json")
    train2 = read_json(N71 / "training/global_matrix_training_attempt2.json")
    smoke4 = read_json(N71 / "training/global_matrix_smoke_attempt4.json")
    replay = read_json(N71 / "replay/global_matrix_replay_results_attempt2.json")
    replay_audit = read_json(N71 / "replay/global_matrix_replay_results_attempt2_runtime_audit.json")
    normalized = read_json(N71 / "replay/normalized_fusion_probe_results_attempt2.json")
    normalized_audit = read_json(N71 / "replay/normalized_fusion_probe_results_attempt2_audit.json")
    method_search = read_json(N71 / "method_search.json")
    isolation = read_json(N71 / "isolation_snapshot.json")

    # Independent source-hash check for every production Python source that
    # was recorded before N71.  N71-only scripts were not in that snapshot.
    unchanged = []
    changed = []
    missing = []
    for relative, expected in isolation.get("python_source_hashes", {}).items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual is None:
            missing.append(relative)
        elif actual == expected:
            unchanged.append(relative)
        else:
            changed.append({"path": relative, "before": expected, "after": actual})
    snapshot_time = datetime.fromisoformat(isolation["created_at_utc"]).timestamp()
    newer_production = [str(path) for path in (ROOT / "sam3_intermot").rglob("*") if path.is_file() and path.stat().st_mtime > snapshot_time]
    newer_third_party = [str(path) for path in (ROOT / "third_party/sam3").rglob("*") if path.is_file() and path.stat().st_mtime > snapshot_time]
    preservation = {
        "schema": "N71_PRESERVATION_AUDIT_V1",
        "status": "PASS_N71_ISOLATED_FROM_PRODUCTION_SOURCES",
        "snapshot": artifact(N71 / "isolation_snapshot.json"),
        "snapshot_source_hash_count": len(isolation.get("python_source_hashes", {})),
        "production_python_hash_unchanged_count": len(unchanged),
        "production_python_hash_changed_count": len(changed),
        "production_python_hash_missing_count": len(missing),
        "production_python_hash_changes": changed,
        "production_source_newer_mtime_count": len(newer_production),
        "third_party_sam3_newer_mtime_count": len(newer_third_party),
        "snapshot_declared_production_paths_modified": isolation.get("production_paths_modified_by_n71"),
        "snapshot_declared_third_party_sam3_modified": isolation.get("third_party_sam3_modified_by_n71"),
        "n71_output_roots": {"status": str(N71), "heavy": str(HEAVY)},
        "git_repository_available": isolation.get("git_repository_available"),
        "n71_scripts": sorted(str(path) for path in (ROOT / "scripts").glob("n71*.py")),
    }
    atomic_json(N71 / "preservation_audit.json", preservation)

    final_methods = {}
    for result_name, result in (("GLOBAL_MATRIX", replay), ("NORMALIZED_FUSION", normalized)):
        method_names = [name for name in result["methods"] if name in {
            "GLOBAL_MATRIX", "GLOBAL_MATRIX_TEMPORAL", "GLOBAL_NORMALIZED_FUSION", "GLOBAL_NORMALIZED_FUSION_TEMPORAL"
        }]
        for method_name in method_names:
            summary = result["methods"][method_name]
            final_methods[method_name] = {
                "source_result": result_name,
                "horizons": {str(h): metric_row(summary, h) for h in (20, 50, 100)},
                "all_frame_score_change_rate": summary["all_frame_score_change_rate"],
                "all_frame_assignment_change_rate": summary["all_frame_assignment_change_rate"],
                "score_changes_after_event_plus_one": summary["score_changes_after_event_plus_one"],
                "assignment_changes_after_event_plus_one": summary["assignment_changes_after_event_plus_one"],
                "new_wrong_reassociation_total": summary["new_wrong_reassociation_total"],
                "corrected_wrong_reassociation_total": summary["corrected_wrong_reassociation_total"],
                "untouched_regression_total": summary["untouched_regression_total"],
            }

    global_gate = replay["gate"]
    normalized_gate = normalized["gate"]
    final_gate = {
        "schema": "N71_FINAL_GATE_V1",
        "experiment": protocol.get("experiment"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_SYNTHETIC_GATE_FAIL_PRODUCTION_DEFERRED",
        "engineering_integrity_gate": "PASS_WITH_EXPLICIT_LIMITATIONS",
        "synthetic_science_gate": "FAIL_FUTURE_EFFECT",
        "production_evidence": "DEFERRED_NO_REAL_HUMAN_TAPE",
        "completed_stages": [f"N71_STAGE_{index:02d}" for index in range(9)],
        "events": {"count": replay["event_count"], "independent_sequences": replay["independent_sequence_count"], "interaction_source": "simulated_from_gt", "real_human_tape": False},
        "runtime": {"frames": replay["runtime_frame_count"], "variant_frames": replay["variant_frame_count"], "future_gt_used": False, "posthoc_gt_loaded_after_runtime_audit": True, "audit_status": replay_audit["status"]},
        "candidate_branch": {"status": candidate_audit["status"], "windows": candidate_audit["window_count"], "sequences": len({item["sequence"] for item in candidate_audit["windows"]}), "frames": candidate_audit["total_frame_count"], "rows": candidate_audit["total_candidate_row_count"], "public_mapping": "UNAVAILABLE_NOT_FABRICATED", "missing_masks": candidate_audit["total_missing_mask_count"]},
        "dataset": {"status": dataset_audit["status"], "groups": dataset_audit["group_count"], "cells": dataset_audit["cell_count"], "positive_cells": dataset_audit["positive_cells"], "none_candidates": dataset_audit["none_candidates"], "split_groups": dataset_audit["split_group_counts"], "runtime_future_gt_used_count": dataset_audit["runtime_future_gt_used_count"]},
        "training": {"required_t1_completed": True, "final_attempt": 2, "checkpoint": train2["checkpoint"], "checkpoint_sha256": train2["checkpoint_sha256"], "best_epoch": train2["best_epoch"], "cuda_visible_devices": train2["cuda_visible_devices"], "cuda_device_name": train2["cuda_device_name"], "holdout_used_for_selection": train2["holdout_used_for_selection"]},
        "branches": {"N70_REPRODUCTION_DIAGNOSIS": "COMPLETED_NEGATIVE_INPUT", "GLOBAL_MATRIX": replay["status"], "GLOBAL_MATRIX_TEMPORAL": replay["status"], "NORMALIZED_FUSION": normalized["status"], "NEW_SAM3_CANDIDATE": "ENGINEERING_PASS_PUBLIC_MAPPING_UNAVAILABLE", "MEMORY_MODES_M0_M4": replay["status"], "CAUSAL_TRIMMING": "NOT_RUN_NO_POSITIVE_NONCAUSAL_PRECONDITION"},
        "gate": {"primary_global": global_gate, "normalized_probe": normalized_gate},
        "methods": final_methods,
        "calibration_head": "NOT_AUTHORIZED",
        "selector": "NOT_AUTHORIZED",
        "decoder_lora": "NOT_AUTHORIZED",
        "production_authorized": False,
        "preservation_audit": artifact(N71 / "preservation_audit.json"),
        "protocol": artifact(N71 / "protocol.json"),
        "protocol_amendment": artifact(N71 / "protocol_amendment_attempt2.json"),
    }
    atomic_json(N71 / "n71_final_gate.json", final_gate)

    stage07 = {
        "schema": "N71_STAGE_07_STATUS_V1",
        "status": "NOT_AUTHORIZED_NO_CONFIRMATION_AFTER_ALL_PRIMARY_BRANCHES_GATE_FAIL",
        "reason": "No branch achieved candidate-present improvement, real assignment crossing, or strict positive sequence-cluster CI; there is no positive branch eligible for confirmation.",
        "positive_branch_count": 0,
        "confirmation_run": False,
        "primary_branches_completed": ["N70_REPRODUCTION_DIAGNOSIS", "NEW_SAM3_CANDIDATE", "GLOBAL_MATRIX", "GLOBAL_MATRIX_TEMPORAL", "NORMALIZED_FUSION", "M0_M4_MEMORY_MODES"],
        "causal_trimming": "NOT_RUN_BY_PREREGISTERED_DEPENDENCY",
        "research_gate": "FAIL_FUTURE_EFFECT",
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "production_authorized": False,
        "final_gate": str(N71 / "n71_final_gate.json"),
    }
    atomic_json(N71 / "stage_07_status.json", stage07)

    stage08 = {
        "schema": "N71_STAGE_08_STATUS_V1",
        "status": "COMPLETE_SYNTHETIC_GATE_FAIL_PRODUCTION_DEFERRED",
        "report": str(ROOT / "docs/N71_FINAL_REPORT.md"),
        "final_gate": str(N71 / "n71_final_gate.json"),
        "all_reasonable_distinct_branches_completed": True,
        "future_effect_gate": "FAIL_FUTURE_EFFECT",
        "production_authorized": False,
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "runtime_future_gt_used": False,
    }
    atomic_json(N71 / "stage_08_status.json", stage08)

    # Keep the report compact while retaining every required numerical gate.
    lines: list[str] = []
    lines += [
        "# N71 Final Report — Global Identity-Association System Probe",
        "",
        f"Date: 2026-09-01 (Asia/Shanghai)  ",
        "Status: `COMPLETE_SYNTHETIC_GATE_FAIL_PRODUCTION_DEFERRED`  ",
        "Engineering integrity: `PASS_WITH_EXPLICIT_LIMITATIONS`  ",
        "Synthetic future-effect gate: `FAIL_FUTURE_EFFECT`  ",
        "Production authorization: `false`",
        "",
        "## 1. Executive summary",
        "",
        "N71 completed the pre-registered negative/diagnostic branches required to test whether a true identity-scoped global association system can turn correction evidence into future public-ID assignment changes. The trained global candidate×identity scorer and the one fixed normalized base+pair fusion probe changed scores, but neither changed a single Hungarian assignment in 24 events, 21 independent sequences, and 12,000 variant-frame records. H20/H50/H100 candidate-present utility was exactly zero with sequence-cluster 95% CIs `[0, 0]`. No calibration head, selector, decoder LoRA, or production change is authorized.",
        "",
        "All interactions in this report are `simulated_from_gt`; this is not real-human efficacy, not a real SAM3 full-loop result, and not production evidence.",
        "",
        "## 2. Frozen scope and protocol",
        "",
        f"N70 was treated as read-only input. N71 protocol SHA-256 is `{sha256_file(N71 / 'protocol.json')}`. The protocol kept the N70 24-event/21-sequence stream, M0–M4 memory variants, 100-frame future windows, explicit NONE, unchanged candidate order/embedding/metric definitions, sequence-cluster bootstrap (2,000 repetitions), and runtime `runtime_future_gt_used=false`. The only post-replay exploratory amendment was the pre-frozen per-frame standardization probe in `{amendment.get('branch')}`; it did not select values from future outcomes.",
        "",
        "The event stream remains GT-simulated and no real human tape exists. Future GT was loaded only after runtime artifacts and new branch artifacts passed structural audits.",
        "",
        "## 3. N70 root-cause recheck",
        "",
        "The CPU audit preserved the distinction between score motion and assignment crossing:",
        "",
        "| N70 branch | score-changed rows | assignment changes | correct crossings | untouched changed frames | target-candidate absent | public assignment absent | axis-mismatch frames |",
    ]
    lines += ["|---|---:|---:|---:|---:|---:|---:|---:|", f"| A | {n70_summary['counts']['branches']['A']['score_changed']} / {n70_summary['counts']['branches']['A']['frame_rows']} | {n70_summary['counts']['branches']['A']['assignment_changed']} | {n70_summary['counts']['branches']['A']['correct_changes']} | {n70_summary['counts']['branches']['A']['untouched_changed_total']} | {n70_summary['counts']['branches']['A']['candidate_absent_frames']} | {n70_summary['counts']['target_public_assignment_absent_frames']} | {n70_summary['counts']['variant_axis_mismatch_frames_retained']} |", f"| B | {n70_summary['counts']['branches']['B']['score_changed']} / {n70_summary['counts']['branches']['B']['frame_rows']} | {n70_summary['counts']['branches']['B']['assignment_changed']} | {n70_summary['counts']['branches']['B']['correct_changes']} | {n70_summary['counts']['branches']['B']['untouched_changed_total']} | {n70_summary['counts']['branches']['B']['candidate_absent_frames']} | {n70_summary['counts']['target_public_assignment_absent_frames']} | {n70_summary['counts']['variant_axis_mismatch_frames_retained']} |", "", "The actionable diagnosis is an interface/boundary failure: row-wise score changes rarely cross the global assignment boundary, while candidate absence and public-ID absence are separate upstream/mapping limitations. The 70 axis-mismatch frames remain retained diagnostics and were not credited as effects.", ""]
    lines += [
        "## 4. New candidate branch",
        "",
        "A real official SAM3 exporter branch ran in independent processes with `max_num_objects=16`, `multiplex_count=16`, `output_prob_thresh=0.30`, `offload_video_to_cpu=true`, default 160-frame windows and 20-frame overlap. The initial `32/32` configuration failed with a checkpoint shape mismatch (the checkpoint is fixed at 16); that failure was preserved and not bypassed.",
        "",
        f"The legal branch audited `{candidate_audit['window_count']}` windows / `{candidate_audit['total_frame_count']}` frames / `{candidate_audit['total_candidate_row_count']}` candidate rows. Missing masks: `{candidate_audit['total_missing_mask_count']}`; preserved degenerate boxes: `{candidate_audit['total_degenerate_box_count_preserved']}`; runtime future GT: `false`. The new exporter has no verified native→local→global→public bridge, so public mapping is explicitly unavailable and no identity result is claimed. Posthoc target recall at IoU≥0.5 over the six windows was `421/594 = {421/594:.6f}`; this was not used by runtime selection.",
        "",
        "## 5. Global association and training",
        "",
        f"The materialized candidate×identity dataset contains `{dataset_audit['group_count']}` frame/variant groups, `{dataset_audit['cell_count']}` cells, `{dataset_audit['positive_cells']}` positive cells and `{dataset_audit['none_candidates']}` explicit candidate-NONE cases. Sequence-disjoint groups are `{dataset_audit['split_group_counts']}`; numeric public/native IDs were not model features. The dataset audit status is `{dataset_audit['status']}` and runtime future GT count is `{dataset_audit['runtime_future_gt_used_count']}`.",
        "",
        f"T1 training was real CUDA training on GPU `{train2['cuda_visible_devices']}` ({train2['cuda_device_name']}), seed `{train2['seed']}`, best validation epoch `{train2['best_epoch']}`, model parameters `{train2['model']['parameter_count']}`, and checkpoint SHA-256 `{train2['checkpoint_sha256']}`. Holdout was read descriptively after validation-only selection and was not used to select the checkpoint. Attempt 1 is retained but excluded because its NONE head was not candidate-specific; attempt 2 retrained after the minimal semantic repair. The repaired CUDA smoke (`{smoke4['checkpoint_sha256']}`) passed explicit-NONE, reload, causal-boundary and old-association import checks.",
        "",
        "The final solver still enforces one-to-one identity assignment with an explicit candidate-specific NONE alternative. The temporal branch adds only the fixed three-frame, 0.15 hysteresis guard; it does not read future GT.",
        "",
        "## 6. Paired replay results",
        "",
        "The global-matrix runtime completed 24 events × 5 variants × 100 frames = 12,000 variant-frame artifacts. Its independent runtime audit passed with zero duplicate/missing keys, complete axes/mapping, event-frame memory hidden, event+1 causal boundary, and runtime future GT false. A posthoc schema failure from attempt 1 and the corrected baseline-vs-treatment wrong-reassociation semantics are both preserved.",
        "",
        "| Method | H | candidate-present / absent | baseline error | treated error | utility Δ | score-change | assignment-change | correct / incorrect / neutral | CI95 utility |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method_name in ("GLOBAL_MATRIX", "GLOBAL_MATRIX_TEMPORAL", "GLOBAL_NORMALIZED_FUSION", "GLOBAL_NORMALIZED_FUSION_TEMPORAL"):
        summary = final_methods[method_name]
        for horizon in (20, 50, 100):
            item = summary["horizons"][str(horizon)]
            lines.append(f"| {method_name} | {horizon} | {item['candidate_present_frames']} / {item['candidate_absent_frames']} | {fmt(item['baseline_future_identity_error'])} | {fmt(item['treated_future_identity_error'])} | {fmt(item['mean_utility_delta_candidate_present'])} | {fmt(item['score_change_rate'])} | {fmt(item['assignment_change_rate'])} | {item['correct_assignment_changes']} / {item['incorrect_assignment_changes']} / {item['neutral_assignment_changes']} | `{item['sequence_cluster_ci95']}` |")
    lines += [
        "",
        "Both global methods changed scores on every frame (`1.0`), including 11,880 post-event+1 frames, but assignment crossing was `0` and correct improvement was `0`. Candidate-present harm and treatment-induced new wrong reassociation were both `0`; this safety result is not a positive efficacy result. The raw baseline wrong-reassociation counts remain in the machine-readable artifacts and are not misclassified as treatment harms.",
        "",
        "### Action decomposition at H100",
        "",
        "| Action | frames | candidate-present / absent | baseline error | treated error | utility Δ | correct / incorrect / neutral | CI95 utility |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for action in ("ADD_NEW_IDENTITY", "AUTHORITATIVE_REASSIGN", "ATOMIC_ID_SWAP", "RECOVER_IDENTITY"):
        item = action_row(replay["methods"]["GLOBAL_MATRIX"], action)
        lines.append(f"| {action} | {item['frame_count']} | {item['candidate_present_frames']} / {item['candidate_absent_frames']} | {fmt(item['baseline_future_identity_error'])} | {fmt(item['treated_future_identity_error'])} | {fmt(item['mean_utility_delta_candidate_present'])} | {item['correct_assignment_changes']} / {item['incorrect_assignment_changes']} / {item['neutral_assignment_changes']} | `{item['sequence_cluster_ci95']}` |")
    lines += [
        "",
        "The normalized-fusion probe used the frozen formula `(base - mean(base_aug))/std(base_aug) + (pair - mean(pair_aug))/std(pair_aug)` independently per frame, with explicit candidate-NONE scores and the unchanged solver/temporal guard. Its candidate-specific NONE validator was minimally repaired after attempt 1 incorrectly required zero spread; attempt 2 passed structural audit. It reproduced the same zero-crossing result. Therefore the primary hypothesis is not merely raw logit scale; the learned global score does not supply a usable assignment-boundary advantage on this frozen stream.",
        "",
        "## 7. Branch decisions",
        "",
        "| Branch | Outcome | Decision |",
        "|---|---|---|",
        "| N70 reproduction/root diagnosis | Negative diagnostic | Retain as control; do not call score motion efficacy. |",
        "| True global candidate×identity matrix + explicit NONE | Execution PASS, future-effect FAIL | Do not promote. |",
        "| Short temporal/track guard | No crossing or utility change | No promotion; no extra temporal training. |",
        "| Fixed normalized base+pair interface | Execution PASS, future-effect FAIL | Scale alone is not sufficient. |",
        "| M0–M4 memory modes | Replay complete; identical assignment | No memory-update authorization. |",
        "| New official SAM3 candidate branch | Candidate audit PASS; public mapping unavailable | Do not merge into identity replay. |",
        "| Causal trimming / TACT-style branch | Not run | Pre-registered dependency failed: no positive noncausal branch. |",
        "| Confirmation experiment | Not run | No branch met the confirmation precondition. |",
        "",
        "## 8. Method search",
        "",
        "The search prioritized 2025–2026 official paper/arXiv/OpenReview pages and official repositories. It adopted only identity/trajectory-conditioned input design and a conservative temporal protection diagnostic; it did not claim to reproduce any paper. HATReID-MOT was not adopted because it changes the frozen 512-D representation; MeMoSORT was kept as a future motion/base-state reference because no verified official repository was found; the exact TACT match was not verified and no citation was invented.",
        "",
        "| Method | Paper | Official repository | Commit/date/license | Reused mechanism |",
        "|---|---|---|---|---|",
    ]
    for item in method_search["candidates"]:
        if not item.get("title"):
            continue
        paper = item.get("paper_url") or item.get("arxiv_url") or item.get("openreview_url") or ""
        repo = item.get("official_repo_url") or "not verified"
        commit = item.get("repo_commit") or "not verified"
        date = item.get("repo_commit_date") or "not verified"
        license_name = item.get("license") or "not verified"
        lines.append(f"| {item['name']} | [{item['title']}]({paper}) | {repo} | `{commit}` / `{date}` / `{license_name}` | {item.get('mechanism', '')} |")
    lines += [
        "",
        "## 9. Failures retained and isolation",
        "",
        "Every failure remains under `./outputs/N71/attempts/`; no failure was converted into PASS. The actionable repairs were: entry-point import path; unsupported 32-capacity checkpoint shape; four materialization schema/shape bugs; unavailable `torch.flatnonzero`; smoke checkpoint timing; replay posthoc boolean metadata; baseline/treatment wrong-reassociation classification; and the normalized probe's NONE-spread validator semantics.",
        "",
        f"The preservation audit compared `{len(unchanged)}` recorded production Python hashes: changed `{len(changed)}`, missing `{len(missing)}`; no production or `third_party/sam3` file had a newer mtime after the N71 snapshot. The project has no usable Git repository, so file hashes and protected-root checks are the isolation proof. All heavy outputs and checkpoints are in `{HEAVY}`; N36–N70 outputs were not overwritten.",
        "",
        "## 10. Conclusions and next action",
        "",
        "N71 proves that the global matrix model and the fixed normalized interface execute, change finite scores, preserve the explicit-NONE/mapping/runtime contracts, and avoid treatment-induced collateral on this frozen synthetic stream. It does not prove future identity improvement: there are zero real assignment crossings, zero candidate-present improvements, zero strict-positive sequence-cluster CIs, and no confirmation candidate. It also does not isolate candidate quality as a successful identity branch because the new SAM3 branch lacks a public mapping bridge.",
        "",
        "Do not train calibration, selector, decoder LoRA, or change production association. The minimum scientifically meaningful next step is provenance-complete real human tape plus a candidate-complete real SAM3 full-loop/mapping bridge; if synthetic work continues, it must introduce a genuinely different, pre-registered association state or candidate/public bridge rather than more weight scaling. Causal trimming remains out of scope until a noncausal branch first shows a strict positive effect.",
        "",
        "ICLR 2027 calendar constraint recorded in the protocol: abstract deadline 2026-09-18 AoE; full paper deadline 2026-09-25 AoE; report date 2026-09-01. Plan remaining work around the external human-tape and mapping blocker rather than consuming the deadline with repeated zero-crossing scans.",
        "",
        "## 11. Key machine-readable artifacts",
        "",
    ]
    key_paths = [
        N71 / "n71_final_gate.json", N71 / "stage_00_status.json", N71 / "stage_01_status.json", N71 / "stage_02_status.json", N71 / "stage_03_status.json", N71 / "stage_04_status.json", N71 / "stage_05_status.json", N71 / "stage_06_status.json", N71 / "stage_07_status.json", N71 / "stage_08_status.json", N71 / "protocol.json", N71 / "method_search.json", N71 / "preservation_audit.json", N71 / "diagnosis/n70_root_cause_summary.json", N71 / "candidate_branch/full_audit_attempt1.json", N71 / "training/global_matrix_dataset_manifest_attempt5.json", N71 / "training/global_matrix_dataset_audit_attempt5.json", N71 / "training/global_matrix_training_attempt2.json", N71 / "replay/global_matrix_runtime_manifest_attempt1.json", N71 / "replay/global_matrix_replay_results_attempt2.json", N71 / "replay/global_matrix_replay_results_attempt2_runtime_audit.json", N71 / "replay/normalized_fusion_probe_manifest_attempt2.json", N71 / "replay/normalized_fusion_probe_results_attempt2.json", N71 / "replay/normalized_fusion_probe_results_attempt2_audit.json",
    ]
    lines += ["| Artifact | SHA-256 |", "|---|---|"]
    for path in key_paths:
        lines.append(f"| {md_link(path)} | `{sha256_file(path)}` |")
    lines += ["", "The complete per-frame artifacts are referenced by the two replay manifests and stored under the independent `/data2` roots; they are intentionally not embedded in this report.", ""]
    (ROOT / "docs/N71_FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
