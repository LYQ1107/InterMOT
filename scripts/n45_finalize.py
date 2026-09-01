#!/usr/bin/env python3
"""N45 strict attribution gate; leaves the N44 report/result untouched."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
N44_REPORT = ROOT / "docs/N44_FINAL_REPORT.md"
N44_GATE = ROOT / "outputs/n44/stage_05_status.json"
N44_RESULT = ROOT / "outputs/n44/replay/paired_replay_results.json"
N45_S1 = ROOT / "outputs/n45/stage_01_status.json"
N45_S2 = ROOT / "outputs/n45/stage_02_status.json"
N45_S3 = ROOT / "outputs/n45/stage_03_status.json"
N45_RESULT = ROOT / "outputs/n45/replay/attribution_results.json"
N45_BLOCK = ROOT / "outputs/n44/blocked_input_real_human_tape.json"
N45_REG = ROOT / "outputs/n45/n45_sidecar_targeted_regression.json"
N45_ALIGN = ROOT / "outputs/n45/n45_alignment_targeted_regression.json"
OUT = ROOT / "outputs/n45"
GATE = OUT / "stage_04_status.json"
REPORT = ROOT / "docs/N45_FINAL_REPORT.md"
N45_BLOCK_OUT = OUT / "blocked_input_real_human_tape.json"
N45_RUNTIME = OUT / "replay/runtime"
VARIANTS = {"M0", "M1", "M2", "M3", "M4"}
BRANCHES = {"no_write", "write_baseline", "write_plus_n44"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_signature(row: dict) -> list[tuple]:
    return [
        (item.get("native_tid"), item.get("box"), item.get("confidence"))
        for item in row.get("rows", [])
    ]


def validate_runtime_artifacts() -> dict:
    """Validate the concrete N45 three-branch runtime contract."""
    files = sorted(N45_RUNTIME.glob("*.json"))
    failures: list[str] = []
    if len(files) != 24:
        failures.append(f"runtime_artifact_count={len(files)} expected=24")
    checked = 0
    for path in files:
        try:
            payload = read(path)
        except Exception as exc:
            failures.append(f"{path.name}: unreadable {type(exc).__name__}")
            continue
        variants = payload.get("variants")
        if set(variants or {}) != VARIANTS:
            failures.append(f"{path.name}: variants={sorted((variants or {}).keys())}")
            continue
        if payload.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
            failures.append(f"{path.name}: runtime_future_gt_used is not false")
        for variant in sorted(VARIANTS):
            branches = variants[variant]
            if not BRANCHES.issubset(set(branches or {})):
                failures.append(f"{path.name}/{variant}: branches={sorted((branches or {}).keys())}")
                continue
            for branch in sorted(BRANCHES):
                trace = branches[branch]
                if not isinstance(trace, list) or len(trace) != 100:
                    failures.append(f"{path.name}/{variant}/{branch}: frame_count={len(trace) if isinstance(trace, list) else 'non-list'}")
                    continue
                frames = [int(row.get("frame")) for row in trace]
                if frames != list(range(frames[0], frames[0] + 100)):
                    failures.append(f"{path.name}/{variant}/{branch}: frame gap/duplicate")
                for row in trace:
                    checked += 1
                    if row.get("runtime_future_gt_used") is not False:
                        failures.append(f"{path.name}/{variant}/{branch}/{row.get('frame')}: runtime future GT flag")
                    native = [item.get("native_tid") for item in row.get("rows", []) if item.get("native_tid") is not None]
                    if len(native) != len(set(native)):
                        failures.append(f"{path.name}/{variant}/{branch}/{row.get('frame')}: duplicate native ID")
            write_trace = branches["write_baseline"]
            plus_trace = branches["write_plus_n44"]
            if isinstance(write_trace, list) and isinstance(plus_trace, list) and len(write_trace) == len(plus_trace) == 100:
                for write_row, plus_row in zip(write_trace, plus_trace):
                    if _candidate_signature(write_row) != _candidate_signature(plus_row):
                        failures.append(f"{path.name}/{variant}/{write_row.get('frame')}: write/plus candidate rows differ")
                    if write_row.get("public_id_order") != plus_row.get("public_id_order"):
                        failures.append(f"{path.name}/{variant}/{write_row.get('frame')}: write/plus public-ID axis differs")
    return {"pass": not failures, "artifact_count": len(files), "trace_rows_checked": checked, "failure_count": len(failures), "failures": failures[:100]}


def main() -> None:
    s1, s2, s3 = (read(path) for path in (N45_S1, N45_S2, N45_S3))
    n44_gate, n44_result, n45, block, reg, align = (read(path) for path in (N44_GATE, N44_RESULT, N45_RESULT, N45_BLOCK, N45_REG, N45_ALIGN))
    runtime_contract = validate_runtime_artifacts()
    checks = {"n45_stage_01": s1.get("status") == "PASS", "n45_stage_02": s2.get("status") == "PASS", "n45_stage_03_complete": s3.get("status") == "PASS" and s3.get("metrics", {}).get("validation", {}).get("trace_rows_checked") == 36000, "three_branches": runtime_contract["pass"], "runtime_artifact_count_24": runtime_contract["artifact_count"] == 24, "runtime_variants_m0_m4": runtime_contract["pass"], "runtime_each_branch_100_frames": runtime_contract["pass"], "write_plus_candidate_rows_and_axis_equal": runtime_contract["pass"], "same_candidate_rows": s3.get("gate_checks", {}).get("same_candidates") is True, "write_baseline_original_n42": s3.get("gate_checks", {}).get("write_baseline_is_original_n42_fused_assignment") is True, "plus_diff_only_sidecar": s3.get("gate_checks", {}).get("write_plus_diff_only_sidecar") is True, "runtime_future_gt_false": n45.get("runtime_future_gt_used") is False, "posthoc_after_validation": n45.get("gt_loaded_only_after_runtime_validation") is True, "neutral_assignment_category_present": all("assignment_change_neutral_count" in n45["effects"][effect][variant][str(h)] for effect in ("memory", "incremental") for variant in ("M0", "M1", "M2", "M3", "M4") for h in (20, 50, 100)), "decomposition_closes": all(n45["effects"]["incremental"][variant][str(h)]["assignment_decomposition_closes"] for variant in ("M0", "M1", "M2", "M3", "M4") for h in (20, 50, 100)), "equal_sequence_bootstrap": n45.get("bootstrap_protocol") == "sequence_mean_then_equal_sequence_cluster_bootstrap", "checkpoint_authorization_false": s2.get("gate_checks", {}).get("production_authorized_explicitly_false_in_repair") is True and read(ROOT / "outputs/n45/frozen_checkpoint_authorization.json")["production_authorized"] is False, "targeted_regression": reg.get("status") == "PASS" and align.get("status") == "PASS", "real_human_tape": False, "real_sam3_full_loop": False}
    incremental = n45["effects"]["incremental"]
    m2 = incremental["M2"]
    strict_incremental_effect = all(float(incremental[variant][str(h)]["sequence_cluster_bootstrap_95ci"]["lower"]) > 0 for variant in ("M2", "M3", "M4") for h in (20, 50, 100))
    all_coverage_zero = all(float(incremental[variant][str(h)]["identity_utility"]) == 0 and int(incremental[variant][str(h)]["assignment_change_count"]) == 0 for variant in ("M1", "M2", "M3", "M4") for h in (20, 50, 100))
    if all_coverage_zero:
        incremental_status = "N44_NOT_EXERCISED"
    elif strict_incremental_effect:
        incremental_status = "POSITIVE_OFFLINE_ONLY"
    else:
        incremental_status = "STRUCTURAL_HYPOTHESIS_FAILED"
    gate = {"schema": "N45_FINAL_ATTRIBUTION_GATE_V1", "status": "N45_COMPLETED_GATE_FAILED", "research_gate": "FAIL_TRUE_INCREMENTAL_EFFECT_AND_REAL_INPUT", "authorization": {"calibration_head": "NOT_AUTHORIZED", "decoder_lora": "NOT_AUTHORIZED", "production_interface_changed": False}, "checks": checks, "incremental_effect_status": incremental_status, "memory_effect_status": "MEASURED_SEPARATELY", "failure_root_cause": {"attribution_repair": "N44 no_write versus write_plus_N44 was not an attributable N44 comparison; N45 now measures write_baseline versus write_plus_N44 directly.", "incremental_effect": "The true N44 increment is zero for M2 (neutral assignment changes only) and negative for M1/M3/M4, with no correct incremental changes; strict positive CI gate fails.", "real_input": block["exact_blocker"], "n44_legacy": "N44 result/report are preserved as provisional legacy evidence and are not overwritten."}, "side_by_side": {"n44_provisional_result": str(N44_RESULT), "n45_attribution_result": str(N45_RESULT), "n44_m2_all_effect": n44_result["aggregates"]["M2"], "n45_m2_memory_effect": n45["effects"]["memory"]["M2"], "n45_m2_incremental_effect": m2}, "next_action": "Do not run another blind training or gate scan. Obtain the external N40 provenance-complete human tape and real SAM3 full-loop; only after that decide whether a new hypothesis about the active public-ID/candidate interface justifies another actual experiment."}
    stage = {"status": gate["status"], "protocol": gate["schema"], "command": ["python", "scripts/n45_finalize.py"], "inputs": {"n44_report": str(N44_REPORT), "n44_gate": str(N44_GATE), "n44_result": str(N44_RESULT), "n45_stage_01": str(N45_S1), "n45_stage_02": str(N45_S2), "n45_stage_03": str(N45_S3), "n45_attribution": str(N45_RESULT), "n45_runtime": str(N45_RUNTIME), "n40_blocker": str(N45_BLOCK), "targeted_regression": str(N45_REG), "alignment_regression": str(N45_ALIGN)}, "outputs": {"gate": str(GATE), "report": str(REPORT), "n45_real_input_blocker": str(N45_BLOCK_OUT)}, "metrics": {"incremental_effect_status": incremental_status, "n45_incremental_m2": {str(h): m2[str(h)] for h in (20, 50, 100)}, "n45_incremental_m1_m3_m4_identity_utility": {v: {str(h): incremental[v][str(h)]["identity_utility"] for h in (20, 50, 100)} for v in ("M1", "M3", "M4")}, "runtime_contract": runtime_contract, "runtime_trace_rows": s3["metrics"]["validation"]["trace_rows_checked"], "runtime_proposal_totals": s3["metrics"]["runtime"]["totals"], "real_human_tape": False, "real_full_loop": False, "n44_provisional_preserved": True}, "gate_checks": checks, "failure_root_cause": gate["failure_root_cause"], "next_action": gate["next_action"], "runtime_future_gt_used": False, "finished_at": now()}
    source_bytes = N45_BLOCK.read_bytes()
    block_copy = {"schema": "N45_BLOCKED_INPUT_REAL_HUMAN_TAPE_V1", "status": block["status"], "source_artifact": str(N45_BLOCK), "source_sha256": hashlib.sha256(source_bytes).hexdigest(), "checks_completed": block["checks_completed"], "checks": block["checks"], "exact_blocker": block["exact_blocker"], "minimal_next_step": block["minimal_next_step"], "fabrication_or_relabeling": block["fabrication_or_relabeling"], "runtime_future_gt_used": False, "downstream_authorized": False, "old_simulated_artifacts_relabelled": False}
    N45_BLOCK_OUT.write_text(json.dumps(block_copy, indent=2) + "\n", encoding="utf-8")
    block_ref = dict(block_copy); block_ref["n45_reference"] = "N45 retains the N44 BLOCKED_INPUT_REAL_HUMAN_TAPE evidence unchanged."; (OUT / "blocked_input_real_human_tape_reference.json").write_text(json.dumps(block_ref, indent=2) + "\n", encoding="utf-8")
    GATE.write_text(json.dumps(stage, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# N45 Final Report — true N44 attribution repair",
        "",
        f"Date: {now()}  ",
        f"Status: `{gate['status']}`. The research objective remains open; this report closes only the N45 attribution-repair unit.",
        "",
        "## Why N44 was provisional",
        "",
        "N44 compared `no_write` against `write_plus_N44`. It did not materialize the unchanged N42 write branch, so the reported N44 utility and 0/12/42/92 assignment changes were memory-write plus sidecar effects, not an attributable N44 increment. The original N44 report, result, runtime files, checkpoint and failures are preserved unchanged.",
        "",
        "## N45 protocol and integrity",
        "",
        "N45 reads the frozen N42 runtime source and frozen N44 checkpoint and materializes, for every one of 24 events, M0–M4, and 100 future frames: (1) N42 `no_write`, (2) N42 `write_baseline` with original fused assignment and no N44, and (3) `write_plus_N44`. Candidate rows (native ID, box, confidence) are identical across branches; dynamic public-ID axes are retained per branch and recorded by intersection/branch-only sets. The write baseline and plus branch use the exact write-ID axis. Runtime validation checked 36000 trace rows, no duplicate/missing frames, candidate completeness, and `runtime_future_gt=false`; GT was loaded only after validation.",
        "",
        "A targeted regression found and repaired two non-scientific attribution issues: N42 NONE uses `-1` while N44 explicit dummy columns use indices beyond the public-ID count, and N42 no/write can have different active public-ID universes despite identical candidate rows. These are now normalized/recorded rather than conflated with candidate mismatch. Accepted boosts were checked against recorded changed cells; hard/NONE semantics and current fused baseline were verified. The immutable N44 checkpoint payload lacked an explicit authorization field; N45 records its source hash and `production_authorized=false` in [frozen_checkpoint_authorization.json](./outputs/n45/frozen_checkpoint_authorization.json) without changing the checkpoint.",
        "",
        "## Corrected Stage 01/02 diagnostics",
        "",
        "The corrected candidate ceiling is 13270 oracle-correct assigned candidate rows divided by 16383 baseline assigned-known candidate rows: `0.8099859611`. The old 0.0763552042 total-cell denominator is retained only as `oracle_correct_rate_over_total_cell_count` legacy diagnostic. The frozen N44 audit contains zero hard-negative cells; N44 code skipped hard-negative cells if encountered. The corrected contract explicitly records zero training hard-negative examples rather than claiming they were included.",
        "",
        "## Memory effect versus true N44 increment",
        "",
        "All values below are event means; CIs are sequence-mean then equal-sequence bootstrap (seed 4444, 2000 replicates). Assignment columns are `changed / correct / incorrect / neutral / no-change`.",
        "",
        "| Variant/effect | Horizon | Utility | Target IoU Δ | Error reduction | Assignment | CI lower |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[f"| M2 memory no_write→write | H{h} | {n45['effects']['memory']['M2'][str(h)]['identity_utility']:.15f} | {n45['effects']['memory']['M2'][str(h)]['target_iou_delta']:.15f} | {n45['effects']['memory']['M2'][str(h)]['future_identity_error_reduction']:.15f} | {n45['effects']['memory']['M2'][str(h)]['assignment_change_count']} / {n45['effects']['memory']['M2'][str(h)]['assignment_change_correct_count']} / {n45['effects']['memory']['M2'][str(h)]['assignment_change_incorrect_count']} / {n45['effects']['memory']['M2'][str(h)]['assignment_change_neutral_count']} / {n45['effects']['memory']['M2'][str(h)]['assignment_no_change_count']} | {n45['effects']['memory']['M2'][str(h)]['sequence_cluster_bootstrap_95ci']['lower']:.15f} |" for h in (20, 50, 100)],
        *[f"| M2 true N44 increment write→plus | H{h} | {m2[str(h)]['identity_utility']:.15f} | {m2[str(h)]['target_iou_delta']:.15f} | {m2[str(h)]['future_identity_error_reduction']:.15f} | {m2[str(h)]['assignment_change_count']} / {m2[str(h)]['assignment_change_correct_count']} / {m2[str(h)]['assignment_change_incorrect_count']} / {m2[str(h)]['assignment_change_neutral_count']} / {m2[str(h)]['assignment_no_change_count']} | {m2[str(h)]['sequence_cluster_bootstrap_95ci']['lower']:.15f} |" for h in (20, 50, 100)],
        "",
        "The true N44 M2 increment is exactly zero utility at H20/H50/H100, with 6/9/15 neutral assignment changes and zero correct or incorrect target changes. M1/M3/M4 true increments are slightly negative, with one incorrect change and no correct changes at each horizon. The selected-but-no-assignment-change counts are retained per frame and per horizon in every event artifact; M2 has 2 such selections through H20/H50/H100. Therefore this is not `N44_NOT_EXERCISED`: the gate selected proposals and caused bounded assignment activity, but the structural hypothesis failed to produce benefit.",
        "For completeness, the true incremental identity utility (H20/H50/H100) is M0 `0/0/0`, M1 `-0.000307395890/-0.000058155979/-0.000027586811`, M2 `0/0/0`, M3 equal to M1, and M4 equal to M1. The machine-readable result contains target-IoU delta, future identity-error reduction, re-correction, assignment decomposition, untouched-ID regression, and both CI bounds for every event, variant and horizon. Runtime proposal totals are 28 considered, 14 selected, 5 selected-but-no-assignment-change, 14 changed cells and 18 changed assignments; these are application counts, not efficacy metrics.",
        "",
        "## Provenance and next step",
        "",
        "All 24 events remain `simulated_from_gt`; there is no real human tape and no real SAM3 full-loop. Three feasibility checks are retained: the N34 sentinel is unavailable with zero events, the N40 contract audit reports `BLOCKED_INPUT_REAL_HUMAN_TAPE`, and the inventory finds only a GT-derived synthetic fallback with no external UI export. The original N44 blocker is retained at [blocked_input_real_human_tape.json](./outputs/n44/blocked_input_real_human_tape.json), and N45's hash-bound copy is [here](./outputs/n45/blocked_input_real_human_tape.json). Standard MOT/TrackEval metrics remain `NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT`.",
        "",
        "N45 does not authorize production calibration or LoRA. Do not run another blind training/threshold scan. The next scientifically meaningful step is external N40 provenance-complete human tape plus a real candidate-complete SAM3 full-loop; after that, decide whether the active public-ID/candidate-universe behavior justifies a new hypothesis and a new actual training experiment.",
        "",
        "Artifacts: [Stage 01](./outputs/n45/stage_01_status.json), [Stage 02](./outputs/n45/stage_02_status.json), [Stage 03](./outputs/n45/stage_03_status.json), [Stage 04 gate](./outputs/n45/stage_04_status.json), [attribution result](./outputs/n45/replay/attribution_results.json), [targeted regression](./outputs/n45/n45_sidecar_targeted_regression.json), and [axis regression](./outputs/n45/n45_alignment_targeted_regression.json).",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": stage["status"], "gate": str(GATE), "report": str(REPORT)}))


if __name__ == "__main__":
    main()
