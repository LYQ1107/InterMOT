#!/usr/bin/env python3
"""Finalize the corrected N45 attribution without overwriting N45 outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "outputs/n46/n45_attribution_repair/normalized_attribution_results.json"
REG = ROOT / "outputs/n46/n45_attribution_repair/targeted_regression.json"
STATUS_IN = ROOT / "outputs/n46/n45_attribution_repair/status.json"
N40 = ROOT / "outputs/n40/stage_01_status.json"
OLD = ROOT / "outputs/n45/replay/attribution_results.json"
OUT = ROOT / "outputs/n46/n45_attribution_repair"
GATE = OUT / "final_gate.json"
STATUS = OUT / "final_status.json"
REPORT = ROOT / "docs/N45_ATTRIBUTION_REPAIR_FINAL_REPORT.md"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
HORIZONS = (20, 50, 100)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    result = load(RESULT); reg = load(REG); old = load(OLD); n40 = load(N40)
    checks = {
        "normalized_result_pass": result.get("status") == "PASS", "targeted_regression_pass": reg.get("status") == "PASS", "all_24_events": result.get("event_count") == 24, "all_5_variants": result.get("variant_count") == 5, "all_horizons": result.get("horizons") == list(HORIZONS), "runtime_future_gt_false": result.get("protocol", {}).get("runtime_future_gt_used") is False and result.get("runtime_validation", {}).get("runtime_future_gt_used") is False, "gt_only_after_runtime_validation": result.get("protocol", {}).get("gt_loaded_only_after_runtime_validation") is True, "axis_normalized": result.get("axis_reconciliation", {}).get("both_assignment_maps_normalized") is True, "old_n45_not_modified": result.get("axis_reconciliation", {}).get("old_n45_result_modified") is False, "equal_sequence_bootstrap": result.get("protocol", {}).get("bootstrap") == "sequence_mean_then_equal_sequence_cluster_bootstrap", "simulated_provenance": result.get("interaction_source") == "simulated_from_gt" and result.get("real_human_tape_created") is False, "standard_mot_not_computable": result.get("id_switch_metric") == "NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT", "production_authorized": False,
    }
    m2 = result["effects"]["incremental"]["M2"]
    gate = {
        "schema": "N45_NORMALIZED_ATTRIBUTION_REPAIR_GATE_V1", "status": "N45_ATTRIBUTION_REPAIR_COMPLETED_GATE_FAILED", "research_gate": "FAIL_FUTURE_EFFECT_AND_REAL_INPUT", "checks": checks,
        "legacy_status": "N45 old attribution_results.json is preserved as provisional because baseline candidate_public_ids were not axis-normalized.",
        "corrected_status": "N45 normalized attribution is structurally complete; M2 incremental utility is zero with neutral-only changes, so no efficacy pass.",
        "corrected_m2_increment": {str(h): {k: m2[str(h)][k] for k in ("identity_utility", "target_iou_delta", "future_identity_error_reduction", "recorrection_proxy_reduction", "assignment_change_count", "assignment_change_correct_count", "assignment_change_incorrect_count", "assignment_change_neutral_count", "assignment_no_change_count", "sequence_cluster_bootstrap_95ci")} for h in HORIZONS},
        "memory_effect_status": "MEASURED_SEPARATELY_AND_NEGATIVE",
        "provenance_status": "SIMULATED_FROM_GT_ONLY",
        "real_input_status": "BLOCKED_INPUT_REAL_HUMAN_TAPE",
        "authorization": {"calibration_head": "NOT_AUTHORIZED", "decoder_lora": "NOT_AUTHORIZED", "production_interface_changed": False},
        "failure_root_cause": "N45 baseline mapping used raw candidate_public_ids, including IDs outside public_id_order, while plus mapping used assignment columns. The normalized repair maps no/write/plus uniformly through assignment columns and the active public-ID axis. The corrected M2 increment remains zero utility and neutral-only.",
        "next_action": "Keep the broader objective open; obtain N40-compliant real human tape and real SAM3 full-loop before any new experiment or authorization.",
    }
    GATE.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# N45 Normalized Attribution Repair — Final Report", "", f"Date: {datetime.now(timezone.utc).isoformat()}  ", "Status: `N45_ATTRIBUTION_REPAIR_COMPLETED_GATE_FAILED`. This closes only the isolated attribution-repair unit; the broader scientific objective remains open.", "",
        "## Correction scope", "",
        "The original N45 runtime and `outputs/n45/replay/attribution_results.json` are preserved unchanged as legacy evidence. A contract audit found that N45 `slim()` used N42 `candidate_public_ids` for the write baseline, while write-plus used assignment columns through `public_id_order`. In 78 frozen write-source frames across variants, raw candidate IDs included an ID outside the active axis; this could appear as a false baseline→plus mapping change. Example: frame 84 of `n37-dancetrack0008-0060-authoritative_reassign-002` contains raw `101001` where the axis-normalized assignment is `NONE`.", "",
        "The repair uses the frozen N42 no-write/write branches and the isolated repair2 runtime, maps every branch by assignment column through its current public-ID axis, and recomputes posthoc assignment changes using native-ID mappings. It does not use raw candidate_public_ids as assignment evidence and does not alter the N45 checkpoint, seed, metrics or production interface.", "",
        "## Runtime and evaluation integrity", "",
        "The repaired runtime is a real complete frozen replay: 24 events × 5 variants × 100 future frames = 12000 runtime frames. M0 is the exact no-sidecar control; M1–M4 apply only the frozen N44 sidecar. Posthoc evaluation covers the same event/frame axes, with GT opened only after runtime validation. Every event remains `simulated_from_gt`, not a real human tape.", "",
        "The repaired runtime/posthoc outputs pass the targeted alignment regression: 24 event files, all variants and horizons, direct `runtime_future_gt_used=false`, assignment+axis normalization, and old-result immutability. The old N45 totals `28/14/5/14/18` remain the application-count legacy reference; repair2 reproduces those runtime totals while correcting mapping attribution.", "",
        "## Corrected effects", "",
        "| Effect / variant | H20 utility | H50 utility | H100 utility |", "|---|---:|---:|---:|",
    ]
    for effect, label in (("memory", "memory no-write→write"), ("incremental", "N44 increment write→plus")):
        for v in VARIANTS:
            vals = [result["effects"][effect][v][str(h)]["identity_utility"] for h in HORIZONS]
            lines.append(f"| {label} {v} | {vals[0]:.15f} | {vals[1]:.15f} | {vals[2]:.15f} |")
    lines += ["", "Corrected M2 incremental assignment decomposition is:", "", f"- H20: `{m2['20']['assignment_change_count']} / {m2['20']['assignment_change_correct_count']} / {m2['20']['assignment_change_incorrect_count']} / {m2['20']['assignment_change_neutral_count']} / {m2['20']['assignment_no_change_count']}` (changed/correct/incorrect/neutral/no-change).", f"- H50: `{m2['50']['assignment_change_count']} / {m2['50']['assignment_change_correct_count']} / {m2['50']['assignment_change_incorrect_count']} / {m2['50']['assignment_change_neutral_count']} / {m2['50']['assignment_no_change_count']}`.", f"- H100: `{m2['100']['assignment_change_count']} / {m2['100']['assignment_change_correct_count']} / {m2['100']['assignment_change_incorrect_count']} / {m2['100']['assignment_change_neutral_count']} / {m2['100']['assignment_no_change_count']}`.", "", "Thus the corrected M2 incremental effect is exactly zero utility at all horizons and every corrected assignment change is neutral. The old `6/9/15` counts are retained only in the old N45 result and are not used as corrected efficacy evidence. The separately measured M2 memory effect remains negative (`-0.009265268392`, `-0.009392946035`, `-0.007457178589`).", "", "All variants, target IoU, future identity error, re-correction, untouched regression and equal-sequence bootstrap CIs are in `normalized_attribution_results.json`; standard MOT/TrackEval remains `NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT`.", "", "## Real-input gate", "", f"N40 remains `{n40['status']}`. No direct human public_id, human-confirmed BOX/CLICK/CONFIRMED_MASK, lossless ROI digest, UI/session/annotator timestamps or candidate-complete real future tape is available. The only fallback remains GT-derived synthetic data. The real-input blocker is preserved at `outputs/n46/BLOCKED_INPUT_REAL_HUMAN_TAPE.json`; no simulated record was relabeled.", "", "## Artifacts and decision", "", "- Corrected result: `outputs/n46/n45_attribution_repair/normalized_attribution_results.json`.", "- Corrected targeted regression: `outputs/n46/n45_attribution_repair/targeted_regression.json`.", "- Corrected stage status: `outputs/n46/n45_attribution_repair/status.json`.", "- Corrected gate: `outputs/n46/n45_attribution_repair/final_gate.json`.", "- Preserved old result: `outputs/n45/replay/attribution_results.json`.", "- All failure attempts remain under `outputs/n46/attempts`.", "", "No new training was started. The corrected attribution does not show a positive N44 increment, and real human tape/full-loop gates are absent. Calibration, decoder LoRA and production changes remain unauthorized. The broader research objective remains open pending provenance-complete human input and real full-loop evaluation."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    final_status = {"status": gate["status"], "protocol": gate["schema"], "command": ["python", "scripts/n45_normalized_attribution_finalize.py"], "inputs": {"normalized_result": str(RESULT), "targeted_regression": str(REG), "old_n45_result": str(OLD), "n40_audit": str(N40)}, "outputs": {"gate": str(GATE), "report": str(REPORT), "status": str(STATUS)}, "metrics": {"corrected_m2_increment": gate["corrected_m2_increment"], "axis_mismatch_frames": result["axis_reconciliation"]["write_source_frames_with_candidate_public_id_axis_mismatch"], "old_result_preserved": True}, "gate_checks": checks, "failure_root_cause": gate["failure_root_cause"], "next_action": gate["next_action"], "runtime_future_gt_used": False, "gt_loaded_posthoc": True, "finished_at": datetime.now(timezone.utc).isoformat()}
    STATUS.write_text(json.dumps(final_status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": final_status["status"], "gate": str(GATE), "report": str(REPORT)}))


if __name__ == "__main__":
    main()
