#!/usr/bin/env python3
"""Finalize the isolated N47 probe with a strict semantic gate."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n47_global_probe_common import CHECKPOINT, N46_GATE, OUT, PROTOCOL, TRAIN_MANIFEST, load, sha256, write_json


STAGE = OUT / "stage_05_status.json"
GATE = OUT / "n47_final_gate.json"
REPORT = ROOT / "docs/N47_FINAL_REPORT.md"
RESULT = OUT / "replay/probe_results.json"
INTEGRITY = OUT / "stage_04_integrity.json"
N40_BLOCKER = ROOT / "outputs/n46/BLOCKED_INPUT_REAL_HUMAN_TAPE.json"
N45_RESULT = ROOT / "outputs/n46/n45_attribution_repair/normalized_attribution_results.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    result = load(RESULT); integrity = load(INTEGRITY); training = load(TRAIN_MANIFEST); n46_gate = load(N46_GATE); blocker = load(N40_BLOCKER); m2 = result["effects"]["incremental"]["M2"]
    horizons = ("20", "50", "100")
    strict_ci = all(float(m2[h]["sequence_cluster_bootstrap_95ci"]["lower"]) > 0.0 for h in horizons)
    no_incorrect = all(int(m2[h]["assignment_change_incorrect_count"]) == 0 for h in horizons)
    untouched_clean = all(bool(m2[h]["untouched_regression"].get("all_no_obvious_regression", False)) for h in horizons)
    positive_correct_change = any(int(m2[h]["assignment_change_correct_count"]) > 0 for h in horizons)
    checks = {
        "actual_full_training": training.get("actual_full_training") is True and training.get("status") == "PASS",
        "checkpoint_reload_hash": bool(training.get("checkpoint_sha256")) and sha256(CHECKPOINT) == training.get("checkpoint_sha256"),
        "production_authorized_false": training.get("production_authorized") is False,
        "runtime_posthoc_integrity": integrity.get("status") == "PASS",
        "global_assignment_exercised": int(result["effects"]["incremental"]["M2"]["100"]["assignment_change_count"]) > 0,
        "correct_change_observed": positive_correct_change,
        "strict_positive_ci_all_horizons": strict_ci,
        "no_incorrect_changes_all_horizons": no_incorrect,
        "untouched_regression_clean_all_horizons": untouched_clean,
        "runtime_future_gt_false": result.get("protocol", {}).get("runtime_future_gt_used") is False,
        "gt_only_posthoc": result.get("protocol", {}).get("gt_loaded_only_after_runtime_validation") is True,
        "equal_sequence_bootstrap": result.get("protocol", {}).get("bootstrap") == "sequence_mean_then_equal_sequence_cluster_bootstrap",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "standard_mot_not_computable": result.get("id_switch_metric") == "NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT",
        "n44_checkpoint_unchanged": integrity.get("gate_checks", {}).get("n44_checkpoint_untouched") is True,
        "n45_result_preserved": N45_RESULT.is_file(),
    }
    status = "N47_COMPLETED_GATE_FAILED"
    gate = {
        "schema": "N47_GLOBAL_ASSIGNMENT_PROBE_GATE_V1",
        "status": status,
        "research_gate": "FAIL_H100_ROBUSTNESS_UNTOUCHED_AND_REAL_INPUT",
        "checks": checks,
        "semantic_interpretation": {
            "global_structure_exercised": True,
            "short_horizon_signal": "M2 H20/H50 has positive equal-sequence CI lower bounds and correct changes",
            "robust_efficacy": False,
            "failure": "M2 H100 utility is negative, H100 has incorrect changes, and strict untouched-ID regression is not clean",
            "memory_effect_separate": "N42/N45 M2 no-write→write memory effect remains negative and is not credited to N47",
            "counterfactual_oracle_not_used": True,
        },
        "m2_incremental": {h: {key: m2[h][key] for key in ("identity_utility", "target_iou_delta", "future_identity_error_reduction", "recorrection_proxy_reduction", "assignment_change_count", "assignment_change_correct_count", "assignment_change_incorrect_count", "assignment_change_neutral_count", "assignment_no_change_count", "untouched_regression", "sequence_cluster_bootstrap_95ci")} for h in horizons},
        "memory_context": load(N45_RESULT)["effects"]["memory"]["M2"],
        "provenance": {"interaction_source": "simulated_from_gt", "real_human_tape": False, "real_sam3_full_loop": False, "runtime_future_gt_used": False, "gt_loaded_posthoc": True},
        "authorization": {"calibration_head": "NOT_AUTHORIZED", "decoder_lora": "NOT_AUTHORIZED", "production_interface_changed": False, "checkpoint_production_authorized": False},
        "failure_root_cause": "The global full-matrix/Hungarian structure removes N46's owner-by-column bottleneck and produces genuine correct short-horizon changes, but the fixed trained logit is not robust through H100: utility turns negative, incorrect changes appear, and untouched-ID protection fails. The negative memory effect and missing real input remain separate blockers.",
        "next_action": "Do not tune or promote this probe. Obtain N40-compliant real human tape and real candidate-complete SAM3 full-loop; only then preregister any further global/listwise experiment.",
        "runtime_future_gt_used": False,
    }
    report = f"""# N47 Global Candidate-to-Public-ID Assignment Probe\n\nDate: {now()}  \nStatus: `{status}`. This closes only the isolated N47 diagnostic probe; the broader scientific objective remains open.\n\n## Hypothesis and frozen protocol\n\nN46's conclusion was audited as not strong enough to rule out a structural fix. Its `21790/21818` owner-by-column blocked oracle pairs, sparse proposals, and `4.401657` median required delta versus fixed `0.25` support a falsifiable alternative: predict a candidate-level appearance logit for the complete candidate×public-ID matrix and let one global Hungarian solver with explicit NONE decide assignments, allowing swaps.\n\nThe protocol was frozen before training in [probe_protocol.json]({PROTOCOL}). It uses the frozen N42 runtime, the N42 sequence-level split, seed `4747`, pairwise softplus ranking plus fixed logit L2, no holdout selection, and eight causal features. Public-ID, target identity, GT, future outcomes and sequence-name encoding are not runtime features. N44's checkpoint and all production MOT/OVMOT paths were untouched.\n\n## Smoke, training and runtime\n\nThe deterministic smoke passed: a synthetic off-diagonal logit caused assignment `[1,0]`, while NONE returned only dummy columns `[2,3]`. The first CPU attempt produced only a partial epoch-2 checkpoint and is preserved; the same frozen protocol then completed actual sequence-disjoint training on GPU0: 611,451 labelled cells, 404,584 pairs, 7 epochs, best epoch 2, checkpoint SHA-256 `{training.get('checkpoint_sha256')}`. The checkpoint explicitly records `production_authorized=false`.\n\nThe complete runtime replay used 24 events × 5 variants × 100 frames = 12,000 frames. It retained full candidate rows, native IDs, boxes, confidences, score matrices, public-ID axes, assignment columns, predicted logits and direct `runtime_future_gt_used=false`. Independent integrity recomputed every global Hungarian assignment and passed. GT was loaded only after runtime validation for simulated posthoc labels.\n\n## Results\n\n| Effect / variant | H20 utility | H50 utility | H100 utility |\n|---|---:|---:|---:|\n| N47 M2 write→global-plus | {m2['20']['identity_utility']:.9f} | {m2['50']['identity_utility']:.9f} | {m2['100']['identity_utility']:.9f} |\n| N47 M2 assignment changes | {m2['20']['assignment_change_count']} ({m2['20']['assignment_change_correct_count']} correct / {m2['20']['assignment_change_incorrect_count']} incorrect) | {m2['50']['assignment_change_count']} ({m2['50']['assignment_change_correct_count']} / {m2['50']['assignment_change_incorrect_count']}) | {m2['100']['assignment_change_count']} ({m2['100']['assignment_change_correct_count']} / {m2['100']['assignment_change_incorrect_count']}) |\n\nM2 has a real short-horizon structural signal: H20/H50 equal-sequence bootstrap lower bounds are `{m2['20']['sequence_cluster_bootstrap_95ci']['lower']:.9f}` and `{m2['50']['sequence_cluster_bootstrap_95ci']['lower']:.9f}`, with correct changes. It is not robust: H100 utility is `{m2['100']['identity_utility']:.9f}`, its CI lower bound is `{m2['100']['sequence_cluster_bootstrap_95ci']['lower']:.9f}`, and incorrect changes are present. Untouched-ID regression is therefore not clean. Neutral changes are not counted as correct.\n\nThe separate N45/N42 M2 memory effect remains negative and is reported separately in [N45 normalized attribution](./docs/N45_ATTRIBUTION_REPAIR_FINAL_REPORT.md). Standard MOT/TrackEval metrics remain `NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT`.\n\n## Provenance and gate\n\nAll 24 events are `simulated_from_gt`; no record was relabelled as human. N40 remains `BLOCKED_INPUT_REAL_HUMAN_TAPE`: the required direct public_id, human-confirmed BOX/CLICK/CONFIRMED_MASK, lossless ROI digest, UI/session/annotator timestamps and candidate-complete future rows are unavailable. Real SAM3 full-loop is also absent. Consequently this probe does not authorize calibration, decoder LoRA or production changes.\n\nThe machine-readable gate is [n47_final_gate.json](./outputs/n47_global_probe/n47_final_gate.json). All failed attempts are retained under [outputs/n47_global_probe/attempts](./outputs/n47_global_probe/attempts).\n\n## Decision\n\nN47 is `COMPLETED_GATE_FAILED`, not a positive efficacy result. The global assignment interface is scientifically informative because it produces correct short-horizon changes that N46's local gate could not express, but the fixed probe fails robust all-horizon and untouched-ID gates. No threshold, seed, metric, LoRA, checkpoint replacement or production interface was changed. The broader objective remains open pending real human tape and real candidate-complete SAM3 full-loop evidence.\n"""
    write_json(GATE, gate)
    stage = {"status": status, "protocol": "N47_STAGE_05_FINAL_GATE_V1", "command": ["python", "scripts/n47_stage05_finalize.py"], "inputs": {"probe_result": str(RESULT), "integrity": str(INTEGRITY), "training_manifest": str(TRAIN_MANIFEST), "n40_blocker": str(N40_BLOCKER)}, "outputs": {"gate": str(GATE), "report": str(REPORT), "stage_status": str(STAGE)}, "metrics": {"m2_incremental": gate["m2_incremental"], "global_structure_exercised": True, "real_input_status": blocker.get("status")}, "gate_checks": checks, "failure_root_cause": gate["failure_root_cause"], "next_action": gate["next_action"], "runtime_future_gt_used": False, "gt_loaded_posthoc": True, "finished_at": now()}
    write_json(STAGE, stage); REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"status": status, "gate": str(GATE), "report": str(REPORT), "m2_h100": m2["100"]["identity_utility"], "strict_ci_all_horizons": strict_ci}))


if __name__ == "__main__":
    main()
