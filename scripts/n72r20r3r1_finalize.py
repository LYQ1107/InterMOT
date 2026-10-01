#!/usr/bin/env python3
"""Write terminal N72R20R3R1 reports after the static gate."""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from scripts.n72r20r3_common import ROOT, SEQUENCES, write_zstd_jsonl
from scripts.n72r20r3r1_train_eval import _json, _write_json, aggregate_seed_predictions


OUT = ROOT / "outputs/N72R20R3R1"
DECISION = "FAIL_EXPLICIT_NONE_CROSS_SEQUENCE_VERIFICATION"


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def _headroom() -> dict[str, Any]:
    predictions_by_seed: list[list[dict[str, Any]]] = []
    worker_paths = sorted(Path(path) for path in glob.glob(str(OUT / "workers" / "V2_DUAL_STATE__seed*/worker_result.json")))
    for path in worker_paths:
        payload = _json(path)
        seed = str(payload["seeds"][0])
        predictions_by_seed.append(payload["prediction_archive"]["V2_DUAL_STATE"][seed])
    predictions = aggregate_seed_predictions(predictions_by_seed)
    labels = read_zstd_jsonl(ROOT / "outputs/N72R20R3/presence/frame_posthoc_labels.jsonl.zst")
    label_map = {(str(row["sequence"]), int(row["frame"]), str(row["state_condition"])): row for row in labels}
    rescue: list[dict[str, Any]] = []
    verifier_correct = 0
    base_correct = 0
    base_wrong = 0
    predicted_candidates = 0
    for prediction in predictions:
        key = (str(prediction["sequence"]), int(prediction["frame"]), str(prediction["state_condition"]))
        label = label_map[key]
        truth_uid = label.get("best_target_candidate_uid") if label.get("candidate_set_present") else None
        base_uid = label.get("base_candidate_uid")
        predicted_uid = prediction.get("predicted_uid")
        if predicted_uid is not None:
            predicted_candidates += 1
        if truth_uid is not None and base_uid == truth_uid:
            base_correct += 1
        if truth_uid is not None and base_uid != truth_uid:
            base_wrong += 1
            if predicted_uid == truth_uid:
                verifier_correct += 1
                if len(rescue) < 100:
                    rescue.append({"sequence": key[0], "frame": key[1], "state_condition": key[2], "base_uid": base_uid, "verifier_uid": predicted_uid, "truth_uid": truth_uid})
    return {
        "stage": "N72R20R3R1",
        "status": "READ_ONLY_STATIC_HEADROOM_ONLY",
        "source": "frozen R3 posthoc labels + R3R1 static predictions",
        "association_rescue_run": False,
        "solver_called": False,
        "public_authority_changed": False,
        "base_assignment_unchanged": True,
        "candidate_created": False,
        "runtime_future_gt_used": False,
        "candidate_predictions": predicted_candidates,
        "base_correct_present_frames": base_correct,
        "base_wrong_present_frames": base_wrong,
        "verifier_correct_when_base_wrong": verifier_correct,
        "examples": rescue,
        "interpretation": "Logged only; no verifier prediction was applied to association because the static gate failed.",
    }


def _causal_not_run() -> None:
    causal_dir = OUT / "causal"
    causal_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "N72R20R3R1",
        "status": "NOT_RUN_STATIC_GATE_FAILED",
        "causal_commit_replay_authorized": False,
        "reason": DECISION,
        "runtime_future_gt_used": False,
        "human_anchor_changed": False,
        "public_authority_changed": False,
        "candidate_created": False,
        "wrong_write_rate": None,
        "correct_write_retention_vs_r2_c0": None,
        "accepted_writes": None,
        "wrong_writes": None,
        "correct_writes": None,
    }
    _write_json(causal_dir / "causal_replay.json", payload)
    _write_json(causal_dir / "per_sequence.json", {"stage": "N72R20R3R1", "status": payload["status"], "sequences": {sequence: {"status": payload["status"], "writes": None, "wrong_writes": None, "correct_writes": None} for sequence in SEQUENCES}})
    write_zstd_jsonl(causal_dir / "memory_write_audit.jsonl.zst", [])


def _f(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _report_text(
    *,
    result: Mapping[str, Any],
    selected: Mapping[str, Any],
    loso: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    baseline: Mapping[str, Any],
    per_sequence: Mapping[str, Any],
    selection: Mapping[str, Any],
    audit: Mapping[str, Any],
    tests_summary: str,
) -> str:
    lines = [
        "NO — explicit NONE-class supervision did not make the frozen InterMOT identity representation a safe and useful cross-sequence open-set identity verifier.",
        "",
        "# InterMOT N72R20R3R1 — Explicit-NONE Identity Verification",
        "",
        "Final decision: `FAIL_EXPLICIT_NONE_CROSS_SEQUENCE_VERIFICATION`.",
        "",
        "The primary question was whether a human-confirmed identity can be recognized against hard competing identities while rejecting candidate-set absence. The frozen-backbone verifier failed the preregistered static gate, so no causal memory replay or association authority was opened.",
        "",
        "## Goal and protocol",
        "",
        f"- Goal: `{result['goal']}`; central question: {result['central_question']}",
        f"- Source stage: N72R20R3; R3 decision unchanged after Phase 0 audit: `{audit['R3_FINAL_DECISION_UNCHANGED']}`.",
        "- Data lineage: eight frozen DanceTrack train sequences, 8,414 future frames, S0/S1 frozen states; GT was used only for training/posthoc labels.",
        "- Formal protocol: 8 sequence-held-out folds × seeds 720301/720302/720303; no frame-IID split.",
        "- OSNet, N72R18 GRU, candidate generation, solver, public authority and SAM3 were frozen.",
        "",
        "## Primary static result",
        "",
        f"Selected primary model: `{selection['selected_model']}`. Pooled negative FPR = `{_f(selected.get('false_present_rate'))}`; open-set correct-ID recall = `{_f(selected.get('open_set_correct_identification_recall'))}`; macro recall = `{_f(selected.get('macro_open_set_correct_identification_recall'))}`; macro FPR = `{_f(selected.get('macro_false_present_rate'))}`.",
        "",
        "| Method | Params | Neg FPR | P0 FPR | P1 FPR | Open-ID recall | Macro recall | NONE recall | Candidate accuracy | Causal wrong-write | Causal retention |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    baseline_values = baseline.get("pooled_by_condition", [])
    if baseline_values:
        lines.append(
            f"| R3 B4 Logistic (S0/S1 pooled rows) | 11 | {_f(sum(float(x.get('false_present_rate', 0.0)) for x in baseline_values) / len(baseline_values))} | {_f(sum(float(x.get('p0_false_present_rate', 0.0)) for x in baseline_values) / len(baseline_values))} | {_f(sum(float(x.get('p1_false_present_rate', 0.0)) for x in baseline_values) / len(baseline_values))} | {_f(sum(float(x.get('open_set_correct_identification_recall', 0.0)) for x in baseline_values) / len(baseline_values))} | — | — | — | not run | not run |"
        )
    for name in ("V1_PAIRWISE_THRESHOLD_NONE", "V2_ANCHOR_ONLY", "V2_LEARNED_STATE_ONLY", "V2_DUAL_STATE"):
        item = loso["reports"][name]
        params = 39842 if name == "V2_DUAL_STATE" else 36626
        lines.append(
            f"| {name} | {params} | {_f(item.get('false_present_rate'))} | {_f(item.get('p0_false_present_rate'))} | {_f(item.get('p1_false_present_rate'))} | {_f(item.get('open_set_correct_identification_recall'))} | {_f(item.get('macro_open_set_correct_identification_recall'))} | {_f(item.get('none_recall_on_absent'))} | {_f(item.get('candidate_top1_accuracy_on_present'))} | not run | not run |"
        )
    lines += [
        "",
        "Gate thresholds were fixed before formal held-out evaluation: pooled FPR ≤ 0.02, pooled open-ID recall ≥ 0.60, macro recall ≥ 0.40, macro FPR ≤ 0.05. The selected model failed all four performance checks; all eight sequences were nevertheless reported, and no candidate was created.",
        "",
        "## Sequence-cluster uncertainty",
        "",
        f"The 2,000-repetition sequence-cluster bootstrap used seed `{bootstrap['seed']}`. 95% intervals: negative FPR `{bootstrap['intervals_95']['negative_fpr']}`, open-ID recall `{bootstrap['intervals_95']['open_recall']}`, macro recall `{bootstrap['intervals_95']['macro_recall']}`.",
        "",
        "## Frozen primary model by sequence",
        "",
        "| Sequence | Absent | Present | FPR | P0 FPR | P1 FPR | Open-ID recall | Candidate accuracy | NONE recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sequence in SEQUENCES:
        item = per_sequence[sequence]
        lines.append(
            f"| {sequence} | {item.get('candidate_set_negative_count', 0)} | {item.get('candidate_set_positive_count', 0)} | {_f(item.get('false_present_rate'))} | {_f(item.get('p0_false_present_rate'))} | {_f(item.get('p1_false_present_rate'))} | {_f(item.get('open_set_correct_identification_recall'))} | {_f(item.get('candidate_top1_accuracy_on_present'))} | {_f(item.get('none_recall_on_absent'))} |"
        )
    lines += [
        "",
        "## Phase 0 and lineage audit",
        "",
        f"- Bug A corrected diagnostic: public-ID-resolved base candidate presence changed 13,778 rows, but gate metric delta was 0.0.",
        f"- Bug B corrected candidate coverage changed from 0.670135 to 0.849673 in the aggregate diagnostic, but B4 features, calibration, gate metrics and R3 final decision were unchanged.",
        f"- Historical R3 SHA audit: `{json.dumps(result['r3_immutability'], sort_keys=True)}`.",
        "- The R3 historical output directory was not overwritten.",
        "",
        "## Causal and association boundary",
        "",
        "Static failure means `causal_commit_replay_authorized=false`. Causal memory writes, solver redesign, association rescue, SAM3 rerun, DanceTrack val/test, LoRA, Transformer, GRU training and R4 authority were not run. Future-association headroom is logged read-only and was not applied.",
        "",
        "## Tests and known baseline issue",
        "",
        f"R3R1 focused tests and full pytest result: {tests_summary}. The four failures are the fixed-version TrackEval old CLI handling of `SEQMAP_FILE` (the parser stores the argument as a list and later passes it to `os.path.isfile`), not evidence that all tests pass.",
        "",
        "## Terminal interpretation",
        "",
        "The result is a negative answer to the only N72R20R3R1 question: explicit NONE supervision, as implemented in this small verifier over the frozen representation, did not provide reliable cross-sequence open-set identity recognition. `next_association_authority_stage_authorized=false`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-summary", default="PENDING")
    args = parser.parse_args()
    selected = _json(OUT / "static_eval/selected_report.json")
    loso = _json(OUT / "static_eval/loso_results.json")
    bootstrap = _json(OUT / "static_eval/bootstrap.json")
    baseline = _json(OUT / "static_eval/baseline_v0_r3_b4.json")
    per_sequence = _json(OUT / "static_eval/per_sequence_results.json")["records"]
    selection = _json(OUT / "training/model_selection.json")
    audit = _json(OUT / "r3_postfinal_audit/corrected_r3_diagnostics.json")
    immutability = _json(OUT / "r3_immutability_audit.json")
    storage_after = _json(OUT / "storage_audit_after.json")
    checkpoint_manifest = _json(OUT / "training/checkpoint_manifest.json")
    final_goal = _json(OUT / "FINAL_GOAL.json")
    _causal_not_run()
    headroom = _headroom()
    _write_json(OUT / "future_association_headroom.json", headroom)
    result = {
        "stage": "N72R20R3R1",
        "goal": "Explicit-NONE Identity Verification",
        "central_question": final_goal["central_question"],
        "final_decision": DECISION,
        "source_commit": checkpoint_manifest["records"][0].get("source_commit", "75696491945456849f5ebdb75fc785e087d8fb41"),
        "final_commit": _git_head(),
        "R3_final_decision_unchanged": bool(audit["R3_FINAL_DECISION_UNCHANGED"]),
        "phase0_bug_fixes": {"public_assignment_uid": True, "candidate_coverage": True, "gate_relevant_change": bool(audit["gate_relevant_metric_changed"])},
        "corrected_diagnostics": "outputs/N72R20R3R1/r3_postfinal_audit/corrected_r3_diagnostics.json",
        "architecture": selection["selected_model"],
        "trainable_parameters": 39842,
        "model_checkpoint_sha256": [item["sha256"] for item in checkpoint_manifest["records"] if item["model"] == selection["selected_model"]],
        "loso_folds": 8,
        "seeds": [720301, 720302, 720303],
        "static_pooled_metrics": {key: selected.get(key) for key in ("false_present_rate", "open_set_correct_identification_recall", "presence_recall", "none_recall_on_absent", "candidate_top1_accuracy_on_present", "nll", "brier", "ece", "auroc_present", "auprc_present")},
        "static_macro_metrics": {key: selected.get(key) for key in ("macro_false_present_rate", "macro_open_set_correct_identification_recall")},
        "static_gate": selected["static_gate"],
        "per_sequence_metrics": per_sequence,
        "bootstrap_intervals": bootstrap,
        "p0_p1_metrics": {name: {key: loso["reports"][name].get(key) for key in ("p0_false_present_rate", "p1_false_present_rate", "p0_count", "p1_count")} for name in loso["reports"]},
        "causal_metrics": None,
        "causal_commit_replay_authorized": False,
        "OSNet_frozen": True,
        "GRU_frozen": True,
        "solver_frozen": True,
        "public_authority_frozen": True,
        "runtime_future_gt_used": False,
        "association_rescue_run": False,
        "val_accessed": False,
        "test_accessed": False,
        "sam3_rerun": False,
        "training_required": False,
        "storage_free_gib_after": storage_after.get("filesystem", {}).get("free_gib"),
        "r3_immutability_all_unchanged": bool(immutability["all_unchanged"]),
        "r3_immutability": {"all_unchanged": bool(immutability["all_unchanged"]), "source_commit": immutability.get("source_commit"), "files": immutability.get("after", {}).get("files", [])},
        "next_association_authority_stage_authorized": False,
        "tests_summary": args.tests_summary,
    }
    _write_json(OUT / "FINAL_RESULT.json", result)
    _write_json(
        OUT / "stage_status.json",
        {
            "stage": "N72R20R3R1",
            "goal": "Explicit-NONE Identity Verification",
            "goal_reference": "outputs/N72R20R3R1/FINAL_GOAL.json",
            "goal_frozen": True,
            "phase": "STATIC_FAIL",
            "status": DECISION,
            "last_artifact": "outputs/N72R20R3R1/FINAL_RESULT.json",
            "R3_FINAL_DECISION_UNCHANGED": bool(audit["R3_FINAL_DECISION_UNCHANGED"]),
            "training_index": "PASS_R3R1_TRAINING_INDEX_COMPLETE",
            "formal_loso": "PASS_FORMAL_LOSO_COMPLETE",
            "static_gate": selected["static_gate"],
            "causal_replay": "NOT_RUN_STATIC_GATE_FAILED",
            "causal_commit_replay_authorized": False,
            "next_association_authority_stage_authorized": False,
            "runtime_future_gt_used": False,
            "sam3_rerun": False,
            "val_accessed": False,
            "test_accessed": False,
            "storage_audit_after": storage_after,
            "tests": args.tests_summary,
        },
    )
    report = _report_text(result=result, selected=selected, loso=loso, bootstrap=bootstrap, baseline=baseline, per_sequence=per_sequence, selection=selection, audit=audit, tests_summary=args.tests_summary)
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": DECISION, "selected_model": selection["selected_model"], "static_gate_pass": selected["static_gate"]["pass"], "causal": "NOT_RUN_STATIC_GATE_FAILED"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
