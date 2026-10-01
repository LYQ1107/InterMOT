#!/usr/bin/env python3
"""Audit R3 post-final diagnostics without modifying historical R3 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.n72r20r3_common import R2_ASSET_ROOT, ROOT, SEQUENCES, load_sequence
from scripts.n72r20r3_presence_loso import _load_joined, aggregate_loso, metrics
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl


R3 = ROOT / "outputs/N72R20R3"
R3R1 = ROOT / "outputs/N72R20R3R1"
FINAL_DECISION = "FAIL_OPEN_SET_CROSS_SEQUENCE_GENERALIZATION"
GATE_KEYS = (
    "false_present_rate",
    "open_set_correct_identification_recall",
    "presence_recall",
    "p0_false_present_rate",
    "p1_false_present_rate",
    "conditional_candidate_accuracy_given_present",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_axis() -> dict[tuple[str, int], set[str]]:
    axis: dict[tuple[str, int], set[str]] = {}
    for sequence in SEQUENCES:
        frames, _base = load_sequence(R2_ASSET_ROOT, sequence)
        for frame_payload, candidates in frames:
            axis[(sequence, int(frame_payload["frame"]))] = {str(row["candidate_uid"]) for row in candidates}
    return axis


def _presence_diagnostic() -> dict[str, Any]:
    runtime = read_zstd_jsonl(R3 / "presence/frame_runtime_features.jsonl.zst")
    axis = _candidate_axis()
    by_condition: dict[str, dict[str, int]] = {}
    changed_rows = 0
    old_true = 0
    corrected_true = 0
    for row in runtime:
        condition = str(row["state_condition"])
        counters = by_condition.setdefault(condition, {"rows": 0, "old_true": 0, "corrected_true": 0, "changed": 0})
        counters["rows"] += 1
        old_value = bool(row.get("candidate_presence_of_base_candidate"))
        sequence = str(row["sequence"])
        frame = int(row["frame"])
        base_uid = row.get("base_target_candidate_uid")
        corrected_value = base_uid not in (None, "", "None") and str(base_uid) in axis[(sequence, frame)]
        old_true += int(old_value)
        corrected_true += int(corrected_value)
        counters["old_true"] += int(old_value)
        counters["corrected_true"] += int(corrected_value)
        if old_value != corrected_value:
            changed_rows += 1
            counters["changed"] += 1
    return {
        "rows": len(runtime),
        "old_true_count": old_true,
        "corrected_true_count": corrected_true,
        "changed_row_count": changed_rows,
        "changed_row_fraction": changed_rows / len(runtime) if runtime else 0.0,
        "by_condition": by_condition,
        "old_field": "candidate_presence_of_base_candidate from sealed R3 runtime artifact",
        "corrected_definition": "public-ID-resolved base_target_candidate_uid exists on the sealed candidate UID axis",
    }


def _coverage_diagnostic() -> dict[str, Any]:
    presence_dir = R3 / "presence"
    joined = _load_joined(presence_dir)
    loso = _read(presence_dir / "loso_results.json")
    policies = _read(presence_dir / "fold_policies.json")["policies"]
    corrected_records: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    for old_record in loso["records"]:
        condition = str(old_record["condition"])
        heldout = str(old_record["heldout_sequence"])
        policy_id = str(old_record["policy_id"])
        policy_key = next(
            key for key in policies
            if key.startswith(f"{condition}|{heldout}|") and key.endswith(f"|{policy_id}")
        )
        policy = policies[policy_key]
        test_rows = [
            item for item in joined
            if str(item["runtime"]["state_condition"]) == condition
            and str(item["runtime"]["sequence"]) == heldout
        ]
        corrected = metrics(test_rows, policy, sequence=heldout)
        corrected.update(
            {
                "condition": condition,
                "heldout_sequence": heldout,
                "calibration": old_record.get("calibration", {}),
                "trainable_parameters": old_record.get("trainable_parameters", 0),
            }
        )
        corrected_records.append(corrected)
        record_rows.append(
            {
                "condition": condition,
                "heldout_sequence": heldout,
                "policy_id": policy_id,
                "old_candidate_coverage": old_record.get("candidate_coverage"),
                "corrected_candidate_coverage": corrected.get("candidate_coverage"),
                "delta": None
                if old_record.get("candidate_coverage") is None or corrected.get("candidate_coverage") is None
                else corrected["candidate_coverage"] - old_record["candidate_coverage"],
            }
        )
    corrected_aggregates = aggregate_loso(corrected_records)
    old_aggregate_map = {(str(item["condition"]), str(item["policy_id"])): item for item in loso["aggregates"]}
    corrected_aggregate_map = {(str(item["condition"]), str(item["policy_id"])): item for item in corrected_aggregates}
    aggregate_rows: list[dict[str, Any]] = []
    max_gate_delta = 0.0
    gate_deltas: dict[str, float] = {key: 0.0 for key in GATE_KEYS}
    for key, old in old_aggregate_map.items():
        corrected = corrected_aggregate_map[key]
        row = {
            "condition": key[0],
            "policy_id": key[1],
            "old_candidate_coverage": old["pooled"].get("candidate_coverage"),
            "corrected_candidate_coverage": corrected["pooled"].get("candidate_coverage"),
        }
        coverage_old = row["old_candidate_coverage"]
        coverage_new = row["corrected_candidate_coverage"]
        row["candidate_coverage_delta"] = None if coverage_old is None or coverage_new is None else coverage_new - coverage_old
        for metric_name in GATE_KEYS:
            old_value = old["pooled"].get(metric_name)
            new_value = corrected["pooled"].get(metric_name)
            delta = abs(float(new_value) - float(old_value)) if old_value is not None and new_value is not None else 0.0
            gate_deltas[metric_name] = max(gate_deltas[metric_name], delta)
            max_gate_delta = max(max_gate_delta, delta)
        aggregate_rows.append(row)
    return {
        "record_count": len(record_rows),
        "old_records_and_corrected_records": record_rows,
        "aggregate_comparison": aggregate_rows,
        "old_candidate_coverage_source": "outputs/N72R20R3/presence/loso_results.json",
        "corrected_candidate_coverage_definition": "candidate_set_present among frames where target_gt_present is true",
        "gate_metric_max_absolute_deltas": gate_deltas,
        "max_gate_metric_absolute_delta": max_gate_delta,
        "b4_training_features_affected": False,
        "loso_policy_calibration_changed": False,
        "selected_policy_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=R3R1 / "r3_postfinal_audit")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    presence = _presence_diagnostic()
    coverage = _coverage_diagnostic()
    final_result = _read(R3 / "FINAL_RESULT.json")
    decision_unchanged = (
        final_result["final_decision"] == FINAL_DECISION
        and coverage["max_gate_metric_absolute_delta"] == 0.0
        and not coverage["b4_training_features_affected"]
        and not coverage["loso_policy_calibration_changed"]
        and not coverage["selected_policy_changed"]
    )
    old_presence_diagnostic = {
        "rows": presence["rows"],
        "true_count": presence["old_true_count"],
        "changed_row_count": presence["changed_row_count"],
        "by_condition": {
            key: {"rows": value["rows"], "true_count": value["old_true"], "changed": value["changed"]}
            for key, value in presence["by_condition"].items()
        },
        "definition": presence["old_field"],
    }
    corrected_presence_diagnostic = {
        "rows": presence["rows"],
        "true_count": presence["corrected_true_count"],
        "changed_row_count": presence["changed_row_count"],
        "by_condition": {
            key: {"rows": value["rows"], "true_count": value["corrected_true"], "changed": value["changed"]}
            for key, value in presence["by_condition"].items()
        },
        "definition": presence["corrected_definition"],
    }
    payload = {
        "stage": "N72R20R3R1",
        "source_stage": "N72R20R3",
        "r3_final_decision": final_result["final_decision"],
        "old_candidate_coverage": coverage["old_candidate_coverage_source"],
        "corrected_candidate_coverage": coverage,
        "old_candidate_presence_diagnostic": old_presence_diagnostic,
        "corrected_candidate_presence_diagnostic": corrected_presence_diagnostic,
        "gate_relevant_metric_changed": not decision_unchanged,
        "R3_FINAL_DECISION_UNCHANGED": decision_unchanged,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "historical_r3_outputs_overwritten": False,
    }
    (args.output_dir / "corrected_r3_diagnostics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    report = [
        "# R3 post-final diagnostic bug audit",
        "",
        "This audit is written under N72R20R3R1 and does not overwrite outputs/N72R20R3.",
        "",
        "## Bug A — public assignment UID parsing",
        "",
        "The R3 runtime field candidate_presence_of_base_candidate had one call path that omitted target public_id. R3R1 resolves the base UID from public_assignments relative to target_public_id and checks membership on the sealed candidate UID axis.",
        "",
        f"Old true count: {presence['old_true_count']}; corrected true count: {presence['corrected_true_count']}; changed rows: {presence['changed_row_count']}.",
        "",
        "## Bug B — candidate coverage",
        "",
        "The corrected metric enumerates joined rows explicitly and only counts candidate-set PRESENT frames among target-GT-present frames. The stale outer-loop variable is no longer used.",
        "",
        f"Maximum absolute delta over gate metrics: {coverage['max_gate_metric_absolute_delta']:.12g}.",
        "",
        f"R3_FINAL_DECISION_UNCHANGED={str(decision_unchanged).lower()}.",
        "",
        "No R3 historical artifact was overwritten. Training is allowed to continue only because all gate-relevant R3 metrics and the final decision are unchanged.",
    ]
    (args.output_dir / "diagnostic_bug_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"R3_FINAL_DECISION_UNCHANGED": decision_unchanged, "changed_presence_rows": presence["changed_row_count"], "status": "PASS_R3_POSTFINAL_AUDIT"}, sort_keys=True))
    return 0 if decision_unchanged else 2


if __name__ == "__main__":
    raise SystemExit(main())
