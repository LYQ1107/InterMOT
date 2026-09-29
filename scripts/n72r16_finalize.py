#!/usr/bin/env python3
"""Apply the frozen N72R16 decision gates and stop at the identity question."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


GOAL = "Human Identity Representation Probe"
QUESTION = "Can one human-confirmed identity observation reliably recognize the same person against hard competing identities over future frames?"


def h100(summary: dict[str, object]) -> dict[str, object]:
    return summary["horizons"]["H100"]


def gate(row: dict[str, object]) -> dict[str, object]:
    ci = row["hard_negative_win_rate_sequence_cluster_ci95"]
    win = row["hard_negative_win_rate"]
    median = row["median_margin"]
    lower = ci["lower"]
    upper = ci["upper"]
    return {
        "clear_identity_signal": bool(lower is not None and median is not None and lower > 0.50 and median > 0.0),
        "strong_identity_signal": bool(
            win is not None
            and lower is not None
            and median is not None
            and win >= 0.70
            and lower >= 0.60
            and median > 0.03
        ),
        "weak_or_fail_signal": bool((upper is not None and upper <= 0.55) or (median is not None and median <= 0.0)),
        "win_rate": win,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "median_margin": median,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
    val_identity = aggregate.get("identity", {}).get("val")
    val_memory = aggregate.get("memory", {}).get("val")
    if not val_identity:
        raise FileNotFoundError("val single-anchor identity result is required")
    single = gate(h100(val_identity))
    memory_gates: dict[str, object] = {}
    memory_passes: list[str] = []
    if val_memory:
        for variant, summary in val_memory.get("variants", {}).items():
            row = gate(h100(summary))
            single_win = single["win_rate"]
            single_median = single["median_margin"]
            row["improvement_over_single_anchor_win_rate"] = (
                h100(summary)["hard_negative_win_rate"] - single_win
                if h100(summary)["hard_negative_win_rate"] is not None and single_win is not None
                else None
            )
            row["improvement_over_single_anchor_median_margin"] = (
                h100(summary)["median_margin"] - single_median
                if h100(summary)["median_margin"] is not None and single_median is not None
                else None
            )
            row["persistent_memory_pass"] = bool(
                variant != "M1"
                and row["strong_identity_signal"]
                and row["improvement_over_single_anchor_win_rate"] is not None
                and row["improvement_over_single_anchor_win_rate"] >= 0.05
                and row["improvement_over_single_anchor_median_margin"] is not None
                and row["improvement_over_single_anchor_median_margin"] >= 0.01
            )
            if row["persistent_memory_pass"]:
                memory_passes.append(variant)
            memory_gates[variant] = row

    if single["strong_identity_signal"]:
        decision = "PASS_STRONG_SINGLE_ANCHOR_IDENTITY"
        rationale = "Validation H100 single-anchor hard-negative gate is strong."
    elif memory_passes:
        decision = "PASS_PERSISTENT_MEMORY_IMPROVES_IDENTITY"
        rationale = f"Validation H100 oracle memory passes the strong gate and improves M1: {memory_passes}."
    elif single["weak_or_fail_signal"]:
        decision = "FAIL_IDENTITY_REPRESENTATION"
        rationale = "Validation H100 hard-negative signal is weak or non-positive under the frozen gate."
    else:
        decision = "AMBIGUOUS_IDENTITY_REPRESENTATION"
        rationale = "Validation H100 evidence is between the frozen pass and fail gates."
    authorized = decision in {"PASS_STRONG_SINGLE_ANCHOR_IDENTITY", "PASS_PERSISTENT_MEMORY_IMPROVES_IDENTITY"}
    result = {
        "stage": "N72R16",
        "goal": GOAL,
        "central_question": QUESTION,
        "primary_endpoint": "VAL_H100_HARD_NEGATIVE_IDENTITY_WIN_RATE",
        "decision": decision,
        "rationale": rationale,
        "single_anchor_gate": single,
        "persistent_memory_gates": memory_gates,
        "next_association_stage_authorized": authorized,
        "stop_rule_applied": True,
        "forbidden_next_steps_without_new_stage": [
            "SAM3 inference",
            "Hungarian redesign",
            "persistent association integration",
            "requery",
            "TrackEval/full MOT",
            "LoRA",
            "identity decoder training",
            "frequency ablation",
            "intervention-strength ablation",
        ],
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "FINAL_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stage_status_path = args.results_dir / "stage_status.json"
    stage_status = json.loads(stage_status_path.read_text(encoding="utf-8")) if stage_status_path.is_file() else {}
    stage_status.update(
        {
            "stage": "N72R16",
            "goal": GOAL,
            "central_question": QUESTION,
            "goal_frozen": True,
            "status": "COMPLETE",
            "decision": decision,
            "next_association_stage_authorized": authorized,
            "primary_endpoint": "VAL_H100_HARD_NEGATIVE_IDENTITY_WIN_RATE",
        }
    )
    stage_status_path.write_text(json.dumps(stage_status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = args.report.read_text(encoding="utf-8")
    marker = "## Final decision"
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + "\n\n"
    report += (
        "## Final decision\n\n"
        f"- Decision: `{decision}`\n"
        f"- Rationale: {rationale}\n"
        f"- `NEXT_ASSOCIATION_STAGE_AUTHORIZED`: `{str(authorized).lower()}`\n"
        f"- Primary endpoint: `VAL_H100_HARD_NEGATIVE_IDENTITY_WIN_RATE`\n"
        "- The N72R16 stop rule is applied. Any association/MOT/SAM3 work requires a separately authorized next stage.\n"
    )
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
