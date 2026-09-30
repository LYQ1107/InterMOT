#!/usr/bin/env python
"""Write the auditable N72R19 noise matrix, result, and final report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


METHOD_EMA = "EMA_0.90"
METHOD_GRU = "GRU_MEMORY_N72R18"
METHOD_ROBUST = "GRU_RELIABILITY_N72R19"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default="outputs/N72R19/metrics/noise_matrix.json")
    parser.add_argument("--output-dir", default="outputs/N72R19")
    parser.add_argument("--docs-report", default="docs/N72R19_FINAL_REPORT.md")
    return parser.parse_args()


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def signed_pp(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):+.2f} pp"


def number(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def _condition_order() -> list[tuple[str, str]]:
    result = [("clean_0p00", "Clean")]
    names = {
        "wrong_identity_injection": "Wrong identity",
        "missing_observation": "Missing observation",
        "hard_negative_replacement": "Hard-negative replacement",
    }
    for kind, display in names.items():
        for rate in ("0p10", "0p20", "0p30", "0p50"):
            result.append((f"{kind}_{rate}", f"{display} {rate.replace('p', '.')}"))
    return result


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _recovery(path: Path) -> dict[str, object]:
    rows = _read_jsonl(path)
    by_identity: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_identity[(str(row["sequence"]), int(row["track_id"]))].append(row)
    for values in by_identity.values():
        values.sort(key=lambda item: int(item["frame"]))
    next_after_missing: list[bool] = []
    for values in by_identity.values():
        record_by_frame = {int(row["frame"]): row for row in values}
        for index, row in enumerate(values):
            if bool(row["observation_present"]):
                continue
            for future in values[index + 1 :]:
                if bool(future["observation_present"]):
                    score_row = record_by_frame.get(int(future["frame"]))
                    if score_row is not None:
                        next_after_missing.append(bool(score_row["win"]))
                    break
    return {
        "missing_events": sum(not bool(row["observation_present"]) for row in rows),
        "recovery_pairs": len(next_after_missing),
        "next_observed_win_rate": (
            sum(next_after_missing) / len(next_after_missing) if next_after_missing else None
        ),
    }


def _drift_value(matrix: dict[str, object], method: str, condition: str, group: str, field: str):
    return (
        matrix.get("memory_drift", {})
        .get(method, {})
        .get(condition, {})
        .get("drift", {})
        .get(group, {})
        .get(field)
    )


def _make_report(matrix: dict[str, object]) -> tuple[str, dict[str, object]]:
    methods = matrix["methods"]
    cells: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for condition, display in _condition_order():
        try:
            ema = methods[METHOD_EMA][condition]["summary"]
            gru = methods[METHOD_GRU][condition]["summary"]
            robust = methods[METHOD_ROBUST][condition]["summary"]
        except KeyError as exc:
            raise RuntimeError(f"missing N72R19 evaluation cell: {condition}") from exc
        ema_h100 = float(ema["h100_win_rate"])
        robust_h100 = float(robust["h100_win_rate"])
        robust_ci_lower = float(robust["h100_ci95"][0])
        delta = robust_h100 - ema_h100
        cell = {
            "condition": condition,
            "display": display,
            "ema_h100": ema_h100,
            "gru_h100": float(gru["h100_win_rate"]),
            "robust_h100": robust_h100,
            "robust_minus_ema": delta,
            "robust_ci_lower": robust_ci_lower,
            "robust_lower_minus_ema": robust_ci_lower - ema_h100,
            "pass_cell": condition == "clean_0p00" or (delta >= 0.05 and robust_ci_lower >= ema_h100),
            "cell_status": "REFERENCE" if condition == "clean_0p00" else ("PASS" if delta >= 0.05 and robust_ci_lower >= ema_h100 else "FAIL"),
        }
        cells.append(cell)
        if condition != "clean_0p00" and not cell["pass_cell"]:
            failures.append(cell)

    validated = not failures
    noisy_cells = [cell for cell in cells if cell["condition"] != "clean_0p00"]
    robust_better_than_gru = sum(
        float(cell["robust_h100"]) > float(cell["gru_h100"]) for cell in noisy_cells
    )
    result = {
        "stage": "N72R19",
        "goal": "Robust Human Identity Memory Learning",
        "goal_family": "Human Identity Memory Learning",
        "goal_reference": "outputs/N72R19/FINAL_GOAL.json",
        "central_question": "Can learned identity memory outperform EMA under noisy observation conditions?",
        "primary_endpoint": "H100_HARD_NEGATIVE_IDENTITY_WIN_RATE",
        "robust_learned_memory_validated": validated,
        "decision": "ROBUST_LEARNED_MEMORY_VALIDATED" if validated else "ROBUST_LEARNED_MEMORY_NOT_VALIDATED",
        "next_association_stage_authorized": validated,
        "success_rule": "Every noisy evaluation cell must have GRU+Reliability H100 >= EMA H100 + 5 pp and robust H100 CI lower >= paired EMA H100 point estimate",
        "cells": cells,
        "failed_noisy_cells": failures,
        "noisy_cells": len(noisy_cells),
        "robust_beats_clean_gru_noisy_cells": robust_better_than_gru,
        "goal_frozen": True,
        "sam3_required": False,
        "mot_required": False,
        "training_required": True,
    }

    lines = [
        f"Learned Identity Memory 在噪声观察条件下是否超过 EMA？**{'是' if validated else '否'}。**",
        "",
        "FINAL GOAL:",
        "Human Identity Memory Learning",
        "",
        "CENTRAL QUESTION:",
        "“学习一个身份记忆更新模型，能不能在真实噪声观察条件下超过一个简单 EMA identity memory？”",
        "",
        f"N72R19 结果：`{result['decision']}`。",
        "",
        "## Frozen scope",
        "",
        "本阶段只研究身份记忆更新；使用 N72R17 OSNet 512-D 冻结 embedding、N72R18 GRU checkpoint 和 N72R19 噪声 replay。没有运行 SAM3、MOT、Hungarian、TrackEval、HOTA、IDF1、Requery、InterMOT association、LoRA 或新的 ReID encoder 训练，也没有保存 crop。",
        "",
        "## Primary endpoint: H100 hard-negative identity win rate",
        "",
        "| Condition | EMA(0.90) | N72R18 GRU | N72R19 GRU+Reliability | Robust−EMA | Robust CI lower−EMA | Cell |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for cell in cells:
        lines.append(
            f"| {cell['display']} | {pct(cell['ema_h100'])} | {pct(cell['gru_h100'])} | {pct(cell['robust_h100'])} | {signed_pp(cell['robust_minus_ema'])} | {signed_pp(cell['robust_lower_minus_ema'])} | {cell['cell_status']} |"
        )
    lines.extend(
        [
            "",
            "The pre-frozen rule is applied to every noisy cell: robust H100 must exceed EMA by at least 5 percentage points, and the robust sequence-cluster 95% CI lower bound must not fall below the paired EMA point estimate.",
            "",
            "## Required answers",
            "",
            "1. **EMA 为什么失败？** Fixed EMA applies the same update fraction to every present observation. Wrong-identity and hard-negative injections therefore move the state toward a competitor; it has no observation-quality decision, while missing observations simply freeze the state.",
            "2. **GRU 为什么提升？** The learned candidate can transform the observation conditioned on the current identity state, and it was trained with positive-versus-visible-competitor loss rather than only self-similarity. The clean N72R18 GRU is the frozen learned baseline; its noisy degradation is part of this test.",
            f"3. **Reliability 是否有效？** The frozen robust rule is **{'validated' if validated else 'not validated'}**. It beats the clean-trained GRU in {robust_better_than_gru}/{len(noisy_cells)} noisy cells; this count is descriptive, while the decision is against EMA and the frozen CI gate.",
            "4. **错误观察多少比例导致漂移？** Each noise level is a fixed attempted corruption probability; the exact applied and missing counts are recorded in the per-cell summaries and update JSONL. State-change L2, state-change cosine, and anchor drift are reported below.",
            "5. **Missing 后能否恢复？** Recovery is measured as the next observed-frame hard-negative win after a missing update. The per-condition recovery counts and rates are reported below; no future identity is fed into the updater.",
            "6. **Hard negative 是否稳定？** Noise C explicitly replaces observations with the current-frame highest-similarity wrong identity. Its H100 curve and drift are shown as separate cells, so a clean-only gain is not treated as robust evidence.",
            f"7. **是否值得接回 InterMOT？** **{'Only as a separately authorized next-stage experiment; this report does not run integration.' if validated else 'No. The robust memory gate is not validated, so `next_association_stage_authorized=false`.'}**",
            "",
            "## Memory drift and missing recovery",
            "",
            "| Method | Condition | Noise applied / updates | Mean state-change L2 | Mean corrupted-update L2 | Mean missing-update L2 | Mean anchor drift | Next-observed recovery win |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition, display in _condition_order():
        robust_cell = methods[METHOD_ROBUST][condition]["summary"]
        robust_slug = condition
        robust_drift = matrix["memory_drift"][METHOD_ROBUST][robust_slug]["drift"]
        applied = int(robust_cell["noise_applied_count"])
        updates = int(robust_cell["update_count"])
        # The source JSONL path is stored in the matrix only indirectly through
        # the generated output tree; report generation runs beside that tree.
        lines.append(
            f"| N72R19 GRU+Reliability | {display} | {applied}/{updates} ({pct(applied / updates if updates else None)}) | {number(robust_drift['all_updates']['mean_state_change_l2'])} | {number(robust_drift['present_corrupted']['mean_state_change_l2'])} | {number(robust_drift['missing']['mean_state_change_l2'])} | {number(robust_drift['all_updates']['mean_anchor_drift_cosine'])} | pending JSONL audit |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`{result['decision']}`; `ROBUST_LEARNED_MEMORY_VALIDATED={'true' if validated else 'false'}`.",
            f"`next_association_stage_authorized={'true' if validated else 'false'}` is recorded for the next stage, but N72R19 stops here and does not execute association integration.",
            "",
            "## Provenance",
            "",
            "- Final Goal: `outputs/N72R19/FINAL_GOAL.json`",
            "- Frozen protocol: `outputs/N72R19/protocol.json`",
            "- Reference audit: `outputs/N72R19/reference_audit.md`",
            "- Evaluation matrix: `outputs/N72R19/metrics/noise_matrix.json`",
            "- Per-cell records: `outputs/N72R19/noise_results/`",
            "- Memory drift: `outputs/N72R19/memory_drift/`",
        ]
    )
    return "\n".join(lines) + "\n", result


def _replace_recovery_table(report: str, matrix: dict[str, object], output_dir: Path) -> str:
    lines = report.splitlines()
    updated: list[str] = []
    for line in lines:
        if "pending JSONL audit" not in line:
            updated.append(line)
            continue
        parts = line.split("|")
        if len(parts) < 8:
            updated.append(line)
            continue
        method = METHOD_ROBUST.lower()
        condition = parts[2].strip()
        # Display text maps uniquely to the condition order.
        mapping = dict(_condition_order())
        condition_slug = next((slug for slug, display in mapping.items() if display == condition), None)
        if condition_slug is None:
            updated.append(line)
            continue
        record_path = output_dir / "noise_results" / f"{method}__{condition_slug}.jsonl"
        recovery = _recovery(record_path)
        value = pct(recovery["next_observed_win_rate"])
        updated.append(line.replace("pending JSONL audit", f"{value} ({recovery['recovery_pairs']} pairs)"))
    return "\n".join(updated) + "\n"


def main() -> None:
    args = parse_args()
    matrix_path = Path(args.matrix)
    output_dir = Path(args.output_dir)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    report, result = _make_report(matrix)
    report = _replace_recovery_table(report, matrix, output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "FINAL_RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    docs_path = Path(args.docs_report)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(report, encoding="utf-8")

    status_path = output_dir / "stage_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    status.update(
        {
            "stage": "N72R19",
            "goal": "Robust Human Identity Memory Learning",
            "goal_family": "Human Identity Memory Learning",
            "final_goal_path": "outputs/N72R19/FINAL_GOAL.json",
            "central_question": "Can learned identity memory outperform EMA under noisy observation conditions?",
            "status": "FINAL_RESULT_WRITTEN",
            "evaluation_complete": True,
            "training_complete": True,
            "robust_learned_memory_validated": result["robust_learned_memory_validated"],
            "decision": result["decision"],
            "next_association_stage_authorized": result["next_association_stage_authorized"],
            "goal_frozen": True,
            "sam3_required": False,
            "mot_required": False,
        }
    )
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "result": str((output_dir / "FINAL_RESULT.json").resolve()),
                "decision": result["decision"],
                "next_association_stage_authorized": result["next_association_stage_authorized"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
