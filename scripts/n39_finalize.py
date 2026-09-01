#!/usr/bin/env python3
"""Finalize N39 without changing any frozen experiment artifact.

This script only reads the N39 audit/scan/posthoc artifacts and atomically writes
the N39 stage-04 status, final gate, and human-readable report.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "n39"
DOCS = ROOT / "docs"
HORIZONS = ("20", "50", "100")
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
CONFIG_ORDER = (
    ("lambda_assoc", (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)),
    ("human_weight", (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)),
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def atomic_text(path: Path, value: str) -> None:
    atomic_bytes(path, value.encode("utf-8"))


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "null"
    if not finite(value):
        return str(value)
    return f"{float(value):.{digits}f}"


def mean(values: Iterable[Any]) -> float | None:
    selected = [float(value) for value in values if finite(value)]
    return sum(selected) / len(selected) if selected else None


def token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def config_id(mode: str, value: float) -> str:
    return f"{mode}_{token(value)}"


def ordered_config_ids(result: dict[str, Any]) -> list[str]:
    output = []
    for mode, values in CONFIG_ORDER:
        for value in values:
            cid = config_id(mode, value)
            if cid in result["configurations"]:
                output.append(cid)
    return output


def input_hashes() -> dict[str, str]:
    paths = (
        "docs/N38R1_FINAL_REPORT.md",
        "outputs/n38r1/n38r1_final_gate.json",
        "outputs/n38r1/diagnostic_attempt3/score_assignment_summary.json",
        "outputs/n39/scale_audit_protocol.json",
        "outputs/n39/scale_audit_summary.json",
        "outputs/n39/weight_protocol.json",
        "outputs/n39/weight_runs/smoke_attempt2_manifest.json",
        "outputs/n39/weight_runs/full_attempt1_manifest.json",
        "outputs/n39/weight_scan_results.json",
    )
    return {path: sha256(ROOT / path) for path in paths}


def scalar_mean(events: list[dict[str, Any]], variant: str, horizon: str, key: str) -> float | None:
    values = []
    for event in events:
        summary = event["variants"].get(variant, {})
        value = summary.get("horizon_deltas", {}).get(horizon, {}).get(key)
        if finite(value):
            values.append(value)
    return mean(values)


def write_metric_mean(events: list[dict[str, Any]], variant: str, horizon: str, key: str) -> float | None:
    values = []
    for event in events:
        summary = event["variants"].get(variant, {})
        value = summary.get("write_metrics", {}).get("horizons", {}).get(horizon, {}).get(key)
        if finite(value):
            values.append(value)
    return mean(values)


def transition_mean(events: list[dict[str, Any]], variant: str, horizon: str, key: str) -> float | None:
    values = []
    for event in events:
        summary = event["variants"].get(variant, {})
        value = summary.get("transition_diagnostics", {}).get(horizon, {}).get(key)
        if finite(value):
            values.append(value)
    return mean(values)


def transition_sum(events: list[dict[str, Any]], variant: str, horizon: str, key: str) -> int:
    total = 0
    for event in events:
        summary = event["variants"].get(variant, {})
        value = summary.get("transition_diagnostics", {}).get(horizon, {}).get(key, 0)
        if finite(value):
            total += int(value)
    return total


def config_summary(cid: str, config: dict[str, Any]) -> dict[str, Any]:
    events = config["events"]
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        gate = config["gate_checks"][variant]
        bootstrap = config["sequence_cluster_bootstrap"][variant]
        horizons: dict[str, Any] = {}
        for horizon in HORIZONS:
            ci = bootstrap[horizon]
            horizons[horizon] = {
                "identity_utility_delta_mean": ci.get("mean"),
                "identity_utility_delta_ci_lower": ci.get("lower"),
                "identity_utility_delta_ci_upper": ci.get("upper"),
                "event_mean_id_switch_reduction": scalar_mean(events, variant, horizon, "id_switch_reduction"),
                "event_mean_target_missing_rate_reduction": scalar_mean(events, variant, horizon, "target_missing_rate_reduction_no_write_minus_write"),
                "event_mean_target_iou_delta": scalar_mean(events, variant, horizon, "target_iou_delta_write_minus_no_write"),
                "event_mean_recorrection_opportunity_reduction": scalar_mean(events, variant, horizon, "posthoc_recorrection_opportunity_reduction"),
                "write_mean_identity_error_rate": write_metric_mean(events, variant, horizon, "target_identity_error_rate"),
                "write_mean_missing_rate": write_metric_mean(events, variant, horizon, "target_missing_rate"),
                "write_mean_iou": write_metric_mean(events, variant, horizon, "target_mean_iou"),
                "write_mean_id_switch_count": write_metric_mean(events, variant, horizon, "id_switch_count"),
                "write_mean_recorrection_count": write_metric_mean(events, variant, horizon, "posthoc_recorrection_opportunity_count"),
            }
        variants[variant] = {
            "horizons": horizons,
            "h20_score_change_rate": gate.get("score_change_rate_h20_mean"),
            "h20_assignment_change_rate": gate.get("assignment_change_rate_h20_mean"),
            "h20_correct_assignment_change_count": gate.get("correct_assignment_change_count_h20"),
            "h20_incorrect_assignment_change_count": gate.get("incorrect_assignment_change_count_h20"),
            "protected_no_obvious_regression": gate.get("protected_no_obvious_regression"),
        }
    return {
        "config_id": cid,
        "mode": config["mode"],
        "value": config["value"],
        "execution_status": config["status"],
        "future_effect_gate": config["future_effect_gate"],
        "variants": variants,
    }


def action_summary(config: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in config["events"]:
        grouped.setdefault(str(event["action_type"]), []).append(event)
    output: dict[str, Any] = {}
    for action in sorted(grouped):
        events = grouped[action]
        summaries = [event["variants"]["M2"] for event in events]
        transitions = [summary["transition_diagnostics"]["20"] for summary in summaries]
        output[action] = {
            "event_count": len(events),
            "identity_utility_delta_event_mean": mean(summary["horizon_deltas"]["20"].get("identity_utility_delta") for summary in summaries),
            "target_missing_rate_reduction_event_mean": mean(summary["horizon_deltas"]["20"].get("target_missing_rate_reduction_no_write_minus_write") for summary in summaries),
            "target_iou_delta_event_mean": mean(summary["horizon_deltas"]["20"].get("target_iou_delta_write_minus_no_write") for summary in summaries),
            "recorrection_opportunity_reduction_event_mean": mean(summary["horizon_deltas"]["20"].get("posthoc_recorrection_opportunity_reduction") for summary in summaries),
            "assignment_change_rate_event_mean": mean(item.get("assignment_change_rate") for item in transitions),
            "score_change_rate_event_mean": mean(item.get("score_change_rate") for item in transitions),
            "correct_assignment_change_count": sum(int(item.get("correct_assignment_change_count", 0)) for item in transitions),
            "incorrect_assignment_change_count": sum(int(item.get("incorrect_assignment_change_count", 0)) for item in transitions),
        }
    return output


def distribution_table(scale: dict[str, Any]) -> list[dict[str, Any]]:
    dimensions = scale["distribution_by_dimension"]
    order = [
        "all",
        "target_row=true",
        "target_row=false",
        "frame_horizon=event_frame",
        "frame_horizon=future_t_plus_1",
        "frame_horizon=future_h20_after_t_plus_1",
        "frame_horizon=future_h50_after_t_plus_1",
        "frame_horizon=future_h100_after_t_plus_1",
        "variant=M0",
        "variant=M1",
        "variant=M2",
        "variant=M3",
        "variant=M4",
        "action=ADD_NEW_IDENTITY",
        "action=AUTHORITATIVE_REASSIGN",
        "action=ATOMIC_ID_SWAP",
        "action=RECOVER_IDENTITY",
    ]
    rows = []
    for dimension in order:
        value = dimensions.get(dimension)
        if value is None:
            continue
        delta = value["appearance_delta"]
        ratio = value["abs_appearance_delta_over_margin"]
        margin = value["top1_top2_margin"]
        assignment_margin = value["assignment_margin"]
        human = value["human_positive"]
        machine = value["machine_prototype"]
        negative = value["negative"]
        rows.append({
            "dimension": dimension,
            "count": delta.get("count"),
            "delta_median": delta.get("median"),
            "delta_p90": delta.get("p90"),
            "delta_p95": delta.get("p95"),
            "delta_max": delta.get("max"),
            "ratio_median": ratio.get("median"),
            "ratio_p90": ratio.get("p90"),
            "ratio_p95": ratio.get("p95"),
            "ratio_max": ratio.get("max"),
            "top1_top2_margin_median": margin.get("median"),
            "assignment_margin_median": assignment_margin.get("median"),
            "human_positive_median": human.get("median"),
            "human_positive_p95": human.get("p95"),
            "machine_prototype_median": machine.get("median"),
            "negative_median": negative.get("median"),
        })
    return rows


def md_distribution(rows: list[dict[str, Any]]) -> str:
    lines = [
        "|分组|n|Δ median/p90/p95/max|abs(Δ)/margin median/p90/p95/max|top1-top2 margin median|assignment margin median|human+ median/p95|machine median|negative median|",
        "|---|---:|---|---|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"|{row['dimension']}|{row['count']}|{fmt(row['delta_median'])}/{fmt(row['delta_p90'])}/{fmt(row['delta_p95'])}/{fmt(row['delta_max'])}|"
            f"{fmt(row['ratio_median'])}/{fmt(row['ratio_p90'])}/{fmt(row['ratio_p95'])}/{fmt(row['ratio_max'])}|"
            f"{fmt(row['top1_top2_margin_median'])}|{fmt(row['assignment_margin_median'])}|"
            f"{fmt(row['human_positive_median'])}/{fmt(row['human_positive_p95'])}|{fmt(row['machine_prototype_median'])}|{fmt(row['negative_median'])}|"
        )
    return "\n".join(lines)


def md_effect_table(summary_by_config: dict[str, Any], config_ids: list[str]) -> str:
    lines = [
        "|配置|variant|H20 U mean [CI]|H50 U mean [CI]|H100 U mean [CI]|H20 score→assign|正确/错误 assignment change|protected|",
        "|---|---|---|---|---|---|---:|---|",
    ]
    for cid in config_ids:
        config = summary_by_config[cid]
        for variant in ("M1", "M2", "M3", "M4"):
            item = config["variants"][variant]
            ci_cells = []
            for horizon in HORIZONS:
                h = item["horizons"][horizon]
                ci_cells.append(f"{fmt(h['identity_utility_delta_mean'])} [{fmt(h['identity_utility_delta_ci_lower'])},{fmt(h['identity_utility_delta_ci_upper'])}]")
            lines.append(
                f"|{cid}|{variant}|{ci_cells[0]}|{ci_cells[1]}|{ci_cells[2]}|"
                f"{fmt(item['h20_score_change_rate'])}→{fmt(item['h20_assignment_change_rate'])}|"
                f"{item['h20_correct_assignment_change_count']}/{item['h20_incorrect_assignment_change_count']}|{item['protected_no_obvious_regression']}|"
            )
    return "\n".join(lines)


def md_metric_table(summary_by_config: dict[str, Any], config_ids: list[str]) -> str:
    lines = [
        "|配置|variant|窗口|write identity error|write missing|write IoU|write IDSW|write re-corr|ΔIoU|Δmissing|ΔIDSW|Δre-corr|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cid in config_ids:
        config = summary_by_config[cid]
        for variant in ("M2", "M3", "M4"):
            for horizon in HORIZONS:
                h = config["variants"][variant]["horizons"][horizon]
                lines.append(
                    f"|{cid}|{variant}|H{horizon}|{fmt(h['write_mean_identity_error_rate'])}|{fmt(h['write_mean_missing_rate'])}|"
                    f"{fmt(h['write_mean_iou'])}|{fmt(h['write_mean_id_switch_count'])}|{fmt(h['write_mean_recorrection_count'])}|"
                    f"{fmt(h['event_mean_target_iou_delta'])}|{fmt(h['event_mean_target_missing_rate_reduction'])}|"
                    f"{fmt(h['event_mean_id_switch_reduction'])}|{fmt(h['event_mean_recorrection_opportunity_reduction'])}|"
                )
    return "\n".join(lines)


def md_action_table(action_summaries: dict[str, dict[str, Any]], config_ids: list[str]) -> str:
    lines = [
        "|配置|action|n|M2 H20 ΔU|Δmissing|ΔIoU|Δre-corr|assign-change|correct/incorrect|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cid in config_ids:
        for action in sorted(action_summaries[cid]):
            item = action_summaries[cid][action]
            lines.append(
                f"|{cid}|{action}|{item['event_count']}|{fmt(item['identity_utility_delta_event_mean'])}|"
                f"{fmt(item['target_missing_rate_reduction_event_mean'])}|{fmt(item['target_iou_delta_event_mean'])}|"
                f"{fmt(item['recorrection_opportunity_reduction_event_mean'])}|{fmt(item['assignment_change_rate_event_mean'])}|"
                f"{item['correct_assignment_change_count']}/{item['incorrect_assignment_change_count']}|"
            )
    return "\n".join(lines)


def main() -> None:
    result_path = OUT / "weight_scan_results.json"
    scale_path = OUT / "scale_audit_summary.json"
    stage1_path = OUT / "stage_01_status.json"
    stage2_path = OUT / "stage_02_status.json"
    stage3_path = OUT / "stage_03_status.json"
    smoke_path = OUT / "weight_runs" / "smoke_attempt2_manifest.json"
    full_path = OUT / "weight_runs" / "full_attempt1_manifest.json"
    result = load_json(result_path)
    scale = load_json(scale_path)
    stage1 = load_json(stage1_path)
    stage2 = load_json(stage2_path)
    stage3 = load_json(stage3_path)
    smoke = load_json(smoke_path)
    full = load_json(full_path)
    config_ids = ordered_config_ids(result)
    summaries = {cid: config_summary(cid, result["configurations"][cid]) for cid in config_ids}
    actions = {cid: action_summary(result["configurations"][cid]) for cid in config_ids}
    hashes = input_hashes()
    matrix_diff_text = json.dumps(scale.get("matrix_replay_max_abs_diff"), sort_keys=True)

    all_worker_rows = full.get("workers", [])
    worker_returncodes = [row.get("returncode") for row in all_worker_rows]
    worker_output_missing = [row.get("output") for row in all_worker_rows if not (ROOT / str(row.get("output"))).exists()]
    execution_integrity = {
        "stage_01_status": stage1.get("status"),
        "stage_02_smoke_status": smoke.get("status"),
        "stage_02_full_status": stage2.get("status"),
        "stage_02_full_worker_count": full.get("worker_count"),
        "stage_02_full_expected_worker_count": full.get("expected_worker_count"),
        "stage_02_full_unique_worker_keys": len({(row.get("event_id"), row.get("config_id")) for row in all_worker_rows}),
        "stage_02_full_nonzero_returncodes": sum(code != 0 for code in worker_returncodes),
        "stage_02_full_missing_worker_outputs": len(worker_output_missing),
        "stage_03_status": stage3.get("status"),
        "posthoc_event_count": result.get("event_count"),
        "posthoc_independent_sequence_count": result.get("independent_sequence_count"),
        "posthoc_configuration_count": result.get("configuration_count"),
        "runtime_future_gt_used": result.get("runtime_future_gt_used"),
    }

    final_gate = {
        "protocol": "N39_WEIGHTED_ASSOCIATION_INTERFACE_PROBE",
        "status": "COMPLETED_GATE_FAILED",
        "execution_complete": True,
        "research_gate": "FAIL_FUTURE_EFFECT",
        "scientific_gate": result["global_future_effect_gate"],
        "downstream_authorized": False,
        "calibration_head": "NOT_AUTHORIZED",
        "selector": "NOT_AUTHORIZED",
        "decoder_lora": "NOT_AUTHORIZED",
        "event_count": result["event_count"],
        "independent_sequence_count": result["independent_sequence_count"],
        "configuration_count": result["configuration_count"],
        "worker_count": full.get("worker_count"),
        "runtime_future_gt_used": False,
        "frozen_scope": {
            "event_count": result["event_count"],
            "independent_sequence_count": result["independent_sequence_count"],
            "action_types": ["ADD_NEW_IDENTITY", "AUTHORITATIVE_REASSIGN", "ATOMIC_ID_SWAP", "RECOVER_IDENTITY"],
            "configurations": [{"config_id": cid, "mode": result["configurations"][cid]["mode"], "value": result["configurations"][cid]["value"]} for cid in config_ids],
            "variants": ["M0", "M1", "M2", "M3", "M4"],
            "interaction_source": "simulated_from_gt",
            "runtime_future_gt_used": False,
            "posthoc_gt_only": True,
        },
        "execution_integrity": execution_integrity,
        "input_hashes": hashes,
        "stage_01_scale_audit": {
            "status": scale.get("status"),
            "row_count": scale.get("row_count"),
            "input_artifacts": scale.get("input_artifacts"),
            "component_replay_count": scale.get("component_replay_count"),
            "matrix_replay_max_abs_diff": scale.get("matrix_replay_max_abs_diff"),
            "causal_boundary_violations": scale.get("causal_boundary_violations"),
            "runtime_future_gt_true_count": scale.get("runtime_future_gt_true_count"),
            "diagnosis": scale.get("diagnosis"),
        },
        "configurations": summaries,
        "action_breakdown_m2_h20": actions,
        "preserved_failures": [
            "outputs/n39/attempts/weight_scan_lambda_assoc_0_n37-dancetrack0001-0296-authoritative_reassign-001.json",
            "outputs/n39/attempts/stage_03_posthoc_failure.json",
        ],
        "interpretation": {
            "score_change_is_not_assignment_change": True,
            "assignment_change_is_not_correctness": True,
            "stage_01_scale_supports_delta_smaller_than_margin": True,
            "external_lambda_can_cross_some_boundaries_but_not_stably_correct": True,
            "internal_human_weight_can_amplify_positive_anchor_but_not_stably_correct": True,
            "candidate_or_base_or_assignment_boundary_remains_plausible": True,
        },
        "next_action": "Collect real human event tape and run one separately preregistered association-interface probe; do not train calibration head, selector or decoder LoRA from this result.",
        "artifacts": {
            "scale_audit_summary": "outputs/n39/scale_audit_summary.json",
            "scale_audit_table": "outputs/n39/scale_audit_table.jsonl",
            "weight_protocol": "outputs/n39/weight_protocol.json",
            "smoke_manifest": "outputs/n39/weight_runs/smoke_attempt2_manifest.json",
            "full_manifest": "outputs/n39/weight_runs/full_attempt1_manifest.json",
            "posthoc_result": "outputs/n39/weight_scan_results.json",
            "stage_01": "outputs/n39/stage_01_status.json",
            "stage_02": "outputs/n39/stage_02_status.json",
            "stage_03": "outputs/n39/stage_03_status.json",
            "stage_04": "outputs/n39/stage_04_status.json",
        },
        "created_at": now(),
    }
    atomic_json(OUT / "n39_final_gate.json", final_gate)

    report_lines = [
        "# N39 — Weighted Association Interface Probe",
        "",
        f"**Status:** `COMPLETED_GATE_FAILED`  ",
        f"**Research gate:** `FAIL_FUTURE_EFFECT`  ",
        "**结论：** N39 的 14 个预注册权重配置、24 个事件/21 条序列、M0–M4 配对 replay 均完成；严格 future-effect gate 未通过，因此 calibration head、selector 和 decoder LoRA 均未授权。",
        "",
        "## 1. 研究问题与冻结边界",
        "",
        "本轮只区分两个可能的尺度瓶颈：内部 `AppearanceMemory.human_weight` 是否使 human positive anchor 太弱，以及外部 `appearance_score_weight`（本轮记为 `lambda_assoc`）是否使 memory 总分相对 base/Hungarian margin 太小。checkpoint、候选 tape、prefix、未来候选流、Hungarian、指标、H20/H50/H100 和 sequence-cluster bootstrap 均冻结。事件来自 N37 冻结 manifest，`interaction_source=simulated_from_gt`；不能称为历史真实点击。N37/N38R1 证据只读且未覆盖。",
        "",
        "运行时没有读未来 GT：所有 worker 和 replay artifact 的 `runtime_future_gt_used=false`；GT 仅在所有运行 artifact 完成后用于 posthoc scoring。event frame 不读取刚写入的 memory，memory 从 event+1 才可见。",
        "",
        "## 2. Stage 1：公式、调用链和尺度审计",
        "",
        "当前调用链为：`AppearanceMemory._score_components` 分解 machine prototype、human positive 和 hard negative；外部关联接口计算 `appearance_delta = lambda_assoc × memory_total`，再将其加入 base score 形成 fused score 并送入 `linear_sum_assignment`。hard-negative 约束在融合后重新施加，因此不会被外观项覆盖。",
        "",
        "内部与外部参数的作用不同：`human_weight` 只改变 positive-anchor 项；`lambda_assoc` 改变整个 memory total 对 fused score 的贡献。event frame 的 appearance delta 全为 0，符合当前帧写入不可见和 t+1 因果边界。",
        "",
        f"Stage 1 对 N38R1 的 120 个 artifact 做了 {scale.get('row_count')} 条候选级记录，component replay={scale.get('component_replay_count')}，矩阵重放最大绝对误差={matrix_diff_text}，causal violation=0，runtime future-GT=0。完整 median/p90/p95/max 分布见机器表；下表同时覆盖 action、variant、horizon 和 target-row 分组。",
        "",
        md_distribution(distribution_table(scale)),
        "",
        "全局默认尺度的诊断是：appearance delta 的中位数约 `0.2213`，target-row `|delta|/margin` 中位数约 `0.0494`、p95 约 `0.0909`，而 assignment margin 中位数约 `4.7091`。因此“delta 通常小于决策 margin”得到支持；但这只是接口尺度证据，不等于已证明提高权重会改善身份。",
        "",
        "## 3. Stage 2：预注册权重扫描与完整性",
        "",
        "扫描顺序和取值在 replay 前冻结：先 `lambda_assoc ∈ {0, 0.25, 0.5, 1, 2, 4, 8}`，再固定 lambda=1 扫描 `human_weight ∈ {0, 0.25, 0.5, 1, 2, 4, 8}`。smoke 使用 3 个不同 action/sequence 的相同输入，42/42 worker 通过；full scan 使用 336 个独立 worker，336/336 返回码为 0，14/14 配置和 24×5 事件-variant 键完整。",
        "",
        "曾保留两类执行失败事实：第一次 smoke 因 worker 将缺省的既有 `runtime_future_gt_used` 字段误当成 true 而失败，退出码 1；修复为显式默认 false 后 targeted smoke 和完整 smoke 均通过。第一次 posthoc 评分在计算完成后因相对路径 `relative_to(ROOT)` 抛出 `ValueError`，退出码 1；路径归一化 targeted regression 通过后，使用同一 336-worker 输入重跑成功。两份失败 artifact 均未删除。",
        "",
        "## 4. Stage 3：future effect 结果",
        "",
        "以下表格中的 U 是 posthoc identity-utility delta（write−no-write，sequence-cluster CI 为 gate 依据）；IoU/missing/IDSW/re-correction 的 Δ 是事件级 write−no-write 语义对应的冻结定义。`write identity error/missing/IoU/IDSW/re-corr` 是各配置的事件平均写分支指标。完整逐事件、逐帧 audit 保存在 `weight_scan_results.json` 及 336 个 worker artifact 中。",
        "",
        md_effect_table(summaries, config_ids),
        "",
        "### 4.1 M2/M3/M4 的可计算指标",
        "",
        md_metric_table(summaries, config_ids),
        "",
        "### 4.2 M2 H20 action 分解",
        "",
        md_action_table(actions, config_ids),
        "",
        "关键门控结果：",
        "",
        "- `lambda_assoc=0` 消除了 appearance score 变化和 assignment change，验证外部接口确实控制 memory 对分配的影响。",
        "- 提高 `lambda_assoc` 后 M2 H20 assignment-change rate 从默认约 `0.0708` 增至 `0.1917`（lambda=4/8）；但对应 U mean 约 `+0.0310`，95% CI 为 `[-0.0152, +0.0891]`，下界不严格大于 0。",
        "- 提高 `human_weight` 后 M2 H20 assignment-change rate 在 4/8 时约 `0.1667`，U mean 同样约 `+0.0310`，95% CI 为 `[-0.0152, +0.0891]`；低值配置出现 0 或负的整体效应。",
        "- 较大权重确实让更多分配穿过边界，但改变并不稳定地正确：例如 lambda=4/8 或 human_weight=4/8 的 M2 H20 汇总为 44 次 correct、15 次 incorrect，sequence-cluster CI 仍跨 0；其它 variant/action 还出现全为 incorrect 或零改变的情况。",
        "- untouched/protected identity 的 no-obvious-regression 检查通过，candidate/runtime 完整性和 leakage 检查通过；但 M2、M3、M4 的 H20 lower-CI > 0 联合条件没有通过。",
        "",
        "因此，N39 支持“默认外观 delta 相对 assignment margin 偏小”，也表明外部 lambda 和内部 human_weight 都能在高值时推动部分 boundary crossing；但当前数据不能把收益归因于某一个权重，更不能把 score change 当成正确 assignment。最大权重仍未得到 sequence-stable future benefit，故不能冻结任何高权重配置。候选可分性、base/几何项或当前 assignment decision boundary 仍是合理的主要瓶颈。",
        "",
        "## 5. Stage 4 决策与下一步",
        "",
        "N39 是执行完成但科研 gate 失败，不是资源 BLOCKED：`execution_complete=true`，`research_gate=FAIL_FUTURE_EFFECT`。由于 M2/M3/M4 的 sequence-cluster 95% CI 下界没有同时严格大于 0，禁止训练 calibration head、selector 和 decoder LoRA，也不部署当前权重。",
        "",
        "最小下一步是收集带真实人工事件的 event tape，并在不改变候选/Hungarian/指标的前提下预注册一次 association-interface probe；不能继续用 `simulated_from_gt` 事件包装成真实点击，也不能通过换 checkpoint、训练 LoRA 或事后调阈值绕过 gate。",
        "",
        "## 6. 机器可读证据",
        "",
        "- [N39 final gate](../outputs/n39/n39_final_gate.json)",
        "- [Stage 1 status](../outputs/n39/stage_01_status.json)",
        "- [Stage 2 status](../outputs/n39/stage_02_status.json)",
        "- [Stage 3 status](../outputs/n39/stage_03_status.json)",
        "- [Scale audit summary](../outputs/n39/scale_audit_summary.json)",
        "- [Scale audit table](../outputs/n39/scale_audit_table.jsonl)",
        "- [Frozen weight protocol](../outputs/n39/weight_protocol.json)",
        "- [Smoke manifest](../outputs/n39/weight_runs/smoke_attempt2_manifest.json)",
        "- [Full scan manifest](../outputs/n39/weight_runs/full_attempt1_manifest.json)",
        "- [Full posthoc result](../outputs/n39/weight_scan_results.json)",
        "- [Preserved first smoke failure](../outputs/n39/attempts/weight_scan_lambda_assoc_0_n37-dancetrack0001-0296-authoritative_reassign-001.json)",
        "- [Preserved first posthoc failure](../outputs/n39/attempts/stage_03_posthoc_failure.json)",
        "",
        "N39 input hashes are recorded in `outputs/n39/n39_final_gate.json`; N36/N37/N38R1 evidence remains unchanged.",
        "",
    ]
    report = "\n".join(report_lines)
    atomic_text(DOCS / "N39_FINAL_REPORT.md", report)

    stage4 = {
        "stage": "N39-04",
        "status": "COMPLETED_GATE_FAILED",
        "execution_complete": True,
        "research_gate": "FAIL_FUTURE_EFFECT",
        "protocol": "N39_WEIGHTED_ASSOCIATION_INTERFACE_PROBE",
        "event_count": result["event_count"],
        "independent_sequence_count": result["independent_sequence_count"],
        "configuration_count": result["configuration_count"],
        "worker_count": full.get("worker_count"),
        "upstream_checks": result.get("upstream_checks"),
        "runtime_future_gt_used": False,
        "interaction_source": "simulated_from_gt",
        "future_effect_gate": result["global_future_effect_gate"],
        "downstream_authorized": False,
        "calibration_head": "NOT_AUTHORIZED",
        "selector": "NOT_AUTHORIZED",
        "decoder_lora": "NOT_AUTHORIZED",
        "diagnosis": final_gate["interpretation"],
        "execution_integrity": execution_integrity,
        "preserved_failures": final_gate["preserved_failures"],
        "input_hashes": hashes,
        "report": "docs/N39_FINAL_REPORT.md",
        "final_gate": "outputs/n39/n39_final_gate.json",
        "next_action": final_gate["next_action"],
        "finished_at": now(),
    }
    atomic_json(OUT / "stage_04_status.json", stage4)

    print(json.dumps({
        "status": stage4["status"],
        "report": "docs/N39_FINAL_REPORT.md",
        "final_gate": "outputs/n39/n39_final_gate.json",
        "configuration_count": result["configuration_count"],
        "event_count": result["event_count"],
        "independent_sequence_count": result["independent_sequence_count"],
        "worker_count": full.get("worker_count"),
        "downstream_authorized": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
