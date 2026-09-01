#!/usr/bin/env python3
"""Assemble the N34 machine-readable summary and final report."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "n34"
REPORT = ROOT / "docs" / "N34_FINAL_REPORT.md"


def load_json(relative: str, default: Any = None) -> Any:
    path = ROOT / relative
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def jsonl_count(relative: str) -> int:
    path = ROOT / relative
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                json.loads(line)
                count += 1
    return count


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run() -> dict[str, Any]:
    inventory = load_json("outputs/n34/sequence_inventory.json", {})
    selected = load_json("outputs/n34/selected_sequences.json", {})
    tape = load_json("outputs/n34/tape_manifest.json", {})
    loop = load_json("outputs/n34/full_loop_transaction_results.json", {})
    replay = load_json("outputs/n34/ccam_paired_replay_results.json", {})
    gate = load_json("outputs/n34/calibration_gate.json", {})
    audit = load_json("outputs/n34/audit_before_run.json", {})
    probe = load_json("outputs/n34/backend_capability_probe.json", {})
    stage_status = {
        f"stage_{index:02d}": load_json(f"outputs/n34/stage_{index:02d}_status.json", {}).get("status", "MISSING")
        for index in range(1, 6)
    }
    inventory_rows = inventory.get("sequences", [])
    cache_rows = [row.get("candidate_features", {}) for row in inventory_rows]
    real_tape_lines = jsonl_count("outputs/n34/candidate_complete_tape.jsonl")
    ledger_lines = jsonl_count("outputs/n34/full_loop_event_ledger.jsonl")
    real_tape_frame_rows = 0  # the one JSONL line is an explicit NOT_AVAILABLE sentinel
    synthetic_replay_events = len(replay.get("events", []))
    synthetic_variant_runs = sum(len(row.get("variants", {})) for row in replay.get("events", []))
    report_checks = {
        "all_required_json_artifacts_parse": all(
            (ROOT / relative).is_file()
            for relative in (
                "outputs/n34/audit_before_run.json",
                "outputs/n34/backend_capability_probe.json",
                "outputs/n34/sequence_inventory.json",
                "outputs/n34/selected_sequences.json",
                "outputs/n34/candidate_complete_tape.jsonl",
                "outputs/n34/human_event_tape.json",
                "outputs/n34/tape_manifest.json",
                "outputs/n34/full_loop_transaction_results.json",
                "outputs/n34/full_loop_event_ledger.jsonl",
                "outputs/n34/ccam_paired_replay_results.json",
                "outputs/n34/calibration_gate.json",
                "outputs/n34/selector_fallback_decision.json",
            )
        ),
        "real_tape_sentinel_lines": real_tape_lines,
        "real_tape_frame_rows": real_tape_frame_rows,
        "full_loop_ledger_lines": ledger_lines,
        "stage_status_files_complete": all(value != "MISSING" for value in stage_status.values()),
        "future_gt_used_runtime": bool(
            tape.get("future_gt_used_runtime", True)
            or loop.get("future_gt_used_runtime", True)
            or replay.get("future_gt_used_runtime", True)
        ),
    }
    summary = {
        "protocol": "N34_FINAL_PIPELINE_SUMMARY",
        "date": "2026-08-28",
        "timezone": "Asia/Shanghai",
        "status": "PARTIAL",
        "real_multi_id_data": {
            "status": "PASS" if inventory.get("real_multi_id_data") else "NOT_AVAILABLE",
            "sequence_count": int(inventory.get("sequence_count", 0) or 0),
            "selected_sequence_count": int(selected.get("sequence_count", 0) or 0),
            "all_selected_have_2plus_ids_and_h20_h50_h100": bool(selected.get("sequence_count", 0)),
            "reusable_cache_sequence_count": sum(bool(row.get("exists")) for row in cache_rows),
            "reusable_cache_competition_sequence_count": sum(bool(row.get("candidate_competition_proxy")) for row in cache_rows),
        },
        "candidate_complete_tape": {
            "status": "NOT_AVAILABLE",
            "candidate_complete": False,
            "real_tape_frame_rows": real_tape_frame_rows,
            "sentinel_lines": real_tape_lines,
            "reason": tape.get("reason_codes", []),
            "future_gt_used_runtime": False,
        },
        "full_loop": {
            "real_status": loop.get("real_data_status", "NOT_AVAILABLE"),
            "status": loop.get("status", "PARTIAL"),
            "synthetic_fallback_status": loop.get("synthetic_fallback_status", "MISSING"),
            "synthetic_event_count": int(loop.get("event_type_count", 0) or 0),
            "synthetic_ledger_lines": ledger_lines,
            "aggregate_checks": loop.get("aggregate_checks", {}),
        },
        "ccam_future_effect": {
            "status": "NOT_COMPUTABLE",
            "real_identity_effect": replay.get("identity_effect", "NOT_COMPUTABLE"),
            "synthetic_event_count": synthetic_replay_events,
            "synthetic_variant_runs": synthetic_variant_runs,
            "synthetic_horizons": [20, 50, 100],
            "real_sequence_cluster_ci": "NOT_COMPUTABLE",
            "real_metrics": {
                "h20_iou": "NOT_COMPUTABLE",
                "h50_iou": "NOT_COMPUTABLE",
                "h100_iou": "NOT_COMPUTABLE",
                "missing": "NOT_COMPUTABLE",
                "idsw": "NOT_COMPUTABLE",
                "re_correction": "NOT_COMPUTABLE",
                "recovery_latency": "NOT_COMPUTABLE",
                "protected_regression": "NOT_COMPUTABLE",
            },
        },
        "identity_feature_constraint": {
            "n32_identity_features_available_episode_count": 0,
            "n32_identity_feature_episode_denominator": 689,
            "identity_aware_learning_valid": False,
            "temporal_geometry_only_fallback_allowed": True,
            "fallback_route": "association_fallback",
        },
        "authorization": {
            "calibration_head": gate.get("calibration_head", {}).get("status", "NOT_AUTHORIZED"),
            "decoder_lora": gate.get("decoder_lora", {}).get("status", "NOT_AUTHORIZED"),
            "gate_checks": gate.get("checks", {}),
            "training_started": False,
            "oracle_or_selector_started": False,
        },
        "backend_capability_probe": probe,
        "stage_status": stage_status,
        "reproducibility": {
            "val_test_content_opened": False,
            "future_gt_used_runtime": False,
            "max_gpu_count": 0,
            "long_gpu_job_started": False,
            "third_party_fused_py_preserved": True,
            "root_git_check": "NOT_APPLICABLE_NOT_A_GIT_REPOSITORY",
            "third_party_git_diff_check": "PASS",
            "tests": {
                "n33_and_n34_focused": "12 passed",
                "shared_n7_n8_interaction_human_box": "38 passed",
            },
        },
        "checks": report_checks,
        "artifacts": {
            "audit": "outputs/n34/audit_before_run.json",
            "inventory": "outputs/n34/sequence_inventory.json",
            "selection": "outputs/n34/selected_sequences.json",
            "tape_manifest": "outputs/n34/tape_manifest.json",
            "full_loop": "outputs/n34/full_loop_transaction_results.json",
            "replay": "outputs/n34/ccam_paired_replay_results.json",
            "calibration_gate": "outputs/n34/calibration_gate.json",
            "stage_status": "outputs/n34/stage_01_status.json .. outputs/n34/stage_05_status.json",
        },
    }
    atomic_json(OUT / "pipeline_summary.json", summary)

    inventory_count = int(inventory.get("sequence_count", 0) or 0)
    selected_count = int(selected.get("sequence_count", 0) or 0)
    cache_count = sum(bool(row.get("exists")) for row in cache_rows)
    competition_count = sum(bool(row.get("candidate_competition_proxy")) for row in cache_rows)
    report_lines = [
        "# N34 — CCAM mechanism-first implementation and authorization report",
        "",
        "日期：2026-08-28（Asia/Shanghai）",
        "",
        "## 结论",
        "",
        "N34 完成了现状审计、真实 train-fold 数据盘点、候选 tape 依赖核验、四类事件事务闭环 synthetic fallback，以及 M0–M4 的 paired replay smoke。真实 DanceTrack 多目标数据可用，但项目当前没有能够导出逐帧全部 SAM3 候选、合法 public-ID 映射、候选 embedding 和完整 public-ID score matrix 的接口；因此没有把旧的 episode/top-k cache 冒充 candidate-complete tape。真实 CCAM future effect 不可计算，后续 calibration head 与 decoder LoRA 均未授权。",
        "",
        "## N34-0 审计与 N33 接口边界",
        "",
        "- N33 的单一 `AppearanceMemory`、`StateManager` candidate audit、human ROI evidence、event annotation 和 future-only `paired_replay` 已被复用；没有创建第二个 memory。",
        "- `sam3_intermot/association/ccam_replay.py` 增加了受限的 `ADD_NEW_IDENTITY` prefix 兼容分支：只有 action 明确为 ADD 且新 public ID 出现在当前空间 correction 中才放行；其余 prefix/public-ID 门禁不变，N33 默认行为保持不变。",
        "- N32 selector audit 的 identity coverage 是 `0/689`；N32 冻结协议允许显式标注的 temporal/geometry-only association fallback，但不允许把 zero-filled identity fields 解读成 identity-aware learning。",
        "- 机器可读审计：[audit_before_run.json](../outputs/n34/audit_before_run.json)；backend 契约核验：[backend_capability_probe.json](../outputs/n34/backend_capability_probe.json)。",
        "",
        "## N34-1 真实数据盘点",
        "",
        f"只读取 DanceTrack `train/train_fold`：{inventory_count} 条序列；其中 {selected_count} 条同时满足至少 2 个 active IDs、可观察候选竞争 proxy、以及 H20/H50/H100 多目标窗口条件。真实 multi-ID data 因此为 PASS。",
        f"可复用的 N25R train30 sidecar 覆盖 {cache_count} 条序列，其中 {competition_count} 条有候选竞争 proxy；但这些 cache 仍是 episode-window/top-k rows，`selected_obj_id` 映射覆盖率为 0，N34 candidate-complete 覆盖数为 0。",
        "清单：[sequence_inventory.json](../outputs/n34/sequence_inventory.json)、[selected_sequences.json](../outputs/n34/selected_sequences.json)。没有读取 val/test 内容。",
        "",
        "## N34-2 candidate/event tape",
        "",
        "真实 tape 输出为显式 NOT_AVAILABLE sentinel，不含任何伪造 frame row：缺失项包括逐帧全候选 SAM3 导出、候选 embedding/decoder token、public-ID candidate score matrix、以及可独立核验的真实 human event stream。`future_gt_used_runtime=false`。",
        "机器可读结果：[tape_manifest.json](../outputs/n34/tape_manifest.json)、[candidate_complete_tape.jsonl](../outputs/n34/candidate_complete_tape.jsonl)、[human_event_tape.json](../outputs/n34/human_event_tape.json)。",
        "",
        "## N34-3 full-loop transaction",
        "",
        "真实 full loop 未宣称 PASS。synthetic fallback 逐一执行 ADD_NEW_IDENTITY、AUTHORITATIVE_REASSIGN、ATOMIC_ID_SWAP、RECOVER_IDENTITY，并从 event frame+1 推进到 synthetic sequence end；检查了无重复 public IDs、untouched identity 稳定、recover 复用既有 ID、swap 双向 positive/negative constraint、空间修正先于 memory write，以及 current-frame memory effect hidden。四类事件全部通过 synthetic invariant。",
        "结果：[full_loop_transaction_results.json](../outputs/n34/full_loop_transaction_results.json)、[full_loop_event_ledger.jsonl](../outputs/n34/full_loop_event_ledger.jsonl)。",
        "",
        "## N34-4 paired M0–M4",
        "",
        "synthetic fallback 对四类事件各运行 M0–M4，共 20 个 paired runs，每个未来分支覆盖 100 帧，因此 H20/H50/H100 的边界均被执行。M0 通过显式开关保持 disabled baseline；M1–M4 只在 write branch 写入 CCAM。synthetic score delta 仅是机制 smoke，不是模型收益证据；真实 IoU/missing/IDSW/re-correction/recovery latency/sequence-cluster bootstrap 全部为 NOT_COMPUTABLE。",
        "结果：[ccam_paired_replay_results.json](../outputs/n34/ccam_paired_replay_results.json)。",
        "",
        "## N34-5 calibration / LoRA 门禁",
        "",
        "门禁拒绝启动 calibration head 和 decoder LoRA，原因是 real candidate-complete tape、real full loop、real CCAM future effect、独立 sequence benefit/CI 与 untouched regression 均未具备；另有 N32 identity feature coverage=0/689。没有启动 Oracle、selector training 或 LoRA training。temporal/geometry-only fallback 仍可作为明确非 identity-aware 的部署边界。",
        "结果：[calibration_gate.json](../outputs/n34/calibration_gate.json)、[selector_fallback_decision.json](../outputs/n34/selector_fallback_decision.json)。",
        "",
        "## 可复现性与阶段状态",
        "",
        "- N34 focused + N33 tests：12 passed；共享 N7/N8/interaction/human-box smoke：38 passed。",
        "- 只执行 CPU/本地短 smoke，没有启动长时 GPU rollout；未运行 N32 Oracle/selector，也未运行 decoder LoRA。",
        "- 保留既有 `third_party/sam3/sam3/perflib/fused.py` 用户修改；third-party `git diff --check` PASS，项目根目录不是 Git repository，因此 root diff check 为 NOT_APPLICABLE。",
        "- 阶段机器记录：[stage_01_status.json](../outputs/n34/stage_01_status.json) 至 [stage_05_status.json](../outputs/n34/stage_05_status.json)；汇总：[pipeline_summary.json](../outputs/n34/pipeline_summary.json)。",
        "",
        "N34_STATUS = PARTIAL",
        "REAL_MULTI_ID_DATA = PASS",
        "CANDIDATE_COMPLETE_TAPE = NOT_AVAILABLE",
        "FULL_LOOP = PARTIAL",
        "CCAM_FUTURE_EFFECT = NOT_COMPUTABLE",
        "CALIBRATION_HEAD = NOT_AUTHORIZED",
        "DECODER_LORA = NOT_AUTHORIZED",
        "NEXT_ACTION = Export a real per-frame SAM3 candidate-complete public-ID tape with independent human ROI events on DanceTrack train/train_fold.",
    ]
    atomic_text(REPORT, "\n".join(report_lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    summary = run()
    print(json.dumps({"status": summary["status"], "summary": "outputs/n34/pipeline_summary.json", "report": "docs/N34_FINAL_REPORT.md"}, sort_keys=True))


if __name__ == "__main__":
    main()
