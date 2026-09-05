#!/usr/bin/env python3
"""Materialize the N72R10 controller and human-readable terminal status."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/N72R10"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> int:
    gate_path = OUTPUT_ROOT / "stage_09_gate.json"
    milestone_path = OUTPUT_ROOT / "true_requery_milestone_audit.json"
    corpus_path = OUTPUT_ROOT / "training/corpus_manifest.json"
    training_path = OUTPUT_ROOT / "stage_06_training_status.json"
    stage10_path = OUTPUT_ROOT / "stage_10_training_distribution_audit.json"
    stage11_path = OUTPUT_ROOT / "stage_11_root_cause_classification.json"
    gate = read_json(gate_path)
    milestone = read_json(milestone_path)
    corpus = read_json(corpus_path)
    training = read_json(training_path)
    stage10 = read_json(stage10_path)
    stage11 = read_json(stage11_path)
    metrics = gate.get("metrics", {})
    e1 = metrics.get("E1_vs_E0", {})
    e2 = metrics.get("E2_vs_E1", {})
    train = corpus.get("splits", {}).get("train", {})
    validation = corpus.get("splits", {}).get("validation", {})
    validation_eval = training.get("validation_evaluation", {})
    counts = milestone.get("global_counts", {})
    controller = {
        "schema_version": "N72R10_CONTROLLER_STATUS_V2",
        "status": "N72R10_DEVELOPMENT_GATE_FAIL_READY_FOR_REPORT",
        "updated_at_utc": now_utc(),
        "current_stage": "STAGE_11_ROOT_CAUSE_CLASSIFICATION_COMPLETE",
        "current_round": 1,
        "current_root_cause": stage11.get("primary_root_cause"),
        "historical_gate": "FAIL_FUTURE_REQUERY_EFFECT",
        "n72r10_gate": gate.get("research_gate"),
        "historical_outputs_read_only": True,
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "runtime_future_gt_used": False,
        "train_examples": train.get("example_count"),
        "validation_examples": validation.get("example_count"),
        "source_counts_train": train.get("source_counts", {}),
        "source_counts_validation": validation.get("source_counts", {}),
        "future_requery_train_count": train.get("future_rows_total"),
        "future_requery_positive_count": train.get("future_rows_selected_as_label"),
        "future_requery_negative_count": train.get("future_rows_not_label"),
        "validation_future_requery_count": validation.get("future_rows_total"),
        "validation_future_requery_positive_count": validation.get("future_rows_selected_as_label"),
        "validation_future_requery_negative_count": validation.get("future_rows_not_label"),
        "validation_target_accuracy": validation_eval.get("target_candidate_accuracy"),
        "validation_none_accuracy": validation_eval.get("none_accuracy"),
        "validation_requery_accuracy": None,
        "E1_H20": e1.get("20", {}).get("identity_error_reduction"),
        "E1_H50": e1.get("50", {}).get("identity_error_reduction"),
        "E1_H100": e1.get("100", {}).get("identity_error_reduction"),
        "E2_minus_E1_H20": e2.get("20", {}).get("identity_error_reduction"),
        "E2_minus_E1_H50": e2.get("50", {}).get("identity_error_reduction"),
        "E2_minus_E1_H100": e2.get("100", {}).get("identity_error_reduction"),
        "true_correct": {str(horizon): e2.get(str(horizon), {}).get("true_correct_crossing_count") for horizon in (20, 50, 100)},
        "true_incorrect": {str(horizon): e2.get(str(horizon), {}).get("true_incorrect_crossing_count") for horizon in (20, 50, 100)},
        "protected_regression": {
            "E1_vs_E0": {str(horizon): e1.get(str(horizon), {}).get("protected_regression_count") for horizon in (20, 50, 100)},
            "E2_vs_E1": {str(horizon): e2.get(str(horizon), {}).get("protected_regression_count") for horizon in (20, 50, 100)},
        },
        "live_requery_trigger_count": counts.get("trigger_count", 0),
        "live_requery_candidates": counts.get("fresh_candidate_count", 0),
        "fresh_selected": counts.get("fresh_selected_count", 0),
        "fresh_assigned": counts.get("fresh_assigned_target_count", 0),
        "successful_reacquisitions": counts.get("complete_milestone_count", 0),
        "wrong_reacquisitions": counts.get("fresh_assigned_wrong_count", 0),
        "raw_rebind_count": counts.get("raw_rebind_count", 0),
        "public_stable_rebind_count": counts.get("public_stable_rebind_count", 0),
        "runtime_future_gt_used": False,
        "gpu_allocation": {
            "max_gpu_count": 4,
            "requery_gpu_ids": [1, 2, 3, 4],
            "training_gpu_ids": [1],
            "max_concurrent_processes": 4,
            "one_process_per_gpu": True,
        },
        "next_root_cause": "TRAINING_DISTRIBUTION_AND_VALIDATION_COVERAGE; then calibrated target-edge bridge on sequence-disjoint validation",
        "next_stage": "N72R10_TERMINAL_REPORT_NO_PRODUCTION_PROMOTION",
        "authorization": {
            "production": False,
            "calibration": False,
            "selector": False,
            "decoder_lora": False,
        },
        "evidence": {
            "gate": str(gate_path),
            "gate_sha256": sha256_file(gate_path),
            "milestone": str(milestone_path),
            "milestone_sha256": sha256_file(milestone_path),
            "training_distribution": str(stage10_path),
            "training_distribution_sha256": sha256_file(stage10_path),
            "root_cause": str(stage11_path),
            "root_cause_sha256": sha256_file(stage11_path),
        },
    }
    atomic_json(OUTPUT_ROOT / "CONTROLLER_STATUS.json", controller)
    e2_h20 = e2.get("20", {})
    e2_h50 = e2.get("50", {})
    e2_h100 = e2.get("100", {})
    human = f"""# N72R10 状态

更新时间：{controller['updated_at_utc']}（Asia/Shanghai）

当前阶段：N72R10 true future-frame closed-loop replay、训练和 CPU 门控已完成；开发门失败，准备最终报告。

结论：`{gate.get('research_gate')}`。运行完整性为 PASS，但不授权 production、calibration、selector 或 decoder LoRA。

- 32 个事件、18 条序列；四类 action 计数为 ADD 4、ATOMIC 3、AUTHORITATIVE 14、RECOVER 11。
- 591/591 次不确定性触发都加入了 fresh `FUTURE_FRAME_REQUERY` 候选；33 次被模型选中，12 次进入目标 public-ID 分配，29 个被选候选的离线 target IoU ≥ 0.5。
- 21 次 target-quality fresh selection 未被全局 solver 接纳，主要根因是 `D_MODEL_TO_SOLVER_GLOBAL_COMPETITION`；完整 trigger→selection→assignment→raw rebind→public stable→posthoc wrong→correct 路径计数为 1。
- E1−E0 identity-error reduction：H20 {e1.get('20', {}).get('identity_error_reduction')}、H50 {e1.get('50', {}).get('identity_error_reduction')}、H100 {e1.get('100', {}).get('identity_error_reduction')}，三者 sequence-cluster CI 下界为正。
- E2−E1 identity-error reduction：H20 {e2_h20.get('identity_error_reduction')}（CI 下界 {e2_h20.get('ci_lower')}）、H50 {e2_h50.get('identity_error_reduction')}（CI 下界 {e2_h50.get('ci_lower')}）、H100 {e2_h100.get('identity_error_reduction')}（CI 下界 {e2_h100.get('ci_lower')}）。这是有希望的开发增量，但不是 production/论文确认。
- E1 protected regression 仍为 H20/H50/H100 = 7/7/7；validation FUTURE 正标签为 0，训练规模为 3000/200，不能据此冻结 bridge 或声称泛化。

所有交互仍为 `simulated_from_gt`，不是历史真实 human tape。下一最小动作是获得更大的、同一运行 public-authority 完整且 validation 含正 FUTURE 样本的因果事件池，再训练/评估 target-edge bridge；不复制事件、不延长窗口、不重用 N72R7 静态查询。
"""
    atomic_text(OUTPUT_ROOT / "HUMAN_READABLE_STATUS.md", human)
    print(json.dumps({
        "status": controller["status"],
        "current_stage": controller["current_stage"],
        "next_stage": controller["next_stage"],
        "output": str(OUTPUT_ROOT / "CONTROLLER_STATUS.json"),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
