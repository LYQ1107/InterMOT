#!/usr/bin/env python3
"""Create the reproducible N72R3 final gate and report from sealed evidence."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/N72R3"
DOCS = ROOT / "docs"
PROTECTION = OUT / "protection_manifest.json"
EFFECT = OUT / "effect_replay/attempt1/ccam_paired_replay_results.json"
MECHANISM = OUT / "mechanism_rounds/attempt2/mechanism_rounds_summary.json"
STAGE22 = OUT / "stage_22_status.json"
GATE = OUT / "n72r3_final_gate.json"
REPORT = DOCS / "N72R3_FINAL_REPORT.md"
HORIZONS = (20, 50, 100)
VARIANTS = (
    "NO_INTERVENTION",
    "M0_CURRENT_FRAME_CORRECTION_ONLY",
    "M1_HUMAN_EMA_PROTOTYPE",
    "M2_POSITIVE_HUMAN_ANCHORS",
    "M3_NEGATIVE_COMPETITOR_BANK",
    "M4_RELIABILITY_AGE_ADMISSION",
)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
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


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "未定义"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def metric_row(value: dict[str, Any]) -> str:
    ci = value.get("sequence_cluster_bootstrap_95ci", {})
    return " | ".join(
        [
            fmt(value.get("identity_utility")),
            f"[{fmt(ci.get('lower'))}, {fmt(ci.get('upper'))}]",
            fmt(value.get("assignment_change_count"), 3),
            fmt(value.get("assignment_change_correct_count"), 3),
            fmt(value.get("assignment_change_incorrect_count"), 3),
            fmt(value.get("assignment_change_neutral_count"), 3),
            fmt(value.get("wrong_reassociation_frames"), 3),
            fmt(value.get("protected_regression_count"), 3),
        ]
    )


def historical_audit(protection: dict[str, Any]) -> dict[str, Any]:
    checks = []
    for item in protection.get("historical_inputs", []):
        path = Path(item["path"])
        actual = sha256(path) if path.is_file() else None
        checks.append({"path": str(path), "expected": item.get("sha256"), "actual": actual, "match": actual == item.get("sha256")})
    return {"count": len(checks), "all_match": bool(checks) and all(item["match"] for item in checks), "checks": checks}


def status_map() -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(OUT.glob("stage_*_status.json")):
        result[path.stem] = read(path)
    return result


def build_gate(effect: dict[str, Any], mechanism: dict[str, Any], protection: dict[str, Any], protection_audit: dict[str, Any], statuses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stage22 = statuses["stage_22_status"]
    stage21 = statuses["stage_21_status"]
    stage18 = statuses["stage_18_status"]
    stage19 = statuses["stage_19_status"]
    stage16 = statuses["stage_16_status"]
    stage17 = statuses["stage_17_status"]
    return {
        "schema_version": "N72R3_FINAL_GATE_V1",
        "status": "N72R3_COMPLETE_EVIDENCE_EXHAUSTED_MECHANISM_BRANCHES_NO_STRICT_EFFECT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_gate": "FAIL_FUTURE_EFFECT",
        "final_scientific_status": "EXHAUSTED_MECHANISM_BRANCHES_NO_EFFECT_CONFIRMATION",
        "architecture_status": "PASS_PERSISTENT_PUBLIC_IDENTITY_ARCHITECTURE",
        "two_window_persistence": "PASS_TWO_WINDOW_PERSISTENCE",
        "six_window_persistence": "PASS_SIX_WINDOW_PERSISTENCE",
        "candidate_recall_status": "PARTIAL_CANDIDATE_RECALL_PERFORMANCE_ONLY",
        "current_correction_status": "PASS_CURRENT_FRAME_OFFICIAL_CORRECTION",
        "appearance_memory_status": "PASS_TARGET_PUBLIC_512D_MEMORY_WRITE_CAUSAL_BOUNDARY",
        "effect_execution_status": "PASS_EXECUTION_FAIL_FUTURE_EFFECT",
        "mechanism_rounds_status": mechanism.get("status"),
        "event_count": effect.get("event_count"),
        "independent_sequence_count": effect.get("independent_sequence_count"),
        "eligible_event_quota_target": effect.get("event_quota_target"),
        "eligible_event_shortfall": effect.get("eligible_event_shortfall"),
        "action_counts": statuses["stage_14_status"].get("action_counts", {}),
        "structural_gate_checks": {
            "stage18_exact_public_baseline": stage18.get("status") == "PASS_BASELINE_N72R3_PERSISTENT_PUBLIC",
            "public_restore_coverage": stage18.get("public_identity_restore_coverage"),
            "public_renumber_count": stage18.get("public_renumber_count"),
            "lineage_loss_count": stage18.get("lineage_loss_count"),
            "assignment_artifact_complete": stage18.get("assignment_artifact_complete"),
            "runtime_gt_leakage_count": stage18.get("runtime_gt_leakage_count"),
            "stage16_correction_success_rate": stage16.get("correction_success_rate"),
            "stage17_finite_memory_writes": stage17.get("finite_512d_memory_writes"),
            "stage17_first_visible_event_plus_one": stage17.get("first_visible_frame_all_event_plus_one"),
            "stage19_candidate_recall_is_performance_only": stage19.get("candidate_recall_is_performance_only"),
            "stage21_target_scoped_audit": stage21.get("status") == "PASS_STAGE21_TARGET_SCOPED_ASSOCIATION_AUDIT",
            "stage21_non_target_score_cells_bitwise_checked": stage21.get("non_target_score_cells_checked_bitwise"),
            "stage21_target_scope_failures": stage21.get("target_scoped_row_failures"),
        },
        "effect_gate": effect.get("gate", {}),
        "runtime_future_gt_used": False,
        "real_human_tape": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "production_authorized": False,
        "calibration_head_authorized": False,
        "selector_authorized": False,
        "decoder_lora_authorized": False,
        "checkpoint_changed": False,
        "candidate_definition_changed": False,
        "hungarian_solver_changed": False,
        "third_party_sam3_changed": False,
        "historical_outputs_changed": False,
        "historical_input_hash_audit": protection_audit,
        "protected_worktree_policy": protection.get("policy", {}),
        "effect_artifact": {"path": str(EFFECT), "sha256": sha256(EFFECT)},
        "mechanism_rounds_artifact": {"path": str(MECHANISM), "sha256": sha256(MECHANISM)},
        "failure_evidence_retained_under": "outputs/N72R3/attempts/",
        "next_minimum_action": "Collect additional protocol-compliant eligible events or externally supplied real-human tape; do not duplicate events, train downstream modules, alter checkpoint/candidates/Hungarian, or call this exploratory result a PASS.",
    }


def build_report(gate: dict[str, Any], effect: dict[str, Any], mechanism: dict[str, Any], protection_audit: dict[str, Any], statuses: dict[str, dict[str, Any]]) -> str:
    aggregate = effect["aggregate"]
    stage19 = statuses["stage_19_status"]
    stage16 = statuses["stage_16_status"]
    stage17 = statuses["stage_17_status"]
    stage18 = statuses["stage_18_status"]
    stage21 = statuses["stage_21_status"]
    lines: list[str] = []
    lines += [
        "# N72R3 最终报告：跨独立 SAM session 的 Persistent Public Identity",
        "",
        f"生成时间：{gate['created_at_utc']}（Asia/Shanghai 任务环境）  ",
        "最终 research gate：**FAIL_FUTURE_EFFECT**。本报告不是把失败包装成成功：结构性 identity closure 已通过，但当前候选流上的 future identity effect 没有严格确认。",
        "",
        "## 1. 一页结论",
        "",
        "- `public_id` 最终属于 sequence-lifetime 的 `PersistentIdentityRecord`，不属于当前 SAM candidate、raw SAM ID 或 association-local numeric state ID。`mot_track_id == public_id`，lineage、appearance/motion/lost state 与一个外层 TrackManager 由持久 runtime 管理。",
        "- 用户给出的 #1007 合约在 CPU toy stress 中通过：Session A candidate 17 绑定后，session boundary 只清理 session-local candidate/SAM binding；Session B 的 NONE 记录为 LOST，identity 未删除；新 raw SAM candidate 8 再绑定后仍返回 public_id 1007、mot_track_id 1007、lineage 不变。这个 stress 不是科学效果结果，但验证了目标状态机。",
        "- 2-window、6-window exact-public structural baseline 通过：public restore coverage=1、renumber=0、lineage loss=0、assignment artifacts complete、runtime GT leakage=0。",
        "- Stage 16/17 对 6 个冻结 `simulated_from_gt` 事件真实调用官方 SAM3 当前帧纠正，并把真实当前图像 ROI 的 512-D OSNet feature 写入 target public ID；6/6 correction、6/6 finite memory writes，event frame 不读新 memory，首次可见为 event+1。它不是 real human tape。",
        "- Stage 20–22 只找到 6 个 eligible events / 6 个独立序列（目标 40 / 20），没有重复事件凑数。M1/M2 改变目标 score 但没有 assignment crossing；M3/M4 出现少量 crossing，H20 utility 约 0.000749，但 sequence-cluster CI 下界为 0，因此严格 gate 失败。",
        "- 结论：主要瓶颈是候选召回与 association decision boundary 的组合，而不是 persistent public-ID ownership。当前不授权 calibration head、selector、decoder LoRA 或 production promotion。",
        "",
        "## 2. 用户要求的持久身份语义",
        "",
        "```text",
        "Persistent Public Identity #1007",
        "├── public_id = 1007       永久稳定",
        "├── lineage_id              永久稳定",
        "├── TrackManager track      永久稳定",
        "├── association state      显式绑定但不拥有 public authority",
        "├── appearance memory      持久",
        "├── motion state            持久",
        "├── ACTIVE / LOST           可变化",
        "│",
        "├── Session A candidate 17 → assignment",
        "├── Session boundary       → 清空 candidate binding，identity 仍存在",
        "├── Session B: NONE         → identity = LOST",
        "└── Session B candidate 8  → association/rebind → public_id 仍为 1007",
        "```",
        "",
        "Stage 01 的 probe 同时证明旧 N72R2 candidate-first authority 有缺陷：同一个 association state 可以被旧 bridge 接受为两个 public IDs。N72R3 用 immutable state→lineage→persistent public binding、外层 birth decision 和显式 LOST/NONE 取代它；IoU/appearance 只能是 association evidence，不能创建 exact authority。",
        "",
        "## 3. 分阶段状态",
        "",
        "| 阶段 | 结果 | 科学含义 |",
        "|---|---|---|",
        "| 00–01 | protocol frozen；旧 authority circularity confirmed | 输入与边界冻结，旧语义缺陷保留 |",
        "| 02–08 | structural PASS | persistent identity、authority bridge、session boundary、snapshot 通过；不是效果结果 |",
        "| 09–11 | structural PASS | 1/2/6-window public continuity 通过；candidate recall 另行诊断 |",
        "| 12–15 | audit/oracle/atomic transaction PASS | GT 审计修正、模拟 oracle 隔离、故障回滚通过 |",
        "| 16–17 | official correction + 512-D memory PASS | 当前帧事务与 t+1 因果边界真实执行 |",
        "| 18 | exact-public baseline PASS | 6/6 eligible windows 的 public continuity 通过 |",
        "| 19 | candidate recall PARTIAL | 性能诊断，不是 identity continuity gate |",
        "| 20 | runtime execution PASS，eligible shortfall | 6 events×6 variants×H20/H50/H100 已落盘 |",
        "| 21 | target-scoped audit PASS | 380,322 个 non-target score cells bitwise unchanged；global Hungarian 保留 |",
        "| 22 + 5 rounds | execution complete，FAIL_FUTURE_EFFECT | 没有严格正 future utility，停止下游学习授权 |",
        "",
        "## 4. 候选召回诊断",
        "",
        f"Stage 19 使用 {stage19.get('event_count')} 个事件、{stage19.get('independent_sequence_count')} 个独立序列，GT 只在 runtime artifact 冻结后 posthoc 读取。候选召回不是结构性 public-ID gate。",
        "",
        "| horizon | 全部候选 recall | AUTHORITATIVE_REASSIGN | RECOVER_IDENTITY | target GT present / candidate absent |",
        "|---:|---:|---:|---:|---:|",
    ]
    by_h = stage19.get("by_horizon", {})
    by_a = stage19.get("by_action", {})
    for horizon in HORIZONS:
        all_value = by_h.get(str(horizon), {})
        auth = by_a.get("AUTHORITATIVE_REASSIGN", {}).get(str(horizon), {})
        recover = by_a.get("RECOVER_IDENTITY", {}).get(str(horizon), {})
        lines.append(f"| H{horizon} | {fmt(all_value.get('candidate_recall'))} | {fmt(auth.get('candidate_recall'))} | {fmt(recover.get('candidate_recall'))} | {fmt(all_value.get('candidate_absent_frames'), 3)} |")
    lines += [
        "",
        "RECOVER 的 H20 recall 只有 0.35；因此不存在 candidate 的帧不能由 appearance-only score 修复。该结果支持后续 candidate recovery/detection 分支，但不能把 candidate absent 写成 public identity 被删除。",
        "",
        "## 5. Stage 20–22 paired replay",
        "",
        f"输入是冻结 Candidate V2 stream；事件为 {effect.get('event_count')} 个、独立序列 {effect.get('independent_sequence_count')} 个，低于预注册目标 {effect.get('event_quota_target')}/{effect.get('sequence_quota_target')}。Stage 14 action 分布是 {json.dumps(gate.get('action_counts', {}), ensure_ascii=False)}；没有 ATOMIC_ID_SWAP、ADD_NEW_IDENTITY 的 eligible 事件，未跨序列拼接或复制事件。所有交互明确标记 `simulated_from_gt`，不是历史真实点击。",
        "",
        "变体定义：`NO_INTERVENTION`；`M0` 当前帧纠正 only；`M1` human EMA prototype；`M2` prototype + positive anchors；`M3` 加 competitor negatives；`M4` 加固定 reliability/age admission。未来 score 只改变 target public-ID row；最终 assignment 的其他 public ID 被动变化时记作 `solver_coupled_collateral`。",
        "",
        "| variant | horizon | identity utility | 95% CI（sequence cluster） | assignment changes | correct | incorrect | neutral | wrong reassociation | protected regression |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        for horizon in HORIZONS:
            value = aggregate[variant][str(horizon)]
            lines.append(f"| {variant} | H{horizon} | {metric_row(value)} |")
    lines += [
        "",
        "上表列顺序是：utility、CI、assignment changes、correct、incorrect、neutral、wrong reassociation、protected regression。",
        "",
        f"主 gate 预注册为 `{effect.get('gate', {}).get('primary_variant')}` @ H20。M2 H20 utility={fmt(aggregate['M2_POSITIVE_HUMAN_ANCHORS']['20'].get('identity_utility'))}，CI={fmt(aggregate['M2_POSITIVE_HUMAN_ANCHORS']['20'].get('sequence_cluster_bootstrap_95ci', {}).get('lower'))}..{fmt(aggregate['M2_POSITIVE_HUMAN_ANCHORS']['20'].get('sequence_cluster_bootstrap_95ci', {}).get('upper'))}；correct assignment changes=0，所以 gate 失败。M3/M4 H20 有 15 个 correct、0 个 incorrect、5 个 neutral changes，但 CI 下界仍是 0，不能包装为确认。",
        "",
        "## 6. 机制诊断与五轮收口",
        "",
        "- Round 1：候选召回上限。确认 absent candidate 帧是 appearance-only association 的硬上限，尤其 recover 分支；没有添加 GT candidate 或重写候选流。",
        "- Round 2：逐帧 global-assignment residual。典型 required target-row residual 中位数约 7.57；M1/M2/M3/M4 在 cheapest alternative 列的 delta 中位数约 0.181/0.685/0.150/0.145。只比较同一替代列，避免把不同列的 delta 混合。",
        "- Round 3：仅 target row 的假设性 boundary probe。对每个可检查未来帧加入恰好超过 residual 的 1e-6，600/600 per variant 能让 constrained alternative crossing；这是说明 solver 接口可穿越的诊断，不是有效 learned score，也没有改 production。",
        "- Round 4：固定 age/admission。M3 memory read/admitted=600/600；M4 read/admitted=600/480，120 帧在固定 age>80 后拒绝；不按未来指标调 admission。H20/H50 两者结果相同，H100 未产生可靠新增收益。",
        "- Round 5：统计功效与 protected-ID。2000 次、独立序列 cluster bootstrap 正确执行，但只有 6 clusters，主 CI 下界为 0；protected regression 为 0。下一步需要更多独立 eligible 事件或真实人工 tape，不能复制当前 6 个事件。",
        "",
        "Round 2 的第一次实现曾错误地跨列比较 delta，已保存 `outputs/N72R3/attempts/mechanism_rounds_semantic_audit_attempt1.json`，修复后在独立 `mechanism_rounds/attempt2/` 重跑。机制轮次脚本第一次语法失败也保存了退出码 1 与 traceback；这些不是科学 PASS。",
        "",
        "## 7. 因果、映射和保护审计",
        "",
        f"- Stage 16 correction success rate={fmt(stage16.get('correction_success_rate'))}；Stage 17 finite 512-D writes={fmt(stage17.get('finite_512d_memory_writes'))}，event-frame hidden={fmt(stage17.get('event_frame_write_hidden'))}，first visible event+1={fmt(stage17.get('first_visible_frame_all_event_plus_one'))}。",
        f"- Stage 18 public restore coverage={fmt(stage18.get('public_identity_restore_coverage'))}，renumber={fmt(stage18.get('public_renumber_count'))}，lineage loss={fmt(stage18.get('lineage_loss_count'))}，runtime GT leakage={fmt(stage18.get('runtime_gt_leakage_count'))}。",
        f"- Stage 21 candidate stream 在各 variant 完全共享；global Hungarian 保留；target-scoped non-target score cells={fmt(stage21.get('non_target_score_cells_checked_bitwise'), 3)}，row failures={fmt(stage21.get('target_scoped_row_failures'), 3)}。",
        "- 每个 replay artifact 对每帧均保存 candidate UID/native/adapter mapping、base/appearance/fused score matrix、assignment、memory read/admission 和 public state axis；GT 字段没有进入 runtime artifact。",
        "",
        "## 8. 失败事实与修复保留",
        "",
        "| 证据 | 首个 actionable root cause / 处理 |",
        "|---|---|",
        "| `stage01_authority_audit_attempt1_failure.json` | 独立脚本缺少 worktree `sys.path`，`ModuleNotFoundError`；修复入口后重跑 |",
        "| `stage01_authority_audit_attempt2_pre_repair.json` | probe 暴露旧 bridge 可让同一 state 接受两个 public IDs；保留为 root-cause 事实，改用 probe-based gate |",
        "| `stage09_11_failure_attempt1.json` | aggregation 读取不存在的 `runtime_final_audit`；兼容真实 `runtime_audit` 后 targeted rerun |",
        "| `stage13_oracle_pytest_attempt1_failure.json` | toy oracle 字段名不一致（`other_public` vs `other_public_id`）；修复后 11 tests pass |",
        "| `stage15_transaction_pytest_attempt1_failure.json` | rollback restore 替换 identity 对象，外部引用失效；改为 in-place restore 后 10 tests pass |",
        "| `stage16` smoke attempts 1–5 | 依次保留 checkpoint capacity mismatch、adapter 未注册 observation、错误绑定未先释放、`wrong_record` 未返回、causal guard 调用方式错误；各自最小修复后 attempt6 pass |",
        "| `stage16_mapping_test_fixture_attempt1/2_failure.json` | toy fixture 触发真实 checkpoint guard；最终 fixture 显式禁用模型加载，仅用于 adapter mapping test，10/10 pass |",
        "| `mechanism_rounds_py_compile_attempt1_failure.json` | `round3_boundary_probe` 类型注解缺 `]`，退出码1；修复后编译/执行 pass |",
        "| `mechanism_rounds_semantic_audit_attempt1.json` | Round2 跨列比较造成潜在 boundary reachability 误报；修复并以 attempt2 独立输出重跑 |",
        "",
        "失败 artifact 均位于 `outputs/N72R3/attempts/` 或对应 attempt 目录，没有删除、覆盖或将失败改写为 PASS。",
        "",
        "## 9. 隔离、输入哈希与资源边界",
        "",
        "- 所有本轮新代码和输出都在独立 N72R3 worktree / `outputs/N72R3/`；N36–N72R2 历史输出只读，N72R3 protection manifest 的历史输入 hash 全部匹配。",
        f"- checkpoint SHA-256：`{read(OUT / 'protocol.json')['frozen_inputs']['checkpoint']['sha256']}`；checkpoint、candidate definition、Hungarian solver、metric/bootstrap 定义未改变。",
        "- 官方 SAM3 third-party 目录未修改；没有创建 real human tape；没有把 `simulated_from_gt` 改名为 `real_human`。",
        "- Stage 16–17 最多使用四张 GPU，每卡单独事件进程；Stage 20–22 与五轮机制诊断为 CPU-only。",
        "",
        "## 10. 最终授权与下一步",
        "",
        "当前明确禁止：calibration head、selector、decoder LoRA、共享 checkpoint 更新、production identity promotion。N72R3 已经完成结构链与五轮 mechanism evidence，但没有严格 future-effect confirmation。",
        "",
        "最小下一步是：在不改变冻结协议、candidate definition、checkpoint 或 Hungarian 的前提下，采集更多独立 eligible current-frame events，或接入外部 provenance-complete real-human tape；优先解决 recover candidate recall，再重新进行相同 paired replay。不能复制当前 6 个事件、不能用 synthetic 事件冒充真实点击，也不能用一次性 boundary probe 代替真实收益。",
        "",
        "若沿用 ICLR 2027 时间约束：摘要截止 2026-09-18 AoE、全文截止 2026-09-25 AoE；截至 2026-09-02，优先整理结构性贡献与诚实的负 future-effect 结果，不应在缺少证据时赶工训练下游模块。",
        "",
        "## 11. 机器可读证据",
        "",
        "- `outputs/N72R3/n72r3_final_gate.json`：最终 gate 与授权状态。",
        "- `outputs/N72R3/effect_replay/attempt1/ccam_paired_replay_results.json`：Stage 20–22 完整 paired replay 与 posthoc 指标。",
        "- `outputs/N72R3/stage_21_status.json`：target-scoped bitwise audit。",
        "- `outputs/N72R3/mechanism_rounds/attempt2/mechanism_rounds_summary.json`：五轮根因诊断。",
        "- `outputs/N72R3/stage_18_status.json`、`stage_19_status.json`、`stage_16_status.json`、`stage_17_status.json`：结构、召回、官方纠正与 memory 证据。",
        "",
        "**最终不是 PASS：`research_gate=FAIL_FUTURE_EFFECT`; `production_authorized=false`; `real_human_tape=false`。**",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    protection = read(PROTECTION)
    protection_audit = historical_audit(protection)
    statuses = status_map()
    effect = read(EFFECT)
    mechanism = read(MECHANISM)
    gate = build_gate(effect, mechanism, protection, protection_audit, statuses)
    if not protection_audit["all_match"]:
        raise RuntimeError("historical input hash audit failed")
    atomic_json(GATE, gate)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(REPORT.suffix + ".tmp")
    temporary.write_text(build_report(gate, effect, mechanism, protection_audit, statuses), encoding="utf-8")
    os.replace(temporary, REPORT)
    print(json.dumps({"status": gate["status"], "research_gate": gate["research_gate"], "gate": str(GATE), "report": str(REPORT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
