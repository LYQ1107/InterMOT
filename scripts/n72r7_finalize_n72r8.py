#!/usr/bin/env python3
"""Seal the N72R7/R8 evidence without changing any experiment artifact.

This is deliberately a read-existing-artifacts finalizer.  It creates only
small stage/controller summaries and the human-readable report; it never
re-scores, selects, or rewrites a historical result.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
N72R7 = ROOT / "outputs" / "N72R7"
R5_RESULT = N72R7 / "r5_requery_posthoc" / "full_attempt1" / "n72r7_r5_requery_posthoc_results.json"
R5_STATUS = N72R7 / "r5_requery_posthoc" / "full_attempt1" / "stage_r5_posthoc_status.json"
CONF_RESULT = N72R7 / "confirmation" / "posthoc_attempt1" / "n72r7_confirmation_posthoc_results.json"
CONF_STATUS = N72R7 / "confirmation" / "posthoc_attempt1" / "stage_13_confirmation_posthoc_status.json"
CONF_RUNTIME = N72R7 / "confirmation" / "posthoc_attempt1" / "runtime_validation.json"
CONF_AUDIT = N72R7 / "confirmation" / "replay_full_attempt1" / "runtime_audit.json"
CONF_PROTOCOL = N72R7 / "confirmation" / "confirmation_protocol.json"
R5_PROTOCOL = N72R7 / "candidate_generator_protocol.json"
STAGE00_PROTOCOL = N72R7 / "protocol.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def metric(result: dict[str, Any], comparison: str, horizon: int) -> dict[str, Any]:
    value = result.get("aggregate", {}).get(comparison, {}).get(str(horizon))
    if not isinstance(value, dict):
        raise KeyError(f"missing aggregate metric {comparison}/H{horizon}")
    return value


def metric_line(value: dict[str, Any]) -> str:
    ci = value.get("sequence_cluster_bootstrap_95ci", {})
    return (
        f"{value.get('identity_error_reduction')} "
        f"[{ci.get('lower')}, {ci.get('upper')}] / "
        f"assignments={value.get('assignment_change_count')} / "
        f"correct={value.get('posthoc_correct_switch_count')} / "
        f"wrong={value.get('posthoc_wrong_switch_count')} / "
        f"protected-reg={value.get('protected_regression_count')}"
    )


def file_hashes() -> dict[str, str]:
    paths = [
        STAGE00_PROTOCOL,
        R5_PROTOCOL,
        CONF_PROTOCOL,
        R5_RESULT,
        R5_STATUS,
        CONF_RESULT,
        CONF_STATUS,
        CONF_RUNTIME,
        CONF_AUDIT,
    ]
    return {rel(path): sha256(path) for path in paths if path.exists()}


def code_hashes() -> dict[str, str]:
    paths = [
        ROOT / "sam3_intermot" / "reacquisition" / "target_candidate_pool.py",
        ROOT / "sam3_intermot" / "reacquisition" / "hypothesis_beam.py",
        ROOT / "sam3_intermot" / "reacquisition" / "progressive_concept.py",
        ROOT / "sam3_intermot" / "reacquisition" / "target_id_features.py",
        ROOT / "sam3_intermot" / "reacquisition" / "models" / "target_id_decoder.py",
        ROOT / "scripts" / "n72r7_dev_replay.py",
        ROOT / "scripts" / "n72r7_candidate_generator_requery.py",
        ROOT / "scripts" / "n72r7_r5_requery_replay.py",
        ROOT / "scripts" / "n72r7_confirmation_target_stream.py",
        ROOT / "scripts" / "n72r7_run_confirmation_target_stream_batch.py",
        ROOT / "scripts" / "n72r7_confirmation_replay.py",
        ROOT / "scripts" / "n72r7_confirmation_posthoc_score.py",
    ]
    return {rel(path): sha256(path) for path in paths if path.exists()}


def make_stage_summaries(r5: dict[str, Any], confirmation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    r5_h20 = metric(r5, "D2_vs_D0", 20)
    r5_h50 = metric(r5, "D2_vs_D0", 50)
    r5_h100 = metric(r5, "D2_vs_D0", 100)
    conf_h20 = metric(confirmation, "D2_vs_D1", 20)
    stage12 = {
        "schema_version": "N72R7_STAGE12_STATUS_V1",
        "created_at_utc": now_utc(),
        "stage": "12_FREEZE_BEST_CANDIDATE_GENERATOR_MECHANISM",
        "status": "PASS_RESEARCH_ONLY_MECHANISM_FROZEN",
        "mechanism": "CAUSAL_MULTI_QUERY_SAM3_TARGET_REQUERY",
        "decision": "best_candidate_recall_branch_on_development_stream_but_not_production_authorized",
        "development_event_count": r5.get("event_count"),
        "development_independent_sequence_count": r5.get("independent_sequence_count"),
        "interaction_source": r5.get("gate", {}).get("interaction_source"),
        "runtime_future_gt_used": r5.get("gate", {}).get("runtime_future_gt_used"),
        "posthoc_gt_used": True,
        "r5_result": rel(R5_RESULT),
        "r5_h20_identity_error_reduction": r5_h20.get("identity_error_reduction"),
        "r5_h20_sequence_cluster_ci": r5_h20.get("sequence_cluster_bootstrap_95ci"),
        "r5_h50_identity_error_reduction": r5_h50.get("identity_error_reduction"),
        "r5_h50_sequence_cluster_ci": r5_h50.get("sequence_cluster_bootstrap_95ci"),
        "r5_h100_identity_error_reduction": r5_h100.get("identity_error_reduction"),
        "r5_h100_sequence_cluster_ci": r5_h100.get("sequence_cluster_bootstrap_95ci"),
        "candidate_recall_h20": r5_h20.get("candidate_recall"),
        "candidate_recall_h100": r5_h100.get("candidate_recall"),
        "production_authorized": False,
        "calibration_authorized": False,
        "selector_authorized": False,
        "decoder_lora_authorized": False,
        "historical_evidence_preserved": True,
    }
    stage13 = {
        "schema_version": "N72R8_CONFIRMATION_STAGE13_STATUS_V1",
        "created_at_utc": now_utc(),
        "stage": "13_DEFERRED_SEQUENCE_CONFIRMATION_POSTHOC",
        "status": confirmation.get("status"),
        "research_gate": confirmation.get("gate", {}).get("research_gate"),
        "scientific_result": confirmation.get("scientific_result"),
        "event_count": confirmation.get("event_count"),
        "independent_sequence_count": confirmation.get("independent_sequence_count"),
        "interaction_source": confirmation.get("gate", {}).get("interaction_source"),
        "runtime_future_gt_used": confirmation.get("gate", {}).get("runtime_future_gt_used"),
        "posthoc_gt_used": True,
        "result_artifact": rel(CONF_RESULT),
        "runtime_validation": rel(CONF_RUNTIME),
        "runtime_audit": rel(CONF_AUDIT),
        "d2_vs_d1_h20": {
            "identity_error_reduction": conf_h20.get("identity_error_reduction"),
            "sequence_cluster_ci": conf_h20.get("sequence_cluster_bootstrap_95ci"),
            "assignment_change_count": conf_h20.get("assignment_change_count"),
            "posthoc_correct_switch_count": conf_h20.get("posthoc_correct_switch_count"),
            "posthoc_wrong_switch_count": conf_h20.get("posthoc_wrong_switch_count"),
            "protected_regression_count": conf_h20.get("protected_regression_count"),
        },
        "target_session_candidate_recall_h20": conf_h20.get("candidate_recall"),
        "production_authorized": False,
        "calibration_authorized": False,
        "selector_authorized": False,
        "decoder_lora_authorized": False,
        "real_human_evidence": False,
        "historical_evidence_preserved": True,
    }
    return stage12, stage13


def make_gate(r5: dict[str, Any], confirmation: dict[str, Any], stage12: dict[str, Any], stage13: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "N72R8_FINAL_GATE_V1",
        "created_at_utc": now_utc(),
        "status": "N72R8_CONFIRMATION_COMPLETE_FAIL_FUTURE_EFFECT",
        "research_gate": "FAIL_FUTURE_EFFECT",
        "terminal_state": "EVIDENCE_BACKED_EXHAUSTION_OF_REGISTERED_N72R7_ROUTES",
        "development_route": {
            "mechanism": stage12["mechanism"],
            "status": r5.get("status"),
            "event_count": r5.get("event_count"),
            "independent_sequence_count": r5.get("independent_sequence_count"),
            "result": rel(R5_RESULT),
        },
        "confirmation_route": {
            "status": stage13["status"],
            "event_count": confirmation.get("event_count"),
            "independent_sequence_count": confirmation.get("independent_sequence_count"),
            "result": rel(CONF_RESULT),
        },
        "strict_gate": {
            "positive_ci_lower_required": "> 0",
            "r5_h20_ci_lower": metric(r5, "D2_vs_D0", 20).get("sequence_cluster_bootstrap_95ci", {}).get("lower"),
            "r5_h50_ci_lower": metric(r5, "D2_vs_D0", 50).get("sequence_cluster_bootstrap_95ci", {}).get("lower"),
            "r5_h100_ci_lower": metric(r5, "D2_vs_D0", 100).get("sequence_cluster_bootstrap_95ci", {}).get("lower"),
            "confirmation_h20_ci_lower": metric(confirmation, "D2_vs_D1", 20).get("sequence_cluster_bootstrap_95ci", {}).get("lower"),
            "confirmation_assignment_change_count": metric(confirmation, "D2_vs_D1", 20).get("assignment_change_count"),
            "confirmation_identity_error_reduction": metric(confirmation, "D2_vs_D1", 20).get("identity_error_reduction"),
            "confirmation_protected_regression_count": metric(confirmation, "D2_vs_D1", 20).get("protected_regression_count"),
        },
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "production_authorized": False,
        "calibration_authorized": False,
        "selector_authorized": False,
        "decoder_lora_authorized": False,
        "historical_evidence_preserved": True,
        "reason": (
            "R5 increased candidate recall and had a positive H20 development comparison, "
            "but H50/H100 confidence intervals crossed zero and D2 added no future identity "
            "benefit over the current target-session baseline. The reserved two-sequence "
            "confirmation had zero treatment-induced assignment changes and zero identity-error "
            "reduction. No production or downstream learning authorization follows."
        ),
        "stage_12": rel(N72R7 / "stage_12_status.json"),
        "stage_13": rel(N72R7 / "stage_13_status.json"),
    }


def make_controller(base: dict[str, Any], r5: dict[str, Any], confirmation: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    payload = dict(base)
    payload.update(
        {
            "schema_version": "N72R8_CONTROLLER_STATUS_V1",
            "created_at_utc": now_utc(),
            "status": gate["status"],
            "stage": "13_DEFERRED_CONFIRMATION_COMPLETE",
            "research_gate": gate["research_gate"],
            "terminal_state": gate["terminal_state"],
            "real_human_evidence": False,
            "interaction_source": "simulated_from_gt",
            "runtime_future_gt_used": False,
            "production_authorized": False,
            "calibration_authorized": False,
            "selector_authorized": False,
            "decoder_lora_authorized": False,
            "historical_evidence_preserved": True,
            "r5_requery": {
                "result": rel(R5_RESULT),
                "status": r5.get("status"),
                "event_count": r5.get("event_count"),
                "independent_sequence_count": r5.get("independent_sequence_count"),
                "h20": metric(r5, "D2_vs_D0", 20),
                "h50": metric(r5, "D2_vs_D0", 50),
                "h100": metric(r5, "D2_vs_D0", 100),
            },
            "confirmation": {
                "result": rel(CONF_RESULT),
                "status": confirmation.get("status"),
                "scientific_result": confirmation.get("scientific_result"),
                "event_count": confirmation.get("event_count"),
                "independent_sequence_count": confirmation.get("independent_sequence_count"),
                "d2_vs_d1_h20": metric(confirmation, "D2_vs_D1", 20),
                "runtime_validation": rel(CONF_RUNTIME),
            },
            "final_gate": rel(N72R7 / "n72r8_final_gate.json"),
        }
    )
    return payload


def make_human_status(gate: dict[str, Any], r5: dict[str, Any], confirmation: dict[str, Any]) -> str:
    r5_h20 = metric(r5, "D2_vs_D0", 20)
    r5_h100 = metric(r5, "D2_vs_D0", 100)
    conf_h20 = metric(confirmation, "D2_vs_D1", 20)
    r5_ci20 = r5_h20["sequence_cluster_bootstrap_95ci"]
    r5_ci100 = r5_h100["sequence_cluster_bootstrap_95ci"]
    conf_ci = conf_h20["sequence_cluster_bootstrap_95ci"]
    return f"""# N72R8 Confirmation Status

- Stage: deferred-sequence confirmation complete
- Research gate: `{gate['research_gate']}`
- Development route: R5 multi-query SAM3 target re-query, {r5.get('event_count')} events / {r5.get('independent_sequence_count')} independent sequences
- R5 H20 identity-error reduction: `{r5_h20.get('identity_error_reduction')}`, CI `[{r5_ci20.get('lower')}, {r5_ci20.get('upper')}]`
- R5 H100 identity-error reduction: `{r5_h100.get('identity_error_reduction')}`, CI `[{r5_ci100.get('lower')}, {r5_ci100.get('upper')}]`
- Deferred confirmation: {confirmation.get('event_count')} events / {confirmation.get('independent_sequence_count')} sequences
- Confirmation D2−D1 H20 identity-error reduction: `{conf_h20.get('identity_error_reduction')}`, CI `[{conf_ci.get('lower')}, {conf_ci.get('upper')}]`
- Confirmation treatment-induced assignment changes: `{conf_h20.get('assignment_change_count')}`
- Evidence source: `simulated_from_gt`; this is not a real-human study.
- Production, calibration, selector and decoder-LoRA authorization: `false`
"""


def make_report(r5: dict[str, Any], confirmation: dict[str, Any], gate: dict[str, Any], stage12: dict[str, Any], stage13: dict[str, Any]) -> str:
    r5_h20 = metric(r5, "D2_vs_D0", 20)
    r5_h50 = metric(r5, "D2_vs_D0", 50)
    r5_h100 = metric(r5, "D2_vs_D0", 100)
    r5_vs_d1_h20 = metric(r5, "D2_vs_D1", 20)
    conf_d1_h20 = metric(confirmation, "D1_vs_D0", 20)
    conf_d2_h20 = metric(confirmation, "D2_vs_D0", 20)
    conf_d2d1_h20 = metric(confirmation, "D2_vs_D1", 20)
    def ci(v: dict[str, Any]) -> str:
        x = v.get("sequence_cluster_bootstrap_95ci", {})
        return f"[{x.get('lower')}, {x.get('upper')}]"
    hashes = file_hashes()
    code = code_hashes()
    confirmation_protocol = read_json(CONF_PROTOCOL)
    confirmation_protocol_declared_hash = confirmation_protocol.get("protocol_sha256", "MISSING")
    hash_lines = "\n".join(f"- `{key}`: `{value}`" for key, value in hashes.items())
    code_lines = "\n".join(f"- `{key}`: `{value}`" for key, value in code.items())
    return f"""# InterMOT N72R8 Confirmation Report

日期：2026-09-04（Asia/Shanghai）
项目：`InterMOT`
分支：`codex/n72r7-closed-loop-reacquisition`

## 最终结论

本轮完成了 N72R7 注册的候选生成器路线、两条预留序列的 target-session confirmation
以及封存后的 posthoc 评分。科研结论是：

```text
N72R8_STATUS          = COMPLETE_CONFIRMATION_FAIL_FUTURE_EFFECT
RESEARCH_GATE         = FAIL_FUTURE_EFFECT
PRODUCTION            = NOT_AUTHORIZED
CALIBRATION           = NOT_AUTHORIZED
SELECTOR              = NOT_AUTHORIZED
DECODER_LORA          = NOT_AUTHORIZED
REAL_HUMAN_EVIDENCE   = FALSE
```

这不是“模块没有运行”：R5 确实提高了候选覆盖并在开发集 H20 相对旧 B0 基线出现
局部正向结果；但 H50/H100 的 sequence-cluster CI 跨过 0，且相对当前 target-session
基线没有 identity-error 改善。两条预留确认序列中 D2 相对 D1 没有 treatment-induced
assignment change，也没有 identity-error reduction。因此不能授权生产身份跟踪或任何
下游训练。

## 研究边界与冻结输入

- 复用 N72R5R1/N72R6 的 frozen B0/public-assignment stream、候选定义、SAM3 checkpoint、
  Hungarian solver、metric、H20/H50/H100 窗口和 sequence-cluster bootstrap；没有改写
  历史 N36--N72R6 证据。
- R5 是唯一新增候选生成机制：以冻结人工框为查询，使用固定的 CENTER_SHRINK、LEFT、
  RIGHT、UP、DOWN 官方 SAM3 query，独立 session 生成 target-session candidate；不改变
  B0 主流、checkpoint 或求解器。
- confirmation 只使用预留的 `dancetrack0020` ADD 和 `dancetrack0049` ATOMIC 事件，
  没有看 confirmation replay 结果重新选事件、阈值或窗口。
- 事件来源全部是 `interaction_source=simulated_from_gt`，不是历史真实人工点击；真实
  human tape 数量仍为 0。运行时 `runtime_future_gt_used=false`，GT 只在 runtime audit
  封存后用于 posthoc scoring。
- 每个 SAM3 event 使用独立进程；confirmation target batch 为 2/2 PASS、每事件 101
  帧；replay 为 D1/D2 各 2/2 PASS、每事件 101 帧；runtime audit error 为 0。
- `third_party/sam3`、共享 checkpoint、历史输出和指标定义未修改。

## 阶段状态

| 阶段 | 结果 |
|---|---|
| N72R7-00 provenance freeze | PASS，历史输入 hash 保留 |
| N72R7-01 native scope/base-score fix | PASS |
| N72R7-03 union candidate-pool audit | PASS，32/32，runtime GT=false |
| N72R7-05 D1/D2 non-learning replay | PASS execution；`FAIL_FUTURE_EFFECT` |
| N72R7-06/09 decoder mechanism rounds | 已完成，未获得 production authorization |
| N72R7-11 R5 candidate generator + replay | PASS execution；`FAIL_FUTURE_EFFECT` |
| N72R7-12 mechanism freeze | `{stage12['status']}`，research-only |
| N72R8/N72R7-13 deferred confirmation | `{stage13['status']}` |

## R5 全量候选生成器路线

R5 全量为 {r5.get('event_count')} 个事件、{r5.get('independent_sequence_count')} 条独立序列。
候选生成器审计为 32/32 event PASS、4859 行、0 error；re-query 只作为 D2 treatment，
没有把 source candidate 预先绑定 public ID。

| 对比 | H20 | H50 | H100 |
|---|---:|---:|---:|
| R5 D2−D0 identity-error reduction | `{r5_h20.get('identity_error_reduction')}` CI `{ci(r5_h20)}` | `{r5_h50.get('identity_error_reduction')}` CI `{ci(r5_h50)}` | `{r5_h100.get('identity_error_reduction')}` CI `{ci(r5_h100)}` |
| R5 candidate recall | `{r5_h20.get('candidate_recall')}` | `{r5_h50.get('candidate_recall')}` | `{r5_h100.get('candidate_recall')}` |
| treatment assignment changes | `{r5_h20.get('assignment_change_count')}` | `{r5_h50.get('assignment_change_count')}` | `{r5_h100.get('assignment_change_count')}` |
| correct / wrong crossings | `{r5_h20.get('posthoc_correct_switch_count')} / {r5_h20.get('posthoc_wrong_switch_count')}` | `{r5_h50.get('posthoc_correct_switch_count')} / {r5_h50.get('posthoc_wrong_switch_count')}` | `{r5_h100.get('posthoc_correct_switch_count')} / {r5_h100.get('posthoc_wrong_switch_count')}` |
| protected regression | `{r5_h20.get('protected_regression_count')}` | `{r5_h50.get('protected_regression_count')}` | `{r5_h100.get('protected_regression_count')}` |

H20 相对 D0 的均值/CI 是正的，但 H50/H100 的下界分别为
`{r5_h50['sequence_cluster_bootstrap_95ci']['lower']}` 和
`{r5_h100['sequence_cluster_bootstrap_95ci']['lower']}`；这只能说明短期候选覆盖存在
局部信号，不能支持稳定的 persistent identity future effect。更关键的是 R5 D2−D1：
H20 identity-error reduction 为 `{r5_vs_d1_h20.get('identity_error_reduction')}`，CI
`{ci(r5_vs_d1_h20)}`，delta IoU `{r5_vs_d1_h20.get('delta_iou')}`，正确/错误 crossing
`{r5_vs_d1_h20.get('posthoc_correct_switch_count')}/{r5_vs_d1_h20.get('posthoc_wrong_switch_count')}`。

## 两条预留序列的 N72R8 confirmation

confirmation protocol file SHA-256 为 `{sha256(CONF_PROTOCOL) if CONF_PROTOCOL.exists() else 'MISSING'}`；
协议内部声明的 `protocol_sha256` 为 `{confirmation_protocol_declared_hash}`。
0020 使用冻结前缀 public-axis 之后的显式 ADD allocator authority
`state=17 -> public_id=1016`；0049 使用冻结前缀中的显式 ATOMIC pair
`target=1003 / other=1004`。这些 public ID 不是从 GT、raw SAM ID 或候选 index 推断的。

| 对比 | H20 identity-error reduction | CI | assignment changes | correct / wrong crossings | protected regression |
|---|---:|---:|---:|---:|---:|
| D1−D0 | `{conf_d1_h20.get('identity_error_reduction')}` | `{ci(conf_d1_h20)}` | `{conf_d1_h20.get('assignment_change_count')}` | `{conf_d1_h20.get('posthoc_correct_switch_count')} / {conf_d1_h20.get('posthoc_wrong_switch_count')}` | `{conf_d1_h20.get('protected_regression_count')}` |
| D2−D0 | `{conf_d2_h20.get('identity_error_reduction')}` | `{ci(conf_d2_h20)}` | `{conf_d2_h20.get('assignment_change_count')}` | `{conf_d2_h20.get('posthoc_correct_switch_count')} / {conf_d2_h20.get('posthoc_wrong_switch_count')}` | `{conf_d2_h20.get('protected_regression_count')}` |
| D2−D1 | `{conf_d2d1_h20.get('identity_error_reduction')}` | `{ci(conf_d2d1_h20)}` | `{conf_d2d1_h20.get('assignment_change_count')}` | `{conf_d2d1_h20.get('posthoc_correct_switch_count')} / {conf_d2d1_h20.get('posthoc_wrong_switch_count')}` | `{conf_d2d1_h20.get('protected_regression_count')}` |

confirmation 的 D2 candidate recall 从 D0 的 `{conf_d2_h20.get('candidate_recall')}` 提升到
target-session 输入，但 assignment 没有改变；因此 candidate presence 的局部改善没有
穿过 assignment decision boundary。两序列的 bootstrap cluster 数为
`{conf_d2d1_h20['sequence_cluster_bootstrap_95ci']['clusters']}`，不是 32-event 开发集的
统计替代，不能包装成总体 positive confirmation。

## 失败事实与最小修复

以下事实均保留，未删除或覆盖：

1. protocol freeze attempt 1 把 B0 的 24 个 candidate/public 列误当作 24 个 live
   association states；实际只有 16 个 `ASSIGNED_TO_PUBLIC_ID` 状态。attempt 2 改为读取
   solver 的完整 state→public 轴，并成功冻结协议。
2. target-stream smoke attempt 1 要求不存在的协议顶层 `public_id_inference` 字段；修复
   为检查协议级 runtime flag 和事件级 authority flag，同一 0020 attempt 2 通过。
3. target full batch attempt 1 使用无 PyTorch 的 base Python，`ModuleNotFoundError:
   torch`；attempt 2 使用已验证的 `intermot` interpreter 后 2/2 通过。批处理器现在支持
   显式 `--python`，避免同类环境歧义。
4. confirmation runtime audit 的 smoke 误把单事件目录传给只接受 batch manifest 的审计器，
   首次 `FileNotFoundError` 保留；正式 batch audit 0 error 通过。
5. R5 replay audit 的首次规则错误地要求每一帧都有 re-query 行；“无 re-query 行”在协议
   上是合法的，修正后 smoke/full audit 通过。posthoc 首次 import path failure 也保留。
6. 每次 Python 启动的 `osr_lib` `.pth` warning 是环境噪声；相关命令退出码为 0，未被当
   作模型失败。

这些是执行/审计修复，不是降低 gate、伪造候选、修改 checkpoint 或改写历史结果。

## 机制判定与后续边界

证据链现在区分三件事：

- **candidate presence**：R5 明确增加了候选覆盖；
- **assignment change**：相对当前 target-session D1 的新增改变很少，confirmation 为 0；
- **future correctness**：R5 的短期相对 D0 信号无法延续到 H50/H100，confirmation 没有
  产生 identity-error reduction。

所以当前瓶颈不是“没有调用候选生成器”，而是候选传播/空间质量和公共身份关联决策边界
共同限制了可持续收益。R5 已是本轮预注册候选生成路线的最后证据；不能再通过盲目扩大
query 数、调阈值、换 checkpoint、增加 LoRA rank 或重选事件来追逐 PASS。

由于严格 gate 未通过，本轮不运行或授权 calibration head、selector、decoder LoRA、
production identity promotion。真实部署还需要 provenance-complete real-human event tape；
当前 `simulated_from_gt` 证据不能替代真实点击。

## 公开方法参考

方法检索与 pinned commit 审计见 [`docs/N72R7_EXTERNAL_REFERENCE_AUDIT.md`](N72R7_EXTERNAL_REFERENCE_AUDIT.md)。
本轮只将公开方法作为设计线索，没有复制外部代码：

- [DAM4SAM paper](https://arxiv.org/abs/2411.17576) / [repository](https://github.com/jovanavidenovic/DAM4SAM)，commit `9c954504b39ebca4c412f207be0787c26bfac85a`：recent-memory admission 与 distractor memory 线索。
- [SAM2Long paper](https://arxiv.org/abs/2410.16268) / [repository](https://github.com/Mark12Ding/SAM2Long)，commit `d70b50a7936fec55af201244ecde3d4433aff943`：multi-path hypothesis 线索。
- [TrackTrack CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html) / [repository](https://github.com/kamkyu94/TrackTrack)，commit `ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da`：track-perspective association。
- [MOTIP CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.html) / [repository](https://github.com/MCG-NJU/MOTIP)，commit `ffc0e905ac196a603027eca8d18fb0dff48c8bcc`：trajectory context + current detection identity prediction。
- [SeC paper](https://arxiv.org/abs/2507.15852) / [repository](https://github.com/OpenIXCLab/SeC)，commit `0a797af5028623831c016692169df5c621037170`：progressive concept/pixel-level association 线索。

## ICLR 2027 时间与交付决策

按冻结计划，摘要截止 `2026-09-18 AoE`，全文截止 `2026-09-25 AoE`。以本报告日期计，
分别约剩 14 天和 21 天。当前最稳妥的论文结论是“候选覆盖可提升但没有稳定 future
identity effect”，而不是把局部 H20 正值包装为成功。

| 时间 | 交付 |
|---|---|
| 2026-09-04 | 封存 R5 与两序列 confirmation，锁定失败 gate，禁止下游学习 |
| 2026-09-05--09-10 | 仅做报告/可复现性/真实人工 tape 入口完善，不改实验定义 |
| 2026-09-11--09-17 | 若获得真实 provenance-complete tape，先独立 validator；否则保留负结果论文路线 |
| 2026-09-18 | ICLR 摘要截止（AoE） |
| 2026-09-25 | ICLR 全文截止（AoE） |

## 可复现产物与 hash

机器状态：`{rel(N72R7 / 'n72r8_final_gate.json')}`。主要输入/结果 hash：

{hash_lines}

本轮关键代码 hash：

{code_lines}

阶段状态与控制器摘要：

- `{rel(N72R7 / 'stage_12_status.json')}`
- `{rel(N72R7 / 'stage_13_status.json')}`
- `{rel(N72R7 / 'CONTROLLER_STATUS_attempt3.json')}`
- `{rel(N72R7 / 'HUMAN_READABLE_STATUS_attempt3.md')}`

本报告只报告已存在并通过审计的 artifact；不把 `PARTIAL`、候选存在或局部 H20 均值
自动等同于科研成功。
"""


def main() -> int:
    r5 = read_json(R5_RESULT)
    confirmation = read_json(CONF_RESULT)
    stage12, stage13 = make_stage_summaries(r5, confirmation)
    gate = make_gate(r5, confirmation, stage12, stage13)
    base_path = N72R7 / "CONTROLLER_STATUS_attempt2.json"
    base = read_json(base_path) if base_path.exists() else {}
    controller = make_controller(base, r5, confirmation, gate)
    human = make_human_status(gate, r5, confirmation)
    report = make_report(r5, confirmation, gate, stage12, stage13)

    atomic_json(N72R7 / "stage_12_status.json", stage12)
    atomic_json(N72R7 / "stage_13_status.json", stage13)
    atomic_json(N72R7 / "n72r8_final_gate.json", gate)
    atomic_json(N72R7 / "CONTROLLER_STATUS_attempt3.json", controller)
    (N72R7 / "HUMAN_READABLE_STATUS_attempt3.md").write_text(human, encoding="utf-8")
    report_path = ROOT / "docs" / "N72R8_CONFIRMATION_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": gate["status"],
        "research_gate": gate["research_gate"],
        "report": rel(report_path),
        "final_gate": rel(N72R7 / "n72r8_final_gate.json"),
        "stage_12": rel(N72R7 / "stage_12_status.json"),
        "stage_13": rel(N72R7 / "stage_13_status.json"),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
