#!/usr/bin/env python3
"""Correct N72R8 report semantics without rerunning scientific experiments.

The historical N72R8 files remain immutable.  This script reads their sealed
posthoc results and the confirmation source code, then writes a separate
N72R8R1 correction record.  In particular, raw-binding switch counts are
never presented as true assignment crossings and the two inspected deferred
sequences are not treated as fresh R5 confirmation.
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
OUT = ROOT / "outputs" / "N72R8R1"
R5_RESULT = ROOT / "outputs/N72R7/r5_requery_posthoc/full_attempt1/n72r7_r5_requery_posthoc_results.json"
R5_STATUS = ROOT / "outputs/N72R7/r5_requery_posthoc/full_attempt1/stage_r5_posthoc_status.json"
CONF_RESULT = ROOT / "outputs/N72R7/confirmation/posthoc_attempt1/n72r7_confirmation_posthoc_results.json"
CONF_STATUS = ROOT / "outputs/N72R7/confirmation/posthoc_attempt1/stage_13_confirmation_posthoc_status.json"
OLD_GATE = ROOT / "outputs/N72R7/n72r8_final_gate.json"
OLD_REPORT = ROOT / "docs/N72R8_CONFIRMATION_REPORT.md"
TARGET_STREAM = ROOT / "scripts/n72r7_confirmation_target_stream.py"
CONF_REPLAY = ROOT / "scripts/n72r7_confirmation_replay.py"
OLD_FINALIZER = ROOT / "scripts/n72r7_finalize_n72r8.py"


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
        raise KeyError(f"missing metric {comparison}/H{horizon}")
    return value


def code_fact_audit() -> dict[str, Any]:
    target_text = TARGET_STREAM.read_text(encoding="utf-8")
    replay_text = CONF_REPLAY.read_text(encoding="utf-8")
    finalizer_text = OLD_FINALIZER.read_text(encoding="utf-8")
    return {
        "target_stream_uses_ordinary_target_session_worker": "base.run_event(" in target_text,
        "target_stream_has_no_requery_path_argument": "target_requery_path" not in target_text,
        "confirmation_replay_uses_normal_d1_d2": "variant=\"D1\"" in replay_text and "variant=\"D2\"" in replay_text,
        "confirmation_replay_has_no_target_requery_path": "target_requery_path" not in replay_text,
        "historical_finalizer_contains_exhaustion_phrase": "EVIDENCE_BACKED_EXHAUSTION_OF_REGISTERED_N72R7_ROUTES" in finalizer_text,
        "historical_finalizer_contains_switch_crossing_labels": "correct / wrong crossings" in finalizer_text,
        "target_stream_sha256": sha256(TARGET_STREAM),
        "confirmation_replay_sha256": sha256(CONF_REPLAY),
        "historical_finalizer_sha256": sha256(OLD_FINALIZER),
    }


def corrected_metric(value: dict[str, Any]) -> dict[str, Any]:
    ci = value.get("sequence_cluster_bootstrap_95ci", {})
    return {
        "comparison": value.get("comparison"),
        "horizon": value.get("horizon"),
        "identity_error_reduction": value.get("identity_error_reduction"),
        "sequence_cluster_ci": ci,
        "candidate_recall": value.get("candidate_recall"),
        "assignment_change_count": value.get("assignment_change_count"),
        "true_correct_assignment_crossing_count": value.get("true_correct_crossing_count"),
        "true_incorrect_assignment_crossing_count": value.get("true_incorrect_crossing_count"),
        "posthoc_correct_raw_binding_switch_count": value.get("posthoc_correct_switch_count"),
        "posthoc_wrong_raw_binding_switch_count": value.get("posthoc_wrong_switch_count"),
        "protected_regression_count": value.get("protected_regression_count"),
        "wrong_reassociation_rate": value.get("wrong_reassociation_rate"),
        "directional_improvement_count": value.get("directional_improvement_count"),
        "directional_regression_count": value.get("directional_regression_count"),
        "evaluated_frames": value.get("evaluated_frames"),
        "independent_sequence_count": value.get("independent_sequence_count"),
    }


def make_corrected_stage12(r5: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "N72R8R1_CORRECTED_STAGE12_V1",
        "created_at_utc": now_utc(),
        "stage": "N72R8R1_PROTOCOL_CORRECTION_DEVELOPMENT_ATTRIBUTION",
        "status": "PASS_SEMANTIC_CORRECTION_NO_EXPERIMENT_RERUN",
        "historical_evidence_preserved": True,
        "historical_r5_status": read_json(R5_STATUS).get("status"),
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "attribution": {
            "D2_vs_D0": "combined_learned_pipeline_effect; not attributable to R5 alone",
            "D2_vs_D1": "incremental_R5_multi_query_candidate_generator_effect",
            "interpretation": "D2-D0 short-horizon signal is present, while R5 incremental identity benefit is not established",
        },
        "d2_vs_d0": {
            "H20": corrected_metric(metric(r5, "D2_vs_D0", 20)),
            "H50": corrected_metric(metric(r5, "D2_vs_D0", 50)),
            "H100": corrected_metric(metric(r5, "D2_vs_D0", 100)),
        },
        "d2_vs_d1_incremental_r5": {
            "H20": corrected_metric(metric(r5, "D2_vs_D1", 20)),
            "H50": corrected_metric(metric(r5, "D2_vs_D1", 50)),
            "H100": corrected_metric(metric(r5, "D2_vs_D1", 100)),
        },
        "old_labels_corrected": {
            "posthoc_correct_switch_count": "POSTHOC_CORRECT_RAW_BINDING_SWITCH",
            "posthoc_wrong_switch_count": "POSTHOC_WRONG_RAW_BINDING_SWITCH",
            "true_correct_crossing_count": "TRUE_CORRECT_ASSIGNMENT_CROSSING",
            "true_incorrect_crossing_count": "TRUE_INCORRECT_ASSIGNMENT_CROSSING",
        },
        "code_fact_audit": facts,
        "production_authorized": False,
        "calibration_authorized": False,
        "selector_authorized": False,
        "decoder_lora_authorized": False,
    }


def make_corrected_stage13(confirmation: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    d1_h20 = corrected_metric(metric(confirmation, "D1_vs_D0", 20))
    d2_h20 = corrected_metric(metric(confirmation, "D2_vs_D0", 20))
    d2d1_h20 = corrected_metric(metric(confirmation, "D2_vs_D1", 20))
    return {
        "schema_version": "N72R8R1_CORRECTED_STAGE13_V1",
        "created_at_utc": now_utc(),
        "stage": "N72R8R1_OLD_CONFIRMATION_RECLASSIFICATION",
        "status": "USED_DIAGNOSTIC_SEQUENCE_NOT_FRESH_CONFIRMATION",
        "classification": "R2_ORDINARY_TARGET_SESSION_DIAGNOSTIC_SEQUENCES",
        "fresh_confirmation": False,
        "posthoc_outcomes_inspected": True,
        "sequence_count": confirmation.get("independent_sequence_count"),
        "event_count": confirmation.get("event_count"),
        "sequences": ["dancetrack0020", "dancetrack0049"],
        "reason": "confirmation target stream and replay use ordinary N72R6 target-session/current-target D1-D2; no R5 target_requery_path is passed",
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "metrics_are_diagnostic_only": {
            "D1_vs_D0_H20": d1_h20 | {"attribution": "ordinary_target_session_diagnostic"},
            "D2_vs_D0_H20": d2_h20 | {"attribution": "ordinary_target_session_diagnostic"},
            "D2_vs_D1_H20": d2d1_h20 | {"attribution": "ordinary_target_session_diagnostic"},
        },
        "code_fact_audit": facts,
        "production_authorized": False,
        "calibration_authorized": False,
        "selector_authorized": False,
        "decoder_lora_authorized": False,
    }


def make_corrected_gate(r5: dict[str, Any], confirmation: dict[str, Any], stage12: dict[str, Any], stage13: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "N72R8R1_CORRECTED_GATE_V1",
        "created_at_utc": now_utc(),
        "status": "N72R8R1_PROTOCOL_CORRECTED_RESEARCH_CONTINUES",
        "research_gate": "SHORT_HORIZON_SIGNAL_PRESENT_PERSISTENT_EFFECT_NOT_ESTABLISHED",
        "terminal_state": False,
        "historical_gate_preserved": rel(OLD_GATE),
        "historical_report_preserved": rel(OLD_REPORT),
        "corrections": {
            "switch_crossing_terminology": True,
            "old_two_sequence_confirmation_reclassified": True,
            "unsupported_exhaustion_removed": True,
            "d2_d0_attribution_corrected": "combined_learned_pipeline_effect",
            "d2_d1_attribution_corrected": "incremental_R5_candidate_generator_effect",
        },
        "development_signal": {
            "comparison": "D2_vs_D0_combined_pipeline",
            "H20_identity_error_reduction": metric(r5, "D2_vs_D0", 20).get("identity_error_reduction"),
            "H20_sequence_cluster_ci": metric(r5, "D2_vs_D0", 20).get("sequence_cluster_bootstrap_95ci"),
            "short_horizon_ci_lower_positive": True,
            "H50_ci_crosses_zero": True,
            "H100_ci_crosses_zero": True,
        },
        "incremental_r5": {
            "comparison": "D2_vs_D1_R5_multi_query_generator",
            "H20_identity_error_reduction": metric(r5, "D2_vs_D1", 20).get("identity_error_reduction"),
            "H20_sequence_cluster_ci": metric(r5, "D2_vs_D1", 20).get("sequence_cluster_bootstrap_95ci"),
        },
        "old_confirmation": {
            "status": stage13["status"],
            "event_count": confirmation.get("event_count"),
            "sequence_count": confirmation.get("independent_sequence_count"),
            "not_fresh_confirmation": True,
        },
        "incomplete_routes": [
            "true_future_frame_uncertainty_triggered_requery",
            "source_aware_decoder_trained_on_target_and_requery_sources",
            "trusted_target_embedding_memory",
            "distractor_embedding_memory",
            "neighbor_aware_temporal_verification",
            "calibrated_or_global_learned_assignment_interface",
            "true_R5_final_mechanism_confirmation",
            "fresh_untouched_confirmation",
        ],
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "production_authorized": False,
        "calibration_authorized": False,
        "selector_authorized": False,
        "decoder_lora_authorized": False,
        "historical_evidence_preserved": True,
        "stage_12": rel(OUT / "corrected_stage12.json"),
        "stage_13": rel(OUT / "corrected_stage13.json"),
    }


def make_report(r5: dict[str, Any], confirmation: dict[str, Any], gate: dict[str, Any], stage12: dict[str, Any], stage13: dict[str, Any], facts: dict[str, Any]) -> str:
    d20 = metric(r5, "D2_vs_D0", 20)
    d50 = metric(r5, "D2_vs_D0", 50)
    d100 = metric(r5, "D2_vs_D0", 100)
    r20 = metric(r5, "D2_vs_D1", 20)
    c20 = metric(confirmation, "D2_vs_D1", 20)
    def ci(v: dict[str, Any]) -> str:
        x = v.get("sequence_cluster_bootstrap_95ci", {})
        return f"[{x.get('lower')}, {x.get('upper')}]"
    return f"""# N72R8R1 Protocol Correction

日期：2026-09-05（Asia/Shanghai）
项目：`InterMOT`
分支：`codex/n72r9-temporal-closed-loop`

## 修正后的状态

```text
N72R8R1_STATUS      = N72R8R1_PROTOCOL_CORRECTED_RESEARCH_CONTINUES
RESEARCH_GATE       = SHORT_HORIZON_SIGNAL_PRESENT_PERSISTENT_EFFECT_NOT_ESTABLISHED
TERMINAL            = FALSE
PRODUCTION          = NOT_AUTHORIZED
```

本文件是对旧 N72R8 报告/gate 的语义修正，不覆盖旧文件，也没有重跑科学实验。旧的
`FAIL_FUTURE_EFFECT` 数值证据仍然有效；被移除的只是“已耗尽全部注册路线”这一不受
支持的结论。

## A1：switch 与 crossing 术语

旧报告把 `posthoc_correct_switch_count` / `posthoc_wrong_switch_count` 写成了
“correct / wrong crossings”。这不准确：它们记录的是 posthoc 判定的 raw binding
switch，不能替代 assignment crossing。修正后的字段语义为：

| 原始字段 | 修正标签 |
|---|---|
| `posthoc_correct_switch_count` | `POSTHOC_CORRECT_RAW_BINDING_SWITCH` |
| `posthoc_wrong_switch_count` | `POSTHOC_WRONG_RAW_BINDING_SWITCH` |
| `true_correct_crossing_count` | `TRUE_CORRECT_ASSIGNMENT_CROSSING` |
| `true_incorrect_crossing_count` | `TRUE_INCORRECT_ASSIGNMENT_CROSSING` |

没有修改任何原始数值。

## A2：旧两序列确认重分类

代码审计确认：

- `n72r7_confirmation_target_stream.py` 调用普通 N72R6 target-session worker；没有
  `target_requery_path`。
- `n72r7_confirmation_replay.py` 运行普通 D1/D2，并没有传入 R5 re-query source。
- 因此 `dancetrack0020` 与 `dancetrack0049` 是
  `R2_ORDINARY_TARGET_SESSION_DIAGNOSTIC_SEQUENCES`，状态为
  `USED_DIAGNOSTIC_SEQUENCE / NOT_FRESH_CONFIRMATION`。
- 两条序列的 posthoc outcome 已被查看，后续不能再作为 untouched confirmation。

这不删除它们的运行证据，也不把它们重跑为 R5 confirmation。

## A3：移除 unsupported exhaustion

旧 gate 中的 `EVIDENCE_BACKED_EXHAUSTION_OF_REGISTERED_N72R7_ROUTES` 已不再作为
当前结论。仍未完成或未验证的路线包括：

- future-frame uncertainty-triggered requery；
- target/requery source-aware decoder；
- trusted target / distractor embedding memory；
- neighbor-aware temporal verification；
- calibrated/global learned assignment interface；
- 真正的 R5 final-mechanism confirmation；
- fresh untouched confirmation。

所以当前研究继续，但不降低任何 efficacy threshold。

## A4/A5：正确的因果归因

| 对比 | 正确含义 | H20 identity-error reduction | sequence-cluster CI |
|---|---|---:|---:|
| D2−D0 | combined learned pipeline effect | `{d20.get('identity_error_reduction')}` | `{ci(d20)}` |
| D2−D1 | incremental R5 multi-query candidate-generator effect | `{r20.get('identity_error_reduction')}` | `{ci(r20)}` |

因此，D2−D0 的短期正信号不能归因给 R5 单独；R5 incremental H20 benefit remains
unestablished (`0` mean, CI `{ci(r20)}`).

## 已有结果的正确解读

| 比较 | H20 reduction / CI | H50 reduction / CI | H100 reduction / CI |
|---|---|---|---|
| combined D2−D0 | `{d20.get('identity_error_reduction')} / {ci(d20)}` | `{d50.get('identity_error_reduction')} / {ci(d50)}` | `{d100.get('identity_error_reduction')} / {ci(d100)}` |

H20 CI lower 为 `{d20['sequence_cluster_bootstrap_95ci']['lower']}`，说明存在短期开发
信号；H50/H100 CI lower 分别为 `{d50['sequence_cluster_bootstrap_95ci']['lower']}` /
`{d100['sequence_cluster_bootstrap_95ci']['lower']}`，因此 persistent effect 尚未建立。
这不是“完全没有 effect”，也不是 production success。

## A 阶段验证边界

本阶段只执行了：源代码读取、artifact 语义重分类、JSON/Markdown 生成、后续所需的
静态校验；没有执行 GPU、posthoc 重评分或旧 replay。关键输入 hash：

- `outputs/N72R7/n72r8_final_gate.json`: `{sha256(OLD_GATE)}`
- `outputs/N72R7/r5_requery_posthoc/full_attempt1/n72r7_r5_requery_posthoc_results.json`: `{sha256(R5_RESULT)}`
- `outputs/N72R7/confirmation/posthoc_attempt1/n72r7_confirmation_posthoc_results.json`: `{sha256(CONF_RESULT)}`
- `scripts/n72r7_confirmation_target_stream.py`: `{facts['target_stream_sha256']}`
- `scripts/n72r7_confirmation_replay.py`: `{facts['confirmation_replay_sha256']}`

新机器产物：

- `outputs/N72R8R1/corrected_gate.json`
- `outputs/N72R8R1/corrected_stage12.json`
- `outputs/N72R8R1/corrected_stage13.json`

所有 interaction 仍为 `simulated_from_gt`；real-human evidence 为 0；runtime
`runtime_future_gt_used=false`。calibration、selector、decoder LoRA 与 production 均未授权。
"""


def main() -> int:
    r5 = read_json(R5_RESULT)
    confirmation = read_json(CONF_RESULT)
    facts = code_fact_audit()
    stage12 = make_corrected_stage12(r5, facts)
    stage13 = make_corrected_stage13(confirmation, facts)
    gate = make_corrected_gate(r5, confirmation, stage12, stage13)
    report = make_report(r5, confirmation, gate, stage12, stage13, facts)
    atomic_json(OUT / "corrected_stage12.json", stage12)
    atomic_json(OUT / "corrected_stage13.json", stage13)
    atomic_json(OUT / "corrected_gate.json", gate)
    (ROOT / "docs/N72R8R1_PROTOCOL_CORRECTION.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": gate["status"],
        "research_gate": gate["research_gate"],
        "report": "docs/N72R8R1_PROTOCOL_CORRECTION.md",
        "corrected_gate": "outputs/N72R8R1/corrected_gate.json",
        "corrected_stage12": "outputs/N72R8R1/corrected_stage12.json",
        "corrected_stage13": "outputs/N72R8R1/corrected_stage13.json",
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
