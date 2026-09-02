# N72R3R1 语义修复中间总结

生成时间：2026-09-02（Asia/Shanghai）

本总结只记录 N72R3 的同一冻结 6-event 输入在语义修复后的可复现重算；没有更换
checkpoint、candidate stream、event、M0--M4 定义、memory 参数或评价窗口。所有
事件仍是 `simulated_from_gt`，不是历史真实人工 tape。

## 语义修复结果

- utility 的 identity 方向已修正为 `baseline_error - treatment_error`：wrong→correct 为
  `+1`，correct→wrong 为 `-1`；IoU delta 和历史 composite 分开保存。
- 正式 replay 通过 `sam3_intermot.association.effect_assignment.solve_effect_assignment`
  调用唯一的 candidate×persistent-public+explicit-NONE solver；没有在正式新路径中另写
  Hungarian solver。
- crossing 已拆成 `TRUE_CORRECT_CROSSING`、`TRUE_INCORRECT_CROSSING`、
  `DIRECTIONAL_IMPROVEMENT`、`DIRECTIONAL_REGRESSION`、`NEUTRAL_CHANGE` 和
  `UNCHANGED`。仅 true crossing 可作为“身份分配翻转”的主证据。
- sequence bootstrap 先对同一序列内的多个 event 求均值，再以 sequence mean 为 bootstrap
  unit；固定 seed=7202、repetitions=2000。

## Old vs new（同一冻结输入）

| 项目 | N72R3 旧宽松语义 | N72R3R1 新语义 |
|---|---:|---:|
| M3/M4 H20 assignment changes | 20 | 20 |
| M3/M4 H20 old broad “correct” | 15 | 0 true-correct crossing |
| M3/M4 H20 directional improvement | 未单独记录 | 15 |
| M3/M4 H20 true incorrect crossing | 未拆分 | 0 |
| M3/M4 H20 identity error reduction | 未作为主指标 | 0 |
| M3/M4 H20 sequence-cluster CI lower | 0（旧 composite/分类） | 0 |
| M1/M2 H20 true-correct crossing | 未拆分 | 0 |

新结果因此不是模型效果改善或恶化，而是把旧报告中的“assignment changed 且 IoU 变大”
重新分类为 directional improvement；没有一项满足 wrong→correct 的严格 crossing。

## Gate 判断

N72R3R1 状态为 `PASS_RUNTIME_SEMANTIC_REPAIR` 但科研 gate 仍为
`FAIL_FUTURE_EFFECT`：M3/M4 的 15 个 directional changes 不等于 true identity
crossings，identity error reduction 的 sequence-cluster CI 下界为 0。该结果保留为
探索性机制证据，不授权 calibration、selector、LoRA 或 production promotion。

下一阶段只在独立的 N72R4 路径中验证两件事：事件帧是否真正继承 `t-1` persistent
runtime，以及官方 SAM3 correction 后的 future candidate stream 是否真实传播。当前
N72R4 的 prestate 与持久 structural probe 已完成，尚未将其称为 official full-loop。

## 机器证据

- `outputs/N72R3R1/semantic_audit.json`
- `outputs/N72R3R1/corrected_replay/attempt1/ccam_paired_replay_results.json`
- `outputs/N72R3R1/corrected_replay/attempt1/old_vs_new_comparison.json`
- `outputs/N72R3R1/corrected_replay/attempt1/n72r3r1_gate.json`
- `outputs/N72R4/full_loop/persistent_candidate_probe/persistent_candidate_probe_validation.json`
- `outputs/N72R4/stage_08_status.json`

