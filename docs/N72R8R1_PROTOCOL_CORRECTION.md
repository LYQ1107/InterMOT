# N72R8R1 Protocol Correction

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
| D2−D0 | combined learned pipeline effect | `0.052202283849918436` | `[0.006932870370370368, 0.1301041666666666]` |
| D2−D1 | incremental R5 multi-query candidate-generator effect | `0.0` | `[-0.01666666666666667, 0.01666666666666667]` |

因此，D2−D0 的短期正信号不能归因给 R5 单独；R5 incremental H20 benefit remains
unestablished (`0` mean, CI `[-0.01666666666666667, 0.01666666666666667]`).

## 已有结果的正确解读

| 比较 | H20 reduction / CI | H50 reduction / CI | H100 reduction / CI |
|---|---|---|---|
| combined D2−D0 | `0.052202283849918436 / [0.006932870370370368, 0.1301041666666666]` | `0.016087516087516088 / [-0.005587737813801641, 0.05151821326821326]` | `0.009904153354632588 / [-0.0037118972979279987, 0.030355860373018562]` |

H20 CI lower 为 `0.006932870370370368`，说明存在短期开发
信号；H50/H100 CI lower 分别为 `-0.005587737813801641` /
`-0.0037118972979279987`，因此 persistent effect 尚未建立。
这不是“完全没有 effect”，也不是 production success。

## A 阶段验证边界

本阶段只执行了：源代码读取、artifact 语义重分类、JSON/Markdown 生成、后续所需的
静态校验；没有执行 GPU、posthoc 重评分或旧 replay。关键输入 hash：

- `outputs/N72R7/n72r8_final_gate.json`: `53a0a12c6deaabae86e92e3528c218bcf9e8525d64fce1450bc2be49680449fc`
- `outputs/N72R7/r5_requery_posthoc/full_attempt1/n72r7_r5_requery_posthoc_results.json`: `a454cc2497019700ef2b29830d3274815327823099108251e3f85e543b19681e`
- `outputs/N72R7/confirmation/posthoc_attempt1/n72r7_confirmation_posthoc_results.json`: `21232f634fa344e3251a93c67654a7041d6ed331fb9d56bd0528ee5a66ce60ec`
- `scripts/n72r7_confirmation_target_stream.py`: `11dbb84fb6b93075c8fbc15f407af2b1cbfbefe4c5dad5465af76377cd24cdcf`
- `scripts/n72r7_confirmation_replay.py`: `fbd5b4db2e62783fa9162dd9944df97794aaac22a313a63a63d9e1de4a1f29bb`

新机器产物：

- `outputs/N72R8R1/corrected_gate.json`
- `outputs/N72R8R1/corrected_stage12.json`
- `outputs/N72R8R1/corrected_stage13.json`

所有 interaction 仍为 `simulated_from_gt`；real-human evidence 为 0；runtime
`runtime_future_gt_used=false`。calibration、selector、decoder LoRA 与 production 均未授权。
