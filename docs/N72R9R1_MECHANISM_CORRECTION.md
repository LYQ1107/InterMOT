# N72R9R1 机制标签校正

日期：2026-09-05（Asia/Shanghai）
分支：`codex/n72r10-true-closed-loop`

## 结论

N72R9 的历史数值、报告、checkpoint、候选流和门控文件保持不变。新增的
[`outputs/N72R9R1/corrected_gate.json`](../outputs/N72R9R1/corrected_gate.json) 只校正机制命名：
N72R9 中名为 `TEMPORAL_REQUERY` 的输入来自 N72R7 R5 已冻结的目标会话多查询流，
不是在触发帧重新启动独立 SAM3 并读取当前帧像素的真正未来帧查询。因此本报告将它
称为 `FROZEN_STATIC_REQUERY`（也可描述为 uncertainty-gated static requery reuse），
不再称为 `TRUE_FUTURE_FRAME_REQUERY`。

这不是对 N72R9 失败门的放宽。历史门仍为 `FAIL_FUTURE_REQUERY_EFFECT`，且 N72R9
没有测试 Module 2 的 true closed-loop reacquisition。

## 可审计的 source taxonomy

| 历史字段/来源 | N72R9R1 canonical source | 含义 |
|---|---|---|
| `MAIN_B0_CANDIDATE` | `MAIN_B0_CANDIDATE` | 冻结 N72R6 C0/B0 候选 |
| `TARGET_SESSION_CURRENT_RAW` | `TARGET_SESSION_CURRENT_RAW` | 冻结 N72R6 当前目标会话候选 |
| `TARGET_SESSION_REQUERY` | `STATIC_EVENT_REQUERY` | 冻结 N72R7 R5 目标会话多查询候选 |
| `FUTURE_FRAME_REQUERY` | `FUTURE_FRAME_REQUERY` | 仅预留给 N72R10 新鲜当前帧 SAM3 rescue 候选；N72R9 为 0 |
| `UNKNOWN` | `UNKNOWN` | 不足以支持机制归因 |

旧字段 `TARGET_SESSION_REQUERY` 不能继续无上下文使用，因为它把静态历史流和
真正独立的 future-frame session 混在一起。

## 影响范围

- `TEMPORAL_CURRENT - BASELINE_B0` 的发展信号保留：H20/H50/H100 分别为
  `+0.0978793 / +0.0521236 / +0.0373802`，但 H20 protected regression 为 9，
  仍不能授权生产使用。
- 重新命名后的隔离比较是 `FROZEN_STATIC_REQUERY - TEMPORAL_CURRENT`，其
  H20/H50/H100 为 `0 / -0.0006435 / 0`，不能解释为 true requery 无效。
- N72R9 训练 source 中 `FUTURE_FRAME_REQUERY=0`（train 和 validation），所以
  不能用该 checkpoint 作为新鲜 requery source 的最终模型。
- 所有交互仍是 `simulated_from_gt`，不是真实历史人工点击；运行时 future GT 仍为
  `false`。

## N72R10 的不可替代要求

N72R10 必须使用独立的 `FutureFrameRequerySession`：在触发帧用因果预测框、速度、
尺度、记忆和 uncertainty radius 生成查询，打开事件局部 `f ... event_frame+100`
窗口的独立 SAM3 session，形成实际 `FUTURE_FRAME_REQUERY` 候选；训练语料和验证
集也必须有非零且可审计的该 source，并保留 `NONE`、错误 requery 和遮挡/缺失样本。
运行时不能读取当前帧 GT 或未来 GT，且 fresh candidate 的 raw rebinding 必须保留
immutable public ID。

## 证据与边界

本次仅做文件级机制标注校正，没有重新运行旧 replay，也没有修改
`outputs/N72R9/`。机器证据见 corrected gate；历史 N72R9 final gate 仍是只读输入。
生产、calibration、selector 和 decoder LoRA 均未授权。
