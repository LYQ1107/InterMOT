# N72R2 — Public-ID Closure and Autonomous GT-Simulated Effect Loop

日期：2026-09-02（Asia/Shanghai）
项目：`InterMOT`  ️

## 最终结论

```text
FINAL_STATUS = BLOCKED_CANDIDATE_RECALL
BEST_ROUND   = ROUND_0_BASELINE
```

N72R2 没有完成科研效果验证，也没有把阶段性结果写成 PASS。单窗口的
`TrackManager.final_mot_track_id` authority 和 exact public-ID+NONE 求解器已经通过；
但是独立 session 的多目标候选回收仍不完整，冻结的 handover gate 失败。因此没有
启动模拟人工事件、官方当前帧纠正、public-ID appearance memory、M0--M4/NO_WRITE
replay、H20/H50/H100 评价、future-effect gate 或任何训练。

本轮所有交互证据仍为 `simulated_from_gt` 协议的预留状态；实际 N72R2 event count 为
`0`，没有创建或导入 `real_human` tape。N72R2 的阻塞是工程前置条件（candidate
recall/public mapping），不是对 appearance memory 有效或无效的科学结论。

## 冻结范围与不可变边界

Stage 00 在运行前冻结了：

- checkpoint：`sam3.1_multiplex.pt`，SHA256
  `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`；
- N72R1 Candidate V2、候选顺序、Hungarian 评价、H20/H50/H100 定义和 bootstrap
  `2000` 次/seed `7202`；
- `TrackManager.final_mot_track_id`、显式 IdentityNamespace transaction 或已证明
  transaction 才能成为 public authority；association state ID 不得直接当 public ID；
- 独立 Python/SAM3 session、窗口长度 `160`、overlap `20`、handover gate `1→2→6`；
- runtime 不读取未来 GT。当前 GT 仅在预测 `Y_pre` 冻结之后才可用于受控模拟人工命令，
  后续 GT 只能用于 runtime artifact 冻结后的 posthoc scoring；
- 新产物隔离在 `/data2/usr_for_deadline/SAM3_InterMOT_N72R2/`，没有覆盖 N36--N72R1
  证据、共享 checkpoint 或 `third_party/sam3`。

协议和保护清单：

- `outputs/N72R2/protocol.json`
- `outputs/N72R2/protection_manifest.json`
- `outputs/N72R2/stage_00_status.json`

## Stage 01：same-run public authority 与 exact solver

### 通过的单窗口结果

冻结的 `dancetrack0001` 160-frame window 的独立 exact smoke 为：

| 项目 | 结果 |
|---|---:|
| 帧 | 160/160 |
| Candidate V2 rows | 1,548 |
| same-run authority mapping coverage | 1.0 |
| exact public+NONE solver frames | 160/160 |
| unmapped assignment rows | 0 |
| association state ID 当 public ID | false |
| runtime future GT | false |

authority 来源是 `TrackManager.final_mot_track_id`。每个候选行通过
`candidate_uid → association_state_id → TrackManager authority → public_id` 审计；
public-ID 轴与 association-state 轴分开保存，并保留每候选的显式 NONE slot。

证据：

- `outputs/N72R2/bridge/stage_01_exact_public_assignment_attempt2/done.json`
- `outputs/N72R2/stage_01_status_exact_public_attempt2.json`

第一次 exact smoke 的 `UnboundLocalError`（frame writer 中计数器缺少 `nonlocal`）没有
被覆盖：

- `outputs/N72R2/bridge/stage_01_exact_public_assignment_attempt1/failure.json`

该错误只修复了审计 runner 的计数变量声明，并用相同 frozen window 重跑。之后又做了
一次 CPU-only 合同回归；8 个 focused tests 全部通过。候选/score/metric 定义没有改变。

### 求解器语义修正

候选行选择自身的 `EXPLICIT_NONE` 与 public 轴上没有任何候选的
`NO_CANDIDATE_ASSIGNED` 已明确区分。该修正只影响机器可读审计标签，不改变分数、候选
顺序或 assignment；没有用 0 填充缺失候选，也没有重新运行历史实验。

## Stage 02：multi-window/segment handover

### 原始失败和三轮 recovery

同一序列的 overlap 审计显示，前一个 session 在 overlap 内有 13 个独立历史
`mot_track_id/public_id`，普通第二 session 只有 7 个可持续候选：

```text
initial overlap mapping = 7 / 13 = 0.5384615385
```

保留的 recovery 事实：

1. per-object seed attempt 1：官方 SAM3 对首个 past-state seed 没有返回 observation，
   严格 runner 退出并保留 traceback；
2. attempt 2：失败 seed 被显式记录，重新建立 concept fallback，仍是部分候选输出；
3. attempt 3：同样的官方 box-only recovery 仍无法恢复全部历史候选，concept fallback
   只保留原有 7 个候选。

证据目录：

- `outputs/N72R2/handover/second_window_0416_0575_seed_recovery_attempt1/`
- `outputs/N72R2/handover/second_window_0416_0575_seed_recovery_attempt2/`
- `outputs/N72R2/handover/second_window_0416_0575_seed_recovery_attempt3/`
- `outputs/N72R2/handover/overlap_audit/handover_gate.json`
- `outputs/N72R2/handover/overlap_audit_seed_recovery_attempt3/handover_gate.json`

### 一次有依据的 bulk multi-box targeted smoke

官方 pinned multiplex 代码真实接受 `bounding_boxes` 批量输入，且其 `add_prompt` 会
重置并重建 prompt state。于是新增了 adapter-level 的一次性
`rebind_past_state_boxes`，而不是继续逐对象重复提交。输入是前一个 runtime session
在新边界 frame `416` 的 overlap 状态（对该帧不可见的对象使用更早的已持久化状态），
不是 GT，也不是 frame `416` 之后的提示；loader 只接受 `frame_idx <= 416` 的运行时行。

结果：

| 项目 | 结果 |
|---|---:|
| persisted past-state objects requested | 13 |
| first official multi-box prompt observed | 11 |
| sanitized whole-batch prompt observed | 10 |
| uniquely recovered objects | 10 |
| handover transactions | 9 |
| overlap mapped prior tracks | 9/13 |
| overlap mapping coverage | 0.6923076923 |
| next-session candidate rows | 1,600 |
| next-session frames | 160/160 |
| same-run mapping coverage in that run | 1.0 |
| exact public solver frames | 160/160 |
| runtime future GT | false |

证据：

- `outputs/N72R2/handover/second_window_0416_0575_bulk_rebind_attempt1/done.json`
- `outputs/N72R2/handover/overlap_audit_bulk_rebind_attempt1/handover_gate.json`
- `outputs/N72R2/stage_02_status_final.json`

bulk smoke 中未能唯一回收的输入对象对应 public IDs `7, 8, 11`。其中 public IDs 1
和 7 在 boundary 的历史输出具有相同 box；box-only official prompt 无法证明它们是
两个不同实例。对这样的情况继续用 geometry 或 raw ID 做猜测会制造错误 public
authority，因此被保留为 candidate-recall failure。另有 overlap 事务未覆盖的历史
track 也按缺失计数，没有被标为 PASS。

最终 handover gate：

```text
Stage 02 = BLOCKED_PUBLIC_MAPPING
candidate recall = FAIL_CANDIDATE_RECALL
fixed windows  = 1/6 completed; remaining 5 NOT_RUN
downstream     = unauthorized
```

这里的 `1/6` 只表示单窗口 prerequisite 已完成；不代表 N72R2 完成了六窗口 handover。

## Stage 03--10 状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| 03 causal simulated observer | BLOCKED_PUBLIC_MAPPING | 本地生命周期实现可用；官方 loop 未启动 |
| 04 six action transactions | NOT_RUN_BLOCKED_PREREQUISITE | 等 Stage 02 |
| 05 official current-frame correction | NOT_RUN_BLOCKED_PREREQUISITE | accuracy 未计算 |
| 06 public-ID appearance memory | NOT_RUN_BLOCKED_PREREQUISITE | 未写入任何实验 memory |
| 07 exact public baseline | NOT_RUN_BLOCKED_PREREQUISITE | 只有 Stage 01 单窗口 solver smoke |
| 08 M0--M4/NO_WRITE | NOT_RUN_BLOCKED_PREREQUISITE | 未运行 replay |
| 09 target-scoped association | NOT_RUN_BLOCKED_PREREQUISITE | 未运行 replay |
| 10 strict future-effect gate | NOT_RUN_BLOCKED_PREREQUISITE | 没有 H20/H50/H100 数值 |

Stage 03 的本地 `N72R2SimulatedHumanObserver` 遵守：

```text
begin_prediction(t)
→ freeze Y_pre(t)
→ read current GT(t) only for simulated command
→ official action
→ freeze Y_post(t)
→ hidden memory write
→ first memory read at t+1
```

六类 action 常量和 causal contract 用明确标记的 toy fixture 做了 8 项 focused tests；
这不是实验事件，也不被计入 simulated-human event 数量。

## 效果指标与训练授权

N72R2 因 handover gate 未通过而没有合法的 efficacy denominator。因此以下字段均为
`NOT_RUN/NULL`，绝不是 0：

- current-frame correction accuracy：未运行；
- H20/H50/H100 identity error、IDSW、missing、IoU、wrong reassociation、
  re-correction：未运行；
- assignment changes/correct/incorrect/neutral：未运行；
- untouched/protected regression：未运行；
- sequence-cluster CI95：未运行；
- simulated human events/sequences/actions：`0/0/0`（observer 入口尚未被授权启动）。

下列权限均为 `false`：full-loop、future replay、calibration、selector、decoder LoRA、
production promotion。不能由单窗口 exact solver smoke 推导 appearance memory 或
public-ID 闭环有效。

机器总门：

- `outputs/N72R2/n72r2_final_gate.json`
- `final_status=BLOCKED_CANDIDATE_RECALL`

## 运行隔离和失败保留

- 本轮只使用一张 GPU 执行每个独立窗口；没有并发争抢，同一 GPU 同时最多一个 sequence/
  frame-range 进程；bulk smoke 使用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- 官方 adapter 实际启用 video CPU offload 和 evaluation output CPU offload；state CPU
  offload 没有被声明为启用。没有凭空使用不存在的 SAM3 参数。
- `third_party` 是指向原项目的只读链接；没有修改其源码。
- 没有覆盖 N36--N72R1 outputs，也没有删除失败日志。N72R2 的 code/outputs/report
  与此前 evidence 隔离。
- 失败包括 Stage 02 prepare 的环境 import failure、exact solver smoke 的计数变量
  failure、三轮 per-object recovery failure/partial、以及 bulk rebind partial；均保留
  原始 artifact，修复只作用于对应 runner/adapter。

主要新代码哈希见 `n72r2_final_gate.json` 的
`input_and_code_hashes.modified_code`。冻结输入哈希见
`protocol.json` 与 `protection_manifest.json`。

## 根因判断

当前第一 actionable root cause 是 **跨 session candidate recall / public authority
handover 不完整**：官方 multiplex 的 box-only reinitialization 在同框重复身份、
历史对象暂时不可唯一观察或跨窗口出现的情况下，无法证明所有旧 lineage 仍然存在。
这不是把缺失对象从 raw SAM ID 数字转换为 public ID 就能解决的问题；那样会违背
authority contract。

所以本轮没有进入外观 memory 或 association effect。任何关于“appearance score 改变但
是否改善 assignment”的结论，必须等待 candidate-complete、exact-public、multi-window
runtime 先通过。

## 最小下一步

提供或实现一个经官方 SAM3 验证的 multi-object/session continuation/rebind primitive，
它能够在同一 sequence 的 segment boundary 保留每个 persisted object 的唯一 state/
lineage，并把结果绑定到 `TrackManager.mot_track_id`。输入可以是同一运行时的合法 past
state；不能使用 future GT、raw ID equality、geometry-only guess 或 frame 416 之后的
观测来补齐。

该 primitive 通过 targeted smoke 后，只重跑冻结的 `1→2→6` handover gate；在 gate 通过
以前不启动 simulated event、memory replay、strict future-effect 或训练。

## 机器文件索引

- Final gate：`/data2/usr_for_deadline/SAM3_InterMOT_N72R2/outputs/N72R2/n72r2_final_gate.json`
- Final stage 02：`/data2/usr_for_deadline/SAM3_InterMOT_N72R2/outputs/N72R2/stage_02_status_final.json`
- Final stage 03--10：`/data2/usr_for_deadline/SAM3_InterMOT_N72R2/outputs/N72R2/stage_03_status.json` 至 `stage_10_status.json`
- Research log：`/data2/usr_for_deadline/SAM3_InterMOT_N72R2/worktree/research_log.md`

## Final response fields

```text
FINAL_STATUS: BLOCKED_CANDIDATE_RECALL
BEST_ROUND: ROUND_0_BASELINE
PUBLIC_MAPPING: same_run=1.0; cross_window_overlap=9/13=0.6923076923; handover=FAIL
SIMULATED_HUMAN: events=0; sequences=0; actions=0; runtime_future_gt=false
CURRENT_FRAME_CORRECTION: NOT_RUN
FUTURE: H20=NULL; H50=NULL; H100=NULL
ASSIGNMENT: changes=NULL; correct=NULL; incorrect=NULL; neutral=NULL
SAFETY: untouched=NULL; protected=NULL
STATISTICS: sequence_cluster_CI95=NULL; bootstrap_repetitions=2000 (not run)
ROOT_CAUSE_HISTORY:
  ROUND_0_BASELINE -> same-run authority/exact solver passed only one window
  STAGE_02_RECOVERY_1_TO_3 -> per-object seed/recovery incomplete
  STAGE_02_BULK_REBIND -> 10/13 recovered, 9/13 mapped; duplicate-box ambiguity remains
FINAL_MECHANISM: unresolved cross-session candidate recall/public authority handover
FAILED_BRANCHES: handover; full-loop; six actions; memory; M0-M4; future gate; training
CONFIRMATION: NOT_RUN_BLOCKED_PREREQUISITE
NEXT_DECISION: prove candidate-complete official session rebind, then rerun unchanged 1→2→6
```
