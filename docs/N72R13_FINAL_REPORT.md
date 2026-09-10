# InterMOT N72R13 Final Report

**阶段：** N72R13 — Temporal Intervention Value for Long-Horizon Identity Correction
**日期：** 2026-09-10（Asia/Shanghai）
**分支：** `codex/n72r13-temporal-intervention-value`
**基线：** N72R12 commit `05bcd696de403cb9a3011f7160b63743f146ef80`

## 结论先行

N72R13 的结构化执行和正式评估已完成，但研究门控失败：

```text
N72R13_STATUS                 = STOP_NO_TEMPORAL_INTERVENTION_HEADROOM
FORMAL_ORACLE                = 32/32 events, 18 independent sequences
EFFECTIVE_PAIRED_OPPORTUNITIES = 176
ORACLE_APPLY_TIV             = 0
ORACLE_KEEP_BASE             = 176
TRACKEVAL                    = 288/288 records, H20/H50/H100
OFFICIAL_H100_GLOBAL_HEADROOM = false
VALUE_MODEL                  = NOT_AUTHORIZED
CALIBRATION/SELECTOR/LORA    = NOT_AUTHORIZED
REQUERY_NEXT                 = true
```

最重要的因果限制是：176 个有效机会全部选择了 `KEEP_BASE`。因此没有一次实际
`APPLY_TIV` 被执行或证明有效。Oracle 与冻结 E0 之间出现的 custom identity 差异，
不能被表述为 temporal intervention 的收益，而只能作为“顺序 KEEP 轨迹与历史 E0
轨迹不同”的混杂诊断。

## 1. 研究问题与冻结边界

本轮检验的假设是：对当前纠正产生的候选 proposal，若从 t+1 起比较 `APPLY` 和
`KEEP` 的未来价值，是否能在 H20/H50/H100 上减少目标身份错误、丢失、错误重关联和
再次纠正，同时不伤害其他 ID。主 horizon 是 H20，另记 H5、H50，并完成 H100
正式窗口评估。

只读复用了 N72R12/N72R11R5R1/N72R9 的：

- 32 个事件、18 条独立序列的冻结 protocol；
- N72R11R4 PCTIS checkpoint 和 exact candidate stream；
- public/native/local mapping、Hungarian solver、H20/H50/H100 定义；
- N72R11R5R1 的 E0/E1B sealed runtime；
- 官方 TrackEval checkout，commit `12c8791b303e0a0b50f753af204249e622d0281a`。

没有修改 `third_party/sam3`、TrackEval、历史 N36–N72R12 输出、checkpoint、
candidate definition 或 production association formula。所有事件仍是
`interaction_source=simulated_from_gt`，不是历史真实人工点击；真实 human tape 仍为 0。

输入哈希：

- N72R9 protocol：`e012ecc3bd64fec4409fccd57d920f3690b8b1c4b5a2dd557e03e2bfb43ef0e9`；
- PCTIS checkpoint：`77b41dbe2fc0f03c48ca4b8cda22e6b219eb58e266566eb96fcf0df3fb8c2c16`；
- frozen E1B manifest：`42540a5c7d999ecfd9e10a075e6cefb62e832c7e3a98f772ac81a27daa005dff`。

## 2. 实现内容

新增了以下隔离研究代码：

- `sam3_intermot/reacquisition/temporal_intervention_value.py`：深拷贝
  `TemporalIdentityState`、同一 state 的 KEEP/APPLY 分支、被动 BASE future rollout、
  branch digest 和 runtime seal；
- `scripts/n72r13_interface_audit.py`：接口、签名、冻结输入和代码 hash 审计；
- `scripts/n72r13_validate_temporal_state.py`：toy-only state clone contract smoke；
- `scripts/n72r13_audit_intervention_opportunities.py`：冻结 516 interaction corpus 的
  metadata-only opportunity audit；
- `scripts/n72r13_temporal_oracle.py`：逐事件、逐有效机会的 sequential post-hoc
  Oracle；
- `scripts/n72r13_export_window_trackeval.py` 与
  `scripts/n72r13_run_window_trackeval.py`：独立的 32×3×3 interaction-window
  TrackEval export/run；
- `scripts/n72r13_finalize_gate.py`：读取 sealed runtime 后补齐 custom H100，并执行
  不可放宽的最终门控。

Runtime 的明确规则是：当前 event frame 不读新 memory，第一次可见为 event+1；未来
候选和 BASE solver 不读 GT；GT 只在 runtime seal 完成后用于 post-hoc Oracle 选择和
评分；Oracle artifact 明确记录 `runtime_eligible=false`、`oracle_upper_bound_only=true`。

## 3. 阶段结果

### 3.1 Opportunity audit

`outputs/N72R13/intervention_opportunity_audit.json` 对冻结 516 个 interaction、
30,390 个 frame rows 做了只读审计：

| 指标 | 数值 |
|---|---:|
| interaction events | 516 |
| frame rows | 30,390 |
| selection accepted | 26,072 |
| assignment changed | 3,988 |
| metadata-defined effective opportunities | 3,988 |
| runtime future-GT violations | 0 |

按 action 的 assignment-changed 数量为 `ADD_NEW_IDENTITY=213`、
`ATOMIC_ID_SWAP=528`、`AUTHORITATIVE_REASSIGN=2,533`、`RECOVER_IDENTITY=714`。
这只是冻结 artifact 的机会盘点，不代表 3,988 个机会都在正式 32-event Oracle 中
实际重放。

### 3.2 Paired smoke

唯一允许的 RECOVER paired smoke 为：

```text
event = n72r5-pool-n37-dancetrack0015-0002-recover_identity-001
intervention_frame = 3
future rows = event+1 ... event+20
```

修复后 smoke 通过：初始 state 相等、KEEP/APPLY frame axis 完整、当前帧 assignment
不同、分支 state/assignment 不共享、memory/GT flags 正确、event+1 因果边界正确。

首次 smoke 失败没有被覆盖：原因是 checker 把 smoke 检查字段误当成 runtime GT
标志，并把预审计机会位置误当成唯一运行位置；修正 scanner 后用同一输入重跑通过。
证据保存在 `outputs/N72R13/stage_03_status_attempt_01_failure.json` 及 smoke
failure artifacts。

### 3.3 Sequential Oracle

正式 Oracle 对 32 个事件逐一顺序运行，当前 opportunity 先完成无 GT runtime pair
seal，再执行 post-hoc GT decision：只有 H20 target accuracy 和 global PID accuracy
同时严格大于 0 才允许 `APPLY_TIV`。

| action | effective opportunities | APPLY | KEEP |
|---|---:|---:|---:|
| RECOVER_IDENTITY | 82 | 0 | 82 |
| AUTHORITATIVE_REASSIGN | 51 | 0 | 51 |
| ATOMIC_ID_SWAP | 40 | 0 | 40 |
| ADD_NEW_IDENTITY | 3 | 0 | 3 |
| **合计** | **176** | **0** | **176** |

相对于冻结 E0 的 custom post-hoc 汇总如下。它们保留在报告中，但不能解释成 APPLY
收益，因为 Oracle 的 selected branch 始终是 KEEP：

| horizon | target identity-error reduction | global identity-error reduction | target accuracy delta/event mean | global PID accuracy delta/event mean |
|---:|---:|---:|---:|---:|
| H5 | +0.116883 | +0.016913 | +0.137500 | +0.017337 |
| H20 | +0.052202 | +0.006887 | +0.054642 | +0.006357 |
| H50 | +0.013514 | +0.002012 | +0.014045 | +0.001689 |
| H100 | +0.009904 | +0.002063 | +0.009833 | +0.001743 |

同时，custom H20 的 protected accuracy event mean 为 `-0.001894`，H50 为
`-0.000436`；Oracle H20/H50/H100 的 event-level ID-switch improvement mean
均为 `-0.90625`，即平均没有减少 ID switch。这进一步说明不能用 target-only
custom 正值替代全局因果门控。

### 3.4 Official interaction-window TrackEval

导出与官方运行均通过 `288/288`：32 events × 3 variants
(`E0_BASELINE_B0`, frozen `E1B_PCTIS`, `ORACLE_TIV`) × H20/H50/H100。结果是
interaction-window diagnostic，不是完整 DanceTrack benchmark 分数。

#### H100：Oracle TIV 相对 E0

| metric | E0 | ORACLE_TIV | ORACLE−E0 |
|---|---:|---:|---:|
| HOTA | 0.665767 | 0.662006 | -0.003761 |
| AssA | 0.721408 | 0.718851 | -0.002557 |
| IDF1 | 0.788364 | 0.784188 | -0.004176 |
| DetA | 0.616112 | 0.611318 | -0.004794 |
| MOTA | 0.579139 | 0.569195 | -0.009944 |
| IDSW | 177 | 221 | +44 |

H100 没有任何全局 TrackEval metric 高于 E0，且 IDSW 增加 44。因此正式
`official_h100_headroom=false`。

#### 其他窗口

| horizon | HOTA delta | AssA delta | IDF1 delta | DetA delta | MOTA delta | IDSW reduction |
|---:|---:|---:|---:|---:|---:|---:|
| H20 | -0.002382 | -0.006467 | -0.000863 | +0.000866 | -0.002853 | -17 |
| H50 | -0.003016 | -0.002688 | -0.003855 | -0.003278 | -0.007844 | -30 |
| H100 | -0.003761 | -0.002557 | -0.004176 | -0.004794 | -0.009944 | -44 |

## 4. 失败事实与修复

所有失败证据均保留在 `outputs/N72R13/`，没有用 PASS 覆盖：

1. state validator attempt 1：直接执行时缺少项目 root 的 import path，触发
   `ModuleNotFoundError`；添加显式 root path 后 targeted rerun 通过。
2. paired smoke attempt 1：scanner 错把 smoke-only fields 当成 runtime 禁止字段，且
   opportunity pre-audit 位置假设过窄；最小修正后同一 RECOVER smoke 通过。
3. Oracle CLI attempt 1：命令行传入 `--horizon 100`，parser 尚未声明该参数，退出码
   2；补上冻结值校验后继续。
4. Oracle full attempt 1：错误调用 `replay._load_gt`，导致 32 个 post-hoc event
   failures；修正为冻结 legacy loader，保留原批量失败 artifacts。
5. Oracle full attempt 2：错误调用 `replay._score_pair`，再次导致 post-hoc 失败；
   修正为 `replay.legacy._score_pair`，第三次 32/32 通过。

这些是实现/接口调用错误，不改变 protocol、checkpoint、candidate 或 metric 定义。
当前正式 stage 04 的 `failure_count=0`；早期 `*.failure.json` 仍留在 event 目录中，
不能把它们误计入第三次正式运行的失败数，也不能删除。

每次 Python 启动均可见环境已有的 `osr_lib-1.1.0-nspkg.pth` warning：
`AttributeError: 'NoneType' object has no attribute 'loader'`。它发生在 site package
加载阶段，未改变本轮任何输出，未被误报为实验 runtime failure。

## 5. 门控决策

`outputs/N72R13/n72r13_final_gate.json` 的逻辑为：

- H20 custom target/global identity reduction 均为正；
- 但这不是 APPLY effect，因为 `APPLY_TIV=0`；
- H100 official TrackEval 没有任何全局 metric 改善；
- runtime GT violations 为 0，事件完整性为 32/32，TrackEval 为 288/288；
- 因此 `actual_temporal_intervention_headroom=false`。

最终状态：

```text
outputs/N72R13/stage_05_status.json = STOP_NO_TEMPORAL_INTERVENTION_HEADROOM
outputs/N72R13/stage_06_status.json = NOT_AUTHORIZED_NO_HEADROOM
value_model_training_authorized = false
calibration_head_authorized = false
selector_authorized = false
decoder_lora_authorized = false
REQUERY_NEXT = true
```

本轮没有构建语料、没有训练 Temporal Intervention Value Model，也没有训练 calibration
head、selector 或 decoder LoRA。按照冻结规则，在 Oracle 没有全局 H100 headroom 时
继续训练会把一个没有 APPLY 正样本/没有 temporal causal gain 的 Oracle 结果包装成
学习问题，因而不被授权。

## 6. 当前真正瓶颈与下一步

当前最可靠的判断不是“memory 一定无效”，而是：

1. 在正式 32-event sequential Oracle 中，未来价值规则没有找到任何同时改善 target
   与 global PID 的 APPLY opportunity；因此 temporal intervention 的可学习正例为 0。
2. Oracle 相对 E0 的 custom 正值被官方 H100 TrackEval 的全局负值否定，且 E0/Oracle
   的输入轨迹并非同一 intervention branch；表面 custom gain 主要是 baseline/state/
   candidate trajectory 差异，而不是经批准的 temporal action gain。
3. H100 的 AssA、IDF1、HOTA、MOTA 均下降，IDSW 增加，说明把当前状态继续传播到长
   窗口会放大关联误差；下一轮应优先检查 association decision boundary、候选覆盖和
   base/public-ID state 对齐，而不是扩大权重、换 checkpoint 或增加 LoRA。

最小合法下一步为 `REQUERY_NEXT=true`：冻结一个与 E0 完全等价的 sequential KEEP
control，再设计单独的 association-interface probe，或取得 provenance-complete 的
真实 human event tape。不能用 N72R13 的 `simulated_from_gt` 改名为真实人工证据，不能
把 Oracle-only post-hoc 规则部署为 runtime。

## 7. ICLR 2027 时间窗口

按项目冻结的硬约束（当前日期 2026-09-10，Asia/Shanghai）：

| 日期 | 距今 | 交付含义 |
|---|---:|---|
| 2026-09-18 AoE | 8 天 | 摘要截止；只能提交已经有可复现实证的主结论 |
| 2026-09-25 AoE | 15 天 | 全文截止；若没有真实 human tape 或严格 future-effect，应如实写负结果/方法诊断 |

在这段窗口内不建议进行无诊断依据的大规模训练。若要继续，优先级是：

1. 一次 association-interface / candidate-base alignment 的冻结 probe；
2. 或导入外部 provenance-complete real-human event tape 并先做 schema/完整性审计；
3. 只有未来确实出现有效 APPLY headroom，才可按预注册 12 train / 6 validation
   序列拆分训练 value model。

## 8. 机器可读证据索引

- [stage 00 interface audit](../outputs/N72R13/stage_00_status.json)
- [stage 01 state clone](../outputs/N72R13/stage_01_status.json)
- [stage 02 opportunity audit](../outputs/N72R13/stage_02_status.json)
- [stage 03 paired smoke](../outputs/N72R13/stage_03_status.json)
- [stage 04 Oracle](../outputs/N72R13/stage_04_status.json)
- [opportunity audit](../outputs/N72R13/intervention_opportunity_audit.json)
- [Oracle metrics](../outputs/N72R13/temporal_oracle/temporal_oracle_metrics.json)
- [TrackEval export](../outputs/N72R13/trackeval/export_manifest.json)
- [TrackEval run](../outputs/N72R13/trackeval/trackeval_run_manifest.json)
- [final gate](../outputs/N72R13/n72r13_final_gate.json)
- [stage 06 no-training decision](../outputs/N72R13/stage_06_status.json)

这些 outputs 在当前仓库的 ignore 策略下是本机实验产物；本报告、研究脚本和
`research_log.md` 会提交到当前 GitHub 分支。历史 N36–N72R12 报告与证据均未改写。
