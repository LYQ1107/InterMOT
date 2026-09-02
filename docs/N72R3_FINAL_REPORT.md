# N72R3 最终报告：跨独立 SAM session 的 Persistent Public Identity

生成时间：2026-09-02T09:57:26.814726+00:00（Asia/Shanghai 任务环境）  
最终 research gate：**FAIL_FUTURE_EFFECT**。本报告不是把失败包装成成功：结构性 identity closure 已通过，但当前候选流上的 future identity effect 没有严格确认。

## 1. 一页结论

- `public_id` 最终属于 sequence-lifetime 的 `PersistentIdentityRecord`，不属于当前 SAM candidate、raw SAM ID 或 association-local numeric state ID。`mot_track_id == public_id`，lineage、appearance/motion/lost state 与一个外层 TrackManager 由持久 runtime 管理。
- 用户给出的 #1007 合约在 CPU toy stress 中通过：Session A candidate 17 绑定后，session boundary 只清理 session-local candidate/SAM binding；Session B 的 NONE 记录为 LOST，identity 未删除；新 raw SAM candidate 8 再绑定后仍返回 public_id 1007、mot_track_id 1007、lineage 不变。这个 stress 不是科学效果结果，但验证了目标状态机。
- 2-window、6-window exact-public structural baseline 通过：public restore coverage=1、renumber=0、lineage loss=0、assignment artifacts complete、runtime GT leakage=0。
- Stage 16/17 对 6 个冻结 `simulated_from_gt` 事件真实调用官方 SAM3 当前帧纠正，并把真实当前图像 ROI 的 512-D OSNet feature 写入 target public ID；6/6 correction、6/6 finite memory writes，event frame 不读新 memory，首次可见为 event+1。它不是 real human tape。
- Stage 20–22 只找到 6 个 eligible events / 6 个独立序列（目标 40 / 20），没有重复事件凑数。M1/M2 改变目标 score 但没有 assignment crossing；M3/M4 出现少量 crossing，H20 utility 约 0.000749，但 sequence-cluster CI 下界为 0，因此严格 gate 失败。
- 结论：主要瓶颈是候选召回与 association decision boundary 的组合，而不是 persistent public-ID ownership。当前不授权 calibration head、selector、decoder LoRA 或 production promotion。

## 2. 用户要求的持久身份语义

```text
Persistent Public Identity #1007
├── public_id = 1007       永久稳定
├── lineage_id              永久稳定
├── TrackManager track      永久稳定
├── association state      显式绑定但不拥有 public authority
├── appearance memory      持久
├── motion state            持久
├── ACTIVE / LOST           可变化
│
├── Session A candidate 17 → assignment
├── Session boundary       → 清空 candidate binding，identity 仍存在
├── Session B: NONE         → identity = LOST
└── Session B candidate 8  → association/rebind → public_id 仍为 1007
```

Stage 01 的 probe 同时证明旧 N72R2 candidate-first authority 有缺陷：同一个 association state 可以被旧 bridge 接受为两个 public IDs。N72R3 用 immutable state→lineage→persistent public binding、外层 birth decision 和显式 LOST/NONE 取代它；IoU/appearance 只能是 association evidence，不能创建 exact authority。

## 3. 分阶段状态

| 阶段 | 结果 | 科学含义 |
|---|---|---|
| 00–01 | protocol frozen；旧 authority circularity confirmed | 输入与边界冻结，旧语义缺陷保留 |
| 02–08 | structural PASS | persistent identity、authority bridge、session boundary、snapshot 通过；不是效果结果 |
| 09–11 | structural PASS | 1/2/6-window public continuity 通过；candidate recall 另行诊断 |
| 12–15 | audit/oracle/atomic transaction PASS | GT 审计修正、模拟 oracle 隔离、故障回滚通过 |
| 16–17 | official correction + 512-D memory PASS | 当前帧事务与 t+1 因果边界真实执行 |
| 18 | exact-public baseline PASS | 6/6 eligible windows 的 public continuity 通过 |
| 19 | candidate recall PARTIAL | 性能诊断，不是 identity continuity gate |
| 20 | runtime execution PASS，eligible shortfall | 6 events×6 variants×H20/H50/H100 已落盘 |
| 21 | target-scoped audit PASS | 380,322 个 non-target score cells bitwise unchanged；global Hungarian 保留 |
| 22 + 5 rounds | execution complete，FAIL_FUTURE_EFFECT | 没有严格正 future utility，停止下游学习授权 |

## 4. 候选召回诊断

Stage 19 使用 6 个事件、6 个独立序列，GT 只在 runtime artifact 冻结后 posthoc 读取。候选召回不是结构性 public-ID gate。

| horizon | 全部候选 recall | AUTHORITATIVE_REASSIGN | RECOVER_IDENTITY | target GT present / candidate absent |
|---:|---:|---:|---:|---:|
| H20 | 0.692308 | 未定义 | 未定义 | 36 |
| H50 | 0.683502 | 未定义 | 未定义 | 94 |
| H100 | 0.710884 | 未定义 | 未定义 | 170 |

RECOVER 的 H20 recall 只有 0.35；因此不存在 candidate 的帧不能由 appearance-only score 修复。该结果支持后续 candidate recovery/detection 分支，但不能把 candidate absent 写成 public identity 被删除。

## 5. Stage 20–22 paired replay

输入是冻结 Candidate V2 stream；事件为 6 个、独立序列 6 个，低于预注册目标 40/20。Stage 14 action 分布是 {"AUTHORITATIVE_REASSIGN": 4, "RECOVER_IDENTITY": 2}；没有 ATOMIC_ID_SWAP、ADD_NEW_IDENTITY 的 eligible 事件，未跨序列拼接或复制事件。所有交互明确标记 `simulated_from_gt`，不是历史真实点击。

变体定义：`NO_INTERVENTION`；`M0` 当前帧纠正 only；`M1` human EMA prototype；`M2` prototype + positive anchors；`M3` 加 competitor negatives；`M4` 加固定 reliability/age admission。未来 score 只改变 target public-ID row；最终 assignment 的其他 public ID 被动变化时记作 `solver_coupled_collateral`。

| variant | horizon | identity utility | 95% CI（sequence cluster） | assignment changes | correct | incorrect | neutral | wrong reassociation | protected regression |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| NO_INTERVENTION | H20 | 0 | [0, 0] | 0 | 0 | 0 | 0 | 66 | 0 |
| NO_INTERVENTION | H50 | 0 | [0, 0] | 0 | 0 | 0 | 0 | 148 | 0 |
| NO_INTERVENTION | H100 | 0 | [0, 0] | 0 | 0 | 0 | 0 | 301 | 0 |
| M0_CURRENT_FRAME_CORRECTION_ONLY | H20 | 0 | [0, 0] | 0 | 0 | 0 | 0 | 66 | 0 |
| M0_CURRENT_FRAME_CORRECTION_ONLY | H50 | 0 | [0, 0] | 0 | 0 | 0 | 0 | 148 | 0 |
| M0_CURRENT_FRAME_CORRECTION_ONLY | H100 | 0 | [0, 0] | 0 | 0 | 0 | 0 | 301 | 0 |
| M1_HUMAN_EMA_PROTOTYPE | H20 | 0 | [0, 0] | 0 | 0 | 0 | 0 | 66 | 0 |
| M1_HUMAN_EMA_PROTOTYPE | H50 | 0 | [0, 0] | 0 | 0 | 0 | 0 | 148 | 0 |
| M1_HUMAN_EMA_PROTOTYPE | H100 | 0 | [0, 0] | 0 | 0 | 0 | 0 | 301 | 0 |
| M2_POSITIVE_HUMAN_ANCHORS | H20 | 0 | [0, 0] | 0 | 0 | 0 | 0 | 66 | 0 |
| M2_POSITIVE_HUMAN_ANCHORS | H50 | 0 | [0, 0] | 0 | 0 | 0 | 0 | 148 | 0 |
| M2_POSITIVE_HUMAN_ANCHORS | H100 | 0 | [0, 0] | 0 | 0 | 0 | 0 | 301 | 0 |
| M3_NEGATIVE_COMPETITOR_BANK | H20 | 0.000748657 | [0, 0.0437964] | 20 | 15 | 0 | 5 | 54 | 0 |
| M3_NEGATIVE_COMPETITOR_BANK | H50 | 0.000271355 | [0, 0.0402962] | 50 | 45 | 0 | 5 | 125 | 0 |
| M3_NEGATIVE_COMPETITOR_BANK | H100 | 7.2041e-05 | [0, 0.0211801] | 100 | 50 | 0 | 50 | 296 | 0 |
| M4_RELIABILITY_AGE_ADMISSION | H20 | 0.000748657 | [0, 0.0437964] | 20 | 15 | 0 | 5 | 54 | 0 |
| M4_RELIABILITY_AGE_ADMISSION | H50 | 0.000271355 | [0, 0.0402962] | 50 | 45 | 0 | 5 | 125 | 0 |
| M4_RELIABILITY_AGE_ADMISSION | H100 | 7.2041e-05 | [0, 0.0211801] | 100 | 50 | 0 | 50 | 296 | 0 |

上表列顺序是：utility、CI、assignment changes、correct、incorrect、neutral、wrong reassociation、protected regression。

主 gate 预注册为 `M2_POSITIVE_HUMAN_ANCHORS` @ H20。M2 H20 utility=0，CI=0..0；correct assignment changes=0，所以 gate 失败。M3/M4 H20 有 15 个 correct、0 个 incorrect、5 个 neutral changes，但 CI 下界仍是 0，不能包装为确认。

## 6. 机制诊断与五轮收口

- Round 1：候选召回上限。确认 absent candidate 帧是 appearance-only association 的硬上限，尤其 recover 分支；没有添加 GT candidate 或重写候选流。
- Round 2：逐帧 global-assignment residual。典型 required target-row residual 中位数约 7.57；M1/M2/M3/M4 在 cheapest alternative 列的 delta 中位数约 0.181/0.685/0.150/0.145。只比较同一替代列，避免把不同列的 delta 混合。
- Round 3：仅 target row 的假设性 boundary probe。对每个可检查未来帧加入恰好超过 residual 的 1e-6，600/600 per variant 能让 constrained alternative crossing；这是说明 solver 接口可穿越的诊断，不是有效 learned score，也没有改 production。
- Round 4：固定 age/admission。M3 memory read/admitted=600/600；M4 read/admitted=600/480，120 帧在固定 age>80 后拒绝；不按未来指标调 admission。H20/H50 两者结果相同，H100 未产生可靠新增收益。
- Round 5：统计功效与 protected-ID。2000 次、独立序列 cluster bootstrap 正确执行，但只有 6 clusters，主 CI 下界为 0；protected regression 为 0。下一步需要更多独立 eligible 事件或真实人工 tape，不能复制当前 6 个事件。

Round 2 的第一次实现曾错误地跨列比较 delta，已保存 `outputs/N72R3/attempts/mechanism_rounds_semantic_audit_attempt1.json`，修复后在独立 `mechanism_rounds/attempt2/` 重跑。机制轮次脚本第一次语法失败也保存了退出码 1 与 traceback；这些不是科学 PASS。

## 7. 因果、映射和保护审计

- Stage 16 correction success rate=1；Stage 17 finite 512-D writes=6，event-frame hidden=是，first visible event+1=是。
- Stage 18 public restore coverage=1，renumber=0，lineage loss=0，runtime GT leakage=0。
- Stage 21 candidate stream 在各 variant 完全共享；global Hungarian 保留；target-scoped non-target score cells=380322，row failures=0。
- 每个 replay artifact 对每帧均保存 candidate UID/native/adapter mapping、base/appearance/fused score matrix、assignment、memory read/admission 和 public state axis；GT 字段没有进入 runtime artifact。

## 8. 失败事实与修复保留

| 证据 | 首个 actionable root cause / 处理 |
|---|---|
| `stage01_authority_audit_attempt1_failure.json` | 独立脚本缺少 worktree `sys.path`，`ModuleNotFoundError`；修复入口后重跑 |
| `stage01_authority_audit_attempt2_pre_repair.json` | probe 暴露旧 bridge 可让同一 state 接受两个 public IDs；保留为 root-cause 事实，改用 probe-based gate |
| `stage09_11_failure_attempt1.json` | aggregation 读取不存在的 `runtime_final_audit`；兼容真实 `runtime_audit` 后 targeted rerun |
| `stage13_oracle_pytest_attempt1_failure.json` | toy oracle 字段名不一致（`other_public` vs `other_public_id`）；修复后 11 tests pass |
| `stage15_transaction_pytest_attempt1_failure.json` | rollback restore 替换 identity 对象，外部引用失效；改为 in-place restore 后 10 tests pass |
| `stage16` smoke attempts 1–5 | 依次保留 checkpoint capacity mismatch、adapter 未注册 observation、错误绑定未先释放、`wrong_record` 未返回、causal guard 调用方式错误；各自最小修复后 attempt6 pass |
| `stage16_mapping_test_fixture_attempt1/2_failure.json` | toy fixture 触发真实 checkpoint guard；最终 fixture 显式禁用模型加载，仅用于 adapter mapping test，10/10 pass |
| `mechanism_rounds_py_compile_attempt1_failure.json` | `round3_boundary_probe` 类型注解缺 `]`，退出码1；修复后编译/执行 pass |
| `mechanism_rounds_semantic_audit_attempt1.json` | Round2 跨列比较造成潜在 boundary reachability 误报；修复并以 attempt2 独立输出重跑 |

失败 artifact 均位于 `outputs/N72R3/attempts/` 或对应 attempt 目录，没有删除、覆盖或将失败改写为 PASS。

## 9. 隔离、输入哈希与资源边界

- 所有本轮新代码和输出都在独立 N72R3 worktree / `outputs/N72R3/`；N36–N72R2 历史输出只读，N72R3 protection manifest 的历史输入 hash 全部匹配。
- checkpoint SHA-256：`0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`；checkpoint、candidate definition、Hungarian solver、metric/bootstrap 定义未改变。
- 官方 SAM3 third-party 目录未修改；没有创建 real human tape；没有把 `simulated_from_gt` 改名为 `real_human`。
- Stage 16–17 最多使用四张 GPU，每卡单独事件进程；Stage 20–22 与五轮机制诊断为 CPU-only。

## 10. 最终授权与下一步

当前明确禁止：calibration head、selector、decoder LoRA、共享 checkpoint 更新、production identity promotion。N72R3 已经完成结构链与五轮 mechanism evidence，但没有严格 future-effect confirmation。

最小下一步是：在不改变冻结协议、candidate definition、checkpoint 或 Hungarian 的前提下，采集更多独立 eligible current-frame events，或接入外部 provenance-complete real-human tape；优先解决 recover candidate recall，再重新进行相同 paired replay。不能复制当前 6 个事件、不能用 synthetic 事件冒充真实点击，也不能用一次性 boundary probe 代替真实收益。

若沿用 ICLR 2027 时间约束：摘要截止 2026-09-18 AoE、全文截止 2026-09-25 AoE；截至 2026-09-02，优先整理结构性贡献与诚实的负 future-effect 结果，不应在缺少证据时赶工训练下游模块。

## 11. 机器可读证据

- `outputs/N72R3/n72r3_final_gate.json`：最终 gate 与授权状态。
- `outputs/N72R3/effect_replay/attempt1/ccam_paired_replay_results.json`：Stage 20–22 完整 paired replay 与 posthoc 指标。
- `outputs/N72R3/stage_21_status.json`：target-scoped bitwise audit。
- `outputs/N72R3/mechanism_rounds/attempt2/mechanism_rounds_summary.json`：五轮根因诊断。
- `outputs/N72R3/stage_18_status.json`、`stage_19_status.json`、`stage_16_status.json`、`stage_17_status.json`：结构、召回、官方纠正与 memory 证据。

**最终不是 PASS：`research_gate=FAIL_FUTURE_EFFECT`; `production_authorized=false`; `real_human_tape=false`。**
