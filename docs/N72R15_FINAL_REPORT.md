# N72R15 Repaired Persistent Identity Association

**日期：** 2026-09-10（Asia/Shanghai）
**项目：** InterMOT / SAM3_InterMOT
**实验状态：** `FAIL_FUTURE_EFFECT`
**结构状态：** `PASS`（formal replay、post-hoc audits、interaction-window TrackEval 均完成）
**科研结论：** 当前修复已经接入正确的持久公共身份状态与全局关联求解器，但没有在冻结数据和冻结评价上产生足够的未来身份收益；不授权下游训练、消融或生产推广。

## 1. 最终结论

N72R15 修复并验证了三个结构性问题：

1. `TARGET_SESSION_CURRENT_RAW` 不再进入未来 solver candidate pool；未来 solver 只使用 `MAIN_B0_CANDIDATE`。
2. 持久状态证据改为 row/column-centered relative edge，并采用固定 `tanh` 和 row-maximum-preserving 融合；没有改变 checkpoint、候选定义、Hungarian 求解器或评价定义。
3. treatment 不再无条件把当前错误 assignment 写回可信机器状态。`E1I` 只保留人工目标状态，`E1J` 只有 base/treatment public-ID consensus 时才允许机器 box/velocity/appearance 更新；disagreement 不写 trusted state。

上述结构审计全部通过，但效果门禁失败：

- `E1J_TRUSTED_GLOBAL_RELATIVE_STATE - E0_BASELINE_B0` 在 H100 的 TrackEval HOTA/AssA/IDF1 分别为 `-0.00002760/-0.00003097/-0.00002561`，不是预注册的 `≥0.003/0.005/0.005`。
- H100 official interaction-window IDSW 为 `177` 对 `177`，没有变差，但也没有形成正向效果。
- H20 的 target identity-error reduction 为 `0`，不是严格大于 `0`；sequence-cluster bootstrap 的 H100 global reduction CI 为 `[0, 0]`。
- 因此 `STRONG_POSITIVE_STATUS=false`、`ablations_authorized=false`、`downstream_training_authorized=false`。本轮不运行 frequency/strength ablation、calibration head、selector 或 decoder LoRA。

这不是“代码没有执行”的结论：状态 edge 在未来帧发生了大量变化，然而只有极少数 assignment 穿过 solver decision boundary，且没有相对 B0 的正确 target assignment change。

## 2. 冻结协议与实验边界

本轮复用 N72R14/N72R9 的冻结 32-event、18-independent-sequence protocol，事件 action 计数为：

| action | events |
|---|---:|
| `ADD_NEW_IDENTITY` | 4 |
| `AUTHORITATIVE_REASSIGN` | 14 |
| `ATOMIC_ID_SWAP` | 3 |
| `RECOVER_IDENTITY` | 11 |

每个事件使用 event frame 加未来 100 帧，三个正式变体为：

- `E0_BASELINE_B0`：MAIN B0 candidate pool，无持久 state edge。
- `E1I_HUMAN_RELATIVE_STATE`：只写人工目标状态，未来不做 trusted machine update。
- `E1J_TRUSTED_GLOBAL_RELATIVE_STATE`：所有 public trusted state 参与 relative edge；只有 base/treatment public-ID consensus 才允许机器状态更新。

固定状态 edge 为：

```text
raw = appearance + 1.0 * motion + 0.5 * native_continuity - 0.1 * gap
relative = 0.5 * (row_centered(raw) + column_centered(raw))
relative_delta = tanh(relative)
fused = base + relative_delta + row_shift
```

`row_shift` 被定义为保持每一行最大值不变的平移量；9600 个 future rows 的 row-max preservation 检查全部通过。求解仍为唯一的 `solve_effect_assignment`，没有增加第二个 Hungarian 或 greedy solver。

所有交互均为 `simulated_from_gt`，不是历史真实人工点击；`not_real_human_evidence=true`。GT 只用于离线 post-hoc scoring，runtime `runtime_future_gt_used=false`、`runtime_gt_read=false`。

## 3. 实现内容

新增的核心实现和审计脚本：

- `sam3_intermot/association/trusted_persistent_public_state.py`
- `sam3_intermot/association/relative_persistent_state_edge.py`
- `scripts/n72r15_run_repaired_persistent_state_formal.py`
- `scripts/n72r15_interface_audit.py`
- `scripts/n72r15_aggregate_metrics.py`
- `scripts/n72r15_candidate_coverage_posthoc.py`
- `scripts/n72r15_state_edge_component_diagnosis.py`
- `scripts/n72r15_state_propagation_audit.py`
- `scripts/n72r15_integrity_audits.py`
- `scripts/n72r15_export_window_trackeval.py`
- `scripts/n72r15_run_window_trackeval.py`
- `scripts/n72r15_finalize_gate.py`

没有修改 `third_party/sam3`、TrackEval、N36--N72R14 历史输出、共享 checkpoint 或 MOT/OVMOT 目录。

## 4. 结构完整性结果

最终 formal manifest 为 `outputs/N72R15/formal_attempt_04/formal_manifest.json`：

| 检查项 | 结果 |
|---|---:|
| unique events | 32 |
| independent sequences | 18 |
| variants/event | 3 |
| runtime rows | 9696 = 32 × 3 × 101 |
| event-frame rows | 96 |
| future rows | 9600 |
| candidate UIDs checked | 72,864 |
| duplicate runtime keys | 0 |
| missing runtime rows | 0 |
| event-frame memory read | 0（全部为 false） |
| first memory-visible frame | event + 1 |
| target-session candidate in solver | 0 |
| runtime GT violations | 0 |
| public-ID inference violations | 0 |
| row-max failures | 0 |

Post-hoc audit 状态全部通过：

- `PASS_N72R15_STATE_PROPAGATION`
- `PASS_N72R15_CANDIDATE_COVERAGE_POSTHOC`
- `PASS_N72R15_STATE_EDGE_COMPONENT_DIAGNOSIS`
- `PASS_N72R15_SELF_REINFORCEMENT_AUDIT`
- `PASS_N72R15_ASSIGNMENT_CARDINALITY_AUDIT`

## 5. TrackEval interaction-window 结果

TrackEval 使用固定 commit `12c8791b303e0a0b50f753af204249e622d0281a`，三个 horizon 各有 96 条 per-event variant 记录，总计 `288/288`；duplicate 和 missing 都是 0。它们是 interaction-window diagnostic，不是完整 DanceTrack 官方 benchmark 分数，`official_dancetrack_benchmark_score=false`。

### 5.1 H100 pooled first screen

| variant | HOTA | AssA | IDF1 | IDSW | DetA | MOTA |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.66576746 | 0.72140773 | 0.78836437 | 177 | 0.61611196 | 0.57913935 |
| E1I | 0.66577311 | 0.72144438 | 0.78838104 | 176 | 0.61608988 | 0.57913935 |
| E1J | 0.66573986 | 0.72137677 | 0.78833876 | 177 | 0.61608730 | 0.57909592 |

相对 B0 的 H100 delta：

| comparison | ΔHOTA | ΔAssA | ΔIDF1 | ΔIDSW | ΔDetA | ΔMOTA |
|---|---:|---:|---:|---:|---:|---:|
| E1I − B0 | +0.00000565 | +0.00003665 | +0.00001667 | −1 | −0.00002208 | 0 |
| E1J − B0 | −0.00002760 | −0.00003097 | −0.00002561 | 0 | −0.00002466 | −0.00004342 |

H20/H50/H100 的 TrackEval runner 均为 PASS；没有把 H20 或单个事件的局部变化包装成全局成功。

### 5.2 Causal post-hoc metrics

下表的 CI 是以独立 sequence 为 cluster、seed `7215`、2000 repetitions 的 global identity-error reduction CI；target reduction 单独列出。正值才表示 treatment 相对左侧 reference 减少 identity error。

| comparison | horizon | global mean | global 95% CI | target mean | target missing reduction | target IoU delta |
|---|---:|---:|---:|---:|---:|---:|
| E1I − B0 | H20 | −0.00020695 | [−0.00110375, 0] | −0.00195313 | −0.03125 | −0.00133000 |
| E1I − B0 | H50 | −0.00008705 | [−0.00046425, 0] | −0.00080128 | −0.03125 | −0.00054564 |
| E1I − B0 | H100 | −0.00004246 | [−0.00022645, 0] | −0.00035112 | −0.03125 | −0.00023910 |
| E1J − B0 | H20 | 0 | [0, 0] | 0 | −0.03125 | −0.00055183 |
| E1J − B0 | H50 | 0 | [0, 0] | 0 | −0.03125 | −0.00022993 |
| E1J − B0 | H100 | 0 | [0, 0] | 0 | −0.03125 | −0.00011262 |
| E1J − E1I | H20 | +0.00020695 | [0, +0.00110375] | +0.00195313 | 0 | +0.00077817 |
| E1J − E1I | H50 | +0.00008705 | [0, +0.00046425] | +0.00080128 | 0 | +0.00031571 |
| E1J − E1I | H100 | +0.00004246 | [0, +0.00022645] | +0.00035112 | 0 | +0.00012648 |

E1J 相对 E1I 的微小正值不能替代预注册的 E1J 相对 B0 gate；E1I/E1J 相对 B0 都没有正向 future-effect 证据。

### 5.3 Action 分解（H100，E1J − B0）

| action | events | independent sequences | mean global reduction | 95% CI |
|---|---:|---:|---:|---:|
| `ADD_NEW_IDENTITY` | 4 | 4 | 0 | [0, 0] |
| `AUTHORITATIVE_REASSIGN` | 14 | 10 | 0 | [0, 0] |
| `ATOMIC_ID_SWAP` | 3 | 3 | 0 | [0, 0] |
| `RECOVER_IDENTITY` | 11 | 8 | 0 | [0, 0] |

没有 action 类在严格的 E1J-vs-B0 比较中提供正向 cluster-level 证据。

## 6. 机制诊断

### 6.1 State edge 已经传播，但几乎没有改变 solver decision

| variant | state-changed future rows | next-edge changed / checked | solver-changed frames | correct assignment changes | cardinality changes |
|---|---:|---:|---:|---:|---:|
| B0 | 0 | 0 / 0 | 0 | 0 | 0 |
| E1I | 3200 | 3168 / 3168 | 1 | 0 | 1 |
| E1J | 3200 | 3168 / 3168 | 2 | 0 | 1 |

E1J H100 相对 B0 的候选/assignment 统计为：`score_changed_frame_count=3200`、`score_changed_cell_count=175498`、target assignment change `1`、correct change `0`，另有 1 个 neutral/cardinality-related change。E1I 相对 B0 有 1 个 assignment change，且被 post-hoc 归为 incorrect；这些变化远少于 3200 个发生 score/state 改变的 future rows。

### 6.2 State update 安全性

`self_reinforcement_audit` 通过：

- E1I/E1J 的 `changed_and_machine_write_rows=0`；
- E1J disagreement rows 为 2，但 `disagreement_trusted_write_rows=0`；
- 没有检测到错误 assignment 与 machine write 同时发生的自我强化路径。

`assignment_cardinality_audit` 也通过：处理过程没有产生 validator violation；实际观察到 E1I 和 E1J 各 1 个相对 B0 的 cardinality-change row，该事实被保留，没有被改写成“无变化”。

### 6.3 Component evidence 指向 appearance 输入/候选质量，而非单纯 state-edge 连接错误

H100 的 positive-vs-negative edge diagnostic AUC：

| component | E1I | E1J |
|---|---:|---:|
| appearance | 0.5000 | 0.5000 |
| motion | 0.5446 | 0.9596 |
| native continuity | 0.5000 | 0.9114 |
| gap | 0.5027 | 0.4996 |
| raw edge | 0.5767 | 0.9634 |
| relative delta | 0.6107 | 0.9649 |

本轮 artifact 中 appearance、appearance-positive、appearance-prototype、appearance-negative 的分布为零或不可区分（appearance AUC 为 0.5），而 E1J 的可分性主要来自 motion/native-continuity。这个结果支持以下诊断：

- persistent-state edge 确实到达 solver，不能把失败归因于“模块完全未调用”；
- 当前 appearance evidence 在这批 sealed candidate rows 中没有提供可辨识的 target-vs-competitor 信号，或者候选 feature provenance/availability 不足；
- 即使 relative edge 改变，base/geometry/continuity 主导的 assignment margin 仍很少被穿过；
- 因此继续盲目放大 state-edge strength、换 checkpoint 或训练 LoRA 没有当前证据支持。

### 6.4 Candidate coverage 限制

H100 target-visible frame 共 3130：

| source | covered frames | coverage |
|---|---:|---:|
| MAIN B0 | 2685 | 0.857827 |
| target-session diagnostic source | 1065 | 0.340256 |
| target-session-only rescue | 112 | 0.035783 |
| neither source | 333 | 0.106390 |

target-session source 只作为 post-hoc coverage diagnosis，绝没有重新进入未来 solver。333 个 target-visible frames 两类来源都没有覆盖，说明 candidate/base-state 质量仍是可见的上游限制；不能把这类缺失包装成 memory 失败或用未来 GT 补齐。

## 7. 失败事实、根因与修复

所有以下事实均保留在 `outputs/N72R15/`，没有删除或覆盖首次失败：

1. 初始 smoke controller 不识别 `--horizon`；保留 `smoke_controller_failure_attempt_01.json`，补充 parser 后用同一 frozen event 重跑通过。
2. 初始 formal controller 不识别 `--state-edge-scale`；保留 `formal_controller_failure_attempt_01.json`，补充受支持参数并固定 scale=1.0。
3. 第一轮 formal 错把 raw N72R9 stream 当作 control reference；事件 `dancetrack0027:222` 的 raw axis 为 8，而历史 corrected N72R14 E0 为 7。修复为 N72R14 corrected E0 reference 后，targeted regression 和 formal attempt 通过。
4. 第二轮 formal 在 consensus lookup 对 `public_id=null` 做 `int(None)`；修复为过滤显式空 public ID，并保留对应 traceback/失败 artifact。
5. 初始 component audit 对 `target_public_id` 假定错误；随后还发现 exact positive×negative AUC 的 O(N²) 算法对 2,840,460 rows 不可行，进程以 exit 130 保留 traceback。改为等价的 tie-aware sorted Mann–Whitney AUC（O(N log N)），并用 brute-force 小样本等价性测试验证。
6. integrity audit 初次没有处理 event-frame 的 `score_audit=null`，并错误地把 event-frame 的 null candidate pool 当作违规；修复为只对 future rows 验证 MAIN_B0 pool，最终 violation 为 0。
7. causal aggregation 初次使用变化的 bootstrap seed；保留 `causal_metrics_failure_attempt_01.json`，重新生成固定 seed=7215、repetitions=2000 的 `causal_metrics_seed7215.json`。
8. TrackEval wrapper 初次直接 import 失败（`ModuleNotFoundError: scripts`）；保留 `export_failure_attempt_01.json`，仅补充 project-root import path，之后 export/run 均通过。TrackEval checkout 没有改动。
9. finalizer 的两次收口自检分别错误假定 assignment 必须带 target-session 字段、以及把 assignment axis 与 score audit axis 混比，产生 25,600 和 6,400 条脚本级 integrity failure；保留 `finalizer_failure_attempt_01.json`、`finalizer_failure_attempt_02.json`，修复后最终 `integrity_failures=0`。这两次不是模型执行失败，也没有被写成 PASS。

正式执行的最终 manifest 是 `formal_attempt_04`；没有把任何上述失败样本删除、跳过或重新标记为成功。

## 8. 最终机器门禁

最终机器文件：

- `outputs/N72R15/STRONG_POSITIVE_STATUS.json`
- `outputs/N72R15/n72r15_final_gate.json`
- `outputs/N72R15/stage_11_status.json`

最终状态为 `FAIL_FUTURE_EFFECT`，但结构完整性为 PASS。严格条件如下：

| criterion | result |
|---|---|
| all post-hoc audits pass | true |
| formal integrity pass | true |
| H100 E1J−B0 ΔHOTA ≥ 0.003 | false |
| H100 E1J−B0 ΔAssA ≥ 0.005 | false |
| H100 E1J−B0 ΔIDF1 ≥ 0.005 | false |
| H100 E1J IDSW ≤ B0 | true（177 ≤ 177） |
| H20 target identity reduction > 0 | false（0） |
| H100 sequence-cluster CI lower > 0 | false（0） |
| protected accuracy not worse | true（delta 0） |

所以本轮不能授权任何下游学习或生产集成。Oracle、selector、calibration、decoder LoRA 和额外强度/频率消融均未启动。

## 9. 存储清理审计（仅当前 data1 InterMOT 项目）

用户要求清理 `data1` 存储后，我先确认 `/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT` 不是 symlink，且没有运行中的 data1 项目进程；当前 N72R15 执行 worktree 位于独立的 data2 worktree。清理只作用于上述 data1 项目树的精确目标，没有触及其他项目、根目录或共享目录。

已删除的仅是可再生/安装残留：

- `/data1/.../SAM3_InterMOT/envs/sam3_intermot/wheels/` 下 4 个 installer wheel，合计 `1,712,566,563` bytes；运行环境包本体未删。
- 当前项目树内 1,161 个精确 `__pycache__` 目录，约 `197,867,529` bytes；这些会由 Python 重新生成。
- 当前项目根下的精确 `.pytest_cache`。

释放空间为 `1,937,313,792` bytes（约 1.94 GB）；`df -h /data1` 的可用空间从约 8.2 GB 增至约 17.7 GB。为避免误删，以下大对象经过引用审计后保留：

- `outputs/n16/enc_cache`（约 2.0 GB）：仍被历史脚本/文档引用；
- `outputs/n20/gfn_cache_r0`（约 2.1 GB）：仍被历史脚本/文档引用；
- `checkpoints/sam3.1_mirror/sam3.1_multiplex.pt`（约 3.5 GB）：共享 checkpoint，未删除；
- N36--N72R15 输出、失败 artifact、日志、候选数据和报告：均未删除。

没有通过模糊 glob 删除数据，也没有删除 data1 之外的任何路径。

## 10. 可复现性与哈希

机器门禁中保存了完整输入与代码 hash。关键 hash 如下：

- formal manifest：`22f736ae93c8c499bea411ab008f7c87f963c0521a4cc7e77b30a7daa0250bdd`
- protocol：`e012ecc3bd64fec4409fccd57d920f3690b8b1c4b5a2dd557e03e2bfb43ef0e9`
- fixed causal metrics：`819ac7d8d0c05ba1d567f0fb1f0731c9cf7915bad9a493276a51410fe13a6ac0`
- TrackEval run manifest：`afec47a457b1ca2dddb1e10c2d50a2d0644314ec2f0cbef44532518e77de5978`
- TrackEval commit：`12c8791b303e0a0b50f753af204249e622d0281a`
- finalizer：记录于 `outputs/N72R15/n72r15_final_gate.json` 的 `source_code_hashes`。

运行使用 `/home/lwr/anaconda3/envs/intermot/bin/python`。该环境每次启动会输出已有的 `osr_lib-1.1.0-nspkg.pth` `AttributeError: 'NoneType' object has no attribute 'loader'` warning；它没有影响本轮 Python 进程退出码、formal rows 或 TrackEval 结果，未被错误归类为实验失败。

## 11. 下一步建议

最小、可证伪的下一步不是继续扩大 state-edge 权重或训练 LoRA，而是先做一个 provenance-complete 的 appearance feature/候选质量诊断：确认人工目标区域和每个竞争候选是否有真实、有限、非零且不同的 appearance embedding，并在不读取未来 GT 的 runtime sidecar 中保存来源、hash、可用性和 cosine。只有 appearance evidence 在 target-vs-competitor 上确实有方向性、且候选 coverage/assignment boundary 足够，才值得重新冻结一个 association-interface probe。

如果需要真实交互证据，必须采集带 annotator/session/timestamp、event frame、public ID、原始 box/click/mask 和不可丢失 provenance 的 real-human tape；当前 32 个事件不能改名为 real human。没有新的证据前，N72R15 的正确科研决策是保留这个负结果，停止下游模型扩张，并继续使用 `FAIL_FUTURE_EFFECT` 作为唯一门禁结论。
