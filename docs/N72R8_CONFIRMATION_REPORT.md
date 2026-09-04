# InterMOT N72R8 Confirmation Report

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
| N72R7-12 mechanism freeze | `PASS_RESEARCH_ONLY_MECHANISM_FROZEN`，research-only |
| N72R8/N72R7-13 deferred confirmation | `PASS_EXECUTION_FAIL_FUTURE_EFFECT` |

## R5 全量候选生成器路线

R5 全量为 32 个事件、18 条独立序列。
候选生成器审计为 32/32 event PASS、4859 行、0 error；re-query 只作为 D2 treatment，
没有把 source candidate 预先绑定 public ID。

| 对比 | H20 | H50 | H100 |
|---|---:|---:|---:|
| R5 D2−D0 identity-error reduction | `0.052202283849918436` CI `[0.006932870370370368, 0.1301041666666666]` | `0.016087516087516088` CI `[-0.005587737813801641, 0.05151821326821326]` | `0.009904153354632588` CI `[-0.0037118972979279987, 0.030355860373018562]` |
| R5 candidate recall | `0.9494290375203915` | `0.9272844272844273` | `0.9047923322683706` |
| treatment assignment changes | `115` | `154` | `209` |
| correct / wrong crossings | `91 / 21` | `127 / 35` | `181 / 46` |
| protected regression | `2` | `2` | `3` |

H20 相对 D0 的均值/CI 是正的，但 H50/H100 的下界分别为
`-0.005587737813801641` 和
`-0.0037118972979279987`；这只能说明短期候选覆盖存在
局部信号，不能支持稳定的 persistent identity future effect。更关键的是 R5 D2−D1：
H20 identity-error reduction 为 `0.0`，CI
`[-0.01666666666666667, 0.01666666666666667]`，delta IoU `-0.002056007158185461`，正确/错误 crossing
`91/21`。

## 两条预留序列的 N72R8 confirmation

confirmation protocol file SHA-256 为 `5f9d94b70410922e5c35732bb3a0cc87a1833fd76b6d92db7dedbac3458aa2ba`；
协议内部声明的 `protocol_sha256` 为 `03146e0fa608227c3d3011bfda3085cc3dd2db929457bb9818de2669a212f35d`。
0020 使用冻结前缀 public-axis 之后的显式 ADD allocator authority
`state=17 -> public_id=1016`；0049 使用冻结前缀中的显式 ATOMIC pair
`target=1003 / other=1004`。这些 public ID 不是从 GT、raw SAM ID 或候选 index 推断的。

| 对比 | H20 identity-error reduction | CI | assignment changes | correct / wrong crossings | protected regression |
|---|---:|---:|---:|---:|---:|
| D1−D0 | `0.0` | `[0.0, 0.0]` | `0` | `1 / 1` | `0` |
| D2−D0 | `0.0` | `[0.0, 0.0]` | `0` | `1 / 1` | `0` |
| D2−D1 | `0.0` | `[0.0, 0.0]` | `0` | `1 / 1` | `0` |

confirmation 的 D2 candidate recall 从 D0 的 `0.8076923076923077` 提升到
target-session 输入，但 assignment 没有改变；因此 candidate presence 的局部改善没有
穿过 assignment decision boundary。两序列的 bootstrap cluster 数为
`2`，不是 32-event 开发集的
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

机器状态：`outputs/N72R7/n72r8_final_gate.json`。主要输入/结果 hash：

- `outputs/N72R7/protocol.json`: `e07a37c98f474dd9cde8a64de35fc91307be5fb6b229d329234bc2329f9a3dba`
- `outputs/N72R7/candidate_generator_protocol.json`: `45a00d6217bcd2431996de2fce8f0c982242e1ecfd0f2f9b2d7459e6e9a410a4`
- `outputs/N72R7/confirmation/confirmation_protocol.json`: `5f9d94b70410922e5c35732bb3a0cc87a1833fd76b6d92db7dedbac3458aa2ba`
- `outputs/N72R7/r5_requery_posthoc/full_attempt1/n72r7_r5_requery_posthoc_results.json`: `a454cc2497019700ef2b29830d3274815327823099108251e3f85e543b19681e`
- `outputs/N72R7/r5_requery_posthoc/full_attempt1/stage_r5_posthoc_status.json`: `756a1b1daa901bbf99d210d5c2528902c0020767e66929e64ce6865279f9d7aa`
- `outputs/N72R7/confirmation/posthoc_attempt1/n72r7_confirmation_posthoc_results.json`: `21232f634fa344e3251a93c67654a7041d6ed331fb9d56bd0528ee5a66ce60ec`
- `outputs/N72R7/confirmation/posthoc_attempt1/stage_13_confirmation_posthoc_status.json`: `08879d948032c10fa1fc7a610a8444133c267c7deea7953e5fa5c958b3d6b4d5`
- `outputs/N72R7/confirmation/posthoc_attempt1/runtime_validation.json`: `5a1d006757efe72c1c4b555534a06972a2f63a8f0ec6eafef90be9498a00583d`
- `outputs/N72R7/confirmation/replay_full_attempt1/runtime_audit.json`: `dbd6478309422e273b719d1c8acd58ebc9b8dcdf9b9d6145d28de3fe69accb2e`

本轮关键代码 hash：

- `sam3_intermot/reacquisition/target_candidate_pool.py`: `4a3c417fd851800a159b6ba70e87b3100fc537d708ed95be9b4a4cd8df92e69e`
- `sam3_intermot/reacquisition/hypothesis_beam.py`: `f7d3e06bf1e5f5dab0159dd7be790e0d86a48df8ee0fc9eeb2c3e6de6af0d2b8`
- `sam3_intermot/reacquisition/progressive_concept.py`: `c7216b2fca34d99465d2a045558f7e5301a79dfb463b7bbf2ea1718017885db0`
- `sam3_intermot/reacquisition/target_id_features.py`: `70a8ef3443fad1359a6727535719793f04eb15e28920029f6aaf0f4d58abd374`
- `sam3_intermot/reacquisition/models/target_id_decoder.py`: `68e93786a136811b7f919327039518c1746481c8b7ae487aa2af5feb2578c93e`
- `scripts/n72r7_dev_replay.py`: `bb0f211ee6f5193fa80b5ff690d6ffd22a58129dda7eb66f1e4042a46fc22027`
- `scripts/n72r7_candidate_generator_requery.py`: `cdc4f67be27a171fe98d2309d8a0c0319f7a881369ef9a6028cb7ff378c16517`
- `scripts/n72r7_r5_requery_replay.py`: `5c4e6a451705df03f45c9a246b0e48c8b8c7565e12dbe14d34ffb5be1cdfa0eb`
- `scripts/n72r7_confirmation_target_stream.py`: `11dbb84fb6b93075c8fbc15f407af2b1cbfbefe4c5dad5465af76377cd24cdcf`
- `scripts/n72r7_run_confirmation_target_stream_batch.py`: `6ef6c4542f49da446a9df60c2d646c3c08def99d71b84731fe89b2a528c75f3b`
- `scripts/n72r7_confirmation_replay.py`: `fbd5b4db2e62783fa9162dd9944df97794aaac22a313a63a63d9e1de4a1f29bb`
- `scripts/n72r7_confirmation_posthoc_score.py`: `273975f83fb39eed8590c43f156ab7f5aee3d326e3ccfccb5cb465edd3314196`

阶段状态与控制器摘要：

- `outputs/N72R7/stage_12_status.json`
- `outputs/N72R7/stage_13_status.json`
- `outputs/N72R7/CONTROLLER_STATUS_attempt3.json`
- `outputs/N72R7/HUMAN_READABLE_STATUS_attempt3.md`

本报告只报告已存在并通过审计的 artifact；不把 `PARTIAL`、候选存在或局部 H20 均值
自动等同于科研成功。
