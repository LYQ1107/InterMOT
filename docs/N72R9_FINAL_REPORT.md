# InterMOT N72R9 最终报告

日期：2026-09-05（Asia/Shanghai）
分支：`codex/n72r9-temporal-closed-loop`
研究状态：`N72R9_DEVELOPMENT_COMPLETE_NO_FRESH_CONFIRMATION`
研究门：`FAIL_FUTURE_REQUERY_EFFECT`

## 1. 执行结论

N72R9 完成了新的 source-aware temporal identity model、真实运行时不确定性触发的未来帧 requery 接口、32-event development replay 和逐帧完整性审计，但没有完成“独立 future-frame requery 稳定改善身份”的严格研究目标。

最重要的区分是：

- `TEMPORAL_CURRENT - BASELINE_B0` 是 **B0 + current target-session candidate + trained temporal model** 的联合效果。它在本次 development 数据上有正向信号：H20/H50/H100 identity-error reduction 分别为 `0.0978793 / 0.0521236 / 0.0373802`，sequence-cluster 95% CI 下界分别为 `0.0490741 / 0.0219580 / 0.0138136`。
- 真正隔离的 `TEMPORAL_REQUERY - TEMPORAL_CURRENT` 只改变 future-frame requery candidate 的加入。其 H20 为 `0`、CI `[0, 0]`；H50 为 `-0.0006435`、CI `[-0.0008333, 0]`；H100 为 `0`、CI `[-0.0004167, 0.0016667]`。因此 future requery 的增量效果没有通过严格门控。
- 联合路径在 H20 产生了 `9` 次 protected identity regression；所以即使联合信号为正，也不能授权生产部署。
- 触发 requery `591` 次，实际加入新候选 `375` 次，assignment 改变 `31` 次，raw binding switch `42` 次，public ID 改变 `0` 次。完整预注册 milestone（不同 requery candidate 被 selector 选中、public ID 不变、记忆安全且 posthoc 从错误变正确）为 `0`；存在 `1` 个部分满足案例，但 selector 因冻结的 admission margin 选择了 `NONE`，solver 最终选回已有 B0 候选。
- 本轮没有 untouched confirmation sequence：元数据审计的 `65` 条可用 train/val 序列全部有历史触及证据，eligible/reserved 均为 `0`。因此结果只能作为 development evidence，不能作为独立确认或真实人工证据。

最终不授权 calibration、selector、decoder LoRA 或生产 identity promotion。

## 2. 冻结范围和历史结果处理

N72R8R1 的协议修正被保留：旧 dancetrack0020/0049 只是普通 target-session diagnostic，不是 fresh confirmation；旧报告中的 raw-binding switch 与 true assignment crossing 已分开；旧的 route-exhaustion 结论没有被继承。N72R8R1/N72R8 及 N36–N72R8 的历史报告、gate、candidate stream、checkpoint、Hungarian solver 和 metric 定义均未改写。

N72R9 使用冻结的 N72R7 D2 development event set，不使用 val/test 的新序列来制造确认。所有事件均为 `interaction_source=simulated_from_gt`，`not_real_human_evidence=true`；这不能被描述为历史真实点击或真实 human-tape confirmation。

运行时只允许使用当前候选、当前帧图像/候选特征、人工 anchor、causal trusted/distractor/neighbor/temporal state 和当前 assignment margin。每个事件先写完 3 个 variant 的 runtime JSONL 和 manifest，再写 `runtime_event_sealed.json`（`gt_loaded=false`），之后才做 posthoc GT scoring。最终 runtime 审计的 future-GT、GT-before-seal、candidate/mapping/matrix/duplicate 检查全部通过。

## 3. Untouched reservation

reservation 只依据 sequence name、split、availability、frame count、历史引用等 metadata，不读取 future outcome 选择序列。结果如下：

| 项目 | 数值 |
|---|---:|
| 可扫描 train/val sequence | 65 |
| eligible untouched sequence | 0 |
| reserved sequence | 0 |
| fresh confirmation authorized | false |

原因是 65/65 序列均存在 N71/N72 及相关历史 development/training/validation/posthoc 触及证据，而不是本地数据缺失。reservation registry hash 为 `f1e3438ea2245844e34a2dfdddbcbd226d1408d13276fb9bfaa652b9cc952b55`；reservation protocol hash 为 `d9601fa6c7067af233c5d1c03bfcf262fa26decfa61d24dce69380901530b4be`。完整 registry 和失败尝试保存在 `outputs/N72R9/confirmation_reservation/`。

## 4. Protocol、数据和模型

冻结 protocol：32 events、18 sequences、30 train events、2 validation events；action 分布为 ADD `4`、AUTHORITATIVE_REASSIGN `14`、ATOMIC_ID_SWAP `3`、RECOVER_IDENTITY `11`。未来窗口为 event frame+1 到 +100，报告 H20/H50/H100；sequence-cluster bootstrap 使用独立 sequence、seed `7290`、`2000` repetitions。protocol 文件的实际 hash 为 `e012ecc3bd64fec4409fccd57d920f3690b8b1c4b5a2dd557e03e2bfb43ef0e9`；文件内部的 `protocol_sha256=86ff10d0762131596d6074c9826552ca06d05ff82d47336154bca7ad940cce13` 是生成器对不含该字段的 canonical body hash，两者语义不同，均予以保留。

模型 `N72R9SourceAwareTemporalIdentityModel` 使用：

- 530-D candidate feature（含 512-D visual feature 和可审计几何/连续性/质量特征）；
- 4-D 显式 source feature：`MAIN_B0_CANDIDATE`、`TARGET_SESSION_CURRENT_RAW`、`TARGET_SESSION_REQUERY`、`UNKNOWN`；
- 4 个 trusted memory slots、4 个 distractor memory slots、8-D temporal context；
- hidden `96`、1 layer、4 heads；set-level candidate + explicit NONE CE，并加固定 hard-negative pairwise margin；
- causal admission：只使用当时 assignment/score/margin/geometry/source 等运行时信息，不用 posthoc GT。

Source-aware corpus 有 `3000` train examples（16 sequences）和 `200` validation examples（2 sequences）。train/validation source counts 分别为：

| split | MAIN_B0 | TARGET_SESSION_CURRENT_RAW | TARGET_SESSION_REQUERY |
|---|---:|---:|---:|
| train | 23,002 | 2,961 | 0 |
| validation | 1,300 | 200 | 0 |

这暴露出一个重要限制：模型虽然显式编码了 requery source，但训练 corpus 没有 requery examples；因此不能把本轮结果包装为已经解决 training-distribution mismatch。

## 5. 实际训练

训练在 CPU 上完成，GPU 使用数为 `0`（协议上限仍为最多 4 张卡、每卡最多一个独立任务）。checkpoint 只由 validation loss 选择，不使用 H20/H50/H100、IDSW、IoU 或 future outcome 选参。

| 指标 | train | validation |
|---|---:|---:|
| examples | 3000 | 200 |
| best epoch | 5 | 5 |
| loss | 0.5326339（最终训练统计） | 0.4806792（best） |
| overall accuracy | 0.7960 | 0.8450 |
| target-candidate accuracy | 0.8314 | 0.9037 |
| NONE accuracy | 0.5590（390） | 0（13） |

validation NONE accuracy 为 `0/13`，是模型校准和泛化能力的明确限制。checkpoint 已保存，但 `production_authorized=false`。

## 6. Replay、因果边界和完整性

3-event smoke 通过后，完整 32-event replay 以每事件一个独立 Python 子进程串行执行；没有并行争抢 GPU/CPU。每个 event 运行 `BASELINE_B0`、`TEMPORAL_CURRENT`、`TEMPORAL_REQUERY` 三个 variant，每个包含 event frame 加 100 个 future frames。

| 审计项 | 结果 |
|---|---:|
| events | 32/32 |
| sequences | 18 |
| variants | 96 |
| runtime frame rows | 9,696/9,696 |
| candidate rows audited | 80,207 |
| duplicate event/candidate | 0 |
| missing frame/variant | 0 |
| matrix error | 0 |
| mapping error | 0 |
| runtime future GT used | false |
| GT loaded before runtime seal | false |

runtime source counts（candidate output rows）为：B0 `24,302`；current variant 为 B0 `24,302` + current target `3,161`；requery variant 再增加 `979` 个 `TARGET_SESSION_REQUERY` rows。event frame 的 candidate set 为空且 memory read=false；所有 treatment 的首次 memory read 为 event+1。

## 7. 未来 requery 机制证据

requery trigger 只使用运行时 base assignment 是否为 NONE 或 base target top1-top2 margin `<0.25`；每帧最多一次。统计为：

| runtime 事件 | 数量 |
|---|---:|
| trigger | 591 |
| requery source actually applied | 375 |
| applied 后 target assignment changed | 31 |
| applied 后 raw binding switch | 42 |
| public ID changed | 0 |
| complete milestone | 0 |
| posthoc wrong→correct partial cases | 1 |

部分案例为 `dancetrack0027 / event 148 / frame 213`：runtime margin `0.2216201` 触发 requery；current variant 选择 target-session candidate，requery variant 因 admission margin `<0.20` 选择 `NONE`，solver 改为已有 B0 candidate；raw binding switch 被记录且 `public_id_changed=false`，`trusted_memory_update=NO_TRUSTED_UPDATE`，posthoc 从 current 的 IoU `0.3997300` 改善到 `0.9677898`。由于不是 selector 选中了新 requery candidate，故只计为 partial，不计入 complete milestone。

这说明当前代码链路确实能够“运行时不确定性→加入当前帧候选→保持 public ID→发生 raw rebinding”，但尚未证明“requery appearance evidence 被 selector/全局接口可靠采纳并长期改善”。

## 8. Paired future replay 结果

以下 identity-error reduction 定义为 baseline error 减 treatment error；CI 是按 sequence cluster 而非帧的 bootstrap。

### 8.1 联合 temporal/current-target 路径：`TEMPORAL_CURRENT_vs_BASELINE_B0`

| horizon | reduction | 95% CI | assignment changes | true correct / incorrect | wrong reassociation rate | protected regression（H20 gate） |
|---|---:|---|---:|---:|---:|---:|
| H20 | 0.0978793 | [0.0490741, 0.2608912] | 144 | 60 / 0 | 0.1957586 | 9 |
| H50 | 0.0521236 | [0.0219580, 0.1631824] | 202 | 82 / 1 | 0.1840412 | — |
| H100 | 0.0373802 | [0.0138136, 0.1222531] | 303 | 119 / 2 | 0.1837061 | — |

这是联合 development signal，不是单独的 source-aware model 或 requery effect；H20 protected regression 使安全门失败。

### 8.2 真正隔离的 future requery 增量：`TEMPORAL_REQUERY_vs_TEMPORAL_CURRENT`

| horizon | reduction | 95% CI | assignment changes | true correct / incorrect |
|---|---:|---|---:|---:|
| H20 | 0.0000000 | [0, 0] | 9 | 0 / 0 |
| H50 | -0.0006435 | [-0.0008333, 0] | 10 | 0 / 1 |
| H100 | 0.0000000 | [-0.0004167, 0.0016667] | 36 | 1 / 1 |

因此不能用 `TEMPORAL_REQUERY_vs_BASELINE_B0` 的正数来声称 requery 有效；它包含了 current target-session + temporal model 的联合变化。

### 8.3 action 分解（H20）

`TEMPORAL_CURRENT_vs_BASELINE_B0` 的 H20 reduction / CI / true crossing 为：ADD `0.1711` / `[0.025, 0.425]` / `13/0`；AUTHORITATIVE_REASSIGN `0.0233` / `[0, 0.1]` / `6/0`；ATOMIC_ID_SWAP `0.3167` / `[0, 0.9]` / `19/0`；RECOVER `0.1000` / `[0.01875, 0.24375]` / `22/0`。ATOMIC action 同时贡献了全部 9 次 H20 protected regression。requery incremental 的四类 action H20 均无正向 crossing/reduction。

## 9. 门控决策

`outputs/N72R9/n72r9_final_gate.json` 的最终门控为：

- `research_gate=FAIL_FUTURE_REQUERY_EFFECT`；
- runtime integrity：PASS；
- combined temporal/current development CI signal：存在；
- isolated future-requery CI signal：不成立；
- protected identity regression：失败（H20 共 9）；
- fresh confirmation：不可用；
- calibration / selector / decoder LoRA / production：全部 `false`。

本轮不能通过增加权重、扩大训练、换 checkpoint、增加 LoRA rank 或调未来阈值绕过失败。最小下一步是获得真正未被历史 development 触及的独立 sequence，并重新冻结 confirmation reservation；若要支持 real-human/user-study claim，还必须由外部 UI 提供 provenance-complete human event tape。当前本地没有这种输入，不能自行生成或把 `simulated_from_gt` 改名为 real human。

## 10. 公开方法检索

检索优先覆盖 2025/2026 的 GitHub、OpenReview、arXiv 和官方论文页面；未使用外部代码或 checkpoint，仅将公开机制映射为本项目的可审计设计。完整记录见 [`N72R9_EXTERNAL_REFERENCE_AUDIT.md`](N72R9_EXTERNAL_REFERENCE_AUDIT.md)。

| 方法 | 可核验页面 | 固定 revision/date | 本项目采用的边界 |
|---|---|---|---|
| SENTRY（ECCV 2026） | [paper](https://arxiv.org/abs/2606.24449) / [GitHub](https://github.com/HamadYA/SENTRY/tree/dd4486c7eeadd7e7022854e29e95e3101390ce65) | `dd4486c7eeadd7e7022854e29e95e3101390ce65`, 2026-07-15 | 参考 trusted/distractor/neighbor 分离和保守 rescue admission；未复制代码。 |
| InteractTrack / IMAT（CVPR 2026） | [arXiv](https://arxiv.org/abs/2604.01974) / [GitHub](https://github.com/NorahGreen/InteractTrack/tree/5f149d4001a84c8b83129192057bf6dd820f71b3) | `5f149d4001a84c8b83129192057bf6dd820f71b3`, 2026-06-16 | 参考 timestamped interaction/memory protocol；其单目标/language setting 不当作 multi-object public-ID 现成方案。 |
| TCEI（CVPR 2026） | [paper/repo audit](https://github.com/1941Zpf/TCEI/tree/145d1b8431398156f8d9f854430e306fdee39eaa) | `145d1b8431398156f8d9f854430e306fdee39eaa`, 2026-03-30 | 参考 transient/accumulated state 分离；没有提前引入 test-time calibration。 |
| STAR（NeurIPS 2025） | [OpenReview](https://openreview.net/forum?id=fmCnNQjZrr) | OpenReview revision audited 2026-04-21 | 仅用于 neighbor/relational reasoning 诊断。 |
| TrackTrack（CVPR 2025） | [official paper](https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR2025_paper.html) | official CVF page | 仅作为 track-perspective association 背景，不替代 immutable public-ID authority。 |

DAM4SAM、SAM2Long、SeC、HMP、UMOT 和 FDTA 的页面、排除理由及是否使用代码也记录在外部审计中。没有找到可直接替换本项目 exact public-ID/Hungarian contract 的公开 drop-in 方法，因此没有凭空增加新接口。

## 11. 失败事实与修复

本轮失败均保留原始 JSON/traceback，没有删除或覆盖：

- corpus builder 首次 compile failure；随后 authority-axis mismatch 暴露了 full public axis 与矩阵 identity rows 的差异，按显式 identity rows 做最小修复后通过；
- 默认 Python 缺少 torch；改用项目 `intermot` environment；
- training smoke 首次 shape check 错误；修正 candidate feature shape 比较后，同一 smoke 通过；
- aggregation 首次因新增函数返回值漏传 `refs` 触发 `NameError`；`outputs/N72R9/replay/attempts/aggregate_failure_attempt1.json` 保留 traceback，最小修复后同一聚合通过；
- reservation scanner 的前六轮环境/树扫描失败和第七轮零 eligible gate 语义均保留在 `outputs/N72R9/confirmation_reservation/attempts/`。

这些是工程/审计修复，不是降低 efficacy gate，也没有重跑已通过的 N72R7/R8 实验。

## 12. 资源、隔离和 ICLR 日历

N72R9 训练和 replay 实际使用 CPU；GPU 使用 `0/4`。若下一轮拥有合法 untouched 输入，计划最多使用四张 GPU，每张卡同时只运行一个独立序列/frame-range 进程；OOM 只使用既定 `160→100→50` 分片策略。没有修改 `third_party/sam3`、历史 outputs、共享 checkpoint 或 production MOT/OVMOT 配置；新代码、日志和 checkpoint 只在 N72R9 独立范围内。

ICLR 2027 冻结日历：

- 摘要截止：2026-09-18 AoE；
- 全文截止：2026-09-25 AoE；
- 本报告日期：2026-09-05，距离摘要约 13 个日历日、距离全文约 20 个日历日。

在截止日期前最有价值的工作不是盲目增大模型，而是补齐未触及的独立确认数据并保持一次性冻结；如果无法获得该数据，应在论文中明确报告 development signal、requery null effect、protected regression 和 simulated-from-GT 限制。

## 13. 机器证据和代码 hash

主要机器证据：

- [`outputs/N72R9/n72r9_final_gate.json`](../outputs/N72R9/n72r9_final_gate.json)
- [`outputs/N72R9/ccam_paired_replay_results.json`](../outputs/N72R9/ccam_paired_replay_results.json)
- [`outputs/N72R9/replay/full/runtime_audit.json`](../outputs/N72R9/replay/full/runtime_audit.json)
- [`outputs/N72R9/stage_06_requery_status.json`](../outputs/N72R9/stage_06_requery_status.json)
- [`outputs/N72R9/stage_07_replay_status.json`](../outputs/N72R9/stage_07_replay_status.json)
- [`outputs/N72R9/CONTROLLER_STATUS.json`](../outputs/N72R9/CONTROLLER_STATUS.json)

关键 hash：

```text
scripts/n72r9_aggregate_replay.py                         f511fe8b5afff094a961436a903ccecda4f5b3baeee72bcedf88cb64e0564914
scripts/n72r9_temporal_replay.py                           55e142c37dd9f6c800f7d8e2d441b34e5152c6103adc17dcaf80e7da5c731023
scripts/n72r9_run_temporal_replay_batch.py                 f8bb1a3e227193cd7a19165734f26152706bef4a7d01d6552d651bb3ee28ae27
scripts/n72r9_build_temporal_corpus.py                     c189cd2c02e006cb63b76664977fd111b2d80ee9642aed99b8917daa83ceeed3
scripts/n72r9_train_temporal.py                            a305989f5ec8d2b44948932a8b6e7c1beeb796b369931d1f8051c501c099dd6e
sam3_intermot/reacquisition/models/n72r9_source_temporal.py 6e3e72195c8a33f438aa4943e837304ae3826fd352c9052630e7753a54f3a04f
outputs/N72R9/protocol.json                                e012ecc3bd64fec4409fccd57d920f3690b8b1c4b5a2dd557e03e2bfb43ef0e9
outputs/N72R9/training/corpus_manifest.json                 ac84e07be60d08926c23181ead08666ad639a8f8e07b5f3f60a0190c936ec715
outputs/N72R9/training/N72R9SourceAwareTemporalIdentityModel_v1.pt
                                                             27a9f0499cbb67947e0e01f9705df898673b0a8afe19f91018b2864af22c11d5
outputs/N72R9/training/training_history.json                7484e71aa22b7f742634d28ab92d3c12b3aae92473b66417b8b494ab94ee71d7
outputs/N72R9/replay/full/batch_manifest.json               4f8fe39a23c2c9988f776a3d8f5b9655f2148474d48bb533c2751b575e7ba1e4
outputs/N72R9/replay/full/runtime_audit.json                19627fd73cc509812be5b37eb51061e6c64f33ed3a10257b8e4e54d61b991534
outputs/N72R9/ccam_paired_replay_results.json               fa4a17ee5fcf2727739eb759bac7dab2ab7c49c583eaba718e6a87fb4b8717cc
```

最终 Git commit：`5b0c6d6`（本报告和本轮代码均随该提交推送）；本报告不会改变任何历史 N36–N72R8 证据。
