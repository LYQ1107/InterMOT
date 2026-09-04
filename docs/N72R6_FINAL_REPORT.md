# N72R6 Final Report — Target-Scoped Correction Architecture

日期：2026-09-04（Asia/Shanghai）
项目：`InterMOT`
分支：`codex/n72r6-target-scoped-correction`
最新代码提交：`9bab1e4344796e43c035a87f192b70de48d5dded`

## 结论

N72R6 的结构性目标基本完成，但科研 future-effect gate 未通过：

```text
N72R6_STATUS       = FAIL_FUTURE_EFFECT
TARGET_ARCHITECTURE = STRUCTURALLY_AUDITED
RESEARCH_GATE      = FAIL_FUTURE_EFFECT
PRODUCTION         = NOT_AUTHORIZED
CALIBRATION        = NOT_AUTHORIZED
SELECTOR           = NOT_AUTHORIZED
DECODER_LORA       = NOT_AUTHORIZED
```

target-scoped correction 确实阻断了主 B0 candidate stream 的修改，但没有稳定降低
目标身份错误。最终主要瓶颈是 target-session candidate 的传播与空间质量，而不是
native/public 映射或 Hungarian solver 拒绝了合法 target row。

## 冻结输入与边界

- 复用 N72R5R1 的 B0 main stream、40-event policy、checkpoint、candidate 定义、
  Hungarian 求解器、H20/H50/H100 和 sequence-cluster bootstrap 定义。
- 40 个候选中 32 个 `B1_SPATIAL_CORRECTION_ONLY` 为 `APPLIED`，覆盖 18 条独立序列；
  action 数量为 ADD 4、ATOMIC 3、AUTHORITATIVE_REASSIGN 14、RECOVER 11。
- recovery replacement 使用同一事件/同一窗口；没有重新选择事件，没有读取 val/test。
- 全部事件仍是 `interaction_source=simulated_from_gt`，不是历史真实人工点击；真实
  human tape 数量仍为 0。
- runtime 全程 `runtime_future_gt_used=false`。GT 只出现在 sealed replay 完成后的
  posthoc scoring/root-cause audit。
- `third_party/sam3`、N36–N72R5R1 历史证据、checkpoint 和冻结 B0 stream 未修改。

## 实现内容

新增或修改的机制包括：

1. 独立 `TargetScopedCorrectionSession`，main SAM session 不接收人工 prompt。
2. `CorrectionEpoch` 与 persistent public identity 保持绑定，清除旧 native scope，
   重新锚定 human ROI，并冻结 target machine appearance prototype。
3. scope-aware native identity，区分 main session 与 target correction session。
4. candidate merge 与 target-exclusive domain：target-session candidate 只能进入
   明确的 target public ID 或 NONE，不能占用 protected public ID。
5. human-anchor verification gate：固定 cosine threshold `0.85`，只对 event+1
   之后的 target-session candidate 生效，拒绝时进入显式 NONE。
6. 单独注册的 B0 target-main fallback：仅当 target row 被拒绝/缺失时，才允许同帧
   冻结 B0 中原本承载该 target public ID 的 shadowed UID 通过同一固定 anchor gate；
   其他 main row 永远不能进入 target public domain。
7. recovery failure 的合法记录：官方 target observation 缺失不会伪造 observation，
   只写明失败并继续后续帧审计。

## 执行与完整性结果

| 阶段 | 结果 |
|---|---|
| Stage 00 protocol/hash freeze | PASS |
| target-session recovery stream | 32/32 validated；duplicate/missing/unexpected = 0 |
| fallback four-action smoke | 4/4 PASS |
| fallback full C0/C1 replay | 32/32 PASS |
| fallback structural audit | PASS；main mutation/domain/GT/axis violations = 0 |
| fallback posthoc effect | FAIL_FUTURE_EFFECT |
| root-cause audit | PASS |
| native/geometry authority audit | PASS；not primary bottleneck |

关键结构数字：

- full replay：32 events、18 sequences、每事件 101 帧。
- future frame 总数：3200。
- accepted target-session rows：831；其中 831/831 被分配到显式 target public ID。
- 合法 B0 target-main fallback rows：374；374/374 被分配到目标 public ID。
- 既无 target row 也无合法 fallback source 的帧：1995。
- `main_candidate_mutation_count=0`。
- target-domain、runtime-GT flag、public-axis violation 均为 0。
- fallback UID 不在 shadowed 集合、或 replay/audit UID 不一致的记录均为 0。
- target-session native scope match：831/831；target-public base score 非有限或非正：0。

## Future-effect 结果

### 1. recovery-only C1

recovery 修复了最初的 39 个 candidate absence，但没有修复大规模漂移：

- H20 C1−C0：`-0.2054661969`
- sequence-cluster 95% CI：`[-0.3455224787, 0.1312917574]`
- H50：`-0.3170492240`
- H100：`-0.3730811735`
- protected H20 regression：49
- target candidate recall：`0.9878125`
- correct/incorrect crossing：6/7

### 2. fixed human-anchor gate

固定 gate 将一部分 target-session 漂移过滤掉，但过度减少了候选覆盖：

- H20 C1−C0：`-0.3599366830`
- sequence-cluster 95% CI：`[-0.4591578443, -0.0425281125]`
- protected H20 regression：20
- target-session candidate recall：`0.2596875`
- correct/incorrect crossing：5/7

### 3. B0 target-main fallback

回退只允许冻结 B0 中曾经承载目标 public 的行，未重新标记其他 protected 行。它相对
单纯 gate 减轻了负效应，但仍未通过严格 gate：

- H20 C1−C0：`-0.2478247549`
- sequence-cluster 95% CI：`[-0.3615559470, 0.0262414045]`
- H50 C1−C0：`-0.4080980146`
- H100 C1−C0：`-0.4398111516`
- protected H20 regression：20
- target-session candidate recall：`0.2596875`
- correct/incorrect crossing：5/2
- `strict_sequence_cluster_ci_lower_gt_zero=false`

因此局部出现更多 correct crossings 或 protected regression 下降，不能被包装为总体
成功；95% CI 下界没有严格大于 0，且 target identity error/missing 仍上升。

## 根因判定

最终 fallback root-cause audit：

- `TARGET_SESSION_IDENTITY_DRIFT`：1 个事件；
- `TARGET_SESSION_PROPAGATION_FAILURE`：15 个事件；
- 两者同时存在：16 个事件；
- target-candidate absent visible frames：2312；
- target-candidate drift frames：351；
- target-candidate spatial hits：467/831；
- target-session native scope match：100%；
- target-session solver assignment given a row：100%；
- target-session target-public base score：全部有限且为正。

这组证据排除了“目标 row 被 native/geometry/solver authority 拒绝”为主因。当前主因
是 target-only SAM session 提供的候选在未来传播中缺失或空间漂移；human anchor gate
能够提高部分候选的空间精度，却同时牺牲了大量 coverage，B0 fallback 只能部分缓解。

## 失败事实与修复记录

1. 首次 recovery batch 中，`dancetrack0029:0112` 的官方 recovery 在两次支持的
   prompt 尝试后没有返回与 target box 重叠的 observation。原始 failure artifact、
   retry failure 和 traceback 均保留；第三次同事件按协议记录为合法 recovery miss，
   没有伪造候选。
2. fallback replay 第一次结构审计退出码为 1，`domain=374`。逐字段检查确认 374
   条全部是协议允许的 shadowed B0 target-main fallback，而非越权映射；审计器随后以
   最小范围支持该新协议，重跑同一 replay 后通过。原始失败事实没有覆盖。
3. Python 环境每次启动会输出 `osr_lib-1.1.0-nspkg.pth` 的 site-package warning，
   但相关命令退出码均为 0，未改变实验结果。

## C2、训练与生产决策

C2/TVC 没有运行。N72R6 的预注册顺序要求先得到正向 C1，再在相同架构上评估 C2；
recovery、human-anchor gate 和 fallback 三种 C1 均未通过 strict future-effect gate。
因此本报告不对 C2/TVC 的效果作任何推断。

没有启动 calibration head、selector、decoder LoRA 或生产 promotion。增加权重、换
checkpoint、修改 Hungarian、缩短窗口、改变指标或把 simulated-from-GT 改称 real
human 都不被允许。

## 代码与验证

- focused tests：`6 passed`。
- N72R6 相关模块和脚本 `py_compile`：PASS。
- minimal imports：PASS。
- `git diff --check`：PASS。
- `third_party/sam3`：无修改。
- 代码已推送至：<https://github.com/LYQ1107/InterMOT/tree/codex/n72r6-target-scoped-correction>
- 代码提交：`9bab1e4344796e43c035a87f192b70de48d5dded`

核心输入/输出 hash 见：

- `outputs/N72R6/protocol.json`：`9e78d8e7f4e4142ba58ca60c74ac07e667391a37436b06cbe5dced2829476ab4`
- `outputs/N72R6/recovery_target_stream_manifest_attempt3.json`：`cee4a97d8aab64f06bae47e6d4baa26c6cb0e5c0a5e0ced7007152a96cebad3c`
- `outputs/N72R6/ccam_paired_replay_results_human_anchor_fallback.json`：`8424a8cc38635ed54f5ab28a690d7b24d19df276f2fd9ad9aef8050be47edb75`
- `outputs/N72R6/target_root_cause_audit_human_anchor_fallback.json`：`917664d68367912a05766fc1e9eefa258bbdd38eef9a3130a40b8189703a4cf0`
- `outputs/N72R6/native_geometry_authority_audit_human_anchor_fallback.json`：`1c756bb98eafe4b276ce972db0c70c29975f32fd575e2efa7369064cc0292594`。

机器 gate：`outputs/N72R6/n72r6_final_gate.json`。阶段汇总：
`outputs/N72R6/stage_09_status.json`。

## 最小下一步

冻结一个新的 target-session candidate-source/propagation quality probe，保持 B0 main
stream、checkpoint、Hungarian、评价定义和 runtime GT 边界不变，优先解决 candidate
absence/spatial drift，再决定是否有资格运行 human-anchor TVC/C2。真实部署结论仍需要
外部 provenance-complete human tape；当前 synthetic evidence 不能替代它。
