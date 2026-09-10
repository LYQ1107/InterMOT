# InterMOT N72R14 Final Report

Date: 2026-09-10 (Asia/Shanghai)
Branch: `codex/n72r14-persistent-state-association`
Base: N72R13 `e7bb637a400ade5c814fb3eb2b91527f50901c2e`
Research status: **`FAIL_FUTURE_EFFECT`**
Structural status: **PASS**
Final classification: **`PERSISTENT_STATE_EDGE_NOT_YET_EFFECTIVE`**

## First-screen answers

1. **Persistent state真正进入global association了吗？**

   是。G1 在每个 future frame 构造 candidate × all-public-ID 的 state-edge 矩阵，并把它送入同一个 exact public solver。T1 只启用 target public-ID 列；G1 的非 target edge cells 有 `196,738` 个非零值，T1 为 `0`。

2. **G1是否超过E0？**

   否。H100 官方 interaction-window diagnostic 中 G1−B0 为 HOTA `-0.006968`、AssA `-0.003810`、IDF1 `-0.010685`，IDSW 从 `177` 增加到 `207`。Causal global identity-error reduction 为 `-0.005697`，95% sequence-cluster CI `[-0.014118,+0.013279]`。

3. **G1是否超过target-only T1？**

   H100 的 TrackEval 数值只出现很小的混合变化：HOTA `+0.000541`、IDF1 `+0.000766`、IDSW `-9`，但 AssA `-0.000543`；这不是严格 future-effect 成功。Causal G1−T1 的 target identity reduction 为 `0`，H100 global reduction 仅 `+0.000039`，CI 下界为 `0`。

4. **AssA提升多少？**

   相对 B0 没有提升，H100 为 `-0.003810`（约 `-0.381` 个百分点）；相对 T1 为 `-0.000543`。

5. **IDF1提升多少？**

   相对 B0 没有提升，H100 为 `-0.010685`（约 `-1.068` 个百分点）；相对 T1 为 `+0.000766`，幅度很小且没有通过完整 gate。

6. **IDSW减少多少？**

   相对 B0 没有减少，H100 增加 `30`（`177→207`）；相对 T1 减少 `9`（`216→207`）。

7. **RECOVER有没有改善？**

   没有。RECOVER 有 `11` 个事件、`8` 条独立序列；G1−B0 的 H100 global identity-error reduction 为 `-0.012595`，95% CI `[-0.063487,+0.011455]`。RECOVER 不是正向稳定子集。

8. **state propagation是否真正持续到t+1/t+20？**

   结构上是。T1/G1 都有 `3,200` 个 state-changing future frames，检查 `3,168` 次 t→t+1 edge transition，`3,168/3,168` 的 next edge 改变，`0` 次 state changed but edge unchanged；runtime 继续覆盖 H20/H50/H100。它证明 state 连到了 solver 输入，不证明每次 edge 变化都能正确翻转 assignment。

9. **是否触发次数/强度消融？**

   否。固定的 strong-positive gate 未通过，因此没有运行 count/strength ablation，也没有运行 human-budget ablation。

10. **如果消融执行：次数过多是否导致错误累积？强度过大是否导致identity lock-in？**

    不适用；本轮没有授权消融，所以这两个问题没有被测量，不能做结论。

## 1. 研究问题与冻结边界

N72R14 检验的是：把持久 public identity state 放入全局 association 矩阵后，人工纠正写入的状态是否能从 event+1 起改善多身份竞争。实验固定：

- N72R9 冻结 protocol：`32` events、`18` independent sequences、train/train-fold、`H20/H50/H100`；protocol SHA-256 `e012ecc3bd64fec4409fccd57d920f3690b8b1c4b5a2dd557e03e2bfb43ef0e9`。
- 候选生成、candidate order、embedding 定义、checkpoint 来源、Hungarian/exact public solver、none score 与评价定义不变。
- runtime 不读取未来 GT；GT 仅在全部 runtime artifact 通过验证后由 posthoc aggregator 读取。所有新 runtime 记录的 `runtime_future_gt_used=false`。
- 事件仍是 `simulated_from_gt`，不是历史真实点击；真实 human tape 数量仍为 `0`。
- 没有修改 `third_party/sam3`、TrackEval checkout、N36–N72R13 历史输出、共享 checkpoint/config，亦未修改 MOT/OVMOT 项目目录。

本轮 formal replay 没有重新运行 SAM3，直接读取已冻结 candidate streams；“H100”表示未来窗口长度 100，不是声称重新占用 H100 GPU。

## 2. 实现与四个 variant

新增隔离模块：

- `sam3_intermot/association/persistent_public_state.py`：`PublicAssociationMotionState` 明确区分 `association_state_id`、`public_id`、`mot_track_id` 语义；bank 以 `public_id` 为 key，保存 last box、velocity、last seen、raw/native scope、lost age 与 assignment count，并拥有现有 `AppearanceMemory`。
- `sam3_intermot/association/persistent_state_edge.py`：输出 candidate × public-ID 矩阵。对每条 edge 使用 appearance memory、既有 `online_associator.predicted_iou` motion 逻辑、raw/native scope continuity 和 gap；固定
  `r=A+1.0*M+0.5*N-0.1*G`、`Delta=tanh(r)`、`STATE_EDGE_SCALE=1.0`。没有用零向量冒充缺失 feature。
- `scripts/n72r14_run_persistent_state_formal.py`：四 variant 的同一 candidate stream、同一 exact solver、事件帧隐藏/下一帧可见和原子 runtime artifacts。

四个 variant 为：

| Variant | 语义 |
| --- | --- |
| B0 / `E0_BASELINE_B0` | corrected E0，只有 frozen base candidate pool，edge off |
| P0 / `E1F_TARGET_POOL_ONLY` | B0 + target-session current raw pool，edge off |
| T1 / `E1G_TARGET_STATE_ONLY` | P0 + state edge 只进入 target public-ID 列 |
| G1 / `E1H_PERSISTENT_GLOBAL_STATE` | P0 + state edge 进入全部 public-ID 列 |

G1 仍使用唯一 official exact solver wrapper；没有引入第二个正式 solver。状态更新发生在 solver 之后，event frame 不读取刚写入的人类 memory，首次可见是 event+1。

## 3. 失败事实与最小修复

首轮 N72R14 formal runtime 结构上生成了 32×4×101 行，但首次 TrackEval export 在
`n72r5-pool-n37-dancetrack0027-0148-authoritative_reassign-007/E0_BASELINE_B0` 的 relative future frame 74（absolute frame 222）发现 exact-solver assigned row 的退化框：

```text
[571.0000228881836, 307.00000047683716,
 571.0000228881836, 308.00000050105155]
```

该行 `x2==x1`，不能在 posthoc 通过 clipping、drop 或替换框修复。首轮命令退出码为 `2`，traceback 保存在：

- `outputs/N72R14/trackeval/export_failure_attempt_01.json`
- `outputs/N72R14/stage_05_attempt_01_failure.json`
- 首轮原位置 `outputs/N72R14/trackeval/export_failure.json`

首轮 runner 的 actionable root cause 是所有 `build_candidate_pool` 调用错误使用了 `require_positive_geometry=False`，与 N72R11R5R1 corrected E0 的 solver 前正面积策略发生漂移。最小修复是只把 N72R14 event-frame 初始化和每个 future-frame pool 的参数改为 `require_positive_geometry=True`，保留 rejected source rows 在 pool audit 中；没有改候选定义、solver、评价或数据。

同时保留了第一次 CPU interface audit 的字段层级误报：
`outputs/N72R14/interface_audit_failure_attempt_01.json`。修复仅改为从 `score_audit` 读取 frozen matrix/axis，并用同一输入回归；最终接口审计 `32/32` events PASS，toy contract PASS。

修复后的 attempt-02 独立目录没有覆盖首轮 runtime：

- RECOVER smoke：`1` event × `4` variants × `101` rows，PASS。
- formal：`32/32` events、`128` event×variant artifacts、`12,928` runtime rows，PASS；assigned invalid geometry `0`，duplicate/missing `0`。
- window export：`384/384` event×variant×horizon records，PASS。
- pinned TrackEval commit：`12c8791b303e0a0b50f753af204249e622d0281a`。

## 4. 官方 interaction-window 结果

这些是 event-window diagnostic，不是完整 DanceTrack benchmark 分数。每一行由 pinned official TrackEval 对相同 H 窗口汇总。

| Variant | H | HOTA | AssA | IDF1 | IDSW | DetA | MOTA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 20 | 0.680935 | 0.772592 | 0.779271 | 32 | 0.605232 | 0.556068 |
| P0 | 20 | 0.678553 | 0.766126 | 0.778408 | 49 | 0.606098 | 0.553215 |
| T1 | 20 | 0.675099 | 0.767327 | 0.771768 | 45 | 0.599260 | 0.539171 |
| G1 | 20 | 0.675099 | 0.767327 | 0.771768 | 45 | 0.599260 | 0.539171 |
| B0 | 50 | 0.667856 | 0.733672 | 0.774861 | 91 | 0.610643 | 0.559352 |
| P0 | 50 | 0.664840 | 0.730984 | 0.771006 | 121 | 0.607365 | 0.551508 |
| T1 | 50 | 0.659884 | 0.727714 | 0.763710 | 114 | 0.601131 | 0.537650 |
| G1 | 50 | 0.660405 | 0.728591 | 0.764521 | 110 | 0.601360 | 0.538435 |
| B0 | 100 | 0.665767 | 0.721408 | 0.788364 | 177 | 0.616112 | 0.579139 |
| P0 | 100 | 0.662006 | 0.718851 | 0.784188 | 221 | 0.611318 | 0.569195 |
| T1 | 100 | 0.658259 | 0.718141 | 0.776913 | 216 | 0.605079 | 0.554735 |
| G1 | 100 | 0.658800 | 0.717598 | 0.777679 | 207 | 0.606533 | 0.557732 |

H100 G1−B0：HOTA `-0.006968`、AssA `-0.003810`、IDF1 `-0.010685`、IDSW `+30`、DetA `-0.009579`、MOTA `-0.021408`。H100 G1−T1：HOTA `+0.000541`、AssA `-0.000543`、IDF1 `+0.000766`、IDSW `-9`。

## 5. Causal identity and assignment diagnosis

正值表示 treatment 相对 baseline 的 identity-error reduction。Bootstrap 单位是独立序列，seed `7214`，`2000` repetitions。

| Comparison | H | Mean global reduction | 95% CI lower / upper | Target reduction | Assignment changes (correct / incorrect) |
| --- | ---: | ---: | ---: | ---: | ---: |
| P0−B0 | 20 | +0.006357 | +0.001385 / +0.017711 | +0.054642 | 182 (45 / 13) |
| P0−B0 | 50 | +0.001689 | -0.002564 / +0.012066 | +0.014045 | 248 (59 / 38) |
| P0−B0 | 100 | +0.001743 | -0.002178 / +0.011606 | +0.009833 | 342 (89 / 58) |
| T1−P0 | 20 | -0.006280 | -0.014744 / +0.003659 | -0.034030 | 99 (11 / 32) |
| T1−P0 | 50 | -0.005943 | -0.010402 / +0.004256 | -0.033979 | 173 (19 / 72) |
| T1−P0 | 100 | -0.007480 | -0.011904 / +0.002547 | -0.042806 | 339 (25 / 160) |
| G1−P0 | 20 | -0.006280 | -0.014712 / +0.003214 | -0.034030 | 99 (11 / 32) |
| G1−P0 | 50 | -0.005943 | -0.010152 / +0.004358 | -0.033979 | 173 (19 / 72) |
| G1−P0 | 100 | -0.007440 | -0.012007 / +0.002585 | -0.042806 | 339 (25 / 160) |
| G1−B0 | 20 | +0.000077 | -0.011586 / +0.019101 | +0.020611 | 281 (56 / 45) |
| G1−B0 | 50 | -0.004254 | -0.012768 / +0.016358 | -0.019934 | 421 (78 / 110) |
| G1−B0 | 100 | -0.005697 | -0.014118 / +0.013279 | -0.032973 | 677 (112 / 216) |

G1−B0 H100 的 `649` global-assignment-changed frames 中，target crossing 只有 `112` correct、`216` incorrect、`335` neutral；所以 edge 变化确实穿过了部分 assignment boundary，但方向总体不正确。G1−P0 H100 为 `25` correct、`160` incorrect，说明 global edge 相对 target-pool control 带来额外伤害。

### RECOVER 分解

| Horizon | G1−B0 mean global reduction | 95% CI | Events / sequences |
| ---: | ---: | ---: | ---: |
| H20 | -0.012058 | -0.067708 / +0.019360 | 11 / 8 |
| H50 | -0.011560 | -0.057996 / +0.011405 | 11 / 8 |
| H100 | -0.012595 | -0.063487 / +0.011455 | 11 / 8 |

## 6. State-to-solver propagation audit

`outputs/N72R14/attempt_02/state_causal_propagation_audit.json` 对 T1 与 G1 分别验证：

| Variant | State-changing frames | t→t+1 checks | Next edge changed | Unchanged after state change | Mean edge L1 | Mean target edge change | Mean non-target edge change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T1 | 3200 | 3168 | 3168 | 0 | 0.035592 | 0.063171 | 0.030934 |
| G1 | 3200 | 3168 | 3168 | 0 | 0.063817 | 0.063171 | 0.063593 |

因此本轮不是“memory 没有接到 solver”的结构性失败。实际瓶颈在于：全局 edge 虽然改变 fused matrix 与一部分 assignment，却没有形成正确、稳定的 public-ID 决策；同时 P0 的 candidate pool 本身已经改变了结果，G1 相对 P0 进一步恶化。

## 7. Strong-positive gate and authorization

预注册 `STRONG_POSITIVE_PSCA=true` 必须同时满足 H100 G1−B0：ΔHOTA≥`+0.0030`、ΔAssA≥`+0.0050`、ΔIDF1≥`+0.0050`、IDSW 不增加，以及 H20 identity-error reduction>0。

本轮结果：

| Check | Result |
| --- | --- |
| H100 ΔHOTA ≥ +0.0030 | FAIL (`-0.006968`) |
| H100 ΔAssA ≥ +0.0050 | FAIL (`-0.003810`) |
| H100 ΔIDF1 ≥ +0.0050 | FAIL (`-0.010685`) |
| H100 IDSW not increased | FAIL (`177→207`) |
| H20 identity reduction > 0 | PASS (`+0.000077`) |
| Strong positive | **FAIL** |

因此 `outputs/N72R14/STRONG_POSITIVE_STATUS.json` 为 `FAIL`，没有执行 frequency/strength/human-budget ablation。没有训练 calibration head、selector、decoder LoRA，也没有把 PSCA 提升为 production InterMOT 方案。

## 8. Machine-readable evidence

主要 attempt-02 证据：

- `outputs/N72R14/n72r14_final_gate.json`：最终 `FAIL_FUTURE_EFFECT` / `PERSISTENT_STATE_EDGE_NOT_YET_EFFECTIVE`。
- `outputs/N72R14/interface_audit.json`：32-event interface audit PASS。
- `outputs/N72R14/attempt_02/stage_02_status.json`：RECOVER smoke PASS。
- `outputs/N72R14/attempt_02/stage_03_status.json`：formal replay PASS。
- `outputs/N72R14/attempt_02/causal_metrics.json`：posthoc causal metrics PASS。
- `outputs/N72R14/attempt_02/state_causal_propagation_audit.json`：state connection PASS。
- `outputs/N72R14/attempt_02/trackeval/export_manifest.json`：384-record export PASS。
- `outputs/N72R14/attempt_02/trackeval/trackeval_run_manifest.json`：384-record official TrackEval run PASS。

关键 hashes：

```text
formal_manifest.json             c2124f2a500a6f00f44f4264aaf148147acb7e6681d3a7ef9fd4e68af92c6d46
causal_metrics.json              70f1915f2f577e215f328311ff04232a0abcad4db48ba091a3bb4f6f9c6e8d91
state_causal_propagation_audit   3a0e1e08d6930d71d92573104ef0349a8cd70814c389558b536f9eeed8e9a55d
trackeval export_manifest        b023daba43fbd8795e6d7e9391f446b2cf8cc383404c5957a6e2407926a2d9e0
trackeval run_manifest           c4948aa1899af203c33ba83a52027ce4c13d7b13e7eebef0d7fe75c247b2079a
n72r14_final_gate.json           3d3e5313a42691e75aa7dd3d87a030e2abd0e912ff4ceb3bac725014485886e3
```

N72R14 code hashes（提交前）：

```text
sam3_intermot/association/persistent_public_state.py  ebb9043293242382e6d805e84204c21f5f72b61ef3348058ff3249f36a8678db
sam3_intermot/association/persistent_state_edge.py    ea98f47153ae3931fdcf9df0cd8ab6b462dc8745f8eea57fb410407ee84dc866
scripts/n72r14_interface_audit.py                     a8f223a622bafd559eccf30605ae0f43f770af7dc0ed27bbcb79148b9f483dc7
scripts/n72r14_run_persistent_state_formal.py         b1be10b2c3648e4a4013f977d368420bc429f54753bd5d3f2b74522ce4a74e7d
scripts/n72r14_aggregate_metrics.py                   c7a01c3f675761dca254914ea243529174bf94bcc1425f1f59be57d36c97962a
scripts/n72r14_state_propagation_audit.py             033e032102671eb8b045420042e68e682c67b62113b162b76a1f26e98655a0d8
scripts/n72r14_export_window_trackeval.py             ad17ce4f2de4ea94ad8d96e911cb34d6ad63aef911945b5a2d55335a20d79d69
scripts/n72r14_run_window_trackeval.py                 87826d51639d985b9c7fada393dfb7bb177320b26c560de56d31932261d652fc
scripts/n72r14_finalize.py                             3a5924f26e9162957075cec6e3f880dc0a5a874ae608a6785dfdd528fa811090
```

编译检查覆盖两个 association modules、interface audit、formal runner、aggregator、propagation audit、window exporter/runner 与 finalizer，`git diff --check` 通过。conda 的 `osr_lib-1.1.0-nspkg.pth` loader warning 在每次 Python 启动时出现，但没有改变 exit code 或实验产物，已作为环境 warning 保留。

## 9. Final scientific conclusion and next step

N72R14 证明了 persistent public state 可以以明确的 candidate×all-public-ID 矩阵接入 exact global association，并且状态更新会在下一帧改变 solver 输入；但在当前冻结候选、base score、appearance feature coverage 与评价协议下，G1 没有产生可靠的正确 assignment change，H100 还恶化了 HOTA/AssA/IDF1 并增加 IDSW。结论不是“模块未调用”，而是 **global state edge 尚未成为有效的 identity association mechanism**；P0 candidate-pool effect 与 G1 relative-to-P0 的负 crossing 是主要诊断。

最小后续方向是保留本轮 edge/propagation 作为接口诊断证据，冻结一个新的 association decision-boundary / candidate-quality probe，或取得 provenance-complete real-human event tape；不得以增加权重、换 checkpoint、LoRA、selector、calibration 或 count/strength sweep 绕过本轮 gate。

ICLR 2027 时间约束（当前日期 2026-09-10）：摘要截止 2026-09-18 AoE，剩余约 8 天；全文截止 2026-09-25 AoE，剩余约 15 天。若继续研究，应优先完成可解释的 association-interface 诊断和真实人工事件采集，不把本轮负结果包装成方法成功。
