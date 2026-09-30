Learned Identity Memory 在噪声观察条件下是否超过 EMA？**否。**

FINAL GOAL:
Human Identity Memory Learning

CENTRAL QUESTION:
“学习一个身份记忆更新模型，能不能在真实噪声观察条件下超过一个简单 EMA identity memory？”

N72R19 结果：`ROBUST_LEARNED_MEMORY_NOT_VALIDATED`。

## Frozen scope

本阶段只研究身份记忆更新；使用 N72R17 OSNet 512-D 冻结 embedding、N72R18 GRU checkpoint 和 N72R19 噪声 replay。没有运行 SAM3、MOT、Hungarian、TrackEval、HOTA、IDF1、Requery、InterMOT association、LoRA 或新的 ReID encoder 训练，也没有保存 crop。

## Primary endpoint: H100 hard-negative identity win rate

| Condition | EMA(0.90) | N72R18 GRU | N72R19 GRU+Reliability | Robust−EMA | Robust CI lower−EMA | Cell |
|---|---:|---:|---:|---:|---:|---|
| Clean | 78.13% | 91.36% | 46.43% | -31.70 pp | -37.90 pp | REFERENCE |
| Wrong identity 0.10 | 75.27% | 82.14% | 44.36% | -30.91 pp | -36.86 pp | FAIL |
| Wrong identity 0.20 | 71.25% | 72.90% | 42.29% | -28.95 pp | -34.70 pp | FAIL |
| Wrong identity 0.30 | 66.30% | 63.84% | 40.48% | -25.82 pp | -31.39 pp | FAIL |
| Wrong identity 0.50 | 52.39% | 45.94% | 35.60% | -16.78 pp | -21.65 pp | FAIL |
| Missing observation 0.10 | 76.70% | 90.34% | 45.93% | -30.78 pp | -36.94 pp | FAIL |
| Missing observation 0.20 | 74.77% | 88.98% | 45.47% | -29.30 pp | -35.37 pp | FAIL |
| Missing observation 0.30 | 72.58% | 87.43% | 45.01% | -27.57 pp | -33.75 pp | FAIL |
| Missing observation 0.50 | 67.14% | 83.03% | 43.60% | -23.54 pp | -29.72 pp | FAIL |
| Hard-negative replacement 0.10 | 74.21% | 82.36% | 43.96% | -30.25 pp | -36.16 pp | FAIL |
| Hard-negative replacement 0.20 | 68.96% | 73.38% | 41.38% | -27.59 pp | -33.12 pp | FAIL |
| Hard-negative replacement 0.30 | 62.30% | 64.45% | 38.98% | -23.32 pp | -28.53 pp | FAIL |
| Hard-negative replacement 0.50 | 42.57% | 46.69% | 33.95% | -8.62 pp | -13.12 pp | FAIL |

The pre-frozen rule is applied to every noisy cell: robust H100 must exceed EMA by at least 5 percentage points, and the robust sequence-cluster 95% CI lower bound must not fall below the paired EMA point estimate.

## Required answers

1. **EMA 为什么失败？** Fixed EMA applies the same update fraction to every present observation. Wrong-identity and hard-negative injections therefore move the state toward a competitor; it has no observation-quality decision, while missing observations simply freeze the state.
2. **GRU 为什么提升？** The learned candidate can transform the observation conditioned on the current identity state, and it was trained with positive-versus-visible-competitor loss rather than only self-similarity. The clean N72R18 GRU is the frozen learned baseline; its noisy degradation is part of this test.
3. **Reliability 是否有效？** The frozen robust rule is **not validated**. It beats the clean-trained GRU in 0/12 noisy cells; this count is descriptive, while the decision is against EMA and the frozen CI gate.
4. **错误观察多少比例导致漂移？** Each noise level is a fixed attempted corruption probability; the exact applied and missing counts are recorded in the per-cell summaries and update JSONL. State-change L2, state-change cosine, and anchor drift are reported below.
5. **Missing 后能否恢复？** Recovery is measured as the next observed-frame hard-negative win after a missing update. The per-condition recovery counts and rates are reported below; no future identity is fed into the updater.
6. **Hard negative 是否稳定？** Noise C explicitly replaces observations with the current-frame highest-similarity wrong identity. Its H100 curve and drift are shown as separate cells, so a clean-only gain is not treated as robust evidence.
7. **是否值得接回 InterMOT？** **No. The robust memory gate is not validated, so `next_association_stage_authorized=false`.**

## Memory drift and missing recovery

| Method | Condition | Noise applied / updates | Mean state-change L2 | Mean corrupted-update L2 | Mean missing-update L2 | Mean anchor drift | Next-observed recovery win |
|---|---|---:|---:|---:|---:|---:|---:|
| N72R19 GRU+Reliability | Clean | 0/24810 (0.00%) | 0.030765 | n/a | n/a | 0.627205 | n/a (0 pairs) |
| N72R19 GRU+Reliability | Wrong identity 0.10 | 2522/24810 (10.17%) | 0.031323 | 0.034888 | n/a | 0.625203 | n/a (0 pairs) |
| N72R19 GRU+Reliability | Wrong identity 0.20 | 5046/24810 (20.34%) | 0.031752 | 0.034654 | n/a | 0.623017 | n/a (0 pairs) |
| N72R19 GRU+Reliability | Wrong identity 0.30 | 7529/24810 (30.35%) | 0.032186 | 0.034271 | n/a | 0.622029 | n/a (0 pairs) |
| N72R19 GRU+Reliability | Wrong identity 0.50 | 12487/24810 (50.33%) | 0.032893 | 0.033404 | n/a | 0.621382 | n/a (0 pairs) |
| N72R19 GRU+Reliability | Missing observation 0.10 | 2522/24810 (10.17%) | 0.029179 | n/a | 0.000000 | 0.611959 | 44.53% (2488 pairs) |
| N72R19 GRU+Reliability | Missing observation 0.20 | 5046/24810 (20.34%) | 0.027484 | n/a | 0.000000 | 0.594355 | 43.14% (4975 pairs) |
| N72R19 GRU+Reliability | Missing observation 0.30 | 7529/24810 (30.35%) | 0.025807 | n/a | 0.000000 | 0.574831 | 42.59% (7400 pairs) |
| N72R19 GRU+Reliability | Missing observation 0.50 | 12487/24810 (50.33%) | 0.022245 | n/a | 0.000000 | 0.525135 | 41.12% (12227 pairs) |
| N72R19 GRU+Reliability | Hard-negative replacement 0.10 | 2522/24810 (10.17%) | 0.031079 | 0.032562 | n/a | 0.625429 | n/a (0 pairs) |
| N72R19 GRU+Reliability | Hard-negative replacement 0.20 | 5046/24810 (20.34%) | 0.031248 | 0.032536 | n/a | 0.623794 | n/a (0 pairs) |
| N72R19 GRU+Reliability | Hard-negative replacement 0.30 | 7529/24810 (30.35%) | 0.031428 | 0.032102 | n/a | 0.623075 | n/a (0 pairs) |
| N72R19 GRU+Reliability | Hard-negative replacement 0.50 | 12487/24810 (50.33%) | 0.031579 | 0.031119 | n/a | 0.622795 | n/a (0 pairs) |

## Decision

`ROBUST_LEARNED_MEMORY_NOT_VALIDATED`; `ROBUST_LEARNED_MEMORY_VALIDATED=false`.
`next_association_stage_authorized=false` is recorded for the next stage, but N72R19 stops here and does not execute association integration.

## Provenance

- Final Goal: `outputs/N72R19/FINAL_GOAL.json`
- Frozen protocol: `outputs/N72R19/protocol.json`
- Reference audit: `outputs/N72R19/reference_audit.md`
- Evaluation matrix: `outputs/N72R19/metrics/noise_matrix.json`
- Per-cell records: `outputs/N72R19/noise_results/`
- Memory drift: `outputs/N72R19/memory_drift/`
