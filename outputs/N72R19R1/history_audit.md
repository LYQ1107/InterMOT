# N72R19R1 history audit

FINAL GOAL:
Selective Identity Memory Update

CENTRAL QUESTION:
“有选择地更新身份记忆，能否在保留已有长期身份识别能力的同时，避免错误观察污染身份状态？”

本阶段继承的是 N72R18 的成功 learned memory，而不是 N72R19 从头训练的失败 updater。所有结论仍限定在冻结 OSNet 512-D embedding、冻结 protocol、离线 causal replay；不代表已经完成 online candidate association。

## Inherited evidence

| Stage | Evidence | Result | Interpretation |
|---|---|---:|---|
| N72R16 | Single Anchor | H100 36.22% | 一次固定 embedding 不能作为可靠的永久身份状态。 |
| N72R17 | EMA(0.90) | H100 78.13% | 连续更新显著优于单 anchor。 |
| N72R18 | Frozen clean-trained GRU | H100 91.36% | learned identity memory 在 clean causal replay 中有效。 |
| N72R18 | GRU + Reliability Gate | H100 89.60% | gate 不是 GRU 本身的改进，保留为历史诊断。 |
| N72R19 | newly trained robust GRU + Reliability | Clean H100 46.43% | 从头初始化的 robust updater 没有保留 N72R18 memory。 |

N72R19 的正式结论是 `ROBUST_LEARNED_MEMORY_NOT_VALIDATED`。这不等价于 “Learned Identity Memory 失败”。它证明的是：在没有严格继承 `outputs/N72R18/checkpoints/identity_memory_gru.pt` 的情况下，从头训练的 robustness updater 破坏了已有身份记忆能力。因此 N72R19R1 的 base 必须是严格加载并冻结的 N72R18 GRU。

## Provenance

- N72R16: `outputs/N72R16/FINAL_RESULT.json`
- N72R17: `outputs/N72R17/FINAL_RESULT.json`
- N72R18: `docs/N72R18_FINAL_REPORT.md`, `outputs/N72R18/FINAL_RESULT.json`
- N72R19: `docs/N72R19_FINAL_REPORT.md`, `outputs/N72R19/FINAL_RESULT.json`
- Frozen base checkpoint: `outputs/N72R18/checkpoints/identity_memory_gru.pt`
- Frozen train protocol: `/data3/liuyeqiang/InterMOT_N72R17_assets/outputs/N72R17/protocol_train.json`
- Frozen validation protocol: `/data3/liuyeqiang/InterMOT_N72R17_assets/outputs/N72R17/protocol_val.json`

## R1 boundary

R1 studies only `Frozen Identity Memory + Observation Selection + Selective Memory Update`. It does not change the encoder or GRU and does not start SAM3, MOT association, TrackEval, HOTA, IDF1, LoRA, or full-MOT work.
