# N72R19R1 data audit

FINAL GOAL: Selective Identity Memory Update

所有统计基于冻结的 N72R17 protocol 和 N72R16/N72R17 embedding store；`synthetic_corruption_instances` 不计入真实数据量。

| split | sequences | eligible identities | episodes | real observations | competitive frames | mean obs/identity | median | P10/P25/P75/P90 | synthetic corruption instances |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| train_32_sequences | 32 | 365 | 365 | 35421 | 35421 | 97.04 | 100.00 | 90.40/100.00/100.00/100.00 | 117065 |
| dev_8_sequences | 8 | 52 | 52 | 5001 | 5001 | 96.17 | 100.00 | 90.60/99.50/100.00/100.00 | 16659 |
| val_25_sequences | 25 | 273 | 273 | 24810 | 24810 | 90.88 | 100.00 | 65.40/92.00/100.00/100.00 | 81713 |

`competitors_per_frame` 的完整 histogram 和 percentile 位于 `data_audit.json`。
Train-dev 是 sequence-level 32/8；val 的 25 个 sequence 只用于最终冻结评估。
