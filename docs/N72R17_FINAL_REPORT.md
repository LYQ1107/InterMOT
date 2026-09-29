FINAL GOAL:
Human Identity Representation Probe

CENTRAL QUESTION:
“用户点一下这个人以后，我们到底能不能认住他？”

# InterMOT N72R17 — Identity Representation Research Phase

## Scientific question

**Can one human-confirmed identity observation reliably recognize the same person against hard competing identities over future frames?**

N72R17 inherits the N72R16 protocol byte-for-byte: the earliest score-blind eligible human anchor, future gaps through H100, and the same-frame hard negative defined as the most similar competing identity. The primary endpoint remains `VAL_H100_HARD_NEGATIVE_IDENTITY_WIN_RATE`.

## Scope and frozen controls

- No SAM3 inference, MOT association, Hungarian redesign, TrackEval, requery, LoRA, identity-decoder training, or full MOT evaluation was run.
- OSNet x1.0 / Market1501 is the reused N72R16 baseline embedding store.
- The new encoder is frozen public `timm/vit_base_patch32_clip_224.openai`, used as a generic CLIP visual representation. It was not fine-tuned for DanceTrack or ReID.
- The public OPA067/ReID reference pipeline was reviewed, but no compatible trained ReID checkpoint was available; this report must not be read as a CLIP-ReID fine-tuning result.
- Memory-3/5 and EMA/attention are offline diagnostics using future same-identity GT crops causally after each score; they are not deployable online memory claims.
- Video identity encoder: `SKIPPED`. Reason: No compatible public frozen Video-ReID checkpoint and inference stack was already present; adding an unverified implementation would break comparability and scope..
- Only embeddings, metadata, JSONL records, and reports were stored; no crops were written.

## Stage A — encoder benchmark

| Encoder/variant | Split | H20 win | H50 win | H100 win | H100 CI95 | Median margin | MRR | Rank-1 | Rank-2 | Rank-3 |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `osnet_x1_0_market1501` / Single Anchor | train | 61.44% | 45.47% | 35.44% | [30.47%, 41.35%] | -0.020287 | 0.516138 | 35.44% | 49.04% | 59.42% |
| `osnet_x1_0_market1501` / Single Anchor | val | 61.20% | 44.95% | 36.22% | [30.27%, 43.38%] | -0.020571 | 0.534829 | 36.22% | 51.58% | 61.85% |
| `openai_clip_vit_b32_zero_shot` / Single Anchor | train | 50.66% | 36.48% | 28.80% | [24.12%, 34.33%] | -0.020031 | 0.460291 | 28.80% | 42.81% | 52.94% |
| `openai_clip_vit_b32_zero_shot` / Single Anchor | val | 52.06% | 38.11% | 31.02% | [25.63%, 37.57%] | -0.018926 | 0.489671 | 31.02% | 46.24% | 56.89% |

## Stage B — memory diagnostics

| Encoder/variant | Split | H20 win | H50 win | H100 win | H100 CI95 | Median margin | MRR | Rank-1 | Rank-2 | Rank-3 |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `osnet_x1_0_market1501` / ATTENTION_FIXED_ANCHOR_QUERY | train | 71.34% | 57.10% | 46.98% | [42.12%, 53.47%] | -0.003283 | 0.614502 | 46.98% | 60.65% | 69.61% |
| `osnet_x1_0_market1501` / EMA_0.90 | train | 80.06% | 78.74% | 77.60% | [71.73%, 83.38%] | 0.026893 | 0.854422 | 77.60% | 87.47% | 91.95% |
| `osnet_x1_0_market1501` / EMA_0.95 | train | 72.12% | 67.70% | 66.90% | [60.37%, 73.79%] | 0.015732 | 0.778402 | 66.90% | 79.79% | 86.25% |
| `osnet_x1_0_market1501` / EMA_0.99 | train | 63.71% | 50.72% | 44.33% | [38.51%, 51.43%] | -0.006186 | 0.597030 | 44.33% | 58.86% | 68.58% |
| `osnet_x1_0_market1501` / M1 | train | 61.44% | 45.47% | 35.44% | [30.47%, 41.35%] | -0.020287 | 0.516138 | 35.44% | 49.04% | 59.42% |
| `osnet_x1_0_market1501` / M3 | train | 61.04% | 45.16% | 35.47% | [30.49%, 41.39%] | -0.019433 | 0.518242 | 35.47% | 49.56% | 59.73% |
| `osnet_x1_0_market1501` / M5 | train | 60.26% | 44.76% | 35.31% | [30.46%, 41.43%] | -0.018756 | 0.518108 | 35.31% | 49.62% | 59.81% |
| `osnet_x1_0_market1501` / ATTENTION_FIXED_ANCHOR_QUERY | val | 70.71% | 55.95% | 47.65% | [40.93%, 56.08%] | -0.003067 | 0.625571 | 47.65% | 61.85% | 70.73% |
| `osnet_x1_0_market1501` / EMA_0.90 | val | 80.79% | 78.63% | 78.13% | [72.70%, 84.35%] | 0.029427 | 0.859113 | 78.13% | 88.06% | 92.30% |
| `osnet_x1_0_market1501` / EMA_0.95 | val | 72.05% | 68.48% | 67.46% | [60.34%, 75.82%] | 0.017812 | 0.783280 | 67.46% | 80.23% | 86.44% |
| `osnet_x1_0_market1501` / EMA_0.99 | val | 63.20% | 50.72% | 46.00% | [38.72%, 54.81%] | -0.004684 | 0.616052 | 46.00% | 61.00% | 70.60% |
| `osnet_x1_0_market1501` / M1 | val | 61.20% | 44.95% | 36.22% | [30.27%, 43.38%] | -0.020571 | 0.534829 | 36.22% | 51.58% | 61.85% |
| `osnet_x1_0_market1501` / M3 | val | 62.25% | 45.33% | 36.98% | [30.62%, 44.58%] | -0.018917 | 0.541584 | 36.98% | 52.21% | 62.69% |
| `osnet_x1_0_market1501` / M5 | val | 61.95% | 44.89% | 36.93% | [30.51%, 44.72%] | -0.018165 | 0.542521 | 36.93% | 52.67% | 63.07% |
| `openai_clip_vit_b32_zero_shot` / ATTENTION_FIXED_ANCHOR_QUERY | train | 64.30% | 50.65% | 42.38% | [36.39%, 49.40%] | -0.005149 | 0.576266 | 42.38% | 56.33% | 65.79% |
| `openai_clip_vit_b32_zero_shot` / EMA_0.90 | train | 69.61% | 66.48% | 65.85% | [58.87%, 73.09%] | 0.009344 | 0.765264 | 65.85% | 78.06% | 84.24% |
| `openai_clip_vit_b32_zero_shot` / EMA_0.95 | train | 60.98% | 55.39% | 55.24% | [48.24%, 62.84%] | 0.003142 | 0.685021 | 55.24% | 69.29% | 77.34% |
| `openai_clip_vit_b32_zero_shot` / EMA_0.99 | train | 52.84% | 40.65% | 35.94% | [30.31%, 42.44%] | -0.010214 | 0.525896 | 35.94% | 50.74% | 60.99% |
| `openai_clip_vit_b32_zero_shot` / M1 | train | 50.66% | 36.48% | 28.80% | [24.12%, 34.33%] | -0.020031 | 0.460291 | 28.80% | 42.81% | 52.94% |
| `openai_clip_vit_b32_zero_shot` / M3 | train | 50.83% | 37.21% | 29.66% | [25.00%, 35.07%] | -0.017968 | 0.467906 | 29.66% | 43.62% | 54.03% |
| `openai_clip_vit_b32_zero_shot` / M5 | train | 51.24% | 37.13% | 29.63% | [25.07%, 35.03%] | -0.017189 | 0.469145 | 29.63% | 43.80% | 54.41% |
| `openai_clip_vit_b32_zero_shot` / ATTENTION_FIXED_ANCHOR_QUERY | val | 65.37% | 52.48% | 44.78% | [38.22%, 52.52%] | -0.003883 | 0.602743 | 44.78% | 59.11% | 68.54% |
| `openai_clip_vit_b32_zero_shot` / EMA_0.90 | val | 70.02% | 68.72% | 68.36% | [62.49%, 75.18%] | 0.010897 | 0.790017 | 68.36% | 80.76% | 87.05% |
| `openai_clip_vit_b32_zero_shot` / EMA_0.95 | val | 61.84% | 58.65% | 57.61% | [50.91%, 65.53%] | 0.004439 | 0.711024 | 57.61% | 72.36% | 80.48% |
| `openai_clip_vit_b32_zero_shot` / EMA_0.99 | val | 54.10% | 42.19% | 38.00% | [31.50%, 45.70%] | -0.009242 | 0.551133 | 38.00% | 53.37% | 63.95% |
| `openai_clip_vit_b32_zero_shot` / M1 | val | 52.06% | 38.11% | 31.02% | [25.63%, 37.57%] | -0.018926 | 0.489671 | 31.02% | 46.24% | 56.89% |
| `openai_clip_vit_b32_zero_shot` / M3 | val | 52.81% | 38.02% | 31.38% | [25.70%, 38.25%] | -0.017299 | 0.492274 | 31.38% | 46.27% | 57.15% |
| `openai_clip_vit_b32_zero_shot` / M5 | val | 51.79% | 37.22% | 31.10% | [25.15%, 38.20%] | -0.017090 | 0.490430 | 31.10% | 46.29% | 56.94% |

## Required secondary analysis

The validation single-anchor time-gap breakdown is included below for both frozen encoders, with the best measured memory diagnostic shown afterward.

### OSNet single anchor — validation

| Time gap | samples | win rate | mean margin | median margin | P10 | P25 | P75 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1–5 | 1328 | 84.86% | 0.060417 | 0.049747 | -0.015102 | 0.017628 | 0.099908 |
| 6–20 | 3835 | 53.01% | 0.005395 | 0.004794 | -0.083647 | -0.035936 | 0.045842 |
| 21–50 | 7610 | 33.92% | -0.028507 | -0.022697 | -0.117873 | -0.068570 | 0.013066 |
| 51–100 | 12037 | 26.97% | -0.036130 | -0.034073 | -0.120002 | -0.076774 | 0.003456 |

### OpenAI CLIP ViT-B/32 — validation

| Time gap | samples | win rate | mean margin | median margin | P10 | P25 | P75 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1–5 | 1328 | 76.28% | 0.024745 | 0.024848 | -0.022666 | 0.001371 | 0.046949 |
| 6–20 | 3835 | 43.68% | -0.007053 | -0.005547 | -0.062250 | -0.031675 | 0.018435 |
| 21–50 | 7610 | 28.65% | -0.028991 | -0.021795 | -0.094114 | -0.055677 | 0.003701 |
| 51–100 | 12037 | 23.49% | -0.032050 | -0.025780 | -0.090986 | -0.056364 | -0.001678 |

### Best memory diagnostic — `osnet_x1_0_market1501:EMA_0.90` validation

| Time gap | samples | win rate | mean margin | median margin | P10 | P25 | P75 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1–5 | 1328 | 89.31% | 0.067592 | 0.053452 | -0.001456 | 0.024381 | 0.103074 |
| 6–20 | 3835 | 77.84% | 0.039514 | 0.030602 | -0.021842 | 0.003351 | 0.067525 |
| 21–50 | 7610 | 77.17% | 0.039204 | 0.030627 | -0.020650 | 0.002879 | 0.067260 |
| 51–100 | 12037 | 77.60% | 0.039223 | 0.026220 | -0.019357 | 0.002836 | 0.059185 |

## Decision

`TEMPORAL_MEMORY_PROMISING`

A causal frozen memory diagnostic materially improves H100 identity wins and retains a positive sequence-cluster lower bound, warranting a learned human-conditioned memory stage.

- Validation OSNet H100 hard-negative win rate: **36.22%**.
- Validation CLIP H100 hard-negative win rate: **31.02%**; delta over OSNet: **-5.20%**.
- Best memory diagnostic: **osnet_x1_0_market1501:EMA_0.90**, H100 **78.13%**, delta over its same-encoder single anchor **41.91%**.
- Predeclared gates: encoder-limited requires CLIP H100 ≥50%, improvement ≥10 percentage points, and sequence-cluster CI lower >40%; memory-promising requires the analogous memory improvement ≥5 points plus H100 ≥50% and CI lower >40%.

## Authorization and stop rule

- `NEXT_ASSOCIATION_STAGE_AUTHORIZED`: **false**.
- `NEXT_HUMAN_CONDITIONED_IDENTITY_STAGE_AUTHORIZED`: **true**.
- N72R17 stops here. No downstream SAM3/MOT/association or Stage C training starts automatically.

## Provenance

- Asset manifest: `/data3/liuyeqiang/InterMOT_N72R17_assets/outputs/N72R17/asset_manifest.json`
- Aggregate: `/data3/liuyeqiang/InterMOT_N72R17_assets/outputs/N72R17/aggregate.json`
- Protocol manifest: `/data3/liuyeqiang/InterMOT_N72R17_assets/outputs/N72R17/protocol.json`
- Frozen Final Goal: `/data3/liuyeqiang/InterMOT_N72R17_assets/outputs/N72R17/FINAL_GOAL.json`
- Final result: `/data3/liuyeqiang/InterMOT_N72R17_assets/outputs/N72R17/FINAL_RESULT.json`
