# N72R16 Final Report

## Frozen Final Goal

**FINAL GOAL:** Human Identity Representation Probe

**CENTRAL QUESTION:** “用户点一下这个人以后，我们到底能不能认住他？”

Formal question: given one simulated human-confirmed identity observation at time `t`, can the visual identity representation reliably recognize the same person against hard competing identities over future frames?

This report, all N72R16 code, all assets, and all measurements are subordinate to this frozen Goal. The Goal must not be changed to improve MOT, HOTA, a tracker, generic ReID, SAM3, Hungarian matching, or N72R15 restoration.

## Primary endpoint

`VAL_H100_HARD_NEGATIVE_IDENTITY_WIN_RATE` is the primary endpoint. For every competitive future frame:

```text
positive_score = sim(human_anchor, same_identity_future_crop)
hard_negative_score = max_q sim(human_anchor, other_visible_identity_crop_q)
win = positive_score > hard_negative_score
```

The final report must include H20, H50, and H100 win rates, the H100 sequence-cluster 95% confidence interval, and the H100 median positive-minus-hard-negative margin.

## Secondary endpoints

Report Rank-1, MRR, Rank-2, Rank-3, mean/median/P10/P25/P75 margin, and gap buckets 1–5, 6–20, 21–50, and 51–100 frames.

The persistent-memory comparison is a subordinate diagnostic only. Memory-3 and Memory-5 must be labeled `POSTHOC ORACLE POSITIVE MEMORY DIAGNOSTIC` and must not be described as deployable online memory.

## Decision and authorization

The only permitted final decisions are:

- `PASS_STRONG_SINGLE_ANCHOR_IDENTITY`
- `PASS_PERSISTENT_MEMORY_IMPROVES_IDENTITY`
- `AMBIGUOUS_IDENTITY_REPRESENTATION`
- `FAIL_IDENTITY_REPRESENTATION`

`NEXT_ASSOCIATION_STAGE_AUTHORIZED` remains `false` unless one of the first two decisions is actually supported by the frozen validation criteria. This stage does not authorize SAM3 inference, Hungarian redesign, persistent-association integration, requery, TrackEval/full MOT, LoRA, identity-decoder training, or ablations.

## Current status

The Goal is frozen and the probe is complete. No SAM3, tracker, association, TrackEval, training, LoRA, or ablation work was started.

## Protocol and assets

- Official DanceTrack train/val only: 40 train sequences and 25 val sequences.
- Frozen protocol: one earliest eligible GT anchor per identity; at least 20 same-ID observations in the next 100 frames; no embedding score is used for anchor selection.
- Competitive samples: 40,422 train-development and 24,810 val-confirmation frames.
- The downloaded release contains valid zero-based identity IDs. `id=0` is retained as an identity, while negative IDs would be invalid. Original GT files were not rewritten.
- Crops were decoded and resized on the fly. Only float16 512-D embeddings and metadata were cached; no crop images were saved.
- Dataset lineage: `N72R16_NEW_ASSET_LINEAGE`, official DanceTrack Hugging Face fallback `noahcao/dancetrack`, commit `a0ba42ac690c41e9850a20e76a0b9450a6fb6a47`. Baidu direct was attempted first and not used because headless extraction-code verification was unavailable.
- Encoder: public Torchreid OSNet x1.0 Market-1501 checkpoint, input `256x128`, frozen, L2-normalized 512-D output. It is a fresh public dependency and is not claimed equivalent to any historical InterMOT checkpoint.

The full asset manifest, protocol JSON, embedding-store metadata, and raw sample JSONL are outside Git under the configured N72R16 asset root. The tracked Goal state is in `outputs/N72R16/FINAL_GOAL.json`, `outputs/N72R16/stage_status.json`, and `outputs/N72R16/FINAL_RESULT.json`.
After SHA verification and successful extraction, the three source ZIP archives were removed; the manifest retains their historical filenames, sizes, hashes, and source commit.

## Results

The primary validation result is:

| Horizon | Hard-negative win rate | MRR | Rank-1 | Rank-2 | Rank-3 | Mean margin | Median margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| H20 | 0.6120 | 0.7323 | 0.6120 | 0.7337 | 0.8094 | 0.0195 | 0.0170 |
| H50 | 0.4495 | 0.6054 | 0.4495 | 0.5945 | 0.6903 | -0.0091 | -0.0077 |
| H100 | **0.3622** | 0.5348 | 0.3622 | 0.5158 | 0.6185 | -0.0222 | **-0.0206** |

`VAL_H100_HARD_NEGATIVE_IDENTITY_WIN_RATE = 0.3622`, with a 2,000-repetition sequence-cluster 95% CI of `[0.3038, 0.4323]`. The H100 margin percentiles are P10 `-0.1130`, P25 `-0.0661`, and P75 `0.0193`.

Validation time-gap breakdown:

| Gap | Win rate | MRR | Median margin |
|---|---:|---:|---:|
| 1–5 | 0.8486 | 0.9008 | 0.0497 |
| 6–20 | 0.5301 | 0.6739 | 0.0048 |
| 21–50 | 0.3392 | 0.5193 | -0.0227 |
| 51–100 | 0.2697 | 0.4599 | -0.0341 |

The train-development H100 result was win rate `0.3544`, sequence-cluster CI `[0.3066, 0.4131]`, and median margin `-0.0203`; it is consistent with the validation failure and was not used to tune the probe.

## Persistent-memory diagnostic

These are explicitly `POSTHOC ORACLE POSITIVE MEMORY DIAGNOSTIC` results, not deployable online memory:

| Variant | Val H100 samples | Win rate | Sequence-cluster CI | Median margin |
|---|---:|---:|---:|---:|
| M1 single anchor | 24,810 | 0.3622 | [0.3038, 0.4323] | -0.0206 |
| M3 anchor + earliest 2 prior same-ID observations | 24,264 | 0.3698 | [0.3079, 0.4435] | -0.0189 |
| M5 anchor + earliest 4 prior same-ID observations | 23,718 | 0.3693 | [0.3074, 0.4456] | -0.0182 |

M3/M5 do not satisfy the frozen strong gate and do not provide a meaningful enough improvement to authorize the next association stage.

## Final decision

- Decision: `FAIL_IDENTITY_REPRESENTATION`
- Rationale: Validation H100 hard-negative signal is weak or non-positive under the frozen gate.
- `NEXT_ASSOCIATION_STAGE_AUTHORIZED`: `false`
- Primary endpoint: `VAL_H100_HARD_NEGATIVE_IDENTITY_WIN_RATE`
- The N72R16 stop rule is applied. Any association/MOT/SAM3 work requires a separately authorized next stage.
