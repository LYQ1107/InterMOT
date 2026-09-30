Does Selective Identity Memory Update preserve the strong clean memory while reducing corruption damage?

选择性身份记忆更新是否在保留原有身份记忆能力的同时减少了错误观察造成的污染？

# N72R19R1 Final Report

FINAL GOAL: **Selective Identity Memory Update**

Central question: **Can selective observation updates preserve the strong learned identity memory while preventing corrupted observations from damaging future identity recognition?**

最终决策：`FAIL_CLEAN_PRESERVATION`

`SELECTIVE_IDENTITY_MEMORY_VALIDATED=false`；`next_candidate_stream_stage_authorized=false`。

## 1. Protocol and causal boundary

The experiment starts from the strictly loaded, frozen N72R18 GRU identity
memory and an immutable first human anchor. The frozen OSNet 512-D embedding
cache and inherited N72R17 protocol are unchanged. The selector uses only the
current observation, current visible competitors, the current dynamic state,
the immutable human anchor, and causal online bookkeeping.

The fixed protocol has 32 train sequences, 8 dev sequences, and 25 validation
sequences / 273 episodes / 24,810 real competitive observations. One
method-independent corruption manifest is shared by every method. Val was not
used to train, tune the threshold, or choose a seed. The final val summaries
use 2,000 sequence-cluster bootstrap repetitions with seed 72191.

No SAM3 inference, MOT association, Hungarian redesign, TrackEval, HOTA, IDF1,
LoRA, or encoder training was run.

## 2. Preservation gate

The always-accept wrapper reproduces the inherited N72R18 GRU point result:

| method / clean condition | H20 | H50 | H100 | H100 sequence-cluster 95% CI | median margin |
|---|---:|---:|---:|---|---:|
| B2 Frozen N72R18 GRU / clean | 92.795% | 91.584% | **91.358%** | [87.737%, 94.727%] | 0.08430 |

This matches N72R18 and passes the preservation implementation gate.

The dev-frozen proposed rule is B5 Future-Utility Selector, threshold 0.20,
seed 72193 selected by dev robust macro subject to clean preservation. On the
unseen val set:

| proposed B5 / clean | H20 | H50 | H100 | H100 sequence-cluster 95% CI |
|---|---:|---:|---:|---|
| seed 72193 | 91.826% | 90.104% | **89.702%** | [85.940%, 93.391%] |
| three-seed mean ± std | 91.846% ± 0.016% | 90.125% ± 0.053% | **89.558% ± 0.259%** | — |

The selected B5 clean regression is −1.657 percentage points versus B2;
the three-seed mean regression is −1.800 points. Both fail the frozen clean
rule (`H100 >= 90%` or regression no worse than 1.5 points).

## 3. Primary robustness endpoint

`ROBUST_H100_MACRO` is the unweighted mean of Wrong Identity 20%, Wrong
Identity 30%, Hard Negative 20%, and Hard Negative 30% H100 cells.

| method | clean H100 | Wrong20 | Wrong30 | HardNeg20 | HardNeg30 | primary macro |
|---|---:|---:|---:|---:|---:|---:|
| B1 EMA(0.90) | 78.134% | 71.467% | 66.763% | 71.463% | 66.320% | 69.003% |
| B2 Frozen GRU | 91.358% | 73.208% | 64.397% | 73.853% | 64.534% | **68.998%** |
| B3 similarity threshold 0.35 | 77.993% | 55.413% | 47.259% | 54.087% | 46.711% | 50.868% |
| B4 correctness selector, seed 72193 | 90.742% | 70.879% | 62.435% | 71.878% | 62.789% | 66.995% |
| B5 future-utility selector, seed 72193 | 89.702% | 72.688% | 64.486% | 71.475% | 61.733% | 67.596% |
| B5 three-seed mean ± std | 89.558% ± 0.259% | 71.356% ± 1.346% | 63.521% ± 0.897% | 70.082% ± 1.220% | 60.563% ± 0.976% | 66.380% ± 1.105% |

For the selected B5 seed, the paired sequence-cluster delta versus B2 over the
four primary cells is −1.403 percentage points, with 95% CI
[−2.618, −0.203] points. The point delta is negative and the CI lower bound is
not above zero. The robustness success rule therefore also fails.

The 50% stress cells remain weak: selected B5 reaches 46.461% on Wrong50 and
45.107% on HardNeg50. The simple similarity threshold is not sufficient; its
dev-frozen threshold 0.35 has both poor clean preservation and much lower
robust H100.

## 4. Secondary identity metrics

For selected B5 on clean val, H100 rank-1 / rank-2 / rank-3 accuracy is
89.702% / 94.974% / 96.667%, MRR is 0.93460, and the positive-minus-hard-
negative margin is mean 0.09431, median 0.08261, P10 −0.00113, P25 0.03663,
P75 0.14252.

Clean H100 win rate by time gap is:

| gap | 1–5 | 6–20 | 21–50 | 51–100 |
|---|---:|---:|---:|---:|
| B2 Frozen GRU | 94.428% | 92.229% | 90.762% | 91.119% |
| selected B5 | 93.976% | 91.082% | 88.936% | 89.275% |

The loss is therefore not isolated to a single horizon; it persists in the
longer gaps that the memory update is meant to improve.

## 5. Selection and memory-damage diagnostics

At Wrong Identity 20% for selected B5, good-observation acceptance is 89.958%,
bad-observation acceptance is 55.017%, and bad-observation rejection is
44.983%. ROC-AUC is 0.8812 and PR-AUC is 0.9713. At Hard Negative 20%,
good acceptance is 95.104%, but hard-negative acceptance is 92.361%, so only
7.639% of hard negatives are rejected. Hard negatives are the most difficult
competing observations for this selector; wrong-identity replacements are
easier to reject.

On Wrong20, selected B5 memory damage has mean −0.01786, median −0.01298,
P90 0.02887, P95 0.10716, and catastrophic-damage rate 8.155% at the
dev-frozen threshold damage > 0.05. On HardNeg20 the corresponding values are
mean −0.01930, median −0.01798, P90 0.05371, P95 0.10862, and 10.441%.
The selector reduces some damage relative to always-accept B2, but it does so
with a clean-memory loss and still accepts too many hard negatives.

For selected B5 on Wrong20, state drift is mean 0.11093 for clean accepted
updates, 0.29886 for corrupted accepted updates, and approximately zero for
corrupted rejected updates. Human-anchor drift remains a diagnostic reference;
the anchor itself is never updated.

## 6. Correctness versus future utility

The future-utility objective is slightly better than the correctness selector
on the four-cell primary macro at val when averaged over three seeds:

| selector | primary H100 macro mean ± std | clean H100 mean ± std |
|---|---:|---:|
| B4 correctness BCE | 66.010% ± 0.790% | 90.709% ± 0.025% |
| B5 future utility | 66.380% ± 1.105% | 89.558% ± 0.259% |

This is a small robustness advantage accompanied by a clean-preservation
tradeoff, not a validated improvement. It does not justify entering a real
candidate stream.

## 7. Evidence ablation

The A0–A5 mechanism diagnostic was run on the dev split with threshold 0.20;
A1–A4 use the same correctness-supervision ablation protocol and one fixed
seed. A5 is the full 13-feature correctness selector (B4 seed 72193).

| ablation | evidence | dev clean H100 | dev primary macro |
|---|---|---:|---:|
| A0 | Frozen GRU | 98.420% | 73.670% |
| A1 | memory similarity only | 98.420% | 73.660% |
| A2 | + human anchor | 98.420% | 73.670% |
| A3 | + competition margin | 98.420% | 72.625% |
| A4 | + prospective state drift | 98.420% | 66.567% |
| A5 | full observation evidence | 98.400% | 71.281% |

This diagnostic does not support the claim that adding more evidence features
automatically improves the identity endpoint. In particular, the prospective
state-drift feature was not beneficial in this frozen ablation.

## 8. Final decision and stop rule

The formal result is:

`FAIL_CLEAN_PRESERVATION`

The experiment answers the Goal negatively for this selective update design:
the selector can reject some corrupted writes and reduce part of the damage,
but it does not preserve the strong N72R18 clean identity memory on unseen val,
and its primary robust macro is below the frozen GRU baseline. The result is
not evidence for `PASS_SELECTIVE_MEMORY_UPDATE`.

No next candidate-stream, association, SAM3, full MOT, TrackEval, HOTA, IDF1,
LoRA, or additional selector stacking is authorized by this stage.

Supporting artifacts: `outputs/N72R19R1/FINAL_RESULT.json`,
`outputs/N72R19R1/metrics/final_val_all.json`,
`outputs/N72R19R1/metrics/dev_ablations.json`, and
`outputs/N72R19R1/leakage_audit.md`.
