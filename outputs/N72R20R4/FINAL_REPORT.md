# N72R20R4 Final Report

## Q1–Q5: identity-to-trajectory transfer

FINAL GOAL: Causal Identity-to-Trajectory Transfer: Globally Consistent Online Association for Long-Term MOT.

Frozen scientific specification: `FINAL_GOAL.json`. Final decision: `FAIL_GLOBAL_ASSOCIATION_AUTHORITY`.

Q1: The historical 90 local corrections did not enter the chosen G0 treatment in any of the eight folds. The old path also discarded non-target Hungarian reallocations and independently re-read the sealed next-frame state. Its IoU-based shadow N01 was never an IDSW count. The new implementation audit and complete causal intervention table separate these quantities.

Q2: Real R3R2 Adapter ensemble score API was called 8414 times in the no-write outer comparison. Full-sequence HOTA=0.5808919181126544, AssA=0.5433304753115834; compare causal baseline HOTA=0.5808919181126544, AssA=0.5433304753115834. The API counter includes empty-candidate calls: 8329 calls have real candidates, giving 24987 actual query-tower forwards across three model seeds.

Q3: Independent legacy reconstruction exactly reproduces all eight native trajectory SHA and TrackEval results. Dynamic runtime and identity-off pass exact A/A. Dynamic births/deaths change the baseline; native HOTA=0.5824390062628039 and causal HOTA=0.5808919181126544. Selected treatment ΔHOTA=0.0, ΔAssA=0.0. The historical/lifecycle difference is not attributed to identity learning. Compared directly with the historical frozen COMBINED replay, new causal baseline ΔHOTA=-0.0016438659980867465, ΔAssA=0.0013768874641366668, ΔDetA=-0.005160132046555965, ΔIDSW=33.0; macro ΔHOTA CI=[-0.019871759419963834, 0.012483285691329485]. This comparison includes lifecycle/export changes and does not isolate a learned-identity effect; it does not show a better global tracker.

Q4 — Representation: full-future competitive-candidate win rate is 0.23524950705293493 for strict Adapter/P0, 0.2737752161383285 for raw ReID, and 0.6896708630365539 for Adapter/P1. These are offline candidate diagnostics, not global MOT metrics or clean GT-replay equivalents. Other real candidates, including unmatched detections, are competitors.

Q4 — Authority: the exact global-margin certificate blocks 7451/7452 assigned target frames even for the strongest preregistered fixed weight. Positive label count is 0; this underpowered counterfactual intervention family cannot establish superiority of structured authority over simpler models. No stronger heldout-informed weight is retroactively presented as an unbiased LOSO result.

Q4 — Global conflict and memory: observed global collisions=0. Absence of harmful overrides is not evidence of safe useful overrides when the assignments never change. P1 accepted writes=7452, wrong-write rate=0.3455448201825013, retention=1.0; P2 accepted writes=684, wrong-write rate=0.2412280701754386, retention=0.1064178798441665. Actual GRU query hashes evolve independently of the ordinary tracker-state hash; their evolution does not by itself imply correct identity storage or trajectory improvement.

Q4 — Candidate quality and detection: real target candidate coverage given visibility=0.7984261501210653. The single-target posthoc oracle produces ΔHOTA=0.015008329212410909, ΔAssA=0.04245455959476152, ΔDetA=-0.01514537522862136; it is not a guaranteed global upper bound because competing ownership is coupled. Official treatment ΔDetA=0.0 is separated from identity ranking. Cached candidates cannot recover a person for whom no valid candidate exists.

Q4 — Domain shift: historical GT-replay representation evidence does not establish performance on real SAM3 boxes over a whole trajectory. This stage measures that operational shift but does not isolate it from crop quality, time gap and contaminated observation histories. With no trajectory interventions, this run cannot identify the best stronger controller or prove that identity evidence is intrinsically incapable of improving MOT.

Q5: All 25 previously accessed VAL sequences were evaluated with a policy frozen from train-development inner selection; no VAL tuning. VAL ΔHOTA=0.0, ΔAssA=0.0, ΔDetA=0.0, ΔIDF1=0.0, ΔIDSW=0.0. Development gate=False, generalization gate=False.

## Table A — full-sequence Global MOT

| Method | HOTA | AssA | DetA | LocA | IDF1 | MOTA | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Native Baseline | 0.5824390062628039 | 0.5418544920565236 | 0.628156812640047 | 0.902063824384534 | 0.6232634541611995 | 0.5940227453054747 | 217.0 | 7622.0 | 13651.0 |
| B1_CAUSAL_BASELINE | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| B2_ADAPTER_NO_WRITES | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| B3_CONSENSUS_MEMORY | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| B4_FIXED_INNER_SELECTED | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| B5_SCALAR | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| B6_MLP | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| B7_STRUCTURED | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| B8_SHUFFLED_STATE | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| G1_FIXED_0.1 | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| G1_FIXED_0.25 | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| G1_FIXED_0.5 | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| G1_FIXED_1 | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| HISTORICAL_R3R2_ADAPTER7 | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| LEGACY_BASE_SCORE | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| LOGISTIC_GATE | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| MEMORY_P2_RELIABLE | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| R2_LOW_CONFIDENCE | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| R2_RECOVERY | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| R4_CONFLICT_GUARD | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| R6_PERSISTENCE | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| RAW_REID | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| SEED_LOGISTIC_720321 | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| SEED_LOGISTIC_720322 | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| SEED_LOGISTIC_720323 | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| SEED_MLP_720321 | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| SEED_MLP_720322 | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| SEED_MLP_720323 | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| SEED_SCALAR_720321 | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| SEED_SCALAR_720322 | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| SEED_SCALAR_720323 | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| SEED_STRUCTURED_720321 | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| SEED_STRUCTURED_720322 | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| SEED_STRUCTURED_720323 | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| SELECTED_TREATMENT | 0.5808919181126544 | 0.5433304753115834 | 0.6230911682274578 | 0.9010765894985205 | 0.6175494256925472 | 0.5785506479767257 | 250.0 | 9157.0 | 12902.0 |
| VAL Baseline | 0.4878651520620449 | 0.5118588672229311 | 0.46917169533441927 | 0.8609944576195884 | 0.5211827253117807 | 0.343289747188516 | 1832.0 | 54861.0 | 91164.0 |
| VAL Treatment | 0.4878651520620449 | 0.5118588672229311 | 0.46917169533441927 | 0.8609944576195884 | 0.5211827253117807 | 0.343289747188516 | 1832.0 | 54861.0 | 91164.0 |

## Table B — actual trajectory interventions

| Method | Changed frames | N01 | N10 | Global conflicts | Persistent corrections H30 | ΔHOTA |
|---|---:|---:|---:|---:|---:|---:|
| B1_CAUSAL_BASELINE | 0 | 0 | 0 | 0 | 0 | 0.0 |
| B2_ADAPTER_NO_WRITES | 0 | 0 | 0 | 0 | 0 | 0.0 |
| B3_CONSENSUS_MEMORY | 0 | 0 | 0 | 0 | 0 | 0.0 |
| B4_FIXED_INNER_SELECTED | 0 | 0 | 0 | 0 | 0 | 0.0 |
| B5_SCALAR | 0 | 0 | 0 | 0 | 0 | 0.0 |
| B6_MLP | 0 | 0 | 0 | 0 | 0 | 0.0 |
| B7_STRUCTURED | 0 | 0 | 0 | 0 | 0 | 0.0 |
| B8_SHUFFLED_STATE | 0 | 0 | 0 | 0 | 0 | 0.0 |
| G1_FIXED_0.1 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| G1_FIXED_0.25 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| G1_FIXED_0.5 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| G1_FIXED_1 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| HISTORICAL_R3R2_ADAPTER7 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| LEGACY_BASE_SCORE | 0 | 0 | 0 | 0 | 0 | 0.0 |
| LOGISTIC_GATE | 0 | 0 | 0 | 0 | 0 | 0.0 |
| MEMORY_P2_RELIABLE | 0 | 0 | 0 | 0 | 0 | 0.0 |
| R2_LOW_CONFIDENCE | 0 | 0 | 0 | 0 | 0 | 0.0 |
| R2_RECOVERY | 0 | 0 | 0 | 0 | 0 | 0.0 |
| R4_CONFLICT_GUARD | 0 | 0 | 0 | 0 | 0 | 0.0 |
| R6_PERSISTENCE | 0 | 0 | 0 | 0 | 0 | 0.0 |
| RAW_REID | 0 | 0 | 0 | 0 | 0 | 0.0 |
| SEED_LOGISTIC_720321 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| SEED_LOGISTIC_720322 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| SEED_LOGISTIC_720323 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| SEED_MLP_720321 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| SEED_MLP_720322 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| SEED_MLP_720323 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| SEED_SCALAR_720321 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| SEED_SCALAR_720322 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| SEED_SCALAR_720323 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| SEED_STRUCTURED_720321 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| SEED_STRUCTURED_720322 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| SEED_STRUCTURED_720323 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| SELECTED_TREATMENT | 0 | 0 | 0 | 0 | 0 | 0.0 |

## Table C — real frozen GRU writes

| Policy | Accepted | Wrong rate | Correct retention | HOTA | Safety pass |
|---|---:|---:|---:|---:|---|
| P0 | 0 | None | 0.0 | 0.5808919181126544 | False |
| P1 | 7452 | 0.3455448201825013 | 1.0 | 0.5808919181126544 | False |
| P2 | 684 | 0.2412280701754386 | 0.1064178798441665 | 0.5808919181126544 | False |
| SELECTED | 0 | None | 0.0 | 0.5808919181126544 | False |

Wrong-write denominator is accepted writes. Correct-write retention denominator is eligible assigned correct observations. First wrong writes, accepted-write cascade length and per-sequence state divergence are in the machine artifacts.

## Protocol, uncertainty and resource limitations

- 8-fold LOSO uses six fit sequences and cyclic inner validation; these sequences are a repeatedly used development benchmark. The three training seeds are 720321/720322/720323.
- Historic seven-sequence Adapter checkpoint metadata discrepancy is documented. Strict six-sequence models are used for inner selection; historical SHA-verified models are an outer-only inference comparison.
- Paired 2000-draw sequence-cluster bootstrap reports sequence-macro Δ CI; combined official TrackEval Δ is a different estimator and reported separately.
- Candidate pipeline is reused SAM3/OSNet. No new candidates, dataset downloads or backbone training. Simulated GT-box human anchors are not real-human evidence; future runtime GT is forbidden.
- M5 local metrics use merged intervention windows; overlapping corrections do not provide independent additive effect estimates. Single-target oracle headroom is diagnostic only.
- Raw trajectories/checkpoints are external; committed manifests bind their hashes and code/config lineage.
- Source refresh and Git delivery statuses are recorded separately; successful scientific execution is not sufficient to mark overall Goal complete.


## Paired uncertainty

DEV ΔHOTA=0.0; sequence-macro 95% CI=[0.0, 0.0]; CI lower > 0=False.
VAL ΔHOTA=0.0; sequence-macro 95% CI=[0.0, 0.0]; CI lower > 0=False.

## All 25 VAL sequences — no tuning

| Sequence | Baseline HOTA | Treatment HOTA | ΔHOTA | ΔAssA | ΔDetA | ΔIDF1 | ΔIDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| dancetrack0004 | 0.4086781316369378 | 0.4086781316369378 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0005 | 0.8752005444681449 | 0.8752005444681449 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0007 | 0.7010404996808832 | 0.7010404996808832 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0010 | 0.8528941002306901 | 0.8528941002306901 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0014 | 0.328325932204303 | 0.328325932204303 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0018 | 0.8070064515636208 | 0.8070064515636208 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0019 | 0.08701653765407548 | 0.08701653765407548 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0025 | 0.6847161467577001 | 0.6847161467577001 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0026 | 0.047057666635629436 | 0.047057666635629436 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0030 | 0.7776434330838042 | 0.7776434330838042 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0034 | 0.25477211361906354 | 0.25477211361906354 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0035 | 0.3803120705286087 | 0.3803120705286087 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0041 | 0.11934440836853906 | 0.11934440836853906 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0043 | 0.41969548596002637 | 0.41969548596002637 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0047 | 0.38359884195671584 | 0.38359884195671584 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0058 | 0.7635526822990695 | 0.7635526822990695 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0063 | 0.3205521655841224 | 0.3205521655841224 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0065 | 0.9029531643798206 | 0.9029531643798206 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0073 | 0.17865287024882082 | 0.17865287024882082 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0077 | 0.7184054466830901 | 0.7184054466830901 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0079 | 0.5384920137012219 | 0.5384920137012219 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0081 | 0.36738455594013203 | 0.36738455594013203 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0090 | 0.3069499312471537 | 0.3069499312471537 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0094 | 0.24103623720607595 | 0.24103623720607595 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| dancetrack0097 | 0.873294069795949 | 0.873294069795949 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Cached-feature runtime (not live SAM3/OSNet FPS)

Timing includes causal association and stateless Adapter projection, excludes disk loading, backbone inference, export and evaluator. CPU timing noise can make small incremental times negative.

| Method | Cached-feature FPS | ms/frame | Incremental ms/frame vs causal baseline |
|---|---:|---:|---:|
| B1_CAUSAL_BASELINE | 381.618540120839 | 2.6204177597958194 | 0.0 |
| B2_ADAPTER_NO_WRITES | 248.5852429844435 | 4.022764939681395 | 1.4023471798855762 |
| B3_CONSENSUS_MEMORY | 181.43168461515188 | 5.511716446447453 | 2.8912986866516333 |
| B4_FIXED_INNER_SELECTED | 244.22205705789972 | 4.094634252314572 | 1.4742164925187529 |
| B5_SCALAR | 236.874822631756 | 4.22163904500139 | 1.6012212852055707 |
| B6_MLP | 218.55342891646842 | 4.575540200662796 | 1.9551224408669767 |
| B7_STRUCTURED | 218.9970359886385 | 4.5662718469481005 | 1.9458540871522811 |
| B8_SHUFFLED_STATE | 235.44125648378775 | 4.24734396568623 | 1.626926205890411 |
| G1_FIXED_0.1 | 245.74635708348663 | 4.069236312871459 | 1.4488185530756392 |
| G1_FIXED_0.25 | 241.2825575290376 | 4.1445184029917 | 1.5241006431958812 |
| G1_FIXED_0.5 | 242.77854303444835 | 4.118980151627765 | 1.498562391831946 |
| G1_FIXED_1 | 250.85359002050095 | 3.9863890324163798 | 1.3659712726205602 |
| HISTORICAL_R3R2_ADAPTER7 | 245.47864739395808 | 4.0736740674440135 | 1.453256307648194 |
| LEGACY_BASE_SCORE | 370.10127679129175 | 2.7019631185004584 | 0.0815453587046388 |
| LOGISTIC_GATE | 229.87189173259432 | 4.350249142958641 | 1.7298313831628216 |
| MEMORY_P2_RELIABLE | 236.94919413429724 | 4.220313994540211 | 1.599896234744392 |
| R2_LOW_CONFIDENCE | 250.01022545369923 | 3.9998363994323722 | 1.3794186396365529 |
| R2_RECOVERY | 245.56959448938923 | 4.072165375682163 | 1.4517476158863432 |
| R4_CONFLICT_GUARD | 239.00574812266987 | 4.183999790192281 | 1.5635820303964623 |
| R6_PERSISTENCE | 239.71343179495688 | 4.171647756707133 | 1.551229996911313 |
| RAW_REID | 365.7232633047115 | 2.7343078779400067 | 0.11389011814418729 |
| SEED_LOGISTIC_720321 | 246.1256868754084 | 4.062964791262162 | 1.4425470314663427 |
| SEED_LOGISTIC_720322 | 240.84516995148144 | 4.152045067797919 | 1.5316273080020992 |
| SEED_LOGISTIC_720323 | 242.60441726654915 | 4.121936489315037 | 1.5015187295192178 |
| SEED_MLP_720321 | 242.021592464664 | 4.131862739255397 | 1.511444979459578 |
| SEED_MLP_720322 | 232.8779958410647 | 4.294093979933094 | 1.6736762201372748 |
| SEED_MLP_720323 | 247.37368274286703 | 4.042467205533143 | 1.4220494457373236 |
| SEED_SCALAR_720321 | 241.58468572781334 | 4.13933522726135 | 1.5189174674655306 |
| SEED_SCALAR_720322 | 234.75913575745625 | 4.259685131202562 | 1.639267371406742 |
| SEED_SCALAR_720323 | 234.30922012996774 | 4.267864488837935 | 1.6474467290421158 |
| SEED_STRUCTURED_720321 | 238.91901105360984 | 4.185518747922554 | 1.565100988126735 |
| SEED_STRUCTURED_720322 | 247.82985818255008 | 4.035026317383459 | 1.41460855758764 |
| SEED_STRUCTURED_720323 | 238.54490841141651 | 4.192082768227892 | 1.571665008432073 |
| SELECTED_TREATMENT | 384.91755864522634 | 2.5979589071479263 | -0.02245885264789313 |
| VAL_BASELINE | 274.031938830045 | 3.6492096661046554 | 0.0 |
| VAL_TREATMENT | 270.46237801844825 | 3.6973719129682054 | 0.04816224686354995 |

## Target-conditioned continuity (distinct from Global MOT)

Missing intervals count visible-target frames with no correct clicked-ID output; reacquisitions count correct returns after such a gap. Identity continuity changes use consecutive outputs matched to a GT identity. These diagnostics do not replace TrackEval IDSW.

| Dev sequence | Baseline target recall | Selected target recall | Matched identity changes | Reacquisitions | Wrong identity takeover frames | Persistent H30 corrections |
|---|---:|---:|---:|---:|---:|---:|
| dancetrack0001 | 0.93974175035868 | 0.93974175035868 | 4 | 25 | 4 | 0 |
| dancetrack0002 | 0.22807017543859648 | 0.22807017543859648 | 15 | 40 | 600 | 0 |
| dancetrack0023 | 0.6684894053315106 | 0.6684894053315106 | 25 | 75 | 146 | 0 |
| dancetrack0024 | 0.9342105263157895 | 0.9342105263157895 | 2 | 18 | 3 | 0 |
| dancetrack0039 | 0.9833472106577852 | 0.9833472106577852 | 6 | 10 | 7 | 0 |
| dancetrack0057 | 0.9625407166123778 | 0.9625407166123778 | 9 | 14 | 9 | 0 |
| dancetrack0062 | 0.23192019950124687 | 0.23192019950124687 | 2 | 24 | 693 | 0 |
| dancetrack0072 | 0.19240953221535745 | 0.19240953221535745 | 5 | 41 | 132 | 0 |

## Verification and checkpoints

Stage suite: 31 passed, 0 failed. Selected dependency regression: 74 passed, 1 failed (unchanged historical branch-name assertion). Full-repository PASS is not claimed.
SHA verified 322 dev/VAL trajectory+trace exports; all causal runtime core hashes match the pre-formal freeze. Source ref verified through GitHub connector and matches the initial source HEAD; direct transport failures and verification timing are in SOURCE_LINEAGE.json.
Official TrackEval re-evaluation of all frozen dev ablations and both all-25 VAL streams exactly reproduces every combined/per-sequence metric. Original evaluator logs remain unchanged; this is reproducibility verification, not policy reselection.

Frozen GRU: `/data3/liuyeqiang/InterMOT/outputs/N72R18/checkpoints/identity_memory_gru.pt`, SHA256 `94ba9c44e177b254ee54984195e839f37c78a648c010ffc723dd3f1d553976a2`. Strict Adapter checkpoint paths/SHA are in `adapter/STRICT_FOLD_CHECKPOINTS.json`; all authority paths/SHA are in `association/training/*.json` and `checkpoints/SEALED_EVIDENCE.json`; final VAL checkpoint paths/SHA are in `val/FROZEN_POLICY.json`.
Final science decision `FAIL_GLOBAL_ASSOCIATION_AUTHORITY`. NEXT_STAGE_AUTHORIZED=False. No downstream training, new candidate generation or test evaluation was started.
