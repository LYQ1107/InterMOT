# N72R21 — complete T1, frozen validation, T2 still active

Final Goal: **One Click, Persistent Identity: Causal Long-Term Human Tracking Across Occlusion, Reappearance and Independent Recordings**.

This is an intermediate scientific record, not final closure. All 72 T1 fits and all 120 outer scene/case replays/evaluations completed. Source P0/P1 state seals cover all 336 registered fold/seed/policy/scenes. Historical 795-file hashes were reverified unchanged.

## Complete T1 results

Three-seed mean sequence-macro all-visible recall on all eight historically exposed TRAIN scenes:

| Training → deployment | Recall | Sequence-cluster 95% CI |
|---|---:|---:|
| P0 → P0 | 20.93% | 15.23–27.52% |
| P1 → P1 | 21.61% | 15.40–28.44% |
| P1 → P0 (mismatch) | 21.44% | 15.42–28.20% |
| MIXED → P1 | 21.24% | 15.26–27.90% |
| MIXED → P0 | 21.23% | 15.36–27.95% |

Matched P1 versus matched P0 differs by +0.680 percentage points (paired sequence CI +0.0066 to +1.600 points), below the preregistered three-point effect threshold. This also changes the training condition/weights, not just deployed memory. Four descriptive contrasts are reported, without multiple-comparison success selection. The isolated same-MIXED-model P1-versus-P0 bank effect is +0.0056 points (CI −0.254 to +0.304), not evidence of useful persistent memory. P1-trained P0-versus-P1 deployment differs by −0.171 points (CI −0.452 to +0.111).

P1→P1 pooled non-target-or-unverified write rates across the three seeds are 79.37%, 78.97%, 78.47%; strictly verified correct-observation retention is 25.23%, 25.09%, 26.00%. MIXED→P1 rates are 79.62%, 79.17%, 78.61%, with retention 24.53%, 24.78%, 25.65%. These fail joint wrong-write ≤2% / retention ≥60%. Unmatched writes are conservatively counted as not verified target, not claimed to be individually identified other people. P0's undefined wrong-write rate and zero retention cannot pass.

## T2 evidence and independent frozen sequence protocol

T2 actual fitting is active using T1-produced current states and sealed real paired write/no-write futures. First outer scene 0001 has all eleven controls/all three seeds. Full K8/K4/K1, anchor-only, fixed safety, no delay and uniform-bank attention all have the same 20.87% recall because full/fixed safety writes are zero there. This is an empirical no-write negative result, not capacity robustness or memory safety. Full-model target-unavailable FPR across seeds is 99.17%, 90.83%, 99.72%. The unconditional write diagnostic has roughly 76–80% conservative write error and 22–24% correct retention. These one-scene observations have no sequence-cluster CI and are not used to change training, thresholds or candidate policy.

The separate validation protocol was committed at `3230d28` before N72R21 VAL pixels/GT/runtime access. It fixes all 25 existing sequences, all first-eligible one-click targets (273), the lexicographically first TRAIN outer configuration 0001 and all three INNER-selected seeds. All VAL scenes are absent from these six FIT plus one INNER scene lists. Existing historical SAM3/OSNet candidates cover 25,508 frames and 188,971 crops; source SHA checks pass, no duplicate data/download/inference. Rule anchor/mean/bank and learned anchor/mean/unsafe/full controls run at unchanged operating points. VAL never fits a parameter or selects a seed, checkpoint, target, sequence or candidate policy. Historical project exposure of this benchmark is disclosed; it is not a virgin cross-recording or identity-disjoint benchmark.

GT-simulated click preparation necessarily parses annotations to find the first eligible target observation. Its GT identity labels are kept in a separate file. Runtime receives only real clicked-crop embeddings, geometry, anonymous episode tokens and current/past candidates. A separate evaluator verifies every compared sequence runtime seal before future GT reads. Secondary identity-claim PR-AUC/ECE masks/counts UNKNOWN unmatched rank1 identities, unlike primary candidate-availability calibration. High availability PR-AUC cannot be relabelled as reliable identity recognition.

## Existing LaSOT alternative, tests and remaining work

The existing SAM3 checkpoint is reverified in place: 3,502,755,717 bytes, SHA256 `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`. No weights or data downloaded/copied. A frozen person-window candidate/identity diagnostic waits for all 24 T2 fits and a genuinely idle sole GPU0, caps allocation at 12 GiB, and retains partial evidence on error. It uses the three existing contiguous TRAIN windows and real machine crops. Unknown FPS disables temporal input features explicitly: timestamps remain null, frame counts remain actual original numbers. Single-target annotations do not verify other identities or physical absence; invalid boxes stay UNKNOWN. This deployment distribution-shift diagnostic cannot pass time-valid, cross-session or cross-day science. CHIRLA still lacks lawful accessible media; no fabricated T3 experiment replaces it.

Latest actual tests: **72 focused passed; 786 repository passed, 5 failed**, four warnings in 56.61s. Four failures remain fixed-version TrackEval CLI list/path handling and one historical literal-branch assertion. T2 complete-cohort inference, full frozen VAL, lawful domain fallback, failure visualizations, final five tables/report and exact Git closure remain pending. No downstream stage is authorized.
