# Historical lessons for N72R21

Final Goal: **One Click, Persistent Identity: Causal Long-Term Human Tracking Across Occlusion, Reappearance and Independent Recordings**.

This is new explicit user authorization, not an automatic promotion from a historical PASS. Existing artifacts and scientific decisions remain immutable. Machine-readable results and checkpoint manifests are recorded in `HISTORICAL_RESULTS.json`; `HISTORY_BEFORE.json` seals 795 historical reports, results, manifests and core code files.

| Stage | Actual evidence | What N72R21 must not infer |
| --- | --- | --- |
| N72R18 | Clean, offline GT-positive replay: GRU H100 91.36%, EMA 78.13%; gate did not beat GRU. | Deployable selection, absence rejection, or safe self-updating memory. |
| N72R20 | Feature shadow signal, but frozen N72R15 association tape unavailable. No `outputs/N72R20/FINAL_RESULT.json`; report lives in `docs/`. | A completed formal association result. |
| R3 | Pooled appearance/presence signal; no LOSO operating point met FPR/recall safety. | Pooled separation implies cross-scene calibrated identity recognition. |
| R3R1 | Explicit NONE supervision failed cross-sequence verification. | Adding a NONE logit alone fixes unknown/absent targets. |
| R3R2 | Genuine Adapter Rank-1 improvement to 68.75%; open-set FPR 23.47%, correct recall 53.24%; full requirement failed. | Good closed-set ranking establishes safe open-set use. |
| R3R2R1 | Broad calibration/uncertainty/temporal exploration; selected FPR 8.25%, correct recall 22.76%; `FAIL_ABSENCE_SEPARABILITY`. | Repeated threshold exploration on the same scenes supplies independent confirmation. |
| R3R2R2 | Joint identity/availability training; P1 localization remains a bottleneck; shadow wrong writes 25.30%. | Ranking or correct-write fraction proves contamination control. |
| R3R2R3 | Candidate refinement changed few valid boxes; VAL HOTA/AssA did not improve. | Local correction counts or successful inference equal generalization. |
| R4 | Causal/global solver audit passed; actual identity interventions remained zero. P1 raises ranking but wrong writes 34.55%. | No intervention damage proves a useful controller; HOTA is the one-click Goal. |
| R4R1 | Natural counterfactual positives exist; native-only DEV signal did not generalize to VAL. All P0–P6 memory safety tests failed. | C6/GRU improved trajectories, zero writes are safe useful memory, or prior VAL is virgin. |

N72R21 therefore measures target recognition, target/NONE decisions and reappearance directly. It retains an immutable human anchor, evaluates simple anchor/mean/bank controls, separates availability from physical/visible presence, and compares genuinely causal matched versus mismatched memory states. Cross-recording identity must come from authoritative global metadata. Ordinary motion/native state is reset at independent-recording boundaries. Clean positive replay remains a diagnostic, not runtime proof.

Reused frozen models retain their actual supervision lineage. Historical seven-sequence Adapter metadata is not silently treated as six-fit proof; strict six-fit checkpoints and all-eight-TRAIN checkpoints have distinct evaluation roles. The eight historic DanceTrack development sequences and 25 previously evaluated VAL sequences are disclosed as exposed, not independent new data.

Historical regression: R4R1 reported **714 passed, 5 failed**, not full PASS. Four pinned TrackEval direct-CLI failures pass list-valued `SEQMAP_FILE` to a scalar file API; the fifth asserts an older branch literal. Historical tests and third-party code are not rewritten to hide this. N72R21's new focused suite is reported separately.
