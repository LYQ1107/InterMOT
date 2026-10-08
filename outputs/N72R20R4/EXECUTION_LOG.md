# N72R20R4 execution log

FINAL GOAL: Causal Identity-to-Trajectory Transfer: Globally Consistent Online Association for Long-Term MOT.

Frozen goal: [FINAL_GOAL.json](FINAL_GOAL.json). Task started 2026-10-08 (Asia/Shanghai).

- Initial working tree clean; no existing R4 branch or local files found. Source local HEAD: `2b80763fe8f875defa697adeae638679ff72fd6b`.
- Goal activated in the application and frozen before training or formal evaluation.
- Initial `git fetch origin` failed with a TLS handshake error. Direct-network attempt stalled and was stopped; connector ref read also failed at transport. Source remote refresh will be retried and recorded before formal evaluation.
- Python 3.12.3, torch 2.10.0+cu128, CUDA available. GPU 0 idle; unrelated GPU processes preserved.
- Initial disk approximately 101 GiB available, below 105 GiB warning. No dataset download, candidate regeneration or backbone training planned. Heavy work must stop below 100 GiB; external outputs will be streamed/compressed and budgeted.
- M0 initial findings: legacy transformed base score, target-only solver merge, frozen per-frame state, pre-event output omission, and assumed identity/quality scores for new SAM3 candidates require explicit audits and regression coverage.

All later checkpoints must include code/config/source hashes, metrics, pass/fail and resumable locations. A component failure never closes the overall goal.

- M0 completed: 24 historical Adapter SHA/architecture checks and N72R18 GRU SHA passed. Source audit reveals historic final seven-sequence fit mislabeled as six; strict six-sequence adapter training started with three seeds and existing frozen training episodes.
- M1 regression suite: 17 passed. Actual tower inference, complete global reassignment/NONE, shared candidate hard failure, branch clone isolation, changed next-frame score matrix, pre-click outputs, actual score-before-GRU-write and zero-write safety covered.
- M2 attempt 1 input failure preserved. Thirteen zero-area candidates in dancetrack0072 were all unassigned in the native source. Shared pre-solver validity filter implemented and logged, without altering original assets or clipping outputs.
- M2 attempt 2: all eight independently reconstructed legacy trajectories have exact native SHA and zero assignment mismatches. New causal baseline and identity-off trajectories have exact A/A outputs/state/birth/death/NONE/SHA and identical official TrackEval metrics. `PASS_CAUSAL_BASELINE_EQUIVALENCE`.
- Native baseline HOTA/AssA/DetA = 0.5824390063/0.5418544921/0.6281568126, IDSW 217. New dynamic causal baseline = 0.5808919181/0.5433304753/0.6230911682, IDSW 250. This lifecycle difference is explicitly attributed to NONE births and finite non-target lifetime, not learned identity benefit.

- M3 completed: 24 strict six-sequence Adapter checkpoints, all three seeds. R4 runtime suite expanded to 20 passing tests.
- Formal M4–M11 LOSO running with frozen exact-global-margin features, per-source fit-only calibration, and all preregistered authority/memory/repair variants. The pre-finalization preparatory attempt remains preserved separately; see `audit/LOSO_PREPARATORY_ATTEMPT_01.json`.
- Stage commit `8ab5bb5` pushed successfully to the R4 GitHub branch. Subsequent remote equality/source-refresh retries encountered TLS handshake failures, not evidence of an authentication failure. A dedicated SSH authentication public key was supplied to the user; no private credential was requested or disclosed.
- Storage remains close to the 100 GiB heavy-work stop line. Byte-identical sealed R4 export files may share storage after full SHA and byte verification, retaining every original result path and manifest. No historical assets or other projects are removed.
- VAL Adapter refit defaults to CPU because the initially idle GPU was subsequently occupied by other jobs. No unrelated process is stopped.
- Posthoc-only tools now distinguish real candidate coverage/rank, exact global score-margin bounds, ordinary tracker-state divergence, and frozen GRU query evolution. Intervention-window audit covers every preregistered outer variant and preserves all original GT columns; no posthoc result is used to tune a policy.
- R4 suite: 26 passed, compilation passed. Selected dependency regression: 74 passed, 1 failed only because an R3R2R3 test pins its old branch name. Test unchanged and failure retained in `checkpoints/TEST_RESULTS.json`; no whole-repository PASS is claimed.
