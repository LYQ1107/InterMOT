# N72R20R4 — causal identity-to-trajectory transfer

FINAL GOAL: Causal Identity-to-Trajectory Transfer: Globally Consistent Online Association for Long-Term MOT.

Central question: 用户点一下这个人以后，我们能否真正持续认住他，并让长期身份记忆提高整个 MOT 系统的 HOTA 和 AssA？

Scientific decision: `FAIL_GLOBAL_ASSOCIATION_AUTHORITY`. Correctness passed; scientific effect/generalization did not. `NEXT_STAGE_AUTHORIZED=false`.

The complete report, all 25 VAL rows, uncertainty, ablations and cached-feature timing are in [FINAL_REPORT.md](../outputs/N72R20R4/FINAL_REPORT.md). Machine result: [FINAL_RESULT.json](../outputs/N72R20R4/FINAL_RESULT.json).

- Independent native reconstruction and causal identity-off A/A passed. All 34 development ablations, eight LOSO folds, three seeds and the preregistered repair ladder were evaluated with complete trajectories.
- DEV baseline/treatment HOTA `0.5808919181`, AssA `0.5433304753`, DetA `0.6230911682`, IDSW `250`; all paired deltas are zero. Every inner fold selected G0.
- Frozen G0/P0 VAL baseline/treatment HOTA `0.4878651521`, AssA `0.5118588672`, DetA `0.4691716953`, IDSW `1832`; all paired deltas and macro-bootstrap intervals are zero. VAL had prior research exposure and was not tuned this stage.
- Real GRU P1 wrong-write rate `2575/7452=34.55%`; P2 `165/684=24.12%`, with correct retention `519/4877=10.64%`. Neither memory safety gate passed.
- The strongest preregistered residual is provably insufficient to beat the global assignment margin in `7451/7452` assigned target frames. All actual assignment-change counts remain zero. The resulting 9,972 controller examples contain no beneficial labels, so controller superiority is not established.
- The posthoc single-target oracle increases HOTA by `0.0150083` but decreases DetA by `0.0151454`; it is neither a runtime method nor a guaranteed global upper bound.
- This negative result does not prove that a better-calibrated future identity intervention is impossible. It does not authorize downstream experiments automatically.

Checkpoint paths and SHA256 are enumerated in [SEALED_EVIDENCE.json](../outputs/N72R20R4/checkpoints/SEALED_EVIDENCE.json). All 322 dev/VAL trajectory+trace exports and frozen runtime core hashes were verified. Official re-evaluation exactly reproduced every metric, retaining original evaluator logs.

Stage tests: 30 passed. Selected dependency regression: 74 passed, 1 unchanged historical branch-name assertion failed; no whole-repository PASS is claimed. No backbone training, new candidate generation, test evaluation, unrelated process termination or historical asset deletion occurred. Simulated GT-derived human anchors are not real-human evidence.

Scientific completion and Git delivery are tracked separately in [stage_status.json](../outputs/N72R20R4/stage_status.json). The application Goal is not complete until final local/remote HEAD equality and a clean worktree are verified.
