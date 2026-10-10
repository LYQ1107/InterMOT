# N72R21R2 — execution in progress

FINAL GOAL: Event-Level Causal Identity Association: Safe One-Click Intervention with Fresh-Sequence Generalization for Online MOT

CENTRAL QUESTION: 用户点击一个人一次之后，我们能否让模型识别真正值得干预的身份关联事件，在尽可能保留原 MOT 正确轨迹的基础上修复身份错误，并在新的独立视频中稳定提高目标身份正确率、AssA、IDF1 和 HOTA？

The sole frozen Goal is `outputs/N72R21R2/FINAL_GOAL.json`; the before-effects protocol is `outputs/N72R21R2/PREREGISTRATION.json`. Application Goal is active. This is a new authorized research stage, not a reinterpretation of R1's negative result.

## Current verified scope

The R1 and original worktrees were clean at bootstrap. R2 starts from R1 commit `1bdb7ff99aaaae6b889b37b525f51bf68e48047d` on `codex/n72r21r2-event-causal-mot-generalization`. Initial Git HTTPS/SSH and connector reads failed; their evidence is preserved. A later fresh GitHub API read verified that the R1 branch still points to that exact parent. The frozen bootstrap protocol retains its original pending-source snapshot; the separate publication receipt records recovery rather than rewriting history.

R1 remains **FAIL_GLOBAL_MOT_TRANSFER**. Its final regression receipt is925 passing /5 historical failures, not all tests passing. Old two-scene results are not fresh-sequence generalization. This stage must prepare the inherited FIT16/INNER8 inputs even if the historical failure reproduces.

CONFIRM8 is unchanged and may not guide fitting, thresholds or policy selection. At bootstrap there were no new candidate extractions, model fits or policy evaluations; the actual progress below supersedes that initial snapshot. Scientific decision is **PENDING**, not PASS or FAIL. MOT is primary; SOT is deferred. No downstream stage is authorized.

## Actual M0 and fixed fresh-input pilot

Eight newly executed historical joint rollouts completed: CLICK_C0 and all three frozen original ACIB Full seeds on0001/0002. Per-frame legacy semantic state hashes/outputs and exported trajectory bytes match the originals. Pinned TrackEval was actually invoked again; all nine combined and per-sequence metrics match exactly. The historical semantic hash excludes prototype/motion tensors, so it is not mislabeled as a full tensor fingerprint. This is successful reproduction of a scientific negative result, not a new method PASS.

The three preregistered fresh FIT videos0074/0020/0032 completed real SAM3/OSNet extraction, full original-axis/current-UID/feature integrity checks and actual full C0/Shadow baselines. The nine deterministic sole clicks give6 valid initializations and3 failures, retained without replacement. Every valid Shadow run preserves complete C0 states, trajectory bytes and all nine TrackEval metrics. Different clicks are averaged inside their video, not independent scene clusters.

For0020, the one valid click has121 strict positive-available frames over479 visible future frames (25.26% coverage); two other clicks cannot initialize. For0074 all three clicks initialize, with89.98%/80.83%/98.41% coverage, but C0 identity-correct counts are625/240/1176 and verified-other takeovers358/700/0. Thus fresh data contains both coverage limitations and real association errors on existing positive candidates. For0032, one click fails with no positive candidate; the two valid clicks have70.34%/77.50% coverage and C0 identity-correct counts214/87. These are input/baseline diagnostics, not policy gains or independent confirmation.

Full FIT16/INNER8 preparation is now running incrementally with at most two isolated physical GPUs and immediate per-video baselines. It is not conditional on the old development gate. No trained R2 model, adaptive correction policy or scientific improvement has yet been established.

First extraction attempts collided on physicalGPU0 because the upstream builder calls `.cuda()` without the adapter device argument. Both own workers failed with OOM before completing tapes; they had already exited when one SIGINT was attempted. No foreign process was touched. Separate versioned workers isolate one physical GPU per process. Completed inference is never repeated merely for a NumPy-count JSON serialization failure: input integrity is rechecked in a versioned writer, preserving original partial JSON/logs. Only new regenerable worker temporary output files were removed after their predeclared exact-target cleanup; no unique historical data or weights were deleted.

The earlier corrected-environment/fixture full regression was935 passing /5 historical failures (54.11s). Two933/7 attempts remain archived: initial subprocess `python` PATH errors, then zstd refusing newly created local symlinks. Exact same-filesystem hardlinks resolved only that local input plumbing without copying data or modifying tests/third-party source. The current V4 regression is **959 passing /5 historical failures** (55.67s), and all34 R2-focused tests pass. The five remain four old `SEQMAP_FILE` list/type CLI failures and one historical branch-literal assertion. No all-tests-pass claim is made.

Initial offline baseline-event mining is complete on the fixed pilot. Observed error intervals and sampled probes are not independent causal corrections. V1 current-inclusive baseline windows are explicitly archived as such; V2 uses future offsets1..H, separating the current action. Neither auxiliary observation is mislabeled as an actual counterfactual reward.

## M3/M4 progress checkpoint, not final result

The local SHA catalog `docs/N72R21R2_LOCAL_EVIDENCE_SHA.json` freezes an actual progress snapshot:9/24 fresh sequences have full candidate integrity and complete baseline evaluation. The fixed three-video pilot has274 sealed event positions and1,935 actually executed full-global same-prestate branches;1,682 have complete future H100. Source-prefix reconstruction adds full current candidate/public axes, raw global score matrices, current identity scores, complete prestate prototype/motion snapshots and strictly past3/8-step feature histories. Runtime workers enforce a file-access guard against GT/offline labels.

Separate offline labeling preserves currentt and futurek=1..H, positive candidate availability versus physical visibility, verified OTHER versus UNKNOWN/NONE, target N01/N10, other-person damage, same-target fragments, takeover/recovery, memory-write provenance and state divergence. Incomplete windows have null supervision labels. Of the pilot arms,83 have observed safe positive H100 components,880 have observed harm, and482 have severe other-person harm.873 executed-action configurations are duplicates. These counts overlap and are correlated across windows/actions/sole clicks: they are **not83 independent corrections, not a deployable policy and not Gate1 PASS**.

Actual preregistered101-frame TrackEval windows completed on all three pilot videos, including all feasible registered arms. All nine metrics and unchanged detection multisets were audited. Window HOTA/AssA are not summed or substituted for full adaptive-policy performance. Subsequent fresh videos use versioned tensor-inclusive before/after fingerprints and actual target motion evidence. V1 pilot traces retain their weaker angular-prototype/actor-box proxies explicitly; missing full motion measurements are marked unavailable, not invented.

The upstream SAM3 demo compactor can discard `multistep_point_inputs`, while the official trim helper indexes it. A process-local versioned compatibility wrapper retries only that exact KeyError with an optional debug-field view, preserving tensor references, official trim flags, model weights, thresholds and hard constraints. A real160-frame0020 prefix is exactly equal in exported UIDs/boxes/scores/features; this is not a whole-video or active-fallback equivalence proof.0027 recovered with4 recorded native schema retries and completed its full baseline. An abruptly ended0069 worker had no catchable receipt; its termination cause remains unknown. Its two exact partial temporary files were relocated, SHA-verified and preserved, not deleted; the full extraction was rerun in a separate versioned attempt and completed.

Fixed P2–P9 low-cost controls are now executing complete own-state policy trajectories with no new fitting: strict recovery, low-baseline-margin intervention, identity margin/global regret, competitor protection, causal confirmation, bounded selective native-bonus discount, target-specific NONE and conservative reattachment. P0/P1 reference the actual complete baselines rather than invented alias runs. The native control re-solves the entire global assignment, only halves the target-column native bonus under frozen current/past evidence, and preserves other columns/core cues/hard negatives. Zero interventions are reported as vacuous, never scientific success. Adaptive causal-onset/harm qualification remains pending.

M1, M2/M3 and M4 have bounded live drivers. All24 fresh FIT/INNER videos remain required. No R2 model has yet been fitted; training/objective/on-policy/memory/open-set/generalization audits and conditional confirmation are unfinished. Application Goal remains **ACTIVE**, scientific decision remains **PENDING**, and next-stage authorization remains false.

## Execution boundaries

Reuse existing DanceTrack TRAIN images, SAM3/OSNet weights and original Python environment. Preserve historical evidence. All runtime decisions use only current real candidate UIDs and causal state. Sole click initializes identity, not unconditional override authority. Every intervention uses the complete global assignment and committed feedback. GT labels stay offline. Frames, seeds and propagated errors are not independent causal events.

Only code, tests and necessary compact documentation are destined for GitHub. Data, embeddings, weights, media and bulk traces remain local. Completion requires actual experiments and final evidence-backed closure, not this bootstrap record.
