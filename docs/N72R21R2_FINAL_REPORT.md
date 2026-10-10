# N72R21R2 — execution in progress

FINAL GOAL: Event-Level Causal Identity Association: Safe One-Click Intervention with Fresh-Sequence Generalization for Online MOT

CENTRAL QUESTION: 用户点击一个人一次之后，我们能否让模型识别真正值得干预的身份关联事件，在尽可能保留原 MOT 正确轨迹的基础上修复身份错误，并在新的独立视频中稳定提高目标身份正确率、AssA、IDF1 和 HOTA？

The sole frozen Goal is `outputs/N72R21R2/FINAL_GOAL.json`; the before-effects protocol is `outputs/N72R21R2/PREREGISTRATION.json`. Application Goal is active. This is a new authorized research stage, not a reinterpretation of R1's negative result.

Earlier sections preserve progress checkpoints. The latest verified memory/open-set checkpoint below supersedes their input counts; it is not a final scientific decision.

## Current verified scope

The R1 and original worktrees were clean at bootstrap. R2 starts from R1 commit `1bdb7ff99aaaae6b889b37b525f51bf68e48047d` on `codex/n72r21r2-event-causal-mot-generalization`. Initial Git HTTPS/SSH and connector reads failed; their evidence is preserved. A later fresh GitHub API read verified that the R1 branch still points to that exact parent. The frozen bootstrap protocol retains its original pending-source snapshot; the separate publication receipt records recovery rather than rewriting history.

R1 remains **FAIL_GLOBAL_MOT_TRANSFER**. Its final regression receipt is925 passing /5 historical failures, not all tests passing. Old two-scene results are not fresh-sequence generalization. This stage must prepare the inherited FIT16/INNER8 inputs even if the historical failure reproduces.

CONFIRM8 is unchanged and may not guide fitting, thresholds or policy selection. At bootstrap there were no new candidate extractions, model fits or policy evaluations; the actual progress below supersedes that initial snapshot. Scientific decision is **PENDING**, not PASS or FAIL. MOT is primary; SOT is deferred. No downstream stage is authorized.

## Actual M0 and fixed fresh-input pilot

Eight newly executed historical joint rollouts completed: CLICK_C0 and all three frozen original ACIB Full seeds on0001/0002. Per-frame legacy semantic state hashes/outputs and exported trajectory bytes match the originals. Pinned TrackEval was actually invoked again; all nine combined and per-sequence metrics match exactly. The historical semantic hash excludes prototype/motion tensors, so it is not mislabeled as a full tensor fingerprint. This is successful reproduction of a scientific negative result, not a new method PASS.

The three preregistered fresh FIT videos0074/0020/0032 completed real SAM3/OSNet extraction, full original-axis/current-UID/feature integrity checks and actual full C0/Shadow baselines. The nine deterministic sole clicks give6 valid initializations and3 failures, retained without replacement. Every valid Shadow run preserves complete C0 states, trajectory bytes and all nine TrackEval metrics. Different clicks are averaged inside their video, not independent scene clusters.

For0020, the one valid click has121 strict positive-available frames over479 visible future frames (25.26% coverage); two other clicks cannot initialize. For0074 all three clicks initialize, with89.98%/80.83%/98.41% coverage, but C0 identity-correct counts are625/240/1176 and verified-other takeovers358/700/0. Thus fresh data contains both coverage limitations and real association errors on existing positive candidates. For0032, one click fails with no positive candidate; the two valid clicks have70.34%/77.50% coverage and C0 identity-correct counts214/87. These are input/baseline diagnostics, not policy gains or independent confirmation.

Full FIT16/INNER8 preparation is now running incrementally with at most two isolated physical GPUs and immediate per-video baselines. It is not conditional on the old development gate. The small model pilot described below validates actual optimization, not reliable adaptive correction or scientific improvement.

First extraction attempts collided on physicalGPU0 because the upstream builder calls `.cuda()` without the adapter device argument. Both own workers failed with OOM before completing tapes; they had already exited when one SIGINT was attempted. No foreign process was touched. Separate versioned workers isolate one physical GPU per process. Completed inference is never repeated merely for a NumPy-count JSON serialization failure: input integrity is rechecked in a versioned writer, preserving original partial JSON/logs. Only new regenerable worker temporary output files were removed after their predeclared exact-target cleanup; no unique historical data or weights were deleted.

The earlier corrected-environment/fixture full regression was935 passing /5 historical failures (54.11s). Two933/7 attempts remain archived: initial subprocess `python` PATH errors, then zstd refusing newly created local symlinks. Exact same-filesystem hardlinks resolved only that local input plumbing without copying data or modifying tests/third-party source. The current V4 regression is **959 passing /5 historical failures** (55.67s), and all34 R2-focused tests pass. The five remain four old `SEQMAP_FILE` list/type CLI failures and one historical branch-literal assertion. No all-tests-pass claim is made.

Initial offline baseline-event mining is complete on the fixed pilot. Observed error intervals and sampled probes are not independent causal corrections. V1 current-inclusive baseline windows are explicitly archived as such; V2 uses future offsets1..H, separating the current action. Neither auxiliary observation is mislabeled as an actual counterfactual reward.

## M3/M4 progress checkpoint, not final result

The local SHA catalog `docs/N72R21R2_LOCAL_EVIDENCE_SHA.json` freezes an actual progress snapshot:9/24 fresh sequences have full candidate integrity and complete baseline evaluation. The fixed three-video pilot has274 sealed event positions and1,935 actually executed full-global same-prestate branches;1,682 have complete future H100. Source-prefix reconstruction adds full current candidate/public axes, raw global score matrices, current identity scores, complete prestate prototype/motion snapshots and strictly past3/8-step feature histories. Runtime workers enforce a file-access guard against GT/offline labels.

Separate offline labeling preserves currentt and futurek=1..H, positive candidate availability versus physical visibility, verified OTHER versus UNKNOWN/NONE, target N01/N10, other-person damage, same-target fragments, takeover/recovery, memory-write provenance and state divergence. Incomplete windows have null supervision labels. Of the pilot arms,83 have observed safe positive H100 components,880 have observed harm, and482 have severe other-person harm.873 executed-action configurations are duplicates. These counts overlap and are correlated across windows/actions/sole clicks: they are **not83 independent corrections, not a deployable policy and not Gate1 PASS**.

Actual preregistered101-frame TrackEval windows completed on all three pilot videos, including all feasible registered arms. All nine metrics and unchanged detection multisets were audited. Window HOTA/AssA are not summed or substituted for full adaptive-policy performance. Subsequent fresh videos use versioned tensor-inclusive before/after fingerprints and actual target motion evidence. V1 pilot traces retain their weaker angular-prototype/actor-box proxies explicitly; missing full motion measurements are marked unavailable, not invented.

The upstream SAM3 demo compactor can discard `multistep_point_inputs`, while the official trim helper indexes it. A process-local versioned compatibility wrapper retries only that exact KeyError with an optional debug-field view, preserving tensor references, official trim flags, model weights, thresholds and hard constraints. A real160-frame0020 prefix is exactly equal in exported UIDs/boxes/scores/features; this is not a whole-video or active-fallback equivalence proof.0027 recovered with4 recorded native schema retries and completed its full baseline. An abruptly ended0069 worker had no catchable receipt; its termination cause remains unknown. Its two exact partial temporary files were relocated, SHA-verified and preserved, not deleted; the full extraction was rerun in a separate versioned attempt and completed.

Fixed P2–P9 low-cost controls are now executing complete own-state policy trajectories with no new fitting: strict recovery, low-baseline-margin intervention, identity margin/global regret, competitor protection, causal confirmation, bounded selective native-bonus discount, target-specific NONE and conservative reattachment. P0/P1 reference the actual complete baselines rather than invented alias runs. The native control re-solves the entire global assignment, only halves the target-column native bonus under frozen current/past evidence, and preserves other columns/core cues/hard negatives. Zero interventions are reported as vacuous, never scientific success. Adaptive causal-onset/harm qualification remains pending.

M1, M2/M3 and M4 have bounded live drivers. All24 fresh FIT/INNER videos remain required. The following additional pilot does not replace the mandatory full corpus. Main training/objective/on-policy/memory/open-set/generalization audits and conditional confirmation are unfinished. Application Goal remains **ACTIVE**, scientific decision remains **PENDING**, and next-stage authorization remains false.

## Actual event-learning development checkpoint

The new actual-trajectory utility interface invokes the pinned TrackEval preprocessing and HOTA/CLEAR/Identity metric implementations directly. Before computing utility labels, every feasible arm of all three preregistered pilot101-frame windows reproduced the existing CLI: **189 all-nine-metric comparisons match within1e-10**. Future training rewards then use original offsets1..100, excluding action-time t. L5 is the preregistered equal-weight paired delta HOTA/AssA versus own KEEP; missing futures are null. These overlapping event-window rewards are not summed or substituted for whole-policy performance.

Eight small model families and seven separate objective interfaces are implemented, with a measured-trigger requirement for any additional relational structure. Only the six preregistered scalar/logistic pilot fits have actually run so far: three seeds each, optimization on FIT0074/0020 and held-out FIT0032 internal validation. Each fit has180 optimizer steps with nonzero gradients; selected scalar checkpoints change6 elements and logistic checkpoints change123. Training uses616 distinct usable arms and303 internal-validation arms after removing duplicate executed configurations and incomplete H100. The actual FIT labels include25 safe-benefit and264 harm cases. Equal video/sole-click/observed-interval/event/arm weighting controls correlation without claiming the intervals are proven independent causal roots. Normalization and optimizer never use INNER/CONFIRM/VAL/TEST observations.

Those pilot checkpoints are **TRAINED_UNCALIBRATED_NOT_DEPLOYABLE**, not qualified main models. None of their held-out FIT arms passes the fixed benefit>=.8, risk<=.02, value>0 score diagnostic. Actual own-policy pilot trajectories now test the fixed conservative point on all three frozen FIT pilot videos; they cannot be described as fresh INNER generalization. Current/past feature-distribution shift from baseline-only supervision is explicitly recorded, not hidden. Other structure/loss/main-corpus fits remain pending, not fabricated from the pilot.

The first four fixed-control videos had zero effective interventions. On0044, Recovery Only and Baseline Uncertain have the same6 direct decisions each across the first two valid clicks; these are **not12 independent corrections**. Each policy adds only1 identity-correct frame on click1 and none on click0, with zero target N10. Nevertheless paired full-video HOTA/AssA/IDF1 all decrease: click0 deltas are−.001676/−.002130/−.006304, click1 deltas−.001607/−.001762/−.005076. This is direct evidence that target N01/N10 alone does not establish global safety. Own-policy tensor-inclusive prefix reconstruction and same-prestate current-action/KEEP future branches are now being executed to diagnose other-person damage. P7 scorer actions are identified from actual assignment changes even when their action-family string is KEEP. No zero-action or one-shot result is used to claim Gate1 PASS.

The V5 full regression is **979 passed /5 unchanged historical failures** (63.67s), including all20 new learning/trajectory/runtime tests. The failing tests remain the four historical direct-CLI SEQMAP_FILE type failures and the historical fixed-branch assertion. Runtime/model/metric interfaces introduce no new failure in that run. Subsequent driver additions do not modify the executing frozen model or candidate sources.

The0044 own-policy onset audit subsequently completed. Tensor-inclusive actual prefix and current-action outputs reproduce the executed policy; every diagnostic forks its own exact prestate into current-action-then-KEEP and OWN_KEEP. For either of the two identical-output fixed policies, click0 has48 future other-person damaged public-frames over its sampled arms, including one severe-harm arm, with zero target N01/N10. Click1 has13 other-person damaged public-frames and one target N01, again zero target N10. These sums include overlap and the two policies duplicate each other; they are not independent harm counts. No audited arm is a safe positive. This establishes a concrete causal other-person damage mechanism for the observed global degradation; incomplete windows remain incomplete and the stage's overall final decision is still pending.

## Fresh24, zero-authority memory and current-axis Open-Set checkpoint

All **16 FIT +8 INNER** videos now have actual newly extracted SAM3 candidates, full-axis integrity checks, deterministic sole-click preparation and C0/Shadow baseline receipts. No old scenes replaced new videos. The three scheduled clicks all fail initialization in0027 and0012; these two videos remain in the24-video census and are not silently replaced or counted as successful recognition. Consequently the current Open-Set optimizer has source rows from15 FIT videos and calibration rows from7 INNER videos, while requiring all24 source receipts before fitting.

The six scalar/logistic event-authority pilot models completed all18 model/video evaluations and36 valid-click full own-state trajectories. They execute zero effective interventions, with zero N01/N10 and zero paired deltas in all nine metrics. This is **vacuous**, not safe correction success. The registered42 main architecture/objective/seed fits have a live prerequisite driver but cannot start until all24 fresh counterfactual labels, actual pinned future-trajectory utilities and simple full-video controls are complete. Main joint-state/on-policy comparisons, nonvacuous global-MOT qualification and conditional confirmation remain unfinished.

0099 exposed a context-reconstruction bug: using the identity actor's already-normalized anchor as tracker input normalized it an extra time. The maximum anchor change was1.4901161193847656e-8;14 CF context positions failed the strict tensor-inclusive prestate test even though semantic states and outputs matched. The failed V1 log, compressed partial labels and contexts remain immutable. V2 uses the exact original sealed raw anchor and actually reproduces **all139** original CF tensor prestates, verifying every arm and source-isolation receipt. The original991 CF arms were not rerun or changed; corrected offline labels and actual paired window TrackEval completed. A previously absent canonical label receipt explicitly links successful V2 evidence and the original failure, rather than reclassifying the V1 attempt as success. Future terminal instances are handled by the bounded versioned recovery driver.

M-A now changes only the actual identity Bank: FROZEN, MEAN, DELAYED, MULTICUE, PENDING_TRUSTED and ROLLBACK. Every changing-bank case preserves full tensor tracker state versus actual FROZEN at each original frame; all complete C0 outputs/semantic states/trajectory bytes and all nine full-video metrics remain exact. Writes are actual globally committed observations, not uncommitted proposals. RISK_LEARNED is rejected until a fresh fitted writer exists. M-B remains conditional on genuine TRAIN-side safe authority; a current-identity calibration point cannot authorize it.

On the first completed0020 memory example, MEAN accepts422 writes:121 TARGET,189 VERIFIED_OTHER and112 UNKNOWN, giving71.327% wrong-or-UNKNOWN contamination. MULTICUE accepts72 actual TARGET writes, zero observed contamination,59.504% retention of the121 correct C0-committed observations, and competitive available-frame Rank-1 of110/121 versus108/121 for FROZEN. DELAYED/ROLLBACK accept66 correct writes (54.545% retention), PENDING_TRUSTED19 (15.702%). These are correlated observations in one video, not a population2% safety guarantee. In particular MULTICUE's retention remains below the frozen60% gate. Constant HOTA under zero authority is isolation proof, not evidence of a successful memory method. All24 M-A cases continue; risk-aware writes, uncertainty/frontier intervals and any conditional M-B integration are not complete.

M8 independently collected every32 original post-click frame in all24 fresh videos, scoring **every real current UID plus explicit NONE** before the actual C0 commit. Current-axis queries preserve the original tensor tracker state and causal pending state. Labels are added only after the entire video's registered runtime seals: TARGET, VERIFIED_OTHER, UNKNOWN, correct/incorrect NONE; physical absence and visible-without-positive-candidate are separate. UNKNOWN is never a verified identity hard negative.

Four fresh current-correctness heads (SCALAR, LOGISTIC, MLP, same-architecture MLP_HARD_NEGATIVE) completed all **12** three-seed actual fits, with FIT-only normalization/optimization and INNER-only epoch/temperature/claim calibration. The hard-negative contrast weights only FIT verified-other look-alikes; UNKNOWN remains its own class. All selected checkpoints were strictly reloaded and rescored on623 actual INNER current frame groups, before labels; accepted-claim counts reproduce the original fit evaluations exactly. This is sampled current-identity inference, **not dense full-video learned intervention or future safety**. The simple margin-labelled reference also includes its explicitly implemented relative-NONE challenger; it must not be misrepresented as an isolated pure-margin ablation. Learned temperature NLL is hierarchical row-weighted; simple-control temperature NLL averages sampled frame groups and is not described as the identical weighting scheme.

Eleven heads find no registered nonvacuous empirical safe point. LOGISTIC seed730101 accepts12 correct identity claims in3 videos at p>=.9, margin>=.15 and UNKNOWN<=.02: micro coverage1.926%, available-target recall4.563%, zero observed wrong/UNKNOWN claims. Its two sibling seeds abstain. Across all three logistic seeds averaged within each video, available-target recall is **2.164% video-macro**, with descriptive seven-video cluster bootstrap95% CI **[0.078%,4.693%]** (2000 resamples, seed730104). This is selected-INNER description, not untouched generalization or evidence that population claim risk is<=2%. Other learned families have zero selected recall; their degenerate zero intervals are explicitly not safety success. Unsafe fixed diagnostic points retain substantial wrong/UNKNOWN/incorrect-NONE outcomes, rather than dropping them from the denominator.

Local availability artifacts include the measured cutoff-grid Recall@verified-other-row-FPR2, per-sequence precision/coverage/false-presence/UNKNOWN metrics and strict current runtime reload evidence. Verified-other-row FPR is not physically absent false-presence FPR. Exact CF-event repair opportunities missed by rejection still require an exact-frame current-axis diagnostic; every32 observations are **not** substituted by nearest-frame imputation. No best seed is selected for confirmation.

The V6 full regression completed with **988 passed /5 unchanged historical failures** in57.87s. V7 then completed with **991 passed /5 unchanged historical failures** in65.09s, including the two strict current predictor tests and all24 main-scope prerequisite test. The four historical TrackEval CLI SEQMAP_FILE type failures and historical fixed-branch assertion remain visible. No third-party code or tests were altered to hide them.

Live source/control/memory drivers retain failures and refuse duplicate ownership. Disk remains above the60GiB reserve, existing environment/weights/images are reused, and no new dataset or checkpoint is downloaded. The application Goal remains **ACTIVE**, final scientific decision **PENDING**, next-stage authorization false, and CONFIRM/VAL/TEST/SOT unopened. This checkpoint is code, tests and compact protocol/documentation only; weights, candidate features, GT, source images and bulk evidence remain local.

## Deployment-matched controlled-state checkpoint (still ACTIVE)

The original live CF/training/association sources are unchanged. A separate
observer reuses the exact existing LearnedEventBridge implementation on an
isolated always-KEEP view. It copies current branch features, past3/8 history
and CANDIDATE/NONE confirmation metadata before any future fork; after the
actual commit it accepts only causal metadata, never the view's post-KEEP
tracker/actor state. Unit tests match exact deployment features and preserve
the real rejected state. The original C0 CF feature confirmation default0 is
retained; it is not silently changed to match deployment.

The separately frozen source protocol covers all24 fresh development videos.
For each valid click and source family it chooses the first chronological
already registered CF frame with H150 available. Non-KEEP/non-delayed sources
must be currently applicable, feasible and assignment-changing. No future
reward, truth or best scene is used for selection. KEEP, rejection, raw/learned
identity top, alternative, recovery and delayed sources are replayed from
their real C0 prefixes; inapplicable cases remain explicit, not replaced.
Probe offsets1/5/20/50 have deployment-matched current features and nested
one-current-action/own-KEEP H100 futures. Original sealed source outputs,
semantic state and available full tensor states must agree. Actor tensors,
tracker-manager metadata, observer histories and source core state are
checked for clone isolation. Frozen weights' ephemeral inference caches are
shared but never treated as causal state or reused after future forks.

The first0074 runtime is genuinely executing: rejected, alternative and
raw/learned source cases each have actual global assignment changes, while
KEEP/delayed zero-effect cases remain zero-effect. This is **controlled
one-shot treatment-state supervision, NOT model-generated on-policy closure,
not independent safe corrections and not a full-MOT PASS**. It uses the
actual P0 no-write bank. Safely updated memory states still require measured
nonvacuous G4; unsafe mean writes or oracle positives cannot supply that label.

Every originally frozen CF frame additionally collects all actual current
UIDs plus explicit NONE before C0 commit. A separately frozen diagnostic
reloads all12 current verifiers at their existing selected INNER points,
seals all scores without truth input, then joins offline exact-frame labels
to actual own-future CF opportunities. It separates safe current N01 repairs
from future-only improvement, excludes delayed actions from the direct-current
denominator and keeps UNKNOWN distinct. These are correlated event-frame
opportunities, not independent onsets or executed MOT repairs. The diagnostic
is running/waiting for complete source videos, not yet a measured conclusion;
every32/nearest-frame imputation and threshold retuning are prohibited.

Three paired state-source modes (Baseline/Treatment/Mixed), each with the
same SMALL_MLP/L3 and3 seeds, are frozen before optimization. All use the same
mixed INNER set for epoch selection; normalizers and gradients use only the
respective FIT source. Treatment phases require an actual effective source
action strictly before the probe, not future or not-yet-delayed effects.
The driver waits for all24 source receipts; it does not optimize a ready subset.
Insufficient benefit/harm diversity is a measured supervision failure, not
a meaningless fit. True learned-policy correction and staged policy authority,
own-deployment distribution audit, safely updated memory, full generalization
gates and conditional confirmation remain unfinished.

Full regression V8 actually completed **996 passing /5 unchanged historical
failures** in124.34s. The five are the four historical TrackEval scalar
SEQMAP_FILE CLI failures and the historical branch-literal assertion. Its5
new observer/source-selection tests pass. Subsequent R2-focused75 tests pass
in7.44s, including the later exact-CF and all24 state-source prerequisite tests.
No all-tests-pass claim or third-party/test rewrite is made. Existing
environment/weights/candidates are reused. The actual filesystem has about
76.9GiB free, above the60GiB reserve. Only code, compact protocols and necessary
documentation are published; bulk evidence and weights stay local. Goal is
**ACTIVE**, scientific conclusion **PENDING**, next-stage authorization false.

## Execution boundaries

Reuse existing DanceTrack TRAIN images, SAM3/OSNet weights and original Python environment. Preserve historical evidence. All runtime decisions use only current real candidate UIDs and causal state. Sole click initializes identity, not unconditional override authority. Every intervention uses the complete global assignment and committed feedback. GT labels stay offline. Frames, seeds and propagated errors are not independent causal events.

Only code, tests and necessary compact documentation are destined for GitHub. Data, embeddings, weights, media and bulk traces remain local. Completion requires actual experiments and final evidence-backed closure, not this bootstrap record.
