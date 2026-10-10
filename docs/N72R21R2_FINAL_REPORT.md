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

## Actual memory-risk inputs, clustered frontier and wait-owner recovery

This is a progress checkpoint, not the final stage decision. The original
live identity/association/candidate sources remain frozen and unchanged.
An independently frozen fresh writer protocol reconstructs strictly past
bank and pending embeddings from SHA-verified actual committed UID/frame
references under all six existing M-A trajectories. It adds ten own-bank
features to the existing32 current features. None of the inputs are oracle
positive memory, uncommitted proposals, future observations or GT labels.
Whole-video runtime seals precede the separate TARGET/VERIFIED_OTHER/UNKNOWN
labels. UNKNOWN contributes to contamination risk, not verified-person
hard-negative training.

The first0074 input/label receipt has2670 rows (445 per teacher bank policy):
1506 TARGET,792 VERIFIED_OTHER,372 UNKNOWN;2210 query rows have a nonempty
past bank. These are correlated observations and six policy replicas in
one video, not2670 independent identity events. Thirteen of the required24
source videos are ready at this checkpoint. LOGISTIC/MLP ×3 seeds are
registered but **zero fresh writer fits have run**: all24 source receipts
and actual M-A metrics are required before optimization. The new head
predicts **current committed-identity correctness**, not future association
safety. Actual own-bank full-video writer replay, held-state feature shift
and all-nine-metric C0 equivalence remain pending. A selected teacher-state
point is not G4; the separately registered p=.5 diagnostic is explicitly
unqualified. No hidden multicue/anchor/native/delay gate is added to this
head and no association permission is granted.

An immutable input-hash memory frontier verifies actual per-case accepted
write provenance, complete full-video seals, unchanged clicks/opportunities
and all nine actual C0 metrics. It averages seeds **inside** video clusters,
separates pooled exposure rates from equal-video macro rates, and reports
2000 whole-video bootstrap resamples (seed730104). Video-local GT identity
clusters are an additional description, not cross-video person-disjoint
proof. Independent causal write-event roots are not established; their CI
is explicitly unavailable, not replaced by frames, write bursts or error
intervals. Zero writes have undefined risk. An empirical [0,0] bootstrap
interval cannot establish population risk<=2%; a separately named binomial
interval concerns ANY-bad-write clusters, not per-write contamination.

The first snapshot contains13 actual fixed-policy video receipts, including
one all-initialization-failed video, hence **12 valid FIT video clusters**.
No INNER result or future risk-head replay is included. Partial pooled
actual accepted-write contamination/retention is:

| Fixed M-A memory | Accepted writes | OTHER+UNKNOWN risk | Correct retention |
| --- | ---: | ---: | ---: |
| FROZEN | 0 | Undefined | 0% |
| MEAN | 29521 | 49.629% | 100% |
| DELAYED | 2494 | 22.334% | 13.026% |
| MULTICUE | 2851 | 22.729% | 14.815% |
| PENDING_TRUSTED | 789 | 23.067% | 4.082% |
| ROLLBACK | 2494 | 22.334% | 13.026% |

These write counts are exposures, not independent sample sizes. For
MULTICUE the equal-video macro correct retention is13.277%, and the
descriptive video-macro contamination95% CI is[18.951%,54.759%]. Its paired
competitive Rank-1 macro improvement is only+.015363. All full-MOT deltas
are exactly zero by design under M-A, proving isolation only. None of these
partial observed fixed-policy points satisfies2% contamination/60% retention;
the stage remains PENDING, rather than declaring a final memory result from
an incomplete role census.

The wait-only original42-job main fit driver ended with unified-process
exit143, zero main fits and14/24 prerequisite videos. Its termination cause
is unknown; the old marker and exit receipt are preserved. A separate V2
owner verifies the old process is no longer the live owner, refuses to adopt
any existing main attempt, and waits for exactly the same24 prerequisites
and42 frozen jobs. It changes no model, objective, seed, threshold or split.

Controlled-state V1 replay failed on fresh nonpilot tapes because the old CF
V2 compactor records `before=null` for primitive future KEEP commits, while
their `after` tensor fingerprints are available. A versioned wrapper does
not compare a real hash to JSON null: it checks every nonnull direct-before,
every available preceding sealed-after as the current-before reference,
and every available current-after. Missing pilot tensor evidence remains
UNAVAILABLE. Actual51-frame KEEP prefixes on0037/0044/0069 each pass1 direct
before,50 previous-after-before and51 after checks, including original
outputs/UIDs/semantic states/actions, under the GT-file guard. This is an
engineering smoke, not full-video controlled supervision or on-policy
closure. Old CF tapes, source-case frames/actions, partials and failures
remain untouched. Recovery waits for all original24 attempts to terminate
so it neither duplicates the counterfactual resource slot nor double-counts
recovered videos in the original driver's completed/failed census.

Saved-log/JUnit full regression V9 is **1006 passed /5 unchanged historical
failures** in117.92s. The shell pipeline's zero return code belongs to tee,
not pytest success; XML/log retain every failure. Five additional frontier
tests pass, and the frontier+driver-recovery focused check has7 passing tests
(4.69s). No third-party or historical test changes hide the failures. The
subsequent separately saved full V10 regression is **1013 passed /5 unchanged
historical failures** in110.59s, including frontier/recovery tests. Frozen protocols are
`memory/RISK_WRITE_PROTOCOL_V1.json`, `memory/SAFETY_FRONTIER_PROTOCOL_V1.json`,
`training/MAIN_WAIT_RECOVERY_PROTOCOL_V2.json` and
`on_policy/JOINT_STATE_TENSOR_RECOVERY_PROTOCOL_V2.json` beneath stage outputs.
They serve the same Final Goal, keep next-stage authorization false, and do
not replace unfinished main architecture/objective/on-policy/MOT/generalization
requirements. Only source/tests/compact protocols and this explanation are
published; weights, source rows, GT, traces, media and bulk evidence stay local.

## Pure margin and frozen-authority support diagnostics — still PENDING

Two new controls remove the earlier relative-NONE bonus confound:
`PURE_REAL_CANDIDATE_MARGIN` ranks only real identity UIDs while retaining
NONE probability mass; `PURE_UID_PLUS_NONE_MARGIN` ranks the full unchanged
UID+NONE axis. Both reuse the existing selected temperature2.0, actual
all24 sealed current sources and the same nonvacuous INNER selector. Neither
changes association or trains a model. Both select CALIBRATION_ABSTAIN,
which is not a safety PASS.

At the fixed, explicitly unqualified p=.5/margin=.05 diagnostic point, the
623 INNER sampled axes yield62 identity claims:32 correct TARGET,6 verified
OTHER,24 UNKNOWN. Identity precision is51.613%, available-target recall
32/263=12.167%, and OTHER+UNKNOWN identity-claim contamination48.387%.
Verified-OTHER **row** FPR is6/2042=.294%; this is not the8.056% no-positive
false-presence rate (29/360) or4.762% physically-absent false-presence rate
(1/21). UID+NONE also accepts36 correct NONE decisions, so its pooled
accepted-decision risk30/98=30.612% is not identity-claim contamination30/62.
The shared hierarchical Brier=.087630 and true-axis NLL=1.753194 are current
score diagnostics, not trained UNKNOWN calibration or future-safe authority.

The separate frozen exact-CF diagnostic is running incrementally, without
nearest32-frame imputation. First0074/0020/0032 plus the all-init-failed0027
are complete. In those three actual pilot videos there are16 correlated
safe direct-current H100-component opportunities. Both selected controls
miss16/16; each loose diagnostic retains the correct repair UID at only1/16.
These are claimed identities, not executed MOT corrections or independent
causal onsets. All24 role results remain required before the final summary.

A new OFFLINE frozen-hand-filter audit covers the complete16 FIT receipt
census (one all-initialization-failed source remains in the denominator),
not INNER. There are104 safe direct-current event-frame groups /104 distinct
executed safe action configs after excluding194 duplicate config rows.
Every config is vetoed by the existing global_regret<=.2 ceiling;101 are
also vetoed by forbidding any displaced public ID,72 by anchor advantage,
and4 by anchor cosine. Veto counts overlap. Even hypothetical ideal model
scores and satisfied confirmation delay cannot pass any of these104 groups.
Actual C0 source confirmation counts remain0; assuming them satisfied is
explicitly an upper-bound diagnostic, not own-policy history or deployment.
Thus zero pilot interventions cannot alone establish inadequate learned
capacity: the fixed authorization support itself excludes these observed
repairs. Safe component labels still do not establish full-policy MOT gain.
Snapshot:
`training/authority_support_v1/snapshots/9ffaae87072cafd797c0acb00560579292d6e52a93e8475971e3bf02efcbd67d.json`.

The original gate remains an immutable control. A separate frozen FIT-only
support-ablation pilot uses the existing LOGISTIC_RISK models/all three seeds,
the original three FIT pilot videos, identical raw sealed-click anchors and
four explicit variants: legacy, no global-cost ceiling, no cost/displacement
ceilings, and no cost/displacement/anchor-advantage ceilings. All retain the
same learned benefit/risk/value thresholds,3-frame own confirmation, current
quality/anchor cosine, NONE filters, exact full-global feasibility, hard
negative constraints, weights and causal feedback. Runtime feature values
are never falsified to satisfy the old gate. There are36 registered full-video
cells, one CPU worker, no new fit or best-seed selection. All nine pinned
MOT metrics are measured; nonzero actions require own-prefix harm-onset
audits before any G1 claim. These variants have no confirmation authority.
Raw-anchor legacy is rerun rather than silently adopting the old extra-
normalized pilot as a tensor-matched control. Preflight failures caused by
an incorrect family identifier and missing protocol are retained; no replay
or frozen protocol existed at those failures.

The old offline diagnostic driver also exited143 while its marker still
said ACTIVE. The cause is unknown. Its old marker/logs and successful0051
utility receipt remain untouched. A new versioned owner verifies both old
owner/child absent, seals54 existing successful operations, and only resumes
missing dependency-ready operations from the unchanged all24/72-operation
schedule. It changes no scientific job or split.

Saved full regression V12 is **1023 passed /5 unchanged historical failures**
in110.06s, with pytest exit1 retained. It precedes the three new support-
ablation tests. The subsequent all-R2 focused run has101 passing tests
(7.54s), with the corrected family-registration test separately passing.
Four historical scalar-SEQMAP CLI tests and one historical branch-literal
test remain failures; no old tests or third-party source are rewritten.

The subsequent saved full V13 run, including support-ablation tests, is
**1026 passed /5 unchanged historical failures** in107.07s, pytest exit1.
The first three actual support-ablation cells (seed730101/raw-anchor legacy,
all three original FIT pilot videos) complete pinned full-MOT evaluation:
zero effective decisions and all nine paired metric deltas exactly zero.
This is matched-control reproducibility, not nonvacuous correction. The
remaining ablation cells are still running and are not inferred from that
zero-action control.

A separate frozen **read-only linear source/own-state diagnostic** now
reproduces seed730101 raw-anchor legacy predictions over all three pilot
videos (score max absolute error2.861e-6, measured logit error0). Five causal
input dimensions are constant in the actual pilot FIT optimizer rows.
Pending confirmation has source value0 and scale.05 but reaches normalized
200 online; its absolute contribution reaches14.622/16.132/8.373 in the
benefit/risk/value logits. Previous candidate agreement reaches normalized
19.993, with2.407/2.836/.533 contributions. Their coefficients changed from
initialization by only about1.34e-6/2.68e-6. This is a measured unsupported
feature excursion, not proof of which dimension caused failure. Across
9656/1596/3456 non-KEEP current choices, neither actual scores nor hypothetical
scores with these constant-dimension contributions removed meet the frozen
benefit/risk/value conditions. Removing them alone therefore has no observed
score-only recovery in this diagnostic. No counterfactual score was deployed,
weight refitted or future safety inferred. The initial read-only probe's
numpy-int64 JSON serialization failure/partial file are preserved; the
versioned successful record is `training/SUPPORT_LINEAR_SOURCE_SHIFT_DIAGNOSTIC_V2.json`.
This remains pilot data support/distribution evidence, not an architecture
failure or substitute for the pending matched-state/main/on-policy studies.

The subsequent saved full V14 regression is1028 passed /5 unchanged
historical failures (90.12s, pytest exit1); all103 R2-focused tests pass
(7.84s). This does not turn the historical failures into an all-pass result.

This is a progress checkpoint, not scientific closure. All24 own CF/state,
main architecture/objective/full-MOT/model-on-policy, independent memory-risk,
open-set repair misses/density/generalization and final audit requirements
remain. CONFIRM/VAL/TEST/SOT are unopened; next-stage authorization is false.

## Own-prefix onset and finite MAIN full-MOT checkpoint

FINAL GOAL remains **Event-Level Causal Identity Association: Safe One-Click
Intervention with Fresh-Sequence Generalization for Online MOT**. The full
stage remains ACTIVE/PENDING, not scientific closure.

The matched hard-filter support pilot now has all36 actual full-video cells
and all36 separate onset-audit receipts. All36 cells have zero effective
decisions and all nine actual paired MOT deltas equal zero. Empty onset
receipts explicitly say NOT_RUN_NO_EFFECTIVE_ACTION: no cloned intervention
was invented and no zero-action safety PASS is awarded. Removing these
named hand filters alone did not recover interventions for the fixed three
existing pilot weights. This does not establish a MAIN-model, matched-state,
or architectural failure.

The new own-onset helper includes complete controller feature history and
branch-pending buffers in clone fingerprints, not only tracker/identity
core state. Every actual effective decision, if any, is reconstructed from
the same whole own-policy prefix and split into own KEEP versus actual
current action followed by own KEEP through H100. The parent is unchanged;
both branches retain complete global ownership, current outputs/actions,
prototype/motion state and tensor seals. Nonoverlapping candidate windows
remain correlation controls, not independent causal-root proof. The first
new test attempt failed collection because of a test-module import path;
only the unfrozen test helper was repaired, without changing frozen science.
The two onset tests and five compact/selection tests subsequently pass.

`mot/MAIN_POLICY_PROTOCOL_V1.json` is frozen before any MAIN weight/outcome
is inspected. It preserves the registered42 real fits and all three seeds.
Every model is scheduled for full original-video INNER8 evaluation at two
already-preregistered benefit thresholds (.8 and .5), with risk<=.02,
confirmation3, identical raw sole-click anchors, P0 frozen memory, the same
real candidate tapes and full global solver. The explicit support policy
removes cost/displacement/anchor-advantage hard ceilings but retains original
feasibility, hard negatives, quality/anchor cosine, strict NONE semantics,
learned risk/value and own confirmation. It is development-only, not a
qualified deployed gate. No extra capacity, epochs, fits, seed selection or
exhaustive hard-filter Cartesian search are added.

INNER selection shares one operating point across all three seeds in each
family/objective group. Harm/nonvacuity precede video-macro HOTA/AssA/IDF1,
with deterministic name ties; incomplete H100 effects cannot qualify the
screen. All-vacuous/unsafe choices remain unqualified diagnostic selections.
Selected-point FIT16 evaluation follows; actual pinned full-video metrics
and own-onset audits are mandatory. Clicks and seeds are averaged inside
video before the fixed2000-resample paired-video95%CI. Independent G1 root
proof and complete G0/G1/G2 closure are still separate mandatory audits;
this selector never authorizes CONFIRM, VAL, TEST, SOT or the next stage.

The runtime verifies newly replayed C0 against sealed original outputs and
states, and tensor equality throughout the zero-action prefix. It preserves
every frame's complete global output/action/core seals and sampled own
causal features, without copying giant solver matrices/feature caches.
The fixed existing pilot51-frame runtime smoke matches C0 tensors/outputs
on all51 frames. Actual compact round trips over all three fixed first-pilot
episodes preserve decisions/KEEP proposals/tensor seals. Trace extrapolation
is3.69GiB under the stated2000-frame/three-click planning assumption;6GiB
includes planned trajectories/evaluation/branch room, not an unbounded
storage permission. Actual free-space checks retain the60GiB floor. No raw
images, new environment, historical assets or unique evidence are deleted.

The single-CPU MAIN evaluation owner is launched and waits for all24 actual
source prerequisites and complete three-seed weights; zero MAIN fits and
zero MAIN full-video results exist at this checkpoint. The first complete
V15 regression reports **1035 passed /5 unchanged historical failures**
(69.83s, actual pytest exit1); all110 R2-focused tests pass (8.68s, exit0).
The four scalar-SEQMAP CLI failures and one
historical branch-literal failure remain visible. Confirmation remains
unopened and scientific success remains pending.

## Matched-state full-MOT adapter and original-scene audit (V9 checkpoint)

The FINAL GOAL remains **Event-Level Causal Identity Association: Safe
One-Click Intervention with Fresh-Sequence Generalization for Online MOT**.
Scientific closure is **PENDING**, application Goal is **ACTIVE**, and
confirmation/VAL/TEST/SOT and the next stage remain unauthorized/unopened.

The finite matched-state evaluation adapter is frozen in
`outputs/N72R21R2/on_policy/STATE_POLICY_PROTOCOL_V1.json`
(SHA256 `a24d91f060ccdf8536fb0e6c0b56fd9d1b03d2533932ee1919a5344c3070557c`).
It evaluates exactly the existing three training-state sources
BASELINE_STATE/TREATMENT_STATE/MIXED_STATE, SMALLMLP/L3, and all three
registered seeds. It adds **no optimizer runs, capacity choices, thresholds
or best-seed selection**. Actual successful weights and all24 controlled
sources are required. It reuses the unchanged frozen MAIN full-global
runtime, zero-action tensor checks, actual own-prefix one-shot onset
branches and pinned nine-metric evaluator under isolated state-policy
output paths. Both registered points run on INNER8; one shared point per
state source is then evaluated on FIT16 across all3 seeds (at most288
video/model/point cells). Failed seeds are retained, not replaced by a
successful subset. The one-CPU queue is launched and waiting for its
actual source/weight prerequisites; **no state-policy full-MOT result
exists at this checkpoint**. Controlled-source training is not actual
learned-model-generated on-policy or staged-training closure.

The M10 full-video tables now use all24 original candidate sources and
the density grouping frozen before policy effects. There are11 MIXED,
7 MEDIUM,5 CROWDED and1 SPARSE videos; all-click initialization failures
in0027 and0012 are retained, leaving15 FIT and7 INNER usable video clusters.
Nine actual full-video metrics are reported for C0 and eight fixed simple
policies, with paired video bootstrap intervals, target recall, N01/N10
and verified-other takeover. Clicks/seeds stay inside video. A singleton
has no informative video-cluster CI. Candidate density is **not** GT
people count. These are original full-video evaluations stratified by
density; posthoc masked TrackEval is explicitly **NOT_RUN**, not silently
substituted for complete trajectories.

Selected descriptive C0 values (fraction units, not percentage points):

| Split / original density | Usable videos | HOTA | Target recall: available / visible |
| --- | ---: | ---: | ---: |
| FIT / CROWDED | 3 | 0.280435 | 0.704546 / 0.276456 |
| FIT / MEDIUM | 5 | 0.536532 | 0.606725 / 0.535168 |
| FIT / MIXED | 7 | 0.413217 | 0.579849 / 0.427126 |
| FIT / SPARSE | 0 | undefined | undefined |
| INNER / CROWDED | 2 | 0.286453 | 0.565667 / 0.352253 |
| INNER / MEDIUM | 2 | 0.335561 | 0.641078 / 0.194313 |
| INNER / MIXED | 2 | 0.415664 | 0.653518 / 0.491446 |
| INNER / SPARSE | 1 | 0.208799 | 1.000000 / 0.278546 |

`data/VIDEO_DENSITY.json` contains the full40-sequence scope census;
confirmation entries remain unopened, not invented effect records.
`mot/DENSITY_BASELINE_SIMPLE_FULL_VIDEO_V1.json` contains the actual
all24 C0/simple tables. MAIN and matched-state effect tables remain pending
their actual own full-video results.

The separate frozen M10 original-scene descriptor protocol
(`data/SCENE_CHARACTERISTICS_PROTOCOL_V1.json`, SHA256
`f5df320fe6aa369558385ec12d17ec9dce40f942658c7ade6b1efb420b8cc932`)
also completed all24 FIT/INNER videos. It verifies sealed actual C0 and
candidate/anchor/GT hashes before descriptive analysis. It reports
original-frame GT people counts, actual visibility annotations, GT-box
overlap proxy, current verified-other/UNKNOWN competition, positive
candidate ownership, physical/strict-candidate observational returns,
and raw-anchor cosine margins. UNKNOWN is not a verified competing
identity; UNASSIGNED is not another public identity's ownership. These
offline observations are neither deployable GT-aided features nor proven
independent causal roots. No new online rollout, representation, memory
training, density relabeling or threshold selection is performed.

The sole sparse video0052 has2–4 GT people (mean3.709). Its three clicks
have strict-positive available exposures307/311/309 versus physically
visible1119/1196/1013; their verified-other raw-anchor win fractions are
286/307,166/311 and229/309. This descriptive variation and single-video
support do **not** establish that fewer people yield safer MOT intervention.
All24 videos have constant GT visibility annotation1; box overlap is
explicitly a geometric proxy, **not physical occlusion ground truth**.

All24 fixed zero-authority memory results are now sealed. The actual
frontier is still partial because learned risk-writer full-video results
are pending. Across15 usable FIT videos, MEAN and MULTICUE pooled
wrong+UNKNOWN write rates are49.06% and24.98%, with correct-observation
retention100% and12.39%. Across7 usable INNER videos they are69.94% and
25.27%, with retention100% and18.50%. None satisfies the2% contamination /
60% retention point gate; these pooled exposure rates are not independent
event confidence claims. HOTA equality is zero-authority isolation, not
evidence that writing improves global association. The registered fresh
risk-head optimization has completed all6 actual fits after all24 inputs
became available (three LOGISTIC and three MLP seeds, nonzero gradient steps,
129/1475 changed weights respectively, strict loaded INNER scores equal).
All six selected sampled teacher-state points abstain; the launched own-bank
full-video audit is still required and actual own-policy write safety remains
unresolved. Zero accepted writes
have undefined contamination risk and cannot qualify G4.

The completed V17 full regression reports **1048 passed /5 unchanged
historical failures** (85.72s, actual shell/pytest exit1), and all123
R2-focused tests pass (4.95s, observed exit0), preserving the
four pinned scalar-SEQMAP CLI incompatibilities and the historical
branch-literal expectation. V16's completed log/JUnit were recovered
(1043 passed /5 failures), but its original terminal exit status was not
available after context compaction and is **not claimed observed**.
Only code, tests, this necessary report and two compact frozen protocols
are included in the V9 Git checkpoint; data, weights, GT, embeddings,
runtime outputs, media and bulk result manifests remain local.

## Complete registered-group closure protocol and first risk video (V10)

The unchanged FINAL GOAL remains ACTIVE; scientific closure is PENDING.
All14 registered MAIN architecture/objective groups (42 weights) and three
controlled state-source groups (9 weights) now have a frozen whole-group
summary protocol: `mot/DEVELOPMENT_CLOSURE_PROTOCOL_V1.json`, SHA256
`f1c7e4c7e590a26ec82f9c1fd631740718ab27f014ed86eadd41ed20074a1352`.
It adds no optimizer, threshold, candidate, identity-memory or association
choice. It was frozen before any MAIN/state-policy full-MOT outcomes.

Every group requires both original INNER operating points on every planned
video and seed, the exactly recomputed shared selection, then selected-point
FIT16 on all three seeds. Missing/failed initializations remain in the census;
no ready/successful subset is reported as complete. Actual source/weight/
runtime/onset/log/artifact hashes, nonzero optimization, original density,
full zero-action tensor/trajectory C0 equality and paired metrics are checked.
Pinned `tracker`/`per_sequence` metadata are kept separate from nine scalar
metrics; percentage-scaled rates and fractional FP/FN/IDSW are rejected.

The summary gives full-video absolute/paired nine metrics, per-video and
original-density results, video-cluster intervals, target recall, N01/N10,
non-target origin proxies and every actual one-shot action label. G2 numerical
targets and the CI are reported separately; zero baseline IDSW allows no
increase. G1 uses only valid negative upper bounds on possible independent
roots, original N01>N10 and severe-harm criteria. Enough action rows or spaced
windows **never prove independent causal roots**; no new per-seed minimum is
introduced. Event precision/recall remain undefined until separately verified
roots exist. These summaries never authorize confirmation or close the entire
task. The first sealed census truthfully contains0 complete /17 pending groups.

Fresh risk-writer full-video evaluation has now completed the first FIT video,
0074, all six fitted heads, both registered points and all three clicks.
Its36 actual full-video tracker outputs have all nine pinned TrackEval metrics
equal to C0; changing memory with zero association authority is isolation,
not MOT improvement. All six selected operating points have0 actual accepted
writes, undefined contamination risk and0 retention. No safety PASS is claimed.

At the fixed0.5 **UNQUALIFIED DIAGNOSTIC** point, the six individual seeds have
wrong+UNKNOWN write rates30.27%–35.24% and correct-observation retention
61.64%–76.04%. After averaging seeds inside this one video, LOGISTIC/MLP pooled
risks are32.30%/34.74%, retention68.22%/71.30%, and available-target Rank-1
deltas versus frozen memory are+0.051073/+0.054023. Better descriptive identity
ranking therefore coexists with unacceptable writes at these points. This is
one FIT development video, not independent generalization or population2%
safety evidence. The remaining23 videos are still required; the immutable
frontier snapshot is PARTIAL, fixed24 / fresh-risk1.

V18 actual full regression: **1055 passed /5 unchanged historical failures**,
65.22s, observed exit1. All130 R2-focused tests pass,6.82s, observed exit0.
Seven new closure tests cover complete seed/video grids, failed initialization,
metric metadata and units, exact C0 pairing, current/future onset completeness,
no zero-action success, UNKNOWN separation and no frame/window-as-root claim.
The historical tests and pinned third-party source remain unchanged.

## Dead state-source owner and optimizer-evidence census (V11)

The same FINAL GOAL is ACTIVE; the scientific decision remains PENDING.
Fresh process/cwd checks found that tensor-recovery V2 owner2990146 and
its marked child3220398 were no longer present while the old marker still
said ACTIVE. Its terminal exit code was unavailable after the handoff:
**no exit code, signal, OOM cause or scientific failure is inferred**.
The immutable dead-owner observation retains hashes of the old marker,
log, completed0037 click0 receipt and both runtime files, including the
interrupted click1 partial. These are neither overwritten nor adopted as a
completed video.

A separately frozen V3 supervisor repeats the **same nineteen registered
V1-failed videos** through the unchanged frozen V2 runtime and offline-label
code, in separate output/log/asset paths. It changes no source-case/frame,
action, sole click, candidate, feature, label, identity memory or association.
The original five complete videos and all original failure evidence remain
sealed. Only after both new full-video receipts and their artifacts verify
can a previously absent canonical receipt be installed. This recovery is
engineering preparation, not model-generated on-policy closure or G1/G2 PASS.

The new census distinguishes fit-attempt records, measured no-optimization
supervision failures and verified actual optimizer models. A model counts
only with a registered successful status, matching frozen protocol and source
hashes, positive optimizer/nonzero-gradient steps, changed weight elements
and a present selected checkpoint with its exact recorded hash. It supports
the actual integer/per-tensor/MAIN-per-tensor change schemas. A
`FAIL_MEASURED_STATE_SOURCE_SUPERVISION_DIVERSITY` record with0 optimizer
steps is retained as an attempt, **never counted as a trained model**.
Unverifiable records remain separately visible; no successful subset becomes
full experiment completion, deployment permission or scientific success.

The first live verified census has17/42 registered MAIN fits,6 pilot fits,
12 current-axis fits and6 fresh memory-risk fits. Controlled state-source
fits remain0/9 pending all24 sources. All26 new recovery/census tests pass
(4.13s, observed exit0).
V3 has started0037 under a new live owner, but its full video is not yet
complete. CONFIRM/VAL/TEST/SOT remain unopened.

V19 complete regression: **1081 passed /5 unchanged historical failures**,
78.308s, actual pytest exit1 observed by its supervising subprocess. The
156 R2-focused tests pass (8.707s, observed exit0). The five failed test
identities match V18 exactly; no skipped tests or errors, no historical
expectation or pinned third-party edits, and no claim that all tests pass.

The next live census advances to19 actual MAIN fits and5 sealed full-MOT
cells for SCALAR/730101/P0. This includes0012 with all original click
initializations failed, **not a successful zero-action evaluated video**.
The other four videos (0096/0008/0016/0006), nine valid clicks, have0 effective
interventions and0 own one-shot labels; their measured all-nine deltas are
zero. This is vacuous C0 preservation, not causal identity benefit or a
complete three-seed/point/group comparison. The independently verified
memory frontier now contains fixed24/fresh-risk3, still explicitly PARTIAL.

## Actual data deliverables and full-task semantic census (V12)

The FINAL GOAL is unchanged and ACTIVE. Whole-task scientific closure remains
PENDING, not redefined around the completed preparation or code delivery.
The task's74 named files, six final tables and35 engineering-test requirements
are now explicitly inventoried. Presence of a file, a placeholder, a pass total
or a narrow preparation proof is never counted as full-task completion.
At this checkpoint13 requested files exist, five have a narrowly verified
configuration/input-preparation scope, and **none of the six final tables is
yet claimed semantically complete**. Existing versioned research evidence must
still be summarized and verified, alongside remaining actual experiments.

`data/CANDIDATE_EXTRACTION.json` and `data/CANDIDATE_INTEGRITY.json` now summarize
all16 original FIT and8 original INNER videos: **24,890 complete original
frames and178,975 real candidates**. Existing tapes were decoded again through
the unchanged frozen integrity audit, checking complete original frame axes,
unique current UIDs/embedding offsets, finite features, stored tape/index/done
hashes, original geometry-count density and no runtime GT. SAM3/OSNet checkpoint
file hashes were independently reread. No extraction, runtime rollout,
training, threshold selection, future/confirmation truth access, raw-image copy
or candidate change was added. The stored f16 tape hash is not misrepresented
as reproduction of the original pre-storage f32 feature hash.

The original extraction manifest's empty `completed_sequences` was a frozen
preparation record, not the current completion census; it remains untouched.
The two requested completed data reports explicitly link that manifest and
all24 actual source receipts/assets. Original per-video integrity seals were
recomputed identically, not rewritten or replaced.

The all40-role Table1 **preparation** report retains the unopened CONFIRM8 and
already-exposed historical8 without fabricating fresh results. Across the
24 new videos there are72 registered sole-click attempts,59 valid and13
retained failures: FIT41 valid/7 failed; INNER18 valid/6 failed. Videos0027 and
0012 have no valid clicks and are not successful zero-action evaluations.
Every valid click has actual same-input baseline coverage. Video-local GT
identity counts are explicitly **not independent cross-scene people**;
independent correction/N10-root counts remain unavailable, not filled with
frame runs, click counts or seeds.

The initial new report adapter incorrectly expected a top-level initialization
`sequence`; the frozen schema instead stores it in each click. Its observed
exit1, exact failed code hashes and absence of aggregate reports were retained
in `audit/DATA_DELIVERY_V1_SCHEMA_FAILURE_ATTEMPT1.json`. Only this new adapter
was repaired and regression-tested against the actual headerless/per-click
format. Scientific inputs, initialization, candidates, memory, matcher,
association, training code and frozen experiment protocols were not edited.

Fourteen new tests cover all74 names/six tables/35 test semantics, matching
input/role/click lineage and candidate axes, retained failed initializations,
unopened confirmation, unproven independent units and the distinction between
file presence/narrow proof/full completion. V20 complete regression:
**1095 passed /5 unchanged historical failures**, actual pytest exit1 observed.
All170 R2-focused tests pass with observed exit0. Failure identities match V19;
no historical expectation or pinned third-party change, no all-tests-PASS claim.

At the preparation checkpoint MAIN optimization has advanced to35/42 verified
fits and11 own-policy full-MOT cells, with0/17 whole registered groups complete.
Controlled sources remain5/24 while the isolated V3 worker progresses through
0037 click2; no partial is promoted to a full source. Fresh risk replay has
completed4/24 videos, and its fixed24/fresh-risk4 uncertainty snapshot remains
PARTIAL. These independent registered branches continue. CONFIRM/VAL/TEST/SOT
remain unopened; no on-policy/M-B/confirmation authorization is inferred.

Only code, tests and this necessary explanation are included in V12 Git
delivery. The generated data summaries, Table1 preparation, requirement ledger,
weights, raw data and large evidence remain local.

## V13 actual all24 event-label evidence, not scientific closure

FINAL GOAL remains **Event-Level Causal Identity Association: Safe One-Click
Intervention with Fresh-Sequence Generalization for Online MOT**. The full
original M0-M11 task remains ACTIVE/PENDING; next-stage authorization is false.

The new offline auditor `scripts/n72r21r2_event_delivery_v1.py` completed the
original FIT16+INNER8, with actual supervisor exit0 observed. It verified every
registered runtime seal and raw arm SHA before opening that video's original
GT, reconstructed current candidate matching and the original source-prefix
public identity-origin proxy, then recomputed **every label field** from actual
paired trajectories. Labels match exactly for all17,567 arms and1,665,342
current-plus-future **correlated arm-frame observations**. This does not add
independent data or certify policy improvement. Current t, future k=1..H,
UNKNOWN, verified OTHER, incomplete futures, duplicate action configurations,
other-track harm, takeover, fragmentation and available motion/prototype
measurements are kept distinct. The origin proxy excludes clicked-person
fragments and is not an independent-person or global-IDF1 ground truth claim.
Own KEEP outputs and semantic states match their original sealed source prefix
through every selected window. Neither runtime, classifier, association,
candidate generator nor training inputs were modified.

Six requested local reports now exist: `events/EVENT_SCHEMA.json`,
`EVENT_CORPUS.json`, `EVENT_LABEL_AUDIT.json`, `DIRECT_VS_PROPAGATED.json`,
`COUNTERFACTUAL_BRANCHES.json` and `ACTION_FEASIBILITY.json`. Their references
point to the actual immutable source tapes and24 recomputation receipts. They
do not turn the old corpus's preparation-time pending annotations into new
results or replace failed attempts. Original0099/0082 context/label failures,
partial tapes and logs were rehashed and retained alongside successful repairs.
All13 failed sole-click initializations remain, including the two all-failed
videos; empty arms are not successful zero-action scientific evaluations.

There are2,503 registered CF positions,15,665 complete H100 arms,8,054 duplicate
executed-action configurations and792 observed safe-positive H100 arms
(FIT570, INNER222). None is an independent beneficial correction-root count.
The177 overlapping-window union components across video-local anonymous
identities are correlation bookkeeping, not causal boundaries or independent
bootstrap units. Independent beneficial/N10 roots and event precision/recall
remain null. The original7,611 observed-harm counter also includes observed
current harm in incomplete arms; it must not be presented as a complete-H100
population risk denominator. Full-video own-policy results and independent
root proof or valid negative bounds are still required for G1/G2.

An important evidence boundary is explicit:15,632 arms have actual current
tensor-inclusive prestate agreement with their sealed context. The1,935 arms
in the three original pilot videos lack that retrospective tensor proof and
retain semantic-only prestate evidence. No new source replay or invented
tensor hash upgrades them. The actual22 representative CLI windows cover154
arms; two all-failed videos are correctly NOT_RUN. The original189 pilot
API/CLI comparisons across all nine pinned metrics remain exactly equal.
Current-inclusive101-frame CLI windows and future-only100-frame trajectory
utilities are different evidence, and overlapping window HOTA is never summed
into a whole-video policy effect.

V22 complete regression on current event/utility source: **1121 passed /5
unchanged historical failures**, actual pytest exit1 observed. All196 current
R2-focused tests pass, with observed exit0. Both supervisor completion and
log/JUnit hashes were recorded. The required `tests/FOCUSED.json` and
`tests/REGRESSION.json` reference these actual runs; they do not infer coverage
of all35 required runtime semantics merely from pass counts. The earlier V21
run (1113/5, focused188) and the separate8 utility tests are retained, not
retroactively recounted. A compile-check syntax error in the new reporter was
fixed before its protocol or runtime existed; no frozen scientific evidence
was changed by that correction.

Only the new source, tests and this necessary explanation are included in V13
Git delivery. All generated event reports, raw GT, labels, trajectories,
weights and bulk assets remain local. Controlled source recovery, exact-current
availability diagnostics, fresh memory-risk replay and registered own-policy
full-MOT groups continue; these evidence reports do not finish the Goal.

### Actual component benefit is not actual global association benefit

The subsequent `scripts/n72r21r2_component_global_utility_v1.py` run also
completed all24 original videos with actual exit0 observed. It pairs every
sealed arm label with its actual pinned future-only H100 nine-metric utility,
retains incomplete and duplicate records explicitly, and reports all arms,
distinct action configurations and distinct effective one-shot arms separately.
No threshold, model, best action, seed or video was selected from this analysis.
This is **POSTHOC DIAGNOSTIC, NOT POLICY SELECTION**.

Among complete H100 distinct effective arms, FIT contains242 component-safe
positive arms, but only32 improve both actual window HOTA and AssA;49 instead
reduce at least one of them. INNER contains102 component-safe positive arms,
but only3 improve both global window metrics;25 reduce at least one. Thus
identity-component benefit and prestate-origin-proxy non-target safety do not
guarantee positive global association utility. All3 INNER positive-global arms
come from0016, not three independent sequences. These are still correlated
window-arm counts, not independent corrections, population precision, policy
qualification or whole-video results. The local receipt is
`events/COMPONENT_VS_GLOBAL_UTILITY_DIAGNOSTIC_V1.json`; all raw pair identities,
hashes, video/role counts and incomplete-window handling are retained. The
frozen42 MAIN fits, original operating points and pending full-policy evaluation
are unchanged; this diagnostic does not authorize confirmation or select a
scientific final decision.

## Execution boundaries

Reuse existing DanceTrack TRAIN images, SAM3/OSNet weights and original Python environment. Preserve historical evidence. All runtime decisions use only current real candidate UIDs and causal state. Sole click initializes identity, not unconditional override authority. Every intervention uses the complete global assignment and committed feedback. GT labels stay offline. Frames, seeds and propagated errors are not independent causal events.

Only code, tests and necessary compact documentation are destined for GitHub. Data, embeddings, weights, media and bulk traces remain local. Completion requires actual experiments and final evidence-backed closure, not this bootstrap record.
