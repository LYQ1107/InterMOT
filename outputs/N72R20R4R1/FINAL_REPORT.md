# N72R20R4R1 — Final Scientific Report

FINAL GOAL: Global Counterfactual Association Opportunity Discovery and Safe Persistent Identity Intervention.

CENTRAL QUESTION: 用户点一下这个人以后，我们能否依靠长期身份记忆，准确识别哪些在线关联值得纠正、何时应当推翻短期跟踪证据，并将这种纠错持续转化为 HOTA 和 AssA 的提升？

Canonical immutable goal: `FINAL_GOAL.json`; methods: `PREREGISTRATION.json`. Decision: **FAIL_VAL_GENERALIZATION**. Reliable positive transfer is unconfirmed; statistically significant harm is not proved. Scientific success and next-stage authorization false. M0–M13 and R0–R8 execution complete, with the explicit zero-activation positive-history learning exception.

## First substantive section — Q1–Q6

### Q1 — Why were all 9,972 R4 samples negative?

R4 used one target per sequence and a strength-0.5 residual, not a fixed edge. Labels required changed assignments; independently recomputed certificates block the strongest registered residual on 7,451/7,452 assigned frames. This is insufficient action space/strength, not proof that associations are correct. The new solver fixes an edge, removes its candidate row/public column, solves the entire remainder with NONE and commits every affected state, revealing natural positives. The identity/availability utility still is not identical to HOTA improvement.

### Q2 — How many global correction opportunities exist?

Fifty-one sequence-local identities give 2,923 sampled identity-frame events and 18,667 feasible non-KEEP actions: H5 beneficial 997, harmful 16,879, neutral 791. Positives comprise 481 immediate available-person corrections, 400 valid NONE/availability decisions and 116 other future benefits, not 997 independent reacquisitions. Oracle-correct beneficial actions number 881. GT-free Adapter/raw-anchor/base-score union covers 656/997, 893/997, 985/997 at K1/K3/K5; actual inner-selected K covers 900/997 (90.27%). These posthoc diagnostics use strict six-fit Adapters, not oracle runtime or heldout selection. Missing-candidate events have no correct edge despite feasible rejection/wrong alternatives.

### Q3 — When does short-term history lock wrong associations?

Exact float32 source score: 1.5×appearance + predicted IoU + 0.5×native − 0.1×gap, followed by native bonus 3, optional positive bonus 5 and immutable −1e9 hard negatives. Native contributes 3.5. Of 511 sampled wrong assignments with a correct candidate, 501 activate native evidence. Wrong-minus-correct local-score medians on 0002/0023/0062/0072: 4.500/4.445/4.401/4.315; correct-edge global-regret medians: 8.948/8.831/8.805/8.450. Target-native removal changes 16 current assignments and corrects only two: global competition remains.

Actual source positive-history/hard-negative activation is zero. Synthetic preservation tests are not learning evidence. Positive reliability and the combination of two learned native/positive components are NOT_RUN: the positive component is unidentifiable. Real positive-soft-zero equivalence and native-only/full-joint controls are executed instead. Explicitly posthoc target-scope native-zero replay shares the 0062 gain but harms 0023/0072; it is not a selected policy. Global scalar and target-conditioned discount have different scopes, preventing isolated claims about learning. Native heads fitted on real P1 states extrapolate on P0; observed state-feature shift is disclosed, not tuned away.

### Q4 — Does C6 outperform simpler controllers?

No demonstrated causal/full-MOT advantage. All strict fit folds pass natural-label gates. C2/C3/C6 ensembles reproduce baseline; C6 L0–L3 do not improve trajectories. C4 makes 11 immediate changes, N01/N10=0/175; C5 seven, 0/485, with lower HOTA. Structured benefit/damage/write-risk heads are genuinely trained, but fit loss is not success. C5 seeds 720321/2/3 yield N10=506/319/159, HOTA 0.57881348/0.57692127/0.57709836; all C6-L0 single seeds remain baseline.

The strongest DEV signal is separate 33-parameter native reliability, not C6/GRU writes: N01/N10=489/0, ΔHOTA +0.00635119, concentrated in 0062 under P0. Adding C6 does not improve that trajectory. Nested final selection gives 489/16 and a slightly smaller gain; writes elsewhere do not establish a marginal memory benefit.

### Q5 — Can useful memory and safe writes coexist?

Not at frozen ≤2% wrong-write and ≥60% retention goals. P1 raises competitive Rank-1/strict hard-negative win rate 22.97%→68.53% but accepts 7,452 writes, 2,579 wrong (34.61%). P2–P6 also fail (Table 3). P6 causal rollback reaches 18.28%/52.55%; rollback is never selected using GT. P1 per-sequence mean divergence from anchor ranges 0.666–0.956; P6 0.650–0.866. P0 drift is recorded null because no write hook emits it; its query is frozen by construction.

Competitive Rank-1 is strict candidate ordering with ties as losses, not full-MOT identity accuracy. Primary one-to-one IoU≥0.5 labels classify unmatched observations as non-target/invalid; they do not prove every unmatched crop depicts another physical person. Secondary max-IoU diagnostics do not replace primary labels or import R4's older totals. Final DEV writes 260 (243 correct/17 wrong): wrong rate 6.54%, retention 4.55%, FAIL. VAL P0 zero writes mean undefined wrong rate/0% retention, not safety PASS.

### Q6 — Did HOTA/AssA really improve?

DEV full nested-selected trajectories: HOTA 0.5808919181→0.5872323133 (+0.0063403952), AssA 0.5433304753→0.5551611833 (+0.0118307080), DetA +0.0000111112. Point gate passes; paired eight-sequence macro ΔHOTA 95% CI [−0.0001304818,+0.0254290194] crosses zero. Official combined metrics are re-evaluated over complete outputs/GT, not means/sums of fold/window HOTA. Gains concentrate in 0062; 0072 slightly degrades.

Per section 26, point signal authorizes frozen VAL confirmation, not science PASS. Unified native-only C0/P0 is selected solely from inner results (mean inner ΔHOTA +0.00678711 and deterministic simplicity tie-break), not outer gains. Three heads refit on all eight TRAIN sequences and historical all-eight-TRAIN Adapter SHAs/splits are frozen before new VAL input. VAL is historically exposed, not virgin, not tuned; TEST untouched.

All 25 VAL sequences: HOTA 0.4878651521→0.4871461418 (−0.0007190103), AssA 0.5118588672→0.5103823298 (−0.0014765375), DetA −0.0000361023, IDF1 +0.0000676340, IDSW 1832→1833. Paired macro HOTA CI [−0.0014836671,+0.0003238601] crosses zero. Independently reconstructed N01/N10=35/0, 527 changed output frames and zero conflicts do not become global HOTA gain. Native-inclusive correction onsets persist H5 2/6, H10 1/6, H30 0/3. Non-target damage proxy is zero, showing it does not replace official global trajectory matching. Positive generalization is unconfirmed.

## Table 1 — Opportunity Discovery

Unit: sampled sequence-identity-frame events. Feasible means any legal non-KEEP action, not guaranteed recovery. Beneficial/harmful count events with any corresponding H5 action and may overlap. Competitor-owned is nonexclusive; do not sum rows. Runtime uses inner-selected K, never oracle selection.

| Event Family | Total | Feasible | Runtime Proposed | Beneficial | Harmful |
|---|---:|---:|---:|---:|---:|
| Wrong Match | 511 | 511 | 511 | 428 | 511 |
| NONE Recovery | 80 | 80 | 80 | 71 | 79 |
| Competitor Owned | 591 | 591 | 591 | 499 | 590 |
| Missing Candidate | 591 | 591 | 591 | 361 | 540 |
| Valid Rejection | 91 | 91 | 91 | 39 | 90 |

Already-correct 1,650 events remain in the corpus. Missing-candidate positives concern NONE/availability, not nonexistent-person recognition. Action-level details: `opportunity/FINAL_EVENT_TABLE.json`.

## Table 2 — Association Authority

Changed: complete-trajectory output-ownership differences, including propagated/native-only effects. Immediate action changes separate. N01/N10: independent future target-frame counts. H30: successful/eligible correction-episode onsets; 0/0 is undefined.

| Method | Changed | Immediate Changes | N01 | N10 | Persistent H30 | ΔHOTA | ΔAssA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed Weight C1 | 0 | 0 | 0 | 0 | 0/0 | 0 | 0 |
| Learnable Scalar C2 | 0 | 0 | 0 | 0 | 0/0 | 0 | 0 |
| Logistic C3 | 0 | 0 | 0 | 0 | 0/0 | 0 | 0 |
| MLP C4 | 717 | 11 | 0 | 175 | 0/0 | −0.00380262 | −0.00711239 |
| Action Ranker C5 | 510 | 7 | 0 | 485 | 0/0 | −0.00230763 | −0.00431834 |
| Structured C6-L0 | 0 | 0 | 0 | 0 | 0/0 | 0 | 0 |
| Native reliability only | 810 | 0 | 489 | 0 | 5/18 | +0.00635119 | +0.01186923 |
| Nested final selection | 1353 | 6 | 489 | 16 | 5/18 | +0.00634040 | +0.01183071 |

Counts/proportions/failure reasons/per-sequence breakdown: `authority/FINAL_FUNNEL_DIAGNOSTICS.json`, `authority/INTERVENTION_FUNNEL.json`, `authority/outer/`. Native discount can alter the solver without authority approval. Mixed action/frame/episode units are not one monotonic denominator.

## Table 3 — Safe Memory

Actual inner-selected policies on strict outer sequences. Wrong denominator: accepted writes; retention: eligible correct observation opportunities on each policy trajectory. All rows FAIL joint safety; competitive Rank-1 equals strict hard-negative win criterion here.

| Policy | Accepted Writes | Wrong Rate | Correct Retention | Rank-1 / Hard-Negative Win Rate | HOTA |
|---|---:|---:|---:|---:|---:|
| P0 | 0 | undefined | 0% | 22.97% | 0.58089192 |
| P1 | 7452 | 34.61% | 100% | 68.53% | 0.58089192 |
| P2 | 684 | 24.12% | 10.65% | 30.76% | 0.58089192 |
| P3 | 3540 | 21.72% | 56.86% | 49.73% | 0.58089192 |
| P4 | 2941 | 18.57% | 49.15% | 44.77% | 0.58089192 |
| P5 | 752 | 65.43% | 5.34% | 28.86% | 0.58089192 |
| P6 | 3134 | 18.28% | 52.55% | 47.14% | 0.58089192 |

Memory-only full trajectories stay baseline despite changed ranks. Actual inner Pareto frontiers, denominators, drift and rollback events are preserved in `memory/`.

## Table 4 — Full MOT

Complete dynamic-lifecycle outputs/all GT/identical pinned evaluation. “Safe” names a tested family, not achieved safety. Joint ablation, nested final selection and unified VAL winner are distinct.

| Method | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|---|---:|---:|---:|---:|---:|---:|
| R4 baseline reproduced | 0.58089192 | 0.54333048 | 0.62309117 | 0.61754943 | 0.57855065 | 250 |
| Adapter only / P0 | 0.58089192 | 0.54333048 | 0.62309117 | 0.61754943 | 0.57855065 | 250 |
| Selected safe memory + C6, no native discount | 0.58089192 | 0.54333048 | 0.62309117 | 0.61754943 | 0.57855065 | 250 |
| Fixed association C1 | 0.58089192 | 0.54333048 | 0.62309117 | 0.61754943 | 0.57855065 | 250 |
| Learned association C6-L0 | 0.58089192 | 0.54333048 | 0.62309117 | 0.61754943 | 0.57855065 | 250 |
| Full joint C6 + native + selected memory | 0.58724311 | 0.55519971 | 0.62307914 | 0.62790948 | 0.57856954 | 249 |
| Nested final selection | 0.58723231 | 0.55516118 | 0.62310228 | 0.62775281 | 0.57847508 | 254 |
| Raw identity control | 0.58089192 | 0.54333048 | 0.62309117 | 0.61754943 | 0.57855065 | 250 |
| Shuffled identity control | 0.58089192 | 0.54333048 | 0.62309117 | 0.61754943 | 0.57855065 | 250 |
| VAL baseline, 25 sequences | 0.48786515 | 0.51185887 | 0.46917170 | 0.52118273 | 0.34328975 | 1832 |
| VAL frozen treatment | 0.48714614 | 0.51038233 | 0.46913559 | 0.52125036 | 0.34328531 | 1833 |

Full joint matches native-only trajectories despite writes elsewhere: no marginal C6/GRU benefit established. No favorable outer metric chooses VAL policy.

## Controls, uncertainty and objective limitations

Folds: 0001/0002/0023/0024/0039/0057/0062/0072; six fit, cyclic next-sequence inner, one outer. Policies frozen before outer input. Thirty-six variants yield 288 sequence-variant records; exact configuration aliases reuse a verified real rollout, not guessed metrics. SHA-bound ownership deltas reconstruct full trajectories from original geometry. Source A/A, new KEEP, P1/P2 trajectories and all nine original official metrics reproduce. Independent audits verify zero conflicts, pre-click/click preservation and post-runtime-only GT evaluation.

Point gate and CI reporting are separate preregistered fields, neither weakened. Bootstraps: 2,000 paired sequence-cluster draws, seed 720401, eight DEV/25 VAL clusters. Macro CI differs from official combined delta. DEV 0062 HOTA 0.46449913→0.53230984; 0072 0.15468034→0.15433239; other final-selection sequences have unchanged official metrics. Complete per-sequence/all-variant metrics and other CI: `dev_trackeval/`, `val/RESULT.json`, `val/DELTA.json`.

All actions have independent H1/H5; 679 SHA-stratified windows have H10/H30/all-GT/all-output TrackEval. H10 positives/harmful 212/310; H30 237/317, not population estimates. Of 199 H5-positive-stratum windows, HOTA rises in 34, falls in 89, equals in 76; mean local delta −0.00249878. Identity/availability/damage/write-risk utility is not consistent full-MOT reward. Baseline-origin damage can depend on birth-number changes and does not capture all global trajectory semantics. No posthoc label/model/threshold changes; overlapping window gains are not summed as full HOTA.

## Models, provenance and latency

New checkpoints: 384 C2–C6 (three seeds, epochs 10/30, C2/3/4 L0, C5 L1, C6 L0–L3, eight folds), 48 six-fit native/write, three all-eight-TRAIN VAL native heads. All 435 SHA/split/feature allowlists verified. Parameters: C2 3, C3 75, C4/C5 2065, C6 2101, ten-feature reliability 33. No backbone/decoder training. Twenty-four six-fit source Adapters and three all-eight-TRAIN VAL Adapters independently checked. Paths/SHAs: `checkpoints/CHECKPOINT_MANIFEST.json`.

Frozen GRU SHA `94ba9c44e177b254ee54984195e839f37c78a648c010ffc723dd3f1d553976a2`; encoder `2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154`. Source `36b0e198fa084ce1288fd134736bea159a2ea731` verified remotely before work; misleading older Adapter metadata not accepted as six-fit proof. Historical hashes unchanged.

Before inner/outer results, deployment neutralized fitted-constant P0 auxiliary-state columns to avoid unfitted coefficients pretending to learn memory reliability. Fit normalization/weights preserved; five family tests reproduce committed inference bit-for-bit. Separate P1 reliability uses genuinely varying causal states, with P0 deployment-shift caveat. Later changes only route VAL splits/add posthoc/integrity analyses/isolate fixtures; frozen DEV numerics unchanged.

VAL CPU replay: 300.90 s baseline/514.23 s treatment over 25,508 frames, 11.80/20.16 ms/frame. Excludes candidate generation/batch Adapter projection; NOT live SAM3 FPS. Per-sequence latency/continuity/native-inclusive persistence: `val/OWNERSHIP_AND_PERSISTENCE.json`.

## R0–R8 closure

Actual evidence: R0 oracle/runtime K coverage; R1 force-edge/inner strength grids; R2 fixed/global/conditioned native/zero-activation positive exception; R3 Adapter/raw/shuffled/memory ranking; R4 ownership/full MOT; R5 P0–P6/Pareto/causal rollback; R6 C2–C6/four losses/heldout values; R7 independent persistence; R8 visible-but-uncovered candidates. Details: `REPAIR_DIAGNOSTICS_R0_R8.json`. Conditional T6 audit finds misleading Adapter ranking in 339/411 competitive sampled runtime-proposal events, not an unconditional future Rank-1 rate.

Secondary bottlenecks: controller value generalization, contamination/retention, discrimination, persistence, P1→P0 native shift and proxy-versus-MOT mismatch. Engineering invariants PASS is not science PASS. Confirmations are simulated from GT, not a real-human study. Positive-history learner explicitly NOT_RUN for zero activation; actual equivalence/preservation alternatives executed.

## Regression, storage and delivery

Stage 85 passed; stage/R4 related 116 passed. Activated full suite **714 passed, 5 failed, 4 warnings**. Four historical direct-CLI TrackEval calls pass list `SEQMAP_FILE` to `stat`; one historical test asserts an older branch literal. Old tests/third-party preserved. Formal compatibility entry reproduces all nine source metrics. Genuine commands/stdout/SHA: `audit/REGRESSION_RESULTS.json`. Earlier unactivated PATH also caused two missing-python errors; these are absent in final activated run.

No dataset/checkpoint/environment rebuild/download, TEST, live backbone, LoRA, historical deletion or unrelated GPU termination. Quota free declined from about 100.1 GiB below unchanged 98 GiB experiment floor; writer attribution unmeasured. No new experiments/caches. Only existing-evidence reports/seals/Git delivery proceed. External assets about 12 MiB, below 48 MiB cap. Owned evaluator temporary inputs removed after sealing; historical assets unchanged.

Branch `codex/n72r20r4r1-global-opportunity-safe-authority`; authenticated SHA-identical non-force GitHub API fast-forwards, no SSH-authentication success claim. External per-HEAD receipts bind final SHA without self-reference. Exact local/remote equality and clean tree are verified before Goal closure.

Stop here. Separately authorized future work could test trajectory-aligned train-only supervision and deployment-matched reliability before a new frozen evaluation. This is a recommendation, not automatic next-stage work. `NEXT_STAGE_AUTHORIZED=false`.
