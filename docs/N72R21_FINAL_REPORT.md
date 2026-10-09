# N72R21 final evidence report — MOT primary, SOT deferred

Generated UTC: 2026-10-09T15:04:41.382140+00:00. Scientific decision: **FAIL**. Next stage authorized: **false**.

## Frozen Goal and latest direction

FINAL GOAL: One Click, Persistent Identity: Causal Long-Term Human Tracking Across Occlusion, Reappearance and Independent Recordings

CENTRAL QUESTION: “用户只需要点击一个人一次，在其离开画面、重新出现、换衣服、跨摄像头或跨独立录制时，系统能否可靠判断他是否出现、在哪里，并持续认住他，而不把其他人误认为他？”

The latest explicit user instruction makes interactive **MOT the primary track** and defers SOT/LaSOT. The original Goal is retained unchanged for provenance; `RESEARCH_DIRECTION_OVERRIDE.json` takes precedence. No new SOT downloads, training, inference or OOM retries were started after the direction change.

## Outcome and scientific limits

The current frozen learned system has **not established reliable persistent identity or a beneficial MOT intervention**. On two preregistered TRAIN sequences, all 12 learned full-MOT conditions have lower HOTA, AssA and IDF1 and higher IDSW than clicked C0. All 16 conditions have zero registered strict H100 future benefit. These are real joint multi-object trajectories, not independently predicted target episodes merged into a tracker.

The completed 25-sequence frozen VAL is a **single-click identity-subsystem evaluation on shared cached SAM3 candidate tapes**, not full MOT, an end-to-end online SAM3 causal/latency test, a virgin benchmark, or proof of globally unseen persons. No VAL threshold, model, checkpoint, memory or candidate-policy selection was performed. The predetermined TRAIN outer0001 model configuration was used; its FIT/INNER contained no VAL sequences.

No locally authorized CHIRLA video/dense annotation exists. Official metadata and identity/protocol membership were audited, but cross-camera/day/session metrics remain **NA, zero executed episodes**. Within-clip seconds and visible-GT gaps are not days, independent sessions, physical absence or verified occlusion. The failure is scoped to this frozen system and protocol, not a universal impossibility theorem about human identity representations.

The scientific result does not alone mark the app Goal complete; semantic delivery and exact clean Git publication are separately verified.

## Quantitative frozen gates

| Gate | Status | Interpretation |
|---|---|---|
| 0 | NOT_PROVEN_END_TO_END_ONLINE_PIXEL_CAUSALITY | GT-free current/past replay checked; upstream cached-pixel causality/latency not proven |
| 1 | FAIL | Frozen independent-sequence effect, safety and usefulness checks |
| 2 | FAIL | Correct identity Recall at primary target-unavailable FPR≤2%, required ≥60% |
| 3 | NOT_EVALUABLE_NO_LAWFUL_LOCAL_CROSS_RECORDING_MEDIA | No actual cross-recording media/episodes; no PASS |
| 4 | FAIL | Wrong/unverified writes≤2% AND correct retention≥60%; zero-write is not PASS |

Gate thresholds come from the preexisting `F1_WITHIN_VIDEO_DEVELOPMENT.json`, referenced by the frozen VAL protocol. They were not adjusted after results. Detailed boolean checks and paired sequence-cluster intervals are in `FINAL_RESULT.json`.

## Table A — One-click target identity (not full MOT)

Panel A1: all 15 registered conditions on all 25 VAL sequences, 273 simulated sole clicks and 245,018 future target-frame decisions per condition. Box recall is IoU≥0.5; strict UID recognition is separately reported because an overlapping wrong candidate can satisfy box overlap. Takeover means a verified other identity, not an unmatched UNKNOWN.

| Condition | Box recall | Strict UID recall | Wrong-ID frames | Takeover runs | Reacq recall | Recovered-only median delay (s) |
|---|---|---|---|---|---|---|
| ACIB_ANCHOR_ONLY_SEED72101 | 16.71% | 15.56% | 130025 | 15281 | 33.75% | 0.3500 |
| ACIB_ANCHOR_ONLY_SEED72102 | 15.74% | 14.63% | 125239 | 15760 | 31.26% | 0.3500 |
| ACIB_ANCHOR_ONLY_SEED72103 | 17.27% | 16.12% | 134900 | 16213 | 36.90% | 0.3500 |
| ACIB_FULL_SEED72101 | 16.71% | 15.56% | 130025 | 15281 | 33.75% | 0.3500 |
| ACIB_FULL_SEED72102 | 15.74% | 14.63% | 125239 | 15760 | 31.26% | 0.3500 |
| ACIB_FULL_SEED72103 | 17.27% | 16.12% | 134900 | 16213 | 36.90% | 0.3500 |
| ACIB_MEAN_PROTOTYPE_SEED72101 | 16.33% | 15.24% | 121856 | 14322 | 29.94% | 0.3500 |
| ACIB_MEAN_PROTOTYPE_SEED72102 | 15.42% | 14.35% | 121677 | 14875 | 29.28% | 0.3500 |
| ACIB_MEAN_PROTOTYPE_SEED72103 | 17.31% | 16.18% | 133065 | 15311 | 35.21% | 0.3500 |
| ACIB_WITHOUT_SAFE_WRITE_SEED72101 | 16.40% | 15.30% | 123044 | 14048 | 30.09% | 0.3500 |
| ACIB_WITHOUT_SAFE_WRITE_SEED72102 | 15.56% | 14.48% | 123107 | 14691 | 29.50% | 0.3500 |
| ACIB_WITHOUT_SAFE_WRITE_SEED72103 | 17.56% | 16.41% | 133850 | 14930 | 35.36% | 0.3500 |
| RAW_OSNET_ANCHOR | 7.71% | 7.39% | 30180 | 15193 | 20.64% | 0.7000 |
| RULE_MEAN_PROTOTYPE | 9.77% | 9.31% | 67291 | 14223 | 18.52% | 1.0500 |
| RULE_TEMPORAL_BANK | 12.23% | 11.79% | 55075 | 13292 | 21.96% | 0.7500 |

Failed return windows remain in reacquisition recall; conditional median delay includes recovered returns only. It is not an unconditional recovery estimate.

Panel A2: all nine historically exposed eight-TRAIN comparator cohorts. These are development diagnostics with their frozen historical operating points, not independent VAL; do not transplant them into Panel A1. Historical R3R2/Existing Memory comparisons were not registered on this VAL.

| Condition | Box recall | Strict UID recall (if evaluated) | Wrong-ID frames | Takeover runs | Reacq recall | Median delay (if evaluated) |
|---|---|---|---|---|---|---|
| B0_RAW_ANCHOR | 9.96% | NA | 7481 | 3576 | 44.72% | NA |
| B1_R3R2_ADAPTER | 6.12% | NA | 5525 | 2783 | 29.81% | NA |
| B2_R4_CAUSAL_P1 | 53.29% | NA | 11316 | 1301 | 54.66% | NA |
| B3_R4R1_NATIVE | 52.55% | NA | 11948 | 1195 | 57.76% | NA |
| B4_P0 | 53.29% | NA | 11316 | 1301 | 54.66% | NA |
| B4_P1 | 53.29% | NA | 11316 | 1301 | 54.66% | NA |
| B4_P4 | 53.29% | NA | 11316 | 1301 | 54.66% | NA |
| B4_P6 | 53.29% | NA | 11316 | 1301 | 54.66% | NA |
| B5_ONLINE_NO_LTM | 53.29% | NA | 11316 | 1301 | 54.66% | NA |


Panel A3: seed-average inside each sequence, then sequence macro and clustered 95% CI. All 25 clusters; bootstrap 2,000, fixed seed72104. Seeds and frames are not independent bootstrap units.

| Family | Box recall macro | 95% CI | Strict UID macro | Deployed unavailable-target FPR |
|---|---|---|---|---|
| ACIB_ANCHOR_ONLY | 18.73% | [0.1407998979446225, 0.23912300644638831] | 17.46% | 76.55% |
| ACIB_FULL | 18.73% | [0.1407998979446225, 0.23912300644638831] | 17.46% | 76.55% |
| ACIB_MEAN_PROTOTYPE | 18.59% | [0.13870993193924486, 0.2390969632556099] | 17.37% | 72.63% |
| ACIB_WITHOUT_SAFE_WRITE | 18.78% | [0.14038335779589636, 0.241211201197928] | 17.55% | 73.85% |
| RAW_OSNET_ANCHOR | 8.23% | [0.05646728028914038, 0.11231269766505037] | 7.88% | 19.14% |
| RULE_MEAN_PROTOTYPE | 11.31% | [0.07264721121879358, 0.16023915779154113] | 10.79% | 45.62% |
| RULE_TEMPORAL_BANK | 13.39% | [0.08653549404622504, 0.18913425039435633] | 12.93% | 34.38% |


## Table B — Open-set recognition and distinct calibration

These are exact pooled tied-score diagnostic curves over the complete cohort, not average per-sequence AP, and never new deployment thresholds. Primary positives mean a strictly matched target candidate exists; primary Recall@FPR2 counts correct ranking among those positives. Primary availability AP/ECE does **not** measure correct-identity claim calibration. Secondary joint-identity calibration excludes only unmatched UNKNOWN rank1 candidates, reports exclusions, and retains structural empty sets as verified negatives.

| VAL condition | Primary correct Recall@FPR2 | Availability AP | Availability ECE | Verified identity AP | Verified identity ECE | UNKNOWN excluded |
|---|---|---|---|---|---|---|
| ACIB_ANCHOR_ONLY_SEED72101 | 4.33% | 0.7265 | 0.2471 | 0.3639 | 0.2801 | 47910 |
| ACIB_ANCHOR_ONLY_SEED72102 | 4.21% | 0.7338 | 0.2185 | 0.3244 | 0.2420 | 49153 |
| ACIB_ANCHOR_ONLY_SEED72103 | 4.12% | 0.7171 | 0.2591 | 0.3579 | 0.2382 | 47017 |
| ACIB_FULL_SEED72101 | 4.33% | 0.7265 | 0.2471 | 0.3639 | 0.2801 | 47910 |
| ACIB_FULL_SEED72102 | 4.21% | 0.7338 | 0.2185 | 0.3244 | 0.2420 | 49153 |
| ACIB_FULL_SEED72103 | 4.12% | 0.7171 | 0.2591 | 0.3579 | 0.2382 | 47017 |
| ACIB_MEAN_PROTOTYPE_SEED72101 | 4.51% | 0.7338 | 0.2116 | 0.3765 | 0.2724 | 46983 |
| ACIB_MEAN_PROTOTYPE_SEED72102 | 4.30% | 0.7397 | 0.1952 | 0.3278 | 0.2412 | 48244 |
| ACIB_MEAN_PROTOTYPE_SEED72103 | 4.13% | 0.7229 | 0.2373 | 0.3683 | 0.2347 | 45755 |
| ACIB_WITHOUT_SAFE_WRITE_SEED72101 | 4.53% | 0.7343 | 0.2135 | 0.3751 | 0.2760 | 47192 |
| ACIB_WITHOUT_SAFE_WRITE_SEED72102 | 4.31% | 0.7406 | 0.2029 | 0.3272 | 0.2436 | 48381 |
| ACIB_WITHOUT_SAFE_WRITE_SEED72103 | 4.17% | 0.7233 | 0.2471 | 0.3677 | 0.2366 | 45945 |
| RAW_OSNET_ANCHOR | 2.18% | 0.6723 | 0.3082 | 0.2649 | 0.6318 | 46399 |
| RULE_MEAN_PROTOTYPE | 0.60% | 0.5816 | 0.3656 | 0.2268 | 0.7300 | 56389 |
| RULE_TEMPORAL_BANK | 2.70% | 0.6799 | 0.3335 | 0.3028 | 0.6602 | 45200 |


Exact pooled T2 principal-control curves, all fixed operating-point false-presence counts/rates and UNKNOWN denominators are available in the machine-readable Table B/A and `evaluation/OPEN_SET_METRICS.json`.

## Table C — Recovery, density and long-term scope

| Scenario / full seed | Strict UID recall | Return windows | Strict recovered | Failed returns | First accept wrong windows | Takeover seconds |
|---|---|---|---|---|---|---|
| ACIB_FULL_SEED72101 | 15.56% | 1366 | 409 | 957 | 723 | 6525.9000 |
| ACIB_FULL_SEED72102 | 14.63% | 1366 | 371 | 995 | 698 | 6293.1000 |
| ACIB_FULL_SEED72103 | 16.12% | 1366 | 437 | 929 | 755 | 6768.0000 |


| Cross-recording scenario | Target recall | ReID | Takeover | Valid episodes |
|---|---|---|---|---|
| CROSS_CAMERA | NA | NA | NA | 0 |
| CROSS_DAY | NA | NA | NA | 0 |
| CROSS_CAMERA_AND_DAY | NA | NA | NA | 0 |
| CROSS_SESSION | NA | NA | NA | 0 |


Every compared condition, including failures, has all-visible versus candidate-conditional recall, unavailable-target/NONE errors, actual ≤5 / >5–20 / >20-second within-recording bins, sparse0–4 / medium5–8 / crowded≥9 current-candidate bins, raw-anchor hard-negative strata and GT overlap proxies. Full breakdowns have source seals, complete-frame A/A checks and strict wrong-identity takeover duration/recovery. GT overlap is not physical occlusion truth. `LOW_DENSITY.json`, `HIGH_DENSITY.json`, `REAPPEARANCE.json` and machine-readable Table C retain the complete cohorts; no easy-scene selection.

## Table D — Memory safety AND usefulness

All three seeds of all 11 T2 controls are shown. These are frozen inference capacity/module state-shift diagnostics, not separately retrained architectures. Correct retention is strictly correct accepted writes divided by available target observations. Wrong means verified other OR unmatched unverified write (upper bound), with the categories separate in frame-count breakdowns.

| T2 control / seed | Writes | Wrong/unverified | Wrong rate | Correct retention | Box recall | Reacq |
|---|---|---|---|---|---|---|
| ANCHOR_ONLY_P0/72101 | 0 | 0 | NA | 0.00% | 19.60% | 63.98% |
| ANCHOR_ONLY_P0/72102 | 0 | 0 | NA | 0.00% | 19.36% | 65.84% |
| ANCHOR_ONLY_P0/72103 | 0 | 0 | NA | 0.00% | 19.49% | 65.84% |
| BANK_ONLY_IDENTITY_P1/72101 | 54017 | 45995 | 85.15% | 20.05% | 16.47% | 52.17% |
| BANK_ONLY_IDENTITY_P1/72102 | 54111 | 45834 | 84.70% | 20.69% | 16.54% | 50.31% |
| BANK_ONLY_IDENTITY_P1/72103 | 54253 | 46113 | 85.00% | 20.35% | 16.43% | 47.83% |
| FIXED_CURRENT_SAFE_NO_FUTURE/72101 | 17 | 3 | 17.65% | 0.03% | 19.56% | 63.98% |
| FIXED_CURRENT_SAFE_NO_FUTURE/72102 | 1099 | 868 | 78.98% | 0.58% | 19.35% | 65.84% |
| FIXED_CURRENT_SAFE_NO_FUTURE/72103 | 1045 | 666 | 63.73% | 0.95% | 19.05% | 65.84% |
| FULL_K1/72101 | 0 | 0 | NA | 0.00% | 19.60% | 63.98% |
| FULL_K1/72102 | 0 | 0 | NA | 0.00% | 19.36% | 65.84% |
| FULL_K1/72103 | 212 | 124 | 58.49% | 0.22% | 19.29% | 65.84% |
| FULL_K4/72101 | 0 | 0 | NA | 0.00% | 19.60% | 63.98% |
| FULL_K4/72102 | 0 | 0 | NA | 0.00% | 19.36% | 65.84% |
| FULL_K4/72103 | 184 | 106 | 57.61% | 0.19% | 19.31% | 65.84% |
| FULL_K8/72101 | 0 | 0 | NA | 0.00% | 19.60% | 63.98% |
| FULL_K8/72102 | 0 | 0 | NA | 0.00% | 19.36% | 65.84% |
| FULL_K8/72103 | 185 | 101 | 54.59% | 0.21% | 19.33% | 65.84% |
| MEAN_PROTOTYPE_P1/72101 | 47440 | 37225 | 78.47% | 25.54% | 20.23% | 61.49% |
| MEAN_PROTOTYPE_P1/72102 | 45689 | 35704 | 78.15% | 24.96% | 19.86% | 63.98% |
| MEAN_PROTOTYPE_P1/72103 | 46713 | 36919 | 79.03% | 24.48% | 19.49% | 64.60% |
| UNIFORM_BANK_ATTENTION/72101 | 0 | 0 | NA | 0.00% | 19.60% | 63.98% |
| UNIFORM_BANK_ATTENTION/72102 | 0 | 0 | NA | 0.00% | 19.36% | 65.84% |
| UNIFORM_BANK_ATTENTION/72103 | 189 | 103 | 54.50% | 0.21% | 19.34% | 65.84% |
| WITHOUT_AVAILABILITY/72101 | 0 | 0 | NA | 0.00% | 20.58% | 66.46% |
| WITHOUT_AVAILABILITY/72102 | 0 | 0 | NA | 0.00% | 20.91% | 66.46% |
| WITHOUT_AVAILABILITY/72103 | 188 | 103 | 54.79% | 0.21% | 20.34% | 66.46% |
| WITHOUT_DELAY/72101 | 0 | 0 | NA | 0.00% | 19.60% | 63.98% |
| WITHOUT_DELAY/72102 | 0 | 0 | NA | 0.00% | 19.36% | 65.84% |
| WITHOUT_DELAY/72103 | 328 | 200 | 60.98% | 0.32% | 19.26% | 65.84% |
| WITHOUT_SAFE_WRITE_P1/72101 | 47948 | 37614 | 78.45% | 25.83% | 20.45% | 61.49% |
| WITHOUT_SAFE_WRITE_P1/72102 | 46149 | 36077 | 78.18% | 25.18% | 20.01% | 62.73% |
| WITHOUT_SAFE_WRITE_P1/72103 | 47295 | 37418 | 79.12% | 24.69% | 19.63% | 63.98% |


FULL_K8 TRAIN seed1/2 write nothing; seed3 writes185, with101 wrong/unverified (54.59%). A null zero-write wrong rate is not 0%; zero retention fails usefulness. The stricter two-scene full-MOT pilot has zero FULL writes in all three seeds and FULL/P0 byte-identical trajectories. These scoped facts must not be generalized into “all T2/VAL seeds never write.” Actual VAL writes:

| VAL condition | Writes | Wrong/unverified | Wrong rate | Correct retention | Joint gate |
|---|---|---|---|---|---|
| ACIB_ANCHOR_ONLY_SEED72101 | 0 | 0 | NA | 0.00% | False |
| ACIB_ANCHOR_ONLY_SEED72102 | 0 | 0 | NA | 0.00% | False |
| ACIB_ANCHOR_ONLY_SEED72103 | 0 | 0 | NA | 0.00% | False |
| ACIB_FULL_SEED72101 | 0 | 0 | NA | 0.00% | False |
| ACIB_FULL_SEED72102 | 0 | 0 | NA | 0.00% | False |
| ACIB_FULL_SEED72103 | 0 | 0 | NA | 0.00% | False |
| ACIB_MEAN_PROTOTYPE_SEED72101 | 179778 | 145503 | 80.93% | 25.58% | False |
| ACIB_MEAN_PROTOTYPE_SEED72102 | 174812 | 142535 | 81.54% | 24.09% | False |
| ACIB_MEAN_PROTOTYPE_SEED72103 | 193734 | 157349 | 81.22% | 27.15% | False |
| ACIB_WITHOUT_SAFE_WRITE_SEED72101 | 181845 | 147450 | 81.09% | 25.67% | False |
| ACIB_WITHOUT_SAFE_WRITE_SEED72102 | 177194 | 144621 | 81.62% | 24.31% | False |
| ACIB_WITHOUT_SAFE_WRITE_SEED72103 | 197100 | 160190 | 81.27% | 27.54% | False |
| RAW_OSNET_ANCHOR | 0 | 0 | NA | 0.00% | False |
| RULE_MEAN_PROTOTYPE | 118769 | 97827 | 82.37% | 15.63% | False |
| RULE_TEMPORAL_BANK | 100781 | 74260 | 73.68% | 19.79% | False |


Frozen, unsafe single-positive, consensus P4/P6, delayed/no-delay, current-safe/no-future and learned reliability policies are all retained in Table D JSON. Removing modules does not by itself establish their retrained causal effect.

## Table E — Primary MOT transfer diagnostic

Same frozen GT-free candidate input, joint one-to-one assignments, native public identity states and unchanged frozen C0 association path. All32 real complete rollouts, two TRAIN sequences, all16 cases. Official pinned TrackEval HOTA/AssA/IDF1 shown on0–100 scale; IDSW is an actual count. No formal generalization CI is claimed from two scenes.

| MOT condition | HOTA | AssA | IDF1 | IDSW |
|---|---|---|---|---|
| ACIB_ANCHOR_P0_SEED72101 | 30.7629 | 14.7679 | 30.5585 | 637 |
| ACIB_ANCHOR_P0_SEED72102 | 39.6021 | 24.3306 | 40.2845 | 879 |
| ACIB_ANCHOR_P0_SEED72103 | 29.9412 | 13.8095 | 29.6451 | 752 |
| ACIB_FULL_SEED72101 | 30.7629 | 14.7679 | 30.5585 | 637 |
| ACIB_FULL_SEED72102 | 39.6021 | 24.3306 | 40.2845 | 879 |
| ACIB_FULL_SEED72103 | 29.9412 | 13.8095 | 29.6451 | 752 |
| ACIB_MEAN_P1_SEED72101 | 39.8154 | 25.0292 | 39.8243 | 576 |
| ACIB_MEAN_P1_SEED72102 | 38.0068 | 22.3984 | 37.7606 | 783 |
| ACIB_MEAN_P1_SEED72103 | 36.8014 | 20.8523 | 37.6560 | 678 |
| ACIB_UNSAFE_P1_SEED72101 | 39.5490 | 24.4083 | 39.7058 | 555 |
| ACIB_UNSAFE_P1_SEED72102 | 38.4707 | 22.9782 | 39.7058 | 780 |
| ACIB_UNSAFE_P1_SEED72103 | 37.2153 | 21.5058 | 38.2138 | 672 |
| CLICK_C0 | 53.7183 | 44.0748 | 53.5383 | 90 |
| NO_HUMAN_C0 | 53.7183 | 44.0748 | 53.5383 | 90 |
| RAW_ANCHOR | 51.3872 | 40.3413 | 51.8371 | 385 |
| RAW_MEAN_P1 | 49.1212 | 36.8050 | 49.7107 | 402 |


The model learned target-only candidate states, then was evaluated in actual joint MOT state: this distribution shift is explicit. Same numeric detection multisets are verified per frame, but ID-conditioned HOTA/CLEAR matching can change DetA/LocA/FP/FN; these metrics are not asserted equal. The original erroneous detection-metric equality assertion failed after successful TrackEval. Its exact source, original outputs and failure record remain archived. Versioned evaluator R1 checks exact Decimal detection multisets; no runtime, model, source candidates or trajectories were rerun/altered.

SOT is **DEFERRED_BY_USER**, not a further delivery prerequisite. Prior auxiliary OSTrack/LaSOT trim results and every failed SAM3 SOT resource attempt remain preserved. They are neither complete current identity evaluation nor MOT success; no new SOT continuation is authorized.

## Failure branches and lawful resume boundaries

| Branch | Actual evidence / disposition | Safe next branch (not started) |
|---|---|---|
| F1 data access | Official CHIRLA metadata obtained; gated HF/contact terms and ScienceDB access did not yield lawful local media. Existing within-video experiments completed; SOT expansion now deferred. | Resume official scoped media access after user handles provider terms; no cookies/tokens requested, no bypass |
| F2 cross-recording labels | Official global identity and membership metadata checked; Tracking/ReID path-prefix overlap and train0 single-camera limits disclosed; zero actual cross-recording video episodes. | Use one official lawful protocol with actual media, dense annotations and frame/PTS metadata; no guessed IDs or lineage mixing |
| F3 candidate missing | All-visible/conditional recall, visible-without-candidate and GT-gap counts reported separately across all25 VAL scenes. | Future TRAIN-only candidate-coverage diagnosis, not identity-controller expansion or heldout tuning |
| F4 encoder confusion | Four frozen representations on actual same competitive candidate crops; second ReID minus OSNet CI includes zero; general features worse; no best-backbone selection. | Research representation failure only with a new preregistered MOT-compatible TRAIN plan |
| F5 open-set | All-future target-unavailable negatives; exact primary and secondary pooled curves/calibration plus UNKNOWN and takeover duration. | No posthoc VAL threshold selection; do not call Rank1 improvement reliable identity |
| F6 contamination | Actual unsafe/control/delayed/anchor conditions and paired own-state training; conservative safe memory fails joint retention/safety. | Immutable-anchor fallback evidence retained; no new policy/integration auto-start |
| F7 cross-day | Not evaluable without independent-recording media; within-video long bins are not appearance-change proof. | Official identity/FPS/recording metadata first, no stitched-day fiction |
| F8 generalization | Frozen25 VAL completed without selection, contrasted with real FIT/INNER and exposed TRAIN; full-MOT TRAIN transfer worsens all12 learned cases. | Inspect coverage/state/scene gap on TRAIN; do not fit VAL or infer all causes from two MOT scenes |
| F9 storage | Personal mount retains 90.13GiB above60GiB reserve; no material deletion this delivery. | No heavy/duplicate downloads or environment; retain unique historical assets |


## Reproducibility, repairs and resource receipts

Actual124 fits: original archivedT0 four, numerical-repairedT0_AMP_R1 twenty-four, T1_CAUSAL_V1 seventy-two, T2_COUPLED_V1 twenty-four. The120 repaired/causal fits are eligible; original four are retained, not mixed into repaired results. Actual248 best/latest files have SHA, schema, finite tensors, strict architecture loads and CPU probability-contract tests. Best is selected by TRAIN INNER, latest is resume state. Loader checks are engineering, never identity performance. All weights stay local, not Git.

Original T0 FP16 scale/gradient failure is retained. The repaired trainer is versioned; the exact original trainer is proven by Git095c5b8 and its frozen SHA. The first catalog preflight incorrectly required that archived trainer equal the repaired current source; that catalog-only error and its repair are recorded. No original weight, trainer record, runtime or sealed schema was silently rewritten.

Latest actual complete repository regression: **826 passed, 5 failed**, 0 skipped. Four failures are the pinned old TrackEval CLI SEQMAP_FILE list-versus-path interface; one asserts a historical literal branch name. Focused N72R21:112 passed. The first delivery run omitted per-process venv PATH; three bare-python legacy subprocesses failed to launch. Its XML/diagnostic are preserved; the corrected R2 process PATH restores the actual four-CLI/one-branch classification. No test/third-party source was patched to hide failures.

Final history receipt: 795 historical metadata/code files and 463 models unchanged. All16 local licensed real examples decode with correct frame counts/FPS. Four MOT clips were visually inspected; both recovery and failure retained. GT is posthoc overlay only, sole-click inset is not a second click, cross-recording examples are unavailable. No raw pixels, faces, features, candidate caches, datasets or checkpoints are published to Git.

No other-user process was terminated. Existing environment/data reused; no new training, candidate inference, Hungarian redesign, LoRA or downstream association stage was launched during final evidence assembly.

## Commands and source evidence

Use the existing environment, repository PYTHONPATH and single-thread CPU limits. These commands reproduce posthoc summaries/audits only; they do not authorize additional experiments:

```bash
cd /data3/liuyeqiang/InterMOT
source .venv/bin/activate
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/n72r21_validation_summary.py
python scripts/n72r21_validation_summary.py --pooled-curves
python scripts/n72r21_failure_breakdowns.py --scope VAL
python scripts/n72r21_pooled_t2_curves.py
python scripts/n72r21_final_evidence.py
python scripts/n72r21_delivery_receipts.py --final
python scripts/n72r21_final_report.py
python -m pytest -q tests/test_n72r21*.py
```

`FINAL_RESULT.json`, `evaluation/TABLES_A_TO_E.json`, frozen protocols, runtime/evaluation seals, checkpoint manifest, failure analyses, JUnit XML and history/storage/video receipts carry the complete machine-readable evidence and source SHA. `docs/N72R21_MOT_TRAIN_PILOT.md` retains the focused MOT report.

Publication state is verified separately by the delivery audit and Git receipt; no report text can make an unverified push true. **NEXT_ASSOCIATION_STAGE_AUTHORIZED=false.** Stop at this scoped result; no autonomous next stage.
