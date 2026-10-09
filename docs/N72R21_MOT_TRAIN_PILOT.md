# N72R21 — MOT-first TRAIN transfer diagnostic

## Outcome and scope

Latest user direction: **interactive MOT first; SOT deferred**. The [direction override](../outputs/N72R21/RESEARCH_DIRECTION_OVERRIDE.json) supersedes older SOT milestones without rewriting the original frozen Goal. Existing SOT results/failures remain preserved; no further SOT acquisition, inference, training or resource retry is started.

All **32 full multi-object rollouts** completed: two DanceTrack TRAIN sequences × 16 preregistered cases. Actual pinned TrackEval completed with return code **0**, after a versioned repair of our evaluator-only assertion. **Every learned variant/seed worsened combined HOTA, AssA, IDF1 and IDSW relative to clicked C0.** This is a failed engineering transfer diagnostic, not evidence that interaction helps MOT and not a final independent-generalization conclusion. Goal remains active; no downstream stage, association redesign or new training is authorized by this result.

## Protocol and causal boundary

[Protocol](../outputs/N72R21/protocol/MOT_TRAIN_PILOT.json) was frozen before pilot initialization/replay/evaluation. Sequences 0001/0002 have 703/1,203 original frames. Each treated rollout has one person clicked once, using the first registered deterministic sole-click episode; no-human C0 has no click. Both initial bindings succeeded (clicked-box IoU 0.9795/0.9264). No target or scene was replaced based on performance.

Identical existing SAM3/OSNet candidate tapes and GT-free geometry filtering feed all methods. Every valid current candidate is output once with a unique public ID per frame. These are complete joint multi-object trajectories, **not merged single-target episodes**. Existing joint scores, exact solver, NONE/births and machine-state updates are unchanged. Identity proposals use the existing forced-edge joint solve; only actually accepted observations may commit proposed writes. Hard-infeasible proposals keep the current joint baseline and veto the proposed write.

Seeds 72101–72103 use their own TRAIN outer-fold model selected by INNER loss, excluding the evaluated sequence from FIT and INNER. No weights, thresholds or candidates were retuned. Models were not trained on this joint-MOT state distribution; this is explicitly a frozen-weight transfer diagnostic. All 32 runtimes were sealed before offline future truth. Runtime uses only initialization/current/past candidate observations, not future GT. Cached candidate replay does **not** prove end-to-end online SAM3 causality or pixel-to-output latency. Both scenes are historically exposed TRAIN data; two scenes cannot establish independent or cross-recording generalization.

## Actual combined metrics — all cases, no selected seed

HOTA/AssA/IDF1 use the **0–100 scale**; IDSW is the two-scene combined count. Unrounded combined/per-sequence metrics, including DetA, LocA, MOTA, FP/FN, and target/write audits are in [RESULT.json](../outputs/N72R21/mot_pilot/RESULT.json).

| Case | Seed | HOTA | AssA | IDF1 | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: |
| NO_HUMAN_C0 | — | 53.7183 | 44.0748 | 53.5383 | 90 |
| CLICK_C0 | — | 53.7183 | 44.0748 | 53.5383 | 90 |
| RAW_ANCHOR | — | 51.3872 | 40.3413 | 51.8371 | 385 |
| RAW_MEAN_P1 | — | 49.1212 | 36.8050 | 49.7107 | 402 |
| ACIB_ANCHOR_P0 | 72101 | 30.7629 | 14.7679 | 30.5585 | 637 |
| ACIB_ANCHOR_P0 | 72102 | 39.6021 | 24.3306 | 40.2845 | 879 |
| ACIB_ANCHOR_P0 | 72103 | 29.9412 | 13.8095 | 29.6451 | 752 |
| ACIB_FULL | 72101 | 30.7629 | 14.7679 | 30.5585 | 637 |
| ACIB_FULL | 72102 | 39.6021 | 24.3306 | 40.2845 | 879 |
| ACIB_FULL | 72103 | 29.9412 | 13.8095 | 29.6451 | 752 |
| ACIB_UNSAFE_P1 | 72101 | 39.5490 | 24.4083 | 39.7058 | 555 |
| ACIB_UNSAFE_P1 | 72102 | 38.4707 | 22.9782 | 39.7058 | 780 |
| ACIB_UNSAFE_P1 | 72103 | 37.2153 | 21.5058 | 38.2138 | 672 |
| ACIB_MEAN_P1 | 72101 | 39.8154 | 25.0292 | 39.8243 | 576 |
| ACIB_MEAN_P1 | 72102 | 38.0068 | 22.3984 | 37.7606 | 783 |
| ACIB_MEAN_P1 | 72103 | 36.8014 | 20.8523 | 37.6560 | 678 |

Clicked and no-human C0 metrics are identical: initialization alone produced no future MOT benefit. FULL worsened all four metrics on **each sequence** as well as combined, for all seeds; this is not an effect of pooling alone.

## Target benefit, collateral damage and memory

The fixed future visible-target denominator is 1,892 frames. Baseline strict UID correctness is 920 frames; FULL gives 446/361/257. FULL has 139/55/59 current-frame corrections (`N01`) but 613/614/722 regressions (`N10`). These frame counts are not independent events.

FULL has **zero accepted writes in this two-scene pilot** for all seeds and matches P0 exported trajectories. Wrong-write rate is undefined and correct observation retention is zero: useful safe memory does not pass. This statement is pilot-specific; the separate eight-scene T2 identity-module cohort has rare writes in seed72103 and is not wholly zero-write.

Unsafe learned P1/mean diagnostics write but have conservative non-target-or-unverified write rates around **77.9–86.2%** (raw mean P1: 79.4%). UNKNOWN unmatched writes are counted separately, not called verified other identities. Every case has zero registered windows where a current correction retains strict target benefit in **all** future frames through H100. This conservative overlapping-window diagnostic is not an independent-event win rate or new success endpoint.

FULL also loses thousands of baseline-correct non-target public-origin/frame assignments. This is a one-sided collateral-damage proxy, not distinct people or net benefit; official full-MOT metrics provide the global comparison. Complete JSON preserves H20/H50/H100, candidate coverage, takeover, reappearance and unsuccessful returns. Strict UID identity is separate from overlapping-box accuracy.

## Evaluator-only R1 repair and tests

Original TrackEval succeeded; **our own** subsequent guard incorrectly required identical DetA/LocA/CLEAR FP/FN for identical detections. Pinned [HOTA](../third_party/MOTIP/TrackEval/trackeval/metrics/hota.py) matches using global identity alignment; [CLEAR](../third_party/MOTIP/TrackEval/trackeval/metrics/clear.py) prioritizes previously matched IDs. ID trajectories can change matched detections and derived metrics with unchanged geometric input. This is not an upstream, environment, model or tracker failure.

Exact failed evaluator source, original logs/output directory and [failure record](../outputs/N72R21/mot_pilot/ORIGINAL_EVALUATOR_FAILURE.json) remain preserved. [R1](../outputs/N72R21/mot_pilot/EVALUATOR_ASSERT_REPAIR_R1.json) was registered before re-evaluation. Its replacement guard checks the exact per-frame exported multiset `(frame,x,y,width,height,confidence)`, preserving multiplicity and ignoring public ID/order, alongside existing frame-axis/UID/seal checks. **All 32 detection multisets match.** Official TrackEval was repeated on unchanged trajectories in a new `trackeval_R1` directory. No runtime, weight, threshold or association rerun/change occurred.

Actual tests: **15 R1 preflight passed; 93 N72R21 focused passed; 807 repository passed / 5 failed**. Existing failures: four pinned TrackEval old-CLI `SEQMAP_FILE` list-versus-path failures and one historical branch-literal assertion. Old tests/upstream code were not patched to hide them. Actual JUnit and classifications are retained under `outputs/N72R21/tests/`.

## Reproduction and next bounded work

Use existing `.venv` with `PYTHONPATH=/data3/liuyeqiang/InterMOT`. `python scripts/n72r21_mot_pilot_verify.py` checks all 32 seals/artifacts, six frozen models and unchanged fitting sources and derives all-case target totals, without fitting or GT-based selection. `python scripts/n72r21_progress.py` refreshes counts while retaining MOT-primary/SOT-deferred direction. Sealed replay/versioned evaluation refuses overwrite; do not invoke the ready driver to rerun completed trajectories.

Finish the already-running frozen DanceTrack identity-module evaluations and failure decomposition. Do not treat target-only recall as MOT benefit, select thresholds/models using VAL, start SOT retries or expand MOT association changes on these negative TRAIN results. Retain transfer failure and collateral damage when deciding further research.
