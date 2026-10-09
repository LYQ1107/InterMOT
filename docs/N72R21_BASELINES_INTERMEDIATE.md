# N72R21 baseline diagnostic — intermediate, not scientific closure

Final Goal: **One Click, Persistent Identity: Causal Long-Term Human Tracking Across Occlusion, Reappearance and Independent Recordings**.

The application Goal remains active. These are eight repeatedly exposed DanceTrack TRAIN scenes, 52 first-eligible real-image `SIMULATED_ONE_CLICK_FROM_GT` targets, not independent final validation or a real-human study. One initialization per target; all future runtime decisions are sealed before GT evaluation. New thresholds are not selected from outer outcomes. B0 uses uncalibrated smoke defaults, B1 uses historical fit/inner-frozen score/margin settings, and the actual tracker baselines retain their native NONE mechanism. Consequently their fixed operating points are descriptive, not a fair calibrated gate or a selected scientific winner.

| Actual comparator | All-visible target recall | Verified wrong-person frames | Wrong-write rate | Correct observation retention |
| --- | ---: | ---: | ---: | ---: |
| Raw OSNet immutable anchor | 9.96% | 7,481 | undefined: zero writes | 0% |
| Strict six-fit R3R2 Adapter, immutable anchor | 6.12% | 5,525 | undefined: zero writes | 0% |
| Actual R4 P1 causal state | 53.29% | 11,316 | 40.69% | 69.48% |
| Actual R4R1 NativeReliability, P0 deployment | 52.55% | 11,948 | undefined: zero writes | 0% |
| Existing online tracker, no long-term memory | 53.29% | 11,316 | undefined: zero writes | 0% |
| Existing P4 delayed confirmation | 53.29% | 11,316 | 20.91% | 30.57% |
| Existing P6 rollback diagnostic | 53.29% | 11,316 | 20.81% | 35.36% |

Source: `outputs/N72R21/baselines/DEVELOPMENT_SUMMARY.json`, nine configurations, all eight sequences complete. P0/P1 tracker comparisons duplicate no-LTM trajectory outputs by design; this is verified on real sealed UID decisions in `AUTHORITY_VS_SCORE_AUDIT.json`. Identity-score or GRU-state changes do not establish action authority. NativeReliability is a historical P1-trained/P0-deployed comparator, not assumed state-matched. No active memory policy satisfies the joint wrong-write ≤2% / useful retention ≥60% gate; zero writes never pass.

Frozen representation diagnostic covers 39,760 future frames with a correctly matched target candidate and at least one **verified different-identity** competitor: raw OSNet hard-negative win rate **32.98%**, strict Adapter ensemble **28.48%**. Unresolved/duplicate unmatched candidates are not guessed to be other people. Missing candidates are excluded from this discrimination denominator and reported separately in all-visible tracking metrics. These results motivate the requested frozen encoder controls before enlarging a controller; they do not authorize a final representation failure/success decision before the remaining research work.

B6 is actual [official OSTrack](https://github.com/botaoye/OSTrack), not a fabricated SOT number. One officially linked 256/CE checkpoint, 370,179,249 bytes, SHA256 `8e01d6251569ac84dbb53fdd062cd938551be2d889582f789aa3070196f42f41`, was strictly loaded with `weights_only=True`. Its 41 needed source files retain original bytes, verified against the upstream Git tree; device-neutral tracking agrees with five original real-frame steps within 1e-5. All eight registered TRAIN scenes / 52 clicks are complete, with 54,990 future frames. Pooled visible-target IoU≥0.5 recall is 0.371487, SOT success AUC 0.330413 and normalized precision@0.2 is 0.347547. Verified wrong-person takeover occupies 34,065 frames. All 2,108 visible-GT gap frames emit a box; no physical out-of-view claim follows from missing GT. There are 161 reappearance episodes: 62.73% reacquire at some point, 44.10% within one actual second. Original SOT has no NONE/persistent identity head. Its pixel-based input differs from the SAM3 candidate axis and is reported separately. Public research-release provenance is recorded; separate off-repository weight redistribution terms are not explicitly stated, so weights are not republished. No final success or next-stage authorization follows.

One last-scene input failure revealed 13 zero-area boxes in the sealed raw SAM3 tape. The versioned repair removes only invalid current geometry, identically across methods, without GT/score selection or changes to raw data, historical code, thresholds or seven earlier result seals. This is a documented engineering protocol repair, not falsely presented as a discovery preregistered before it occurred.

Pending: actual frozen second-ReID/general visual controls; complete official cross-recording metadata/media protocol; actual new three-seed T0–T3 learning, causal-state/memory ablations and frozen generalization; proper final uncertainty/gates, final reports and final Git closure. No downstream stage is authorized.
