No. On the frozen eight-sequence DanceTrack-train lineage, the human-initialized identity memory showed useful raw presence-related signal, but no leave-one-sequence-out policy could reliably abstain on absent candidate sets and identify the target across sequences under the preregistered safety and usefulness gates.

# N72R20R3 Final Report — Open-Set Persistent Identity Recognition

## 1. Final decision

`FAIL_OPEN_SET_CROSS_SEQUENCE_GENERALIZATION`

`next_association_authority_stage_authorized=false`

The final decision is not `FAIL_OPEN_SET_PRESENCE_SIGNAL`: the frozen features have measurable pooled separation. The failure is that this signal does not yield a safe and useful cross-sequence operating point under LOSO evaluation.

## 2. Frozen goal

This stage has one frozen goal:

`Open-Set Persistent Identity Recognition`

Central question:

> Can a human-initialized persistent identity memory determine whether the target identity is represented in the current candidate set, abstain when it is not, and identify the correct candidate when it is present?

The source lineage is N72R20R2 commit `b2abc90fd90bc5eb36b31f753561ea0aa3da900b`. The exact eight DanceTrack train sequences were reused:

`dancetrack0001`, `dancetrack0002`, `dancetrack0023`, `dancetrack0024`, `dancetrack0039`, `dancetrack0057`, `dancetrack0062`, `dancetrack0072`.

Candidate generation, the candidate axis, OSNet features, the N72R18 GRU, public-ID authority, and the exact assignment solver were frozen. SAM3 was not rerun. DanceTrack val/test, MOT/TrackEval, training, LoRA, and association rescue were not run.

## 3. Why R2 failed

R2 established that the persistent-memory state could change scores but did not safely control runtime memory writes: its formal decision was `FAIL_RUNTIME_MEMORY_COMMIT`. The R2 C0 baseline accepted 2,000 wrong writes among 6,889 accepted writes. R3 therefore isolates the prior question from association authority:

- R3 asks whether the target is represented by any current candidate.
- R3 does not change the assignment, solver, public ID, or memory authority.
- Any future authority or rescue question belongs to R4 and remains unauthorized.

## 4. Failure taxonomy

The frozen eight-sequence development tape contains 8,414 future frames and 49,202 candidate rows. Taxonomy labels use GT only after runtime artifacts were written.

| Class | Meaning | Count |
|---|---|---:|
| P0 | Target GT identity absent | 154 |
| P1a | Best available localization exists but IoU < 0.50 | 439 |
| P1b | Target candidate unavailable and selected object is not best available | 1,226 |
| P1 total | Candidate-set absence due to P1a/P1b | 1,665 |
| P2 | Target candidate exists, frozen base assignment is wrong | 1,706 |
| P3 | Target candidate exists, frozen base assignment is correct | 4,889 |

Candidate-set PRESENT is `P2 + P3 = 6,595`; candidate-set ABSENT is `P0 + P1 = 1,819`. P0/P1/P2/P3 never enter runtime decisions.

## 5. R2 wrong-write reclassification

The original R2 failure is preserved unchanged. Of the 2,000 historical C0 wrong writes:

| Reclassification | Count | Fraction |
|---|---:|---:|
| P0 target absent | 68 | 3.40% |
| P1 target candidate unavailable | 817 | 40.85% |
| P1a localization gray case | 439 | 21.95% |
| P1b wrong available object | 378 | 18.90% |
| P2 true candidate association error | 1,115 | 55.75% |
| Other / insufficient evidence | 0 | 0.00% |

This is a diagnostic decomposition, not a reinterpretation of the R2 decision.

## 6. Candidate-set presence benchmark

Runtime generated 16,828 rows: S0 human-anchor-only and S1 R2-style causal learned-state features for the 8,414 future frames. Runtime rows contain no future GT, GT ID, GT IoU, taxonomy, or post-hoc label. A separate post-hoc join contains 25,242 rows including the explicitly marked S2 oracle-clean diagnostic.

The required pooled gate was negative FPR `<= 0.02` and open-set correct-ID recall `>= 0.60`, plus the same thresholds for every held-out sequence. No policy passed both.

## 7. Human anchor vs learned state

S0 keeps the human-confirmed anchor fixed. S1 scores before each R2-style state update and then follows the frozen R2 base-assignment update trajectory. S2 is an oracle-clean post-hoc diagnostic only; it is not a deployable runtime condition and was not eligible for policy selection.

The best pooled audit feature had AUROC `0.7660916705798405` for PRESENT versus ABSENT (`human_anchor_top1_score` / S0 learned top-1 score). Other useful raw features included score mean AUROC `0.751303`, learned top-1-minus-median AUROC `0.729662`, and motion evidence AUROC `0.707961`. Raw AUROC is not sufficient: safe thresholding and cross-sequence generalization failed.

## 8. B0–B3 results

The table reports pooled held-out LOSO metrics. “Wrong write” and “retention” are post-hoc eligibility diagnostics only; no R3 causal memory replay was run.

| Condition / method | Params | Neg FPR | P0 FPR | P1 FPR | Presence recall | Open-set correct ID recall | Conditional top-1 | Wrong write* | Retention* | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S0 B0 Always Present | 0 | .953271 | 1.000000 | .948949 | 1.000000 | .273995 | .233916 | .222579 | .297198 | FAIL |
| S0 B1 Absolute Score | 0 | .038483 | .000000 | .042042 | .111751 | .044579 | .788414 | .497191 | .036613 | FAIL |
| S0 B2 Absolute + Margin | 0 | .056075 | .006494 | .060661 | .112358 | .044731 | .733479 | .497191 | .036613 | FAIL |
| S0 B3 Top1 − Mean | 0 | .008246 | .012987 | .007808 | .009704 | .006065 | .506107 | .269231 | .007773 | FAIL |
| S0 B3 Top1 − Median | 0 | .017042 | .019481 | .016817 | .025625 | .012889 | .329224 | .192308 | .017181 | FAIL |
| S0 B3 Z-score | 0 | .025838 | .012987 | .027027 | .025019 | .003942 | .174487 | .413793 | .003477 | FAIL |
| S1 B0 Always Present | 0 | .953271 | 1.000000 | .948949 | 1.000000 | .702957 | .613245 | .258440 | .925547 | FAIL |
| S1 B1 Absolute Score | 0 | .000000 | .000000 | .000000 | .000152 | .000152 | .180743 | .000000 | .000205 | FAIL |
| S1 B2 Absolute + Margin | 0 | .000000 | .000000 | .000000 | .000152 | .000152 | .180743 | .000000 | .000205 | FAIL |
| S1 B3 Top1 − Mean | 0 | .051127 | .006494 | .055255 | .011827 | .000455 | .002626 | .981366 | .000614 | FAIL |
| S1 B3 Top1 − Median | 0 | .002749 | .000000 | .003003 | .002578 | .000000 | .000000 | 1.000000 | .000000 | FAIL |
| S1 B3 Z-score | 0 | .038483 | .012987 | .040841 | .063078 | .036998 | .271060 | .344086 | .049908 | FAIL |

S0 B3 Top1 − Mean demonstrates the tradeoff clearly: pooled FPR is below 2%, but open-set correct-ID recall is only 0.6065%. S1 B0 has high recall but accepts nearly every absent frame. No B0–B3 method passed the static gate.

## 9. B4 authorization and result

B4 tiny logistic presence calibration was authorized under the preregistered rule because an existing S0/S1 feature had pooled AUROC `>= 0.60` and B0–B3 were limited. It used 11 trainable parameters over existing causal features; no identity model or encoder was retrained.

| Condition / method | Params | Neg FPR | P0 FPR | P1 FPR | Presence recall | Open-set correct ID recall | Conditional top-1 | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| S0 B4 Logistic Presence | 11 | .074217 | .051948 | .076276 | .227293 | .055345 | .294187 | FAIL |
| S1 B4 Logistic Presence | 11 | .184717 | .415584 | .163363 | .244276 | .208036 | .794696 | FAIL |

B4 did not establish a safe operating point. Its S1 held-out sequence results show the cross-sequence problem directly:

| Held-out sequence | Negative count | Positive count | FPR | Open-set correct ID recall |
|---|---:|---:|---:|---:|
| dancetrack0001 | 35 | 667 | .028571 | .718141 |
| dancetrack0002 | 215 | 987 | .000000 | .028369 |
| dancetrack0023 | 374 | 1108 | .122995 | .676895 |
| dancetrack0024 | 51 | 711 | .000000 | .004219 |
| dancetrack0039 | 49 | 1192 | .000000 | .024329 |
| dancetrack0057 | 29 | 592 | .000000 | .030405 |
| dancetrack0062 | 247 | 955 | .032389 | .067016 |
| dancetrack0072 | 819 | 383 | .343101 | .002611 |

The complete S0/S1 per-sequence table for all 14 policies is embedded in `FINAL_RESULT.json` and `presence/loso_results.json`.

## 10. LOSO cross-sequence results

The protocol has exactly eight sequence folds, with each sequence held out once and the other seven used for calibration. There was no frame-level random split. There are 112 held-out records and 14 aggregate policy rows. Sequence-cluster bootstrap used 2,000 repetitions; it resampled sequences, not correlated frames.

No policy simultaneously passed:

- pooled negative FPR `<= 2%`;
- pooled open-set correct-ID recall `>= 60%`;
- every held-out sequence at the same thresholds.

The static signal is therefore not cross-sequence robust enough for a frozen runtime presence policy.

## 11. P0 versus P1 absence analysis

P1 is the larger absence source: 1,665 frames versus 154 P0 frames. This matters because P1 tests whether the interface can abstain when the target is real but no candidate sufficiently represents it; handling P0 alone would not answer the open-set candidate question.

The P0/P1 FPR split is reported for every LOSO method in the primary table. For example, S1 B4 has P0 FPR `.415584` and P1 FPR `.163363`, while S0 B3 Top1 − Mean has P0 FPR `.012987` and P1 FPR `.007808` but almost no useful identification recall.

## 12. Causal memory replay

Causal replay was intentionally not run. The precondition—a passing static presence gate—was false, so running a memory-commit replay would not answer the preregistered R3 question and would risk conflating presence with association authority.

`outputs/N72R20R3/causal/causal_replay.json` records `NOT_RUN_STATIC_PRESENCE_GATE_FAILED`. Therefore causal wrong-write rate and causal retention are `null`, not zero and not a claimed failure measurement.

## 13. Memory contamination result

No new R3 runtime contamination claim is made: no causal writes were executed, and all runtime rows report `runtime_future_gt_used=false`. The historical R2 wrong-write population remains 2,000 and is preserved in the post-hoc reclassification artifact. S2 oracle-clean rows are explicitly posthoc-only and cannot be used as runtime evidence.

## 14. Limitations

- The evidence is limited to the frozen eight DanceTrack train sequences and their sealed candidate/base-score tapes.
- DanceTrack val/test, MOT datasets, and a fresh SAM3 candidate generation were intentionally excluded.
- GT is used only for post-hoc taxonomy and evaluation joins.
- P1 mixes detector/localization limitations with candidate-interface absence; R3 reports the distinction but does not repair it.
- B4 is a tiny calibration diagnostic, not a newly trained identity representation.
- No association lambda scan, solver change, identity decoder, LoRA, or memory architecture change was authorized.

## 15. Scientific interpretation

The answer to “can we recognize the user-selected person among current candidates and abstain otherwise?” is no under the required cross-sequence operating guarantee. The representation is not devoid of information: fixed-anchor scores separate PRESENT from ABSENT in pooled diagnostics. The failure is that the score is not calibrated or stable enough across scenes to provide both safe abstention and useful correct identification. A high-recall always-present policy is not open-set recognition, and a low-FPR policy with near-zero recall is not a usable identity recognizer.

## 16. Next-stage authorization

`next_association_authority_stage_authorized=false`

N72R20R4 identity-aware association authority is not authorized. Do not use this result to justify solver intervention, association rescue, VAL evaluation, or full MOT.

## 17. Runtime invariants

- Candidate axis was read from the sealed tape and never changed.
- Presence evaluation creates no candidates and never changes public-ID assignment.
- Runtime scoring is before any S1 state update.
- Runtime code rejects GT and post-hoc taxonomy keys.
- Human anchor and public/state binding remain immutable.
- `runtime_future_gt_used=false` for runtime features, labels, LOSO, and stop artifacts.
- S2 oracle-clean data is marked `posthoc_oracle_only=true` and is not a final policy.
- R2 historical artifacts are byte-hash unchanged.
- SAM3 rerun, val/test access, training, and association rescue are false/not started.

## 18. Reproducibility / hashes

- Branch: `codex/n72r20r3-open-set-identity`.
- R2 source commit: `b2abc90fd90bc5eb36b31f753561ea0aa3da900b`.
- Frozen N72R18 checkpoint SHA-256: `94ba9c44e177b254ee54984195e839f37c78a648c010ffc723dd3f1d553976a2`.
- Frozen OSNet encoder SHA-256: `2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154`.
- R2 immutability audit: `all_unchanged=true` for `FINAL_GOAL.json`, `FINAL_RESULT.json`, `FINAL_REPORT.md`, and `commit_gate_decision.json`.
- Storage after evaluation: `115.735 GiB` free; hard stop is `100 GiB`; no download or deletion was performed.
- Primary machine-readable result: `outputs/N72R20R3/FINAL_RESULT.json`.
- Full per-policy metrics: `outputs/N72R20R3/presence/loso_results.json`.
- Fold policies and calibration provenance: `outputs/N72R20R3/presence/fold_policies.json`.
- Runtime/post-hoc provenance manifests: `outputs/N72R20R3/presence/runtime_feature_manifest.json` and `posthoc_label_manifest.json`.
- Future R4 read-only headroom: `outputs/N72R20R3/future_association_headroom.json`.
- R3 targeted tests: `42 passed, 0 failed`.
- Full suite: `324 passed, 4 failed`; all four are the known pinned TrackEval CLI `SEQMAP_FILE` list/type failures and do not exercise R3 code.

The final output is a terminal R3 result. No downstream association or VAL work is authorized from it.
