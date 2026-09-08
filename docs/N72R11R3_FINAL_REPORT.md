# N72R11R3 Final Report

## Decision

`N72R11R3_STATE_ALIGNED_TEMPORAL_AND_GLOBAL_EDGE` completed its authorized path with
research gate **FAIL_V3_GENERALIZATION_AFTER_STATE_ALIGNMENT**. The state/feature
contract repairs are structurally validated, but neither the ordinary finetune nor
the one permitted category-balanced repair makes V3 ready. The mandated downstream
Bridge and live branches were therefore not authorized. Production authorization is
false. The research direction is **REVISE**, not a claim of success.

All events in this development corpus are `simulated_from_gt`; real-human evidence
count remains zero. Runtime rollout and replay report `runtime_future_gt_used=false`.

## Frozen inputs and repaired contract

- Baseline commit: `1f59671309d676c19903ccca9439011b7bbc8887`.
- Corrected bootstrap corpus: `516` events, `24,690` train examples, `5,700`
  validation examples, fixed `12/6` sequence split; resource-censored development.
- On-policy corpus: train `24,690` rows / `419` events / `12` sequences and validation
  `5,700` rows / `97` events / `6` sequences. It was produced by causal V3 self-selection;
  offline labels were copied for supervision and not used for runtime decisions.
- Shared temporal schema is the eight-dimensional
  `frame_horizon_over_100, tanh_causal_top_score, tanh_causal_second_score,
  tanh_causal_margin, previous_fused_target_score_clipped,
  previous_assignment_uncertainty, trusted_age_over_100,
  has_future_frame_requery_candidate` contract.
- Shared `TemporalIdentityState` now governs corpus construction, V3 rollout and
  runtime: only exact target agreement plus finite score/margin/feature admits trusted
  memory; selected non-targets alone can enter distractor memory; geometry/native
  binding follows exact assignment; unselected candidates do not pollute memory.
- TargetEdgeBridge has one explicit 14-D scalar builder and requires explicit
  `motion_iou`; train/runtime no longer silently use different feature meanings.
- Bridge loss code was corrected to target-vs-best-other public competition, but no
  Bridge checkpoint was trained because the V3 prerequisite failed.

## Failures and repairs retained

1. The first on-policy attempt failed with `ValueError: selected candidate feature has
   zero norm` when a valid zero-filled missing embedding was passed to state memory.
   The original artifact is retained at
   `outputs/N72R11R3/onpolicy_corpus/attempts/onpolicy_failure_20260908T100422Z.json`.
2. Attempt 2 was killed with exit code `137`/`SIGKILL`. The kernel evidence identified
   host global OOM (about `6.4 GiB` available, no swap), not a CUDA OOM traceback.
   The failure record is retained at
   `outputs/N72R11R3/onpolicy_corpus_attempt_02/attempts/process_exit_137_global_oom.json`.
   The remedy was two sequential CPU split workers with bounded BLAS/OpenMP threads;
   no input or model definition changed.
3. The split workers completed atomically in attempt 3. The repaired state semantics
   preserve zero-filled decoder tokens but pass missing state features as `None`, so
   they cannot become fabricated memory vectors.
4. The first full self-rollout after that repair exposed the same missing-feature
   boundary in `n72r11_train_v3.py`; its failure is retained at
   `outputs/N72R11R3/v3_selected_eval/attempts/self_rollout_failure_20260908T103010Z.json`.
   A one-event targeted regression and the full 5,700-row self-rollout then passed.
5. The first E0/E1 aggregator rejected all 32 valid children because older `done.json`
   files omitted top-level `runtime_gt_read` while their authoritative runtime seals
   explicitly contained `false`. The failed aggregate is retained at
   `outputs/N72R11R3/component_replay_e0e1_attempt_01/component_metrics.json`.
   The validator was minimally repaired to inspect the runtime seal, and the same
   manifest re-aggregated with zero failures.

## V3 training and readiness

The corrected bootstrap V3 used the frozen two-epoch bootstrap configuration. A normal
one-epoch finetune used learning rate `2.5e-4`. A second and only permitted repair used
train-only action-type × label-source inverse-square-root balancing, raw clipping
`[0.5, 3.0]`, then mean normalization. No H20/H50/H100 or validation future outcome
was used for weighting or checkpoint choice.

| Candidate | Target-candidate accuracy | NONE accuracy | Future-requery accuracy/count | Readiness |
|---|---:|---:|---:|---|
| Selected bootstrap | 0.511730 | 0.216716 | 0.642857 / 42 | FAIL |
| Ordinary finetune | 0.525299 | 0.238166 | 0.571429 / 42 | FAIL |
| Category-balanced finetune | 0.505520 | 0.262574 | 0.738095 / 42 | FAIL |

The fixed thresholds remained target candidate `>=0.80`, NONE `>=0.50`, future
requery `>=0.60` with count `>=50`. The selected checkpoint was the bootstrap because
its fixed validation loss `5.5877319` was lower than ordinary finetune `7.6736831`
and category-balanced finetune `8.3648421`. Thus V3 was frozen as not ready after the
single permitted repair; no Bridge training was started.

## Authorized component replay: E0/E1 only

Because V3 was not ready, the protocol allowed only the development replay
`E0_BASELINE_B0` versus `E1_V3_LEGACY_INJECTION`. All `32/32` frozen events completed,
with zero child failures, over `18` independent sequences. E2/E3 were not run, so
Bridge and live increments are correctly `null`, not zero and not inferred.

| Comparison | H20 | H50 | H100 |
|---|---:|---:|---:|
| Pooled identity-error reduction E1−E0 | -0.019576 | -0.081081 | -0.068690 |
| Sequence-cluster 95% CI lower | -0.106251 | -0.157491 | -0.121157 |
| Assignment-change rate | 0.535073 | 0.416988 | 0.370288 |
| Correct crossings | 64 | 92 | 176 |
| Incorrect crossings | 76 | 218 | 391 |
| Directional improvements | 13 | 24 | 74 |
| Directional regressions | 115 | 196 | 300 |
| Protected regressions | 8 | 19 | 25 |

The sequence-cluster bootstrap used seed `7211`, `2,000` repetitions and independent
sequence clusters. The overall effect is negative at all horizons and every lower CI
is below zero. E1 changes assignments substantially, but the changes are more often
wrong than correct, especially at H50/H100. This does not recover the N72R10 direction.

By action, pooled identity-error reduction at H20/H50/H100 was:

- `ADD_NEW_IDENTITY`: `+0.144737 / +0.058824 / +0.020672`;
- `ATOMIC_ID_SWAP`: `+0.233333 / +0.033333 / +0.097973`;
- `AUTHORITATIVE_REASSIGN`: `-0.128405 / -0.143284 / -0.130212`;
- `RECOVER_IDENTITY`: `-0.018182 / -0.084095 / -0.068519`.

The positive ADD/ATOMIC slices do not overturn the negative pooled effect or the
strongly negative reassignment/recovery slices. Bounded-window metrics are not HOTA
or IDF1 claims.

## Required scientific answers

1. **Did state-mismatch repair remove the R2 degradation?** Structurally, the
   train/runtime temporal contract and memory admission semantics are aligned and
   pass CPU/targeted/runtime audits. Scientifically, no: corrected V3 still fails
   readiness and E1 is negative.
2. **Did corrected V3 + legacy injection recover N72R10?** No. E1−E0 is negative
   overall at H20/H50/H100, with negative sequence-cluster lower bounds.
3. **Did corrected Bridge reduce model→solver refusal?** Not tested. Bridge training
   was forbidden by the V3 gate, so no Bridge accuracy or refusal reduction is claimed.
4. **Did Bridge protect untouched IDs?** Not tested. No E2/E3 protected-ID result exists.
5. **What is the independent true-live requery increment?** Not measured. E3 was not
   authorized; the value is `null`.
6. **Did H100 recover a positive value?** No. E1−E0 H100 pooled reduction is
   `-0.068690`, CI lower `-0.121157`.
7. **Main direction:** **REVISE**. Keep persistent public identity and exact global
   assignment as structural foundations, but revise V3 supervision/candidate-base
   signal and obtain provenance-complete real-human evidence before another Bridge or
   live experiment. Do not scale Bridge/LoRA/checkpoints to bypass this gate.

## Isolation and reproducibility

All new code is confined to InterMOT modules/scripts; no `third_party/sam3` file was
modified and no N36–N72R2 historical evidence was overwritten. New outputs are under
`outputs/N72R11R3/`. No MOT/OVMOT project, shared checkpoint or shared configuration was
modified. The formal OOM event was rerun only once at full window and passed the resource
check; the other historical 31 events and secondary corpus were not rerun.

Key machine-readable artifacts:

- Controller: `outputs/N72R11R3/CONTROLLER_STATUS.json`
- Final gate: `outputs/N72R11R3/n72r11r3_final_gate.json`
- E0/E1 manifest: `outputs/N72R11R3/component_replay_e0e1_attempt_01/component_replay_manifest_attempt_01.json`
- E0/E1 metrics: `outputs/N72R11R3/component_replay_e0e1_attempt_01/component_metrics_attempt_02.json`
- On-policy corpus: `outputs/N72R11R3/onpolicy_corpus_attempt_03/corpus_manifest.json`

The full authorized branch remains explicitly non-production and non-real-human.
