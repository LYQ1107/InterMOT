# R4 association implementation audit

Frozen goal: `outputs/N72R20R4/FINAL_GOAL.json`.

## Code findings

| Item | File/function/lines | Actual behavior | Status | Trajectory impact | R4 repair / required evidence |
|---|---|---|---|---|---|
| M0-A | `scripts/n72r20r3r2r3_pipeline.py:203` `base_identity_scores` | sigmoid of sealed base_score_matrix[:,target], no CrossSceneIdentityAdapter inference | FAIL_IDENTITY_ADAPTER_NOT_IN_ASSOCIATION | Rank gains are not transferred into the actual score matrix | Strict SHA/axis-verified tower inference from branch-owned causal GRU state |
| M0-B | `scripts/n72r20r3r2r3_pipeline.py:758` `rollout_sequence` | plain linear_sum_assignment without explicit NONE; extracts target row, retains all other baseline public assignments | FAIL_TARGET_COLUMN_MERGE | candidate collisions, stolen observations and NONE recovery are not globally committed | Complete solve_exact_public_assignment output, hard candidate/public uniqueness checks |
| M0-C | `scripts/n72r20r3r2r3_pipeline.py:758` `rollout_sequence` | every frame reads sealed base_scores and frozen geometry; no treatment IdentityState update | FAIL_FROZEN_STATE_FEEDBACK | interventions cannot influence next-frame base scores, motion or appearance | Each tracker owns StateManager/IdentityState and recomputes current scores from prior branch state |
| M0-D | `scripts/n72r20r3r2r3_pipeline.py:758` `rollout_sequence` | pre-event continue skips trajectory append for all public IDs | FAIL_PRE_EVENT_TRAJECTORY_OMISSION | truncated non-target trajectories confound full-sequence evaluation when event_frame>0 | ordinary online births/assignment outputs before click; residual visible only after event |
| M0-E | `scripts/n72r20r3r2r3_pipeline.py:758` `rollout_sequence` | new SAM3 identity copied from current max, quality=0.75, geometry=0.7 | FAIL_ASSUMED_NEW_CANDIDATE_IDENTITY | localization benefit includes unsupported appearance/quality authority | R4 uses existing real OSNet candidates; new candidates disabled |
| M5 | `scripts/n72r20r3r2r3_pipeline.py:874` `association_shadow` | local target IoU correction and threshold commit booleans, no GRU write or next-state rollout | FAIL_SHADOW_IS_NOT_MEMORY_OR_TRACK_EFFECT | 90 shadow N01 is not 90 IDSW repairs or actual accepted writes | pair complete global trajectories; future continuity and real memory updater audits |
| Export | `scripts/n72r20r3r2r3_pipeline.py:846` `export_trajectory_tracker` | sorts rows, clamps width/height to >=1; no cross-ID candidate uniqueness validation | AUDIT_EXPORT_GEOMETRY | writer does not repair underlying assignment semantics | validate geometry/uniqueness before export and hash exact trajectories |
| Native | `scripts/n72r20r3r2r3_pipeline.py:347` `export_baseline` | exports exact sealed R2 public assignments; original tape builder does causally update states | SEALED_CAUSAL_BASELINE_SOURCE | valid reference but no new online treatment state | reconstruct independently with same initialization and legacy lifecycle; compare frame assignments and SHA |
| Eval | `scripts/n72r20r3r2r3_pipeline.py:384` `run_trackeval_many` | wrapper normalizes pinned CLI; shared config and full sequence GT | VALID_PINNED_EVALUATOR | evaluator cannot turn malformed runtime into causal evidence | reuse wrapper with identical input axes, frame range and split |
| SAM | `scripts/n72r20r3r2r3_pipeline.py:909` `targeted_sam3_refinement` | dev schedule selects earliest posthoc P1a frame; real SAM3 prompts use causal boxes | POSTHOC_DEV_SCHEDULE | posthoc scheduling is not a deployable trigger | disabled in R4; no candidate regeneration |

## Supporting APIs audited

- `public_assignment.solve_exact_public_assignment`: explicit state/public axes, candidate×public matrix, independent NONE slots; complete solve reused without modification.
- `branch_public_replay.apply_exact_frame`: applies all solver rows, preserves outer birth vs solver NONE; runtime and StateManager independently owned. R4 reuses its lifecycle principles without requiring old fixed public bridge sessions.
- `online_associator.score_matrix_pairwise`: real current feature, previous prototype, velocity/native continuity and gap; reusable base scorer.
- `learned_identity_memory.update_from_consensus`: frozen GRU, immutable anchor, public binding, real accepted-update audit; R4 adds acceptance gates and explicit denominators.
- `cross_scene_adapter`: separate 512-D query/candidate towers, 265472 parameters, actual cosine inference.

## Critical training-lineage discrepancy

`n72r20r3r2_representation.run_representation` first selects epochs on 6/1, then fits the final adapter on all seven non-heldout sequences. `_save_checkpoint` nevertheless writes `parameter_fit_sequences` as six and omits the final seventh sequence. Heldout H is absent, but inner V is actually included in the final fit. Such checkpoints must not be used to select association policy on V. R4 records actual seven-sequence lineage and trains strict six-sequence adapter instances for current 6/1/1 selection. Backbone and GRU remain frozen.

## Historical 90 corrections

G1/G3 changed 1440 target-frame choices with N01=90/N10=0 in shadow reports. Formal inner selection chose G0 in all eight folds, so those association interventions did not enter COMBINED trajectories. Furthermore, their global row reallocations were discarded and their state was never propagated. They are local-IoU diagnostics and cannot be credited as IDSW or HOTA improvements.

## Test evidence

Pending real R4 regression suite and independently generated A/A trajectories; no structural finding is a scientific PASS.
