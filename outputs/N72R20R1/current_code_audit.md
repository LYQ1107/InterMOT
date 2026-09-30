# N72R20R1 Current Code Audit

## Frozen scope

Final Goal: **Fresh-Lineage Learned Identity Memory Integration**.

Central question: **Can the N72R18 learned identity memory improve future
public-ID association when evaluated on a newly generated provenance-complete
frozen candidate/base-score lineage under the current InterMOT runtime?**

This stage does not reproduce or restore N72R15. It establishes an independent
fresh experimental lineage. The historical N72R20 artifact
`outputs/N72R20/frozen_tape_recovery_audit.json` remains
`BLOCKED_NOT_RECOVERED` and is not modified or used as a formal input.

## Existing pipeline and ownership

The current runtime already contains the complete path:

```text
SAM3
  -> candidate generation
  -> candidate bridge
  -> candidate/public scoring
  -> exact assignment
  -> explicit NONE
  -> persistent public identity
  -> TrackManager
  -> TrackEval
```

The R1 insertion is only the frozen N72R18 learned identity signal:

```text
candidate observation
  -> same-lineage OSNet embedding
  -> N72R18 public identity state
  -> learned identity evidence
  -> relative state edge
  -> row-max-preserving fusion
  -> existing exact assignment
  -> public identity
  -> trusted machine update
```

No SAM3 source, exact assignment solver, or public-ID authority is being
reimplemented.

| Component | Current implementation | R1 decision and evidence |
|---|---|---|
| Official SAM3 candidate runtime | `sam3_intermot/backend/sam3_backend.py` (`Sam3Backend`, `start_video` line 243, `detect_concept` line 292, `propagate` line 649, candidate exports lines 1450/1554) | Reuse the adapter and official pinned submodule. It carries boxes, presence/native provenance and no public identity. The caller supplies machine OSNet features. |
| SAM3 checkpoint loader | `sam3_intermot/adaptation/sam3_loader.py` (`build_tracker_model`, line 33) | Keep strict checkpoint identity and official builder behavior; no wrapper fork. |
| Candidate bridge | `sam3_intermot/association/decoder_candidate_bridge.py` | Candidate provenance/merging only. Its legacy `build_decoder_assignment` is not a second solver; R1 fresh base tape uses the explicit effect/public assignment path below. |
| Existing appearance state | `association/appearance_memory.py`, `trusted_persistent_public_state.py` | Retain score-before-update and consensus-only trusted writes as fixed runtime semantics. |
| Relative state edge | `association/relative_persistent_state_edge.py` (`fuse_row_max_preserving`) | Reuse row/column-centered relative evidence and row-max preservation. This preserves each candidate's competition with explicit NONE. |
| Exact assignment | `association/effect_assignment.py` line 42 -> `association/public_assignment.py` line 40 | One `scipy.optimize.linear_sum_assignment` implementation with explicit per-candidate NONE columns. Solver is frozen. |
| Persistent identity owner | `identity/persistent_runtime.py` (`SequencePersistentIdentityRuntime`, line 211) | Candidate bindings remain session-local; public/MOT IDs and lineages remain persistent. |
| Public-ID authority | `identity/public_authority.py` (`PublicAuthorityBridge`, line 110, `bind_identity_state` line 139) | Immutable state-to-public/MOT binding and frame-aware explicit NONE remain authoritative. |
| Frozen learned memory | `association/learned_identity_memory.py` (`LearnedIdentityMemoryBank`) | Strictly loads N72R18 `GRU_MEMORY`, initializes `z0=x_human`, scores before update, and updates only on equal non-NONE base/treatment UIDs. |
| Learned relative edge | `association/learned_identity_state_edge.py` | Adds only centered learned evidence and calls the existing row-max-preserving fusion. It does not call a solver or assign a public ID. |
| Feature contract | `identity_probe/encoders.py`, `identity_memory/encoder.py` | Frozen OSNet x1.0 Market1501, 256x128 RGB ImageNet normalization, 512-D L2-normalized output. N72R5 features are forbidden. |

## Fixed causal order

For every future frame the R1 replay must execute:

1. Read the current learned state and candidate features.
2. Compute the frozen base matrix and the learned treatment edge.
3. Fuse treatment scores while preserving row maxima.
4. Run the same exact solver and explicit NONE policy for E0/E1/E2.
5. Commit public/TrackManager decisions through the existing authority.
6. Only after the decision, perform consensus-only machine memory updates.

The event-frame human initialization is the only privileged write. The human
anchor is immutable and is never replaced by the learned state. Missing
features, NONE decisions, and base/treatment disagreement do not update learned
memory. Runtime rows must carry `runtime_future_gt_used=false`; GT is opened
only by post-hoc evaluation code after the tape is sealed.

## Historical evidence inherited, not repeated

- N72R15: structure passed but the scientific gate was `FAIL_FUTURE_EFFECT`; no
  downstream training or integration was authorized from that result.
- N72R18: the frozen GRU is the learned identity-memory checkpoint and its
  identity-probe protocol is the feature lineage inherited here.
- N72R19: robust noisy-memory validation failed; R1 does not retrain or repair
  that model and tests its frozen checkpoint in a fresh InterMOT lineage.
- N72R19R1: selective update validation ended `FAIL_CLEAN_PRESERVATION`; no
  selector is imported into R1.
- N72R20: formal assignment replay was blocked because the historical N72R15
  tape was unavailable. That blocked artifact remains unchanged.

## R1 invariants

1. Fresh candidate and base-score tapes are generated from the current runtime;
   old N72R20 tapes are diagnostic only and cannot be R1 formal inputs.
2. E0, E1 and E2 read the same sealed candidate rows, features, axes and base
   matrix. They do not regenerate candidates independently.
3. The candidate UID axis, public-ID axis, state axis, solver, NONE score and
   authority mapping are shared across variants.
4. No training, LoRA, new encoder, new matcher, Hungarian variant, or SAM3
   source modification is in scope.
5. DanceTrack train is development-only (`dancetrack0001/0002` smoke first);
   DanceTrack val is formal and remains untuned until the config is frozen.

## Audited source hashes

These are Git blob hashes at the R1 base commit (`669d42f`). They identify the
interfaces audited above:

```text
sam3_backend.py                    8a4cd7a8f483381681c9e9aaab4716ca9f31965d
sam3_loader.py                     73468678695284668f4f3f3540163ef69a969160
decoder_candidate_bridge.py        565feca67b3eba40628553633c7f7cfbb28a81dd
appearance_memory.py               5127c29a784d3784805dc69a3d602e382981595c
trusted_persistent_public_state.py f44d2068c30fe16b732ce2d9524e3b5b3891ce6b
relative_persistent_state_edge.py  baadcc6727b42ffc3813fc9826f1a22304e7dbd3
effect_assignment.py               5de22000896429868316e7a692e081976ae12a70
public_assignment.py               a9cbba74bef6eb8bf7cbcf405e191a957c23eaef
persistent_runtime.py              68ffff31f71962258a2d0430ac315375e95fe12f
public_authority.py                0162ef9a96323808f2cb923372d232a3f1129b71
identity_memory/updater.py         aa871e4d13421456e6cfd8def3c25a08c8fc8209
learned_identity_memory.py         861609e2bffdbe0aaf8fc9bd72e1c7af42188564
learned_identity_state_edge.py     36d00fa4c6868dbf4f72ec0ea18c399e72e21925
```

Audit result: **R1 integration seam is available.** The two-sequence fresh
candidate tape and GT-blind base-score tape were generated and sealed. The
post-hoc identity shadow passed its candidate-signal gate, but the assignment
shadow triggered `FAIL_LEARNED_MEMORY_DECISION_INACTIVE`: learned scores
changed on every future smoke frame while the frozen exact public assignment
changed on zero frames. Therefore no formal VAL/TrackEval conclusion is
claimed and the frozen stop rule ends this stage before downstream causal
promotion.

## Fresh-lineage execution evidence

- Candidate validation: `outputs/N72R20R1/candidate_tape_validation.json` —
  `PASS_N72R20R1_CANDIDATE_TAPE_VALIDATION`.
- Base-score construction: `scripts/n72r20r1_build_base_score_tape.py` uses
  the existing `score_matrix_pairwise` implementation, not a new matcher;
  runtime GT was not opened.
- Fresh tape seal:
  `outputs/N72R20R1/fresh_tape_manifest.json` — `fresh_tape_sealed=true`.
- Shared replay: E0/E1/E2 use the same candidate UID axes, features, base
  matrices, exact solver, NONE score and public axes.
- Assignment-shadow gate:
  `outputs/N72R20R1/assignment_shadow/train_smoke_summary.json`.
