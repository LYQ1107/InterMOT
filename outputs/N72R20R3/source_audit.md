# N72R20R3 source audit

## Frozen lineage

- R3 base commit: `b2abc90fd90bc5eb36b31f753561ea0aa3da900b`.
- Historical R2 decision remains `FAIL_RUNTIME_MEMORY_COMMIT`.
- The candidate and base-score tapes are the sealed eight-sequence R2 dev lineage under `/data3/liuyeqiang/InterMOT_N72R20R2_assets`; no SAM3 rerun is needed or allowed.

## 1. How current learned scores are computed

`LearnedIdentityMemoryBank.score_matrix` receives the current normalized
512-dimensional identity state and the current normalized frozen OSNet feature
for every candidate. It computes a dot product, which is cosine similarity,
for each public/state record against each candidate. The output is a
state-by-candidate matrix. Candidate UIDs are checked for collisions and the
matrix is required to be finite. No calibration transform, background model,
or identity-presence classifier is applied.

The R2 forensics and commit replay use the same frozen bank and score-before-
update order. The base score tape is produced by the existing
`score_matrix_pairwise` scorer and the unchanged `solve_effect_assignment` /
`solve_exact_public_assignment` path.

## 2. When memory is updated

`LearnedIdentityMemoryBank.update_from_consensus` runs after the current frame
has been scored and assigned. It updates only when the base and treatment
public assignments are equal, non-`NONE`, and the candidate feature is
available. It invokes the frozen N72R18 GRU, records hashes, and protects the
human anchor. In the historical R2 C0 replay, every non-NONE base assignment
was accepted as the update candidate; this is why a wrong base assignment could
contaminate the persistent state.

## 3. Why a candidate top1 always exists when candidates exist

The learned score matrix supplies one finite scalar per available candidate.
An arg-sort/arg-max over a non-empty candidate list therefore returns one of
those candidate UIDs, even when every score is poor or the target is absent.
The current learned-memory API has no abstention branch between scoring and
top1 selection.

The exact public assignment solver has explicit per-candidate `NONE` columns,
but that is a global assignment option with score `0.0`; it is not an
identity-specific test of whether the human-initialized target is represented
in the candidate set. R2's failure is therefore consistent with a closed-set
ranker being forced to name a candidate.

## 4. Identity-specific NONE score

There is no identity-specific learned-memory `NONE` score or calibrated
identity-presence logit. `identity_memory_commit.evaluate_commit` can reject a
missing base assignment, a missing feature, a non-top1 assignment, or a weak
margin, but it still assumes the learned top1 is a candidate whenever the
candidate list is non-empty. `learned_identity_rescue` also only constructs a
target-column residual and does not add presence semantics.

## 5. Absolute cosine calibration

Absolute dot-product/cosine scores have not been calibrated for presence.
N72R18's declared endpoint is hard-negative ranking: a positive future
identity score must exceed the most similar visible competing identity. Its
training and validation artifacts report ranking metrics, not a target-absent
or candidate-set-absent calibration curve. R2 tested commit thresholds and
margins, but did not establish a cross-sequence open-set threshold.

## 6. What N72R18 training contained

The N72R18 training configs record `same_identity_observation_source` as
`future_gt_crop_offline_causal_replay`, `score_then_update=true`, and a hard-
negative contrastive temperature. The inherited protocol contains visible
same-frame competitors and future same-identity observations. It does not
declare examples whose target identity is absent from the current candidate
set, nor an explicit `NONE`/identity-absence loss. The updater consumes a
previous state and one observation; it does not consume a candidate-set
presence label.

Thus N72R18 can learn useful closed-set identity state geometry without being
trained to say that none of the current candidates represents the target.

## R3 consequence

R3 must add only a small, auditable open-set presence/abstention layer over the
existing frozen evidence. Runtime rows are generated before any same-frame
memory update and contain no GT labels. GT is joined later into a separate
post-hoc table for taxonomy and evaluation. Assignment authority and candidate
generation remain frozen.
