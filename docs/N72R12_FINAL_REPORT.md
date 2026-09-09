# InterMOT N72R12 Final Report

## Direct answer

**Did selective intervention beat E0? NO.**

N72R12 is `FAIL_SAFE_INTERVENTION_NOT_SUFFICIENT`.  The deterministic
counterfactual gate and the one permitted learned-safe gate both completed their
runtime and post-hoc evaluation chains, but neither passed the pre-registered
global-quality gate against the no-intervention baseline E0.  No production
promotion, calibration head, selector, decoder LoRA, or live requery is
authorized.

## Scope and frozen inputs

The question was whether PCTIS could be made useful by deciding *whether to
alter the current global assignment at all*, rather than always committing its
proposal.  The frozen runtime is:

```text
human-conditioned persistent identity memory
    -> frozen PCTIS proposal
    -> exact base assignment A0
    -> exact proposal assignment A1
    -> counterfactual safety decision
    -> commit A0 or A1
```

The experiment used the existing N72R9 protocol: 32 events, 18 independent
DanceTrack sequences, action counts AUTHORITATIVE_REASSIGN=14,
RECOVER_IDENTITY=11, ADD_NEW_IDENTITY=4, and ATOMIC_ID_SWAP=3.  The candidate
stream, public/native mapping, checkpoint, exact Hungarian solver, admission
rules, H20/H50/H100 definitions, post-hoc scoring and sequence-cluster
bootstrap were not changed.  Runtime did not read future GT; GT was read only
after sealed runtime rows for post-hoc labels and scoring.  Every event remains
`interaction_source=simulated_from_gt`; real-human tape is 0.  These simulated
events must not be described as historical human clicks.

Key frozen hashes:

| Input | Evidence | Hash/value |
|---|---|---|
| N72R9 protocol | `outputs/N72R9/protocol.json` | `e012ecc3bd64fec4409fccd57d920f3690b8b1c4b5a2dd557e03e2bfb43ef0e9` |
| PCTIS checkpoint | `outputs/N72R11R4/pctis_onpolicy_finetune/pctis_onpolicy_finetuned.pt` | `77b41dbe2fc0f03c48ca4b8cda22e6b219eb58e266566eb96fcf0df3fb8c2c16` |
| pinned TrackEval | `third_party/MOTIP/TrackEval` | commit `12c8791b303e0a0b50f753af204249e622d0281a` |
| learned gate checkpoint | `outputs/N72R12/gate_training/safe_gate.pt` | `f9a7e2a34c54233aa3f22668ac33948f03f407ee957050a292f133a738184474` |
| learned corpus manifest | `outputs/N72R12/gate_corpus/manifest.json` | `dd165d3e6ec8c783f8b395f2d3f24bc44a1c96ff98df81d56c2244b83a57c871` |

The source audit was run at base head
`31f3fecbb51c20cc8adba308fec8e48525e99b09` on branch
`codex/n72r12-safe-intervention`; the pushed commit containing this report is
the branch tip recorded by GitHub.

## Method implementation

`sam3_intermot/association/counterfactual_safe_intervention.py` implements the
fixed `COUNTERFACTUAL_SAFE_V1` policy.  APPLY requires all of the following:

1. the frozen PCTIS selection is accepted and has an explicit selected UID;
2. the proposal changes the target public assignment and selects that UID;
3. the selected candidate has valid geometry and is unowned or owned by the target;
4. no non-target public ID changes;
5. the proposed target edge strictly dominates the existing best-other edge and
   has positive target-edge gain.

Every other case commits the base solver.  A rejected proposal is not allowed
to update trusted or distractor memory.  No new score threshold, cosine
threshold, checkpoint, candidate source, solver, or metric was introduced.
E1B remained the unconditional PCTIS control.  E1C runtime rows contain base,
proposal and committed matrices/deltas, exact solver audits, mapping audits and
the causal event-frame/event+1 boundary.

The learned module is deliberately small and fixed: a 37-D causal feature
vector and `Linear(37,64)-GELU-LayerNorm-Linear(64,32)-GELU-Linear(32,1)`.
Its 0.5 threshold is fixed.  E1D is the intersection of the learned decision
and the deterministic CSI decision; the learned model cannot override a
deterministic KEEP.

## Execution gates

| Stage | Result | Evidence |
|---|---|---|
| Interface and CSI implementation | PASS | `outputs/N72R12/stage_00_status.json`, `stage_01_status.json` |
| E1C integration and causal smoke | PASS | `stage_02_status.json`, `smoke/smoke_validation.json` |
| Post-hoc Oracle headroom | PASS, runtime-ineligible | `oracle/oracle_headroom.json`, `stage_04_status.json` |
| Compile audit | PASS | `stage_05_status.json` |
| Deterministic formal runtime | PASS, 32/32 | `formal_safe/formal_safe_manifest.json`, `stage_07_status.json` |
| Deterministic causal aggregation | PASS, 32 events/18 sequences | `causal/formal_causal_metrics.json`, `stage_08_status.json` |
| Deterministic TrackEval | PASS, 288/288 records | `trackeval/trackeval_run_manifest.json`, `stage_09_status.json` |
| Deterministic scientific gate | FAIL | `stage_11_status.json` |
| Learned safe-gate corpus | PASS structurally | `gate_corpus/manifest.json`, `stage_12_status.json` |
| Learned safe-gate training | PASS as development training; readiness FAIL | `gate_training/training_manifest.json`, `stage_13_status.json` |
| Learned formal runtime | PASS, 32/32 and 3,232 frames | `formal_learned/formal_learned_manifest.json`, `stage_14_status.json` |
| Learned TrackEval | PASS, 384/384 records | `trackeval_learned/trackeval_run_manifest.json`, `stage_15_status.json` |
| Learned scientific gate | FAIL | `stage_17_status.json` |
| Final decision | FAIL_SAFE_INTERVENTION_NOT_SUFFICIENT | `n72r12_final_gate.json` |

All newly generated JSON artifacts were atomically written.  Existing N36--
N72R11R5R1 evidence and `third_party/sam3`/TrackEval were not rewritten.

## Post-hoc Oracle headroom

The Oracle used GT after runtime and selected E1B only when E0 was wrong, E1B
was correct, and no protected identity regressed.  It is explicitly marked
`gt_used_for_decision=true`, `runtime_eligible=false`, and
`oracle_upper_bound_only=true`.  It is not a method, not a runtime result, and
does not authorize promotion.  Its H20/H50/H100 identity-error headroom was
`+0.065253/+0.028314/+0.017891`, with lower bootstrap bounds
`+0.030081/+0.012169/+0.007615`; this only shows that selective intervention
has post-hoc headroom in the frozen rows.

## Deterministic CSI results

The formal CSI stream covered 3,200 future frames: 290 APPLY and 2,910 KEEP.
Runtime validation found 32/32 complete events, no duplicate or missing event,
no unavailable event, no candidate-axis error, and no runtime future-GT read.

### Causal target-centered comparison

Values are identity-error reductions (positive is better) and sequence-cluster
95% CI over 18 independent sequences, seed 7211 and 2,000 repetitions.

| Comparison | H20 | H50 | H100 |
|---|---:|---:|---:|
| E1B - E0 | +0.042414 [-0.041032,+0.201799] | -0.000644 [-0.066751,+0.132664] | -0.008307 [-0.046482,+0.094342] |
| E1C - E0 | +0.040783 [-0.041032,+0.193767] | -0.000644 [-0.066445,+0.127138] | -0.008307 [-0.047357,+0.091093] |

E1C H20 has more correct than incorrect crossings (55/30), but protected
regression is 7 at H20, 8 at H50 and 14 at H100.  Therefore the strict gate
fails even before global TrackEval: its requirement is not merely a positive
pooled H20 mean.

### Interaction-window TrackEval

These are interaction-window diagnostics, not official full DanceTrack
benchmark scores.

| H | Variant | HOTA | AssA | IDF1 | IDSW | DetA | MOTA |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20 | E0 | 0.680935 | 0.772592 | 0.779271 | 32 | 0.605232 | 0.556068 |
| 20 | E1B | 0.676341 | 0.765893 | 0.775195 | 51 | 0.602501 | 0.546193 |
| 20 | E1C | 0.676331 | 0.765909 | 0.775195 | 50 | 0.602485 | 0.546412 |
| 50 | E0 | 0.667856 | 0.733672 | 0.774861 | 91 | 0.610643 | 0.559352 |
| 50 | E1B | 0.662308 | 0.729586 | 0.766764 | 128 | 0.603938 | 0.543228 |
| 50 | E1C | 0.662445 | 0.730246 | 0.767104 | 118 | 0.603667 | 0.544448 |
| 100 | E0 | 0.665767 | 0.721408 | 0.788364 | 177 | 0.616112 | 0.579139 |
| 100 | E1B | 0.660161 | 0.718145 | 0.780025 | 236 | 0.608541 | 0.561075 |
| 100 | E1C | 0.660414 | 0.718722 | 0.780404 | 222 | 0.608517 | 0.561857 |

At H100, E1C-E0 is HOTA `-0.005353`, AssA `-0.002686`, IDF1 `-0.007961`,
and IDSW `+45`.  It improves over unconditional E1B by reducing IDSW by 14,
but remains worse than E0 on every primary global metric.

### Counterfactual confusion diagnosis

The following is a post-hoc classification on visible H100 target frames.  A
harmful proposal is E1B changing an E0-correct target to wrong or causing a
protected regression.  A beneficial proposal is E1B changing an E0-wrong
target to correct with no protected regression.  CSI APPLY/KEEP is read from
the sealed runtime decision; it is not used to define the proposal labels.

| Quantity | Overall |
|---|---:|
| visible target frames | 3,130 |
| harmful proposals | 153 |
| harmful proposals blocked | 45 |
| beneficial proposals | 105 |
| beneficial proposals applied | 21 |
| visible APPLY | 289 |
| visible KEEP | 2,841 |
| harm-block rate | 0.294118 |
| benefit-retention rate | 0.200000 |
| intervention precision | 0.072664 |

The all-frame CSI count is 290 APPLY/2,910 KEEP; one APPLY is in a target-
absent frame and is therefore excluded from the visible-frame precision table.
The action-level pattern is important: deterministic APPLY/KEEP was ADD 3/397,
ATOMIC 48/252, AUTHORITATIVE 80/1320, and RECOVER 159/941.  RECOVER H20 was
still unfavorable (18 correct versus 26 incorrect crossings; reduction
`-0.036364`), and its H100 reduction was `-0.095370`.

## Learned safe gate

The deterministic gate did not pass, so the protocol required one small learned
gate.  No additional model family or hyperparameter search was performed.

Corpus facts:

- 30,390 source rows, 29,890 included samples, 972 positive and 28,918 negative;
- 24,202 train examples from 12 sequences and 5,688 validation examples from 6
  disjoint sequences;
- 488 train and 12 validation target-not-visible rows were excluded;
- labels were generated offline after runtime sealing; runtime future GT remained
  false; the corpus is `simulated_from_gt`, not real-human evidence.

Fixed training configuration:

```text
10 epochs, batch 256, AdamW, lr 1e-3, weight_decay 1e-4
BCEWithLogitsLoss, train-only capped positive weight 10
seed 7212, checkpoint = minimum validation BCE, threshold = 0.5
```

The actual forward/backward, finite-gradient, checkpoint save and restore smoke
passed.  Best validation BCE was `0.309315` at epoch 1.  Readiness failed:
precision `0.020325` (required >=0.75), recall `0.112360`, predicted APPLY
`492`, unsafe APPLY rate `0.979675` (required <=0.10).  This checkpoint is
therefore a development diagnostic only.

### Learned E1D results

E1D formal runtime passed 32/32 events and 3,232 frame rows.  Deterministic
safety allowed 219 APPLY, the learned classifier predicted 109 APPLY, and the
final CSI∩learned decision committed 83 APPLY; 26 learned APPLY predictions
were blocked by the deterministic safety ceiling.

| Comparison | H20 | H50 | H100 |
|---|---:|---:|---:|
| E1D - E0 identity reduction | +0.052202 [+0.000827,+0.191221] | +0.010296 [-0.042244,+0.116941] | +0.009904 [-0.028876,+0.087625] |
| E1D - E1B identity reduction | +0.009788 [-0.024807,+0.056944] | +0.010940 [-0.028748,+0.037968] | +0.018211 [-0.019289,+0.034498] |

E1D correct/incorrect crossings were 51/19 at H20, 71/55 at H50 and 103/72
at H100.  Protected regressions remained 7/8/14, so the learned target-
centered improvement is not a clean global-safe result.

| H | Variant | HOTA | AssA | IDF1 | IDSW |
|---:|---|---:|---:|---:|---:|
| 20 | E1D | 0.677678 | 0.765816 | 0.776695 | 49 |
| 50 | E1D | 0.663024 | 0.729871 | 0.768885 | 120 |
| 100 | E1D | 0.660915 | 0.718277 | 0.783053 | 224 |

At H100, E1D-E0 is HOTA `-0.004852`, AssA `-0.003130`, IDF1 `-0.005311`,
and IDSW `+47`.  E1D improves over E1B at H100 (HOTA +0.000754, AssA
+0.000132, IDF1 +0.003028, IDSW -12), but not over E0.  The learned strict
gate is false because global metrics do not improve and protected regressions
are nonzero.

## Failure and repair ledger

No failure was deleted or converted into a PASS.

| Attempt | Root cause | Minimal repair/evidence |
|---|---|---|
| safe-gate corpus attempt 1 | exact split fields were nested in the frozen source manifest; the exception path also had a flush indentation defect | read the nested manifest and repair the exception path; original failure retained in `gate_corpus/corpus_failure_attempt_01.json` |
| corpus replay | fixed NPZ width 14 was mistaken for actual candidate count | slice by metadata candidate count; `gate_corpus/corpus_failure.json` retains the axis failure |
| training smoke | metadata included excluded target-not-visible rows while NPZ held included rows only | align on `included_in_training=true`; `gate_training/training_failure.json` retained |
| E1D short smoke | safe checkpoint argument was shadowed in `run_event`; then short horizon was unsupported by the baseline validator | remove parameter clobber; use same event at full H100; both failed attempts retained under `learned_smoke/attempts/` |
| learned export | missing `json` import after complete manifest construction | add import and rerun the same export; `trackeval_learned/export_failure_attempt1.json` retained |
| learned TrackEval | wrapper received TrackEval root instead of pinned `scripts/run_mot_challenge.py` | correct only the runner path; first logs/partial manifest retained; final run is 384/384 |
| final status audit | post-hoc script first read a runtime-omitted `dataset_gt_id`, then passed Git subcommands as one argument | use frozen protocol field and split Git arguments; `status_writer_failure_attempt_00_schema.json` and `status_writer_failure_attempt_01.json` retained |

The environment emitted the pre-existing `osr_lib-1.1.0-nspkg.pth` import
warning.  It did not change experiment inputs or results and was not treated as
a model or metric failure.

## Isolation and reproducibility

- No files under `third_party/sam3` or `third_party/MOTIP/TrackEval` changed.
- No N36--N72R11R5R1 output or report was overwritten.
- New experiment artifacts are under `outputs/N72R12/`; they are ignored by the
  repository and are accompanied by machine-readable manifests and hashes.
- The safe-gate training checkpoint is independent of the frozen PCTIS
  checkpoint; PCTIS parameters, SAM3 backbone, candidate generator and exact
  solver were frozen.
- Compile and `git diff --check` passed for the N72R12 modules and scripts.  The
  requested low-value full-repository test sweep was not run; the targeted
  compile/smoke checks are recorded in Stage 05/06.
- `runtime_future_gt_used=false` and `runtime_gt_read=false` are present in the
  sealed runtime manifests and validators.  Post-hoc GT use is explicitly
  separated in the causal and TrackEval aggregators.

## Final decision and next step

The safe intervention idea has useful target-centered signal—especially at H20
and relative to unconditional PCTIS—but it does not close the E0 gap.  The
learned gate is not ready and, even under formal replay, global AssA/IDF1/HOTA
remain below E0 while IDSW remains higher.  This is evidence against continuing
threshold tuning or adding more gate capacity, not evidence that a larger model
should be tried immediately.

Keep the persistent public-identity and exact-assignment foundations as
research infrastructure.  Do not promote CSI/E1D, train a selector or
calibration head, add decoder LoRA/Bridge, or start live requery.  The minimum
next defensible experiment is one newly frozen association-interface probe that
explains the remaining target-edge versus global-assignment conflict, or a
provenance-complete real-human event tape.  Simulated-from-GT events cannot be
used as a substitute for that tape.

Machine-readable final records:

- `outputs/N72R12/n72r12_final_gate.json`
- `outputs/N72R12/CONTROLLER_STATUS.json`
- `outputs/N72R12/intervention_confusion.json`
