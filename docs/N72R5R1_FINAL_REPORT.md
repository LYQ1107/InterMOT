# InterMOT N72R5R1 Final Report

- Generated: `2026-09-04` (Asia/Shanghai)
- Final controller status: `EXHAUSTED_PRE_REGISTERED_MECHANISMS_NO_EFFECT`
- Research gate: `FAIL_FUTURE_EFFECT`
- Final root cause: `GEOMETRY_NATIVE_OR_CANDIDATE_STREAM_DECISION_BOUNDARY_DOMINATES_APPEARANCE_STATE`
- Runtime future GT: `false`
- Interaction source: `simulated_from_gt`; this is not real-human evidence
- Real human event tape: `0`

## Executive decision

The missing N72R5 public-association layer is now implemented and independently validated
without rerunning the frozen Stage07 SAM3 candidate workers. All 40 events, 20 independent
sequences, 200 branches, 20,200 frame rows, and 146,176 candidate rows are connected to a
persistent public-ID plus explicit `NONE` decision axis. Stage08 exact association and Stage09
runtime validation pass with full decision coverage and no runtime future-GT use.

The corrected 40-event V0 posthoc evaluation still fails the strict future-effect gate. The
primary B4-minus-B0 result is negative at every horizon, with protected-ID regression and more
incorrect than correct first-future crossings. Six evidence rounds were completed, including
the branch-isolation repair and a causal persistence probe. No mechanism is authorized for
production promotion, calibration, selector training, or decoder LoRA. The correct terminal
state is exhaustion of the preregistered synthetic mechanisms, not PASS and not a claim that
simulated events are historical human interactions.

## Frozen scope and provenance

The following N72R5 artifacts were reused read-only:

- `outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json`;
- the Stage07 official candidate stream and CPU audit;
- the original checkpoint, candidate definition, candidate order, Hungarian implementation,
  H20/H50/H100 definitions, and sequence-cluster bootstrap (`seed=7202`, `repetitions=2000`);
- no val/test data and no new synthetic event.

Stage07 was not rerun. `max_num_objects=24` and the N72R5 capacity contract remained frozen.
No file under `third_party/sam3` and no N36--N72R5 historical evidence was modified. All new
runtime/effect artifacts are under `outputs/N72R5R1/`; the authoritative corrected V0 run is
isolated at `outputs/N72R5R1/controller/round_05_branch_isolation_v0/full/`.

All events retain `interaction_source=simulated_from_gt` and
`not_real_human_evidence=true`. The simulated oracle supplies public authority only inside the
posthoc-controlled experiment; it must not be relabeled as a real human tape.

## Stage08 exact persistent public association

| Check | Result |
|---|---:|
| Selected events | `40/40` |
| Independent sequences | `20` |
| Branch artifacts | `200/200` |
| Frame rows | `20,200/20,200` |
| Candidate rows observed | `146,176` |
| Formal candidate decision coverage | `1.0` |
| Public assignment completeness | `true` |
| Runtime future GT | `false` |
| Stage08 status | `PASS_N72R5R1_EXACT_PUBLIC_ASSOCIATION` |

The formal decision contract is `ASSIGNED_TO_PUBLIC_ID` or `EXPLICIT_NONE` for candidates,
and `ASSIGNED` or `NO_CANDIDATE_ASSIGNED` for identities. The 72 null `target_public_id`
fields in the branch manifest are intentional B0/no-intervention or unavailable-event target
annotations; they are not silently filled. The sidecars still carry the complete candidate
decision axis and the validator treats `NONE` as a valid decision.

Every branch owns one isolated simulated oracle initialized from the common prefix mapping.
This repaired a real cross-branch contamination bug: the old sequential oracle caused four ADD
events to report `ADD_TARGET_ALREADY_HAS_PREFIX_PUBLIC` in B2--B4 after B1 committed the same
target. The isolated run has zero mapping conflicts, zero Y_PRE mismatches, and zero new
inconsistent events. Branch-local mappings are retained for audit.

Action preconditions in the corrected V0 run were:

- `32` events with `APPLIED` action transactions;
- `4` events with `ATOMIC_OTHER_CANDIDATE_UNAVAILABLE`;
- `4` events with `TARGET_CANDIDATE_UNAVAILABLE`;
- the eight incomplete transactions remain explicit evidence and are not relabeled as action
  success. The structural Stage08 branch status is valid because candidate association is
  complete for every branch.

## Stage09 public/runtime validation

The independent validator reports `200/200` expected branches, `0` missing branch keys,
duplicate keys, extra keys, frame errors, or validator errors, `146,176` candidate decision rows,
and `0` legacy outer-birth status rows. It reports `strict_pass=true`, `posthoc_gt_opened=false`,
and `runtime_future_gt_used=false`.

Stage09 is a structural/runtime pass only. It is not a future-effect claim.

## Corrected V0 Stage10 effect result

The authoritative result is the dedicated V0 run after branch-oracle isolation, with
`TVC_MODE=V0` and `PERSISTENCE_MODE=OFF`. It contains 585 event metrics (`5` pairs × `39`
evaluable events × `3` horizons); one event remains unavailable because its target public mapping
cannot be resolved. The sequence-cluster unit is the independent sequence, not the frame.

### Identity-error reduction (higher is better)

| Pair | H20 mean [95% CI] | H50 mean [95% CI] | H100 mean [95% CI] |
|---|---|---|---|
| B1−B0 | `-0.480853` [`-0.559101`, `-0.121408`] | `-0.504792` [`-0.602364`, `-0.132551`] | `-0.526172` [`-0.626235`, `-0.214823`] |
| B2−B1 | `-0.005128` [`-0.013158`, `0.001974`] | `-0.001319` [`-0.003759`, `0.000789`] | `0.000190` [`-0.001716`, `0.001579`] |
| B3−B1 | `0.000000` [`0.000000`, `0.000000`] | `0.000000` [`0.000000`, `0.000000`] | `0.000000` [`0.000000`, `0.000000`] |
| B4−B2 | `0.000000` [`0.000000`, `0.000000`] | `0.000000` [`0.000000`, `0.000000`] | `0.000000` [`0.000000`, `0.000000`] |
| B4−B0 | `-0.485982` [`-0.564105`, `-0.123497`] | `-0.506111` [`-0.603379`, `-0.133285`] | `-0.525982` [`-0.626231`, `-0.214503`] |

At H20, the first-future-frame crossing counts and protected regression counts were:

| Pair | Assignment changes | Correct crossings | Incorrect crossings | Protected regression |
|---|---:|---:|---:|---:|
| B1−B0 | 27 | 5 | 17 | 976 |
| B2−B1 | 2 | 0 | 0 | 54 |
| B3−B1 | 0 | 0 | 0 | 0 |
| B4−B2 | 0 | 0 | 0 | 0 |
| B4−B0 | 28 | 5 | 17 | 1,014 |

The primary gate fails on both effect direction and protected identity safety. A positive local
crossing count cannot override the negative sequence-cluster effect.

## Autonomous root-cause rounds

### Round 01 — 40-event decision-boundary audit

The new exact-public artifacts, rather than the old six-event evidence, were audited. There were
39 usable events across 19 sequences and 15,600 frame records. Root-cause counts were:

| Classification | Count |
|---|---:|
| Candidate absent | 3,890 |
| Correct candidate present, target loses competitor | 3,940 |
| Solver competition | 5,402 |
| Target already correct | 2,368 |

Candidate absence is `3,890/15,600 = 0.249359` rows; candidate-present decision errors account
for the majority. The global required target-row residual had median `8.865210`, p90
`9.382172`, and max `10.393968`, while the observed V0 residual was capped at `1.0` (median and
p90 among finite rows). This supports a decision-boundary/scale problem, not a
candidate-absence-only explanation.

### Round 02 — protected transaction audit

The protected-ID transaction run compared 200 branches and 40 events. It produced zero event
assignment changes, zero event+1 changes, zero future assignment changes, and zero action-status
changes across 116 applied branches with protected locks. The lock is safe but has no useful
future effect and was not promoted.

### Round 03 — 40-event feature-separability audit

The posthoc target-versus-competitor table contains 10,092 finite pairs. Direction-correct rates
were anchor `0.740884`, persistent prototype `0.624892`, and temporal feature `0.197759`. The
fused-gap quantiles were median `-4.438776`, p90 `4.555855`, p95 `4.782185`, max `5.211033`.
At H20, 87/113 inspected pairs were target-present but wrongly assigned. Appearance has usable
information, but it is not reliably strong enough to dominate the global solver.

### Round 04 — small learned TVC_V1 probe

Because the feature gate was non-null, the permitted small verifier was trained without future
effect selection: sequence-disjoint split; `8,088` training pairs and `2,004` holdout pairs;
logistic residual model with 3 audited features, seed `7202`, 400 epochs, learning rate `0.05`,
L2 `0.01`, and maximum residual `8.0`. Holdout AUC was `0.6628215624638946`; model SHA-256 was
`833f9ad00795332644ad33202ed62c83dbeda640826df1b89ec46e0a9010b441`.

On the branch-isolated V1 replay, B3−B1 had H20 mean `0.288160` with CI
`[0.084237,0.308715]`, 15 changes, 13 correct and 0 incorrect; B4−B2 had the same incremental
signal. However B4−B0 remained negative (`-0.197822`, CI `[-0.327683,0.018381]`) with 8
incorrect crossings and 928 protected regressions. V1 is an incremental probe, not a safe
system mechanism.

### Round 05 — branch-oracle isolation repair

This engineering repair made the four treatment branches independent. The old sequential-oracle
inconsistency count was 4; after creating a fresh branch oracle from the same prefix mapping it
was 0 across all 40 events and 200 branches. The repaired V1 effect still failed the full gate,
so this repair did not manufacture a scientific PASS.

### Round 06 — persistence audit and causal probe

The persistence audit found the decisive runtime symptom in the 32 applied B1 events:

- event+1 B0 target correct: `20/32`; B1 target correct: `9/32`;
- B1 target public ID overwritten by a non-target: `22` events;
- target public ID lost/none: `0` at event+1;
- candidate stream changed in `32/32` events and all `3,200/3,200` H100 future frame rows;
- H100 B1 target-public overwrite: `2,407/3,200` (`0.752188`);
- H100 target lost/none: `66/3,200` (`0.020625`).

A single opt-in persistence probe froze the machine prototype after the event for non-B0
treatments while preserving motion and official future state. It passed the causal boundary:
event-frame write hidden, event+1 visible, `3,200` expected/observed freeze frames, and zero
boundary errors. It changed 3,105 finite fused-score rows with maximum absolute delta `0.656545`,
but changed zero B1 assignments across 4,040 compared frames. This rejects “machine appearance
overwrite alone” as the actionable bottleneck: score changes are absorbed by geometry/native/
candidate-stream decisions.

## Mechanism routing and what was/was not falsified

1. Exact persistent public association: structurally completed and validated.
2. Current TVC_V0: tested on the corrected 40-event public axis; no correct crossing and
   negative B4−B0 effect.
3. Existing official image-grounded recovery branch: B2−B1 gave no positive effect; candidate
   absence was not dominant, so the preregistered improved Recovery V2 branch was not triggered.
4. TVC_V1: allowed by feature evidence and produced an incremental B3−B1 signal, but failed the
   end-to-end B4−B0 gate and protected-ID safety.
5. Temporal identity context: temporal features were audited, but the preregistered trigger
   “appearance representation insufficient” was not satisfied; V1 showed usable appearance
   direction and AUC. No unsupported large temporal model was introduced.
6. Persistence/state probe: causally valid, but it changed scores without a B1 assignment
   crossing. It supports the final decision-boundary diagnosis.

This routing is why the controller ends after six evidence rounds. It does not claim that every
possible future association interface is impossible; it records that the preregistered
mechanisms on this frozen candidate stream did not establish a safe future effect.

## Answers to the required scientific questions

1. **Public association:** completed structurally with persistent public IDs and explicit NONE.
2. **B0--B4:** the corrected V0 table above is authoritative; primary B4−B0 is negative.
3. **Spatial correction:** no improvement; B1−B0 is strongly negative.
4. **Image recovery:** no positive incremental B2−B1 effect.
5. **TVC:** V0 had no correct crossing; V1 had a local incremental signal but was unsafe end to end.
6. **V0 failure:** required global-assignment residuals far exceed observed TVC residuals.
7. **Learned/temporal TVC:** V1 was triggered and tested; a separate temporal model was not.
8. **Dominant root cause:** candidate-present global association/decision-boundary competition,
   amplified by correction-induced state/candidate-stream drift.
9. **H20/H50/H100 primary means:** `-0.485982`, `-0.506111`, and `-0.525982`.
10. **Statistical confirmation:** none; all primary lower bounds are negative.
11. **Unsupported mechanisms:** V0, spatial correction as a beneficial treatment, B2 recovery,
    the tested V1 end-to-end system, and persistence freezing as a standalone fix.
12. **Best mechanism:** no production mechanism is promoted; V1 is only the strongest
    incremental diagnostic, while exact association is structurally accepted.

## Failure and repair provenance

No failure evidence was deleted. Preserved artifacts include Stage08 historical attempts;
Round02 initial `NameError`/`TypeError`; Round03 initial failure and corrected attempt2; Round06
`KeyError('horizon')`; persistence-probe `NameError`, non-finite JSON, and smoke `AttributeError`;
and the no-environment Stage10 invocation at
`outputs/N72R5R1/controller/round_05_branch_isolation_v0/attempts/stage10_wrong_root_attempt.json`.

The last item was an operational invocation error: exit code 1 using the script's default output
root because `N72R5R1_RUN_ROOT` was omitted. It was not used as the authoritative V0 result; the
explicit-root rerun completed and produced the metrics above. The unrelated `osr_lib` `.pth`
startup warning appeared during Python commands, but compilation, imports, Stage08, Stage09,
and Stage10 reached their intended code paths.

## Isolation, validation, and authorization

The required `py_compile` set and minimal runtime import probe passed. No broad historical test
suite was rerun. No SAM3 checkpoint, backbone, candidate generator, Hungarian solver, metric, or
third-party SAM3 file was changed. No calibration head, selector, decoder LoRA, or production
association promotion is authorized.

| Authorization | Decision |
|---|---|
| Production efficacy promotion | `false` |
| Calibration head | `false` |
| Selector | `false` |
| Decoder LoRA | `false` |
| Confirmation experiment | `false` |
| Runtime future-GT use | `false` |

The minimum scientifically meaningful next step is external provenance-complete real-human tape
and a newly frozen association-interface probe. This report does not claim real-human efficacy.

## Reproducibility artifacts and hashes

- [controller status](../outputs/N72R5R1/CONTROLLER_STATUS.json)
- [human-readable status](../outputs/N72R5R1/HUMAN_READABLE_STATUS.md)
- [corrected V0 Stage08 manifest](../outputs/N72R5R1/controller/round_05_branch_isolation_v0/full/stage08_runtime_manifest.json)
- [corrected V0 Stage09 validation](../outputs/N72R5R1/controller/round_05_branch_isolation_v0/full/stage09_validation.json)
- [corrected V0 Stage10 effect](../outputs/N72R5R1/controller/round_05_branch_isolation_v0/full/stage10_effect_scoring.json)
- [decision-boundary summary](../outputs/N72R5R1/controller/round_01_decision_boundary/decision_boundary_summary.json)
- [feature-separability summary](../outputs/N72R5R1/controller/round_03_feature_separability_attempt2/feature_separability_summary.json)
- [TVC_V1 audit](../outputs/N72R5R1/controller/round_04_tvc_v1/audit/round_04_mechanism_audit.json)
- [branch-isolation audit](../outputs/N72R5R1/controller/round_05_branch_isolation/audit/round_05_mechanism_audit.json)
- [persistence audit](../outputs/N72R5R1/controller/round_06_persistence_audit/persistence_audit_summary.json)
- [persistence-probe audit](../outputs/N72R5R1/controller/round_06_persistence_probe/audit/round_06_persistence_probe_audit.json)

Selected SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| Frozen N72R5 final gate | `05c3390e25f74b1d57d7d6cb078c055c779616f6c1c2197293ac45139632448c` |
| Frozen Stage07 CPU audit | `4cdcdc7ab9a486da39ba006025165927901ac222a4241023aedf09498260228b` |
| Frozen event policy | `4f9be3fb7487ef005b5826d51b8c24ec1a7aaf8b247be359effcec2498e0bff1` |
| Corrected V0 Stage08 | `0ba205c7a3b24d8b816ad2893c27905fc09b4dc922a9ecc217072959b9329b0d` |
| Corrected V0 Stage09 | `2abc272a96e298cdb6c181b463fa43fa505894027e66d51fc0c2145e6c23ec5a` |
| Corrected V0 Stage10 | `925b620020cb701d20e255b60c3c1720d10859f6931675484c3ad6c4b3695538` |
| Feature-separability summary | `73401a29c9b7fe1de75a1c2d4b3df35d94c7e8e52fd6fa7cc8af02b65affab04` |
| TVC_V1 model | `833f9ad00795332644ad33202ed62c83dbeda640826df1b89ec46e0a9010b441` |
| Persistence-probe audit | `e13a435e9a7b1abfe79bf6668e8762d71b0651dd85c6c7021bab493054e3e2f9` |

The code additions are isolated to `branch_public_replay.py` and the `n72r5r1_*.py` scripts;
the code and this report are pushed before handoff.
