# N72R11R2 — Official SAM3 Streaming Memory Trimming and V3/Bridge Development

**Project:** InterMOT / SAM3_InterMOT  
**Run:** N72R11R2  
**Report date:** 2026-09-07 (Asia/Shanghai)  
**Overall status:** `BLOCKED_N72R11R2_INCOMPLETE_FORMAL_REPLAY_AND_V3_NOT_READY`  
**Production authorization:** `false`

## 1. Executive decision

N72R11R2 produced useful engineering and development evidence, but it did not
produce a complete scientific or production result.

The frozen formal development replay required 32 event records. Thirty-one
events have sealed PASS artifacts and one event remains an explicit child
failure caused by CUDA out-of-memory inside official SAM3 future propagation.
The combined manifest therefore remains incomplete. The resource-censored
development route is valid as a diagnostic/training route, but it is not a
replacement for the missing formal event.

Temporal V3 and TargetEdgeBridge were actually trained in isolated outputs.
V3 readiness failed its preregistered source-validation thresholds, so neither
checkpoint is a production candidate. No Oracle, calibration head, selector or
decoder LoRA was run or authorized.

The correct terminal conclusion is a resource/incompleteness block, not a
model PASS and not a claim that the identity hypothesis has been disproved.

## 2. Frozen scope and provenance

- The pinned official SAM3 source was kept unchanged at commit
  `4cbac146c1b5a1e3a7f5c6a894901090b4dfd65b`.
- Checkpoint, candidate definition, public/native mapping contract, Hungarian
  evaluation, event protocol, H20/H50/H100 windows and GT posthoc protocol
  were not changed.
- All interaction records in this development route are
  `interaction_source=simulated_from_gt`; this is not real-human evidence.
- `runtime_future_gt_used=false` was retained in the runtime and posthoc
  provenance audits. GT was used only for offline corpus labels and posthoc
  scoring.
- No file under `third_party/sam3` was modified.
- New evidence was written below `outputs/N72R11R2/`; historical N36--N72R11R1
  evidence was not overwritten.

The formal event-level replay reuses the frozen N72R9 development protocol
(32 events, 18 sequences) through the N72R11 worker contract. Its aggregate
metrics use independent-sequence bootstrap with seed `7211` and `2000`
repetitions. These metrics are descriptive while one required event is
unsealed/failed.

## 3. Official streaming implementation and equivalence

The implementation added official SAM3 propagation through an isolated
streaming path and release of completed sessions after observation
materialization. The principal modified files are:

- `sam3_intermot/backend/sam3_backend.py`
- `sam3_intermot/interaction/target_correction_session.py`
- `sam3_intermot/reacquisition/future_requery_session.py`
- `sam3_intermot/reacquisition/live_requery_controller.py`
- `scripts/n72r11_generate_secondary_interaction.py`
- `scripts/n72r11_on_demand_replay.py`
- `scripts/n72r11_build_causal_corpus.py`
- `scripts/n72r11_train_v3.py`
- `scripts/n72r11_train_target_edge_bridge.py`
- `scripts/n72r11_run_formal_replay_batch.py`

The first two streaming-equivalence attempts preserved the official
`KeyError: 'multistep_point_inputs'` failure. The minimal adapter delayed the
trim call until a compatible non-conditioning output was present, without
changing the checkpoint, candidate stream, precision or metric protocol.
Attempt 3 then compared the same 41-frame development smoke:

- scientific candidate/mask mismatches: `0`
- candidate frames and rows compared: `41 / 41`
- official propagation: active
- runtime future GT: `false`

The equivalence result is
`outputs/N72R11R2/streaming_equivalence_audit.json` with status
`PASS_SCIENTIFIC_CANDIDATE_EQUIVALENCE`.

This does **not** mean official memory trimming was effective. The trim flag
was requested, but the effective trim count was `0`; 34 observations retained
the schema blocker `past_output_missing_multistep_point_inputs`. The verified
mechanisms were video/output CPU offload, not state CPU offload:

| Mechanism | Observed state |
|---|---|
| output-to-CPU for evaluation | enabled |
| video-to-CPU | enabled |
| official state-to-CPU | disabled/not available |
| official trim requested | yes |
| official trim effective | no, 0 frames |

The requested singleton fallback was also rejected. The pinned checkpoint has
fixed 16-object tensor shapes, while a `max_num_objects=1` load failed before
candidate generation. `singleton_backend_equivalence_audit.json` records
`SINGLETON_NOT_SEMANTICS_PRESERVING`; singleton execution was not used to
claim a memory repair.

## 4. Resource audit and failure history

The frozen resource schedule contained 527 event-policy instances. The audit
found:

| Quantity | Count |
|---|---:|
| required schedule | 527 |
| observed records | 527 |
| unique records | 527 |
| duplicate records | 0 |
| missing schedule records | 0 |
| invalid/unclassified records | 0 |
| retained executable records | 516 |
| residual CUDA-OOM records | 11 |

The 11 resource-censored records were retained as failures and excluded from
the executable development corpus. They are:

```text
n72r5-pool-n37-dancetrack0012-0040-add_new_identity-012:secondary:050
n72r5-pool-n37-dancetrack0012-0040-add_new_identity-012:secondary:070
n72r5-pool-n37-dancetrack0015-0002-recover_identity-001:secondary:005
n72r5-pool-n37-dancetrack0015-0002-recover_identity-001:secondary:010
n72r5-pool-n37-dancetrack0015-0002-recover_identity-001:secondary:015
n72r5-pool-n37-dancetrack0015-0002-recover_identity-001:secondary:020
n72r5-pool-n37-dancetrack0015-0194-authoritative_reassign-001:secondary:005
n72r5-pool-n37-dancetrack0016-0553-authoritative_reassign-007:secondary:035
n72r5-pool-n37-dancetrack0024-0369-recover_identity-113:secondary:000
n72r5-pool-n37-dancetrack0033-0154-recover_identity-108:secondary:005
n72r5-pool-n37-dancetrack0062-0291-add_new_identity-001:secondary:070
```

The resource audit is
`outputs/N72R11R2/resource_censoring_audit.json` with status
`PASS_RESOURCE_CENSORED_DEVELOPMENT`. “PASS” here means the censoring
classification is internally complete; it does not turn the 11 OOM items into
successful model executions.

Formal replay supervision proceeded through three preserved attempts:

1. Attempt 1 ended before harvest. Its source manifest contained 14 sealed
   PASS records, four stale `RUNNING` records and 14 `NOT_RUN` records.
2. Attempt 2 resumed the 18 unfinished records but its supervisor also ended
   before harvest. It is retained as provenance and is not used as a source of
   PASS records.
3. Attempt 3 was run detached with four isolated workers on GPUs 1--4 and
   completed naturally with 17 PASS records and one `FAIL_CHILD`.

The sole failed formal event is:

```text
event:    n72r5-pool-n37-dancetrack0062-0291-add_new_identity-001
action:   ADD_NEW_IDENTITY
sequence: dancetrack0062
returncode: 1
error:    CUDA out of memory during official SAM3 future propagation
```

The preserved traceback reports a 2.00 MiB allocation request with only
1.06 MiB free on a 39.39 GiB device; 37.43 GiB was allocated, 322.38 MiB was
reserved, and the process reported 39.39 GiB in use including non-PyTorch
memory. This is a real execution failure, not a validator or metric failure.

The combined audit is
`outputs/N72R11R2/formal_replay_resource_censored_combined_manifest.json`:

- required events: `32`
- sealed PASS events: `31`
- unsealed/failed events: `1`
- duplicate event IDs: `0`
- missing/failed event: the `dancetrack0062` event above
- combined status: `INCOMPLETE_FORMAL_DEVELOPMENT_REPLAY`

The previous full-window smoke failure and all supervisor interruption facts
remain preserved. The current environment does not provide another verified,
equivalent low-risk memory mechanism. The lawful next execution would require
a genuinely larger/less-contended memory environment or a supported official
SAM3 state-memory offload mechanism; shortening the window, changing precision,
changing the checkpoint or deleting the failed event would violate the frozen
protocol.

## 5. Independent runtime-row audit

The first independent checker was intentionally retained after it exposed a
schema assumption: legacy E0 rows use `BASELINE_B0`, E0 materializes
`exact_global_solver_output`, and E0 has a fused-score matrix without the
modern base-score matrix. The raw replay artifacts were not changed.

After accepting those observed, documented schema variants, the repaired audit
passed for the 31 sealed events:

- runtime rows: `9,393`
- variant artifacts: `93` (`31 × 3`)
- event-frame rows: `93`
- candidate rows: `77,440`
- legacy E0 score-schema rows: `3,100`
- audit errors: `0`
- runtime future GT: `false`
- posthoc GT in runtime rows: `false`

The result is
`outputs/N72R11R2/formal_replay_resource_censored_runtime_audit_attempt_02.json`
with status `PASS_FORMAL_RUNTIME_ROWS_31_EVENTS`. It is explicitly a 31-event
audit and cannot satisfy the 32-event scientific completeness gate.

## 6. Development metrics (descriptive only)

The corrected aggregate is
`outputs/N72R11R2/formal_replay_resource_censored_metrics_attempt_02.json`.
Its status is `PASS_METRICS_FOR_31_SEALED_EVENTS_INCOMPLETE_FORMAL_REPLAY`.
The metrics were computed only from sealed PASS records and use the frozen
independent-sequence bootstrap. They are not a full formal gate.

For the live bridge path versus baseline (`E2_vs_E0`), the aggregate results
were:

| Horizon | Identity-error reduction | Bootstrap 95% CI | Assignment changes | Correct / incorrect crossings | Protected regressions |
|---|---:|---|---:|---:|---:|
| H20 | +0.035413 | [-0.138979, +0.390649] | 371 | 105 / 84 | 10 |
| H50 | -0.096410 | [-0.275483, +0.225692] | 892 | 190 / 335 | 20 |
| H100 | -0.180528 | [-0.345926, +0.086496] | 1885 | 285 / 832 | 26 |

The internal V3 bridge increment (`E2_vs_E1`) had small positive aggregate
reductions, but its bootstrap lower bounds were exactly zero at all three
horizons: `0`, `0`, and `0`. The combined path therefore does not meet a
strict positive future-effect gate, and the missing 32nd event prevents a full
gate evaluation regardless.

At H20, the action decomposition for `E2_vs_E0` was:

| Action | Events / sequences | Reduction | 95% CI lower bound | Correct / incorrect crossings |
|---|---:|---:|---:|---:|
| ADD_NEW_IDENTITY | 3 / 3 | +0.517857 | +0.200000 | 32 / 3 |
| ATOMIC_ID_SWAP | 3 / 3 | +0.333333 | -0.050000 | 21 / 1 |
| AUTHORITATIVE_REASSIGN | 14 / 10 | -0.101167 | -0.298359 | 20 / 46 |
| RECOVER_IDENTITY | 11 / 8 | -0.009091 | -0.400000 | 32 / 34 |

The positive ADD result is based on only three events and cannot be promoted
to a general result. Reassignment and recovery dominate the available event
count and are not positive in this incomplete development replay.

## 7. Corpus, V3 and TargetEdgeBridge training

The first resource-mode corpus build failed because the merged R1 schema did
not contain the required `returncode` field. That failure is preserved. The
same frozen inputs were rebuilt successfully in:
`outputs/N72R11R2/training_v3_resource_censored_attempt_02/`.

Corpus facts:

- resource-censored executable events: `516`
- train examples: `24,690`
- validation examples: `5,700`
- sequence split: fixed `12 train / 6 validation`
- labels: offline box-IoU labels only
- runtime future GT: `false`
- source: `simulated_from_gt`

V3 training was real isolated training, not a placeholder:

- smoke: `PASS_N72R11_TRAINING_SMOKE`, finite loss `2.777571`
- bootstrap checkpoint: `v3_bootstrap.pt`
- checkpoint SHA-256:
  `4abae660d80ce105618ae25eb9bc2d91a6b2dd2073e51fca4599d9a643c81eb2`
- fixed training: two epochs, seed `7211`
- epoch 1: train loss `0.556575`, validation accuracy `0.452281`, validation loss `2.641313`
- epoch 2: train loss `0.330574`, validation accuracy `0.412632`, validation loss `3.264498`

The first self-rollout failed on an invalid zero feature in distractor memory;
the failure is preserved. A CPU-only rerun after the minimal feature-validity
fix passed with 97 events, 5,700 examples, 4,466 accepted candidates and
1,234 `NONE` outputs. This is a causal self-rollout audit, not proof of
future tracking efficacy.

The V3 readiness audit is
`outputs/N72R11R2/v3_training_resource_censored/v3_readiness_audit.json` with
status `SOURCE_VALIDATION_UNDERCOVERED_OR_V3_NOT_READY`:

| Readiness quantity | Observed | Required |
|---|---:|---:|
| future-positive count | 4,348 | >= 50 |
| future-positive accuracy | 0.451932 | >= 0.60 |
| NONE accuracy | 0.265533 | >= 0.50 |
| target-candidate accuracy | gate false / undercovered | >= 0.80 |

The action breakdown also shows a severe ADD source problem: future-positive
accuracy `0.043891` and NONE accuracy `0.040138`. The readiness gate is false;
the checkpoint is not a production selector.

TargetEdgeBridge was also actually trained, but remains development-only:

- smoke: `PASS_N72R11_BRIDGE_SMOKE`
- input dimension: `14`
- residual scale: `4.9380065`
- fixed training: three epochs
- final validation accuracy: `0.456842`
- checkpoint SHA-256:
  `973376ca5fb4e48ffbf21b2d1413ffbbbfb46b29c0f08fa216fbdf684b2125ab`
- status: `PASS_N72R11_BRIDGE_TRAINING`, `production_authorized=false`

## 8. Preserved failures and repairs

The following facts remain as separate artifacts; none was deleted or silently
converted to PASS:

| Fact | Preserved evidence |
|---|---|
| official trim schema failure (`multistep_point_inputs`) | `outputs/N72R11R2/streaming_equivalence_attempt_01/`, `attempt_02/`, `INTERFACE_MISMATCH.json` |
| full-window OOM smoke | `outputs/N72R11R2/oom_smoke_attempt_01/` |
| singleton checkpoint shape mismatch | `outputs/N72R11R2/singleton_equivalence_1x1_attempt_01/`, `singleton_backend_equivalence_audit.json` |
| supervisor interruption, attempt 1 | `formal_replay_resource_censored_attempt_01/supervisor_interruption_audit_attempt_01.json` |
| supervisor interruption, attempt 2 | `formal_replay_resource_censored_attempt_02/supervisor_interruption_audit_attempt_02.json` |
| final formal child OOM | `formal_replay_resource_censored_attempt_03/attempts/` and worker log |
| initial corpus schema failure | `training_v3_resource_censored/attempts/corpus_failure_20260906T164056Z.json` |
| initial V3 self-rollout failure | `v3_training_resource_censored/attempts/self_rollout_failure_20260906T164654Z.json` |
| initial metrics provenance-flag error | `formal_replay_resource_censored_metrics_attempt_01_semantic_flag_failure.json` |
| initial runtime-row schema assumption | `formal_replay_resource_censored_runtime_audit_attempt_01_schema_failure.json` |

The Python environment emitted a recurring `osr_lib-1.1.0-nspkg.pth`
`AttributeError` warning during invocations. It was non-fatal: commands
returned their expected codes and produced valid artifacts. It was not
classified as a worker or scientific failure.

## 9. Storage and isolation

The prior storage cleanup moved approximately 5.4 GiB of old N15/N18/N21/N22/N30
artifacts to the recoverable quarantine directory
`/data2/usr_for_deadline/intermot_storage_quarantine_20260906/`. It did not
delete active checkpoints, third-party SAM3 code, N36--N40 evidence or the
N72R11R2 evidence required by this report.

Code changes were made in the isolated Git worktree
`/data2/usr_for_deadline/SAM3_InterMOT_N72R5/worktree`, on branch
`codex/n72r11r2-streaming-memory`, with remote
`https://github.com/LYQ1107/InterMOT.git`. The final commit and remote branch
verification are recorded after the final self-check. The new scripts are
also included:

- `scripts/n72r11r2_audit_formal_replay.py`
- `scripts/n72r11r2_aggregate_formal_metrics.py`
- `scripts/n72r11r2_audit_formal_runtime_rows.py`
- `scripts/n72r11r2_write_terminal_status.py`

## 10. Final machine-readable gate and next step

The terminal machine-readable files are:

- `outputs/N72R11R2/CONTROLLER_STATUS.json`
- `outputs/N72R11R2/stage_14_formal_replay_status.json`
- `outputs/N72R11R2/stage_15_final_gate.json`

They record:

- no active long process at the safe boundary;
- 527/527 resource schedule coverage, 516 executable and 11 residual OOM;
- 31/32 formal event PASS and one preserved failure;
- zero duplicate formal event IDs;
- runtime-row audit clean for the 31 selected sealed events;
- V3 readiness false;
- Bridge trained but not authorized;
- production, Oracle, calibration, selector and decoder LoRA all false.

The only defensible next step is to obtain a genuinely larger/less-contended
GPU-memory environment, or verify an official SAM3 state-memory offload
interface without changing the experiment definition, and rerun only the
preserved failed event. If that cannot be provided, retain this block and do
not spend the remaining schedule on blind weight scaling or downstream
training.

## 11. ICLR 2027 schedule constraint

Using the project’s fixed dates:

- abstract deadline: **2026-09-18 AoE**;
- full-paper deadline: **2026-09-25 AoE**;
- report date: **2026-09-07**, leaving approximately 11 days to the abstract
  deadline and 18 days to the full-paper deadline.

The schedule should prioritize one narrow memory-environment recovery and the
missing formal event only. A complete 32-event gate and V3 readiness decision
must precede any claim of a learned production mechanism. A failed recovery
must remain a clearly documented systems limitation rather than be hidden by
resource censoring.

## 12. Reproducibility index

| Evidence | Path |
|---|---|
| terminal gate | `outputs/N72R11R2/stage_15_final_gate.json` |
| controller status | `outputs/N72R11R2/CONTROLLER_STATUS.json` |
| formal combined audit | `outputs/N72R11R2/formal_replay_resource_censored_combined_manifest.json` |
| corrected descriptive metrics | `outputs/N72R11R2/formal_replay_resource_censored_metrics_attempt_02.json` |
| corrected runtime audit | `outputs/N72R11R2/formal_replay_resource_censored_runtime_audit_attempt_02.json` |
| resource audit | `outputs/N72R11R2/resource_censoring_audit.json` |
| streaming equivalence | `outputs/N72R11R2/streaming_equivalence_audit.json` |
| V3 readiness | `outputs/N72R11R2/v3_training_resource_censored/v3_readiness_audit.json` |
| Bridge training | `outputs/N72R11R2/bridge_training_resource_censored/stage_13_bridge_training.json` |

**Final conclusion:** N72R11R2 demonstrates a scientifically equivalent
official streaming candidate path and completes isolated development training,
but it does not complete the formal replay and does not establish V3 readiness
or a future-effect/production result. The correct status remains
`BLOCKED_N72R11R2_INCOMPLETE_FORMAL_REPLAY_AND_V3_NOT_READY`.
