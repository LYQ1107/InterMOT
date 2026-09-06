# InterMOT N72R11R1 — OOM Recovery and Terminal Resource-Gate Report

**Date:** 2026-09-06 (Asia/Shanghai)
**Branch:** `codex/n72r11r1-oom-recovery`
**Terminal status:** `BLOCKED_INTRINSIC_OFFICIAL_SAM3_MEMORY`
**Research status:** `BLOCKED_INCOMPLETE_SECONDARY_CORPUS`

## Executive decision

N72R11R1 completed the authorized recovery procedure for the 37 preserved
N72R11 secondary-interaction OOM keys, but 11 keys still failed with genuine
official SAM3 `OutOfMemoryError` exceptions. The final merged schedule contains
527 unique keys, 516 PASS and 11 `FAIL_CHILD`, with zero duplicate and zero
missing keys. The complete-secondary-corpus gate therefore remains false.

This is a bounded resource block, not a scientific success or a scientific
failure claim. The causal corpus builder, Temporal V3 training,
TargetEdgeBridge training, formal E0/E1/E2 replay and any production
promotion were not run because their prerequisite corpus was incomplete.

All N72R11R1 interactions remain `simulated_from_gt`; no real-human evidence
was created. `third_party/sam3`, N36--N72R11 historical evidence, the frozen
checkpoint, candidate schedule and evaluation definitions were not modified.

## Frozen scope and invariants

The recovery used the same N72R11 schedule and protocol:

- 527 event-policy keys across 18 sequences;
- 428 train and 99 validation keys;
- action counts: ADD_NEW_IDENTITY 58, ATOMIC_ID_SWAP 51,
  AUTHORITATIVE_REASSIGN 231, RECOVER_IDENTITY 187;
- event protocol SHA256:
  `367ca05823fec21ad2141bd1f731504c96c36d2cbc76456d215933c4ac828aa6`;
- secondary schedule SHA256:
  `7cfbef5d65498e5c079e3a4178ca3dce9dd20ff854a3ef1fb69ccb0580a7eac3`;
- runtime future-GT use remained `false`;
- no horizon shortening, frame removal, precision change, checkpoint change,
  candidate replacement, solver change, metric change or threshold change.

The known smoke key was kept exactly as specified:
`n72r5-pool-n37-dancetrack0001-0296-authoritative_reassign-001:secondary:060`.

## N72R11 baseline retained before R1

The previous N72R11 evidence was not overwritten:

| Evidence | PASS | Failure | Interpretation |
|---|---:|---:|---|
| Formal attempt 02 | 457 | 70 | 37 OOM plus 33 invalid-box `ValueError` |
| Fresh retry attempt 03 | 33 | 37 | all remaining failures OOM; the 33 invalid-box cases were repaired |
| N72R11 merged baseline | 490 | 37 | 527 unique keys, incomplete |

The original failure artifacts and the previous partial merge remain under
`outputs/N72R11/`. R1 derived its retry set from those manifests; no event ID
was manually copied or dropped.

## R1 implementation

The following implementation changes were made on the isolated R1 branch:

| File | Change |
|---|---|
| `scripts/n72r11_run_secondary_interaction_batch.py` | Added live `nvidia-smi` GPU snapshots, optional compute-app probing, idle-first/free-memory scheduling, physical-GPU telemetry, and a maximum of four concurrent children. |
| `scripts/n72r11_generate_secondary_interaction.py` | Passed the real backend keyword `official_batched_grounding_batch_size=1`; split SAM3 geometry propagation from post-session OSNet materialization; recorded phase memory telemetry. |
| `sam3_intermot/reacquisition/frozen_feature_materializer.py` | Added a short-lived frozen 512-D OSNet materializer that batches boxes per frame and releases the encoder after each materialization call. |
| `sam3_intermot/reacquisition/live_requery_controller.py` | Releases completed SAM3 requery sessions before post-session feature materialization and retains only observation/audit data. |
| `scripts/n72r11r1_build_oom_retry_manifest.py` | Reconstructs the exact 37-key retry manifest from immutable N72R11 evidence. |
| `scripts/n72r11r1_build_remaining_oom_manifest.py` | Derives the 36 non-smoke IDs from the required smoke artifact. |
| `scripts/n72r11r1_build_retry_manifest.py` | Seals smoke plus dynamic-retry results and parses child failure evidence. |
| `scripts/n72r11r1_audit_retry_results.py` | Performs key, lifecycle, telemetry, runtime-GT and physical-GPU overlap auditing. |
| `scripts/n72r11r1_write_terminal_status.py` | Writes the machine-readable terminal controller and final-gate artifacts. |

The production formula, candidate order, Hungarian solver and official SAM3
code were not changed. The backend's actual runtime policy reported:

- official batched grounding: enabled;
- `official_batched_grounding_batch_size`: `1`;
- video CPU offload: enabled;
- output CPU offload: enabled;
- official state CPU offload: `false` and not claimed as available;
- allocator: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

The post-session controller audit flag was also made immutable at construction
time. This matters because `close()` releases the callable while the audit
must still state that post-session materialization was used.

## Required full-window smoke

The required smoke ran once, with no `--horizon` override, in an independent
process selected by the dynamic scheduler:

- key: `n72r5-pool-n37-dancetrack0001-0296-authoritative_reassign-001:secondary:060`;
- event frame: 296;
- secondary frame: 356;
- frozen end frame: 396;
- sealed frame count: 41;
- result: `PASS_N72R11_SECONDARY_INTERACTION`;
- runtime future GT: `false`.

The smoke memory telemetry was:

| Phase | Allocated MiB | Reserved MiB | Max allocated MiB | Max reserved MiB |
|---|---:|---:|---:|---:|
| after target SAM | 5,583 | 6,616 | 6,349 | 6,616 |
| after target feature materialization | 1,261 | 1,468 | 6,349 | 6,616 |
| during live SAM peak | 31,284 | 33,806 | 36,109 | 36,440 |

The smoke's pre-close audit correctly recorded
`post_session_feature_materializer=true` and
`feature_materialization_phase=after_sam3_session_release`, with event-frame
memory read disabled and first visibility at frame 357. The already sealed
smoke JSON was not overwritten. Its post-close summary had a stale legacy
representation because the immutable audit flag was added after that smoke;
the targeted toy controller regression then passed with the corrected
post-close representation. The full smoke was not rerun solely for this
provenance-field repair, as required. All 25 PASS artifacts from the later
dynamic retry contain the corrected post-close audit fields.

## Dynamic retry result

`outputs/N72R11R1/oom_retry_manifest.json` contained exactly 37 unique OOM
keys. The required smoke passed, so
`outputs/N72R11R1/remaining_oom_event_ids.json` deterministically selected the
other 36 keys. Attempt 05 ran only those 36 keys.

The scheduler audit reported:

| Metric | Value |
|---|---:|
| Selected retry keys | 36 |
| PASS | 25 |
| `OutOfMemoryError` failures | 11 |
| Discovered physical GPUs | 0--9 |
| Maximum concurrent children | 4 |
| Idle-GPU launches | 36 |
| Shared/external-compute launches | 0 |
| GPU snapshot probe failures | 0 |
| Same physical GPU interval overlaps | 0 |
| Runtime future GT | `false` |

All 11 failures have independent failure JSON artifacts and full child logs.
Their failure type is `OutOfMemoryError` in every case. The failed keys are:

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

The child exceptions were observed after a clean, high-free idle launch. The
requested allocation sizes were 2--82 MiB; reported free memory at failure
was 1.12--73.12 MiB, with 37.26--37.49 GiB already allocated on a 39.39 GiB
device. This is consistent with official SAM3 future-propagation memory
exhaustion, not external GPU contention or a validator classification error.

## Strict merge and completeness gate

The existing `n72r11_merge_secondary_batch_manifests.py` was used with the
previous 490-PASS merged manifest and the sealed R1 retry manifest. It emitted
the following immutable result:

| Check | Result |
|---|---:|
| Required event-policy keys | 527 |
| Merged records | 527 |
| PASS | 516 |
| `FAIL_CHILD` | 11 |
| Duplicate keys | 0 |
| Missing keys | 0 |
| Unavailable/failed keys | 11 |
| `PASS_ALL_SELECTED` | `false` |

The merge status is `PARTIAL_WITH_FAILURES`. Failed evidence was retained;
partial rows were not promoted to PASS. Since the exact 527/527 PASS gate did
not pass, no incomplete rows were sent to corpus construction or training.

## Downstream scientific stages

The following stages are intentionally `NOT_RUN_UPSTREAM_BLOCKED`:

| Stage | Status | Reason |
|---|---|---|
| Causal corpus | `NOT_RUN_UPSTREAM_BLOCKED` | Requires 527/527 complete secondary rows |
| Temporal V3 training/validation | `NOT_RUN_UPSTREAM_BLOCKED` | No complete causal corpus |
| TargetEdgeBridge training/validation | `NOT_RUN_UPSTREAM_BLOCKED` | No complete causal corpus |
| Formal E0/E1/E2 replay | `NOT_RUN_UPSTREAM_BLOCKED` | No complete corpus and no authorization |
| Paired scoring/protected-ID analysis | `NOT_RUN_UPSTREAM_BLOCKED` | No formal replay |
| Oracle/selector/production promotion | `NOT_RUN_UPSTREAM_BLOCKED` | Completeness gate failed |

Consequently, this report contains no fabricated corpus size, future-positive
distribution, V3 accuracy, bridge loss, E1/E2 effect, protected regression or
solver-refusal number for R1. The corresponding controller fields are `null`,
which means “not run,” not zero effect.

## Preserved failure and audit facts

One post-run read-only diagnostic summary command initially failed because it
attempted to JSON-encode a tuple-keyed composite counter:

```text
TypeError: keys must be str, int, float, bool or None, not tuple
```

The failure was retained in the machine audit metadata, corrected by
stringifying that composite key, and the identical read-only audit then
passed. It did not modify scientific artifacts.

The earlier R1 smoke audit representation issue was handled similarly: the
sealed smoke was retained, the code was minimally repaired, and a targeted
toy controller regression passed. No historical N36--N72R11 output was
rewritten, and no failure log was deleted.

## Integrity and reproducibility hashes

The governance file was read from
`/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/AGENTS.md`;
its SHA256 is
`35c67f21b0a86d0a3abd2e9899115d742bd812c714c4485eefb95a781adb4bce`.

Important frozen and R1 artifact hashes:

| Artifact | SHA256 |
|---|---|
| `outputs/N72R11/event_protocol.json` | `367ca05823fec21ad2141bd1f731504c96c36d2cbc76456d215933c4ac828aa6` |
| `outputs/N72R11/secondary_event_manifest.json` | `7cfbef5d65498e5c079e3a4178ca3dce9dd20ff854a3ef1fb69ccb0580a7eac3` |
| `outputs/N72R11R1/oom_retry_manifest.json` | `9da906d9743a1e7c3c86e6191e96295e955d6495aab4a8d826d57bf20d3bd047` |
| `outputs/N72R11R1/remaining_oom_event_ids.json` | `82fb210f70f121cad9f55bab7c3cd41d60d3dbdc78ce8a63e5a4a3d29ee0b340` |
| `outputs/N72R11R1/secondary_retry_manifest_attempt_05.json` | `1e245e5308cc022ea87ca7f656d10abd47b590ea6396bc98d936c569a5c99884` |
| `outputs/N72R11R1/secondary_batch_merged_attempt_05.json` | `84a285e42fc0dcf57e33a3db2e0e5afe41863428ce99e426f574ed97a9b92822` |
| `outputs/N72R11R1/retry_output_audit_attempt_01.json` | `f7cb7b9afd7e80bf6effd1300cbf5f7225ba353740dcc5242dd629cd50ec7dcd` |
| `outputs/N72R11R1/n72r11r1_final_gate.json` | `ffe4bedadcf900143fc471e7b271a4776204462a1dff51599bd0782d6a5ab3c2` |

Modified-code hashes at report creation:

| File | SHA256 |
|---|---|
| `sam3_intermot/reacquisition/live_requery_controller.py` | `971e7f6d9c0d6c4b49dde77dad96de481f2a0b87769c4bb2a62f1943bff0758c` |
| `sam3_intermot/reacquisition/frozen_feature_materializer.py` | `e787f876bcb243cfda99cd85fa7679dc0aeb19e1046c78011c2b4f1a3677f976` |
| `scripts/n72r11_generate_secondary_interaction.py` | `feb4f0f2dd74cd08d35d6dcea87c7ffe8db447e50233407542d1b983b6359837` |
| `scripts/n72r11_run_secondary_interaction_batch.py` | `b12506968b4a8643c0ace01cd1f776ddb745c5f0bf7de4ee8cdac85137af6bd3` |
| `scripts/n72r11r1_build_oom_retry_manifest.py` | `32ce897398b9a5825bf6b9c97e24f3a303f4e2d394a0d9d6dbaa2335d780a44b` |
| `scripts/n72r11r1_build_remaining_oom_manifest.py` | `59aab35f25faacc1b469ec93b0e29dfcc3754a2279ba4d153342d8371dea1d89` |
| `scripts/n72r11r1_build_retry_manifest.py` | `b383e3d4c73c62444f0ad08b041be68a6329766616aa026f078520b523e493c6` |
| `scripts/n72r11r1_audit_retry_results.py` | `04b30873e6b0983de5f165fe2d414ea3fa600b08322147d75b523e1e8226eec0` |
| `scripts/n72r11r1_write_terminal_status.py` | `74d12858c780120beffbcafcd479437441c663d95df8dfdbfcddee59ca330e9a` |

Validation completed with the InterMOT environment:

- targeted post-session controller regression: PASS;
- required Python compilation: PASS;
- minimal import/compile and `git diff --check`: PASS;
- dynamic scheduler audit: PASS for telemetry/key/lifecycle invariants;
- complete secondary gate: FAIL because 11 official SAM3 OOM failures remain.

## Terminal root cause and lawful next step

The allowed R1 memory engineering path has been exercised:

1. independent event processes and fresh backend/session ownership;
2. serialized SAM3 geometry/propagation followed by released-session OSNet
   materialization;
3. official grounding batch size 1;
4. `expandable_segments:True` allocator configuration;
5. live dynamic scheduling over all physical GPUs, with idle-first selection;
6. full-window high-free-GPU retry of every non-smoke OOM key.

The remaining failures are therefore terminal for this environment under the
frozen protocol: `INTRINSIC_OOM_AFTER_SERIALIZED_FEATURE_PIPELINE`.

The only defensible future recovery is to provide a genuinely larger or
less-contended execution environment, or an already-supported official SAM3
state-memory offload mechanism, and rerun only the 11 preserved keys. The
future run must retain the same checkpoint, candidate stream, full window,
metrics and runtime-GT boundary. Shortening the window, changing precision,
changing the solver, replacing the event set or treating partial artifacts as
PASS is not permitted.

## ICLR 2027 timing interpretation

Using the frozen project schedule, the ICLR 2027 abstract deadline is
**2026-09-18 AoE** and the full-paper deadline is **2026-09-25 AoE**. On
2026-09-06, these are approximately 12 and 19 calendar days away. N72R11R1
does not justify spending that remaining window on blind training: the next
research-valid action is resource recovery, followed by the existing strict
527-key gate. Until that gate passes, there is no scientifically valid R1
corpus or downstream effect claim.

## Machine-readable evidence map

- Controller status: `outputs/N72R11R1/CONTROLLER_STATUS.json`
- Final gate: `outputs/N72R11R1/n72r11r1_final_gate.json`
- Retry audit: `outputs/N72R11R1/retry_output_audit_attempt_01.json`
- Frozen 37-key retry manifest: `outputs/N72R11R1/oom_retry_manifest.json`
- Remaining 36-key manifest: `outputs/N72R11R1/remaining_oom_event_ids.json`
- Sealed retry manifest: `outputs/N72R11R1/secondary_retry_manifest_attempt_05.json`
- Strict merged manifest: `outputs/N72R11R1/secondary_batch_merged_attempt_05.json`
- Dynamic scheduler manifest and child logs: `outputs/N72R11R1/secondary_oom_retry_attempt_05/`
- Required smoke: `outputs/N72R11R1/secondary_oom_smoke_attempt_04/`
- Human-readable terminal status: `outputs/N72R11R1/HUMAN_READABLE_STATUS.md`
