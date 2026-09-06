# N72R11 — On-Demand Closed-Loop Reacquisition + Assignment-Aware Target Edge Bridge

## Executive decision

N72R11 is **`BLOCKED_N72R11_SECONDARY_OOM`**, not a research or production
PASS. The on-demand controller, true uncertainty-frame wiring and isolated
target-edge bridge pass focused engineering checks. The required complete
secondary interaction corpus does not exist: 490 of 527 selected event-policy
executions passed, while 37 still fail in official SAM3 future propagation with
CUDA out-of-memory. Therefore the causal corpus, Temporal V3 training, bridge
training/validation, formal E0/E1/E2 replay and any production promotion were
not authorized.

All N72R11 interactions are `simulated_from_gt`; no real-human evidence was
created. Historical N36--N72R10 evidence, shared checkpoints, frozen candidate
definitions, metrics and `third_party/sam3` were not modified.

## Frozen scope and input integrity

N72R11 retained the N72R10 true future-frame contract: uncertainty is detected
from runtime state, a new causal SAM3 session queries the current frame, only a
selected source may propagate, raw/native binding may change while immutable
public ID remains stable, and the first memory effect is event+1. Runtime never
read future GT; GT remained confined to the simulated event construction and
posthoc-only paths.

The frozen secondary schedule contains 527 event instances across 18 sequences,
with 428 train and 99 validation instances. Action counts are ADD 58, ATOMIC
51, AUTHORITATIVE_REASSIGN 231 and RECOVER 187. The schedule hash is
`7cfbef5d65498e5c079e3a4178ca3dce9dd20ff854a3ef1fb69ccb0580a7eac3`; the event
protocol hash is `367ca05823fec21ad2141bd1f731504c96c36d2cbc76456d215933c4ac828aa6`.
The governance file hash is
`35c67f21b0a86d0a3abd2e9899115d742bd812c714c4485eefb95a781adb4bce`.

## Implementation and structural checks

The new controller is in
`sam3_intermot/reacquisition/live_requery_controller.py`. It probes one
current-frame session, commits at most one selected source, materializes the
observation rows and audit, then closes/releases that session before returning.
The target-edge bridge is isolated in
`sam3_intermot/association/target_edge_bridge.py`; no backbone, candidate
generator or Hungarian solver was changed. The V3 model definition is isolated
in `sam3_intermot/reacquisition/models/n72r11_temporal_v3.py`.

The focused toy smoke (`attempt_04`) passed the controller close/release,
causal boundary, finite tensor shape/loss and hard-negative contract. The prior
invocation/signature failures remain in `outputs/N72R11/attempts/` rather than
being overwritten. A short real runtime smoke also sealed E0/E1/E2 artifacts;
its horizon was shorter than formal H100 and its posthoc artifact explicitly
reports `NOT_SCIENTIFIC_SMOKE`.

The protected-regression audit reconstructed seven historical protected cases
and identified `TARGET_EDGE_OVER_STEALS_PROTECTED_INCUMBENT`. This was treated
as a safety failure, not hidden by changing thresholds or the evaluation
definition.

## Secondary batch and retained failures

The first formal batch used four isolated workers on GPUs 1--4 and ended as:

| Attempt | PASS | failures | interpretation |
|---|---:|---:|---|
| attempt 2 | 457 | 70 | 37 OOM, 33 finite-box validation failures |
| retry attempt 3 | 33 | 37 | all 33 finite-box failures repaired; every remaining failure was OOM |
| merged | 490 | 37 | 527 unique keys, zero duplicate keys, zero missing keys, incomplete |

The 33 non-OOM failures were repaired with the smallest semantics-preserving
finite-box validation change and passed on the same frozen event-policy keys.
The 37 OOM keys were retried in fresh processes on GPUs 5--8 with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. No frames, future window,
checkpoint, H20/H50/H100 definition, candidate stream or public-ID protocol
was changed. The merge manifest remains `PARTIAL_WITH_FAILURES`; it is not a
complete corpus.

The causal corpus builder correctly refused this manifest and preserved the
failure at
`outputs/N72R11/training_v3/attempts/corpus_failure_20260906T074809Z.json`:
`secondary batch is not complete PASS_ALL_SELECTED`.

## Final memory repair and actionable root cause

The final bounded repair released `prefix_snapshot`, `prefix_outputs` and other
correction-time references after the selected session's observations and
audit were materialized, called garbage collection and emptied the CUDA cache,
and retained only observation data needed by the controller. It did not clear
official continuation state before future propagation. The toy close/release
test passed.

The same official event-policy key
`n72r5-pool-n37-dancetrack0001-0296-authoritative_reassign-001:secondary:060`
was then rerun with its full frozen window. It still failed in official SAM3
segmentation during `propagate_if_selected`: allocation request 1.19 GiB,
only 1.05 GiB free on a 39.39 GiB device, with 36.41 GiB allocated. The full
traceback is retained in
`outputs/N72R11/secondary_oom_lifecycle_smoke_attempt_04/attempts/`.

The backend requests official video/output CPU offload. The pinned official
`init_state` path does not expose a verified state-CPU-offload mechanism, so
state offload was not claimed or enabled; earlier unsupported/mismatch paths
were not reused. This is a genuine resource failure in official future
propagation, not a metric or policy failure. Three bounded memory repair
families have been exhausted: isolated allocator-aware workers, fresh-GPU
retry, and post-materialization session release.

## Gate and downstream authorization

The machine gate is
`outputs/N72R11/n72r11_final_gate.json`. Its decisive counts are:

- required selected keys: 527;
- complete PASS keys: 490;
- retained child failures: 37, all OOM after retry;
- duplicate keys: 0;
- missing keys: 0;
- complete-batch gate: false;
- runtime future GT: false;
- real-human evidence: false.

Because completeness is false, stages 07--17 that require a complete causal
pool are explicitly `NOT_RUN_UPSTREAM_BLOCKED` or blocked in their individual
status files. No incomplete rows were used to train Temporal V3 or the bridge;
there is no valid N72R11 training checkpoint and no formal causal effect number.
This is intentional evidence preservation, not a skipped scientific result.

## External method audit

The public-method audit is in
[`docs/N72R11_METHOD_AUDIT.md`](N72R11_METHOD_AUDIT.md), with paper/project URLs,
pinned repository commits and dates. It covers MOTIP, TrackTrack, SENTRY, TCEI
and InteractTrack. These methods informed design comparison only; none was
silently imported and none substitutes for the frozen N72R11 paired replay.

## Isolation and verification

The implementation was checked with the InterMOT environment by importing the
association, persistent-runtime, target-edge and requery modules, followed by
full Python compilation of `sam3_intermot` and `scripts`. The repository
isolation check showed branch
`codex/n72r11-on-demand-edge-bridge`, base HEAD
`b949d9a36198fefc2ce22f369480fbcecd8be99c`, no changed path under
`third_party/sam3`, and no historical output modification. The old unsupported
Git option invocation is retained as an invocation failure; the compatible
branch query was used for the actual check.

## ICLR 2027 schedule and practical next step

The controlling dates are the abstract deadline **2026-09-18 AoE** and full
paper deadline **2026-09-25 AoE**. N72R11 does not justify spending that window
on blind additional training. The only defensible recovery is to obtain a
genuinely less-contended/larger-memory execution environment, or use an
already-supported official memory mechanism, and rerun only the 37 preserved
OOM event-policy keys with the exact frozen protocol. If those keys complete,
the complete-batch audit must pass before corpus construction or training is
considered. Shortening windows, changing precision/checkpoints, replacing
events, or treating partial rows as PASS remains prohibited.

## Reproducibility map

- Final machine gate: `outputs/N72R11/n72r11_final_gate.json`
- Controller status: `outputs/N72R11/CONTROLLER_STATUS.json`
- Human-readable status: `outputs/N72R11/HUMAN_READABLE_STATUS.md`
- Secondary schedule: `outputs/N72R11/secondary_event_manifest.json`
- Retry list: `outputs/N72R11/secondary_retry_manifest_attempt_03.json`
- Partial merge: `outputs/N72R11/secondary_batch_merged_attempt_03.json`
- Corpus gate failure: `outputs/N72R11/stage_06_status.json`
- Final lifecycle failure: `outputs/N72R11/secondary_oom_lifecycle_smoke_attempt_04/attempts/`
- Focused contract smoke: `outputs/N72R11/attempts/n72r11_focused_smoke_attempt_04.json`
- Method source record: `outputs/N72R11/method_audit.json`
