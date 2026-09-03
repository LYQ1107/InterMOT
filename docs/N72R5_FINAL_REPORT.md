# N72R5 Final Report

- Generated: `2026-09-03T18:43:44.433415+00:00`
- Machine gate: `N72R5_STRUCTURAL_FULL_LOOP_PASS_EFFICACY_BLOCKED_NO_PUBLIC_MAPPING`
- Research gate: `BLOCKED_EXACT_PUBLIC_MAPPING_AND_REAL_HUMAN_TAPE`
- Runtime future GT: `False`
- Interaction source: `simulated_from_gt` (not real-human evidence)

## Executive conclusion

N72R5 completed the official SAM3 candidate-stream full-loop structurally, but it did not produce a scientific future-effect result. The 40-event policy is explicitly `simulated_from_gt`, and the official branch artifacts retain `public_id=null` with `NOT_ASSIGNED_IN_OFFICIAL_BRANCH`. Therefore exact public-ID association, posthoc future-effect scoring, production promotion, calibration, selector training, and decoder LoRA remain unauthorized.

This is a provenance/authority block, not a claim that the mechanism works or fails. No public ID was inferred from a native ID, candidate index, or dataset GT ID.

## Frozen scope and event policy

- Events: `40`; independent sequences: `20`.
- Action counts: `{"ADD_NEW_IDENTITY": 5, "ATOMIC_ID_SWAP": 8, "AUTHORITATIVE_REASSIGN": 14, "RECOVER_IDENTITY": 13}`.
- Source split: frozen train/train_fold candidate tape; no val/test was introduced.
- All events remain `simulated_from_gt`; they must not be described as historical human clicks.
- Stage 01--04 mechanism findings were retained: candidate absence and candidate-present decision errors are distinct; image recovery had no recall gain; TVC had no correct crossing; feature separability was not informative.

## Stage 07 structural result

| Check | Result |
|---|---:|
| Events | `40/40` |
| Official branches | `200/200` |
| Unique worker keys | `200` |
| Duplicate worker keys | `0` |
| Missing worker keys | `0` |
| Frame rows | `20200/20200` |
| Candidate rows | `146176` |
| Native/adapter mapping fields complete | `146176` |
| Runtime future GT | `False` |
| Structural integrity | `True` |

Each event has the five preregistered branches `B0`--`B4`, 101 frame rows from the event frame through event+100, a shared frozen pre-state hash, and event-frame memory-read suppression. The Stage07 CPU audit is independent of the worker execution and reports no structural error.

## Exact association and future-effect gate

- Public-ID assigned candidate rows: `0`.
- Public-ID unassigned candidate rows: `146176`.
- Exact public association evaluated: `False`.
- Posthoc future effect evaluated: `False`.
- Future identity effect: `None`.

Because the authoritative public-ID axis is absent, there is no valid identity-error, ID-switch, re-correction, or H20/H50/H100 public-ID effect number to report. Filling the missing axis from raw SAM IDs, candidate indices, dataset GT IDs, or a future heuristic would violate the frozen protocol, so the finalizer intentionally leaves the effect null.

## Failure and repair provenance

- Stage07 attempts 1--4 remain preserved as blocked/partial attempts; their status files and failure artifacts were not deleted or rewritten.
- Attempt5 completed after the already-recorded memory/observation engineering repairs: `200/200` branches, `0` new failure artifacts.
- The Stage03/Stage04 negative mechanism gates remain scientific negative findings, not converted into PASS efficacy evidence.
- N36--N72R4 historical outputs and all earlier failure evidence remain read-only inputs.

## Authorization decision

`production_authorized=false`, `training_authorized=false`, `calibration_authorized=false`, `selector_authorized=false`, and `decoder_lora_authorized=false`. The structural full-loop pass cannot authorize downstream learning because exact public authority and real-human evidence are still missing.

## Reproducibility artifacts

- [Machine-readable final gate](/data2/usr_for_deadline/SAM3_InterMOT_N72R5/worktree/outputs/N72R5/n72r5_final_gate.json)
- [Stage07 CPU audit](/data2/usr_for_deadline/SAM3_InterMOT_N72R5/worktree/outputs/N72R5/audits/stage07_attempt5_cpu_audit.json)
- [Official full-loop manifest](/data2/usr_for_deadline/SAM3_InterMOT_N72R5/worktree/outputs/N72R5/mechanism_rounds/round_07_official_full_loop_attempt5/official_full_loop_manifest.json)
- [Frozen event manifest](/data2/usr_for_deadline/SAM3_InterMOT_N72R5/worktree/outputs/N72R5/mechanism_rounds/round_06_event_policy/real_event_manifest.json)
- [Pinned regression result](/data2/usr_for_deadline/SAM3_InterMOT_N72R5/worktree/outputs/N72R5/tests/n72r5_regression_result.json)
- [Preserved N72R5 attempts](/data2/usr_for_deadline/SAM3_InterMOT_N72R5/worktree/outputs/N72R5/attempts)

Input SHA-256 values are recorded in `n72r5_final_gate.json`. The pinned regression completed `193 passed, 0 failed` with three existing interpreter warnings; the isolated worktree used the populated sibling TrackEval checkout only as a read-only test path. The final gate also records the source Git HEAD observed at finalization.

## Minimum next action

Collect provenance-complete real-human event JSONL and raw input files, including direct public IDs and a same-run authoritative resolver/mapping that survives session boundaries. Validate the mapping before running the unchanged exact public association and future-effect protocol. Do not relabel `simulated_from_gt` events as real human data and do not start calibration, selector, or decoder LoRA before that gate.
