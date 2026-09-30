FINAL GOAL: **Human Correction Driven Persistent Identity Adaptation**

CENTRAL QUESTION: **When a human corrects a tracking error, can the system learn from this correction and improve future identity tracking?**

# InterMOT N72R20 Final Report

This report is governed by [`outputs/N72R20/FINAL_GOAL.json`](/data3/liuyeqiang/InterMOT/outputs/N72R20/FINAL_GOAL.json). The revised Final Goal and central question are frozen; this report records construction of the train-only interactive correction environment.

The requirement-by-requirement completion audit is recorded in [`docs/N72R20_COMPLETION_AUDIT.md`](/data3/liuyeqiang/InterMOT/docs/N72R20_COMPLETION_AUDIT.md). It records the train-only correction environment; no learned correction-benefit decision is issued.

## Current status

`PASS_TRAIN_CORRECTION_EVENTS`

No downstream scientific decision has been issued. The checkpoint was SHA256-verified, the bounded train-only candidate smoke completed for two sequences, and 1,043 simulated correction events were generated from 1,497 train correction opportunities. A val run was started under the previous direction but was stopped and quarantined after the revised Goal arrived; it is not evidence and no val result is reported. No MOT evaluation, training, LoRA, selector training, or full association work has been started in N72R20.

The train smoke worker generated GT-free per-sequence SAM3 candidate caches for `dancetrack0001` and `dancetrack0002`. Historical B2 immediate records were produced from those caches and then consumed only for offline correction-event discovery; no val aggregate or current-stage learned correction evaluation was run.

## Frozen goal

- Goal: Human Correction Driven Persistent Identity Adaptation.
- Central question: When a human corrects a tracking error, can the system learn from this correction and improve future identity tracking?
- Ultimate application: Interactive Multi-Object Tracking.
- N72R20 role: Interactive Correction Environment Construction.
- Runtime boundary: real SAM3 candidates; GT is used only offline to discover errors and simulate correction events.
- Current output: correction event JSONL and correction opportunity/error statistics on a small train-only smoke.
- Event types: identity switch, missed target, and wrong recovery.
- Next stage: N72R21 Learned Correction Memory Update.
- Training, MOT/TrackEval, val evaluation, and downstream association remain out of scope.

Historical context is recorded in [`outputs/N72R20/history_context.md`](/data3/liuyeqiang/InterMOT/outputs/N72R20/history_context.md). The module boundary and reuse decisions are recorded in [`outputs/N72R20/module_reuse_audit.md`](/data3/liuyeqiang/InterMOT/outputs/N72R20/module_reuse_audit.md).

## Assets resolved

- `DANCETRACK_ROOT=/data3/liuyeqiang/InterMOT_N72R16_assets/dataset`.
- DanceTrack train: 40/40 complete sequences.
- DanceTrack val: 25/25 complete sequences.
- All frozen N72R17 protocol sequence names are present; image and GT audits passed with no frame-count mismatch.
- Lineage: `N72R16_NEW_ASSET_LINEAGE`.
- Frozen N72R18 GRU, frozen N72R16 OSNet checkpoint, and `sam3.1_multiplex.pt` are present and SHA-recorded in `outputs/N72R20/asset_manifest.json`.
- DanceTrack was not downloaded or copied, no proxy was used, and DanceTrack test was not downloaded.
- The official endpoint was gated; a public Hugging Face mirror was used after matching the published official SHA256. The source and verification are recorded in [`outputs/N72R20/checkpoint_access_audit.json`](/data3/liuyeqiang/InterMOT/outputs/N72R20/checkpoint_access_audit.json).
- The train-only smoke output is recorded in [`outputs/N72R20/smoke_report.md`](/data3/liuyeqiang/InterMOT/outputs/N72R20/smoke_report.md). No completed val cache was retained; the interrupted partial cache was quarantined outside the candidate root.

## Current boundary

The requested checkpoint recovery, two-sequence train smoke, and correction-event construction are complete. The previous frozen-val attempt was stopped after the Goal correction and its partial cache was quarantined. N72R20 stops at train error discovery and simulated correction-event construction; learning from corrections is reserved for N72R21.

The train-derived confirmation threshold exists as a historical diagnostic at
`outputs/N72R20/confirmation_threshold.json`; it is not used as a current
N72R20 correction-learning endpoint.

## Software verification

- Targeted N72R18/R1/backend checks: `16 passed`.
- N72R20 bridge, aggregation, and candidate-connector tests: `13 passed`.
- Full suite with the project `.venv` on PATH: `241 passed, 4 failed` (the four
  failures are unchanged third-party TrackEval interface failures; the three
  new correction-event tests pass).
- All four failures reach the same fixed TrackEval `12c8791` interface defect: the CLI supplies `SEQMAP_FILE` as a one-element list while `mot_challenge_2d_box.py` calls `os.path.isfile` on it. This is a third-party/legacy test infrastructure issue, not an N72R20 dataset or research-code failure. The prohibited TrackEval/full-MOT path was not modified.

## Software prepared and current-stage work

- `scripts/n72r20_candidate_stream_smoke.py`: one process/one sequence; the two train smoke sequences completed at 160 written frames each.
- `scripts/n72r20_identity_bridge_eval.py`: frozen OSNet anchor, B0/B1/B2/oracle, causal all-frame replay, GT-posthoc target/competitor labels, and immediate/2-frame policies.
- `scripts/n72r20_aggregate_identity_bridge.py` and the val runners are preserved as historical identity-probe infrastructure, not the current N72R20 endpoint.
- `sam3_intermot/interaction/correction_event.py`: frozen correction-event schema.
- `sam3_intermot/identity_memory/correction_update.py`: record-only future update interface; it does not change memory.
- `scripts/n72r20_generate_correction_events.py`: offline GT error discovery and simulated correction-event generation from the train smoke.

## Required next action

The event generator has run on the existing `dancetrack0001` and `dancetrack0002` train caches. Validate the committed schema and statistics, then stop. N72R21 may later study how to learn from these events.
