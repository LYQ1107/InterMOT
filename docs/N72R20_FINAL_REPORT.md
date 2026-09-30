FINAL GOAL: **Human-Initialized Identity Memory in Real Candidate Streams**

CENTRAL QUESTION: **Can one human initialization be recognized reliably in future real SAM3 candidates?**

# InterMOT N72R20 Final Report

This report is governed by [`outputs/N72R20/FINAL_GOAL.json`](/data3/liuyeqiang/InterMOT/outputs/N72R20/FINAL_GOAL.json). The Final Goal and central question are frozen; this document is an in-progress report until the real SAM3 candidate-stream validation is run.

The requirement-by-requirement completion audit is recorded in [`docs/N72R20_COMPLETION_AUDIT.md`](/data3/liuyeqiang/InterMOT/docs/N72R20_COMPLETION_AUDIT.md). It leaves the scientific decision unissued because this task stops after the bounded train smoke and does not authorize frozen val.

## Current status

`BLOCKED_VAL_NOT_AUTHORIZED`

No final scientific decision has been issued. The checkpoint was SHA256-verified and the bounded train-only candidate smoke completed for two sequences. No val evaluation, MOT evaluation, training, LoRA, selector training, or full association work has been started in N72R20.

The train smoke worker generated GT-free per-sequence SAM3 candidate caches for `dancetrack0001` and `dancetrack0002`. The target-centric evaluator and aggregate command were not run; candidate coverage, identity rank, memory contamination, and the frozen endpoint remain unmeasured.

## Frozen goal

- Goal: Human-Initialized Identity Memory in Real Candidate Streams.
- Central question: Can one human initialization be recognized reliably in future real SAM3 candidates?
- Formal question: Can a single human identity initialization, together with the frozen learned identity memory from N72R18, reliably recover and maintain the same identity from a real SAM3 candidate stream over future frames?
- Primary comparison: Frozen N72R18 GRU versus Human Anchor Only on identical real SAM3 candidates.
- Horizons: H20, H50, H100.
- Interaction source: `simulated_from_gt`; this is not real-human evidence.
- Next interactive MOT stage: not authorized.
- Memory variants: B0 human anchor only, B1 EMA(0.90), frozen N72R18 GRU B2, and explicitly labeled `ORACLE_CORRECT_UPDATE_OFFLINE_ORACLE_DIAGNOSTIC`.
- Update diagnostic: immediate machine update versus 2-frame confirmation. The confirmation margin must be frozen from train records before val; no val threshold tuning is allowed.

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
- The train-only smoke output is recorded in [`outputs/N72R20/smoke_report.md`](/data3/liuyeqiang/InterMOT/outputs/N72R20/smoke_report.md). No val cache was created.

## Current blocker

The requested checkpoint recovery and two-sequence train smoke are complete. N72R20 remains blocked because this task explicitly stops before frozen val; no scientific identity decision can be issued from a train smoke.

The missing threshold manifest is a downstream dependency, not an independent blocker: it will be produced from train-only B2 records by `scripts/n72r20_freeze_confirmation_threshold.py` after real candidate observations exist. It cannot be fabricated before that evidence.

## Software verification

- Targeted N72R18/R1/backend checks: `16 passed`.
- N72R20 bridge, aggregation, and candidate-connector tests: `13 passed`.
- Full suite with the project `.venv` on PATH: `238 passed, 4 failed`.
- All four failures reach the same fixed TrackEval `12c8791` interface defect: the CLI supplies `SEQMAP_FILE` as a one-element list while `mot_challenge_2d_box.py` calls `os.path.isfile` on it. This is a third-party/legacy test infrastructure issue, not an N72R20 dataset or research-code failure. The prohibited TrackEval/full-MOT path was not modified.

## Software prepared but not executed

- `scripts/n72r20_candidate_stream_smoke.py`: one process/one sequence; the two train smoke sequences completed at 160 written frames each.
- `scripts/n72r20_identity_bridge_eval.py`: frozen OSNet anchor, B0/B1/B2/oracle, causal all-frame replay, GT-posthoc target/competitor labels, and immediate/2-frame policies.
- `scripts/n72r20_aggregate_identity_bridge.py`: H20/H50/H100, rank-1/2/3, MRR, margins, sequence-cluster bootstrap, paired B2-vs-B0 delta, and the four N72R20 decision labels.
- `scripts/n72r20_run_frozen_val_candidates.py`, `scripts/n72r20_run_streaming_val_bridge.py`, and `scripts/n72r20_run_identity_bridge_eval.py`: explicit gates for storage, full val/streaming val, and frozen threshold provenance.

## Required next action

The next possible action requires explicit authorization for frozen DanceTrack val. Until then, do not run val, TrackEval, MOT, association, training, or any identity-memory modification.
