Does the learned identity memory transfer from offline GT replay to a real SAM3 candidate stream after only one human initialization?

# InterMOT N72R20 Final Report

This report is governed by [`outputs/N72R20/FINAL_GOAL.json`](/data3/liuyeqiang/InterMOT/outputs/N72R20/FINAL_GOAL.json). The Final Goal and central question are frozen; this document is an in-progress report until the real SAM3 candidate-stream probe is run.

## Current status

`DATASET_READY_SAM3_CHECKPOINT_MISSING`

No final scientific decision has been issued. No SAM3 inference, candidate cache, val evaluation, MOT evaluation, training, LoRA, selector training, or full association work has been started in N72R20.

The executable bridge is prepared but not run: the worker generates a GT-free per-sequence SAM3 candidate cache, and the target-centric evaluator replays every post-anchor frame causally. GT is read only afterward to label target coverage, visible-identity hard negatives, rank, wrong writes, and recovery. The aggregate command keeps candidate coverage separate from identity discrimination and cannot authorize the next association stage unless the frozen N72R20 decision is `PASS_REAL_CANDIDATE_IDENTITY_MEMORY`.

## Frozen goal

- Goal: Human-Initialized Identity Memory in Real Candidate Streams.
- Central question: Can a single human identity initialization, together with the frozen learned identity memory from N72R18, reliably recover and maintain the same identity from a real SAM3 candidate stream over future frames?
- Primary comparison: Frozen N72R18 GRU versus Human Anchor Only on identical real SAM3 candidates.
- Horizons: H20, H50, H100.
- Interaction source: `simulated_from_gt`; this is not real-human evidence.
- Next interactive MOT stage: not authorized.
- Memory variants: B0 human anchor only, B1 EMA(0.90), frozen N72R18 GRU B2, and explicitly labeled `ORACLE_CORRECT_UPDATE_OFFLINE_ORACLE_DIAGNOSTIC`.
- Update diagnostic: immediate machine update versus 2-frame confirmation. The confirmation margin must be frozen from train records before val; no val threshold tuning is allowed.

## Assets resolved

- `DANCETRACK_ROOT=/data3/liuyeqiang/InterMOT_N72R16_assets/dataset`.
- DanceTrack train: 40/40 complete sequences.
- DanceTrack val: 25/25 complete sequences.
- All frozen N72R17 protocol sequence names are present; image and GT audits passed with no frame-count mismatch.
- Lineage: `N72R16_NEW_ASSET_LINEAGE`.
- Frozen N72R18 GRU and frozen N72R16 OSNet checkpoint are present and SHA-recorded in `outputs/N72R20/asset_manifest.json`.
- DanceTrack was not downloaded or copied, no proxy was used, and DanceTrack test was not downloaded.
- The current checkpoint/cache recheck is recorded in [`outputs/N72R20/checkpoint_access_audit.json`](/data3/liuyeqiang/InterMOT/outputs/N72R20/checkpoint_access_audit.json): no local SAM3 checkpoint or candidate cache was found, and the NAS path is not locally visible.

## Current blocker

`sam3.1_multiplex.pt` is not present on the local search roots, the historical NAS path is not mounted, and the local Hugging Face client reports `Not logged in`. The official SAM3 repository requires authenticated checkpoint access. Therefore the required two-sequence smoke test cannot honestly be run yet.

The missing threshold manifest is a downstream dependency, not an independent blocker: it will be produced from train-only B2 records by `scripts/n72r20_freeze_confirmation_threshold.py` after real candidate observations exist. It cannot be fabricated before that evidence.

## Software verification

- Targeted N72R18/R1/backend checks: `16 passed`.
- N72R20 bridge and aggregation tests: `8 passed`.
- Full suite with the project `.venv` on PATH: `233 passed, 4 failed`.
- All four failures reach the same fixed TrackEval `12c8791` interface defect: the CLI supplies `SEQMAP_FILE` as a one-element list while `mot_challenge_2d_box.py` calls `os.path.isfile` on it. This is a third-party/legacy test infrastructure issue, not an N72R20 dataset or research-code failure. The prohibited TrackEval/full-MOT path was not modified.

## Software prepared but not executed

- `scripts/n72r20_candidate_stream_smoke.py`: one process/one sequence; train smoke is limited to 100–200 frames, while val requires explicit `--frozen-eval` and full-sequence mode.
- `scripts/n72r20_identity_bridge_eval.py`: frozen OSNet anchor, B0/B1/B2/oracle, causal all-frame replay, GT-posthoc target/competitor labels, and immediate/2-frame policies.
- `scripts/n72r20_aggregate_identity_bridge.py`: H20/H50/H100, rank-1/2/3, MRR, margins, sequence-cluster bootstrap, paired B2-vs-B0 delta, and the four N72R20 decision labels.
- `scripts/n72r20_run_frozen_val_candidates.py`, `scripts/n72r20_run_streaming_val_bridge.py`, and `scripts/n72r20_run_identity_bridge_eval.py`: explicit gates for storage, full val/streaming val, and frozen threshold provenance.

## Required next action

After authenticated official checkpoint access is supplied, configure its path through the N72R20 asset manifest or CLI, record size and SHA256, run only the two-train-sequence smoke test, and stop if candidate coverage or storage gates fail. Then run train-only bridge development, freeze the confirmation threshold, and only then consider the explicit frozen-val command. Do not download any additional dataset or start full val before those gates.
