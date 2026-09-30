Does the learned identity memory transfer from offline GT replay to a real SAM3 candidate stream after only one human initialization?

# InterMOT N72R20 Final Report

This report is governed by [`outputs/N72R20/FINAL_GOAL.json`](/data3/liuyeqiang/InterMOT/outputs/N72R20/FINAL_GOAL.json). The Final Goal and central question are frozen; this document is an in-progress handoff report until the real SAM3 candidate-stream probe is run.

## Current status

`DATASET_READY_SAM3_CHECKPOINT_MISSING`

No final scientific decision has been issued. No SAM3 inference, candidate cache, val evaluation, MOT evaluation, training, LoRA, selector training, or full association work has been started in N72R20.

## Frozen goal

- Goal: Human-Initialized Identity Memory in Real Candidate Streams.
- Central question: Can a single human identity initialization, together with the frozen learned identity memory from N72R18, reliably recover and maintain the same identity from a real SAM3 candidate stream over future frames?
- Primary comparison: Frozen N72R18 GRU versus Human Anchor Only on identical real SAM3 candidates.
- Horizons: H20, H50, H100.
- Interaction source: `simulated_from_gt`; this is not real-human evidence.
- Next interactive MOT stage: not authorized.

## Assets resolved

- `DANCETRACK_ROOT=/data3/liuyeqiang/InterMOT_N72R16_assets/dataset`.
- DanceTrack train: 40/40 complete sequences.
- DanceTrack val: 25/25 complete sequences.
- All frozen N72R17 protocol sequence names are present; image and GT audits passed with no frame-count mismatch.
- Lineage: `N72R16_NEW_ASSET_LINEAGE`.
- Frozen N72R18 GRU and frozen N72R16 OSNet checkpoint are present and SHA-recorded in `outputs/N72R20/asset_manifest.json`.
- DanceTrack was not downloaded or copied, no proxy was used, and DanceTrack test was not downloaded.

## Current blocker

`sam3.1_multiplex.pt` is not present on the local search roots, the historical NAS path is not mounted, and the local Hugging Face client reports `Not logged in`. The official SAM3 repository requires authenticated checkpoint access. Therefore the required two-sequence smoke test cannot honestly be run yet.

## Software verification

- Targeted N72R18/R1/backend checks: `16 passed`.
- Full suite with the project `.venv` on PATH: `225 passed, 4 failed`.
- All four failures reach the same fixed TrackEval `12c8791` interface defect: the CLI supplies `SEQMAP_FILE` as a one-element list while `mot_challenge_2d_box.py` calls `os.path.isfile` on it. This is a third-party/legacy test infrastructure issue, not an N72R20 dataset or research-code failure. The prohibited TrackEval/full-MOT path was not modified.

## Required next action

After authenticated official checkpoint access is supplied, configure its path through the N72R20 asset manifest or CLI, record size and SHA256, run only the two-train-sequence smoke test, and stop if candidate coverage or storage gates fail. Do not download any additional dataset or start full val before that gate.
