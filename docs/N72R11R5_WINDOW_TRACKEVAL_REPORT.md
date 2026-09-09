# N72R11R5 — Interaction-Window TrackEval Diagnostic

**Final status: `BLOCKED_INCOMPLETE_WINDOW_TRACKEVAL`**

## Conclusion

This post-hoc diagnostic did not alter SAM3, PCTIS, Hungarian association, public IDs, candidates, thresholds, runtime artifacts, or E2. It is blocked before a complete 32-event × 3-variant × 3-horizon TrackEval matrix can be evaluated.

The blocked condition is an input-integrity failure in the sealed N72R11R4 solver output, not a TrackEval score and not evidence for or against the research hypothesis. No incomplete aggregate, imputed box, or official DanceTrack benchmark score is reported.

## Scope and frozen inputs

- Evaluation scope: `INTERACTION_WINDOW_DIAGNOSTIC`.
- Each pseudo sequence is exactly `[event_frame+1, event_frame+H]`, remapped to frames `1..H`, for `H20/H50/H100`.
- Variants: `E0_BASELINE_B0`, `E1A_V3`, `E1B_PCTIS`.
- Expected matrix: 32 events, 18 independent original sequences, 96 pseudo windows, 288 per-event variant records.
- TrackEval values, if available in a future valid run, must come only from the pinned official checkout. `official_dancetrack_benchmark_score=false` because these are local interaction windows.
- All N72R11R4 events are `simulated_from_gt`; they are not historical real-human evidence.
- E2 was not evaluated and remains `NOT_MEASURED_INCOMPLETE_FORMAL_REPLAY`.

### Frozen input hashes

- `E1A_V3` metrics: `/data2/usr_for_deadline/SAM3_InterMOT_N72R5/worktree/outputs/N72R11R4/formal_e1a_metrics.json`; SHA-256 `96f3544ac0e66e03c3b9b437c6269ca852e4557e59ab8e3b8feefbbc554187b7`.
  - source manifest: `/data2/usr_for_deadline/SAM3_InterMOT_N72R5/worktree/outputs/N72R11R4/formal_e1a_attempt_01/exact_v3_replay_manifest_attempt_01.json`; recorded SHA-256 `ee8610f9b661ec145bf37cfa1c7909cc20bbdcacb9dffec9a533fe94bd79ae79`.
- `E1B_PCTIS` metrics: `/data2/usr_for_deadline/SAM3_InterMOT_N72R5/worktree/outputs/N72R11R4/formal_e1b_metrics_attempt_02.json`; SHA-256 `41802fef72a5bdde662ad21cdbd4b702ae5cffaffd246789c990b6817669436e`.
  - source manifest: `/data2/usr_for_deadline/SAM3_InterMOT_N72R5/worktree/outputs/N72R11R4/formal_e1b_attempt_01/exact_v3_replay_manifest_attempt_01.json`; recorded SHA-256 `7d8c333bdbb31ae6b557b3b93df98d5eef2349bed028b8776eca6893980b5bb4`.
- Frozen protocol: `/data2/usr_for_deadline/SAM3_InterMOT_N72R5/worktree/outputs/N72R9/protocol.json`; recorded SHA-256 `e012ecc3bd64fec4409fccd57d920f3690b8b1c4b5a2dd557e03e2bfb43ef0e9`.
- Pinned TrackEval commit: `12c8791b303e0a0b50f753af204249e622d0281a`.

## Execution record

1. Export smoke for `n72r5-pool-n37-dancetrack0001-0296-authoritative_reassign-001` passed: 1 event, 3 horizons, 9 export records.
2. The first exporter smoke failed before experiment logic because the direct script did not add the project root to `sys.path`; the failure is retained in `outputs/N72R11R5/smoke_attempt1/export_failure_attempt1.json`. The import-path repair was targeted and the same smoke then passed.
3. TrackEval smoke attempt 1 failed because the pinned CLI parsed the default `SEQMAP_FILE=None` as a one-element list. The failure is retained in `outputs/N72R11R5/smoke_attempt1/trackeval_raw/h020/trackeval_failure_attempt1.json`.
4. TrackEval smoke attempt 2 failed because the pinned CLI similarly parsed `OUTPUT_FOLDER` as a list. The failure is retained in `outputs/N72R11R5/smoke_attempt1/trackeval_raw/h020/trackeval_failure_attempt2.json`.
5. The runner was repaired locally by using the supported `SEQ_INFO` path and a narrow compatibility unwrap for scalar CLI arguments; the identical H20 smoke then passed with 3/3 records. The pinned TrackEval submodule was not modified.
6. The full exporter was run CPU-only in one blocking process. Its first failure and the subsequent lossless preflight regression are both retained; no full TrackEval run was started after the preflight found the same sealed invalid rows.

## Actionable blocking evidence

- Artifact: `outputs/N72R11R5/export_failure_invalid_assigned_boxes.json`.
- Status: `BLOCKED_INVALID_SEALED_ASSIGNED_BOXES`; reason: `sealed exact-solver assigned rows violate x2>x1 or y2>y1`.
- Invalid exact-solver assigned rows: **42**; de-duplicated logical `(event, variant, frame, public_id)` rows: **42**.
- Affected events: **2** — `n72r5-pool-n37-dancetrack0027-0148-authoritative_reassign-007, n72r5-pool-n37-dancetrack0033-0154-recover_identity-108`.
- Per sealed variant: `{'E0_BASELINE_B0': 14, 'E1A_EXACT_ONPOLICY_V3_LEGACY': 14, 'E1B_PCTIS_LEGACY': 14}`.
- Representative row: `dancetrack0027`, absolute frame `222` (relative H100 frame `74`), public ID `1007`, box `[571.0000228881836, 307.00000047683716, 571.0000228881836, 308.00000050105155]`; therefore `x2 == x1`.
- The rows have `solver_status=ASSIGNED_TO_PUBLIC_ID`, explicit exact global solver authority, and no alternate box field. Clipping, dropping, replacing with a one-pixel box, or selecting a different candidate would change the sealed solver output and violate the protocol.
- No historical runtime artifact was modified; `allowed_repairs=[]`.
- Affected absolute frames by event:
  - `n72r5-pool-n37-dancetrack0027-0148-authoritative_reassign-007`: `[222, 232, 233, 235, 236, 238, 239, 240, 244, 246]`.
  - `n72r5-pool-n37-dancetrack0033-0154-recover_identity-108`: `[234, 248]`.

## Machine-readable gate

- Export status: `NOT_AVAILABLE`.
- TrackEval status: `NOT_RUN`; records `0/288`.
- Aggregation status: `BLOCKED_INCOMPLETE_WINDOW_TRACKEVAL`.
- Final status: `BLOCKED_INCOMPLETE_WINDOW_TRACKEVAL`; metrics available: `False`.
- Because the complete gate did not pass, no pooled/paired TrackEval aggregate and no HOTA/AssA/DetA/IDF1/MOTA/IDSW claim is made here.

## Questions required by the protocol

- Q1 (PCTIS vs B0 association): not estimable in N72R11R5 because the complete official interaction-window matrix was blocked before aggregation.
- Q2 (PCTIS vs V3): not estimable for the same reason.
- Q3 (association vs detection): not estimable; the diagnostic cannot use the one-event smoke as a 32-event result.
- Q4 (interpretation of the earlier `+0.042414` identity-error reduction): that N72R11R4 value remains a frozen legacy post-hoc identity metric, not an interaction-window TrackEval result. N72R11R5 does not reinterpret it or upgrade it to a benchmark claim.

## Required next step

The smallest protocol-preserving recovery is upstream: repair/re-seal the two affected N72R11R4 runtime windows so every exact solver-assigned future box has finite positive width and height, or obtain an explicitly authorized equivalent runtime regeneration. Then rerun the lossless exporter preflight and, only if it passes, the complete pinned TrackEval matrix. N72R11R5 itself must not alter those boxes or omit the rows.

No calibration head, selector, decoder LoRA, E2 evaluation, threshold change, checkpoint change, or GT-ID remapping was performed or authorized by this blocked result.

## Artifacts

- Machine-readable final gate: `outputs/N72R11R5/FINAL_STATUS.json`.
- Export failure: `outputs/N72R11R5/export_failure_invalid_assigned_boxes.json`.
- Stage statuses: `outputs/N72R11R5/stage_01_status.json`, `outputs/N72R11R5/stage_02_status.json`, `outputs/N72R11R5/stage_03_status.json`.
