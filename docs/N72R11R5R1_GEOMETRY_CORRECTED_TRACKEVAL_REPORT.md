# N72R11R5R1 — Positive-Geometry Corrected Interaction-Window TrackEval

**Final structural status: `PASS_N72R11R5R1_WINDOW_TRACKEVAL`**
**Research effect gate: `FAIL_FUTURE_EFFECT`**

## Executive conclusion

The sealed N72R11R4 exact-solver rows were not edited.  N72R11R5R1 applied the registered positive-area geometry policy before model/solver execution, regenerated E0 from the frozen sources, reran E1A/E1B on all 32 frozen events, and completed the official pinned TrackEval interaction-window matrix. This is a structural diagnostic completion, not evidence that the causal treatment improves identity tracking.

The corrected causal gate remains `FAIL_FUTURE_EFFECT`: the H20 sequence-cluster lower bounds are not strictly positive for either E1A or E1B. No calibration, selector, decoder LoRA, E2, or production promotion is authorized.

## 1. Frozen scope and provenance

- Frozen N72R9 protocol: `/data2/usr_for_deadline/SAM3_InterMOT_N72R5/worktree/outputs/N72R9/protocol.json`; protocol SHA-256 `e012ecc3bd64fec4409fccd57d920f3690b8b1c4b5a2dd557e03e2bfb43ef0e9`.
- Events/sequences: `32` events, `18` independent sequences; action and event order unchanged.
- Future window: exactly event frame +1 through +100; H20/H50/H100 are prefixes of that same window.
- Runtime GT: `runtime_future_gt_used=false`; GT was used only for offline/post-hoc metrics and TrackEval ground truth.
- Interaction provenance: all events remain `simulated_from_gt`; there is no real-human tape.
- Checkpoint, candidate definition, Hungarian solver, embedding and metric definition were not changed. `third_party/sam3` and the TrackEval submodule were not modified.

## 2. Original failure and actionable root cause

The prior N72R11R5 export was blocked by 42 sealed assigned rows with non-positive boxes in two events (`dancetrack0027` and `dancetrack0033`). Post-hoc clipping, dropping, replacement, or candidate substitution would have changed sealed solver output and was forbidden.

The corrected source audit found 40 invalid *candidate-source* rows across 57,987 source rows: `c0_source=14`, `c1_source=14`, `requery_source=10`, `target_stream_source=2`; affected sequences were dancetrack0008 (2), 0015 (6), 0027 (24), and 0033 (8). No frame became an empty solver pool (`empty_solver_pool_frame_count=0`). These rows were rejected before model/solver, with bounded audit records; no geometry was invented.

A separate exporter attempt then failed because the protocol label `validation` was blindly mapped to the physical `val` directory for dancetrack0051/0052. The frozen N72R6 GT audit records the exact train GT hashes, and N72R6/N72R9 runtime code reads these sequences from `train`; after verifying both hashes and each frozen N72R7 source manifest, the exporter recorded the resolution as `VALIDATION_LABEL_RESOLVED_TO_FROZEN_TRAIN_SOURCE`. The original failure is preserved in [export_failure_attempt1.json](../outputs/N72R11R5R1/export_failure_attempt1.json) with no fabricated traceback.

## 3. Geometry repair and smoke

The only runtime change was the opt-in `require_positive_geometry=true` policy. Normalization remains finite-box normalization; invalid candidates are separated and logged before model/solver. The baseline was regenerated from the same frozen source streams rather than filtering an old E0 artifact after the fact.

- Smoke event: `n72r5-pool-n37-dancetrack0027-0148-authoritative_reassign-007`; E1A and E1B both passed.
- E0 corrected runtime hashes: `['4f3e1e7c86921ae2e5c8619acd19a15ebe99c41a2f18279e1418bc3812aa2211', '4f3e1e7c86921ae2e5c8619acd19a15ebe99c41a2f18279e1418bc3812aa2211']`; identical across E1A/E1B: `True`.
- Smoke assigned/candidate geometry violations: `0`; runtime GT violations: `0`.

## 4. Corrected replay completeness

- E1A: `{'pass': True, 'manifest_status': 'PASS_ALL_SELECTED', 'record_count': 32, 'unique_record_count': 32, 'duplicate_event_count': 0, 'missing_event_count': 0, 'total_filtered_geometry_candidates': 30, 'filtered_geometry_by_source': {'MAIN_B0_CANDIDATE': 28, 'TARGET_SESSION_CURRENT_RAW': 2}, 'failures': []}`.
- E1B: `{'pass': True, 'manifest_status': 'PASS_ALL_SELECTED', 'record_count': 32, 'unique_record_count': 32, 'duplicate_event_count': 0, 'missing_event_count': 0, 'total_filtered_geometry_candidates': 30, 'filtered_geometry_by_source': {'MAIN_B0_CANDIDATE': 28, 'TARGET_SESSION_CURRENT_RAW': 2}, 'failures': []}`.
- Both formal manifests contain 32 unique event records, zero missing/duplicate events, zero validator failures, 101 frames per event, and runtime future-GT false.
- The 30 filtered invalid candidates per formal treatment are source rows rejected by the new pre-solver policy; they are not missing event rows and are not post-hoc replacements.

### Corrected E1A (V3)

| Horizon | identity-error reduction | 95% CI | assignment change | correct / incorrect | protected regression |
|---:|---:|---|---:|---:|---:|
| H20 | -0.019576 | [-0.106251, 0.181495] | 0.535073 | 64 / 76 | 8 |
| H50 | -0.081081 | [-0.157491, 0.065641] | 0.416988 | 92 / 218 | 19 |
| H100 | -0.068371 | [-0.120793, 0.073667] | 0.372524 | 177 / 391 | 25 |

### Corrected E1B (PCTIS)

| Horizon | identity-error reduction | 95% CI | assignment change | correct / incorrect | protected regression |
|---:|---:|---|---:|---:|---:|
| H20 | 0.042414 | [-0.041032, 0.201799] | 0.274062 | 56 / 30 | 7 |
| H50 | -0.000644 | [-0.066751, 0.132664] | 0.176319 | 80 / 81 | 8 |
| H100 | -0.008307 | [-0.046482, 0.094342] | 0.139936 | 114 / 140 | 14 |

## 5. Corrected causal result

The corrected custom causal aggregator was reused; it is complete as an execution artifact even where its scientific effect status is negative. E1A H20 reduction is -0.019576 with CI [-0.106251, 0.181495]. E1B H20 reduction is +0.042414 with CI [-0.041032, 0.201799]. Neither lower bound is > 0. At longer horizons both treatments are non-positive on the point estimate, and incorrect crossings remain substantial.

The comparison artifact shows that the geometry-policy rerun did not create a new causal signal: corrected and old R4 values are equal for almost all fields; the only visible change is a one-row E1A H100 correct crossing / tiny identity reduction difference caused by removing invalid source candidates upstream. This is a contract repair, not a performance improvement.

## 6. Interaction-window TrackEval result

The exporter created 96 pseudo sequences and 288 unique `(event, logical_variant, horizon)` records. The pinned TrackEval checkout was commit `12c8791b303e0a0b50f753af204249e622d0281a`; H20/H50/H100 each completed 96 records. These are local interaction-window diagnostics, not official DanceTrack benchmark scores.

| Horizon | Variant | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|---:|---|---:|---:|---:|---:|---:|---:|
| H20 | E0_BASELINE_B0 | 0.680935 | 0.772592 | 0.605232 | 0.779271 | 0.556068 | 32.000000 |
| H20 | E1A_V3 | 0.670045 | 0.762085 | 0.594339 | 0.767056 | 0.530173 | 44.000000 |
| H20 | E1B_PCTIS | 0.676341 | 0.765893 | 0.602501 | 0.775195 | 0.546193 | 51.000000 |
| H50 | E0_BASELINE_B0 | 0.667856 | 0.733672 | 0.610643 | 0.774861 | 0.559352 | 91.000000 |
| H50 | E1A_V3 | 0.654656 | 0.723419 | 0.595113 | 0.757433 | 0.524664 | 155.000000 |
| H50 | E1B_PCTIS | 0.662308 | 0.729586 | 0.603938 | 0.766764 | 0.543228 | 128.000000 |
| H100 | E0_BASELINE_B0 | 0.665767 | 0.721408 | 0.616112 | 0.788364 | 0.579139 | 177.000000 |
| H100 | E1A_V3 | 0.651746 | 0.708435 | 0.601315 | 0.768713 | 0.540492 | 360.000000 |
| H100 | E1B_PCTIS | 0.660161 | 0.718145 | 0.608541 | 0.780025 | 0.561075 | 236.000000 |

The official TrackEval matrix is structurally complete, but lower HOTA/AssA/IDF1 and higher IDSW for E1A/E1B relative to E0 in the pooled rows are consistent with the failed causal gate. No metric is used to retroactively select a treatment or alter the protocol.

## 7. Answers to the twelve required questions

1. **Why was the original exporter blocked?** Forty-two sealed assigned rows had zero/non-positive bbox area; post-hoc repair was illegal.
2. **What was changed?** Only pre-solver positive-geometry eligibility and bounded source rejection audit; the frozen sealed rows remain unchanged.
3. **Were candidate definitions or solver semantics changed?** No. Source order and original source-axis indices are preserved; invalid rows are rejected only when the explicit flag is enabled.
4. **Did filtering empty a solver pool?** No; the source audit found zero empty-after-filter solver frames.
5. **Was E0 regenerated correctly?** Yes. It was rebuilt from frozen sources with the same policy, and the two smoke E0 hashes are identical.
6. **Was the full corrected replay complete?** Yes: E1A and E1B are each 32/32 with unique event keys and no runtime validator failures.
7. **Was the GT split mismatch silently ignored?** No. The requested `validation` label and resolved physical `train` source, exact hashes, and frozen-manifest evidence are recorded in the export protocol.
8. **Was runtime future GT used?** No. Runtime artifacts, corrected replay audit, export and aggregation all report false; GT appears only in post-hoc evaluation.
9. **What is the corrected E1A effect?** H20 -0.019576, CI lower -0.106251; H50 -0.081081, lower -0.157491; H100 -0.068371, lower -0.120793.
10. **What is the corrected E1B effect?** H20 +0.042414, CI lower -0.041032; H50 -0.000644, lower -0.066751; H100 -0.008307, lower -0.046482.
11. **Did TrackEval complete and is this a benchmark score?** 288/288 completed across all three horizons; it is an interaction-window diagnostic, explicitly not an official DanceTrack benchmark score.
12. **What is authorized next?** Preserve the geometry/source resolver and TrackEval diagnostic as research artifacts, but stop downstream learning. The next scientific step requires a new, pre-registered association/interface or provenance-complete real-human event tape; do not increase LoRA rank, change checkpoint, tune thresholds, or call E2 to bypass the failed gate.

## 8. Failure preservation, E2, and authorization

- Historical R5 invalid-box evidence remains at `outputs/N72R11R5/export_failure_invalid_assigned_boxes.json`; it was not overwritten. Current exporter split failure is at `outputs/N72R11R5R1/export_failure_attempt1.json`.
- The six corrected source-audit candidate rejection groups and all formal replay logs remain under `outputs/N72R11R5R1/`; no failure was relabeled as PASS.
- E2 remains `NOT_MEASURED_INCOMPLETE_FORMAL_REPLAY`; no E2 runtime was started.
- Calibration head, selector, decoder LoRA and production integration remain `NOT_AUTHORIZED` because the strict future-effect gate failed.

## 9. Machine-readable artifacts

- [final gate](../outputs/N72R11R5R1/final_gate.json)
- [geometry source audit](../outputs/N72R11R5R1/geometry_source_audit.json)
- [corrected replay audit](../outputs/N72R11R5R1/corrected_replay_audit.json)
- [corrected-vs-old comparison](../outputs/N72R11R5R1/geometry_repair_effect.json)
- [export manifest](../outputs/N72R11R5R1/export_manifest.json)
- [TrackEval run manifest](../outputs/N72R11R5R1/trackeval_run_manifest.json)
- [TrackEval aggregation](../outputs/N72R11R5R1/aggregation_manifest.json)
- [stage 01](../outputs/N72R11R5R1/stage_01_status.json), [stage 02](../outputs/N72R11R5R1/stage_02_status.json), [stage 03](../outputs/N72R11R5R1/stage_03_status.json), [stage 04](../outputs/N72R11R5R1/stage_04_status.json), [stage 05](../outputs/N72R11R5R1/stage_05_status.json)

## 10. Reproducibility hashes

- Branch: `codex/n72r11r5r1-geometry-trackeval`; commit used for finalization: `9b9a683abb153facd588ac5eeec5b9386b1e6f94`.
- TrackEval commit: `12c8791b303e0a0b50f753af204249e622d0281a`.
- Code and input SHA-256 records are in `outputs/N72R11R5R1/final_gate.json` under `hashes`; historical outputs modified: `false`.

## 11. Gate checklist

- `geometry_source_audit`: `True`
- `corrected_smoke`: `True`
- `corrected_replay`: `True`
- `corrected_causal_metrics`: `True`
- `export`: `True`
- `trackeval`: `True`
- `aggregation`: `True`

The structural TrackEval completion status must not be confused with the failed research-effect gate. This report therefore closes N72R11R5R1 as a completed diagnostic while leaving the appearance/identity causal claim unresolved.
