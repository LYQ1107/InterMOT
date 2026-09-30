# N72R20 Runtime Reuse Audit

Final Goal: `outputs/N72R20/FINAL_GOAL.json`

Central question: When a human corrects a tracking error, can the system learn from this correction and improve future identity tracking?

Current role: construct a train-only interactive correction environment; GT is
offline error-discovery truth and never a runtime input.

Audit status: checkpoint restored, train-only smoke completed, and correction
events generated for the two bounded train sequences. The official endpoint
was gated; a SHA256-verified public mirror was used. Val was stopped and its
partial cache quarantined after the Goal correction.

## Reuse decisions

| Existing component | N72R20 decision | Boundary |
|---|---|---|
| `sam3_intermot/backend/sam3_backend.py` | Reuse `Sam3Backend` directly | It already wraps the pinned SAM3 multiplex API, CPU video/output offload, process-controlled streaming, `detect_concept`, `propagate`, and complete candidate export. Do not create a second backend or modify `third_party/sam3`. |
| `Sam3Backend.export_frame_candidates_v2` | Reuse its provenance/schema contract | Supply machine OSNet features only; N72R20 storage must omit dense masks by default and use compact per-sequence records. No public identity is inferred by the exporter. |
| `sam3_intermot/adaptation/sam3_loader.py` | Do not use as the multiplex runtime | It is a detector/model adaptation helper, not a replacement for the pinned `Sam3Backend` video predictor. |
| `sam3_intermot/association/decoder_candidate_bridge.py` | Reuse only candidate normalization/provenance ideas if needed | N72R20 is target-centric. Do not turn this stage into a new global Hungarian redesign. |
| `scripts/n72r7_run_candidate_generator_batch.py` | Reuse one-independent-process-per-unit and rotating per-unit logs | N72R7's event requery worker and outputs are not N72R20 candidate data; no historical candidate rows are silently relabeled as real-stream rows. |
| `scripts/n72r7_candidate_generator_requery.py` | Reuse causal provenance conventions and OSNet crop extraction pattern | Its target-session requery protocol is event-specific and is not the N72R20 all-candidate stream definition. |
| `scripts/n72r15_candidate_coverage_posthoc.py` | Reuse IoU `>= 0.50` coverage decomposition and GT-posthoc boundary | Keep `candidate unavailable` separate from `identity discrimination failure`; do not reuse N72R15 outputs as N72R20 evidence. |
| `scripts/n72r20_candidate_stream_smoke.py` / `scripts/n72r20_run_candidate_smoke.py` | New thin N72R20 path-portable wrapper around the existing backend/export contract | The worker reads image frames only, uses the frozen OSNet checkpoint, stores compact metadata plus float16 features, and is deliberately not an identity associator. It must be run only after the checkpoint audit passes. |
| `scripts/n72r20_identity_bridge_eval.py` / `scripts/n72r20_aggregate_identity_bridge.py` | Preserved historical identity-probe layer | Its train records are an error-discovery source; the old val endpoint is superseded and not a current-stage result. |
| `scripts/n72r20_generate_correction_events.py` | Current N72R20 error-discovery adapter | Reads GT only after GT-free train candidate records exist, classifies identity switch/missed target/wrong recovery, and writes compact simulated human correction events. |
| `sam3_intermot/interaction/correction_event.py` | Current event data contract | Stores candidate ids, pre-correction state, corrected embedding, and candidate context; no image or dense tensor persistence. |
| `sam3_intermot/identity_memory/correction_update.py` | Future update boundary | Record-only dummy adapter; it returns memory unchanged and reserves learning for N72R21. |
| `sam3_intermot/identity_probe/dataset.py` | Reuse sequence/GT parsing and validation concepts | The frozen N72R17 protocol and the N72R20 asset manifest are the authority for paths and anchors. GT is evaluation truth only after runtime rows are produced. |
| `sam3_intermot/identity_memory/selective.py::load_frozen_n72r18_gru` | Reuse strict frozen N72R18 GRU loader | The N72R20 core comparison is frozen GRU versus immutable human-anchor baseline. N72R19/R1 selector models are not promoted into this stage. |
| `sam3_intermot/identity_memory/updater.py` | Reuse the already serialized GRU architecture/state contract | No new memory architecture or training is authorized. |
| `sam3_intermot/identity_memory/candidate_identity_matcher.py` | Use the new thin candidate-to-identity connector for frozen feature scoring/ranking | It is not a selector, tracker, SAM3 modification, or training path. |

## N72R20 causal contract

- Human initialization is `simulated_from_gt`, once at the frozen N72R17 earliest eligible anchor.
- Future GT is not a runtime input. It is read only for posthoc candidate coverage, identity ranking, contamination, and recovery metrics.
- The target must first be evaluated against the complete real SAM3 candidate set. Candidate absence is not scored as an identity-ranking error.
- B0 is immutable human anchor only; B1 is EMA(0.90); B2 is the frozen N72R18 GRU. Oracle-correct updates are diagnostic only.
- Immediate update versus two-frame confirmation is an engineering contamination diagnostic, not a new selector research direction.
- The two-frame margin threshold must be frozen from train-only B2 records before val; a val-derived threshold is prohibited.
- A candidate row matching no target or other visible identity is not silently promoted to a hard negative; hard negatives require IoU `>= 0.50` with another visible GT identity.
- Runtime memory updates are replayed over every frame after the anchor, including target-not-visible frames. Those frames are retained for causal state evolution but excluded from target-visible coverage/ranking denominators.
- Candidate data is streamed or split by sequence; no monolithic cache, crop PNG archive, dense mask cache, or raw SAM tensor dump is allowed.

## Current boundary and next permitted action

The dataset, frozen identity assets, checkpoint, train-only smoke, and train
correction events are ready. Stop after validating the event artifacts. No
val/MOT/association work or training is authorized in the current task.
