# N72 real-human event tape collection contract

## Current state

There is currently no verified real-human event tape in this project.  The
N37/N39/N41/N42/N70/N71 event records are frozen synthetic controls marked
`interaction_source=simulated_from_gt`; they are useful for mechanism
diagnostics but must not be renamed or imported as human evidence.

N72 provides an ingestion boundary, not a simulator.  The external UI or
annotator must emit one JSON object per line and preserve the referenced raw
files.  The validator rejects missing provenance, GT-derived fields, machine
masks relabeled as human masks, nonzero runtime future-GT flags, incomplete
candidate windows, and non-exact identity mappings.

## External collection workflow

1. Use only an audited `train` or `train_fold` sequence and freeze the UI
   version, session ID, annotator pseudonymous hash, and timestamps.
2. At the selected event frame, preserve the original frame bytes and SHA-256.
   The annotator directly enters the `public_id`; the system must not infer it
   from IoU, appearance, GT, or a future frame.
3. Capture exactly one raw human input: a BOX, CLICK points, or a
   CONFIRMED_MASK.  Store the raw UI payload and digest.  A confirmed mask
   must be lossless and explicitly marked `machine_candidate_mask=false`.
4. Run the official spatial correction first.  Freeze the current-frame
   output before writing memory; the event frame must not read the newly
   written memory.  The first allowed memory read is event frame + 1.
5. Export the complete candidate tape covering the event frame and the
   non-overlapping H20/H50/H100 future ranges.  Each candidate needs the raw
   official ID, adapter external ID, segment-local ID, sequence-global ID,
   candidate UID, frame/mask/feature provenance, and an exact public mapping or
   an explicit non-exact status.
6. Validate before any full-loop or replay command.  Rejected records are
   written to the append-only failure directory and are never silently
   repaired.  No N72 scientific result is authorized by a schema-only PASS.

## JSONL event schema

The following is a compact contract skeleton.  Real collection must replace
every toy value and must keep the raw referenced files available:

```json
{
  "protocol": "N72_REAL_HUMAN_EVENT_TAPE_V1",
  "event_id": "seq_frame_action_unique",
  "interaction_source": "real_human",
  "human_confirmed": true,
  "runtime_future_gt_used": false,
  "sequence": "dancetrackXXXX",
  "split": "train",
  "event_frame": 123,
  "public_id": 17,
  "public_id_source": "human_direct",
  "action_type": "AUTHORITATIVE_REASSIGN",
  "ui_version": "external-ui-version",
  "session_id": "session-id",
  "annotator_id_hash": "64-hex-digest",
  "event_start_timestamp": "2026-09-02T00:00:00+08:00",
  "event_end_timestamp": "2026-09-02T00:00:02+08:00",
  "frame_image_ref": "raw/seq/000123.jpg",
  "frame_image_sha256": "64-hex-digest",
  "candidate_tape_ref": "candidates/seq_event.json",
  "human_input": {
    "kind": "BOX",
    "origin": "external_human_ui",
    "human_confirmed": true,
    "raw_payload_ref": "raw/seq/000123.ui.json",
    "raw_payload_sha256": "64-hex-digest",
    "box_xyxy": [10.0, 20.0, 80.0, 160.0]
  },
  "human_embedding": {
    "derived_from": "BOX",
    "source_kind": "human_roi_encoder",
    "feature_dim": 512,
    "finite": true,
    "norm": 1.0,
    "sha256": "64-hex-digest"
  },
  "prefix_range": [0, 122],
  "future_ranges": {"H20": [124, 143], "H50": [124, 173], "H100": [124, 223]},
  "spatial_correction": {
    "status": "PASS",
    "current_frame_output_frozen_before": true,
    "correction_before_memory_write": true,
    "backend_prompt_route": "native_box"
  },
  "mapping_audit": {
    "status": "EXACT",
    "source": "direct_user_public_id",
    "public_id": 17,
    "raw_native_id": 41,
    "adapter_external_id": 5041,
    "segment_local_id": "chunk-0:41",
    "sequence_global_id": "dancetrackXXXX:g41",
    "stable": true
  },
  "memory_audit": {
    "event_frame_read": false,
    "current_frame_write_hidden": true,
    "first_visible_frame": 124,
    "write_after_spatial_correction": true
  }
}
```

The candidate tape is a `N72_CANDIDATE_TAPE_V1` JSON object with
`frame_manifest` and `rows`, or JSONL with exactly one `frame_manifest` record.
It must include the event frame and all 100 future frames, with no duplicate
frame, raw-ID, or candidate-UID keys.  Runtime candidate rows must set
`runtime_future_gt_used=false`.  Posthoc GT labels are not event-input fields
and must never be placed in this tape.

## Validation command

From the project root, after collecting raw files externally:

```bash
PYTHONPATH=. /home/lwr/anaconda3/envs/intermot/bin/python \
  scripts/n72_real_human_event_cli.py \
  --input /path/to/external/events.jsonl \
  --candidate-root /path/to/candidate_exports \
  --raw-root /path/to/raw_ui_capture \
  --failure-dir outputs/N72/human_tape/attempts \
  --report outputs/N72/human_tape/import_report.json
```

The command exits zero only when every input record passes.  It is an import
and contract check only; it does not start SAM3, full-loop, replay, training,
calibration, selector, or LoRA.  The append-only recorder in
`sam3_intermot/interaction/n72_real_human.py` uses binary JSONL append,
`fsync`, and never truncates a prior event.

## Official backend input limitations

The current official SAM3 multiplex backend exposes box prompts.  Its
`add_points` path raises `NotSupportedError`; `correct_object` therefore uses
`box_fallback_from_click` for click input.  A mask prompt is converted to its
box because the official request path does not accept a native mask; record
this as `box_fallback_from_mask`.  Only BOX can use `native_box`.  The raw
click/mask payload remains preserved so a later supported backend can replay
the original human input without claiming that the current backend consumed a
native point or mask prompt.

`HumanInteraction(source="human")` in the internal interaction API is not a
proof of provenance.  Realness comes from the external UI origin,
human-confirmation, raw payload digest, annotator/session metadata, and the
strict validator above.

## What is still required

The minimum external input is one or more genuine UI sessions producing the
event JSONL plus raw frame/input files and a candidate export with complete
exact mapping axes.  Until that exists, N72 must remain blocked from any claim
of real-human future efficacy.  No synthetic or GT-derived record should be
generated to satisfy the quota.
