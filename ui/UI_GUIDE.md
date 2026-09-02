# N72R1 external human UI

This is an ingestion boundary for direct external annotations. It is not a
simulator and it does not infer a public ID from GT, candidate order, or a
numeric tracker ID.

Start it from the isolated N72R1 worktree with paths chosen under a separate
N72R1 output root:

```bash
cd /path/to/SAM3_InterMOT_N72R1/worktree
PYTHONPATH="$PWD" /path/to/intermot/bin/python ui/n72r1_human_ui.py \
  --host 127.0.0.1 --port 8762 \
  --raw-root /path/to/N72R1/human_events/raw_namespace \
  --event-root /path/to/N72R1/human_events/validated_namespace \
  --annotator-id '<annotator-account>'
```

The annotator must submit the complete V2 record through `/api/events` and
confirm the action, frame, and raw input. Existing IDs are selected directly
by the annotator. `ADD_NEW_IDENTITY` has no user-supplied public ID; the
runtime allocator owns that assignment. A confirmed mask must remain a
lossless PNG/NPZ/RLE payload with an independent digest; a machine candidate
mask cannot be relabeled as human evidence.

Every accepted record is server-finalized as `real_human`, retains the exact
request bytes in the raw namespace, and must have `runtime_future_gt_used:
false`. The event-frame write is hidden from the event frame and becomes
eligible for memory reads at event+1.

Validate the append-only tape with the project validator before any replay:

```bash
PYTHONPATH="$PWD" /path/to/intermot/bin/python \
  scripts/n72r1_validate_real_human_tape.py \
  --event-root /path/to/N72R1/human_events/validated_namespace \
  --raw-root /path/to/N72R1/human_events/raw_namespace
```

An empty result is not experimental evidence. Do not import N37/N39/N41/N42
or later `simulated_from_gt` artifacts into this tape.
