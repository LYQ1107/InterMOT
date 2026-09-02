# N72R1 real-human event collection

This guide defines the external input boundary. It does not create events.

## Current state

No real human tape is present in this N72R1 run. N37/N39/N41/N42 and later simulated
artifacts remain explicitly `simulated_from_gt`; they cannot be relabeled as clicks.

## Start the local ingestion boundary

```bash
/home/lwr/anaconda3/envs/intermot/bin/python ui/n72r1_human_ui.py \
  --raw-root /data2/usr_for_deadline/SAM3_InterMOT_N72R1/human_events/raw_namespace \
  --event-root /data2/usr_for_deadline/SAM3_InterMOT_N72R1/human_events/validated_namespace \
  --annotator-id '<annotator-account>'
```

Open `http://127.0.0.1:8762/`. The external client must POST one complete JSON object
to `/api/events`; the server stores exact request bytes before validating. A submission
is not a real event until server finalization adds a session nonce and
`server_generated_real_human=true`.

## Required annotation

For every event provide the event frame, direct public ID (except ADD, whose ID is
allocated by the server), one of the four action families, raw BOX/CLICK/
CONFIRMED_MASK or ID_SELECTION, confirmation, session/annotator/timestamp, source
frame SHA-256, candidate tape reference, prefix, and H20/H50/H100 ranges.
`runtime_future_gt_used` must be exactly false.

## Backend limitation

The official adapter exposes box correction directly. Point and mask submissions
are preserved losslessly and routed through the adapter's supported point/mask path
when available; if the pinned official predictor lacks a requested form, the event
must record the explicit box fallback route rather than silently claiming native
support. A machine candidate mask is never a confirmed human mask.

## Operational boundary

After the current-frame official correction, memory is written on the event frame but
is hidden there; the first allowed read is event+1. Candidate native, adapter,
segment-local, sequence-global, and public IDs stay separate. Append-only hash-chain
artifacts preserve both accepted and rejected submissions.
