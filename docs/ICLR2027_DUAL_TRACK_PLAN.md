# ICLR 2027 dual-track plan after N72

Date: 2026-09-02 (Asia/Shanghai)  
Project: `SAM3_InterMOT`

## Fixed scientific boundary

The project has a useful negative synthetic mechanism record through N70/N71,
but it does not yet have a verified real-human event tape.  The two tracks
must remain separate:

| Track | Input | Allowed claim | Current state |
|---|---|---|---|
| A — real-human contract closure | External UI records with direct public IDs, raw input digests and exact candidate mapping | Whether a human correction writes usable ID-scoped evidence and changes future public-ID assignment | Blocked on external input; N72 recorder/validator ready |
| B — controlled mechanism analysis | Frozen `simulated_from_gt` controls and posthoc GT labels | Whether score motion crosses the assignment boundary under a stated control | N70/N71 diagnostic only; no production or efficacy promotion |

Track B cannot substitute for Track A.  Synthetic records must retain
`interaction_source=simulated_from_gt`; they must not be described as historical
clicks, boxes, masks, or real user evidence.

## Track A entry gate

Before any full-loop or replay, an external annotator/UI must provide:

* event JSONL with a direct `public_id`, action, event frame, session,
  annotator and timestamp provenance;
* raw event-frame bytes and the raw BOX/CLICK/CONFIRMED_MASK payload with
  SHA-256 digests;
* a candidate tape covering the event frame and H20/H50/H100 future ranges;
* raw official ID, adapter external ID, segment-local ID, sequence-global ID,
  and authoritative public mapping for every candidate needed by the
  assignment audit;
* explicit `runtime_future_gt_used=false` and causal evidence that event-frame
  memory read is false and first read is event+1.

The N72 validator is a necessary contract gate, not a scientific result.  It
must reject any missing/ambiguous/stale/colliding mapping, candidate gap,
machine mask relabeled as human, GT-derived field, or invalid causal boundary.
Only after all records pass may the fixed four-action quota and sequence quota
be audited.  No quota may be satisfied with synthetic or inferred events.

## Track B analysis gate

The frozen N70/N71 boundary evidence may be described as mechanism diagnosis:
appearance scores changed frequently, while global assignment crossings were
rare and the strict future-effect confidence bound was not positive.  It is
not a reason to add another blind weight scan, LoRA, calibration head,
selector, checkpoint, or threshold.  N72's strict five-axis filter found no
eligible N72 rows because the N71 official branch has no public assignment;
N70's old four-axis rows are context only.

If a future controlled experiment is proposed, it must preregister the
interface, event selection, split, seed, candidate stream, horizon, and
sequence-cluster bootstrap before looking at post-treatment results.  A
positive result requires a strict positive lower confidence bound and no
untouched-ID regression or leakage.  Otherwise the result stays diagnostic.

## Resource and isolation plan

No GPU work is authorized by N72 at present.  If Track A later clears the
contract gate, at most four GPUs may be used, with one independent sequence or
frame-range process per GPU and no duplicate long job.  Each process must use
the existing official SAM3 offload settings, atomic artifacts, and the
established 160/100/50-frame fallback only when necessary.  New artifacts,
logs, and checkpoints belong under an explicitly versioned N72+ directory;
N36–N71 outputs, shared checkpoints, and `third_party/sam3` remain read-only.

## ICLR 2027 calendar (hard project dates)

The project-fixed deadlines are abstract submission on **2026-09-18 AoE** and
full paper submission on **2026-09-25 AoE**.  The dates below are a decision
calendar, not a promise that missing external data will appear:

| Date window | Deliverable | Stop condition |
|---|---|---|
| Sep 2–4 | External UI emits a pilot record; run N72 schema/raw/mapping smoke | Any provenance or axis failure is retained and blocks replay |
| Sep 5–10 | Collect the preregistered multi-sequence/four-action tape | If no genuine input arrives, close Track A as input-blocked |
| Sep 11–13 | Full contract audit and real full-loop, one isolated process per unit | Any incomplete event, runtime GT, or causal failure blocks replay |
| Sep 14–16 | Fixed M0–M4 paired future replay and sequence-cluster scoring | Do not tune thresholds or choose events after outcomes |
| Sep 17–18 | Abstract evidence freeze and honest limitation statement | No real-human result means no real-human efficacy claim |
| Sep 19–24 | Full paper, artifact index, negative-control mechanism analysis | Keep synthetic and real tracks explicitly separated |
| Sep 25 | Full-paper submission | Submit only gates and evidence actually completed |

The minimum immediate action is external collection of a provenance-complete
event tape.  Until then, the defensible paper contribution is a reproducible
contract/mapping diagnosis plus a clearly labeled synthetic negative result,
not an end-to-end real-human interactive MOT claim.
