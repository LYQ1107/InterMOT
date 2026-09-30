# N72R20 Smoke Decision

FINAL GOAL: **Human Correction Driven Persistent Identity Adaptation**

CENTRAL QUESTION: **When a human corrects a tracking error, can the system learn from this correction and improve future identity tracking?**

## Decision

`PASS_TRAIN_CANDIDATE_PIPELINE_SMOKE_ONLY`

## Answers

1. **SAM3 candidate pipeline:** Yes for the bounded train smoke. The pinned
   SAM3 loader constructed a multiplex model and both selected train sequences
   produced complete 160-frame compact candidate caches.
2. **Correction environment:** The two train smoke caches and frozen B2
   records provide a bounded real-candidate error-discovery source. GT is used
   only offline to simulate correction events.
3. **Correction events:** Generated under
   `outputs/N72R20/correction_events/`; the record-only updater does not change
   memory and no training was performed.
4. **Current N72R20 boundary:** DanceTrack train only. Val, H20/H50/H100,
   TrackEval, MOT, association, and training are out of scope.

N72R20 is `PASS_TRAIN_CORRECTION_EVENTS`; this is an environment
construction result, not evidence that a learned correction update improves
future tracking. That question is reserved for N72R21.
