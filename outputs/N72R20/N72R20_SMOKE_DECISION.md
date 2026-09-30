# N72R20 Smoke Decision

FINAL GOAL: **Human-Initialized Identity Memory in Real Candidate Streams**

CENTRAL QUESTION: **Can one human initialization be recognized reliably in future real SAM3 candidates?**

## Decision

`PASS_TRAIN_CANDIDATE_PIPELINE_SMOKE_ONLY`

## Answers

1. **SAM3 candidate pipeline:** Yes for the bounded train smoke. The pinned
   SAM3 loader constructed a multiplex model and both selected train sequences
   produced complete 160-frame compact candidate caches.
2. **Candidate coverage:** Not measured. This smoke intentionally did not read
   GT; candidate coverage requires the later posthoc evaluation boundary.
3. **Identity ranking:** Not measured in this smoke. No identity-memory
   evaluation or update experiment was started.
4. **Formal N72R20 evaluation:** Not authorized in this task. Val, H20/H50/H100,
   TrackEval, MOT, association, and training remain stopped.

N72R20 remains `BLOCKED_VAL_NOT_AUTHORIZED`; this smoke is not a scientific
PASS for identity representation and does not authorize the next association
stage.
