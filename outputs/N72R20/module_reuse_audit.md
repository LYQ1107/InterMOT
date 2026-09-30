# N72R20 Module Reuse Audit

This audit is governed by [`FINAL_GOAL.json`](FINAL_GOAL.json). The only new
runtime connection module is
[`sam3_intermot/identity_memory/candidate_identity_matcher.py`](../../sam3_intermot/identity_memory/candidate_identity_matcher.py).

## Reused modules

| Component | N72R20 use | Boundary |
|---|---|---|
| `sam3_intermot/backend/sam3_backend.py` | Official SAM3 candidate generation and propagation | Reused directly; no SAM3 transformer/token/memory change. |
| `sam3_intermot/adaptation/sam3_loader.py` | Existing loader utilities where applicable | Not a replacement for the pinned multiplex video backend. |
| `sam3_intermot/identity_probe/encoders.py` | Frozen N72R16 OSNet x1.0 Market-1501 crop encoder | No retraining; output is a normalized 512-D embedding. |
| `sam3_intermot/identity_memory/selective.py::load_frozen_n72r18_gru` | Strict loading of the frozen N72R18 GRU checkpoint | N72R19R1 selector is not promoted into N72R20. |
| `sam3_intermot/identity_memory/updater.py` / `memory.py` | Existing frozen GRU architecture/state contract | No new memory architecture or training. |
| `scripts/n72r20_identity_bridge_eval.py` | Causal B0/B1/B2/Oracle evaluation | GT is posthoc evaluation truth only. |

## New lightweight connector

`CandidateIdentityMatcher` accepts the compact SAM3 candidate record, obtains
or consumes one OSNet embedding per candidate, computes cosine scores against
one identity state, and returns a stable descending rank. It does not save
images, masks, decoder tensors, or dense feature maps. It rejects GT/public
identity fields in runtime candidate records.

Memory writes remain in the existing N72R20 bridge/evaluator, where the frozen
N72R18 GRU and the explicit immediate/2-frame diagnostics are controlled. The
new connector is not a selector, tracker, Hungarian assignment, or training
path.

## Not modified

- `third_party/sam3/*`
- the OSNet or N72R18 checkpoint contents
- global MOT association and TrackEval
- any training or LoRA path
