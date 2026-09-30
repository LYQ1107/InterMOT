# N72R20 Current Runtime Audit

## Frozen scope

Final Goal: **Learned Identity Memory Integration into InterMOT**.

Central question: **Does the N72R18 learned identity memory, initialized once
by a human correction, improve future public-ID association in the existing
InterMOT runtime when candidate generation, assignment solver, public-ID
authority, and interaction protocol are held fixed?**

This is an audit of the existing runtime. No candidate matcher, identity
memory, association solver, SAM3 backend, or public-ID authority was changed
for this audit.

## Existing path and ownership

| Stage | Existing implementation | Ownership that remains fixed |
|---|---|---|
| Official candidate generation | `sam3_intermot/backend/sam3_backend.py`: `detect_concept`, `correct_object`, `propagate`, and `export_frame_candidates`/`export_frame_candidates_v2` | SAM3 candidate boxes, native IDs, confidence and candidate ordering |
| Persistent runtime identity state | `sam3_intermot/identity/persistent_runtime.py` | session lifecycle, candidate binding, correction epochs, snapshots and restoration |
| Public-ID authority | `sam3_intermot/identity/public_authority.py` | immutable state-to-public/MOT mapping and explicit `EXPLICIT_NONE` resolution |
| Existing appearance state | `sam3_intermot/association/appearance_memory.py` and `trusted_persistent_public_state.py` | score-before-update and N72R15 consensus-only write policy |
| Assignment | `association/effect_assignment.py` -> `association/public_assignment.py` | one exact `linear_sum_assignment` call, explicit NONE columns and public axis |
| Relative state edge | `association/relative_persistent_state_edge.py` | appearance/motion/native-continuity composition and row-max-preserving fusion |
| Official decoder bridge | `association/decoder_candidate_bridge.py` | decoder outputs are candidate evidence; multiplex slots do not become public IDs |

The runtime already has the required public identity and candidate interfaces.
It does not need an MOT rewrite or a SAM3 rebuild for this stage. The missing
research seam is a learned identity-memory signal that can be evaluated inside
the existing candidate-to-public assignment path without taking ownership of
candidate generation, assignment, or public-ID authority.

## Required invariants for the integration

1. Candidate rows, candidate UIDs, boxes, native IDs, scores and ordering must
   be byte-for-byte/protocol-equivalent between baseline and treatment.
2. The assignment solver remains `solve_exact_public_assignment`; no second
   solver and no Hungarian redesign are allowed.
3. Public IDs remain owned by `PublicAuthorityBridge`, never by a SAM3 native
   ID, decoder slot, or learned-memory module.
4. The human correction is the only initialization event. A learned state must
   be read before the future-frame decision and updated only after that
   decision, so the current-frame write cannot affect its own result.
5. Future GT is post-hoc evaluation truth only. It cannot enter runtime state,
   candidate generation, assignment, or memory updates.
6. The frozen N72R18 feature contract is 512-D L2-normalized OSNet output. No
   new encoder or feature normalization may be introduced in the integration.
7. Val, if later run, remains frozen: no threshold, selector, memory, sequence,
   or checkpoint tuning on val.

## Audit conclusion

The current runtime is structurally suitable for an integration replay. The
N72R15 trusted persistent bank already demonstrates the correct placement for a
state signal: it prepares state-owned evidence, fuses it without changing the
candidate-vs-NONE axis, and delegates the final decision to the existing exact
public assignment solver. The N72R18 learned memory can therefore be tested as
an additional state-owned identity signal, not as a replacement tracker.

This document does **not** establish association gain. The scientific result
remains pending the frozen baseline/treatment replay and its H20/H50/H100
TrackEval records.
