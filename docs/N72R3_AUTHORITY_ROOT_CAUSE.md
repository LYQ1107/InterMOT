# N72R3 authority root cause

Date: 2026-09-02T08:00:29.786619+00:00

## Machine conclusion

`CANDIDATE_FIRST_AUTHORITY = true`. The N72R2 active authority architecture is invalid for sequence-persistent public identity:

- `ActiveTrackAuthority` creates its own `TrackManager` and registers candidates before association.
- `ContinuousObserverDriver` receives a different manager from its caller.
- `StateManager._new_pid` is an association-local allocator; it is not the persistent public-ID owner.
- `PublicAuthorityBinding` is candidate-scoped and the old bridge accepts a state changing from one public ID to another.
- `PersistentLineageHandover.match_overlap()` turns IoU/appearance evidence into `status=PASS`, which is not exact authority.
- The real track/output field is `Track.mot_track_id`; `TrackManager.final_mot_track_id` does not exist.
- `IdentityNamespace` is not on the N72R2 authority-smoke active path, while independent state/track/public allocators remain.

The N72R2 final status remains `BLOCKED_CANDIDATE_RECALL` as historical evidence. N72R3 retires its 13/13 candidate-presence requirement as an identity-continuity gate.

## Dependency-free state probe

The old bridge accepted association state `7` → public `11` at frame 10 and state `7` → public `22` at frame 20. This is an architecture defect, not a scientific result; the probe uses fake tracks and no dataset or GT.

```json
{
  "bindings_created": 2,
  "public_ids_for_same_state": [
    11,
    22
  ],
  "resolution_at_frame_20": {
    "binding": null,
    "public_id": null,
    "reason": "multiple_public_authorities",
    "status": "COLLISION"
  },
  "same_association_state_id": 7,
  "same_state_multiple_public_ids_accepted": true
}
```

## Required replacement

N72R3 must make a sequence-lifetime `SequencePersistentIdentityRuntime` the owner of public identity, lineage, appearance/motion/lost state and one injected `TrackManager`. A candidate is assigned to an existing persistent identity; only an outer birth decision may allocate a new identity. Session reset may clear raw/adapter/SAM bindings and mark an identity LOST/NONE, but must not delete or renumber it.

IoU/appearance/motion may produce only recovery/association evidence. Exact authority must come from the persistent identity record and an immutable state → public binding.

## Evidence

Machine audit: `outputs/N72R3/audits/n72r2_authority_semantics.json`

Historical N72R2 evidence is read-only and was not changed.
