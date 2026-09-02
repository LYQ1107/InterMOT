# N72R3 protocol — persistent public identity across independent SAM sessions

This protocol is frozen before the N72R3 implementation. Historical N36–N72R2
outputs, the shared SAM3 checkpoint, and `third_party/sam3` are read-only.

The central contract is:

> A persistent public identity survives a SAM session boundary independently
> of whether a session-local candidate is present.

SAM raw IDs, adapter IDs, candidate UIDs, masks, and backend state are
session/observation-local. `public_id`, `mot_track_id`, the identity lineage,
association state, appearance memory, and motion/lost history belong to the
sequence-persistent identity runtime. Candidate absence at a boundary is a
valid `NO_CANDIDATE_ASSIGNED`/`LOST` observation, not identity deletion.

The former N72R2 `13/13` boundary-candidate requirement is explicitly retired
as an identity-continuity prerequisite. IoU, appearance, and motion matching
can produce association or recovery evidence, but never exact authority.

Frozen numerical protocol:

- Candidate definition: N72R1 Candidate V2.
- Window length/overlap: 160/20 frames; fixed-window checks are 1 → 2 → 6.
- Future horizons: H20/H50/H100.
- Sequence-cluster bootstrap: 2,000 repetitions, seed 7202.
- Maximum four GPUs, one sequence/frame-range process per GPU, and existing
  160 → 100 → 50 OOM sharding only.
- Runtime future-GT reads: zero. For GT-simulated interaction, only current
  GT may be read after `Y_pre` is frozen; future GT is posthoc only.

The machine-readable source of truth is
`outputs/N72R3/protocol.json`; the protection inventory is
`outputs/N72R3/protection_manifest.json`.
