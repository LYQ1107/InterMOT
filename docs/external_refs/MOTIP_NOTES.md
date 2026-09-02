# MOTIP reference notes

This note records an idea-level audit for N72R3R1/N72R4. No MOTIP source code
or lines are copied into InterMOT.

- Repository: [MCG-NJU/MOTIP](https://github.com/MCG-NJU/MOTIP)
- Audited paths: `models/runtime_tracker.py`, `models/motip/id_decoder.py`,
  `models/motip/trajectory_modeling.py`
- Audited commit: `ffc0e905ac196a603027eca8d18fb0dff48c8bcc`
- Audit date recorded by the frozen external-reference artifact:
  `2026-07-30T20:04:18+08:00`
- License recorded by the audit: Apache-2.0

## Reusable mechanism

MOTIP is relevant as a design reference for track-centric trajectory context,
explicit identity prediction, and separating existing-identity prediction from
newborn handling. InterMOT keeps its own immutable sequence-lifetime
`public_id` authority, SAM3 candidate stream, explicit-NONE global assignment,
and outer birth policy. It does not import MOTIP's finite identity vocabulary,
decoder, detector, or backbone.

## Deliberately not reused

MOTIP is not evidence for InterMOT's human-correction future effect. Its code is
not used to select events, alter the Hungarian solver, assign public IDs, or
authorize training. The N72R4 conclusion is determined only by the frozen
InterMOT artifacts and gates.
