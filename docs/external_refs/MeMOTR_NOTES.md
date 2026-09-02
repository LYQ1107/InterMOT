# MeMOTR reference notes

This note records an idea-level audit for N72R3R1/N72R4. No MeMOTR source code
or lines are copied into InterMOT.

- Repository: [MCG-NJU/MeMOTR](https://github.com/MCG-NJU/MeMOTR)
- Audited paths: `models/query_updater.py`, `models/memotr.py`,
  `models/runtime_tracker.py`, `models/motion.py`
- Audited commit: `eb7a177b9cbcb89742ec69b2545ab3af2ea31a80`
- Audit date recorded by the frozen external-reference artifact:
  `2025-10-15T14:56:34+08:00`
- License recorded by the audit: MIT

## Reusable mechanism

MeMOTR is relevant as a design reference for long-memory/query updates,
confidence-conditioned feature updates, lost-track persistence, and optional
motion context. InterMOT uses only this state-design intuition. SAM3 remains
the detector/propagator, and the InterMOT persistent identity runtime owns
public IDs, lineage, session detachment, and candidate authority.

## Deliberately not reused

MeMOTR is not used to alter the frozen candidate definition, memory weights,
association solver, future windows, or strict sequence-cluster gate. It does
not turn the N72R4 simulated-from-GT evidence into real-human evidence and does
not authorize a production memory update.
