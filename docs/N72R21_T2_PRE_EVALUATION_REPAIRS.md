# N72R21 — pre-evaluation repairs and actual completion

Final Goal: **One Click, Persistent Identity: Causal Long-Term Human Tracking Across Occlusion, Reappearance and Independent Recordings**.

This is an intermediate record, not scientific closure or next-stage authorization.

All 72 preregistered T1 fits (eight outer scenes, three seeds, P0/P1/MIXED) completed. All 336 original causal-state scene seals exist. T1 outer comparisons continue; a 2026-10-09 10:28 UTC snapshot contains 90/120 runtime cases and 120/168 T2 paired-source scene seals. Partial summaries have unequal cohorts and must not be compared as complete results. Training completion does not establish identity reliability.

## Original T2 labeler restart failure

The first T2 fit driver stopped before any T2 fit: the original labeler compared freshly constructed integer-key competitive-margin dictionaries to a JSON-loaded sealed record with string keys. Recomputing every label and comparing normalized serialized JSON verifies the existing pilot record exactly, while the original Python dictionaries compare unequal. The original labeler, pilot label bytes, paired trajectories, training adapter, trainer and loss remain unchanged. A separate restart verifier rechecks every source/GT/code/protocol SHA and every regenerated label; genuinely changed contents fail instead of being overwritten. Missing records are produced by the unchanged original labeler. The restarted driver now waits for sealed sources and fits on GPU0 only after the entire T1 fit phase has ended.

## Secondary identity-claim calibration repair

Git `092fd202c24875ae3b09d4f9e204addc30374531` preserves the initial secondary evaluator and deployment controls before repair. No actual T2 runtime or evaluation had occurred. Candidate-availability PR-AUC/ECE is not identity-claim calibration. The initial new secondary helper additionally treated an unmatched rank1 candidate as a known wrong identity. This is incorrect: an unmatched candidate has UNKNOWN identity.

The versioned repair keeps the primary preregistered target-unavailable FPR2 endpoint unchanged. Separate post-seal strict one-to-one GT matching labels rank1 as verified target, verified other, or UNKNOWN; structural empty candidate sets are known no-claim negatives with score zero. UNKNOWN candidates are counted and excluded only from secondary PR-AUC/ECE/wrong-rank FPR calibration. They are not removed from all-visible recall or counted as verified correct selections. Incomplete annotation axes and inconsistent labels fail explicitly. All-UNKNOWN calibration is undefined, not a 0% false-positive success. The original secondary protocol is preserved and the repair references its SHA.

The existing primary evaluator's `wrong_writes` counts every write that is not strictly verified as the target, including unmatched candidates. It is a conservative non-target-or-unverified write count, not proof every counted crop is another identified person. Correct retention is strictly verified target writes divided by all available target observations; no UNKNOWN write inflates retention. Zero writes still cannot pass the joint safety/usefulness gate.

## Frozen deployment controls and tests

T2 tests eleven fixed controls with all three seeds/all 52 clicks: full K8/K4/K1, anchor-only, no safe write, mean prototype, no delay, fixed current safety, bank-only identity, uniform bank attention, and no availability. Same trained weights plus capacity/module deletions are explicitly inference state-shift diagnostics, not independently retrained architecture gains. Cross-session supervision is absent, not fabricated as a trained ablation.

The actual paired FIT pilot has 1,690 future-frame pairs; 1,488 have candidate-logit differences above 1e-4, but only six select a different UID. Bank effects on scores do not themselves prove useful identity decisions. This is one in-sample scene/seed, without a generalization claim.

Latest actual tests: **62 focused passed**; complete repository **776 passed, 5 failed**, four warnings, 51.40s. The four pinned TrackEval CLI list/path failures and historical literal-branch assertion remain, with raw XML. No old tests or third-party interface were patched to hide failure. T2 fitting/replay, independent-domain and lawful cross-recording evidence, final five tables and final Git closure remain active work. CHIRLA legal media access is still unresolved. No next stage is authorized.
