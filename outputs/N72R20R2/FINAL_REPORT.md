FINAL GOAL: Trusted Identity Memory Commit and Association Rescue

CENTRAL QUESTION: “Can runtime learned-identity evidence prevent unsafe machine observations from entering persistent identity memory and selectively rescue ambiguous or incorrect public-ID associations while the candidate stream, exact assignment solver, public-ID authority, frozen N72R18 memory and interaction protocol remain unchanged?”

The answer is no at the current development gate: none of the ten preregistered commit policies simultaneously satisfied the required wrong-write rate and retention thresholds, so association rescue was not authorized.

## Decision

`FAIL_RUNTIME_MEMORY_COMMIT`

`next_association_stage_authorized=false`

This is a runtime memory-commit failure, not evidence that an unrun rescue or VAL experiment succeeded or failed. The R2 stop rule was applied immediately after the commit gate. N72R20 and N72R20R1 artifacts were preserved unchanged.

## Frozen protocol and lineage

- Dataset: DanceTrack train only; the exact eight-sequence dev set was `dancetrack0001`, `dancetrack0002`, `dancetrack0023`, `dancetrack0024`, `dancetrack0039`, `dancetrack0057`, `dancetrack0062`, and `dancetrack0072`.
- Candidate tape: 8,422 frames, 49,202 candidate rows, 53,998,226 bytes; validation passed in `candidate_tape_validation.json`.
- Storage audit after the formal dev run remained `OK` with approximately 118.85 GiB free and a 27.61 GB safe budget; no new dataset or checkpoint copy was made.
- Candidate generation, OSNet, N72R18 GRU, interaction protocol, public-ID authority, and exact solver were held fixed.
- Runtime future GT was not read. GT appears only in post-hoc labels for the commit audit.
- DanceTrack val/test, MOT/TrackEval, training, LoRA, and new identity models were not run.

## Commit gate

The preregistered gate was wrong-write rate `<= 0.02` and correct-write retention versus C0 `>= 0.60`.

| policy | threshold | accepted | correct | wrong | wrong rate | retention | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| C0 | 0.00 | 6889 | 4889 | 2000 | 0.290318 | 1.000000 | FAIL safety |
| C1 | 0.00 | 4558 | 3448 | 1110 | 0.243528 | 0.705257 | FAIL safety |
| C2 | 0.00 | 4558 | 3448 | 1110 | 0.243528 | 0.705257 | FAIL safety |
| C2 | 0.02 | 3729 | 2874 | 855 | 0.229284 | 0.587850 | FAIL both |
| C2 | 0.04 | 2333 | 1801 | 532 | 0.228033 | 0.368378 | FAIL both |
| C2 | 0.06 | 1649 | 1204 | 445 | 0.269861 | 0.246267 | FAIL both |
| C2 | 0.08 | 1216 | 1051 | 165 | 0.135691 | 0.214972 | FAIL both |
| C2 | 0.10 | 870 | 803 | 67 | 0.077011 | 0.164246 | FAIL both |
| C2 | 0.15 | 290 | 265 | 25 | 0.086207 | 0.054203 | FAIL both |
| C3 | 0.00 | 3438 | 2725 | 713 | 0.207388 | 0.557374 | FAIL both |

The complete machine-readable audit is `outputs/N72R20R2/commit_gate_decision.json`, with the original scan in `outputs/N72R20R2/commit_policy_scan.json`.

## Verification

The R2 targeted suite passed 31/31 tests. The full suite passed 282 tests and retained four known failures in legacy TrackEval integration tests. All four fail inside the pinned third-party TrackEval CLI because `SEQMAP_FILE` is received as a one-element list and is passed to `os.path.isfile`; they do not exercise the R2 commit/rescue code.

## Resource and engineering notes

The first unbounded SAM3 attempt reached an out-of-memory condition during a train sequence. It was quarantined. The formal candidate lineage used the official trim setting and bounded 160-frame sessions with runtime seed rebinding; all eight formal tapes passed validation. This changed only runtime resource handling and did not change identity memory, candidate matching, or assignment logic.

## Next action

Do not run rescue, DanceTrack val, MOT/TrackEval, training, LoRA, or association integration from this stage. A future stage would first need an explicitly authorized new protocol addressing the failed trusted-memory commit gate.
