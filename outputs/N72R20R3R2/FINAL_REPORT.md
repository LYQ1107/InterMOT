NO — sequence-held-out identity representation learning passes Phase A, but the complete open-set/causal requirement is not met.

# InterMOT N72R20R3R2 — Cross-Scene Open-Set Identity Representation Learning

Final decision: `FAIL_OPEN_SET_CALIBRATION_AFTER_REPRESENTATION`.

## Phase A — representation

Frozen learned-state baseline Rank-1 `0.3815011372251706`; new runtime-state Rank-1 `0.6874905231235785`; delta `0.3059893858984079`; MRR `0.7868773842617663`; Rank-2/3/5 `0.7807429871114481` / `0.8488248673237301` / `0.9435936315390447`.
Oracle-clean Rank-1 `0.9035633055344958`; sequence-cluster paired 95% CI `[0.15633034534356954, 0.4439950217797137]`; representation gate `True`.

## Phase B — explicit NONE

Pooled negative FPR `0.23474436503573393`; pooled open-set correct-ID recall `0.5323730098559515`; macro negative FPR `0.40467475367641337`; macro recall `0.5424804849934729`; gate `False`.

## Phase C — causal commit replay

Causal metrics: `NOT RUN — Phase B failed`; gate `False`.

OSNet, N72R18 GRU, candidate tape, exact solver and public-ID authority remained frozen. No SAM3 rerun, DanceTrack VAL/TEST access, association override or candidate creation occurred.
next_association_authority_stage_authorized=False
Focused/full test summary: R3R2 focused 36 passed; full pytest 452 passed, 4 known TrackEval SEQMAP_FILE CLI failures.
