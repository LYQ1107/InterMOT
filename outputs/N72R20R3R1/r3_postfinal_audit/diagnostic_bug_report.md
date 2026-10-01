# R3 post-final diagnostic bug audit

This audit is written under N72R20R3R1 and does not overwrite outputs/N72R20R3.

## Bug A — public assignment UID parsing

The R3 runtime field candidate_presence_of_base_candidate had one call path that omitted target public_id. R3R1 resolves the base UID from public_assignments relative to target_public_id and checks membership on the sealed candidate UID axis.

Old true count: 0; corrected true count: 13778; changed rows: 13778.

## Bug B — candidate coverage

The corrected metric enumerates joined rows explicitly and only counts candidate-set PRESENT frames among target-GT-present frames. The stale outer-loop variable is no longer used.

Maximum absolute delta over gate metrics: 0.

R3_FINAL_DECISION_UNCHANGED=true.

No R3 historical artifact was overwritten. Training is allowed to continue only because all gate-relevant R3 metrics and the final decision are unchanged.
