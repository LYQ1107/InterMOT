# N72R20R2 diagnosis of N72R20R1

N72R20R1 remains a completed historical stage with final decision
`FAIL_MEMORY_CONTAMINATION`. Its outputs are read-only inputs for R2 and are
not relabeled or overwritten. Historical N72R20 remains
`BLOCKED_NOT_RECOVERED`.

## Fact A — the learned identity representation is strong

On the fresh real candidate stream, the H100 hard-negative identity win rate
was:

| Method | H100 hard-negative win rate |
|---|---:|
| Human anchor | 0.391304 |
| Frozen N72R18 learned memory | 0.923913 |
| Delta | +0.532609 |

The corresponding deltas were rank-1 `+0.527174`, MRR `+0.342068`, and mean
margin `+0.148894`. R2 therefore does not retrain the OSNet encoder or the
N72R18 GRU. The scientific problem is runtime use of an already strong state.

## Fact B — learned score has no current decision authority

Across 318 future frames, learned scores changed on all 318 frames and 4,549
cells, but the exact solver changed 0 frames, changed 0 public IDs, and made 0
candidate/`NONE` changes. The correct interpretation is that a strong identity
signal is currently converted into no public-assignment action; it is not that
the learned memory lacks identity signal.

## Fact C — the current consensus rule is vacuous

The R1 learned treatment produced no solver changes. Consequently every
learned-memory write had equal base and treatment assignments. The rule
`base assignment == treatment assignment` therefore reduced to “whatever the
base assigns, write it.” That is not an identity-correctness criterion.

## Fact D — contamination is inherited from base assignment

R1 recorded 318 writes and 33 post-hoc wrong writes, a 10.377% wrong-write
rate. Since the assignment shadow changed no assignment, R1 does not support
the claim that learned memory caused those 33 assignment errors. The accurate
interpretation is that baseline machine-assignment errors were admitted into
learned memory because the consensus condition was non-discriminative.

## Fact E — R2 must test temporal contamination cascade

The R2 forensic pass will run-length encode the 33 wrong writes, with special
attention to `dancetrack0002`: first wrong frame, contiguous/near-contiguous
runs, state similarity before and after the first wrong write, subsequent
target-vs-hard-negative margins, and correlation between later errors and
earlier state corruption. This is post-hoc analysis only; it does not alter
the sealed R1 tape.

## Fact F — decision-scale mismatch is a measurable hypothesis

The frozen base scorer uses `sim=1.5`, `iou=1.0`, `native=0.5`, native bonus
`3.0`, and positive bonus `5.0`. The learned edge uses `tanh(relative)` with
state-edge scale `1.0`. R2 will measure base score ranges, assignment margins,
learned residual ranges, and residuals required to cross the exact decision
boundary rather than treating the scale hypothesis as established.

## R2 constraints

- Candidate generation, candidate tape, OSNet features, N72R18 GRU, exact
  assignment, public-ID authority and interaction protocol remain frozen.
- Runtime replay remains GT-blind. GT is joined only in separate post-hoc
  forensic/evaluation passes.
- No new trainable parameters, new neural network, new matcher, second
  Hungarian solver, candidate creation, or VAL tuning is allowed.
- R1 artifacts remain immutable; the first R2 pass reuses the sealed R1
  candidate/base tape and does not rerun SAM3.
