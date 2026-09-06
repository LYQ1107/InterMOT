# N72R10R1 Mechanism Correction

## Scope

This note corrects the historical label of the N72R10 implementation without
rerunning or changing any N72R10 artifact. The frozen N72R10 outputs remain
read-only inputs for N72R11.

## Corrected interpretation

N72R10 did implement a causal future-session lifecycle: the event-frame
correction was sealed first, and the first possible memory read was on
`event_frame + 1`. Its runtime source was therefore a genuine causal future
session.

The formal N72R10 replay wiring, however, used an event-level future candidate
stream that had already been generated and then reused it when the uncertainty
rule fired. The precise label for that wiring is:

`EVENT_PLUS_ONE_FUTURE_SOURCE_WITH_UNCERTAINTY_GATED_REUSE`

It is not yet a test of a fresh SAM3 query created at the uncertainty frame.
That stricter mechanism is the N72R11 subject. The preserved N72R10
implementation label is:

`GENUINE_CAUSAL_FUTURE_SESSION`

The true on-demand uncertainty-frame requery is:

`NOT_YET_TESTED`

## Frozen facts

The N72R10 E2−E1 values are not recomputed or reinterpreted here:

| Horizon | E2−E1 identity-error reduction | 95% CI | Changes | Correct / wrong |
|---|---:|---|---:|---:|
| H20 | 0.01631321370309951 | [0.0016339869281045752, 0.059967320261437904] | 44 | 10 / 0 |
| H50 | 0.014157014157014158 | [0.001111111111111111, 0.050851063829787234] | 69 | 24 / 2 |
| H100 | 0.009904153354632588 | [5.668934240362783e-06, 0.034104144750683754] | 94 | 34 / 3 |

The N72R10 development gate remains failed because of protected regression,
zero validation future-positive coverage, and the below-target training
distribution. No production authorization is inferred by this correction.

## N72R11 boundary

N72R11 must create a fresh `FutureFrameRequerySession` only after a causal
uncertainty trigger, probe its four fixed queries, and allow the selected
session to propagate. It must not use the frozen N72R10 future stream as the
formal on-demand path, alter the exact Hungarian solver, read future GT at
runtime, or rewrite the N72R10 evidence.
