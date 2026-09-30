# N72R19R1 problem diagnosis

FINAL GOAL:
Selective Identity Memory Update

## Problem A — the N72R19 updater did not inherit the proven memory

`scripts/n72r19_train_robust_memory.py` builds a new `GRU_RELIABILITY_GATE` and optimizes all of its parameters. It does not strictly load `outputs/N72R18/checkpoints/identity_memory_gru.pt`, so its 46.43% clean H100 result is a new memory failure, not evidence that the N72R18 91.36% memory is intrinsically invalid. R1 therefore loads the N72R18 GRU with `strict=True`, freezes every GRU parameter, and trains only a small observation selector.

## Problem B — causal order must be restored

N72R18 evaluates the current state against the current/future competitive frame before consuming the same-identity observation. N72R19's training path constructs a corrupted observation, updates the state, and then computes the main contrastive loss on that same frame. That is an `update → score` training target and is not the declared future consequence of an update. R1 evaluates `score → selection → update → future score`, and uses future utility only as offline training supervision.

## Problem C — selection must see competition-aware evidence

The N72R19 reliability module receives only `[previous_state, observation, delta, abs(delta)]`. It cannot see the strongest visible competitor, assigned rank, top1–top2 ambiguity, entropy, prospective state drift, anchor drift, temporal gap, or memory age. R1 computes these explicit low-dimensional features before update. The selector never receives identity labels, corruption labels, future frames, or future utility at inference.

## Design response

1. Keep `human_anchor` immutable and separate from the dynamic state.
2. Use the frozen N72R18 GRU only to form a counterfactual candidate state.
3. Train hard selection on current evidence, with future utility as an offline target rather than an inference feature.
4. Freeze one corruption manifest using the frozen N72R18 clean causal reference state so all methods see identical corruptions.
5. Require the preservation test before interpreting any robustness result.
