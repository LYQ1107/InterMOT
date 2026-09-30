# N72R19R1 leakage audit

FINAL GOAL: Selective Identity Memory Update

This audit records what is available at inference time and what is used only to
construct training labels or evaluation truth.

## Inference-time inputs

- The human-confirmed anchor embedding is immutable for the episode.
- The current frozen OSNet 512-D observation and the current-frame visible
  competitor embeddings are available.
- The selector sees only the 13 causal evidence features in
  `sam3_intermot/identity_memory/selective.py`.
- Temporal gap, trusted-update count, and recent state stability are maintained
  online. They do not contain identity labels or future-frame values.
- The selector checkpoint is a 3,009-parameter MLP. Its inference path does
  not use corruption labels, future utility, future embeddings, or TrackEval.

## Training-only information

- Train targets use the fixed synthetic corruption manifest to identify clean
  versus corrupted observations. This label is never passed as an inference
  feature.
- The future-utility objective uses target and competitor embeddings at offsets
  1, 5, and 10 only while training. The checkpoint records
  `uses_future_utility_as_inference_feature: false`.
- The train/dev split is sequence-level. The 25 validation sequences are not
  loaded by the selector-training script and are not used for threshold tuning.

## Evaluation truth

- Positive identity crops and all visible competitors are used to compute the
  frozen identity endpoint after each causal update.
- The corruption manifest is generated once from the clean frozen N72R18 GRU
  replay and is shared by every method. It is not a learned signal and is not
  used to decide whether a runtime update is accepted.
- Val decisions are made only from the predeclared protocol and dev-frozen
  hyperparameters. No val result is fed back into training or selection.

## Explicit exclusions

No SAM3 inference, MOT association, Hungarian assignment, TrackEval, LoRA,
identity-decoder training, or future-frame query is part of this probe.
