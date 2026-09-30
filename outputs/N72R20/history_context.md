# N72R20 Historical Context

This file is governed by [`FINAL_GOAL.json`](FINAL_GOAL.json). It records the
prior identity-memory evidence as preserved provenance. The current N72R20
question is now:

> When a human corrects a tracking error, can the system learn from this
> correction and improve future identity tracking?

The earlier identity-probe question is superseded and is not the current
N72R20 endpoint.

The earlier results were produced under the offline GT replay protocol. They
are motivation and frozen-memory provenance, not evidence that the memory has
already transferred to a real SAM3 candidate stream.

| Stage | Frozen condition | Validation H100 hard-negative win rate | Interpretation |
|---|---|---:|---|
| N72R16 | Single OSNet human anchor | 36.22% | One embedding was insufficient against hard competitors. |
| N72R17 | OSNet EMA(0.90) diagnostic | 78.13% | Causal positive-memory replay was promising, but was not deployable online evidence. |
| N72R18 | Frozen learned GRU memory | 91.36% | Learned memory improved the offline identity endpoint over EMA; checkpoint is frozen for N72R20. |
| N72R19R1 | Selective update on unseen val | 89.702% for selected B5 | Selective update failed clean preservation; this does not invalidate the frozen N72R18 memory. |

## Boundary carried into N72R20

- N72R16 reported `FAIL_IDENTITY_REPRESENTATION` for the single-anchor
  baseline.
- N72R17 reported `TEMPORAL_MEMORY_PROMISING`; its EMA/Memory-3/5 results
  were offline diagnostics, not a real-stream claim.
- N72R18 reported `PASS_PERSISTENT_MEMORY_IMPROVES_IDENTITY` under causal GT
  replay, while explicitly keeping downstream association unauthorized.
- N72R19R1 reported `FAIL_CLEAN_PRESERVATION` for its selective-update design;
  the frozen N72R18 GRU remains the N72R20 B2 reference.

The current N72R20 correction environment reuses real SAM3 candidate streams
and the same frozen OSNet/N72R18 GRU assets to discover train-only prediction
errors offline and write simulated human correction events. It does not
retrain identity models, modify SAM3, update memory, or promote the N72R19R1
selector. Learned correction updates are reserved for N72R21.

## Source reports

- [`docs/N72R16_FINAL_REPORT.md`](../../docs/N72R16_FINAL_REPORT.md)
- [`docs/N72R17_FINAL_REPORT.md`](../../docs/N72R17_FINAL_REPORT.md)
- [`docs/N72R18_FINAL_REPORT.md`](../../docs/N72R18_FINAL_REPORT.md)
- [`docs/N72R19R1_FINAL_REPORT.md`](../../docs/N72R19R1_FINAL_REPORT.md)
