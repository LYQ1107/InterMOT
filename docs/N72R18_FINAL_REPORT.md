Learned identity memory 是否超过 EMA？

# InterMOT N72R18 — Human Identity Memory Learning

## Frozen Goal

- Goal: `Human Identity Memory Learning`
- Central question: Can a learned identity memory updater build a more reliable long-term identity representation than a fixed EMA from sparse human-confirmed observations?
- Primary endpoint: `VAL_H100_HARD_NEGATIVE_IDENTITY_WIN_RATE`
- Comparator: frozen N72R17 OSNet `EMA(0.90)`

## Result

| Method | H20 | H50 | H100 | H100 sequence-cluster 95% CI | H100 median margin | MRR |
|---|---:|---:|---:|---|---:|---:|
| Single Anchor (N72R17 M1) | 61.20% | 44.95% | 36.22% | [30.27%, 43.38%] | -0.020571 | 0.5348 |
| EMA(0.90) | 80.79% | 78.63% | 78.13% | [72.70%, 84.35%] | +0.029427 | 0.8591 |
| GRU Memory | 92.79% | 91.58% | 91.36% | [87.82%, 94.53%] | +0.084300 | 0.9482 |
| GRU + Reliability Gate | 91.52% | 90.03% | 89.60% | [85.62%, 93.26%] | +0.075193 | 0.9372 |

The frozen success rule is `GRU+Gate H100 > EMA(0.90) by >= 5 pp`. The gate
variant reaches **89.60%**, which is **+11.47 percentage points** over EMA.
Therefore the machine-readable decision is:

`PASS_PERSISTENT_MEMORY_IMPROVES_IDENTITY`

The ungated GRU is higher than the gate variant by 1.76 percentage points, so
the gate is not claimed to improve the learned updater. It rejects 40 of
24,810 replay updates (0.16%) at the fixed `gate < 0.5` diagnostic threshold;
all 40 are in the 1–5 frame bin, and none are in the later bins.

## Interpretation and boundary

The result supports learned identity memory under the offline causal replay
protocol. Future same-identity GT embeddings are scored before they update the
state. This is not evidence of deployable online association, because the
correct future observation is supplied by the diagnostic replay.

No N72R18 SAM3 inference, MOT association, Hungarian redesign, TrackEval,
HOTA, IDF1, requery, LoRA, new crops, new dataset, or CLIP fine-tuning was
run. The full repository pytest was only a diagnostic check; it reported 210
passed and 4 pre-existing TrackEval-related failures.

`next_association_stage_authorized=false` remains frozen. N72R18 stops here.

## Provenance

- Canonical machine-readable result: `outputs/N72R18/FINAL_RESULT.json`
- Canonical full report: `outputs/N72R18/FINAL_REPORT.md`
- Frozen Goal: `outputs/N72R18/FINAL_GOAL.json`
- Protocol metadata: `outputs/N72R18/protocol.json`
- Reference audit: `outputs/N72R18/reference_audit.md`
- Checkpoints: `outputs/N72R18/checkpoints/`
- Validation records and secondary metrics: `outputs/N72R18/metrics/`
