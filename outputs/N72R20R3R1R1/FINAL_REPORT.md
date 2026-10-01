NO — after correcting the one-frame decision protocol and fully isolating outer held-out sequences, explicit-NONE verification over the frozen InterMOT identity representation does not succeed.

# InterMOT N72R20R3R1R1 — Protocol-Corrected Explicit-NONE Identity Verification

Final decision: `FAIL_PROTOCOL_CORRECTED_EXPLICIT_NONE_GENERALIZATION`.

The primary model was pre-frozen as `V2_DUAL_STATE`; no architecture was selected from formal outer scores. The corrected index has exactly `8414` unique `(sequence, frame)` decisions, with `6595` PRESENT and `1819` NONE labels.

## Primary static result

- pooled negative FPR: `0.22649807586586038`
- pooled open-set correct-ID recall: `0.13115996967399546`
- macro FPR: `0.29894651638641284`
- macro open-ID recall: `0.14275970144332548`
- closed-set candidate top-1 / Rank-1: `0.2360879454131918`
- Rank-2: `0.4421531463229719`
- Rank-3: `0.6157695223654284`
- closed-set MRR: `0.47608391398004735`
- static gate pass: `False`

## Protocol and provenance

- 8 outer sequence-held-out folds × seeds 720301/720302/720303.
- Each outer fold trains on 6 sequences, selects epoch on one cyclic inner-validation sequence, then refits on all 7 non-heldout sequences for that epoch.
- Seed probabilities are averaged into one final prediction per real frame; no 2× state-condition rows remain.
- `runtime_future_gt_used`: `False`; `runtime_gt_clean`: `True`.
- Historical R3R1 artifacts unchanged: `True`.

## Sequence-cluster bootstrap

2,000 repetitions with seed `720312`; 95% intervals: `{'negative_fpr': [0.17593123209169054, 0.34600793347981107], 'open_recall': [0.06018387413950119, 0.2287799716480425], 'closed_set_top1': [0.17482407235555303, 0.3188124382233623], 'macro_recall': [0.057513418859410255, 0.24741158629403706]}`.

## Terminal boundary

Causal replay: `NOT_RUN_STATIC_GATE_FAILED`.
Association rescue, solver changes, SAM3, DanceTrack VAL/TEST, TrackEval, LoRA, Transformer, JEV, Mamba, OSNet retraining and GRU retraining were not run.

Bottleneck classification: `BOTTLENECK_CANDIDATE_IDENTITY_REPRESENTATION`.
`next_representation_learning_stage_authorized=True`.
`next_association_authority_stage_authorized=False`.

Focused and full test summary: R3R1R1 focused 46 passed; combined R3R1R1+R3R1 92 passed; full pytest 416 passed, 4 failed (fixed TrackEval SEQMAP_FILE CLI compatibility).
