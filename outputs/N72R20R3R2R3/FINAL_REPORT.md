# InterMOT N72R20R3R2R3 Final Report

## Q1 — Did candidate localization improve?

- Baseline P1a frames: 632; combined P1a→V recovery: 0.0095.
- Baseline mean target IoU: 0.564544; combined mean target IoU: 0.564961.
- Combined valid-candidate coverage: 0.5930.

## Q2 — Did identity-aware evidence improve association?

- Combined AssA: 0.541954 versus baseline 0.541854; IDSW: 217.0 versus 217.0.
- G2 shadow N01=0, N10=0; G3 shadow N01=90, N10=0.

## Q3 — Did complete MOT improve?

- Dev HOTA(AUC): 0.582536 versus baseline 0.582439; DetA: 0.628251 versus 0.628157; AssA: 0.541954 versus 0.541854.
- Final scientific decision: `FAIL_VAL_GENERALIZATION`.

## Required end-to-end table

| Method | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.5824390062628039 | 0.628156812640047 | 0.5418544920565236 | 0.6232634541611995 | 0.5940227453054747 | 217.0 | NA |
| Candidate quality only | 0.5824390062628039 | 0.628156812640047 | 0.5418544920565236 | 0.6232634541611995 | 0.5940227453054747 | 217.0 | NA |
| Box refinement only | 0.5565076543049107 | 0.5891082528206123 | 0.5301683826564632 | 0.6099219743787497 | 0.5653644160652889 | 250.0 | NA |
| Targeted SAM3 only | 0.5825433015333412 | 0.6282809145002605 | 0.541941353524679 | 0.6233636154208275 | 0.5942872256016927 | 217.0 | NA |
| Identity association only | 0.5824390062628039 | 0.628156812640047 | 0.5418544920565236 | 0.6232634541611995 | 0.5940227453054747 | 217.0 | NA |
| Combined Dev | 0.5825357841107411 | 0.6282513002740138 | 0.5419535878474467 | 0.6233636154208275 | 0.594249442702233 | 217.0 | NA |
| Combined VAL | 0.45619931370881145 | 0.4107078295160756 | 0.5100162714932082 | 0.46548970357275227 | 0.3124033968767211 | 1108.0 | NA |

## Localization variants

| Variant | P1a frames | Valid candidate coverage | Mean best IoU | P1a→V recovery |
| --- | ---: | ---: | ---: | ---: |
| Original | 632 | 0.592284 | 0.564544 | 0.000000 |
| Box refine | 632 | 0.506652 | 0.379697 | 0.041139 |
| Targeted SAM3 | 632 | 0.593130 | 0.565015 | 0.011076 |
| Combined | 632 | 0.593009 | 0.564961 | 0.009494 |

## Association variants

| Variant | Changed frames | Wrong→Correct | Correct→Wrong | Net corrections | AssA Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| G0_ADDITIVE | 0 | 0 | 0 | 0 | NA |
| G1_TARGET_COLUMN | 1440 | 90 | 0 | 90 | NA |
| G2_GATED | 0 | 0 | 0 | 0 | NA |
| G3_RECOVERY_ONLY | 1440 | 90 | 0 | 90 | NA |

## Protocol and limitations

- TrackEval is the pinned evaluator invoked through the dedicated local scalar/list compatibility wrapper with identical settings for all dev trackers.
- Outer LOSO uses six fit sequences and one inner validation sequence; heldout GT is used only after model/policy selection for posthoc evaluation.
- Targeted SAM3 status: `PASS_BOUNDED_TARGETED_SAM3`; full SAM3 retraining and LoRA were not used.
- VAL status: `VAL_COMPLETE`; targeted SAM3 status: `PASS_BOUNDED_TARGETED_SAM3`; DanceTrack test was not accessed.
- Historical N72R20R3R2, R3R2R1 and R3R2R2 outputs were not modified.


## VAL result

- Baseline HOTA(AUC): 0.45620288499479394; treatment HOTA(AUC): 0.45619931370881145; delta: -3.5712859824954535e-06.
- VAL AssA delta: -7.41935055981191e-06; DetA delta: -3.367575437951409e-07; targeted SAM3: `PASS_BOUNDED_TARGETED_SAM3` with 29 candidate(s) and causal selection `frozen_causal_event_frame`.
- All 25 VAL base tapes were available; the candidate index uses a train-named compatibility view of DanceTrack val, while TrackEval uses the val split. Offline simulated event anchors use the first frame containing both a candidate and GT; this GT is not used for training, tuning, or runtime decisions.
- VAL GT was used only for simulated event preparation and TrackEval truth; it was not used for training, tuning, or runtime decisions.
