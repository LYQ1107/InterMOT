# What was learned after exhaustive exploration?

The R3R2 cross-scene representation remains the strong part of the system: runtime Rank-1 was `0.687491` and oracle-clean Rank-1 was `0.903563`. The new stage therefore kept the representation, candidate axes, OSNet features and N72R18 state generator frozen.

The full authorized open-set tree evaluated direct NONE comparison, two-stage presence calibration, affine and temperature score calibration, energy/entropy/margin and Mahalanobis set uncertainty, a bounded DeepSets model, conformal selection, P0/P1 multitask supervision, class-balance ratios, runtime state-quality prediction, fixed and learned query mixtures, combined inner selection, Pareto/threshold diagnostics, and temporal EMA/hysteresis.

The best combined inner-selected formal result was pooled FPR `0.082463`, pooled correct-ID recall `0.227597`, macro FPR `0.067036`, and macro recall `0.201144`. The formal gate was `False`. Causal shadow replay was completed for the selected, safest, and frontier policies and was not used to authorize association.

## Terminal decision: `FAIL_ABSENCE_SEPARABILITY`

Final bottleneck classification: `FAIL_ABSENCE_SEPARABILITY`. `next_association_authority_stage_authorized=false`. `next_memory_state_learning_stage_authorized=false`.

No SAM3 rerun, VAL/TEST access, candidate creation, solver modification, public-ID override, OSNet fine-tuning, or association authority was performed.
## Verification

Focused closure tests: `40 passed`; R3R2 representation plus closure focused tests: `76 passed`; full pytest: `492 passed, 4 failed`, all four fixed-TrackEval `SEQMAP_FILE` list/string CLI failures.
