NO — explicit NONE-class supervision did not make the frozen InterMOT identity representation a safe and useful cross-sequence open-set identity verifier.

# InterMOT N72R20R3R1 — Explicit-NONE Identity Verification

Final decision: `FAIL_EXPLICIT_NONE_CROSS_SEQUENCE_VERIFICATION`.

The primary question was whether a human-confirmed identity can be recognized against hard competing identities while rejecting candidate-set absence. The frozen-backbone verifier failed the preregistered static gate, so no causal memory replay or association authority was opened.

## Goal and protocol

- Goal: `Explicit-NONE Identity Verification`; central question: Can explicit NONE-class supervision over frozen InterMOT candidate sets transform the existing human anchor and frozen N72R18 persistent identity state into a cross-sequence calibrated open-set identity verifier that abstains when the target is not represented and selects the correct existing candidate when it is, without changing candidate generation, OSNet, N72R18 GRU, the assignment solver, or public-ID authority?
- Source stage: N72R20R3; R3 decision unchanged after Phase 0 audit: `True`.
- Data lineage: eight frozen DanceTrack train sequences, 8,414 future frames, S0/S1 frozen states; GT was used only for training/posthoc labels.
- Formal protocol: 8 sequence-held-out folds × seeds 720301/720302/720303; no frame-IID split.
- OSNet, N72R18 GRU, candidate generation, solver, public authority and SAM3 were frozen.

## Primary static result

Selected primary model: `V2_DUAL_STATE`. Pooled negative FPR = `0.3002`; open-set correct-ID recall = `0.1000`; macro recall = `0.1044`; macro FPR = `0.3758`.

| Method | Params | Neg FPR | P0 FPR | P1 FPR | Open-ID recall | Macro recall | NONE recall | Candidate accuracy | Causal wrong-write | Causal retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R3 B4 Logistic (S0/S1 pooled rows) | 11 | 0.1295 | 0.2338 | 0.1198 | 0.1317 | — | — | — | not run | not run |
| V1_PAIRWISE_THRESHOLD_NONE | 39842 | 0.0429 | 0.0617 | 0.0411 | 0.0408 | 0.0420 | 0.9571 | 0.0408 | not run | not run |
| V2_ANCHOR_ONLY | 36626 | 0.2903 | 0.4383 | 0.2766 | 0.0691 | 0.0703 | 0.7097 | 0.0691 | not run | not run |
| V2_LEARNED_STATE_ONLY | 36626 | 0.3656 | 0.3961 | 0.3628 | 0.1184 | 0.1175 | 0.6344 | 0.1184 | not run | not run |
| V2_DUAL_STATE | 39842 | 0.3002 | 0.3149 | 0.2988 | 0.1000 | 0.1044 | 0.6998 | 0.1000 | not run | not run |

Gate thresholds were fixed before formal held-out evaluation: pooled FPR ≤ 0.02, pooled open-ID recall ≥ 0.60, macro recall ≥ 0.40, macro FPR ≤ 0.05. The selected model failed all four performance checks; all eight sequences were nevertheless reported, and no candidate was created.

## Sequence-cluster uncertainty

The 2,000-repetition sequence-cluster bootstrap used seed `720311`. 95% intervals: negative FPR `[0.19865271568140422, 0.5241174552223491]`, open-ID recall `[0.05149473702420744, 0.147875090061777]`, macro recall `[0.055732447379164864, 0.1540571318551947]`.

## Frozen primary model by sequence

| Sequence | Absent | Present | FPR | P0 FPR | P1 FPR | Open-ID recall | Candidate accuracy | NONE recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dancetrack0001 | 70 | 1334 | 0.2429 | 0.0833 | 0.2759 | 0.0465 | 0.0465 | 0.7571 |
| dancetrack0002 | 430 | 1974 | 0.5953 | 0.2500 | 0.6053 | 0.0502 | 0.0502 | 0.4047 |
| dancetrack0023 | 748 | 2216 | 0.1257 | 0.0750 | 0.1285 | 0.0162 | 0.0162 | 0.8743 |
| dancetrack0024 | 102 | 1422 | 0.9118 | 0.5000 | 0.9375 | 0.2039 | 0.2039 | 0.0882 |
| dancetrack0039 | 98 | 2384 | 0.6224 | 0.6220 | 0.6250 | 0.1472 | 0.1472 | 0.3776 |
| dancetrack0057 | 58 | 1184 | 0.0000 | 0.0000 | 0.0000 | 0.0279 | 0.0279 | 1.0000 |
| dancetrack0062 | 494 | 1910 | 0.2287 | — | 0.2287 | 0.1618 | 0.1618 | 0.7713 |
| dancetrack0072 | 1638 | 766 | 0.2796 | 0.2571 | 0.2817 | 0.1815 | 0.1815 | 0.7204 |

## Phase 0 and lineage audit

- Bug A corrected diagnostic: public-ID-resolved base candidate presence changed 13,778 rows, but gate metric delta was 0.0.
- Bug B corrected candidate coverage changed from 0.670135 to 0.849673 in the aggregate diagnostic, but B4 features, calibration, gate metrics and R3 final decision were unchanged.
- Historical R3 SHA audit: `{"all_unchanged": true, "files": [{"bytes": 826, "path": "outputs/N72R20R3/FINAL_GOAL.json", "sha256": "c035049c610d48cdca533d3828f5caede26ab2918db06ad453828fcf9ad7bfcd"}, {"bytes": 56182, "path": "outputs/N72R20R3/FINAL_RESULT.json", "sha256": "942b955af1f98cef29fa9d18f369df50d1af46e197a3f4f1674af5cf40d60ab1"}, {"bytes": 13773, "path": "outputs/N72R20R3/FINAL_REPORT.md", "sha256": "266938ccd2b030dcfb4ab54e3b3c872fc86cb1d1dac7a2e0cf7a4f221f3b9aa4"}, {"bytes": 356553, "path": "outputs/N72R20R3/presence/loso_results.json", "sha256": "0ecc2c0ffc396ed44e267e8996b68c37f28b3992cd4adb560bc0085a00e4e5b1"}], "source_commit": "75696491945456849f5ebdb75fc785e087d8fb41"}`.
- The R3 historical output directory was not overwritten.

## Causal and association boundary

Static failure means `causal_commit_replay_authorized=false`. Causal memory writes, solver redesign, association rescue, SAM3 rerun, DanceTrack val/test, LoRA, Transformer, GRU training and R4 authority were not run. Future-association headroom is logged read-only and was not applied.

## Tests and known baseline issue

R3R1 focused tests and full pytest result: 46 focused passed; full pytest 370 passed, 4 failed. The four failures are the fixed-version TrackEval old CLI handling of `SEQMAP_FILE` (the parser stores the argument as a list and later passes it to `os.path.isfile`), not evidence that all tests pass.

## Terminal interpretation

The result is a negative answer to the only N72R20R3R1 question: explicit NONE supervision, as implemented in this small verifier over the frozen representation, did not provide reliable cross-sequence open-set identity recognition. `next_association_authority_stage_authorized=false`.
