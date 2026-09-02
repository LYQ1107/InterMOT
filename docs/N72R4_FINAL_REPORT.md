# N72R4 Final Research Report

> Final status: **M3_SIGNAL_WAS_SOLVER_ARTIFACT**; research gate **FAIL_FUTURE_EFFECT**. No production, calibration, selector, or decoder-LoRA authorization is issued.

## 1. Scope and final conclusion

N72R4 completed the persistent-prestate structural checks, an official SAM3 paired future-propagation run for the six frozen events, the NO-versus-M0 candidate-recall decomposition, and the corrected-stream M0–M4 mechanism probe. All interactions remain `simulated_from_gt`; this is not evidence of a historical real-human tape.

The semantic repair removed the historical broad M3 label as a true identity crossing. In the persistent official-stream replay, M1/M2 produced no assignment changes, while M3/M4 produced assignment changes without any true correct or true incorrect crossing. The track-centric recovery probe accepted five proposals but produced zero identity-error reduction. The supported conclusion is therefore that the previous positive-looking M3 signal was a solver/metric semantic artifact, and the remaining bottleneck is unresolved at the candidate/association decision interface. No further synthetic expansion was used to manufacture statistical confirmation.

The implementation goal remains a persistent public identity whose `public_id`, lineage, TrackManager record, association state, appearance memory, and motion state survive SAM-session boundaries; candidate bindings may clear and status may become `LOST`, but a later candidate must rebind to the same public identity. N72R4 structural evidence supports these invariants, but it does not demonstrate future identity benefit.

## 2. N72R3R1 semantic repair

| Item | Result |
|---|---|
| Utility sign | PASS; primary identity-error reduction is `baseline_error - treatment_error` |
| Assignment solver | PASS; the formal path uses explicit per-candidate NONE through `solve_effect_assignment` → `solve_exact_public_assignment` |
| Crossing taxonomy | PASS; true correct/incorrect crossings are separated from directional IoU changes |
| Sequence bootstrap | PASS; events are averaged within sequence before 2,000-repetition bootstrap |
| Runtime GT | `false`; GT appears only in offline posthoc scoring |

Old broad versus repaired M3 at H20: old assignment changes `20`, old broad-correct labels `15`; repaired assignment changes remain `20`, but true correct crossings are `0`, directional improvements are `15`, and identity-error reduction is `0`. At H50/H100 the same distinction is `50/100` changes, `45/50` directional improvements, and `0` true correct crossings. These are semantic reclassifications, not model changes.

## 3. Persistent state and official full loop

Stage 6/7/8 passed for the frozen six events: event-prestate is captured at `t-1`, public and association axes are sourced from persistent records, and candidate index/raw SAM ID are not public authority. Stage 9 adopted retry `6/6` with paired prefix equivalence `True` and runtime future GT `False`. The original Stage 9 attempt-1 blocked status and logs remain preserved.

The official path is distinct from the frozen-candidate mechanism probe: correction is executed through the official SAM3 branch before future propagation. Stage 11 then keeps the corrected candidate stream fixed across M0–M4 so that memory’s incremental association effect is not confused with spatial correction.

## 4. M0–M4 corrected-stream mechanism results

Primary metric is future identity-error reduction; IoU is reported separately. `true_correct` and `true_incorrect` are strict crossing counts; directional changes are not promoted to crossings.

| Variant | Horizon | Changes | True correct | True incorrect | Directional + | Directional − | Neutral | Identity error reduction | ΔIoU | Missing | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M0_CURRENT_FRAME_CORRECTION_ONLY | H20 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000000000 | 0.000000000 | 0.000000000 | [0.0, 0.0] |
| M0_CURRENT_FRAME_CORRECTION_ONLY | H50 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000000000 | 0.000000000 | 0.000000000 | [0.0, 0.0] |
| M0_CURRENT_FRAME_CORRECTION_ONLY | H100 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000000000 | 0.000000000 | 0.008503401 | [0.0, 0.0] |
| M1_HUMAN_EMA_PROTOTYPE | H20 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000000000 | 0.000000000 | 0.000000000 | [0.0, 0.0] |
| M1_HUMAN_EMA_PROTOTYPE | H50 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000000000 | 0.000000000 | 0.000000000 | [0.0, 0.0] |
| M1_HUMAN_EMA_PROTOTYPE | H100 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000000000 | 0.000000000 | 0.008503401 | [0.0, 0.0] |
| M2_POSITIVE_HUMAN_ANCHORS | H20 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000000000 | 0.000000000 | 0.000000000 | [0.0, 0.0] |
| M2_POSITIVE_HUMAN_ANCHORS | H50 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000000000 | 0.000000000 | 0.000000000 | [0.0, 0.0] |
| M2_POSITIVE_HUMAN_ANCHORS | H100 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000000000 | 0.000000000 | 0.008503401 | [0.0, 0.0] |
| M3_NEGATIVE_COMPETITOR_BANK | H20 | 20 | 0 | 0 | 0 | 2 | 18 | 0.000000000 | 0.000000000 | 0.170940171 | [0.0, 0.0] |
| M3_NEGATIVE_COMPETITOR_BANK | H50 | 50 | 0 | 0 | 0 | 3 | 47 | 0.000000000 | 0.000000000 | 0.168350168 | [0.0, 0.0] |
| M3_NEGATIVE_COMPETITOR_BANK | H100 | 100 | 0 | 0 | 0 | 3 | 97 | 0.000000000 | 0.000000000 | 0.178571429 | [0.0, 0.0] |
| M4_RELIABILITY_AGE_ADMISSION | H20 | 20 | 0 | 0 | 0 | 2 | 18 | 0.000000000 | 0.000000000 | 0.170940171 | [0.0, 0.0] |
| M4_RELIABILITY_AGE_ADMISSION | H50 | 50 | 0 | 0 | 0 | 3 | 47 | 0.000000000 | 0.000000000 | 0.168350168 | [0.0, 0.0] |
| M4_RELIABILITY_AGE_ADMISSION | H100 | 100 | 0 | 0 | 0 | 3 | 97 | 0.000000000 | 0.000000000 | 0.178571429 | [0.0, 0.0] |

M3/M4 H20 had `20` assignment changes, `0` true correct crossings, `0` true incorrect crossings, `2` directional regressions, `18` neutral changes, and missing rate `0.170940171`. H50 had `50` changes with `3` directional regressions; H100 had `100` changes with `3` directional regressions. Protected-ID regression remained zero, but the strict future-effect lower CI stayed `0`, so the gate failed.

By action, the observed M3/M4 changes occurred in `AUTHORITATIVE_REASSIGN` and not in `RECOVER_IDENTITY`; neither action produced a true correct crossing. This does not establish that appearance memory can never help, but it rejects the current frozen mechanism as a confirmed future-effect route.

## 5. Spatial correction and candidate recovery

Stage 10 NO→M0 candidate recall (official future candidate stream):

- H20: `None` → `None` (Δ `None`)
- H50: `None` → `None` (Δ `None`)
- H100: `None` → `None` (Δ `None`)

Correction therefore helps short-horizon candidate availability but degrades the aggregate at H50/H100. That effect is separate from memory’s Mx–M0 increment and is not a demonstrated public-ID gain.

Stage 13 track-centric recovery accepted `5` proposals. R1 preserved the official candidate stream, but identity-error reduction remained `0` at H20/H50/H100 and no true crossings occurred. Recovery is therefore not promoted to a production branch.

## 6. Stage14 expansion policy and downstream gates

A CPU-only, replay-independent policy audit froze `40` events across `24` sequences, with action counts `{'ADD_NEW_IDENTITY': 8, 'ATOMIC_ID_SWAP': 8, 'AUTHORITATIVE_REASSIGN': 8, 'RECOVER_IDENTITY': 16}` and `7` known-failure exclusions. The adopted artifact is attempt4; earlier attempt1/attempt2/attempt3 selections are retained but not adopted. All selected interactions are explicitly `simulated_from_gt` and have no target public ID until a valid persistent prestate is available.

Because the repaired six-event evidence has zero surviving true M3/M4 crossing, the frozen expansion was retained as a reproducible policy artifact but not executed. Stage15 larger replay, Stage16 M3 confirmation preregistration, and Stage17 independent confirmation are `NOT_AUTHORIZED`/`NOT_RUN`. Executing them now would enlarge a synthetic sample after observing the treatment outcome without a surviving primary mechanism precondition.

Stage18 TrackEval is `NOT_RUN`: the available outputs are bounded event windows, not complete legal MOTChallenge sequence files. HOTA, AssA, IDF1, MOTA, Frag and standard full-sequence IDSW are therefore not reported as if measured.

## 7. Failures and preservation

- Stage 6/7 initial prestate failures, Stage 8 initial persistent replay failure, Stage 9 attempt-1 SAM3 hot-start failure, Stage 10 first analysis failure, and Stage14 attempt-1 quota-finalizer failure remain under `outputs/N72R4/attempts/`.
- Stage14 attempt3 is not adopted because it selected the explicitly forbidden `dancetrack0015` atomic candidate `773`; attempt4 excludes all four unresolved `772/773/774/796` candidates and is the only adopted expansion policy.
- No N36/N37/N72R3R1 artifact was overwritten. No `third_party/sam3` file, checkpoint, metric definition, or event protocol was changed.
- The environment emitted a non-fatal `osr_lib` namespace `.pth` warning during tests; the targeted suite still completed `23 passed` with zero test failures.

## 8. Authorization and next step

`production_authorized=false`, `training_authorized=false`, `calibration_authorized=false`, and `decoder_lora_authorized=false`. The next scientifically valid step is provenance-complete real human event tape with direct public-ID authority and candidate/native/local/global mapping evidence. Synthetic-from-GT events must not be relabeled as real human evidence. If a new synthetic association-interface probe is proposed, it needs a new frozen hypothesis and decision-boundary audit before event expansion.

## 9. External reference audit

The mechanism review records only publicly verifiable references and uses them as design context, not as evidence that this project passed its gate:

- [MOTIP](https://github.com/MCG-NJU/MOTIP) — audited runtime trajectory modeling and ID-decoder paths at commit `ffc0e905ac196a603027eca8d18fb0dff48c8bcc` (2026-07-30, Apache-2.0); conceptually relevant to trajectory-conditioned identity association, but no code was copied.
- [MeMOTR](https://github.com/MCG-NJU/MeMOTR) — audited query/memory update and motion paths at commit `eb7a177b9cbcb89742ec69b2545ab3af2ea31a80` (2025-10-15, MIT); conceptually relevant to persistent track memory, but no code was copied.
- Additional provenance-checked references (ByteTrack, BoT-SORT, TrackTrack, TrackEval, and InteractTrack) are catalogued in `outputs/N72R3R1/external_reference_audit.json`; their mechanisms were not treated as a substitute for the frozen InterMOT experiment.

## 10. Machine-readable files

- `outputs/N72R4/n72r4_final_gate.json`
- `outputs/N72R4/mechanism_rounds/round_01_assignment_diagnosis/results.json`
- `outputs/N72R4/mechanism_rounds/round_01_assignment_diagnosis/gate.json`
- `outputs/N72R4/stage_status/stage_14_adopted_attempt4_status.json`
- `outputs/N72R4/stage_status/stage_15_status.json` through `stage_18_status.json`
- `outputs/N72R3R1/n72r3r1_gate.json` and `outputs/N72R3R1/old_vs_new_comparison.json`

Report generated from the machine gate by `scripts/n72r4_finalize_gate.py`; all input hashes are recorded in the gate and round pre-change manifest.
