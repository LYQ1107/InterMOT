# N9 Research Log

## N49.0 N48 read-only gate diagnosis
Hypothesis: N48 repair2's zero accepted cells may be caused by output/training scale underfit rather than absent appearance signal, a data-axis defect, or a candidate ceiling.
Experiment: scanned all three frozen N48 runtime chains (24 events x 5 variants x 100 frames), summarized first gate rejection and raw/bounded residual, uncertainty, global-margin and valid-memory distributions, and independently recomputed row-major candidate/memory pairing, public-ID axis and explicit-NONE Hungarian assignments. Audited the frozen dataset pair labels/splits and checkpoint metadata without reading future GT or changing production code.
Observation: repair2 M2 has 89,000 finite-memory eligible cells but 0 accepted; first rejections are memory-invalid 83,519, global-margin>2 66,379, residual<0.05 22,621, hard/nonfinite 0, uncertainty gate 0 because no cell reaches it. Runtime repair2 residual median/P95 is 0.00184/0.01973 and uncertainty median/P95 is 0.50047/0.50302. R0 and legacy R1 have nonzero selected coverage, so the output-scale collapse is specific to repair2's corrected objective path.
Diagnosis: training underfit plus gate/output scale mismatch; data/feature mismatch and candidate ceiling are not supported. The no-write public-ID axis differs on 30 M2 frames as the retained active-universe memory-effect branch; write-baseline to write-plus axis and all row-major/Hungarian/GT-boundary contracts pass.
Change: none to N48 or production. Added isolated read-only audit `outputs/n49/diagnosis/n49_readonly_diagnosis.py`, JSON/log outputs, and `docs/N49_DIAGNOSIS_REPORT.md`.
Result: `PASS_DIAGNOSTIC_NOT_EFFICACY`; structural follow-up is justified only under a new frozen N49 protocol with runtime GT false and production authorization false.
Keep/Discard: keep the diagnosis and all N48 evidence; discard threshold/seed/metric tuning as a response. Do not claim human efficacy or authorize calibration/LoRA.
Next: freeze the N49 isolation protocol for an assignment-aware residual/risk training experiment, then run only contract smoke/regression before any GPU job.

## N49.1 Assignment-aware isolated follow-up
Hypothesis: the diagnosed repair2 underfit/output-scale mismatch can be tested with a pre-frozen assignment-critical weighting of the existing offline objective and a larger fixed optimization budget, without changing runtime thresholds, candidate streams, public-ID axes or production code.
Experiment: froze `outputs/n49/protocol.json` with N48 dataset/checkpoint hashes, N42 disjoint 14/3/4 sequence split, critical-cell mask, fixed weights, same output units and frozen gate, then built a labels/splits-preserving dataset sidecar. Ran py_compile, smoke and 12,000-frame contract regression; warm-started from the preserved repair2 checkpoint and trained on GPU0 for exactly 32 epochs with one complete accumulated objective step per epoch. Ran 24x5x100 simulated_from_gt paired replay, posthoc-only GT, independent integrity, H20/H50/H100 and equal-sequence bootstrap metrics.
Observation: stage 01/02/03/04 all PASS; checkpoint SHA256 `357a45642db50647e39eb35affb425ffe512991264d0c8fe6f58d6e4620b0f7b`; integrity checked 12,000 runtime and 24,000 source trace frames with zero failures. M2 incremental utility was 0.0 at H20/H50/H100, assignment changes and correct changes were 0, and untouched regression passed.
Diagnosis: the fixed assignment-aware weighting plus larger budget did not produce an exercised simulated incremental effect under the unchanged runtime gate. This is a strict negative/zero simulated result, not evidence that appearance has no signal and not human efficacy evidence. Real human tape and real SAM3 full-loop remain absent hard gates.
Change: only new `outputs/n49` artifacts and logs; N48, N42/N43/N47 and production interfaces were not changed. Preserved first actionable wrapper path failures under `outputs/n49/attempts`.
Result: `N49_SIMULATED_GATE_FAILED`; no calibration head, decoder LoRA, production MOT/OVMOT change, threshold scan, seed scan or metric change authorized.
Keep/Discard: keep complete N49 protocol, dataset, training, replay and integrity evidence; discard this structural variant as a route to positive effect under the current frozen contract.
Next: broader objective remains open but this isolated retry is exhausted for its declared structure. Obtain provenance-complete real human tape plus candidate-complete real SAM3 full-loop; do not relabel GT-derived artifacts or use this zero-effect run to authorize downstream training.

## N51.0 Corrected risk-contract isolation (in progress)
Hypothesis: N50's all-abstain result is still confounded by four concrete artifact-contract defects: duplicate oracle targets are labeled safe, critical-pair delta uses the current assignment instead of the strongest competitor, micro-batch gradients are not normalized by complete train-cell/train-pair denominators, and runtime scalar column 6 uses a row mean instead of the public-ID cell value. A fresh sidecar with explicit class-balanced unsafe-risk loss can test the diagnosis without changing the frozen candidate stream, public-ID axis, solver, gate, seed or production code.
Research plan: audit N50 and frozen source hashes; write fixtures for duplicate labels, signed-advantage sign/reference, full-denominator one-step gradient equivalence and row-major scalar[6]; freeze a new `outputs/n51/protocol.json`; materialize a fresh corrected dataset; run the fixed CUDA training; then run GT-free M0--M4 replay, independent integrity, and posthoc-only simulated metrics. Keep every failure and retain the hard real-tape/full-loop blocker.
Status: implementation and experiments pending; all N51 artifacts must remain `interaction_source=simulated_from_gt`, `not_real_human_evidence=true`, `runtime_future_gt_used=false`, and `production_authorized=false`.

## N51.1 Corrected contract experiment result
Hypothesis: correcting duplicate/ambiguous labels, strongest-competitor signed deltas, full-denominator one-step gradients and per-public-ID scalar[6], together with class-balanced unsafe-risk loss, would exercise the frozen risk gate on the same candidate stream.
Experiment: audited N50; froze `outputs/n51/protocol.json`; passed duplicate/sign/gradient/row-major fixtures; built a fresh 172,519-cell dataset; trained from scratch on GPU0 for exactly 32 epochs; ran 24×5×100 GT-free replay, independent integrity, then posthoc GT metrics.
Observation: 328 duplicate frames became unsafe/invalid with no safe cells and 2,173 critical pairs excluded; train safe/unsafe cells were 8,448/141,435. Training completed (best validation epoch 20; checkpoint SHA256 `48333234bc3eebd57ed159d5ab8fa0195d60d0e894840c0facf3c58b7f847036`). Integrity passed 12,000/12,000 frames, including 690,492/690,492 gate-reason cells. M2 selected 0 cells and had utility 0/correct changes 0 at H20/H50/H100; untouched regression passed and sequence-cluster CI was [0,0].
Diagnosis: the corrected structural path is valid, but the risk head remains near 0.46 for safe and unsafe eligible cells; fixed gate counts are memory-invalid 83,519, global-margin>2 66,379, delta<0.05 523, risk>0.35 22,098, selected 0. Validation/holdout risk unsafe AUC is 0.501/0.538. This is a strict simulated failure, not efficacy evidence.
Change: added only isolated `outputs/n51` scripts/artifacts and reports; no MOT/OVMOT production code, thresholds, seeds, metrics, splits, LoRA or human evidence changed. Preserved launch/integrity failure evidence.
Keep/Discard: keep N51 as corrected-contract negative evidence; discard any interpretation that class balancing alone calibrates the fixed gate. No calibration or decoder LoRA is authorized.
Next: perform a fixed representation/target separability audit before any separately frozen objective; real human tape and real SAM3 full-loop remain hard blockers.

## N51.2 Fixed representation separability diagnosis
Hypothesis: if the input appearance representation itself is non-informative, the N51 risk failure is a representation bottleneck; if it separates safe/unsafe cells while the learned head does not, the bottleneck is model/optimization alignment.
Experiment: without GT loading, threshold scans, seed scans or checkpoint selection, evaluated fixed candidate-memory cosine, base score and oracle-delta directions on the frozen train/validation/holdout splits.
Result: candidate-memory cosine unsafe AUC was 0.739 validation and 0.647 holdout, while N51 learned risk unsafe AUC was 0.501/0.538. The representation has separability in this fixed diagnostic, but N51's risk head did not use it under the frozen objective.
Keep/Discard: keep the representation signal and N51 failure evidence; do not relax the gate. Any model/feature objective follow-up must be a new frozen isolation protocol. Real human tape/full-loop and production authorization remain absent.

## N33.0 CCAM mechanism implementation
Hypothesis: known-ID human corrections can provide future identity evidence when written as target-scoped appearance anchors, while preserving K1 spatial state updates.
Change: added opt-in `AppearanceMemory`, future-only scoring, human/machine write APIs, and optional additive association terms; default path remains legacy.
Status: implementation and focused integration checks PASS in the provisioned project environment; no long run started.
Next: candidate-complete future replay is still required before any identity-learning claim or Stage-A full loop.

## N33.1 CCAM mechanism-first implementation result
Hypothesis: a human-confirmed public ID can receive an independently extracted ROI appearance anchor that affects only later association decisions, while native/geometry/motion/Hungarian behavior stays intact.
Change: strengthened `AppearanceMemory` with capped positive/negative banks, EMA prototype, quality/age/source/event provenance, reliability gating, finite-feature checks, serialization and reset; added additive score decomposition/audit; added module-level `human_evidence()` with box/mask ROI extraction and explicit unavailable/failure statuses; added post-commit ledger/event attachment; added strict candidate-tape validation and deterministic `memory_write=True/False` replay; added M0--M4 ablation driver with atomic artifacts.
Result: `tests/test_n33_ccam.py` passed 10/10; shared K1/N10 smoke tests passed 38/38; touched modules passed `py_compile`; M0 candidate audit, current-frame/future-frame boundaries, and a synthetic all-variant driver smoke passed. The default ablation artifact is `NOT_AVAILABLE` because no candidate-complete tape exists; all M0--M4 rows are `NOT_RUN`, and H20/H50/IDSW/re-correction/protected-regression/IDF1/HOTA are not computed.
Limitations: N32 selector identity feature coverage remains 0/689, so there is no valid identity-aware learning evidence. The repository is not a Git worktree, therefore root `git diff --check` returned the real `Not a git repository` error; `git -C third_party/sam3 diff --check` passed. The pre-existing `third_party/sam3/sam3/perflib/fused.py` modification was preserved. No val/test data, full MOT replay, training, or Stage-A multi-action loop was run.
Keep/Discard: keep the target-scoped non-parametric mechanism and audit boundaries; do not claim identity treatment effect, IDF1/HOTA, or deploy a selector until a real complete-candidate, multi-identity future tape is available.

## Iteration N9.0
Hypothesis: N8 sparse interaction has no gain because one correction does not persist (t+1 retention ~4%, time-to-next-error median 1 frame).
Experiment: N8 canonical 25 (already completed, reused as frozen baseline).
Failure/Observation: B1-B8 flat/slightly negative; unlimited dense correction strong (HOTA 80.57).
Diagnosis: bottleneck is future association persistence, not scheduling.
Change: N9 builds a learned tracklet-relinking association layer with human-confirmed identity memory.
Result: stage started.
Keep/Discard: keep diagnosis.
Next: P0 on DanceTrack train, ReID/SAM features, relinking benchmark, pairwise/set-level/human-conditioned models.

## Iteration N9.1
Hypothesis: P0 tracklet breaks (N8 119k) come from tid disappearing/reappearing, so segment-level relinking episodes will match the error stream.
Experiment: built segment/episode benchmark on val 0004/0005/0007.
Failure/Observation: only 2-3 episodes; P0 tids are all long-lived and the same identity flips between tids on adjacent frames (e.g., gid0 tid 4->2->1->4 at frames 48-50).  The real pattern is per-frame identity instability, not tracklet death.
Diagnosis: episode definition wrong for this backbone; need per-frame decision episodes.
Change: rewrote benchmark as per-frame decision episodes (identity memory vs current rows).
Result: 17-19k episodes per 3 val sequences; ReID cosine AUC 0.92, R@1 98.5% (val 3).
Keep/Discard: keep per-frame formulation; drop segment-only episodes.
Next: association overlay inference; SAM feature probe.

## Iteration N9.2
Hypothesis: per-frame association overlay with ReID/learned scorer keeps identity consistency and improves retention.
Experiment: N9Observer v1 with persistent canonical_map + greedy assignment; 3-seq runs (reid/pairwise/auto/proposed).
Failure/Observation: FN exploded (8993 vs 1352); HOTA dropped to 37-49; duplicate public ids caused row loss.
Diagnosis: persistent tid->pid map made multiple tids map to the same pid over time (all tids persist every frame); rows silently deduplicated in post.
Change: per-frame assignment (frame_assignment) + Hungarian + collision resolution.
Result: no row loss (FN ~1442) but identity consistency collapsed (AssA 18-27, IDSW 685-5339): memory drift from one wrong assignment cascades.
Keep/Discard: discard aggressive per-frame reassignment.
Next: conservative new-tid-only relinking.

## Iteration N9.3
Hypothesis: only genuinely NEW tids (absent in previous frame) should be relinked to known identity memories; this preserves baseline and enables recovery after misses/window boundaries.
Experiment: N9Observer v2 (new-tid gating) + stability bonus + margin-gated memory update; 3-seq runs.
Result: baseline preserved (HOTA 67.90 vs P0 67.92, no collapse); relinks 16-17 per 3 seqs at B8, anchor_used=1; retention at accepted events unchanged (~0-4% at t+1).
Diagnosis: P0 tids rarely disappear, so conservative relinking has almost no opportunity; anchors rarely fire; one-shot corrections still do not persist.
Keep/Discard: keep conservative safety; discard expectation of sparse-gain.
Next: full train split training + calibration gate; decide FAIL_PERSISTENT_IDENTITY_MEMORY vs association-only.

## Iteration N9.4
Hypothesis: SAM3 native per-object features (obj_ptr/maskmem_features) can be extracted via adapter without changing inference.
Experiment: three probe attempts on val 0004 (full-pass handle_stream_request, limited-window, backend.propagate then inspect inference_state).
Failure/Observation: full pass OOM (38GB on 40GB card); limited window crashes with zero-batch tensor error (known pinned-API limitation); propagate-then-inspect returns no retained per-object features in accessible state.
Diagnosis: pinned SAM3 does not expose stable per-object features through the adapter API; extraction requires deep third-party modification or unsafe memory patterns.
Change: none (third-party frozen).
Result: SAM_NATIVE_FEATURE_NOT_USABLE.  Continue ReID route.
Keep/Discard: discard SAM feature route; document in report.
Next: final training on full train split.

## Iteration N9.5
Hypothesis: full 30/10 training makes the learned set-level / human-conditioned association clearly beat ReID and N8.
Experiment: trained pairwise (5 epochs, cal_acc 1.0), set (4 epochs, cal_acc 81.2%), hcpim (fine-tune from set + anchor future objective, cal_acc 87.1%) on 30 train seqs; ran calibration observer (10 seqs) and final 3-seq runs.
Failure/Observation: retention identical across reid/pairwise/auto/proposed and equal to N8 (calibration B8 t+1 53.75%, t+3 40%, t+5 37.5%; TTE median 1).  Final 3-seq: auto/proposed outputs byte-equal to N8 (models never fire relinks); reid/pairwise add 11-17 relinks but HOTA flat (67.89-67.91 vs P0 67.92), IDSW slightly worse.
Diagnosis: (1) the hcpim/set models' logit threshold (0) never passes on real streams, so the learned route contributes nothing; (2) even when relinks fire (reid/pairwise), they do not change retention because P0 tids rarely disappear and the correction's persistence is already fully captured by N8's canonical_map; (3) aggressive per-frame reassignment remains the only way to act on P0's identity instability, but it drifts and collapses.
Change: none (conservative design already maximal; aggressive route rejected by evidence).
Result: FAIL_PERSISTENT_IDENTITY_MEMORY / FAIL_INTERACTION_SPECIFIC_GAIN; canonical 25 not run per protocol.
Keep/Discard: discard human-conditioned route on this backbone; keep benchmark + ReID feasibility numbers.
Next: detection-side interaction / trainable tracker; not N10 scheduling.

# N10 Research Log

## Iteration N10.0
Hypothesis: N9 failed because the association layer was a post-hoc patch on P0 native
identities.  If the final identity is produced by our own online association state
machine that reads observations, motion, history and human-confirmed state, one
correction should change future association trajectories.
Experiment: built anonymous ObservationTapes (all P0 rows + OSNet ReID, native tid as
cue only) for 40 train + 25 val; implemented online identity state machine
(birth/ACTIVE/LOST/REACTIVATED/TERMINATED), pairwise/set scorers, chunk-based
teacher-forced training, and N8-verified-error human interventions (hard bind +
anchor authority + positive/negative native constraints).
Result: pipeline runs; first rollout with untrained scorer fragments identities
(55 states on one sequence).
Diagnosis: absolute logit threshold 0 too permissive; native-tid flips of the P0
backbone cause repeated births without a continuity prior.
Change: native-continuity bonus (+3) for same native tid as identity's last match;
threshold -5; monotonic pid allocator.
Keep/Discard: keep native bonus (cue only); discard permissive threshold 0.
Next: full 30-seq training.

## Iteration N10.1
Hypothesis: a fully trained pairwise scorer plus native continuity is a stable
online AUTO associator comparable to P0.
Experiment: PairwiseMLP (512+512+12 -> 256) trained 4 epochs on 30 train seqs
(teacher-forced chunk walk, hard-event weights); cal decision R@1 = 0.965.
Rollout on 10 calibration seqs: AUTO-Pairwise HOTA 41.402 vs P0 41.355, AssA 34.330
vs 34.264, IDF1 45.837 vs 46.273, IDSW 772 vs 730.  ReID+motion heuristic baseline:
HOTA 41.566, AssA 34.619, IDF1 46.300, IDSW 730.  AUTO-Set collapsed (HOTA 16.666,
IDSW 20822; set R@1 only 0.172).
Diagnosis: pairwise AUTO is stable and within the 1-point AUTO gate; set-level
lightweight model does not learn the online set task with this feature layout.
Change: none for pairwise; keep set as failed comparison.
Keep/Discard: keep pairwise AUTO; discard set-level route.
Next: HUMAN state interventions.

## Iteration N10.2
Hypothesis: accepted N8 verified errors applied as identity-state interventions
(anchor authority, positive/negative native constraints) will persist beyond the
corrected frame.
Experiment: same pairwise checkpoint, HUMAN mode B1/B2/B4/B8 on 10 cal seqs;
same-event no-apply AUTO branches (human_auto) for causal comparison; zero-
intervention invariant checked (human_b0 == pairwise_b0, 10/10 byte-equal).
Result: retention at accepted events improves monotonically with budget
(t+1: B1 0%, B2 20%, B4 42.5%, B8 51.25%) vs AUTO branch (0/14.3/9.5%);
TTE mean 1.0/5.7/5.15/4.33 vs AUTO 1.0/1.25/2.19; median TTE = 1 everywhere.
Official TrackEval: HUMAN slightly degrades with budget (B8 HOTA 41.157 vs AUTO
41.402, AssA 33.913 vs 34.330, IDF1 45.537 vs 45.837, IDSW 819 vs 772, Frag 7253
vs 7227).
Diagnosis: interventions genuinely change the identity state and persist for the
corrected identity, but they also perturb other identities; aggregate official
metrics do not improve and the median time-to-next-error stays 1 frame.
Change: none (protocol stop condition met: median TTE still 1, no official gain).
Keep/Discard: keep state-intervention mechanism as evidence; discard expectation of
official-metric gain on this backbone.
Next: final report (FAIL_INTERACTION_SPECIFIC_GAIN); canonical 25 NOT run.

## Iteration N10.3
Hypothesis: removing the native-tid cue shows whether AUTO depends on P0 identity
continuity.
Experiment: AUTO-Pairwise with native_bonus=0 on 3 cal seqs.
Result: HOTA 44.877 vs 46.677, AssA 39.661 vs 42.682, IDF1 48.004 vs 52.535;
state count per sequence roughly doubles.
Diagnosis: the lightweight AUTO relies on native-tid continuity as a strong
short-term cue; without it the online state machine fragments.
Keep/Discard: keep native cue (documented as cue, not final identity).
Next: final report.

# N11 Research Log

## Iteration N11.0
Hypothesis: N10's "median TTE=1" may be a global-error artifact; correcting
identity may actually persist (event-level gain was real but collateral damage
was large).
Experiment: recomputed N10 logs with same-ID vs global TTE, same-ID retention,
target vs unrelated future errors (10 cal seqs; 80 accepted events at B8).
Observation: same-ID median TTE is also 1 (not only global), but HUMAN mean
same-ID TTE 4.3-5.7 vs AUTO 1.7-3.3; B8 same-ID t+1 51.3% vs AUTO 9.5%; target
errors per interaction 22.34 vs 26.26; unrelated errors per interaction 196.7
vs 86.8 (collateral ~110 extra errors/interaction).
Diagnosis: correction persists for the corrected identity, but the sequence
is dominated by back-to-back association errors (median TTE=1), and the
intervention perturbs other identities.
Change: none (audit only).
Result: N11 must test locality + temporal authority, not re-prove persistence.
Keep/Discard: keep mechanism evidence; discard global-TTE-only interpretation.
Next: deterministic Scope-v0 vs N10 Global.

## Iteration N11.1
Hypothesis: N10 collateral damage comes from global re-solve + permanent
native hard constraints; freezing non-scope AUTO assignments (Scope-v0) should
cut collateral IDSW/Frag without losing target persistence.
Experiment: implemented expiry-aware constraints, scope marking, local
constrained association (freeze non-scope assignments; solve scope subgraph);
ran local_perm_b8 on the same 10 cal seqs and compared with N10 Global B8.
Observation: local_perm_b8 output is byte-identical to global_b8 on all 10
sequences (HOTA 41.157 / AssA 33.913 / IDF1 45.537 / IDSW 819 / Frag 7253).
Diagnosis: post_rows are a re-association of the same detection rows; the
pairwise Hungarian already preserves unrelated assignments because one-to-one
matching of the same row set leaves no free variables for non-scope rows;
freezing non-scope assignments has zero effect.
Change: none.
Result: FAIL_LOCALITY_HYPOTHESIS for the spatial scope; learned scope predictor
has no mechanism basis.
Keep/Discard: discard learned scope predictor; keep Scope-v0 as a documented
negative result.
Next: temporal authority ablations (permanent vs native0 vs decay vs evidence).

## Iteration N11.2
Hypothesis: N10's permanent native constraints and hard human authority are
the collateral source; short-lived/soft authority keeps persistence while
reducing churn.
Experiment: ran local_native0_b8 (no native future constraint), local_native0
decay B1/B2/B4/B8 (hard=1, decay=8), local_native0_evidence_b8, and
local_decay memfreeze_b8 on the same 10 cal seqs; official TrackEval +
collateral audit.
Observation: native0 barely changes anything (41.152/33.912/45.536/823/7254);
decay reduces unrelated assignment changes (30-frame total 1727 -> 1275) but
also reduces same-ID persistence (B8 t+1 51.3->47.5, t+30 19.5->10.4; ID_BREAK
persistence collapses after frame 1); unrelated *errors* per event do not drop
(203.0 vs 196.7); official metrics never beat AUTO-P (best local B8 HOTA 41.297
vs AUTO 41.402, AssA 34.143 vs 34.330, IDF1 45.573 vs 45.837, IDSW 823 vs 772);
memfreeze is byte-identical to decay.
Diagnosis: the damage is not native-tid persistence; it is the density of the
underlying association errors (same-ID median TTE still 1 at B8) plus the fact
that target-persistent corrections shift identities across the sequence in ways
official metrics count as switches; TRUE_MISS_NEW corrections mostly do not
hold (28.9% at t+1, lower under decay).
Change: none (two diagnosed mechanism iterations completed; gate failed).
Result: FAIL_INTERACTION_SPECIFIC_GAIN; canonical 25 NOT RUN; three-seq sanity
NOT RUN.
Keep/Discard: discard learned temporal controller; keep decay/evidence numbers
as ablations; keep mechanism implementation for future detection-side work.
Next: N11 final report; then detection-side interaction or trainable tracker.

# N12 Research Log

## Iteration N12.0
Hypothesis: a sparse human correction can teach the tracker (not just patch
the output) if a lightweight adapter is updated on the current-frame correction
and the update improves future frames.
Experiment: audited pinned SAM3.1 (commit 4cbac146c1b5a1e3a7f5c6a894901090b4dfd65b):
built the standalone tracker model and remapped checkpoint keys
(`tracker.model.*` + `detector.backbone.vision_backbone.*`); 0 missing / 0
unexpected keys.  Implemented LoRA injection (Linear + Conv2d) on the
decoder/memory surface (107 modules, ~0.63M params at rank 8) and a CFA
single-object episode runner (box->points/mask prompt, propagation, GT IoU).
Observation: official propagate/add_prompt paths are @torch.inference_mode;
the underlying `_run_single_frame_inference` / `track_step` / `forward` are
trainable.  bf16 autocast is required for official inference; the fused
`addmm_act` path forbids gradients through the backbone MLP.
Diagnosis: the standalone tracker model is not the supported entry (its
backbone is None in the official stack); single-object propagation drifts to
full-frame within 2-3 frames.
Change: switched to the project's `Sam3Backend` (official full multiplex
pipeline) for future-frame propagation.
Keep/Discard: keep LoRA + loader + runner as infrastructure; discard the
standalone-tracker propagation route.
Next: backend-based CFA baseline.

## Iteration N12.1
Hypothesis: in the official full pipeline, one human box at frame t should
initiate tracking of the corrected identity for future frames (no-update
baseline).
Experiment: Sam3Backend + raw response parsing; box prompt at TRUE_MISS_NEW
event (dancetrack0074 frame 6, gid 1), propagate 15 frames, Target
Recall@1/3/5/10/30 vs GT.
Observation: the project's `_parse_outputs` drops add_prompt responses whose
outputs are a dict of arrays; a permissive parser recovers them.  However, the
identical add_prompt call is non-deterministic on this event: one run returned
5 detections (out_obj_ids len 5), another returned 0 (empty).  When 0, future
frames have no target output and recall is 0.0 at every horizon.
Diagnosis: TRUE_MISS_NEW box prompts are unreliable at the detection level on
DanceTrack; this is the same detection-gap failure N8-N11 measured, now at the
SAM3 prompt level.
Change: none yet (measurement recorded).
Keep/Discard: keep backend CFA runner + raw parser; the no-update baseline must
be collected over many events to separate prompt failure from propagation
failure.
Next: run CFA baselines on a batch of events; then implement the differentiable
inner update on the full pipeline (detector features + track_step with grad +
reset/re-prompt) and compare update vs no-update.

## Iteration N12.2 (current checkpoint)
Status: N12 infrastructure complete for the no-update baseline; the update
branch (online LoRA on the full pipeline) and meta-training are the next
implementations.  Key early findings:
- SAM3.1 checkpoint loads losslessly with the remap (0/0).
- Lightweight LoRA on decoder/memory is injectable and runs under bf16 autocast.
- The official full pipeline tracks a box-prompted object on the bundled
  example video (frames 0-5 stay localized).
- On DanceTrack TRUE_MISS_NEW events, box-prompt detection is unstable
  (0 vs 5 detections across identical calls), so CFA needs event-level
  prompt-success filtering or a detector-side interaction.
- No future GT is used in any inner update; future GT is only for offline eval.

## Iteration N12.3
Hypothesis: box-only prompts may be why TRUE_MISS_NEW events fail at the prompt
layer; the official add_prompt supports text+box jointly.
Experiment: added text="person" to the CFA prompt and reran 16 events
(2 per sequence x 8 calibration sequences).
Observation: prompt success 16/16 (box-only was 1/16).  No-update target
recall: mean 1.0 @ t+1, 0.83 @ t+3, 0.78 @ t+5, 0.74 @ t+10, 0.47 @ t+30;
several events lose the target after 2-5 frames (0075 gid4: 1.0/0.33/0.2/0.1/0.04;
0096 gid2 similar; 0082 gid16: 0 everywhere).
Diagnosis: the human box is not sufficient; the concept text is required for
the detector to produce the object.  This is a detection-layer gate, matching
N8-N11.
Change: CFA prompt now uses text+box.
Keep/Discard: keep text+box as the fair no-update baseline.
Next: online-LoRA update on headroom events.

## Iteration N12.4
Hypothesis: a lightweight LoRA update on the correction frame should improve
future target recall on events where no-update loses the target.
Experiment: shadow standalone tracker differentiable inner update (mask-in-box
+ objectness), copy LoRA back to the official pipeline, reset + re-prompt +
propagate; tested 0075/0096/0082/0086/0080/0074 with rank 8-16, 2-10 steps,
lr 1e-3-1e-2.
Observation: update runs in 0.14-0.70 s but is completely neutral: every delta
recall = 0.0 and branch-to-branch box IoU = 1.0 with 0 changed frames on all
events and configs.
Diagnosis: initial LoRA targets (sam_mask_decoder/maskmem_backbone/transformer)
do not influence the correction-frame interactive SAM head.  After adding
interactive_sam_mask_decoder + interactive_sam_prompt_encoder and a coverage
term, the shadow model's mask output DOES change (full-frame -> slimmer mask,
loss 0.39 -> 0.22), but the official text+box pipeline's future MOT outputs
still do not change.
Change: none that produces an effect on the official output.
Keep/Discard: discard tracker-head online LoRA as a causal adaptation surface
for the SAM3.1 text+box MOT pipeline (detector/association dominated).
Next: decide feasibility gate (candidate: FAIL_VISUAL_ADAPTATION_FEASIBILITY).

## Iteration N12.5 (conclusion)
Visual Adaptation Feasibility Gate: FAIL for the proposed lightweight
identity-selective tracker-head adaptation route.
- The human box alone is unreliable (1/16 prompt success); text+box is needed
  (16/16), i.e., the correction is gated by the detector layer.
- When the prompt succeeds, no-update already tracks t+1 at recall 1.0; several
  events decay to 0.03-0.47 by t+30.
- Online LoRA on the tracker head changes the shadow interactive SAM output but
  not the official pipeline's future output (all deltas 0).
- The effective causal surface would be the detector itself (detector
  fine-tuning), which is outside the lightweight identity-selective scope and
  was not pursued per protocol.
- No meta-training, identity-selective adapter, preservation, sparse MOT, or
  canonical 25 is justified without a measurable adaptation effect.
Next: write N12_FINAL_REPORT.md (negative result) + update final report/gates.

## Iteration N13.1
Hypothesis: a human `person+box` correction can become a persistent
detector-side query by re-injecting predicted boxes as per-frame geometric
prompts during official propagation (EXTERNAL_PERSISTENT_PROMPT_ROUTE).
Experiment: built a streaming PDR runner that (a) forces full-VG propagation
(action_history cleared, batched grounding disabled at runtime), (b) sets
`per_frame_geometric_prompt[t+k]` before each frame, (c) invalidates the
official detector's pre-fetch cache so the prompt is actually re-encoded, and
(d) probes raw detector bbox/scores pre-association.
Observation: on dancetrack0075/0082/0096 TRUE_MISS_NEW events, the injected
box prompts ARE delivered to the detector encoder (nbox=1) but the raw 200
detector queries are byte-identical with/without boxes, on the prompted frame
and future frames (max box diff = 0, max score diff = 0).  Text-only vs
text+box also byte-identical.  N12's 1/16 vs 16/16 was a text-presence effect,
not a box effect.
Diagnosis: the human box is causally inert in the official SAM3.1 detector
when the text prompt is present; the persistent external box-prompt route has
no mechanism to change official MOT output.  The causal levers are text
conditioning (already global) and detector weights.
Change: none for the box route.  Gate A of the detector-adapter fallback is
met (Oracle future-box prompt still fails), so proceed to detector LoRA.
Keep/Discard: keep the PDR runner (used for the adapter evaluation); discard
the box-prompt persistence hypothesis for SAM3.1.
Next: lightweight detector-side LoRA on decoder/dot-product scoring, prove it
changes official outputs, evaluate PDR Target Admission / Delivered Recall.

## Iteration N13.2
Experiment: full-VG PDR baselines on 16 TRUE_MISS_NEW calibration events
(same official pipeline as the adapter; one-shot text-only vs per-frame GT-box
prompts).
Observation: one-shot delivered recall 0.938@1 / 0.812@3 / 0.788@5 / 0.750@10
/ 0.628@30; admission@30 0.730; mean false capture 0.197.  Oracle future-GT
box prompts are byte-identical to one-shot on all 8 probed events (0 changed
frames) — box prompts remain causally inert in full-VG mode too.
Diagnosis: the earlier N12 baseline (partial mode) understated one-shot
delivered recall (e.g., 0075 gid4: 0.037 -> 0.259 in full-VG); comparisons
must use the full-VG A0.
Keep/Discard: keep full-VG A0 as the N13 baseline; discard per-frame box
prompt persistence.
Next: evaluate the detector LoRA adapter against the full-VG A0.

## Iteration N13.3
Hypothesis: lightweight online LoRA on detector decoder + dot-product scoring
(trained on the human frame, 10 steps, lr 1e-3, rank 8, alpha 4) improves
future target recovery.
Experiment: 16 events, fresh process per run (avoids inference-mode cache
poisoning), train on frame t only, then official full-VG PDR eval.
Observation: training works (loss 0.095 -> 0.028, lora_l2 ~86, logits change),
and outputs DO change on 6/16 events.  But delivered recall is worse than A0
at every horizon: 0.812@1 (vs 0.938), 0.729@3 (0.812), 0.713@5 (0.788),
0.706@10 (0.750), 0.587@30 (0.628); mean delta@30 -0.042 (3 improved, 3
degraded, 10 flat).  Gentler configs (steps 2/5, lr 3e-4, preservation loss)
still collapse easy events (0083 gid1: 1.0 -> 0.03 with 5 steps) or flip
0096 gid5 (+0.2 -> -0.333).
Diagnosis: the adapter is causally active but not robust: it can improve hard
long-horizon events (0096 gid2/gid5, 0086 gid3) but frequently degrades
early recall and easy events; no config gives a reliable net gain.
Keep/Discard: discard online detector LoRA as a net-positive mechanism;
record it as causally active but unstable.
Next: finalize N13 as FAIL_DETECTOR_CAUSAL_ROUTE with full honest report.

## Iteration N14.1
Hypothesis: replacing a reserved slot of the detector decoder query_embed
with a per-identity dynamic embedding is on a real causal path (raw detector
candidates and official rows can change), and an empty/unchanged bank is
byte-identical to the official pipeline.
Experiment: causal smoke, 3 TRUE_MISS_NEW events (0074 f6 g1, 0083 f1 g1,
0096 f1 g2), 4 branches x 3 future frames, full-VG mode, instance-level
_run_decoder patch (no third-party source edit).  Branches: O official, I
identity control, R fixed random query, C ROI-pooled query written at the
human frame.
Observation: O==I byte-identical on all 9 frames (raw and official).  R
changes raw on 9/9 and official rows on 7/9.  C changes raw on 6/9 but no
official row changes in the 3-frame window.
Diagnosis: dynamic pre-decoder query injection is causally active; a raw ROI
query is too weak without training.  The patch machinery is a clean no-op with
an empty bank (zero-interaction equivalence holds).
Change: scripts/run_n14_query_causal_smoke.py, outputs/n14/query_causal_smoke.csv,
docs/N14_QUERY_CAUSAL_SMOKE.md.
Keep/Discard: keep the pre-decoder slot injection as the primary architecture
path; keep the smoke harness for regression; discard raw (untrained) ROI query
as sufficient for admission.
Next: Persistent Identity State + HumanWriteEncoder (frozen decoder), episode
dataset, PIR vs N13 one-shot A0.

## Iteration N14.2
Hypothesis: a human-seeded persistent detector query (dynamic query slot +
dynamic reference + per-slot box/score head) improves PIR vs one-shot without
changing B0.
Experiment: causal smoke passed (empty bank byte-identical; random query
changes raw 9/9 and official 7/9).  Trained HumanWriteEncoder + SlotHeadAdapter
(frozen decoder) on 6 sequences / 47 episodes / 564 samples (366 pos, 198
cross-identity neg), 8 epochs.
Observation: training score separation improves (pos 0.28->0.99, neg 0.27-0.93
then down to 0.58); PIR delivered@30 0.666->0.688 (+0.021; 5 better / 3 worse),
false capture 0.161->0.154; big gains on 0086 gid1 and 0096 gid2.  Preliminary
2-identity specificity: q0 correct, q1 not yet.
Diagnosis: the score head needed (a) a direct q input (decoder-path gradients
too weak), (b) cross-identity negatives matched at the other person's ROI
(earlier negatives were degenerate: same ROI as positives).  Box = human
reference in F0; learned box delta is unstable and deferred.
Change: sam3_intermot/persistent_identity/{head_adapter,injection}.py,
scripts/{train_n14_query,run_n14_pir,run_n14_specificity}.py,
outputs/n14/pir_results.csv, docs/N14_QUERY_PIR_MILESTONE.md.
Keep/Discard: keep query+ref injection, similarity score head, cross-identity
negatives; discard learned box delta in F0, bf16 training for this path.
Next: train on 12+ sequences with harder negatives; re-run specificity and PIR;
then multi-query and sparse budgets.

## Iteration N14.3
Hypothesis: contrastive (InfoNCE) training with cross-identity hard negatives
improves identity specificity; rule-based query renewal (update reference with
the delivered box) extends the F0 head over long horizons.
Experiment: rebuilt manifest with 706 positive samples x 4 nearest-other
negatives + 14 absence rows; trained v5 (12 seqs, hidden 1024, 20 epochs,
contrastive weight 10, tau 0.05) using a fast path that skips the decoder
(the F0 slot head only uses q + future ROI, so decoder gradients are
unnecessary for training).
Observation: training separation improved to pos 0.70 / neg 0.095 (epoch 19);
specificity probe improved for q0 (2.67 vs 1.35) but q1 still prefers the
wrong identity (2.94 vs 2.36).  Official PIR with v5 was worse than v4:
delivered@30 0.624 (FC 0.182) vs A0 0.666.  Slot-log inspection showed the
patched slot head is bypassed on the first propagated frame because that
detector chunk is prefetched before the patch is installed; forcing
invalidation (and rule-based renewal) made results much worse (delivered@30
0.331, FC 0.266; several easy events collapsed to 0).
Diagnosis: (1) F0 slot head fires on many negatives (specificity still
partial); (2) the official pipeline's NMS/continuity makes the extra
slot candidate interact destructively once it is strong; (3) renewal without
reliable identity verification corrupts the reference.  The best official
artifact remains v4 (no renewal, no forced invalidation): delivered@30
0.685 vs A0 0.666 (+0.019), FC 0.154 vs 0.161.
Change: scripts/run_n14_pir.py (slot log + renewal experiments),
outputs/n14/pir_slot_log_v5.jsonl, logs/pir_v5*.log, models
human_write_encoder_f0_v5.pt; restored outputs/n14/pir_results.csv to v4.
Keep/Discard: keep contrastive hard-negative training for specificity;
discard naive rule-based renewal and forced prefetch invalidation for F0.
Next: only proceed to multi-query/sparse budgets after either (a) threshold
calibration of v5 recovers PIR, or (b) a verified renewal/reference update
with identity gating; otherwise report FAIL_IDENTITY_SPECIFICITY /
FAIL_PERSISTENT_QUERY_GAIN with v4 as PASS_WITH_NOTES.

## Iteration N14.4
Hypothesis: the prefetch bypass and the renewal failure are confounded; a
corrected causal path (query registered before add_prompt) must be separated
first; then the head's box must be dynamic (learned) instead of static.
Experiment: run_n14_prefetch_audit.py traces run_backbone_and_detection calls
(frame, patch_active, slot score/box, raw hash) for A0/v4/v5 x E0/E1, no
renewal; then trained v6 = v5 contrastive score + learned bounded box delta
(box = ref + 0.5*tanh(box_net), supervised with future GT boxes).
Observation: E0 t+1 is served from the add_prompt-cached chunk (frozen slot);
E1 makes the slot head active at t+1.  Under E1, v4/v5 collapse (delivered@30
~0.0 on 4 events) because the static reference box locks the delivered
trajectory.  v6 restores tracking: delivered@30 0.565 vs A0 0.498 (4 events),
0096 gid2 0.067->0.4; FC 0.347 vs 0.330.  Training: pos 0.70 / neg 0.10; box
IoU 0.935 @t+1 .. 0.667 @t+8.
Diagnosis: static-ref head is the root of the E1/renewal collapse; a learned
box delta is necessary; the corrected causal path must be E1 for all future
runs.  v4 E0 PIR gain is partly a bypass artifact (historical).
Change: injection.py logit+box head (ref+delta), head_adapter.py box_net,
train_n14_query.py (v6), run_n14_prefetch_audit.py, outputs/n14/prefetch_*.
Keep/Discard: keep E1 as CORRECTED_CAUSAL_PATH; keep v6 representation;
discard static-ref head and forced invalidation + naive renewal.
Next: full 14-event PIR on E1 with v6; v6 score distribution + stable-region
calibration; then identity verifier (G_id/G_upd) and verified renewal.

## Iteration N14.5
Hypothesis: a simple output gate (identity score threshold + anchor IoU) can
prevent the slot candidate from derailing delivered continuity on easy events
while keeping hard-event gains.
Experiment: full 14-event E1 run for v6 (learned box delta) and v6 with gate
(score>=0.7, IoU(slot box, anchor)>=0.25); no renewal.
Observation: v6 E1 delivered@30 0.516 vs N13 A0 0.666 (-0.150; 2 better / 5
worse); gated v6 0.509 (no recovery).  Early recall drops on both (0.929 ->
0.714).  Gains only on 0096 gid2 (0.033->0.4) and gid5 (0.733->0.767);
collapses on 0080 gid5 (1.0->0.0), 0082 gid11 (1.0->0.567), 0086 gid3
(0.967->0.0).  Gate did not change the pattern.
Diagnosis: the collapse is not from low-confidence or off-anchor fires; the
extra slot candidate itself disrupts _best_delivery/NMS continuity even when
it passes the gate.  A single-slot head has not produced a net official PIR
gain on the corrected causal path (E1).  The v4 E0 gain is historical and
partly a prefetch-bypass artifact.
Change: gate params in injection.py + audit driver; outputs/n14/prefetch_*.
Keep/Discard: keep E1 as corrected path and v6 representation (best on hard
events); discard simple score/IoU gate as insufficient.
Next: identity verifier with G_id/G_upd + verified renewal (the planned
ICLR-level mechanism); until then N14 official status cannot be PASS.

## Iteration N14.6
Hypothesis: the human query must run as a non-destructive SHADOW candidate
(never enters normal NMS/_best_delivery) and only commit selectively; an
oracle commit measures the upper bound.
Experiment: run_n14_shadow.py computes the v6 shadow candidate separately
from the official branch (fixed human-anchor reference, no renewal), then
simulates oracle commit (GT-only: commit iff AUTO wrong AND shadow box
correct) on the 14 calibration events.
Observation: S1 (shadow never commit) is byte-identical to A0 on all 14
events x 32 frames (raw hash 32/32); easy-frame preservation 1.0 (259/259);
zero unnecessary interventions.  But the shadow candidate itself is useless
on the calibration sequences: shadow IoU with GT = 0.0 at every gap (n=14
per gap) and identity scores mostly ~0.003-0.005 (mean 0.36 driven by a few
outliers).  Oracle selective commit does NOT beat A0 (delivered@30 0.619 vs
0.666; 3 better / 4 worse).
Diagnosis: v6 (trained on 12 sequences) does not generalize to the 8
calibration sequences; both its identity score and learned box delta fail
out-of-training-distribution.  The earlier v6 E1 gains were produced by the
decoder-query injection on frozen outputs, not by the v6 head.  The shadow
architecture (S1) is validated; the candidate quality must be fixed by
training on the full non-calibration training set before G_commit can be
judged.
Change: run_n14_shadow.py + outputs/n14/{shadow_equivalence,commit_dataset,
oracle_commit,intervention_utility,easy_frame_preservation}.csv.
Keep/Discard: keep shadow isolation (S1) and oracle-commit methodology;
discard v6 as the generalizing candidate representation.
Next: rebuild manifest with all 32 non-calibration sequences; train v7
(same architecture); re-run shadow oracle and E1 PIR; then learned G_commit
only if oracle clearly beats A0.

## Iteration N14.7 (v7-v9 diagnostics)
Hypothesis: training on all 32 non-calibration sequences fixes shadow
generalization; then oracle commit should be re-evaluated.
Experiment: v7 (32 seqs, 25 epochs, hidden 1024, InfoNCE+box delta).  Shadow
oracle still showed 0/420 shadow-correct and score ~0.0 on calibration
events.  Debug prints revealed the encoder output q was UNBOUNDED
(||q|| ~ 6.5M-8.3M); box_net saturated at tanh(0.5) and the score collapsed.
Fixed q to unit norm in HumanWriteEncoder (v9) and normalized q/roi inside
the adapter (v8); shadow scores improved only to ~1e-7-0.02 and dbox still
saturated at [0.5,0.5,0.5,-0.5].
Diagnosis: two independent training/inference mismatches: (1) unbounded query
embedding magnitude (fixed by unit-norm q); (2) the box delta was trained on
the ROI at the future GT box, but at inference the shadow uses the ROI at the
fixed human reference -> out-of-distribution -> saturated deltas.  The score
also needs the candidate ROI at the predicted box.
Change: SlotHeadAdapter now takes (q, roi_cand, roi_anchor, ref): score from
roi_cand, box from roi_anchor; training uses GT-box ROI as the candidate
(teacher forcing) and reference ROI as the anchor; shadow computes the box
from the anchor ROI then scores the ROI at the predicted box.
Keep/Discard: keep unit-norm q and decoupled cand/anchor ROIs; discard raw
unbounded query embeddings and GT-ROI-only training for the box head.
Next: v10 with the corrected protocol; re-run shadow oracle; then decide
learned G_commit.

## Iteration N14.8 (final)
Hypothesis: with v10 (unit-norm q, normalized adapter, decoupled
candidate/anchor ROIs, 32 training sequences, corrected box protocol), the
shadow identity score should transfer to the 8 evaluation sequences.
Experiment: direct probe on dancetrack0074 f7/f8/f10: (a) score at the anchor
ROI (target present, ref-IoU 0.67-0.86) = 0.0; (b) score at the GT-box ROI
(teacher-forcing candidate) = 0.0.  Shadow oracle on all 14 events:
shadow-correct 0/420; oracle delivered@30 0.619 vs A0 0.666.  Box_net
checkpoint weights grow without bound over training (ep0 max 1.15 -> ep24 max
24.1) and saturate at inference, explaining earlier box collapse; epoch-mean
IoU masked the late-epoch degeneration.
Diagnosis: the learned single-frame ROI identity representation does not
generalize to unseen DanceTrack sequences after 5 representation iterations
(v5-v10).  The identity score is unusable for G_id/G_commit; verified renewal
cannot be validated.  S1 (shadow never commit) is byte-identical to A0 and
easy-frame preservation under oracle abstention is 1.0, so the shadow
architecture itself is sound.
Change: docs/N14_FINAL_REPORT.md; stage_gate N14_FINAL_STATUS =
FAIL_IDENTITY_SPECIFICITY.
Keep/Discard: keep the causal query path and shadow isolation as positive
results; discard the MLP identity verifier branch.
Next: stronger pretrained appearance encoder or motion/geometry-guided
recovery; otherwise close the persistent identity query branch.

## Iteration N15.0
Hypothesis: a strong pretrained identity backbone can replace the failed
SAM3-native identity representation from N14 and generalize sequence-disjoint.
Experiment: built Human Seed Identity Benchmark (76,199 crops, 19,880
query-delta tasks, 30 train / 10 calibration, val25 locked) and extracted
OSNet (N9 cache), CLIP-ReID ViT-B/16 (occurra HF mirror pth, official
Syliz517 config), DINOv2 ViT-B/14 (timm).
Observation: CLIP-ReID raw is best on calibration: R@1 0.7615, R@5 0.9732,
AUC 0.9060, FPR@1 0.2385, FPR@95TPR 0.6586, margin 0.0192; OSNet 0.7397,
DINOv2 0.6657.  NFC (Pose2ID, CVPR 2025) hurts on small candidate windows.
Diagnosis: pretrained identity features do generalize to unseen DanceTrack
sequences; the N14 0-score collapse is a representation problem, now fixed.
Change: docs/N15_IDENTITY_BACKBONE_AUDIT.md; scripts build/extract/bakeoff.
Keep/Discard: keep CLIP-ReID as IDENTITY_BACKBONE_V1; discard DINOv2/NFC as
primary.
Next: Human Anchor H_i -> I2Q -> shadow -> oracle commit.

## Iteration N15.1
Hypothesis: Linear I2Q (H_i -> detector query Q_i) injected through the N14 E1
path can make the frozen decoder slot output target-specific.
Experiment: trained LinearI2Q (1024 hidden, 3 epochs, 1200 samples) with
decoder-slot supervision (score BCE + box L1); probed slot 199 output on
dancetrack0074 f7 with trained/zero/random/static queries.
Observation: slot score 0.0084 (trained), 0.0048 (zero), 0.0118 (random),
0.0161 (static query embed); slot box never matched GT (IoU ~0); training IoU
stuck at 0.004, score 0.013.
Diagnosis: the frozen class/box head of slot 199 is dead for any injected
query (even the static query embedding scores 0.016); the I2Q cannot steer
the decoder slot.  This is FAIL_IDENTITY_TO_QUERY for the injected branch.
Change: scripts/train_n15_i2q.py, scripts/debug_n15_slot.py.
Keep/Discard: discard injected-decoder-query branch as the candidate source;
keep the pretrained anchor for candidate selection.
Next: identity-conditioned shadow selection + oracle commit.

## Iteration N15.2
Hypothesis: ranking official detector candidates by CLIP cosine with H_i and
oracle-committing only when AUTO is wrong gives a positive upper bound.
Experiment: 14 N13 calibration events x 30 frames; official AUTO + per-frame
shadow selection; oracle commit simulation (GT-only).
Observation: shadow_correct 159/434 (36.6%); AUTO-wrong frames 163, but only
6 have a shadow-correct candidate (3.7%) -> the target identity is almost
never in the detector candidate set when AUTO fails.  Oracle delivered@30
0.6328 vs A0 0.6186 (+0.0142; 3 improved / 0 degraded); @1/3/5/10 unchanged.
Diagnosis: oracle upper bound is now non-degrading (N14 was below A0), but
tiny; recovery is limited by detection-side candidate availability, not by
identity scoring.
Change: scripts/run_n15_selection_oracle.py; outputs/n15/selection_*.csv.
Keep/Discard: keep oracle methodology; keep CLIP selection as shadow state.
Next: learned/rule G_commit.

## Iteration N15.3
Hypothesis: a simple rule gate (G_id cosine >= tau, detector score >= s0,
AUTO uncertain) can realize most of the oracle bound safely.
Experiment: online rule commit on the same 14 events (tau=0.90, s0=0.30);
commits update the delivered trajectory.
Observation: 106 commits, 15 beneficial, 91 harmful; rule delivered@30 0.3186
vs A0 0.6186 (-0.30; 0 improved / 10 degraded).  At tau=0.95 only 2 of 10
AUTO-wrong frames are recoverable while 13 false positives occur on
AUTO-correct frames; tau=0.96 recovers 0.
Diagnosis: cosine margin between same/cross identity in DanceTrack is too
small (pos 0.95 vs neg 0.93; FPR@95TPR 0.66) for a safe commit gate; the
oracle's small upper bound cannot be realized without harming easy frames.
Change: scripts/run_n15_rule_commit.py; outputs/n15/rule_commit_*.csv.
Keep/Discard: discard rule commit; keep threshold analysis as evidence.
Next: verified renewal and multi-query offline checks; final decision.

## Iteration N15.4
Hypothesis: verified renewal (renew anchor only when G_id verifies) improves
long-horizon retrieval; multi-query anchors compete cleanly.
Experiment: offline delta=30 retrieval on calibration using CLIP features:
fixed H(t) vs verified H(t+10) (cos>=0.90) vs naive renewal vs averaged.
Also pairwise swap/conflict at delta=5.
Observation: fixed R@1 0.5129; verified 0.5520; naive 0.5716; averaged 0.5757.
Multi-query d5 swap acc 0.7944, conflict 20.0% (CLIP).  Verified renewal helps
+3.9pp but the simulation uses the GT positive crop as the observation; with
real unverified candidates the small cosine margin implies high false-accept
(FPR@95TPR 0.66), so online verified renewal would inherit commit-gate risk.
Diagnosis: renewal is a real long-horizon lever only if G_id precision
improves; with current representation it cannot rescue the failed commit gate.
Change: scripts/run_n15_renewal_multi.py; outputs/n15/renewal_metrics.csv,
multi_query.csv.
Keep/Discard: keep verified-renewal as future component; discard as the
current net-gain mechanism.
Next: final report and stage gate.

## N15 Final Status
FAIL_SPARSE_NET_GAIN (primary).  Sub-statuses: IDENTITY_GENERALIZATION PASS
(CLIP-ReID); FAIL_IDENTITY_TO_QUERY (injected slot); ORACLE_COMMIT
PASS_WITH_NOTES (+0.0142@30 upper bound); FAIL_COMMIT_GATE (rule harmful);
MULTI_QUERY PASS_WITH_NOTES; VERIFIED_RENEWAL NOT_VALIDATED_ONLINE.
Canonical25 and second-dataset eval NOT RUN (sparse gate failed).

## Iteration N16.0
Hypothesis: the N15 candidate-availability bottleneck (3.7% recoverable
AUTO-wrong frames) generalizes to a large scale; a target-conditioned
re-detection decoder can create the missing candidates from SAM3 features.
Experiment: built HCRED (675,668 episodes; 30 train / 10 calibration) from
P0 AUTO outputs + GT; generic-candidate-miss rate measured per delta.
Observation: generic SAM3 candidate missing in 20.5% of train frames and
34.1% of calibration frames (target present), stable across deltas 1..60.
N15's 3.7% was the AUTO-wrong subset; the candidate-recall ceiling is much
larger (~24% overall).
Diagnosis: candidate creation has substantial headroom (0.66-0.80 recall).
Change: outputs/n16/hcred_manifest.csv, generic_miss_stats.csv; scripts/
build_n16_hcred.py; docs/N16_PLAN.md; method/dataset audits.
Keep/Discard: keep HCRED; P0-based miss labels are the training signal.
Next: HCRD-v0 overfit -> subset training -> HCC eval.

## Iteration N16.1
Hypothesis: a small target-conditioned transformer decoder (CLIP-ReID anchor
+ SAM3 ROI -> target tokens, cross-attend SAM3 1008x1008 encoder features,
K=4 proposals) can localize a human seed at future frames.
Experiment: implemented RecoveryDecoder (2 layers, 4 heads, d=256); fixed
pos-embed layout (memory/pos both [n_tokens,bs,d]); 2-sample smoke then
100-sample overfit (dancetrack0001/0002, 6 epochs).
Observation: decoder runs at full 1008x1008 features (5184 tokens) with
~5MB feature cache per frame; base SAM3 model uses ~17GB, fits in 40GB.
Diagnosis: training is encoder-bound (~1-2s/frame) and was dominated by
backend video reloads when samples bounced between sequences; grouping by
sequence fixes the hot loop.
Change: sam3_intermot/recovery/recovery_decoder.py; scripts/train_n16_
recovery.py (grouped sampling, LRU 96); eval_n16_candidate_creation.py.
Keep/Discard: keep grouped training loop.
Next: overfit result -> train subset -> HCC metrics.

## Iteration N16.2
Hypothesis: a small transformer decoder (target tokens cross-attend full
features) can create missing candidates.
Experiment: HCRD-v0 (4 query tokens, 2 layers) trained on 100 samples; loss
stuck at 4.92, IoU 0.05; even lr=1e-3 did not move (identical per-sample
losses).  Synthetic data trained fine, so the bug was real-data-specific.
Diagnosis: zero-initialized head weights make upstream gradients exactly zero
(obj_head.weight=0 kills all upstream gradients); updates were ~1e-6/step.
Change: switched to dense correlation decoder (HCRD-v1) with non-zero head
init; then cosine heatmap + reference-scale box; then CLIP-anchor-modulated
correlation (identity injected into both template and search features).
Observation: v1 cosine+CLIP fusion reaches train IoU ~0.15 on 100-sample
overfit (vs 0.06 without identity modulation); loss decreases steadily.
Diagnosis: SAM3 features alone are not identity-discriminative (N14
conclusion confirmed); CLIP anchor conditioning is required for localization.
Change: grid 36 for speed; on-disk encoder feature cache (5MB/frame) to make
multi-epoch training feasible; balanced 600-sample train set (200 miss / 200
present / 200 absent).
Keep/Discard: keep cosine+CLIP-fusion correlation decoder; discard v0 query
decoder and box-head-only localization.
Next: train 600 samples x 6 epochs; HCC eval on calibration.

## Iteration N16.3
Hypothesis: HCRD-v1 (trained F0 on 600 balanced train samples) creates
missing candidates on unseen calibration sequences.
Experiment: HCC eval on 250 calibration episodes (79 generic-miss).
Observation: recall@0.5 (present) 0.012, CCR on generic-miss 0.0127,
top1 0.008, false-capture 0.108.  Train IoU was 0.179 -> no transfer.
Diagnosis: frozen SAM3 encoder features are not identity-transferable for
localization (consistent with N14); F0 candidate creation FAILS.
Change: implemented F1 (unfreeze SAM3 transformer encoder) with a
differentiable fallback for the fused addmm_act kernel; full-res cost ~55s
per sample; launched a 60-sample x 2-epoch F1 run as the single justified
upgrade.
Keep/Discard: keep F0 numbers as the frozen-baseline verdict.
Next: F1 result -> final N16 decision.

## Iteration N16.4 (final)
Hypothesis: F1 (unfreeze SAM3 encoder) transfers candidate creation to
calibration.
Experiment: differentiable fused-kernel fallback, 60 samples x 2 epochs,
HCC eval on the same 250 calibration episodes (79 generic-miss).
Observation: F1 CCR = 0.0 (F0 was 1.3%), recall 0.0, false-capture 0.044.
Presence head (F0) fires on 100% of absent frames (ghost rate 1.0).
Diagnosis: small F1 fine-tune overfits and destroys weak F0 localization;
no order-of-magnitude candidate-creation headroom exists at F0 or F1.
Change: docs/N16_FINAL_REPORT.md; outputs/n16/n16_frozen.json; stage gate.
Keep/Discard: keep HCRED and HCRD-v1 as diagnostics; discard the branch.
Next: external-pretrained joint F1/F2 training at scale OR detection-side
interaction (human-box hypotheses); canonical25 not supported.

## Iteration N17.0
Hypothesis: N16's failure was under-training a toy head, not the
candidate-creation hypothesis; a real transformer detector trained on tens of
thousands of HCRED episodes can create missing candidates.
Experiment: sampled 44,001 train episodes (20k natural miss, 12k present, 12k
absent) and 2,606 calibration episodes; 24,745 unique frames; started a 4-GPU
frozen SAM3 encoder-memory cache build (~2h).
Observation: SAM3.1 detector is single-scale (72x72 memory, num_feature_levels
=1); the HTD therefore uses the full 5184-token memory as search features.
Change: docs/N17_PLAN.md; scripts/build_n17_episodes.py; build_n17_feature_
cache.py; sam3_intermot/recovery/htd.py (HTD-v1, 3.8M params, K=4 proposals);
train/eval scripts for cached-feature training.
Keep/Discard: keep HTD-v1 as the formal architecture; discard N16 cosine head.
Next: cache build -> overfit -> 10k -> >=50k training -> calibration HCC.

## Iteration N17.1 (final)
Hypothesis: HTD-v1 with formal-scale training creates missing candidates.
Experiment: 44,001 episodes x 5 epochs (4 GPUs, cached features); HTD-v2
(identity-modulated search) resume + 3 epochs; calibration eval (2,000
episodes, 1,156 generic-miss).
Observation: HTD-v1 best CCR@0.5 = 0.0147 (recall@0.5 present 0.031); HTD-v2
best CCR = 0.0112; ghost on absent 0.81-0.92; false-capture up to 0.133.
Diagnosis: training at 220k effective samples does not create candidates on
unseen sequences; identity modulation of search features does not help and
hurts localization; presence head still not discriminative.
Change: docs/N17_FINAL_REPORT.md; outputs/n17/*; stage gate.
Keep/Discard: keep HTD/HCRED as formal evidence; discard the branch.
Next: alternative backbone feature space or detection-side interaction.

## Iteration N18.0
Hypothesis: the N17 failure closes only "frozen SAM3 dense feature as a
query->full-frame detector"; SAM3 can still track after a correct
(re)initialization, so the missing component is recovery+handoff, not
continuous re-detection.
Experiment: read real SAM3 source (add_prompt/propagate_in_video/remove_object
lifecycle) and identity layer (namespace/lineage/registry); verified
literature/repo candidates for global recovery.
Observation: text/box add_prompt calls reset_state (wipes the session); the
only existing-object refresh path is add_sam2_new_points(existing obj_id)
with points; H1 (new internal track) is directly supported; namespace.recover()
returns a stable public-id triple for canonical rebinding. Person-search
candidates: PSTR (CVPR22, checkpoints), GFN (WACV23, MIT, checkpoints), DSCA
(AAAI25, Google Drive); DiffPS code still unverified, PSDiff not found.
Change: outputs/n18/source_audit.md, outputs/n18/method_audit.md.
Keep/Discard: keep H1 as primary reactivation path; H2 only via real
point-refinement API.
Next: N18.1 Oracle Reactivation benchmark (4 shards, GPUs 3/5/6/9, running).

## Iteration N18.1 (final)
Hypothesis: a perfect recovery box can be handed to SAM3 and future tracking
will materially recover (the hard gate before any person-search work).
Experiment: 200 sustained AUTO-loss events on cal10 (20/seq), 4 shards on
GPUs 3/5/6/9, one blocking launcher, horizon 60. A0=frozen P0 output;
one-shot future == A0 (single-frame correction cannot affect t+1..);
oracle-reactivation=add_prompt(GT box at t)+propagate, no GT after t.
Observation: retention A0 vs react: h1 0.000/0.745, h3 0.170/0.660,
h5 0.296/0.633, h10 0.372/0.520, h30 0.611/0.389, h60 0.565/0.285
(all cluster-bootstrap CIs exclude 0). Error-free run (horizon granularity):
A0 0.0 vs react 14.8. react>A0 overall in 122/200 events, on h1..10 in 147/200.
Diagnosis: SAM3 clearly extends correct same-ID tracking immediately after a
perfect re-init (no 1-5 frame collapse), but a single box seed without
reconditioning/verification drifts and is overtaken by frame-wise P0
re-detection beyond ~30 frames.
Change: scripts/aggregate_n18_oracle.py, analyze_n18_oracle.py;
outputs/n18/oracle_reactivation*.csv, reactivation_retention*.csv.
Keep/Discard: GATE=PASS (reactivation handoff works in the short/medium term);
require verified re-anchoring in the lifecycle for long-gap stability.
Next: N18.2 clone+checkpoint-verify up to 3 recovery candidates; N18.3
official-pretrained HCRED generic-miss benchmark.

## Iteration N18.2
Hypothesis: mature official person-search checkpoints are real and runnable
locally (needed before any adaptation decision).
Experiment: cloned PSTR (CVPR22) and GFN (WACV23); downloaded official
checkpoints: GFN CUHK-SYSU ConvNeXt-B torchscript + pytorch (epoch 29),
PSTR CUHK-SYSU ResNet50.
Observation: GFN torchscript fails on torch 2.5 (embedded torchvision::roi_align
op); GFN pytorch checkpoint loads into the repo's SeqNeXt with 0 unexpected
keys (only 6 expected OIM buffers missing) after key-name compat fixes;
forward pass works (~1.4s/frame, 2048-d embeddings). PSTR needs a legacy mmcv
compile (deferred).
Change: outputs/n18/checkpoints/*; scripts/gfn_recovery_model.py.
Keep/Discard: GFN runnable now; PSTR deferred but checkpoint retained.
Next: N18.3 GFN on HCRED cal set (running, 4 shards GPUs 0/3/5/6).

## Iteration N18.3a (GFN, final)
Hypothesis: a mature official person-search model beats N17's 1.47% CCR on
the same HCRED generic-miss subset.
Experiment: GFN (WACV23, CUHK-SYSU ConvNeXt-B, epoch 29) on all 2,606 cal
episodes; query crop -> cosine rank of gallery det embeddings; 4 shards.
Observation: generic-miss CCR@0.5 top3 = 0.3893 (N17 0.0147, 26x); top1
0.283; recall@0.3 0.531 / @0.7 0.184; GFN detector already finds 57.7% of
targets (selection ceiling); novel (query-created) rescue = 0.0 by design;
absent ghost at sim>=0.6 = 0.84 (cosine alone is not discriminative).
Diagnosis: the N17 failure was representation/task mismatch, not an
impossible subproblem; but selection-only person search cannot create
candidates its detector missed, and absent rejection needs a verifier.
Change: scripts/run_n18_gfn_hcred.py, merge_n18_gfn.py, gfn_recovery_model.py;
outputs/n18/{hcred_recovery,recovery_backbone_benchmark,absence_recovery}.csv.
Keep/Discard: keep GFN as a strong selection baseline; the query-conditioned
question now needs PSTR.
Next: N18.3b PSTR (running).

## Iteration N18.3b (PSTR, final)
Hypothesis: an official one-stage query-person model (PSTR) complements GFN
on the same HCRED generic-miss benchmark.
Experiment: PSTR (CVPR22, CUHK-SYSU ResNet50) runnable via a legacy
torch1.13+mmcv-full1.7.1 env; PSTR's PartAttention ported to stock mmcv
(pure grid_sample, no CUDA ops); all 2,606 cal episodes, 4 shards.
Observation: generic-miss CCR@0.5 top3 = 0.2107 (GFN 0.3893, N17 0.0147);
top1 0.135; PSTR detector finds 72.8% of targets (GFN 57.7%) but ranks
worse; novel rescue 0.0 (selection-only in protocol); absent ghost@sim0.6
only 0.047 (GFN 0.84).
Diagnosis: mature person search is confirmed as the right recovery family;
classic person search cannot create query-conditioned candidates when its
detector misses; PSTR embeddings are much better calibrated for absence.
Change: pstr_env; scripts/pstr_part_attention.py, run_n18_pstr_hcred.py;
outputs/n18/pstr_hcred*.csv.
Keep/Discard: keep GFN as RECOVERY_BACKBONE_V1 (higher CCR); keep PSTR
embedding as a candidate verifier feature.
Next: N18.6 oracle handoff (running), N18.7 verifier.

## Iteration N18.7a (verifier baseline)
Hypothesis: a simple cosine threshold can gate recovery safely.
Experiment: GFN top-1 sim threshold on 2,500 present + 106 absent episodes.
Observation: PSTR t=0.6 -> precision 0.51 / recall 0.21 / false-reactivation
0.047; t=0.7 -> 0.68 / 0.056 / 0.019. GFN top-1 cosine is worse for absence
(86.8% absent false-accept at sim>=0.3; 84.0% at 0.6) while keeping most
present accepts. Cosine alone cannot give both high precision and recall for
either model.
Diagnosis: a small learned verifier with margin/detector-score (and optionally
cross-model agreement) features is needed, precision-priority.
Change: scripts/analyze_n18_verifier.py;
outputs/n18/verifier_metrics_{gfn,pstr}_hcred.csv.
Keep/Discard: discard cosine-only acceptance; keep per-method data.
Next: N18.6 handoff results; N18.7 learned verifier.

## Iteration N18.6a (handoff, repair cycle 1)
Hypothesis: GFN-recovered boxes (oracle-selected) handed to SAM3 keep tracking.
Experiment: 4-shard reactivation on generic-miss recoverable events with
per-sequence state reuse; recovered GFN box at f -> add_prompt -> propagate.
Observation: shards died mid-run on IndexError
(per_frame_geometric_prompt[num_frames]) for events whose horizon reaches the
video end, losing the remaining events of each shard (686/1073 recovered from
logs); a separate data-hygiene bug (PSTR merge overwrote GFN hcred csv) was
also fixed with per-method files.
Change: guard f+1<num_frames; per-method merge outputs; resume-skip logic;
recovered 686 rows from logs.
Keep/Discard: keep per-sequence reuse (10-12s/event); discard unbounded prompt
clears.
Next: resume missing ~387 events; then aggregate handoff retention.

## Iteration N18.6b (handoff, final)
Hypothesis: oracle-selected GFN boxes handed to SAM3 extend tracking.
Experiment: 794 true GFN recoveries (IoU>=0.5) on generic-miss events; A0 vs
reactivated retention at f+1/3/5/10/30/60 (repaired bounds guard, resume run).
Observation: react vs A0 = 0.707/0.264 @1, 0.590/0.352 @5, 0.536/0.405 @10,
0.388/0.469 @30, 0.273/0.478 @60. Box quality dominates: react@1
0.619/0.802/0.917 for IoU 0.5-0.7/0.7-0.9/0.9+.
Diagnosis: recovery->SAM3 closed loop works in the lost gap; long-horizon
stability needs verified re-anchoring; recovery box precision is a lever.
Change: outputs/n18/oracle_recovery_handoff*.csv.
Keep/Discard: keep GFN+SAM3 handoff as the core loop; discard none.
Next: N18.7 learned verifier (margin/detector-score features).

## Iteration N18.7b (learned verifier, final for now)
Hypothesis: margin + detector score + cross-model agreement beats cosine-only.
Experiment: LR, 5-fold sequence-disjoint CV on cal10; features gfn sim/margin/
score/n_dets + pstr sim; label = top-1 correct on present.
Observation: OOF AUC 0.822; t=0.6 -> precision 0.80 / recall 0.19 /
false-reactivation 0.009; t=0.5 -> 0.71 / 0.41 / 0.057. Top coefs: gfn sim
6.74, gfn margin 3.58, pstr sim 1.98, gfn score 1.36.
Diagnosis: a small verifier is sufficient for precision-first gating; absence
set is small (106) so FR is noisy; two-frame confirmation is the next step.
Change: scripts/extract_gfn_verifier_features.py, train_n18_verifier.py;
outputs/n18/{verifier_features,learned_verifier_cv,learned_verifier_curve}.csv.
Keep/Discard: keep LR verifier as the baseline; discard cosine-only.
Next: N18.8 lost trigger rule; N18.9 full online loop.

## N18 Phase-II plan (brief)
Goal: causal online TRACK -> LOST -> RECOVER -> VERIFY -> REACTIVATE -> TRACK
loop (FULL_LOOP_V0) on cal10, then failure accounting decides the single
biggest upgrade; then sparse B1/B2/B4/B8, TrackEval, freeze, canonical25.
V0 design (real source, documented): human authority H_i seeded from the GT
box at the identity's first frame (simulator semantics); ACTIVE delivery
follows frozen P0 AUTO rows by causal IoU/motion until lost; rule trigger =
3 consecutive missing/wrong frames; LOST runs GFN (H_i crop query) + LR
verifier (gfn sim/margin/score/n_dets + pstr sim, t=0.6); ACCEPT reactivates
via an ISOLATED SAM3 add_prompt session (per-lineage state, option B) so
other identities are untouched by the known session reset; public id stays on
the original lineage. Metrics: lifecycle JSONL, trigger P/R/delay, recovery
attempts/accepts, post-reactivation retention, same-ID TTE, re-correction
probability, collateral delta, failure taxonomy F1-F12.
Next: implement scripts/run_n18_full_loop_v0.py; smoke; cal10 run.

## N18.9a (FULL_LOOP_V0 first cal10 run -> two root-cause repairs)
Hypothesis: V0 loop runs end-to-end on cal10; failure accounting then picks
the single bottleneck.
Experiment: 4-shard full run (GPUs 0/3/5/6), one blocking command.
Observation: 2 of 4 shards crashed (IndexError in iou), completed only
dancetrack0074/0082/0096; trace of 0074 showed gid=0 and gid=2 with identical
delivery patterns (correct 4.4%/1.4%) -> winner-take-all collapse.
Diagnosis: (1) GFN returns 1-D det_boxes on single-detection frames ->
boxes[0] is a 0-d scalar; (2) ACTIVE delivery matched each identity to P0
rows independently, so one row can be claimed by many public IDs; (3) the
duplicate check compared dict keys (always unique), not boxes; (4) react_traj
stored np.asarray(None) -> 0-d array(nan) -> iou crash.
Change: reshape GFN outputs to 2-D; greedy one-to-one P0-row assignment;
box-based duplicate detection; preserve None in react_traj; iou shape guard;
added --no-recovery/--no-reactivation ablation switches and per-seq runtime.
Keep/Discard: discard first V0 numbers (invalid multi-ID delivery).
Next: smoke dancetrack0074/120, then clean cal10 rerun in full/human/gfn
modes, failure accounting, one major upgrade.

## N18.9b (FULL_LOOP_V0 clean cal10 + failure accounting)
Hypothesis: after the delivery/GFN shape/reactivation fixes, V0 runs all 10
calibration sequences and the system-level F1-F12 accounting identifies the
single dominant bottleneck.
Experiment: one blocking 4-shard run (GPUs 4-7), modes full/human/gfn;
analyzer scripts/analyze_n18_full_loop.py.
Observation: full = 7698 recovery attempts, 21 verifier accepts (6 correct /
15 wrong), 468 correct top-1 boxes overall; mean re-correction 0.7025 vs
human 0.7154; taxonomy F3/F4 wrong candidate 4457, F10 duplicate-public-id
874, F5 verifier false reject 462, F1 lost-trigger-late 156, F6 15, F11 14,
F12 11, F8 11, F9 1. Lost trigger: 313 episodes, 0 false triggers.
Diagnosis: candidate selection is the largest failure class, so F3/F4 was
split with an offline top-K replay.
Change: scripts/audit_n18_gfn_topk.py, run_n18_gfn_topk_audit.sh,
analyze_n18_topk_audit.py; outputs/n18/tables/gfn_full_loop_topk_*.csv.
Keep/Discard: keep V0 traces; the accounting, not the absolute score, is the
deliverable.
Next: split F3 vs F4 and test the stale-query-anchor hypothesis.

## N18.9c (F3/F4 split + query-anchor diagnosis)
Hypothesis: (a) is GFN failure mostly detector miss or ranking miss; (b) the
V0 query anchor (first appearance, mean gap 334 frames) is the hidden cause
of the ranking loss.
Experiment: offline replay of all 7698 recorded attempts with GFN top-1/3/5/10
and best detection; second replay compares first-appearance H_i with the
last-GT-visible box (upper-bound diagnosis).
Observation: attempt level any-det 60.0%, top1 7.9%, top3 19.2%, top10 38.9%
(F3 40.0%, F4 52.1%); episode level any-det 84.3%, top1 46.7%, top3 58.6%,
top10 69.0%. Fresh anchor lifts top1 7.9->49.9 and top3 19.2->54.1.
Diagnosis: ranking is the larger loss, but it is driven by the stale anchor,
not by a broken GFN representation; a causal trusted-memory anchor is the
first V1 upgrade (GFN fine-tuning is not yet justified).
Change: scripts/audit_n18_query_anchor.py, run_n18_query_anchor_ablation.sh;
full_loop_v0.py added trusted_box/trusted_frame + anchor_policy;
run_n18_full_loop_v0.py added --fresh-anchor/--out-tag;
scripts/run_n18_loop_v1_stage.sh.
Keep/Discard: keep H_i immutable; add M_i (trusted memory) as the query.
Next: FULL_LOOP_V1 on the same cal10 (running).

## N18.9d (FULL_LOOP_V1 trusted memory -> negative)
Hypothesis: a causal M_i updated from P0 score + temporal IoU can reproduce
the fresh-anchor oracle without future GT.
Experiment: full/human/gfn V1 rerun on cal10 with --fresh-anchor.
Observation: full_v1 7959 attempts, 21 accepts, 4 correct (V0 21/6); mean
re-correction 0.7087 vs 0.7025. CPU replay: causal M_i mean anchor age 260
frames, 90.2% correct (oracle gap 1 frame, 100% correct); score carries no
signal (0.855 vs 0.849), temporal IoU only weak signal.
Diagnosis: a safe fresh anchor cannot be produced from available P0 cues;
verified memory needs appearance verification, but GFN/H_i LR has only
2.8-10% recall on correct delivered rows at safe thresholds.
Change: scripts/audit_n18_trusted_anchor.py, audit_n18_delivery_gate.py,
audit_n18_health_features.py + run scripts; trace now stores delivery score,
prev IoU and delivered box.
Keep/Discard: discard naive M_i; H_i remains the deployed anchor.
Next: test candidate-set and confirmation upgrades before any large training.

## N18.9e (top-3 candidates and two-frame confirmation -> no system gain)
Hypothesis: evaluating GFN top-3 instead of top-1, or a two-frame causal
confirmation, raises accepted correct recoveries without a new backbone.
Experiment: full_top3 cal10 rerun; two-frame smoke on 0074/120.
Observation: full_top3 has identical 21 accepts / 6 correct as V0 (all
additional correct top-3 candidates still score below t=0.6). Two-frame smoke
reduced attempts 10->3 and raised re-correction 0.243->0.425 (worse), because
the 0.4/0.3 operating point admits damaging accepts.
Diagnosis: the bottleneck moved to the verifier operating point and its
feature distribution; a loop-distribution verifier calibration was tried.
Change: top-3 candidate export in the runner; use_two_frame/accept/confirm
config in full_loop_v0.py; scripts/train_n18_loop_verifier.py.
Keep/Discard: keep top-3 plumbing; discard t=0.4/0.3 two-frame operating
point as unsafe.
Next: GFN DanceTrack adaptation (Route C) is now the evidence-based major
revision; no further threshold trial-and-error on the frozen models.

## N18.RouteC.0 (audit: GFN training path + 2025/26 literature)
Hypothesis: stale H_i (mean age ~334) breaks GFN ranking (top1 7.9% vs fresh
49.9%); temporal identity representation adaptation can restore it causally.
Experiment: read GFN source (SeqNeXt/SeqRoIHeads/NormAwareEmbedding, OIM +
NT-Xent losses), opened VCLIP/PKP/DiffPS + CHIRLA/GPS/HLT-MOT.
Observation: identity embedding = NormAwareEmbedding on feat_res4(512) +
feat_res5(1024) -> 2048-dim L2; R0 trainable head is only 1.58M params.
Diagnosis: head-only adaptation first (R0); backbone untouched.
Change: outputs/n18/route_c/{gfn_source_audit,method_audit_2025_2026}.md.
Keep/Discard: keep GFN as backbone; literature = conceptual/decision refs.
Next: RouteC.1 GFN feature cache + temporal pairs.

## N18.RouteC.1 (GFN feature cache, train30+cal10, 4 GPUs)
Hypothesis: pre-head RoI features suffice for a projection-level temporal
adaptation; caching them makes R0 cheap and keeps the backbone frozen.
Experiment: one blocking 4-GPU run over 41,796 frames; forward pre-hook on
roi_heads.embedding_head stores feat_res4/feat_res5 per final detection
(exact embedding alignment), plus first-appearance query crops per identity.
Observation: smoke test aligned features at ~1e-8; ~7-21 dets/frame; ~1 GB
cache expected.
Diagnosis: none yet; run in progress.
Change: scripts/build_route_c_feature_cache.py, scripts/run_route_c_cache.sh.
Keep/Discard: keep frozen det_emb in cache for baseline replay.
Next: RouteC.2 pair construction, RouteC.3 overfit, RouteC.4 R0 training.

## N18.RouteC.2 (temporal pair dataset)
Hypothesis: Q1 first-appearance human anchor -> future same-ID pairs with
hard negatives and absent frames is the right supervision for stale-anchor
recovery.
Experiment: 355 cached identities, 246,414 rows (154,844 positives with an
IoU>=0.5 detector match, 67,672 present-but-detector-miss, 23,898 absent);
gaps 1/3/5/10/30/60/120/240/480/480+; hard negatives = nearest person +
highest frozen-GFN-sim wrong person + highest-score wrong person.
Observation: gap distribution heavily weighted to 120/240/480/480+; absent
rows are fewer because empty gallery frames carry no dets.
Diagnosis: >=100k positives satisfied (preferred 300k limited by 355 IDs).
Change: scripts/build_route_c_temporal_pairs.py; CSVs + stats json.
Keep/Discard: keep Q1 only for R0; Q2/Q3 deferred.
Next: R0 head-only training on train30, cal10 sequence-disjoint validation.

## N18.RouteC.3 (R0 overfit sanity)
Hypothesis: with 64 fixed pairs the InfoNCE+margin objective must decrease.
Experiment: 300 steps, batch 64, LR 1e-3 (OneCycle), 3 idle GPUs.
Observation: loss 6.21 -> 0.00; early duplicated-batch and same-identity
negative bugs fixed (identity-aware masking in InfoNCE).
Diagnosis: overfit gate PASS; 64-sample plateau was a bug, not a hypothesis
failure.
Change: scripts/train_route_c_r0.py (identity mask, materialized npz cache).
Keep/Discard: keep head-only R0.
Next: RouteC.4 formal 12-epoch R0 training on train30 (running).

## N18.RouteC.4 (R0 head-only formal training, train30)
Hypothesis: the frozen ConvNeXt/RoI features already carry enough identity
signal such that only the NormAwareEmbedding projection needs temporal
adaptation.
Experiment: 12 epochs x 2000 steps x batch 512 (3 idle A100s, ~35 min);
InfoNCE tau=0.07 + hard-negative margin 0.25 on 154,844 positives; train30
only, cal10 probes as held-out monitor.
Observation: train loss 1.15 -> 1.02 (plateau); monitor val top1 43.9-46.3%
vs frozen 48.8%; long-gap top1 37.0% vs frozen 33.3%; absent mean top1 sim
0.37 vs frozen 0.73.
Diagnosis: head-only shifts the embedding to reject absent/wrong persons but
does not raise stale-anchor recall.
Change: scripts/train_route_c_r0.py; models/r0_best.pt, r0_last.pt.
Keep/Discard: keep r0_best for downstream analysis; R0 alone insufficient.
Next: real-attempt-distribution evaluation (RouteC.5) and R1 decision.

## N18.RouteC.5 (cal10 real V0 attempt-distribution evaluation)
Hypothesis: R0 must beat the frozen head on the exact 7698 causal recovery
attempts, not just on balanced temporal probes.
Experiment: replay V0 attempts with stale H_i queries; gallery dets from the
cached per-frame features; frozen vs R0 embeddings.
Observation (attempt level, all 7698): frozen top1/3/5/10 =
9.1/20.8/29.6/41.8% (reproduces the 7.9/19.2/27.8/38.9 baseline within
replay noise); R0 = 7.9/19.3/28.2/42.0%. Conditional on detector containing
the target: frozen 16.5/38.0/53.9/76.2, R0 14.3/35.2/51.4/76.6. Absent
top1-sim>=0.6 false reactivation: frozen 92.8% -> R0 13.1%.
Diagnosis: R0 fails the primary retrieval gate (slightly worse top1/3/5) but
strongly improves absent rejection; representation headroom appears upstream
of the linear projection.
Change: scripts/eval_route_c_r0.py + r0_* analysis CSVs.
Keep/Discard: keep absent-discrimination finding; R0 ranking not deployable.
Next: fresh-anchor upper bound under R0, then R1 partial-backbone decision.

## N18.RouteC.5b (fresh-anchor upper bound under R0 head)
Hypothesis: if R0's fresh-anchor ranking is strong, the stale-anchor gap is
query-side evidence staleness rather than head/gallery representation.
Experiment: offline last-GT-visible anchor through the R0 head on the same
7698 attempts (offline diagnostic only).
Observation (conditional on detector containing target): fresh top1/top3/
top5/top10 = 75.9/85.6/90.0/95.4%, MRR 0.822; stale R0 = 14.3/35.2/51.4/76.6.
Diagnosis: the adapted head discriminates near-perfectly when the query is
current; a 300+ frame stale crop lacks the target's current appearance, so
gallery/head adaptation alone cannot recover it. Absent rejection improved
92.8% -> 13.1% (top1 sim>=0.6), which makes verified causal anchor refresh
plausible again.
Change: eval_route_c_r0.py --fresh-oracle; r0_fresh_* CSVs.
Keep/Discard: keep R0 head for downstream verification/anchor-refresh; R1
now tests deeper temporal canonicalization of the stale query.
Next: R1 partial-backbone (box_head features[6,7] + embedding_head).

## N18.RouteC.8 (R1 partial-backbone training, running)
Hypothesis: the head-only ceiling is upstream of the linear projection; the
last ConvNeXt head stages (features[6,7] of the ReID branch = box_head) plus
the embedding head may canonicalize stale anchors nonlinearly.
Experiment: crop-level training through the official model 'gt' path; frozen
backbone features[0..5]/RPN/prop_head/predictors; differential LR
(head 1e-3, box_head 1e-4); InfoNCE + margin; 6 epochs x 800 steps; 2 idle
A100s (8,9) via a CropEmbedder + DataParallel wrapper (list-scatter bug in
DataParallel avoided by padding crops into one batch tensor).
Observation: pipeline smoke passes; detection branch untouched by design.
Diagnosis: pending training result.
Change: scripts/train_route_c_r1.py, scripts/run_route_c_r1.sh.
Keep/Discard: R0 head-only retained as a candidate.
Next: R1 eval on the real attempt distribution (RouteC.9).

## N18.RouteC.9 (R1 eval -> negative, shared-head detection conflict)
Hypothesis: box_head+head unfreeze could canonicalize stale anchors
nonlinearly without touching detection.
Experiment: R1 on the real 7,698 attempts; detector preservation check.
Observation: conditional top1/3/5/10 = 15.4/36.4/52.0/75.4 (frozen
16.5/38.0/53.9/76.2); absent sim>=0.6 = 73.3%; candidate set changed on
7,693/7,698 frames because box_head.feat_res5 feeds score_predictor.
Diagnosis: R1 fails ranking and destabilizes detection; shared-head conflict
is real (DiffPS-style).
Change: scripts/eval_route_c_r1.py + r1_* CSVs.
Keep/Discard: discard box_head unfreeze; keep detector-preservation audit.
Next: one evidence-driven detection-independent upgrade.

## N18.RouteC.10 (MLP canonicalizer upgrade -> negative)
Hypothesis: nonlinear capacity on the frozen pre-head features might succeed
where the linear R0 head failed, with zero detection-side effect.
Experiment: 1536-512-512-2048 ReLU MLP, 8 epochs x 2,000 x 512 (~16.4M
samples), sequence-disjoint monitor.
Observation: real-distribution conditional top1/3/5/10 =
12.6/33.0/47.5/73.1 - the worst of all candidates; monitor top1 fell to
23.2% vs frozen 48.8%.
Diagnosis: capacity on the same stale-anchor features overfits train30
identities; the information limit is the stale query itself, not the head.
Change: scripts/train_route_c_upgrade.py, scripts/eval_route_c_upgrade.py.
Keep/Discard: keep R0 head for absent discrimination (92.8% -> 13.1%).
Next: verdict FAIL_TEMPORAL_REPRESENTATION; next stage = causal verified
anchor refresh + verifier retrain on R0 features, then FULL_LOOP re-run.

## N19.0 (plan: Human-Rooted Verified Dynamic Identity Memory, HVDIM)
Hypothesis: stale first-appearance H_i is the confirmed bottleneck; a causal
dynamic appearance memory M_i(t) (verified slots) should narrow the
static->oracle gap (16.5% -> 75.9% conditional top1) and improve FULL_LOOP.
Plan:
  N19.1/2 Oracle causal refresh replay on the real 7,698 V0 attempts:
  memory slots = recent GT-correct delivered observations (causal, <=t);
  K=1/2/4/8; readers Last/MaxSim/Mean/AgeWeighted; report top1/3/5/10, MRR,
  absent FP, anchor age, memory purity.
  N19.3 Oracle FULL_LOOP: anchor_policy=oracle (write on GT-correct delivery)
  + oracle verifier + SAM3 reactivation on cal10.
  N19.4 Gate: if oracle refresh does not improve FULL_LOOP -> FAIL_REFRESH;
  else -> write dataset + learned writer (safety/future utility).
Hard rules: no val25, no future GT at inference, oracle is offline diagnostic
only; if learned writer cannot beat heuristics, stop before complex memory.
Next: implement oracle replay; 2025/26 memory-update audit in parallel.

## N19.1/19.2 (Oracle causal refresh replay, DONE)
Hypothesis: GT-correct delivered observations, written causally into K
appearance slots and read with simple readers, recover most of the
fresh-anchor retrieval headroom on the real 7,698-attempt distribution.
Experiment: replay V0 attempts with slot embeddings from the frozen GFN
cache; slots = delivered=1 & correct=1 frames with GT box (oracle write),
frame <= attempt; K=1/2/4/8; readers Last/MaxSim/Mean/AgeWeighted.
Observation (conditional on detector containing target): static H_i
25.6/53.5/71.3/89.1 (top1/3/5/10); oracle K=2 best 29.2/57.9/75.0/91.0;
K=4 26.2-26.8/55.9/75.6/91.5; K=8 25.3-26.5/54.9/74.4/91.4. Overall
(fallback to static when no slot): static 15.2/37.0/53.7/75.4, MRR 0.3286;
oracle K=8 mean 18.6/41.0/56.8/78.4, MRR 0.360. Absent false-positive at
top1 sim>=0.5 stayed ~99-100% (slots are the same frozen GFN space).
Diagnosis: oracle refresh gives only a modest retrieval lift (+3-4 points
top1) and does not fix absent discrimination; K=2 is the local optimum;
readers differ little. The system-level question is whether the lift
changes FULL_LOOP re-correction, which N19.3 measures.
Change: scripts/run_n19_oracle_refresh.py; outputs/n19/oracle_refresh.csv,
oracle_memory_k.csv.
Keep/Discard: keep K=2 as the default; oracle write remains an upper-bound
diagnostic only.
Next: N19.3 Oracle FULL_LOOP on cal10 (anchor_policy=oracle + oracle
verifier + real SAM3 reactivation), then N19.4 gate.

## N19.3 (Oracle FULL_LOOP, RUNNING)
Hypothesis: with perfect causal writes (GT-correct delivery -> trusted
anchor) and a perfect verifier (GT-correct candidate -> accept), recovery
top1 should rise and re-correction should fall versus N18 V0 (0.7025) and
human control (0.7154).
Experiment: cal10, 4 shards on GPUs 0-3, full_loop_v0 with
--oracle-anchor --oracle-verifier --out-tag oracle_n19; real SAM3
reactivation. Launched 2026-08-14 14:50; blocking wait.
Change: full_loop_v0.py anchor_policy="oracle"; run_n18_full_loop_v0.py
--oracle-anchor/--oracle-verifier; scripts/run_n19_oracle_full_loop.sh.
Keep/Discard: pending gate.
Next: N19.4 analysis (scripts/analyze_n19_oracle_full_loop.py).

## N19.3/19.4 (Oracle FULL_LOOP + gate, DONE)
Observation (cal10, oracle anchor + oracle verifier + real SAM3
reactivation): attempts 4228 (V0 7698 - fewer because recoveries succeed),
261 accepts all correct (false accepts 0; V0 21 accepts / 6 correct / 15
false). mean re-correction 0.6292 unweighted / 0.6216 weighted vs V0
0.7025/0.6913 and human 0.7154/0.7098. Retention 1/3/30/120:
0.840/0.750/0.519/0.311 vs V0 0.400/0.400/0.140/0.200. The remaining
re-correction floor is dominated by detection-side misses and SAM3
reactivation decay, not by write/verification errors.
Diagnosis: PASS_ORACLE_REFRESH - causal verified refresh is a real system
lever (10% re-correction drop, 2x+ retention), but not a silver bullet.
Keep/Discard: keep oracle diagnostics as upper bound; proceed to learned
writer.
Next: N19.5 real train30 write dataset.

## N19.5 (write dataset, cal10 built)
Hypothesis: causal features (appearance sims, tracker state, memory state)
can predict safe vs dangerous memory writes and future utility.
Experiment: write_dataset_cal10 from oracle_n19 FULL_LOOP events (delivered
observations, causal features, GT labels); future utility by replaying V0
attempts with each candidate as query.
Observation: 42,356 rows; GOOD_WRITE 3,736 / SAFE_BUT_REDUNDANT 17,604 /
DANGEROUS_WRITE 21,016. Feature separability (cal10, 9/10 seqs): oracle
memory-slot similarity AUC ~0.97 (GFN and R0), human-root sims 0.62-0.68,
heuristic memory sims ~0.54, oracle memory age strongly informative
(AUC 0.01 direction-inverted).
Keep/Discard: keep; train30 events now being generated.
Next: train30 dataset + writer V0 training.

## N19.5/N19.6/N19.7/N19.8/N19.9/N19.10 (write datasets + Writer V0, DONE)
Experiment: ran FULL_LOOP_V0 (deployed verifier + real SAM3 reactivation) on
train30 (30 seqs, 9,174 attempts, 60 accepts, ~40 min on 4 GPUs). Built
write datasets: train30 114,722 rows, cal10 42,356 rows. Future utility
labels by offline replay (V0 attempts): train30 GOOD 3,724 / SAFE_REDUNDANT
48,979 / DANGEROUS 62,019 (horizon 240); cal10 6,102 / 15,238 / 21,016.
Feature separability: oracle memory-slot sims AUC ~0.97 (GFN/R0), human-root
sims 0.62-0.68, heuristic memory sims ~0.54, oracle memory age strongly
informative. Writer V0: 2-layer MLP (22 features incl. memory sims), DDP on
4 GPUs (3,5,8,9), 30 epochs; cal AUC 0.9854; threshold 0.98 on dataset ->
precision 0.995 / recall 0.408. Offline self-memory simulation revealed
cold-start deadlock without memory; fixed by seeding memory with the Human
Root (H_i) as slot 0 (consistent in dataset/runner/sim). Simulated
purity-freshness at T=0.95: writes 1,564, write precision 0.991, safe-write
recall 0.831, memory purity 0.980; T=0.9: precision 0.959 / recall 0.854 /
purity 0.976. K=2 slightly better than K=1 retrieval on the (candidate,
attempt) replay.
Diagnosis: learned verified write is feasible; deploy T=0.95, K=2, Human
Root seed.
Change: n19_writer_features.py, build_n19_write_dataset.py,
compute_n19_future_utility.py, merge_n19_utility.py, train_n19_writer.py,
simulate_n19_learned_memory.py, run_n19_full_loop_learned.py,
full_loop_v0.py (learned anchor policy + Human Root seed).
Keep/Discard: keep Writer V0 (T=0.95) as the deploy candidate.
Next: FULL_LOOP_N19 cal10 (running), failure accounting, verifier retrain.

## N19.11/N19.12 (verifier retrain + FULL_LOOP_N19, FIRST ATTEMPT)
Hypothesis: retraining the verifier on learned-memory features (memory age,
agreement count, R0 sim) should let more correct fresh-anchor candidates
through than the static-anchor verifier.
Experiment: verifier LR fit on train30 (6,236 rows, 998 pos, AUC 0.937),
calibrated on cal10 (AUC 0.663, threshold 0.93 -> 44 accepts/22 correct in
the offline dataset). FULL_LOOP_N19 with Writer V0 (T=0.95, K=2, Human Root
seed) + new verifier at threshold 0.75.
Observation: attempts 6,440; 119 accepts (24 correct, 95 false -> verifier
precision 20.2%); mean re-correction 0.7326, WORSE than N18 V0 (0.7025).
Retention is poor (0074 retention1 0.125, 0083 0.154, 0096 0.105).
Diagnosis: the retrained verifier transfers poorly from the offline V0
attempt replay to the live learned-memory loop; too many false
reactivations contaminate the loop. Writer write precision in-loop was
~90% (46/51 on 0099), so the write side is not the primary failure; the
verification side is.
Keep/Discard: keep Writer V0; retrain/calibrate verifier further or raise
threshold.
Next: full cal10 at verifier threshold 0.85 (higher precision operating
point), then decide FAIL_FULL_LOOP_GAIN vs N19.14 short-tracklet upgrade.

## N19.12/N19.13 (FULL_LOOP_N19 variants + failure accounting, FINAL)
Experiment (cal10, learned Writer T=0.95/K=2/Human Root seed):
  - old verifier 0.6: 1 accept, re-corr 0.7153 (no gain).
  - verifier v1 0.75: 119 accepts (24 correct/95 false), re-corr 0.7326.
  - verifier v1 0.85: 55 accepts (19/36), re-corr 0.7097.
  - live verifier v2 (thr 0.19): 94 accepts (19/75), re-corr 0.7356.
  - v2 + two-frame confirmation: 9 accepts (7/2), re-corr 0.7460.
  All learned variants fail to beat N18 V0 (0.7025). Fixes applied along
  the way: (a) cfg.write_fn not passed; (b) DDP state-dict prefix;
  (c) cold start solved by Human Root memory seed; (d) slot embeddings
  cached at write time instead of IoU re-lookup; (e) recovery query uses
  the cached slot embedding instead of crop re-encoding; (f) verifier
  memory features computed live.
Diagnosis: write policy is sufficient (offline precision 0.99, in-loop
0.90); the verifier does not transfer from offline replay to the live
learned-memory distribution, so false accepts contaminate the loop and
erase the memory benefit. F3 (target absent) remains ~60% of rejections.
Keep/Discard: keep Writer V0 + datasets + oracle diagnostics; do NOT
deploy the learned verifier.
Verdict: FAIL_FULL_LOOP_GAIN -> FAIL_DYNAMIC_IDENTITY_MEMORY (current
Writer V0 path). Sparse/TrackEval/Canonical25/second dataset NOT run.
Next: iterative online verifier training or short-tracklet verification.

## N20.0 (plan: Causal Multi-Hypothesis Shadow Tracklet Verification, CMSTV)
Hypothesis: single-frame recovery verification fails because one frame
cannot disambiguate hard distractors; running top-K GFN candidates as
isolated shadow hypotheses for H causal frames and committing only after
temporal evidence (identity consistency, motion, memory agreement,
candidate competition, NONE class) should raise correct commits and lower
false commits vs N19 single-frame verifiers.
Plan:
  N20.0 read code + 2025/26 literature/GitHub audit.
  N20.1 top-1/3/5/10 target availability on real attempts with N19 learned
  memory (offline).
  N20.2 oracle shadow propagation (real SAM3 isolated sessions, no public-ID
  binding, H=1/3/5/8).
  N20.3 oracle delayed-commit FULL_LOOP; N20.4 oracle gate.
  If PASS: N20.5-7 train30/cal10 shadow datasets + feature/hard-negative
  analysis; N20.8-13 baselines (single-frame, mean-pool, fixed-H temporal,
  K+1 competition, NONE calibration, adaptive stopping); N20.14-15 oracle/
  learned-memory verifier; N20.16-18 FULL_LOOP_N20 + failure accounting +
  one evidence-driven upgrade if needed; then sparse/TrackEval/freeze/
  Canonical25/stats/second dataset/novelty/ICLR.
Hard rules: no val25, no future GT at inference, shadow state never mutates
public ID or memory before commit, no hindsight relabel, K<=10, max 4 idle
GPUs, no new detector, no Writer V0 retraining.
Next: top-K availability + literature audit in parallel.

## N20.1 (Top-K recovery availability, CORRECTED)
Hypothesis: with N19 learned memory (Writer T=0.95, K=2, Human Root seed)
and the real N19 learned-loop attempt distribution, top-K contains the
correct target often enough for multi-hypothesis shadow verification.
Experiment: fixed the N20.1 pairing bug (previously events_oracle_n19 were
matched with transactions_full from a different loop). Re-ran on matched
learned_n19 events+transactions: 6,371 rankable attempts, 2,699
target-present (42.4%), 3,672 target-absent (57.6%).
Learned-memory ranks (target-present only):
  top1 16.8% / top3 39.9% / top5 56.6% / top10 78.2%.
Static Human-Root ranks: top1 11.9% / top3 36.9% / top5 54.6% /
top10 76.4%.
Diagnosis: top3/5 give 40%/57% theoretical candidate headroom; top1 alone
is insufficient. K=5 is the practical oracle-gate configuration.
Keep/Discard: keep corrected table (topk_recovery_availability.csv).
Next: dump-only no-commit attempt distribution -> shadow cache -> oracle
delayed-commit FULL_LOOP (H=0/1/3/5/8).

## N20.0 (method audit, DONE)
Opened/verified GitHub: OAMOT, KeyRe-ID, CATB, SiamABC (SiamABC WACV 2025).
SAMURAI + FC-Track confirmed at paper level. HLT-MOT GitHub 404
(unverified). No 2025/26 method combines human-root authority + top-K
global recovery + isolated shadow tracklets + K+1 NONE verification +
atomic public-ID commit, so no external code is directly used.
Saved: outputs/n20/method_audit_2025_2026.md.

## N20.1b (no-commit attempt distribution, DONE)
Experiment: ran the N20 delayed-loop runner in dump-only mode (learned Writer
T=0.95, K=2, Human Root seed; no commits) over cal10 to get a clean
no-commit attempt distribution; then replayed the writer over these events
to compute learned-memory ranks. Fixed a performance bug (np.delete used the
full gallery instead of the per-frame slice), cutting cal10 CPU time from
~60 min to ~5 min with identical logic.
Distribution: 6,441 attempts; target-present 2,701 (41.9%); target-absent
3,740 (58.1%). Learned-memory ranks on target-present: top1 17.0% / top3
39.9% / top5 55.5% / top10 77.2% (static root: 11.9/36.8/53.5/75.6).
Saved: outputs/n20/topk_no_commit.csv (used as the shadow-generation plan).
Next: N20.2 real SAM3 shadow tracklets for top-5 correct candidates on 4
idle GPUs (horizon 120), then N20.3 delayed-commit FULL_LOOP variants.

## N20.2/N20.3 (oracle shadow + delayed-commit smoke, RUNNING)
Experiment: real SAM3 shadow generation (K=5, horizon 120) launched on 4
idle GPUs (3/5/8/9) for the 1,499 top-5 correct candidates of the
no-commit distribution. Fixed shadow-step sid parsing in the delayed loop.
Smoke on dancetrack0074 (900 frames, partial cache): H=5 oracle
delayed-commit -> 150 attempts, 2 commits, 2 timeouts, mean re-correction
0.489 vs 0.683 with no commits; retention@5 1.0. Early signal is strongly
positive; full gate tables pending shadow completion.

Fix: memory explosion in the shadow generator (SAM3 session feature cache
grew to ~28GB RSS / 39.5GB VRAM per process after ~90 shadows). Killed the
4 shards, added per-shadow `reset_session` + `torch.cuda.empty_cache()` and
`--skip-existing` resume; relaunched. Now ~9GB RSS / ~26GB VRAM per process
and bounded; 79GB RAM available. Completed rows are preserved and skipped.

## N20.2/N20.3/N20.4 (Oracle shadow + delayed-commit gate, DONE)
Experiment: 1499 real SAM3 shadows (K=5, horizon 120) generated; oracle
delayed-commit FULL_LOOP on cal10 for K=3/5 x H=0/1/3/5/8. Fixed a glob bug
that made K=3 runs see an empty cache. Results (mean re-correction):
  k5_h0 0.6662, k5_h1 0.6681, k5_h3 0.6705, k5_h5 0.6785, k5_h8 0.6798;
  k3_h0 0.6697, k3_h1 0.6714, k3_h3 0.6725, k3_h5 0.6825, k3_h8 0.6823.
All oracle accepts correct; false=0. Baselines: V0 0.7025, Human 0.7154.
Verdict: PASS_ORACLE_SHADOW (lower bound: cache covers the no-commit
distribution; committed loops shift attempts so only ~10-12% of cached
attempts recur; full coverage would be stronger).
Keep: oracle gate tables; shadow cache; missing-attempt union (2675) saved.

## N20.5-N20.13 (shadow dataset + temporal verifier, PARTIAL DONE)
Dataset: 1482 cal10 attempts / 11823 causal evidence steps (22 features).
Feature top AUC at H=5: consecutive_delivered 0.79, shadow_delivered 0.76,
temp_sim_first 0.73. 5-fold sequence-disjoint CV (H=5): GRU AUC 0.783/AP
0.779 > LR 0.703/0.743 > MLP 0.644/0.696. Learned GRU FULL_LOOP_N20 (H=5,
thr 0.5): 58 commits, 36 correct (62.1%), 22 false; mean re-correction
0.6809 vs V0 0.7025 (+0.022) and Human 0.7154 (+0.035); oracle bound
0.6785. LIMITATIONS: only correct-candidate shadows (no wrong/absent/NONE),
no train30 expansion, coverage ~10%, false commits 38% -> learned stage not
finalized; TrackEval/Canonical25/second dataset/ICLR assessment NOT run.
Keep: GRU fold models + dataset + gate; next step is dataset expansion with
wrong/absent hypotheses + K+1 competition + NONE calibration.

## N20 Phase-II wave2 (fuller oracle coverage, DONE)
Generated wave2 correct-candidate shadows for the 2675 missing target-present
top-5 attempts (horizon 120). Two shards were OOM-killed (global memory);
all rows were recovered via --skip-existing resume. Cache now covers 4174
unique attempts. Reran Oracle delayed-commit FULL_LOOP (K=3/5, H=0/1/3/5/8):
  k5_h0 0.6550 (delta_v0 +0.0476), k5_h1 0.6614, k5_h3 0.6636,
  k5_h5 0.6731, k5_h8 0.6709; k3_h0 0.6650, k3_h1 0.6680, k3_h3 0.6700,
  k3_h5 0.6800, k3_h8 0.6752. Shadow starts rose to 237-340 per variant but
  coverage is still ~5-6% of live-loop attempts because commits shift the
  attempt distribution; the Oracle Gate remains a lower bound.
Gate: PASS_ORACLE_SHADOW (oracle variants only; learned variant excluded).
Downstream commit-cost audit: correct commits give 59-89% correct frames at
h1-h30; false commits give 5-24% -> false commit costs ~2-10 wrong frames
plus memory contamination (risk asymmetry confirmed).
Next: complete all-candidate cal10 shadows (resuming shards 1/3), then
train30 all-candidate dataset, K+1 GRU training, FULL_LOOP_K+1, failure
accounting, report.

## N20 Phase-II dataset generation (IN PROGRESS, with OOM notes)
cal10 all-candidate cache complete: 5781 hypotheses (1700 sampled attempts,
top-5, horizon 8). cal10 K+1 dataset built: 44,839 rows; smoke K+1 GRU
training works (NONE precision 0.77, 5 epochs, in-sample only). train30
all-candidate generation: one shard OOM-killed (RSS grew to 66GB during a
long propagation); resumed with --skip-existing. Remaining: train30
completion -> build train30 K+1 dataset -> formal train30->cal10 training ->
FULL_LOOP_K+1.

## N20 Phase-II plan (CMSTV formal)
Goal: turn PARTIAL into PASS_FULL_LOOP_N20 or a defensible
FAIL_SHADOW_TRACKLET_VERIFICATION.
1. N20.5B/C: complete cal10 shadow cache -> all top-K hypotheses for the
   real recovery distribution + wave2 correct-candidate coverage (2675
   missing target-present top-5 attempts, horizon 120, 4 idle GPUs).
2. N20.5D: rerun Oracle delayed-commit gate with the fuller cache; keep the
   lower-bound caveat only if coverage is still partial.
3. N20.6: train30 no-commit distribution (CPU) + all-candidate top-5 shadow
   dataset (sampled; horizon >=8; real GFN distribution; no synthetic
   negatives as primary).
4. N20.7: wrong / stable-wrong / NONE / F3a-F3b audit on cal10+train30.
5. N20.8-13: K+1 set-level baselines; shared GRU temporal encoder; NONE
   calibration; fixed-H 1/3/5/8; adaptive sequential stopping (commit /
   continue / reject-all) with thresholds set only on cal10.
6. N20.14-15: empirical downstream commit-cost audit from N18/N19/N20
   transactions; risk-aware decision (false commit >> missed commit).
7. N20.16-18: Oracle-memory isolation + learned-memory FULL_LOOP V2 +
   failure accounting (F3a/F3b/F4a/F4b/F6/F8/F9/F10/F11/F12/F13/F14).
8. If live shift persists: one train30 on-policy round (N20.19-20). If
   representation is the binding bottleneck: one audited major upgrade
   (N20.21-23).
9. If PASS_FULL_LOOP_N20: freeze -> sparse B1/B2/B4/B8 -> TrackEval ->
   sanity 0004/0005/0007 -> Canonical25 -> stats -> second dataset ->
   novelty -> ICLR assessment. Update docs/N20_FINAL_REPORT.md throughout.
Hard rules: no val25, no future GT at inference, no hindsight relabel,
shadow never writes public ID/memory before commit, cal10 not used as main
training data, max 4 idle GPUs, no Writer/detector changes, no fabricated
references.
Next: launch wave2 shadow generation (2675 attempts, 4 GPUs), then train30
no-commit distribution and all-candidate dataset while it runs.

## N20 Phase-III on-policy runtime (MAJOR ENGINEERING BREAKTHROUGH, RUNNING)
Implemented a TRUE on-policy FULL_LOOP runner
(scripts/run_n20_onpolicy_full_loop.py): at each live LOST attempt it ranks
the current GFN gallery with the live learned memory, creates top-5 SAM3
hypotheses on the fly (one reused base session per sequence; reset_session
isolates hypotheses and keeps video frames -> memory bounded), propagates
H=5 frames, builds the same 30-dim K+1 feature schema, runs the frozen K+1
GRU, and commits / rejects. No offline cache consulted for eligibility.
Key fixes found by debugging: (a) max_frame_num_to_track must be None
(explicit 5 triggers an official SAM3 shape bug [5184,0,256]); break after
the needed horizon; (b) per-frame geometric prompts require
set_frame_geometric_prompt + invalidate_detector_prefetch on the active
session; (c) rank_mem feature was constant 0 in the training dataset
(all-candidate cache rows lacked the field), making its normalization
scale ~1e8; fixed builder to use candidate_rank and retrained (min sd now
0.04).
Smoke on dancetrack0074 (300 frames): 38 attempts, 6 K+1 decisions,
2 real on-policy COMMITs (both false, candidate5; live precision 0% so
far), 35 rejects/timeouts, memory stable (~20-25GB, one session reused).
The full causal chain now runs: LOST -> top-K -> on-the-fly shadows ->
temporal features -> K+1/NONE -> COMMIT -> trajectory change. Live
distribution transfer is NOT yet good (false commits) and needs calibration
or retraining on on-policy data. Full cal10 first-pass is running.
Outputs: outputs/n20/onpolicy_runtime_source_audit.md,
outputs/n20/onpolicy_full_loop_v1/*.

## N21 (Human-Supervised Online Identity Adaptation, HOIA) plan
Hypothesis: human corrections are not just state repairs; they are
high-value deployment-time supervision. A lightweight identity verifier
updated online from correction-derived positives and explicit hard
negatives should reduce repeated identity confusion and future human
corrections beyond offline on-policy retraining.
Baselines:
  A0 = frozen N20 CMSTV (live first-pass: 0074/0075/0080 completed before
       the process was lost; 8/8 live commits were false).
  A1 = offline train30 on-policy aggregation + offline K+1 retrain,
       evaluated on true live cal10.
  A2 = memory-only human adaptation (Human Root + dynamic memory, no
       gradients).
  A3 = A1 + correction-driven online lightweight head/adapter update
       (episodic reset per sequence, causal, cal10 only for calibration).
Execution order: N21.0 freeze N20 baseline; N21.1 2025/26 literature/GitHub
audit; N21.2 train30 true on-policy rollout; N21.3 model-induced hard
negatives; N21.4 offline on-policy retrain; N21.5 live cal10 A1; N21.6-15
correction supervision, ledger, online head/adapter, replay/forgetting/
collateral audits, cal10 freeze; N21.16-17 FULL_LOOP_N21 + failure
accounting; then optional major upgrade, sparse/TrackEval/Canonical25/
stats/second dataset/novelty/ICLR if PASS.
Hard rules: no val25, no future GT in decisions/updates, no detector or
SAM3 backbone online update, no full fine-tune, no cross-sequence adapter
leakage, max 4 idle GPUs, honest baseline comparisons.

## N21 (HOIA) offline phase — result (2026-08-16)
Implemented HumanSupervisionLedger + offline correction-driven online
adaptation experiment on the real N20 all-candidate shadow datasets
(scripts/n21_offline_adaptation_experiment.py).
- A0 frozen cal10 stream (986 full-top5 attempts): 157 corrections
  (92 false commits, 65 missed commits).
- A1 offline retrain (train30 + hard-negative weighting, 40 epochs):
  172 corrections — no gain; shifts false -> missed.
- Best online variant (head-only from A0, lr1e-4, replay32, KL2.0,
  1 step/correction, episodic reset): 153 corrections (-2.5%),
  false 84 (-8.7%), missed 69 (+6.2%) — marginal, not robust.
- Adapter variants worse (189-218 corrections).
- Without KL regularization online updates degrade the stream (216+).
- Same-rank repeats remain 58% of false commits; collateral flips +9/-13.
- Verdict (offline): FAIL_ONLINE_ADAPTATION provisional; decisive live
  train30 rollout + FULL_LOOP_N21 still pending. Report:
  docs/N21_FINAL_REPORT.md.

## N21 Phase-II (CATIL) plan (2026-08-16)
Hypothesis: human corrections become substantially more useful when they
adapt the intermediate multi-frame identity representation (temporal
tracklet encoder) rather than only the final 8.6K head. Phase-I (tiny head)
is NOT sufficient to falsify correction-driven online learning.
Plan: P2-0 read state; P2-1 re-audit 2025/26 lit/GitHub; P2-2 identity
representation source audit; P2-3 true live train30 on-policy rollout
(4 idle GPUs, per-seq .done/resume); P2-4 model-induced hard negatives;
P2-5 multi-frame tracklet identity dataset from real shadow boxes + GFN/R0
embeddings; P2-6 temporal tracklet identity encoder (causal, 2-4 layer);
P2-7 offline train on train30; P2-8 representation gate vs 22-d verifier;
P2-9 optional single external ReID upgrade only if needed; P2-10 C1 LoRA
(50K-200K) in temporal encoder; P2-11 C2 partial FT (0.5M-3M); P2-12 online
training amount 5/20/40/60; P2-13 capacity ladder; P2-14-16 exact-repeat /
collateral / forgetting; P2-17 cal10 freeze; P2-18 live FULL_LOOP C0/C1/C2/
memory-only/offline-retrain; P2-19 failure accounting; P2-20 decision
(PASS_LIGHTWEIGHT_LORA / CAPACITY_LIMITED / FAIL_CORRECTION_DRIVEN_ADAPTATION);
then freeze/sparse/TrackEval/canonical25/stats/second-dataset/novelty/ICLR
only if PASS. No new N22. Final chat: report link only.

## N21 Phase-II (CATIL) result (2026-08-16)
Built real multi-frame tracklet identity dataset (GFN+R0 per shadow frame),
causal 2-layer Transformer encoder, GRU-prior residual scoring, LoRA on QKV,
partial-FT upper bound.
- Frozen visual-only top1-correct 34.7% (random 20%); 22-d GRU prior is the
  strong signal.
- Offline (train30 428 attempts, 30 epochs): cal10 set acc 0.763, top1 0.822.
- Capacity ladder (online): C0 16.5K/211 corrections, C1 LoRA 49K/210,
  C2 partial FT 2.1M/210 vs frozen 211 -> NO gain.
- Online passes 5 vs 20: 211 vs 210 -> no gain; mean positive-logit delta
  +0.0009; 69% same-rank repeats persist.
- Verdict (offline): FAIL_REPRESENTATION_ADAPTATION provisional; bottleneck
  is the identity representation, not capacity or update amount.
- True live train30 rollout running on GPU 5-8: 10/30 seqs, 902 attempts,
  13 commits (4 correct, 9 false). Full report updated:
  docs/N21_FINAL_REPORT.md. GPU9 was used briefly for analysis (disclosed).

## N21 Phase-II status update (2026-08-16 evening)
Rollout: 24/30 train30 true on-policy done (partial aggregate: 3726
attempts, 34 commits, 13 correct / 21 false, precision 0.382; 92
repeated-attempt gids). GPU8 is the only remaining shard; supervisor and
post-rollout watcher (run_n21_phase2_after_rollout.sh) keep running;
resume-safe .done markers verified. Partial distribution comparison:
live 3726 vs offline proxy 748 attempts (ratio ~5.0) -> live distribution
is far more attempt-dense; do not draw final conclusions yet.
Report corrected: Phase-I = 1 pass/correction only (no 5/20/40 claim);
Phase-II = 5/20 passes; status stays PARTIAL; no formal
FAIL_CORRECTION_DRIVEN_ADAPTATION. Next gated GPU steps after 30/30:
trajectory-dump pass -> rebuild true-live tracklet dataset -> retrain
CATIL C0/C1/C2 -> cal10 calibration -> FULL_LOOP_N21.
Resume/skip/atomic verification (run_n21_train30_onpolicy.py +
run_n21_train30_stage.sh): (1) runner checks per-seq .done and prints
SKIP_EXISTING; (2) per-seq outputs written (events/transactions/metrics)
before .done marker; (3) stage retries each shard up to 3 attempts and
resumes via markers; (4) only GPU8 shard remains (attempt 1). No kill/restart
performed; normal slow tasks untouched.

## N21 Phase-II rollout COMPLETE (2026-08-17 ~03:15)
30/30 train30 true on-policy done (all 4 shards RC=0; STAGE.done).
Final aggregate (COMPLETE): attempts 6296, commits 39 (14 correct /
25 false, precision 0.359), timeouts 5964, target-present 5839 /
absent 457, repeat-attempt gids 136. Live vs offline-proxy attempt
ratio 7.54x (6296 vs 835) -> true live distribution is far more
attempt-dense and hard. Launched trajectory-dump pass (same 4 GPUs,
--dump-trajectories, new out dir) to persist real SAM3 shadow
tracklets for the true-live CATIL retrain; prepared converter +
generalized dataset/training scripts (custom csv/npz/model-prefix).
Post-rollout CPU pipeline ran: aggregate + distribution comparison.

## N21 Phase-II interruption + resume (2026-08-17 11:05)
Trajdump pass (12/30 done) was externally killed at ~08:30 (all 4 shards +
supervisors vanished, no OOM in dmesg; GPUs freed). Resumed with setsid
(detached session) so tool-session cleanup cannot kill it: trajdump stage
skips the 12 completed sequences (.done resume) and re-runs the remaining
18 on GPUs 5-8; true-live retrain pipeline waits for 30/30 then auto-runs
convert -> K+1 CSV -> visual npz -> CATIL retrain (C0/C1/C2) -> cal10 eval.
Current aggregate (30/30 rollout): 6296 attempts, 39 commits
(14 correct / 25 false, precision 0.359), 5964 timeouts, live-vs-proxy
attempt ratio 7.54.

## N21 Phase-IIb: true-live retrain COMPLETE (2026-08-18 ~04:47)
Trajdump pass finished 30/30 (interrupted once at 12/30, resumed via .done;
GPU8 was the last shard, ended ~03:58). Fixed converter (added is_correct +
per-frame correct via GT) after pipeline bug; retrain pipeline completed.
True-live train30 dataset: 6003 attempts, 28811 tracklets, coverage 55.4%.
Live-trained CATIL cal10: A0 208, C0 207, C1 207, C2 208 corrections;
online passes 5/20: 208/207; adaptation delta logit +0.0003. vs offline
proxy (211/211/210/210): base improved slightly, online gain still ~0.
Conclusion strengthened: correction-driven representation adaptation shows
no measurable gain even trained on the true on-policy distribution. Formal
FAIL_CORRECTION_DRIVEN_ADAPTATION still gated behind live FULL_LOOP_N21.

## N21 Phase-III plan (2026-08-18)
1) Fix stale report statements (10/30 etc.) - done.
2) True-live cal10 FINAL GATE: L0 frozen GRU / memory-only, L1 live-trained
   CATIL frozen, L2 CATIL C1 LoRA online, L3 CATIL C2 partial-FT online.
   Added on_correction hook to full_loop_v0; new runner
   scripts/run_n21_live_final_gate.py (CATIL live verifier + episodic reset +
   causal online updates; smoke L1/L2 OK). Stage runs L0/L1/L2 in parallel
   on GPUs 5/6/8, then L3 (global <=4 GPU rule; only 3 used).
3) Then analyze -> PASS/FAIL_CATIL_REPRESENTATION_ADAPTATION verdict.
4) If FAIL: R0 source/differentiability/cache audit, GFN-vs-R0
   decomposition, direct R0 adaptation (C0/C1/C2), offline train on
   true-live data, 5/20/40 passes, exact same-distractor recurrence,
   train30 chronological, cal10 freeze, R0 live FULL_LOOP.
5) External ReID replacement only as the final gated step.

## N21 Phase-III L3 resume (2026-08-20 ~11:40)
L3 crashed at 11:19 (RC=120) after ~21h while on dancetrack0096; the
crashed process had the single-instance C2 update path (pre-two-instance
fix) and hit the inference-tensor autograd bug; 7/10 sequences were valid
(0074/0075/0080/0082/0083/0086/0087). Fixed teardown abort with
os._exit(0) after ONPOLICY_DONE (outputs flushed). Resumed on GPU 5 with
the two-instance (decision/online) code; remaining 0096/0098/0099.

## N21 Phase-III CATIL live gate COMPLETE (2026-08-20 23:45)
L0/L1/L2/L3 each completed cal10 10/10 (8428 frames); L3 resumed from the
seven valid per-sequence markers and completed 0096/0098/0099 without
rerunning completed sequences. Final true-live correction events:
L0=73111, L1=72969, L2=73051, L3=72965. False commits: 73/84/91/83;
commit precision: .0395/.0345/.0521/.0460. L2/L3 performed 2425/2434
causal parameter-update events (24250/24340 optimizer steps); L1 is frozen
and performed zero parameter updates (the runner's historical metric was
correction-supervision events, corrected in the final aggregator). The
largest correction reduction is L3 -146 (-0.20%), far below the frozen
10% gate, with +10 false commits. Verdict:
FAIL_CATIL_REPRESENTATION_ADAPTATION. This closes CATIL only; direct R0
source/cache audit and C0/C1/C2 adaptation remain gated next work.

## N22.0 (plan: Correction-Driven Identity Representation Learning, 2026-08-21)
Hypothesis: N21 failed because it updated a late verifier over frozen GFN/R0
inputs; a correction must change the identity space or its causal memory, not
only the final commit boundary. First gate is an identity-source audit plus
live-aligned representation evaluation before choosing R0 adaptation,
tracklet learning, or a new memory formulation.
Protocol: use cal10 real top-5 shadow JSONL and cached GFN/R0 vectors; use
GT only for offline labels; evaluate GFN, R0, GFN+R0 and causal memory
similarity at h=1/3/5/8 with AUC, top-1, top-5 availability, hard-negative
margin and hard-negative error. Do not reuse the historical N21 NPZ for this
gate until its frame convention is repaired.

## N22.1 (identity audit, COMPLETE 2026-08-21)
The source audit confirms that Human Root, dynamic memory and shadow-frame
identity vectors all enter through frozen GFN `emb` and R0 `r0g` spaces;
SAM3 supplies propagation boxes but no persisted identity embedding in the
current pipeline. CATIL receives concatenated 4096-d GFN+R0 vectors only
after those spaces have been computed.

The historical N21 dataset builder uses `r["frames"][:h]` (first token f0),
while the N21 live runner uses `range(1, args.h + 1)` (first evidence f0+1).
This is a one-frame offline/live training-input mismatch. N22 audit therefore
reconstructed prefixes f0+1..f0+8 directly from the shadow cache.

Results on the usable 986-attempt cal10 top-5 stream: raw R0 is strongest
(AUC 0.644 at h=5; top-1 0.190; hard-negative error 0.661), GFN+R0 is
slightly lower (AUC 0.639), and memory similarity is lower still (AUC 0.580
for R0-memory at h=5). All hard-negative margins are negative. Top-5 hit
availability is 0.562 at h=5 and is candidate-pool availability, shared by
all scorers. This is evidence for a representation/memory bottleneck, not a
final-head capacity bottleneck. Artifacts: `docs/N22_identity_audit.md`,
`outputs/n22/identity_representation_eval.csv`,
`outputs/n22/identity_representation_summary.json`.

## N22.2 (literature and method selection, 2026-08-21)
Checked recent primary papers and author repositories for correction-driven
online adaptation, compatible continual ReID, and replay. LIT-LoRA is the
closest online correction precedent: a correction trains a future-producing
LoRA and a validation gate controls later acceptance. OAIMS supplies the
stronger final corrected target idea. LoRA-DRS and Bi-C2R reinforce that an
evolving feature space must remain compatible with old gallery/memory
vectors. No checked work combines these constraints with human hard-negative
MOT corrections, shadow tracklets and true-live false-commit risk. Full
details and official links are in `docs/N22_literature_review.md`.

## N22.3 (CDCIA offline gate, COMPLETE 2026-08-21)
Built `scripts/build_n22_identity_dataset.py` using live-aligned evidence
f0+1..f0+8. The resulting sources contain 4015 train30 candidate rows / 835
attempts and 5781 cal10 rows / 1200 attempts; at h=5 the all-five-valid
protocol leaves 364 and 557 usable groups, respectively. This explicitly
separates missing R0 coverage from an identity zero vector.

Implemented `scripts/n22_cdcia_offline.py`: a shared low-rank residual R0
adapter, correction-style positive-vs-hard-negative margin, and compatibility
penalty. It overfits train30 (AUC 0.954, top-1 .723) but fails the
sequence-disjoint cal10 gate (AUC .500, top-1 .147, 385 corrections) versus
frozen R0 (AUC .644, top-1 .190, 324 corrections). The no-compat and
compatibility variants are both failures. This rules out a blind global
R0-LoRA/low-rank update under this data regime.

## N22.4 (causal prototype-memory hypothesis, offline + smoke)
Implemented `scripts/n22_prototype_offline.py` and the `N22_PROTO` branch of
`scripts/run_n21_live_final_gate.py`. The method keeps frozen R0 as the
canonical coordinate, updates a per-identity positive prototype only after a
human correction, and stores up to two explicit wrong-candidate prototypes;
the current score is positive similarity minus a negative-prototype penalty.
The train-selected alpha/beta=(.05,.05) reduces clean cal10 offline
corrections from 324 to 317 and false commits from 79 to 69, but repeated
wrong attempts remain high (275), so this is only a modest hypothesis.

The 250-frame true-live smoke on `dancetrack0074` completed successfully:
23 recovery attempts, one shadow commit, two positive prototype updates and
one negative update. It verified that updates occur through the causal
`on_correction` callback and affect later shadow decisions only.

## N22.5 (true-live cal10 FULL_LOOP, COMPLETE 2026-08-21)
Ran `N22_PROTO` on all ten cal10 sequences (`8428` frames) with the frozen
N20 K+1 verifier, live SAM3 propagation, shadow tracklets, and the causal
R0 prototype update. The process ended with `ONPOLICY_DONE`; all ten atomic
`.done` markers and per-sequence event/transaction/metric files are present.
The run used one GPU (GPU 3) for 34958.6 recorded runtime seconds, about
9.71 GPU-hours. Aggregate results: 3441 recovery attempts, 226 shadow
commits, 25 correct / 201 false commits (precision .1106), 74255 human
correction events, 74092 repeated corrections, mean re-correction
probability .7271, and 890 causal correction-supervision callbacks. The
prototype memory recorded 231 positive updates, 149 negative updates, and
795 repeated-wrong correction events.

Against N21 L0, N22 has +1144 corrections (+1.56%), +1144 repeated
corrections, and +128 false commits (201 vs 73), so it fails the correction
gate despite increasing correct commits (25 vs 3) and reducing shadow
timeouts. The full aggregate is in
`outputs/n22/live_cal10_proto/live_cal10_summary.json`.

For the requested downstream identity metrics, the event traces were
exported to MOTChallenge format and evaluated with the pinned official
TrackEval. N22_PROTO combined metrics on the same ten train sequences are
HOTA 35.048, AssA 29.245, IDF1 39.813, IDSW 1897; N21 L0 is HOTA 38.087,
AssA 32.268, IDF1 42.891, IDSW 1078. These derived metrics also reject N22
and are recorded in `outputs/n22/trackeval_baselines/trackeval_all.log`.

## N22.6 (final report, COMPLETE 2026-08-21)
Wrote `docs/N22_FINAL_REPORT.md` with the N18–N21 background, N22 identity
audit, literature and official-code review, CDCIA/prototype method pivot,
offline and true-live ablations, causal protocol, TrackEval export, resource
accounting, failure analysis, and ICLR value assessment. Final status:
`FAIL_CORRECTION_DRIVEN_IDENTITY_ADAPTATION` for the tested CDCIA and
N22_PROTO routes; no positive ICLR claim is made.

## N23 (final report, COMPLETE 2026-08-21)
Implemented correction-driven target discovery to address the N18–N22 F3
failure boundary: 58.07% of N20 no-commit attempts had no GFN candidate at
IoU>=0.5. N23 uses a causal human correction query, a deterministic whole-
frame multi-scale window bank capped at 600 windows, frozen N15 CLIP-ReID,
an offline PairRanker ablation, and a separate `NONE` gate. The PairRanker
was rejected for live use after calibration10 top-1 fell to 4.20% versus
37.44% for raw cosine; the deployed configuration is raw cosine plus a
train30-only raw-score gate threshold of 0.6.

Training used 2,631 episodes and 2,854 balanced pairs for the adapter; the
formal cache contains 6,000 train rows and 2,606 calibration rows. The raw
gate reached sequence-disjoint validation AUC 0.8620. On calibration10,
target-present bank availability was 89.80%, raw top-1/top-5 were
37.44%/60.44%, and raw+NONE correct/wrong recovery was 23.40%/27.24% on
target-present rows. Target-absent acceptance remained 16.04%.

The strict causal true-live run completed `ONPOLICY_DONE` for all ten
calibration sequences: 8,428 frames, 2,121 attempts, 314 accepted
recoveries, 97 correct, 207 false, 10 without a post-accept delivery,
31.91% precision among delivered accepts, 70,627 human corrections, and
mean re-correction probability 0.6921. Human corrections decreased only
3.4% from N21 L0 (73,111), while false recoveries exceeded the N21 L0
false-commit count (73) by 134; the correction gate therefore fails.
Official TrackEval on the same ten sequences reports N23 HOTA 34.641,
AssA 30.712, IDF1 39.707, MOTA 9.687, and IDSW 2,136, versus N21 L0
HOTA 38.087, AssA 32.268, IDF1 42.891, MOTA 30.299, and IDSW 1,078.
The run also repeatedly emitted the official SAM3
`tracking_obj.max_num_objects=16` capacity warning without crashing.

Final status: `FAIL_TARGET_CONDITIONED_DISCOVERY`; no positive end-to-end
claim is made. Full report: `docs/N23_FINAL_REPORT.md`. Machine-readable
outputs are in `outputs/n23/live_full_loop_cal10/live_cal10_summary.json`
and `outputs/n23/trackeval/mot_results/N23/pedestrian_summary.txt`.

## N24.0 (initial analysis and literature, COMPLETE 2026-08-25)
N24 tested whether a human correction can be represented as a short causal
identity trajectory over five isolated SAM3 shadow candidates plus `NONE`.
The N18-N23 reports and this log were reconciled before implementation. The
main falsifiable claim was that causal temporal identity reasoning would
separate the corrected target from same-scene distractors and lower later
corrections in a strict live loop. The surviving bottleneck was defined as
identity separation under candidate/propagation shift, with false-commit cost
and open-set rejection treated as first-class constraints.

The primary 2025/2026 literature and official repositories/checkpoints were
audited in `docs/N24_LITERATURE_REVIEW.md`, including MOTIP, DualTemporalMOT,
SENTRY, DTPTrack/InteractTrack, TAPNext/TAPNext++, AllTracker/ReTracker and
related temporal/query-propagation directions. C0 (masked temporal
mean/prototype), C1 (causal Transformer) and C2 (multi-pool temporal
competition) were selected as a controlled capacity progression. No external
checkpoint was adopted.

## N24.1 (real temporal diagnostic, COMPLETE 2026-08-25)
The first diagnostic used 1,002 real SAM3 shadow candidate rows in 240
attempt groups, matched to frozen GFN/R0 features at IoU>=0.5. GFN/R0/fused
prototype top-1 at H=1 were .2180/.2180/.2137 with fused MRR .4916; at H=5
they were .2500/.2585/.2458 with MRR .5023; at H=10 they were
.2288/.2373/.2373 with MRR .4931; and at H=20 they were
.2246/.2373/.2373 with MRR .4867. Mean coverage falls from .987 at H=1 to
.672 at H=20. The real trajectories have mean length 15.75 and maximum 16,
so H=20 includes masked missing observations rather than twenty uniformly
observed frames. This is not a monotonic temporal identity gain.

## N24.2 (dataset, reasoners and offline gate, COMPLETE 2026-08-25)
The compact train30/cal10 manifests contain 4,015/5,781 candidate rows,
835/1,200 attempt groups and 726/1,048 complete top-five groups. Train30
contains 186 NONE groups and cal10 638 NONE groups. The real H=20 shadow
subset contains 104 complete groups. C0/C1/C2 were trained for 30 epochs
using only complete groups. At cal10 H=5, candidate top-1 is .3576/.2096/.1777
and exact decision accuracy is .1546/.2996/.2529 for C0/C1/C2; their commit
precisions are .1505/.1041/.0788. On real-shadow H=5, candidate top-1 is
.2039/.0874/.0874 and commit precision is .2019/.0577/.0541. C0 was the
only model promoted to the strict live run, using the train30-only threshold
.18 and margin .05; train-to-calibration transfer is a documented negative
result.

## N24.3 (strict causal FULL_LOOP and official TrackEval, COMPLETE 2026-08-25)
The C0 H=5 run completed all ten N21 calibration sequences and 8,428 frames.
The canonical aggregate has 5,856 recovery attempts, 107 accepted commits,
15 correct and 92 false (precision .1402), 3,549 target-present attempts,
2,307 target-absent attempts, 73,615 corrections after anchor, 100,603
present frames after anchor, and global identity-weighted mean re-correction
probability .7509. The complete run used only completed per-sequence metric
sources; interrupted duplicate 0082 traces were excluded. Total recorded
runtime is 76,812.8 seconds. The official SAM3 `max_num_objects=16` warning
appeared without a crash.

The exact per-sequence audit is in
`outputs/n24/full_loop/C0_h5_cal10/aggregate_metrics.csv`; the aggregate JSON
is `outputs/n24/full_loop/C0_h5_cal10/aggregate_summary.json`. Exporting the
event traces and running the pinned TrackEval on the same ten DanceTrack train
sequences produced combined HOTA 37.658, AssA 31.841, IDF1 42.447, MOTA
28.080 and IDSW 1,292. The official output is
`outputs/n24/trackeval_C0_h5/mot_results/N24_C0/pedestrian_summary.txt`.

Relative to N21 L0 (HOTA 38.087, AssA 32.268, IDF1 42.891, MOTA 30.299,
IDSW 1,078, 73,111 corrections and 73 false commits), N24 decreases every
primary tracking metric, adds 214 identity switches, increases corrections
by 504 (+.69%) and increases false commits by 19. It is also below N21 on the
correction gate despite being less catastrophic than N22/N23 on false commits.

## N24.4 (final verdict, COMPLETE 2026-08-25)
N24 is a negative result for the tested correction-conditioned temporal
identity formulation: `FAIL_CORRECTION_CONDITIONED_TEMPORAL_IDENTITY_REASONING`.
Short temporal pooling gives only a modest H=5 diagnostic improvement, while
larger causal reasoners lose candidate ranking and transfer. Frozen
appearance features plus offline shadow labels do not solve the
candidate/propagation shift. The next viable research boundary is a jointly
candidate-conditioned, motion-aware, on-policy-trained and NONE-calibrated
system, with causal memory admission tested against the same correction and
TrackEval gates. Final report: `docs/N24_FINAL_REPORT.md`.

## N25.0 (initial analysis and literature, COMPLETE 2026-08-25)
N25 defined CCRIM as a correction-conditioned, target-specific positive/negative
identity memory with relational candidate-set reasoning, explicit `NONE`,
shadow-before-write admission and risk-controlled commit. N9–N24 were reconciled
before implementation. The primary falsifiable gate was pre-training information
sufficiency on sequence-disjoint train30/cal10; no CCRIM model was authorized
before new information beat the strongest frozen GFN/R0 baseline by at least
five top-1 points without increasing target-absent false acceptance.

The 2025/2026 paper and official-code audit is in
`docs/N25_LITERATURE_REVIEW.md`. It covers InteractTrack/IMAT, DAM4SAM/D4SM,
SENTRY, DualTemporalMOT, MOTIP, DTPTrack, TAPNext, LIT-LoRA, OAIMS, continual
ReID compatibility, open-world tracking and the official SAM3 multiplex path.
No external tracker or checkpoint was adopted.

## N25.1 (real SAM3 feature audit, COMPLETE 2026-08-25)
`scripts/n25_feature_audit.py` ran the pinned SAM3.1 multiplex checkpoint with a
real `person` prompt and 60-frame propagation on dancetrack0074 and 0096. The
model exposes `obj_ptr`, `maskmem_features`, image/backbone features and
per-object state internally. Consolidated pointer shape is `[1,16,256]`; after
the official multiplex demux it is `[5,256]` for 0074 and `[16,256]` for 0096.
The observed mask memory is `[1,256,72,72]` (spatial, not a stable candidate
axis), while image features are `[5184,1,256]` frame-level grid features.
The public `Sam3Backend` observation contract exposes boxes/masks/confidence,
not these internal tensors. The state contains object-slot mappings but not a
validated candidate-to-public-ID cache. No zero vectors or guessed mappings
were used. Full audit: `docs/N25_SAM3_FEATURE_AUDIT.md`; machine output:
`outputs/n25/feature_audit/n25_sam3_feature_audit.json`.

## N25.2 (raw on-policy episode data, COMPLETE 2026-08-25)
Stage B was built from the N20 raw `full_shadow_cache_train30/cal10`, not the
N24 compact NPZ. It contains 4,015 train rows / 835 groups and 5,781 cal rows /
1,200 groups, with variable GFN top-5 candidates, actual shadow validity masks,
raw RGB crop descriptors, GFN/R0 features, motion, neighbor summaries and
post-hoc labels isolated from decision features. Raw crop coverage is 100% for
22,353 train and 42,493 cal requested valid steps. Train30 has 732 present and
103 absent groups; cal10 has 787 present and 413 absent groups, so current
candidate-set target availability is 87.66% / 65.58%. Candidate ranks are
incomplete in the inherited stream (cal rank-5 has 1,048 rows, not 1,200).
No explicit human-denial field exists, and dancetrack0087 has no N20 candidate
rows. N23 whole-frame and controlled-union sources were not merged because they
have incompatible event keys; SAM3 proposals were not materialized before the
gate. The complete manifest is
`outputs/n25/dataset/n25_dataset_manifest.json`.

The first slow builder invocation was interrupted before writing aggregate
artifacts and is excluded. After adding bounded frame/crop caches and releasing
per-sequence feature arrays, a second invocation completed both splits. A
field-name error in the post-build scorer was repaired and the gate was rerun
from the completed JSONL without rebuilding the images. No interrupted output
entered a canonical aggregate.

## N25.3 (information gate and oracle decomposition, COMPLETE 2026-08-25)
The machine-readable gate is `outputs/n25/dataset/n25_information_sufficiency.csv`.
At cal10 H=10 the strongest old baseline B2 GFN+R0 has top-1 .3705, pair AUC
.6143 and target-absent false acceptance .4205. The best raw temporal candidate
B4 has top-1 .2861, pair AUC .5486, hardest-negative margin -.0229 and false
acceptance .7585. At H=5 B2 is .3662 while B4 is .2869. Raw+motion and
raw+neighbor variants are no better. B5 SAM3 internal, B10 explicit negative,
and B11 all-legal are explicitly `NOT_COMPUTABLE`, not zero-filled. ECE/Brier,
trained logistic/MLP/listwise models and risk curves were not run because the
pre-training gate already failed; the CSV records this status.

The offline oracle decomposition is in
`outputs/n25/n25_oracle_decomposition.json`. On the current cal10 candidate set,
perfect identity/NONE/commit selection can be 100% safe conditional on a
present candidate, but correct-commit coverage is capped at 65.58% by candidate
availability. The 170-correct/zero-false k5_h0 SAM3 shadow oracle and the N19
261/261 causal refresh oracle are inherited references, not new N25 runs.

## N25.4 (final verdict, COMPLETE 2026-08-25)
The information gate fails the predeclared +5pp, margin, false-acceptance and
SAM3-alignment requirements. CCRIM C0/C1/C2 training, offline ablations, new
on-policy/DAgger data, FULL_LOOP, official TrackEval and canonical val25 were
therefore not run. No checkpoint was created; configuration/checkpoint and
TrackEval status are recorded in `outputs/n25/frozen_config.json`,
`outputs/n25/checkpoint_manifest.json` and `outputs/n25/trackeval_status.json`.
Final status: `FAIL_N25_INFORMATION_SUFFICIENCY`; final report:
`docs/N25_FINAL_REPORT.md`.

## N25-R.0 (protocol and alignment repair, COMPLETE 2026-08-25)
N25-R found that the original train shadow shard 0 contained 335/500 expected
attempts, leaving 165 of 1,000 train groups absent; cal was complete at
1,200/1,200. An independent repaired cache now validates 1,000 groups and
4,765 unique candidate rows without modifying N20/N25. Repaired B2 H5 is
.8264 train versus .3662 cal, so a 46.02-point gap remains. The required
three-sequence SAM3 object-alignment smoke reached .9009 ROI and .8922
pointer/binding coverage but failed the 95% ROI and 5-point missingness gates.
F2 mask, F3 pointer and F4 memory were frozen. Candidate-box F1 remained legal
through a candidate-independent frozen frame grid and passed an exact-key full
cache validator.

## N25-R.1 (deep features and explicit negatives, COMPLETE 2026-08-25)
Symmetric SAM3 F1 and N15 CLIP-ReID features cover every valid observation in
4,765 train and 5,781 cal rows, with per-sequence atomic `.done` artifacts.
The existing N21 ledgers contain 983 explicit-negative occurrences but only 53
conflict-free canonical identity/event/rank keys (51 cal, 2 train). A frozen
B2 H5 same-stream train simulation generated 267 causal corrections, 267
explicit rejection writes and 158 positive writes; current corrections never
score their own event and memory is target-specific.

## N25-R.2 (fixed R5 gate, COMPLETE 2026-08-25)
B10 is a real candidate-rank signal: cal H5 top-1 .5458 versus B2 .3662
(+17.96 points), pair AUC .7480 versus .6147, margin +.0092 versus -.0132,
and candidate-set-absent false acceptance .0726 versus .2324. Eight of nine
sequences improve and the sequence-bootstrap gain CI is [.0807,.2342]. The
train-frozen commit policy transfers at only .7205 precision/.1908 coverage,
so it fails the immutable 90%-precision-at-5%-coverage gate. SAM3 F1 gains
only 1.63 points; the all-legal logistic probe overfits (.9214 train/.3621
cal). No method passes all ten criteria.

## N25-R.3 (final verdict, COMPLETE 2026-08-25)
Final status is `PARTIAL_N25R_FEATURE_SIGNAL`. CCRIM, candidate union,
FULL_LOOP, TrackEval and val25 were not run because R5 did not authorize them.
N25-R establishes that identity-specific human negatives improve ranking but
do not yet transfer as a safe commit rule, while object-conditioned SAM3
identity extraction remains alignment-limited. Final report:
`docs/N25R_FINAL_REPORT.md`.

## 2026-08-25 — N26 protocol freeze and initial hypothesis

- **Hypothesis:** N25-R B10's rank gain can become a safe baseline only if existence/expressibility and commit risk are factorized on genuinely same-policy causal trajectories; otherwise a correction-conditioned set model is required.
- **Modification:** Added `docs/N26_INITIAL_ANALYSIS.md` and `outputs/n26/frozen_protocol.json`. Frozen B10 H5 (`lambda=0.8`), train30/cal10 roles, the immutable N26-A gate, maximum-four-GPU N26-B route, and pre-result FULL_LOOP stress sequences `0075/0082/0099`. Historical N25-R outputs remain untouched; the project has no Git metadata at or above the project root, so git status is recorded as unavailable rather than fabricated.
- **Result:** Protocol is frozen before any N26 prediction or training result. `val25_read=false`; candidate union remains disabled.
- **Failure reason:** NOT_APPLICABLE.
- **Keep:** YES — this is the governing N26 protocol.

## 2026-08-25 — N26-A same-policy causal safety gate

- **Hypothesis:** A factorized existence/commit module on identical frozen-B10 train/cal trajectories can retain B10 ranking while transferring at least 90% commit precision at 5% or greater sequence-OOF coverage.
- **Modification:** Added `scripts/n26a_onpolicy_gate.py`. Replayed train30 and cal10 chronologically with one frozen B10 H5 policy (`lambda=0.8`) and one simulator; only selected-and-rejected candidates enter explicit-negative memory, with all writes applied after the current prediction. Audited raw DanceTrack GT into four states, trained two train30 sequence-OOF regularized heads, and applied leave-one-cal-sequence-out thresholds.
- **Result:** Same-policy B10 cal10 H5 top-1 is 75.56% (no rank regression), pair AUC 0.8766, and margin +0.0175. The OOF safety policy commits 296/1,200 (24.67% coverage) at 87.50% precision; candidate-set-missing false acceptance is 2.66%, commits span all nine sequences, and the largest sequence contributes 45.95%. The sole immutable failure is precision <90%. The raw full-attempt audit identifies train states 891 visible/present, 509 visible/candidate-missing, 100 target-not-visible; cal states 787/728/185. The materialized model subset contains only the first two states; this limitation is retained explicitly.
- **Failure reason:** SCIENTIFIC_FAILURE — sequence-shifted commit risk remains insufficient; artifact counts, causal ordering, state isolation, held-sequence threshold provenance, and rank reproduction checks passed. No implementation/data error was found, so the one repair rerun allowance was not used.
- **Keep:** YES as a strong same-policy rank/safety baseline; NO as an authorized deployable gate. Per protocol, stop calibrator tuning and proceed to N26-B.

## 2026-08-25 — N26-B targeted official-code audit and route selection

- **Hypothesis:** A current official 2025/2026 implementation may expose a reusable correction-conditioned association/memory interface without replacing SAM3.
- **Modification:** Re-verified eight author/organization repositories and exact remote HEAD commits, including InteractTrack, SENTRY, DAM4SAM/D4SM, MOTIP, SAM3.1, CLIP-ReID, and Conformal Risk Training. Recorded code, weight, license, component, four-GPU, and non-transferability evidence in `docs/N26_OFFICIAL_CODE_AUDIT.md` and `outputs/n26/official_code_audit.json`.
- **Result:** No direct model satisfies target-specific positive plus human-explicit-negative memory, K+1 NONE, correction response, and separate commit risk on the frozen SAM3 candidate stream. Selected exactly one local route: Correction-Conditioned Set Association Model (CC-SAM). SENTRY informs temporal/neighbor summaries, DAM4SAM the separation of target and distractor memory, and MOTIP listwise identity tokens; no external code is vendored and SAM3 remains the executor.
- **Failure reason:** Direct-transfer hypothesis rejected due missing interface/checkpoint/license or task mismatch, not due compute.
- **Keep:** YES — use the audit to constrain the one local trainable route.

## 2026-08-25 — N26-B dense causal data and feature completion

- **Hypothesis:** Real H1--H9 prefixes plus every legal upstream attempt can enlarge the training stream without inventing independent human corrections.
- **Modification:** Added `scripts/n26_extract_extra_clip.py` and `scripts/n26_build_dense_dataset.py`. Reconstructed the frozen static GFN top-5 for the 500+500 `target_present=0` attempts, extracted symmetric frozen CLIP-ReID crops, and emitted every real shadow prefix with a shared `parent_event_id` and weights summing to one per parent. Explicit positive, selected-and-rejected human negative, and ordinary model-induced hard-negative tokens retain separate provenance; current feedback applies only from the next parent event.
- **Result:** Round0 train has 1,500 parents / 6,900 states across 28 represented sequences; cal has 1,700 / 11,160 across ten. Parent weights sum to 1,500.0001 / 1,700.0001. Train states are 891 visible+present, 509 visible+candidate-missing, and 100 target-not-visible; cal is 787/728/185. No UNKNOWN arose from the valid raw mapping, but the class remains represented and masked by protocol. Seventeen train and 39 cal temporal states legitimately have no valid candidate and are retained with target `NONE`.
- **Failure reason:** The first four-GPU feature launch failed before data/model work because `scripts` was not on `sys.path`; the minimal import repair passed. A validation assertion initially required every temporal state to have a valid candidate; it was corrected because all-lost states are legal NONE observations, not corrupt data.
- **Keep:** YES — dense artifacts and ledgers pass finite/alignment/parent-weight checks; N25-R history is untouched.

## 2026-08-25 — N26-B CC-SAM Round0/Round1 four-GPU training

- **Hypothesis:** A small listwise set encoder with explicit correction memory, NONE, existence/risk heads, and correction-response loss can retain B10 ranking while reducing later corrections.
- **Modification:** Added `scripts/n26_ccsam_model.py`, `scripts/n26_train_ccsam.py`, and DDP regression smoke. CC-SAM has 583,038 trainable parameters; frozen SAM3 and CLIP-ReID parameters are absent from the optimizer. Round0 used twelve sequence-disjoint selection epochs, selected epoch 3 by minimum total validation loss, then reinitialized seed 26 and fit all 28 represented train sequences for three epochs. The Round0 model rolled out 1,500 parents; Round0 and Round1 trajectories were clustered at 0.5 weight each and refit for five full epochs on four A100s with NCCL DDP and bfloat16 AMP.
- **Result:** Selection epoch 3 has validation loss .5115 and H5 top-1 .8708. Round0 and Round1 full jobs completed with finite losses/gradients, resumable optimizer/RNG checkpoints, and 1,500 effective parent weight per epoch. Final Round1 train loss is .1065 and parent-weighted accuracy .9860. Association training consumed .2132 GPU-hours; extra frozen CLIP extraction consumed .1459 GPU-hours.
- **Failure reason:** A single-batch smoke exposed Python 3.12/PyTorch collation of `numpy.bool_`; scalar conversion fixed it. The first NCCL smoke had a post-success teardown race; a barrier fixed the regression log. The first Round1 rollout wrote the main arrays but failed formatting a relative checkpoint path in its summary; resolving the path was the minimal repair. No protocol, seed, loss, sample, or metric changed.
- **Keep:** YES as reproducible full training evidence and resumable checkpoints; scientific transfer is evaluated separately.

## 2026-08-25 — N26-B strict cal10 gate and final verdict

- **Hypothesis:** Final CC-SAM should achieve at least 90% sequence-OOF commit precision at 5% coverage, no more than 7.26% absent false accept, B10-level ranking, and a significant correction-response effect.
- **Modification:** Froze `outputs/n26/evaluation_protocol.json` before safety outputs. The final model rolled out all 1,700 cal parents. Leave-one-sequence-out thresholds use only the other sequences. Same-checkpoint correction counterfactuals remove only the latest legal past correction. Evaluated frozen-weight memory/NONE/risk interventions, the dense B2/B10/SAM3-ROI proxies, 2,000 sequence bootstraps, human-effort proxies, and failure cases.
- **Result:** The evaluator exactly reproduces N26-A dense B10 H5 top-1 .755644. Final CC-SAM falls to .3373 top-1, .5764 MRR, .6090 pair AUC and -1.3102 hardest-negative margin; Round0 fixed-history diagnostic is only .3493. OOF safety commits 4/1,700, all wrong: precision 0, coverage .235%, absent FA 0, and all commits come from one sequence. Existence accuracy/recall are .6812/.6746. Memory lowers fixed-history re-correction only from .5801 to .5765; its sequence bootstrap CI crosses zero. Adding the latest correction changes rejected selection .21149 -> .21027 and future error .60269 -> .59902, but both bootstrap upper bounds are 0, not strictly below it. The model fails ranking in all nine materialized cal sequences.
- **Failure reason:** SCIENTIFIC_FAILURE, not evaluator misalignment: the aligned B10 score is reproduced exactly, every inference tensor is finite, and memory ablations all remain near 33--34% top-1. The learned absolute candidate scorer has severe train/cal sequence transfer collapse; correction memory supplies only a small non-significant residual effect.
- **Keep:** Keep datasets, source, checkpoints, counterfactuals and failure evidence. Do not deploy CC-SAM, do not start a third route, and do not run FULL_LOOP/TrackEval/final calibration/val25. Final status: `SCIENTIFIC_GATE_FAIL`.

## 2026-08-25 — N27 protocol freeze and initial hypothesis

- **Hypothesis:** N26 failed because its free absolute-logit branch overwrote B10, not because correction memory or SAM3 failed. A sign-constrained bounded residual trained on genuinely independent cross-video parents may improve correction response while preserving B10.
- **Modification:** Added `docs/N27_INITIAL_ANALYSIS.md` and `outputs/n27/frozen_protocol.json`. Frozen B10 (`lambda=0.8`, margin `0.02`), APCR-S residual range `[-0.03,+0.03]`, strict no-correction zero gate, separate positive/explicit-negative/hard-negative provenance, sequence/video split rules, conservative train/external-only safety calibration, immutable cal10 gates, one anchor-repair allowance and APCR-T authorization condition. N25-R/N26 remain immutable; the project is not a Git repository.
- **Data scale:** Target at least 50,000 independent parent keys, 500 identities and 50 videos across audited Tier A sources, without prefix/round/detector-variant inflation. `/data1` begins with about 136GB free and a hard 40GB reserve.
- **Result:** Protocol frozen before N27 data, model or cal10 predictions. `val25_read=false`; candidate union and visual unfreezing remain unauthorized.
- **Failure reason:** NOT_APPLICABLE.
- **Keep:** YES — governing N27 protocol.

## 2026-08-26 — N27 official/data audit and independent episode scale

- **Hypothesis:** A larger, source-audited cross-video parent pool can test a bounded correction residual without manufacturing event count.
- **Modification:** Re-audited official 2025/2026 repositories and local DanceTrack, MOT17, MOT20, BDD100K Tracking, KITTI Tracking and TAO-Amodal. Collapsed MOT17 DPM/FRCNN/SDP to seven original scenes, kept only the BDD image/annotation intersection, admitted standard KITTI `label_02` pedestrians, and excluded TAO for sparse/non-exhaustive labels, mixed source licenses and explicit BDD overlap. Built 55,000 external-train and 12,000 external-held-out parent keys with separate GT/public-detector provenance; retained 1,500 real DanceTrack P2 parents. Extracted 288,829 unique frozen 1280-D CLIP-ReID embeddings in four recoverable fp16 shards.
- **Result:** The main pool contains 5,332 external identities across 122 videos; all 67,000 external keys are unique and the episode ledger has no duplicate keys. Human negatives are only post-prediction selected-and-rejected candidates; ordinary hard negatives remain separate; current feedback is not visible to the current prediction. Disk reserve remains satisfied.
- **Failure reason:** TAO was scientifically excluded rather than used as a sparse-NONE source. The first feature launch was interrupted before accepting artifacts; the four-shard retry completed and passed exact-order, finite-value and normalization validation.
- **Keep:** YES — audits, manifests, caches and causal ledgers are retained; val25 remains unread.

## 2026-08-26 — N27 APCR-S P0/P1/P2 and causal rollout

- **Hypothesis:** A sign-constrained bounded residual can add correction response while preserving the frozen B10 ranking.
- **Modification:** Implemented local APCR-S with exact zero masking, separate positive/explicit-negative/ordinary-hard channels, bounded monotone residuals and separate safety heads. Trained only the residual over frozen B10/CLIP-ReID using four-GPU bfloat16 AMP, dataset-balanced losses, P1 external pretraining, DanceTrack train-fold CV and a final P2 fit. Replayed external held-out, real DanceTrack P2 and historical cal10 ranking-only trajectories causally, with B10/APCR memories isolated by dataset/video/identity.
- **Result:** P0 passes exact anchor reproduction, exact no-memory zero, signs, bounds and monotonicity. P1 held-out static APCR top-1 is .741431 versus B10 .740948; P2 fold-4 CV selects epoch 10 with .912371 APCR/B10 top-1. Final dynamic external APCR is .741045 versus B10 .740658; DanceTrack is .766554 for both. Historical cal10 ranking-only APCR is .755644, and dynamic B10 selection reproduces the frozen reference at zero mismatch.
- **Training resources:** P1 used .042836 GPU-hours; P2 CV .003528 and final P2 .003918 GPU-hours, all with four visible A100s. Checkpoints, per-dataset/per-sequence curves and resumable state are retained.
- **Keep:** APCR-S is retained as a reproducible bounded-residual analysis artifact, not yet as a valid deployable method; APCR-T is not authorized because the evidence does not identify a capacity-limited underfit.

## 2026-08-26 — N27 correction, safety and final gate

- **Hypothesis:** The learned residual should show a sequence-stable counterfactual correction effect and support a conservative commit policy.
- **Modification:** Evaluated same-checkpoint latest-correction counterfactuals, positive/negative/hard controls, 2,000-repetition sequence/dataset group bootstraps, and a pre-cal10 safety certificate using separate existence/commit heads. The safety threshold was frozen from external held-out plus DanceTrack train folds using one-sided Wilson, sequence bootstrap, coverage, concentration and absent-false-accept constraints; then the single predeclared group-conformal fallback was attempted.
- **Result:** External counterfactuals show target probability gain .007220, rejected-identity selection delta -.181283 with sequence-bootstrap CI [-.290552,-.144403], but APCR top-1 gain is only .000386 and its sequence-bootstrap CI [-.000112,.001139], with no majority sequence rank support. DanceTrack response is directionally positive but ranking is unchanged. Neither B10 nor APCR has a non-empty primary safety threshold; the conformal fallback has zero usable coverage. Explicit-negative-only does not beat the ordinary hard-negative control in the external static top-1 table.
- **Failure reason:** `SCIENTIFIC_GATE_FAIL` is a scientific transfer/safety failure, not a cache or alignment failure. Because safety failed, FULL_LOOP, TrackEval, cal10 threshold selection, candidate union and val25 were not authorized.
- **Keep:** Keep all audits, data, checkpoints, counterfactuals and failure artifacts for a future newly frozen protocol. Do not treat APCR-S or B10 as a certified automatic-commit policy.

## 2026-08-26 — N28 LIT-LoRA code audit and route revision

- **Hypothesis:** LIT-LoRA's stable-anchor/live-challenger principle can support real correction-conditioned identity learning, but its released single-object VOS path cannot be transferred to MOT without global assignment and a non-LoRA online-learning control.
- **Modification:** Audited the CVPR 2026 paper, supplementary material, arXiv v2 source, official repository commit `c1a68373d922f2c5656afb54fc584a54cf3d773c`, and its patch against SAM2. Audited the local global-assignment and atomic interaction paths, and added FACT (KBS 2026) as the relevant online-MOT boundary. Wrote `docs/N28_LIT_LORA_DEEP_AUDIT.md` and updated the N28 architecture report.
- **Result:** The public LIT path resets per object/pass, trains Q/K/V LoRA from a full GT mask with an empty prompt, accepts future candidates using GT IoU, and does not release replay/CLIP/memory-adapter/automatic variants. Its active main table contains unresolved raw-value/reduction mismatches. The N28 route is revised to assignment-coupled LCIA with Hungarian, atomic two-sided REASSIGN/ID_SWAP updates, exact zero-reference deltas, and a correction-supervised recursive least-squares (C-RLS) challenger. Large four-GPU meta-training is gated on future benefit beyond both B10 and C-RLS.
- **Failure reason:** NOT_APPLICABLE — this was a research/code audit, not an experiment. No N28 performance result is claimed.
- **Keep:** YES — retain LIT's anchor/challenger principle; do not vendor its code, copy its GT oracle, claim first online-learning MOT, or start large LoRA training before the cached-feature dual-challenger gate.

## 2026-08-26 — N28-A assignment-coupled cached-feature smoke

- **Hypothesis:** A zero-reference, identity-scoped challenger can be added after frozen B10 without breaking global assignment, correction provenance, or rollback semantics.
- **Modification:** Added the frozen machine-readable N28 protocol; implemented the target×candidate+NONE matrix, existing-Hungarian coupling, legal correction compiler, correction-only Sherman–Morrison C-RLS, frozen-backbone/B-only Q-K-V LCIA-LoRA, earliest-checkpoint validator, and atomic challenger snapshots. Reused only the immutable N27 DanceTrack cached-feature row for the engineering smoke; no SAM3 executor or N27 artifact was modified.
- **Result:** N28-A passes exact zero-reference delta/matrix/selection for both challengers, a global assignment conflict test, bilateral REASSIGN, four ID_SWAP constraints, one cached support overfit, B-only gradient scope (3,072 live parameters per smoke identity), unaffected-identity invariance, and byte-identical LoRA+C-RLS rollback. Existing regression tests remain 92 passed. This is an engineering gate, not a future-performance result.
- **Repair note:** The first C-RLS smoke exposed that a zero-initialized state cannot move from a rejected `0` target. The final transaction path maps compiler labels to signed `+1/-1` targets and rejects unselected-candidate pseudo-negatives; the cache, event, protocol and gate criteria were unchanged before rerun.
- **Failure reason:** NOT_APPLICABLE — N28-B causal episodes, meta-training, real SAM3/FULL_LOOP, TrackEval, automatic arbitration and val25 were not run; `val25_read=false`.
- **Keep:** YES — N28-B may be considered only as the next separately frozen causal phase; N28-C remains blocked until LCIA beats B10 and C-RLS on adapter-specific future benefit.

## 2026-08-26 — N28-B causal cached-episode feasibility and conditional N28-C transition

- **Hypothesis:** A correction-supervised relational challenger can reduce future identity errors beyond B10's own analytic memory response while preserving exact no-update behavior; if the random live LoRA passes the predeclared sequence-stable gate, offline episodic meta-training should start automatically.
- **Modification:** Froze `outputs/n28/n28b_frozen_protocol.json`. Added the cached single-identity legal correction projection, vectorized chronological replay, four-cell `B10/adapted × current/cf` attribution, C-RLS, random-live B-only LCIA, no-update and provenance controls, 2,000 sequence-group bootstraps, and the single-process B→conditional-C runner. Reused only the 55,000 external-train and 1,500 real DanceTrack N27 B10 caches plus the frozen APCR-S checkpoint; no images, val25 artifact, SAM3 executor, or N27 file was modified.
- **Result:** N28-B completed in 685.61 seconds on CPU. External train contributed 55,000 parents / 122 sequences / 5,332 identities / 9,416 fixed B10 feedback events; DanceTrack contributed 1,500 / 28 / 76 / 209. Candidate-set-absent rows (7,997 external and 609 DanceTrack) were retained but never converted into human DELETE/NONE supervision. LCIA accepted 1,936/9,625 update attempts and 1,677 response-bearing events. Its future-error and future-re-correction adapter-specific ΔΔ is +0.1883 with sequence-group 95% CI [+0.1664,+0.2114], while the intended direction is strictly negative; majority sequence support is false. C-RLS is +0.2044 [+.1783,+.2241]; APCR-S is -.0008 [-.0030,+.0004] and crosses zero. No-update exact-zero and unaffected-identity gates pass, but the primary LCIA gate fails.
- **Repair note:** The first cached short replay exposed a missing two-dimensional-to-batched LCIA relation adapter; `delta_numpy` now accepts `[candidate, feature]` input and preserves the one-dimensional output. The N28-A smoke and the complete B replay were rerun with the same frozen cache, seeds, update rule and gate. The external cache's identity-episode order was also handled explicitly rather than incorrectly requiring global frame order; per-identity chronological validation remains strict.
- **Failure reason:** `SCIENTIFIC_GATE_FAIL`, not implementation/alignment failure. The same continuous command wrote `outputs/n28/n28c_result.json` as `NOT_AUTHORIZED_N28_B_GATE_FAIL`; N28-C meta-training did not start. `val25_read=false`; real SAM3/FULL_LOOP, TrackEval and automatic arbitration remain unauthorized.
- **Keep:** YES — retain [n28b_result.json](/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/outputs/n28/n28b_result.json), response arrays, controls and failure evidence. Do not treat the cached feasibility result as executor-level MOT performance or authorize N28-C without a newly frozen protocol.

## 2026-08-26 — N29 pretrained SAM3 propagation-decoder route

- **Hypothesis:** N28's cached random relation challenger cannot reject a pretrained official SAM3 decoder. A stable pretrained propagation decoder with identity-scoped exact LIT-style Q/K/V A+B adapters, legal spatial correction supervision, atomic B10 updates and one global candidate assignment should provide a causally testable online route.
- **Modification:** Audited the pinned official SAM3/SAM3.1 source and local checkpoint; wrote `docs/N29_OFFICIAL_CODE_AUDIT.md` and `outputs/n29/frozen_protocol.json`; implemented `sam3_decoder_lit.py`, `corrected_mask_teacher.py`, `decoder_update_transaction.py`, `decoder_candidate_bridge.py`, and the N29-A/B/C/D/E runners. The 21 exact Two-Way Transformer projection targets are external identity-scoped states; the full official decoder base, interactive executor, CLIP-ReID and B10 remain frozen. A pre-existing `third_party/sam3/sam3/perflib/fused.py` user modification was preserved.
- **N29-A result:** PASS. Rank 4 has 35,328 A+B parameters (42 tensors), rank 8 would have 70,656; zero-LoRA equivalence, bias preservation, zero base gradients, B-gradient reachability, current-frame causality, future-only effect, candidate matrix and one global Hungarian assignment all pass. The fixture used three seeds and makes no dataset claim.
- **N29-B result:** PASS as a bounded official-path mechanism pilot. On CUDA device 8, one DanceTrack `train_fold` sequence (`dancetrack0001`), frames 0–5 and one identity, the official runtime exposed propagation inputs to `MultiplexMaskDecoder`; one `BOX_DERIVED_PSEUDO_MASK` correction committed five decoder update steps at adapter version 1. The external adapter inventory is 35,328 parameters with 42 trainable tensors and one identity state. The semantic box path returned no SAM output, so the declared official singleton `add_new_masks` rectangle binding was used; no slot/public-ID guess or third-party edit was made.
- **B diagnostic result:** Anchor and adapted future box IoU are both 0.0 over frames 2–5, each has four `IoU<0.5` errors, and future error delta is 0.0. The decoder candidate reached the two-candidate, three-column matrix and global assignment, but no mechanism benefit or full MOT gain is claimed. No clicks, confirmed masks, real multi-seed result, rank-8 result or seconds were measured.
- **Automatic transition:** Because the B mechanism/binding gate passed, the same workflow entered N29-C. C recorded `NOT_RUN` because there is no complete delivered full-video candidate tape with public-ID trace, recovery and reactivation callbacks. N29-D also recorded `NOT_RUN` because no legal official-decoder support/query episode manifest exists. TrackEval was not authorized.
- **Failure reason:** NOT_RUN_FULL_LOOP_AND_TRACKEVAL is a missing-input/benefit gate, not an implementation failure. `val25_read=false` throughout; no validation/test content was opened. The final report is `docs/N29_FINAL_REPORT.md`, with machine-readable results under `outputs/n29/`.
- **Keep:** YES — retain the official code audit, exact decoder adapter implementation, transaction/bridge tests, bounded real pilot and explicit C/D/E `NOT_RUN` evidence. Do not report the pilot as an end-to-end MOT improvement; next work requires a full train-fold official candidate/public-ID tape and legal confirmed-correction replay before TrackEval.

## 2026-08-26 — N29-R metric validity, real optimization and paired future audit

- **Hypothesis:** After separating DanceTrack identities from public/SAM object IDs, observable decoder optimization on difficult current-time errors could be tested against a correction-write-only control without leaking future selection.
- **Modification:** Added `ReplayIdentityBinding` and corrected `_trial_outputs` to query GT by `dataset_identity` while selecting predictions by `public_id`; added the two direct identity/missingness regressions and the offline N29-B erratum. Added deterministic support/logit/mask/parameter diagnostics, a finite/nonzero/decreasing-support commit validator, atomic rollback, and zero-update controls. The official runtime stream is explicitly closed before adapter creation, and the equivalent batched-matmul LoRA delta avoids the pinned runtime's non-differentiable nested `F.linear` path. No third-party source was changed; the pre-existing `third_party/sam3/sam3/perflib/fused.py` modification remains preserved.
- **R1/R2 result:** Both regressions pass (`5 passed` including optimizer controls). The old N29-B IoU=0 claim is invalid: corrected visible IoU is `.9526386` on frames 2–5 for both anchor and adapted boxes, with elementwise-identical boxes. The official pilot now has five finite update steps, parameter L2 delta `.0758207`, logit L-infinity delta `.25`, three changed binary pixels, and deterministic support loss `.4581505 -> .4575081`; zero-update and strict-future activation tests pass. Support fitting and engineering connectivity are therefore proven, not performance benefit.
- **R3 result:** A causal train-fold manifest was frozen before replay with 50 episodes across 10 sequences, 48 current-IoU triggers and 2 occlusion-recovery triggers; every episode has a 20-frame future query and uses `BOX_DERIVED_PSEUDO_MASK`. The five paired branches completed for all 50 episodes. Correction write-only improves over the difficult anchor, but LoRA-minus-write-only visible-IoU deltas are `-.0001849 [-.0009091,.0003486]` at H5, `.0011958 [-.0004205,.0039660]` at H10 and `.0001472 [-.0015783,.0018952]` at H20 (episode bootstrap 95% CIs); negative-transfer rates are 16%, 26% and 28%. Forty-nine updates commit and one is correctly rolled back because deterministic support loss increased `.8066837 -> .8073753`; the strict all-episode mechanism gate and future-benefit gate both fail. This is a scientific no-benefit result, not a hidden or skipped failure.
- **R5/R4 result:** The real train-fold association audit passes as `REAL_ASSOCIATION` on three predeclared two-identity cases (six identity bindings, six frames with official decoder candidates), using frozen CLIP-ReID/B10 evidence, audited N26 explicit negatives where available, original+official candidates, one global Hungarian with explicit NONE, and `relation_cache_used=false`; unaffected-identity regression is false. It is not `FULL_LOOP_DELIVERED`; TrackEval and end-to-end MOT benefit were not claimed. Accepted/confirmed/oracle masks are unavailable under the authorized box-only DanceTrack protocol, so N29-D is `NOT_TRIGGERED_NO_CONFIRMED_ORACLE_SUPERVISION` and no offline training starts. All artifacts are under `outputs/n29r/`; `val25_read=false` remains true.
- **Keep:** YES for the corrected evaluator, atomic diagnostics/rollback, official decoder integration and failure evidence. Do not retain an online-LoRA performance claim: the current result proves support fitting in most hard episodes but not a stable strict-future mechanism benefit or end-to-end MOT improvement. The next single line is a newly frozen confirmed-mask/teacher or offline episodic protocol, not larger blind replay.

## 2026-08-26/27 — N30 correction-state attribution and strict-future writer

- **Hypothesis:** The approximately five-point N29-R write-only gain is caused by a specific correction state, and an offline correction-conditioned writer trained on future query supervision can improve beyond that state without the instability of current-frame online LoRA.
- **Modification:** Froze and replayed the N29-R 50-episode train-fold manifest through six single-identity branches (A--F), recording target-scoped backend/official tracker/LoRA deltas. Added the real 10-case multi-identity M0--M4 spatial-vs-B10 ablation with one global Hungarian assignment. Added the sequence-disjoint N30 H20 writer split (20 meta-train, 4 selection, 4 calibration), the 991,905-parameter fp32 correction-memory writer with strict-future transaction, a 20-episode overfit gate, and a one-GPU formal training runner. The pre-existing `third_party/sam3/sam3/perflib/fused.py` user modification was preserved; no third-party source was edited.
- **N30-A result:** All 50 episodes passed all six branches. The singleton path explicitly records `b10_update_called=false` / `NOT_APPLICABLE_SINGLE_ID_NO_STATE_MANAGER`; the 50 current write-only runs called the official prompt path and changed only backend+official state, while the LoRA branch added only LoRA state. At H20, official `correct_object` alone is `-.566856` vs anchor, forced singleton reprompt is `-.601583`, current write-only is `+.049517` with episode bootstrap CI `[-.001409,.108233]` and sequence-cluster CI `[-.010251,.115371]`, local bookkeeping alone is `0`, and online LoRA minus write-only is `-.000440` with sequence-cluster CI `[-.001597,.000220]`. Thus the singleton write-only gain is not a B10 claim; the extra binding/reconstruction path is material.
- **N30-B/Gate-A result:** The 10 fixed two-identity train-fold cases pass. M1 official spatial write-only improves future delivered box IoU by `.320049` with sequence CI `[.192184,.485238]`; M2 B10-only is `0`; M3 joint minus the better single write is `.011235` with episode CI `[-.092473,.138863]`; M4 online LoRA minus M3 is exactly `0`. Gate A is `OFFICIAL_TRACKER_STATE_DOMINANT`, so the writer is authorized only as a spatial residual ablation.
- **N30-D/E result:** The frozen H20 manifest is sequence-disjoint and has 28 PASS samples (20/4/4), all query frames strictly after the correction; the final tensor index is PASS 28/28. Gate 2 passes: zero-init equivalence, writer-only gradients (31 parameter tensors; decoder trainable count 0), 400/400 future activations, protected-slot isolation, save/load equality and query loss `0.022349 -> .015312`; the train-only decoder-mask-derived proxy C-B gain is `.026065` with episode CI `[-.027666,.087774]` and 45% negative-transfer episodes, which is mechanism evidence only.
- **N30 training/Gate-3 result:** After Gate 2, the one-GPU CUDA device-8 run used AdamW (`lr=1e-4`, weight decay `1e-4`, clip `1.0`), 30 epochs, 600 steps, rotating H20 future indices, and frozen official decoder; loss fell `.196260 -> .089058`, elapsed `71.30s`, peak allocated memory `4,071,652,352` bytes, and `n30_writer_best.pt` was saved. On selection, D-B proxy is `+.067179` across 80 rows but sequence CI `[-.065244,.282931]`, negative-transfer sequence rate `75%`; calibration is `+.0000089` with CI `[-.014704,.013861]` and 50% negative-transfer sequences. Mean/success/missing criteria pass, but CI, negative-transfer and real unaffected-identity criteria fail/not-run (singleton tape; protected-slot unit passes). Gate 3 is therefore `FAIL`; full-loop/TrackEval was correctly not run.
- **Failure reason:** `SCIENTIFIC_GATE_FAIL`, not hidden skipping or evaluator failure. The learned writer overfits the frozen training proxy and can produce a large selection average driven by one sequence, but does not show sequence-stable future benefit beyond the official write-only state. The N30 failure therefore invalidates the learned-writer performance hypothesis, not the causal official correction path.
- **Keep:** YES for the state audit, multi-identity attribution, strict-future tape, writer implementation, gates and failure evidence. Keep the official tracker write-only path as the current mainline. Do not claim B10, online-LoRA benefit, offline-writer benefit or end-to-end MOT improvement; do not open val25/test. Final report: `docs/N30_FINAL_REPORT.md`.

## 2026-08-27 — N31 correction-state writer/selector causal audit

- **Hypothesis:** After removing N30's public-ID admission and FULL/SPLIT continuation confounds, a target-scoped official correction-state candidate should expose a sequence-stable future-quality signal that a causal writer or selector could learn beyond the strongest fixed write.
- **Modification:** Added the centralized backend `_bind_external_sam_id` mapping helper and singleton/ALREADY_BOUND restoration, strict next-frame resume preparation, lightweight continuation snapshots, target-scoped official `add_new_masks(..., reconditioning=True)` writes, local official interactive-SAM multimask candidates, P0--P7 ablation, candidate rollouts, the expanded train-fold manifest, future-gradient smoke, explicit selector/fallback gates, and a resumable overnight orchestrator. The pinned `third_party/sam3` checkout was not edited; the pre-existing `sam3/perflib/fused.py` user modification was preserved.
- **N31-A/B result:** Mapping regression passed, including BOUND, ALREADY_BOUND and raw-output invariance. FULL/SPLIT no-op equivalence passed on three N30 hard episodes after one minimal repair cycle, selecting `strict_next_frame_resume`; all branches therefore share the same correction prefix and continuation protocol.
- **N31-C result:** All 50 N29-R hard episodes completed P0--P7 and the State Gate passed. P1/P0 raw future signatures are exactly equal. At H20, P0 IoU is `.766850`; P4 corrected rectangle is `.500706` unconditional but `.075451` better than P3 conditional on both outputs (sequence CI `[.013868,.156237]`); P5 current ensure is `.816367`; P7 frozen interactive-SAM pseudo-mask is `.513646`; P6 online-LoRA minus P5 is only `.000299` with sequence CI `[-.001410,.002016]`. P2 prompt success/target-state rates are `.20/.20`, while P3 restores target-state presence to `1.0`; this admission/recovery effect is not called mask quality. The real two-ID protected scope passes with unaffected IDs and per-ID state signatures preserved.
- **N31-D result:** The corrected candidate implementation uses decoder predicted-IoU rank order for S3/S4/S5 and scalar causal features; the formal rollout then completed 300 rows (50 episodes × 6 candidates), with all candidates available. S0 restore-old-state has mean reward `.737147`; S1/S2/S3/S4/S5 have `.280874/.266059/.295190/.291242/.255224`. Every alternative has negative H20 IoU gain versus S0; the strongest alternative is S3 at `-.253204` with sequence CI `[-.333807,-.171602]`. Oracle Gate is `FAIL`, so the candidate library has no demonstrated future-quality upper bound and no larger learned selector is authorized.
- **N31-E/F result:** The frozen sequence-disjoint manifest retains all 689 legal train-fold events over 30 parent sequences with fixed 18/6/6 split, counts `419/114/156` for meta-train/selection/calibration. The meta-train target of 500 is not reached under the already frozen uneven sequence partition; no duplicate events were created and the shortfall is reported. No usable local BDD100K/TAO source was found; the local KITTI label directory was empty. The official future-gradient smoke fails honestly (`requires_grad=false`, writer gradient count `0`, inference-mode/detached path), but Oracle failure takes precedence over selector training; Path A and Path B training are both explicitly `NOT_RUN_ORACLE_FAIL`.
- **Fallback result:** The bounded real multi-ID train-fold association evidence is retained as the final `association_trigger_fallback`: 10 cases, official spatial write-only future IoU `.320049` with sequence CI `[.192184,.485238]`, joint-minus-best-single `.011235` with episode CI `[-.092473,.138863]`, and online-LoRA-minus-joint `0`. This is not a learned correction-state or end-to-end MOT result. Full-loop/TrackEval is `NOT_RUN_LEARN_GATE_FAIL`; ICLR/end-to-end claims remain unestablished.
- **Failure reason:** `SCIENTIFIC_GATE_FAIL` at the candidate Oracle stage, not a hidden implementation failure. The causal state path is real and the state gate passes, but every current-information candidate is worse than restoring the old state on the frozen hard fold. Future gradient is also unavailable through the official inference path. The selector artifacts explicitly record `NOT_RUN_ORACLE_FAIL`; the final bounded route is association/interaction timing, not a larger writer.
- **Resources/keep:** Formal N31-C took `7750.845674s`, formal N31-D `3971.552922s`, and resume gate `383.482265s` on CUDA device 9; the 40 GiB disk reserve remained satisfied. Keep the mapping/resume fixes, target-scoped writer, official decoder audit, candidate evidence and failure artifacts. Do not deploy a correction-state writer/selector or claim B10/LoRA/full-loop benefit. Final report: `docs/N31_FINAL_REPORT.md`.

## 2026-08-27 — N32 strategy-level correction application selector gate

- **Hypothesis:** A causal current/past correction-state feature tape may allow selection among `K0_KEEP_OLD`, `K1_APPLY_ENSURE` and `K2_PROMPT_THEN_RESTORE` without future-information leakage, subject to a policy Oracle and sequence-stable selector gates.
- **Modification:** Completed the frozen 689-episode policy-level retry, semantic reconciliation and merge. The one allowed OOM repair released correction-only `prefix_snapshot`/`prefix_outputs` and temporary feature containers before future propagation, followed by garbage collection and CUDA cache release; the exact smoke passed on attempt 2. Added shared null-denominator semantics, CPU-only B1/B2 reconciliation, canonical 689/2067 strict validation, explicit identity-coverage reporting, and protocol-authorized route artifacts.
- **Policy evidence:** The retry manifest had 268 `(episode, policy)` items. Reconciliation found 264 raw strict PASS, one skipped-existing PASS, two legal B1 zero-visible windows and one legal B2 safe-rollback zero-visible window; real A failures were zero. The supervisor's `PARTIAL` summary is retained as a non-policy artifact and excluded from the 268 policy count. Merge passed with 689 episodes, 2,067 policy rows, zero duplicates/missing/unavailable/NOT_RUN/PARTIAL rows, three legal zero-visible windows and 854 legal undefined drift metrics.
- **Oracle result:** The 689 policy Oracle completed post hoc with gain `0.0152976`, 22 positive sequences and 688 defined H20 IoU episodes, but `winner_not_best_fixed_rate=0.202035` is below the frozen `0.30` gate. Oracle status is `FAIL`; no selector training or temporal fallback was authorized. The route therefore entered the bounded real multi-ID association fallback, which executed as `PASS`/`DIAGNOSTIC_ONLY` and makes no full future-MOT gain claim.
- **Identity limitation:** The feature audit is structurally `PASS` with 30-dimensional, policy-independent finite vectors, but `identity_features_available_episode_count=0` of 689 and coverage `0.0`; all identity feature slots are zero-filled. The selector is explicitly scoped as `TEMPORAL_GEOMETRY_ONLY_FALLBACK`, not identity-aware learning. The frozen selector Learn Gate has no identity-coverage-greater-than-zero requirement, while the full-loop gate still requires primary identity-metric improvement; no identity-aware or full-loop claim is made.
- **Blind boundary/keep:** `val25_read=false`, `test_labels_used=false` and `future_gt_used_for_selector_input=false` throughout. Keep the retry memory repair, semantic validator, reconciliation/merge audit, policy Oracle failure evidence and bounded association fallback. Do not deploy the untrained selector, temporal selector or full loop; the next credible line requires a candidate-complete multi-identity correction tape with actual identity-memory coverage.

人工已经提供 public_id；本任务不是判断框属于谁，而是把人工确认框的真实外观证据写入该 ID，并让它只影响后续帧的关联。

## 2026-08-28 — N34 CCAM mechanism-first pipeline and real-tape authorization gate

- **Hypothesis:** A candidate-complete, correction-conditioned identity tape could make N33 CCAM's future-only effect measurable across multi-identity sequences and authorize a small calibration head before any decoder LoRA.
- **Modification:** Audited the N33 callers and artifacts; added N34 train-fold inventory, backend capability probe, explicit real-tape/event-tape sentinels, a four-event synthetic transaction fallback, M0--M4 paired replay, and the final calibration/LoRA authorization gate. Reused the single N33 `AppearanceMemory` and added only the narrow `ADD_NEW_IDENTITY` validation exception required for a new public ID created by a current spatial transaction. No N32 Oracle/selector or decoder LoRA job was started.
- **Real data result:** DanceTrack `train/train_fold` contains 40 multi-ID sequences. Twenty-four selected sequences satisfy at least two simultaneous IDs, reusable-cache candidate-competition proxy, and H20/H50/H100 future-window availability. The N25R cache has 27 sidecar sequences (24 with competition proxy), but it is episode-window/top-k only, has `selected_obj_id` coverage 0, and cannot provide a valid public-ID mapping or every per-frame SAM3 candidate.
- **Interface result:** The pinned adapter's observation type exposes frame/id/mask/box/confidence/source, not candidate embeddings, decoder tokens or public-ID score matrices. The adapter does expose `offload_video_to_cpu=True`, `async_loading_frames` and conditional `offload_output_to_cpu_for_eval`; these do not create the missing all-candidate identity tape. `CANDIDATE_COMPLETE_TAPE=NOT_AVAILABLE` is therefore a code-contract/dependency result, not an unattempted old-script result.
- **Fallback result:** The four synthetic transaction types (ADD_NEW_IDENTITY, AUTHORITATIVE_REASSIGN, ATOMIC_ID_SWAP, RECOVER_IDENTITY) pass full-loop invariants through 29 future frames. Four synthetic events × five M0--M4 variants run paired replay through 100 future frames, covering H20/H50/H100; M0 has zero delta and M1--M4 show only synthetic score-delta smoke. Real IoU, missing, IDSW, re-correction, recovery latency and sequence-cluster CI remain NOT_COMPUTABLE.
- **Gate result:** Real multi-ID data is PASS; real candidate-complete tape is NOT_AVAILABLE; synthetic full loop/replay are PASS fallbacks; calibration head and decoder LoRA are NOT_AUTHORIZED. N32 identity feature coverage remains 0/689, and the allowed route is explicitly temporal/geometry-only association fallback, not identity-aware learning. Final N34 status is `PARTIAL`; no performance or end-to-end MOT claim is made.
- **Reproducibility/keep:** `12` focused N33+N34 tests and `38` shared N7/N8/interaction/human-box tests passed. `val/test` content was not opened, no long GPU job was started, and the pre-existing `third_party/sam3/sam3/perflib/fused.py` modification was preserved. Keep all N34 audits, sentinels, synthetic ledgers and gate artifacts. The next action is to export a real per-frame SAM3 candidate-complete public-ID tape with independent human ROI events on DanceTrack `train/train_fold`.

## 2026-08-28 — N35 backend export audit

- **Hypothesis:** The pinned SAM3 full-VG response already contains per-frame candidate geometry/masks/scores, so the N34 blocker may be repairable in the project adapter without editing third-party SAM3.
- **Modification:** Added `scripts/run_n35_backend_audit.py`, which records the real `_send_prompt`, `_parse_outputs`, `_apply_stable_ids`, `_output_cache`, `propagate`, `StateManager` candidate audit and pinned official predictor contracts. The audit does not load the checkpoint or read dataset frames.
- **Result:** Official full-VG output exposes `out_obj_ids`, `out_probs`, `out_boxes_xywh` and `out_binary_masks`; the adapter converts these to active-object observations but has no candidate exporter, embedding/decoder token, public-ID mapping or score-matrix export. The pinned `init_state` supports video CPU offload and async frame loading but not state CPU offload. The minimum N35 repair is an adapter-level exporter retaining all candidates, with an explicitly marked machine box-crop feature fallback and separate human ROI extraction.
- **Keep:** YES — N34 real-tape sentinel remains immutable; proceed to one-sequence real SAM3 exporter smoke before any long train-fold export.

## 2026-08-28 — N36 independent sharded real tape, full loop and CCAM future gate

- **Hypothesis:** Reinitializing the pinned official SAM3 predictor for bounded overlapping frame ranges can remove N35's long-sequence state/OOM bottleneck while preserving all per-frame candidates and enabling a causal known-public-ID correction replay.
- **Modification:** Added an adapter-level N36 exporter with one child process/session per 160-frame range and 20-frame overlap, atomic chunk/done artifacts, explicit local-native to sequence-global boundary matching using box/mask/machine-embedding evidence, and a CPU-only chunk/merge validator. The adapter keeps official video CPU offload and output offload, does not claim unsupported state CPU offload, and no `third_party/sam3` file was edited. Added offline train-fold event construction, human-box OSNet extraction, real M3 full-loop execution to sequence end, and real M0--M4 paired replay with sequence-cluster bootstrap.
- **Real tape result:** The complete selected train-fold set passed: 24/24 sequences, 203/203 chunks, 26,691/26,691 frames, 193,969 candidate rows, zero duplicate/missing/unavailable/error rows, and runtime GT reads `0`. The required `dancetrack0001` full 703-frame sharded smoke and the intermediate one-sequence and six-sequence validator ladders are retained. Empty-candidate mapping compatibility is recorded as warnings rather than silently treated as candidate loss; one native/public transition is retained as identity-switch evidence.
- **Event/full-loop result:** The offline event manifest contains 7 legal simulated human events over 6 independent train sequences: `ADD_NEW_IDENTITY=1`, `ATOMIC_ID_SWAP=1`, `AUTHORITATIVE_REASSIGN=4`, `RECOVER_IDENTITY=1`. Prefix state is strictly before each event; human features are extracted from the explicit human box crop and never copied from machine candidates. The runtime view strips offline dataset IDs, human vectors and N8 diagnostics before calling the intervention path; M3 full-loop completed all 7 events to each sequence end. Spatial correction, public/native mapping, new-ID/recovery/swap transaction checks, write-after-spatial ordering, current-frame write invisibility, t+1 future availability, no duplicate public IDs and runtime future-GT absence all passed. The first full-loop audit attempt retained two false FAIL records caused by requiring anchors on protected old IDs; the corrected audit checks only transaction-added IDs and the formal rerun is 7/7 PASS.
- **CCAM result:** All 35 real variant replays (7 events × M0--M4) passed candidate validation and strict future execution. M2 H20 identity-utility cluster mean is `+0.013116` but its 95% sequence-cluster CI is `[0, 0.039347]`; M3 and M4 H20 deltas are `0` with lower CI `0`. Protected-identity regression is false for M2/M3/M4 and the replay leakage checks pass, but the frozen requirement `M2/M3/M4 lower CI > 0` is not met. Therefore `CCAM_EFFECT=NULL`, `CALIBRATION_HEAD=NOT_AUTHORIZED`, and `DECODER_LORA=NOT_AUTHORIZED`; no training or selector was started.
- **Repair/failure evidence:** The first offline event build rejected a frame-25 `dancetrack0012` candidate with no legal pre-event public state; it was retained as a rejection and a later valid event was selected. The first full-loop run retained `full_loop_event_ledger_attempt1.jsonl`; its only failures were the audit-condition bug above. The first replay smoke exposed a missing `FEATURE_DIM` import, then the same smoke passed before the full 35-replay run. These are preserved execution/audit facts, not converted into PASS by overwriting evidence.
- **Resources/keep:** Every SAM3 chunk used an independently exited process and only authorized GPU 0--3 scheduling; no concurrent GPU task was killed. Keep the N36 tape, boundary records, validators, event manifest, full-loop/replay ledgers and strict no-authorization gate. The N36 infrastructure/evaluation is complete, but the identity-memory future-benefit hypothesis is not supported by the required sequence-stable CI; retain the official write path without calibration/LoRA.

## 2026-08-29 — N37 event expansion blocked by real atomic precondition

- **Hypothesis/protocol:** N37 was intended to expand the real train/train_fold event tape to at least 24 events across at least 12 independent sequences, then run causal M0--M4 CCAM replay. The N36 candidate tape, checkpoint, action quotas, future windows and evaluation definitions were frozen and reused; no val/test data or runtime future GT was used.
- **Stage A result:** All 24 source-sequence scan artifacts were available and the deterministic frozen selection produced 24 unique sequence slots with action counts `ADD_NEW_IDENTITY=5`, `AUTHORITATIVE_REASSIGN=4`, `ATOMIC_ID_SWAP=4`, `RECOVER_IDENTITY=11`. Materialization stopped after 5/24 events at `dancetrack0015:772` `ATOMIC_ID_SWAP`; 19 events remain not attempted.
- **Root cause and repair audit:** The first static precondition probe had a real audit bug: it treated `pre_rows[other_auto_tid]` as the other event-frame box, while `build_event` uses the offline event-frame GT box for `other_dataset_gt_id`. The static checker was corrected without changing the N36 builder. A same-input consistency audit then matched the corrected static path and an independent exact builder-precondition reproduction for all four frozen `dancetrack0015` atomic candidates; each maps target and other to the same current public ID, so all four are illegal (`0` valid, `4` rejected). The original materialization failure was retained and the corrected `772` targeted regression reproduced the same `ValueError`.
- **Gate result:** No same-sequence/same-action replacement exists. Cross-sequence fallback, quota/protocol/future-window changes, replay-based selection, and synthetic substitution were prohibited. N37 is therefore `BLOCKED` at Stage A; full loop, paired replay, calibration, selector and decoder LoRA are `NOT_RUN`/`NOT_AUTHORIZED`. This is not a CCAM future-effect conclusion.
- **Evidence/keep:** Keep `outputs/n37/atomic_id_swap_precondition_consistency_audit_attempt1.json`, `outputs/n37/atomic_id_swap_targeted_regression_attempt2.json`, `outputs/n37/atomic_id_swap_replacement_blocked.json`, the original `event_materialization_failure_n37-dancetrack0015-0772-atomic_id_swap-001.json`, all N37 stage statuses and `docs/N37_FINAL_REPORT.md`. N36 artifacts and `third_party/sam3` were not modified.

## 2026-08-29 — N37 global atomic repair, full-loop and final CCAM gate correction

- **Correction to the preliminary entry above:** The first Stage A block was not a final N37 conclusion. The required read-only global audit first recorded `selected_sequences.json` `KeyError:'name'`; schema inspection showed the frozen field is `sequence`, so the error was a field-mapping diagnostic, not candidate insufficiency. The all-24 audit then examined 894 stored atomic candidates, found 36 deterministic replacement-eligible candidates, and selected four legal same-action replacements in the frozen global order. No replay result, quota change, future-window change, or synthetic event was used.
- **Stage A result:** The repaired canonical manifest passed with 24/24 events, 21 independent sequences, exact action counts `ADD_NEW_IDENTITY=5`, `ATOMIC_ID_SWAP=4`, `AUTHORITATIVE_REASSIGN=4`, `RECOVER_IDENTITY=11`, zero duplicate event IDs, and `runtime_future_gt_used=false`. The original `dancetrack0015:772` and `dancetrack0016:397` failures and all preliminary audits remain preserved.
- **Stage B result:** The N37 M3 full loop completed 24/24 events to sequence end. Spatial correction, post-correction memory ordering, hidden current-frame write, event+1 causal boundary, mapping completeness, no duplicate public IDs, and runtime future-GT checks all passed. A wrapper smoke initially failed because its allow-list omitted the explicit `future_gt_used_runtime=false` audit flag; the same event passed after the minimal wrapper-only repair and two targeted regressions.
- **Stage C result:** The first 24×5 replay attempt completed 21/24 events and retained three real validator failures on frame-0 `ADD_NEW_IDENTITY` events: the frozen validator incorrectly required a non-empty prefix even though a new identity has no pre-event state. A narrow validator repair allows an empty prefix only for `ADD_NEW_IDENTITY` with an explicit current-frame correction for that new public ID; the three-event M0–M4 targeted regression passed. The raw first replay result/artifacts were preserved. The corrected run passed 120/120 unique `(event_id, variant)` artifacts, each with a 100-frame future trace and compact per-frame candidate mapping/assignment audit. GT was loaded only after all five variants for each event and used only for post-hoc scoring.
- **Final effect:** M2 H20 sequence-cluster mean is `-0.0105888782`, 95% CI `[-0.0317666345, 0]`; H50 mean/lower is `-0.0107347955/-0.0322043864`; H100 mean/lower is `-0.0085224898/-0.0255674694`. M3 and M4 H20 means/CIs are `[0,0]`. Protected-identity no-obvious-regression and leakage checks pass, but the strict M2/M3/M4 lower-CI-greater-than-zero gate fails. All 24 events have a defined H20 visible target window (minimum 7 visible frames), so this is not a zero-visible denominator artifact. The mechanism decomposition shows appearance scores changing at the first future frame in enabled variants, but assignment changes only once (M2), with the only nonzero H20 effect negative.
- **Decision/keep:** Final status is `PARTIAL` with `execution_complete=true` and `research_gate=FAIL_FUTURE_EFFECT`. Calibration head, selector, and decoder LoRA remain `NOT_AUTHORIZED`; no learning was run. Keep the global pool audit, corrected manifest, full-loop/replay artifacts, raw attempt-1 replay evidence, validator repair, and all N36/third-party boundaries. Final gate/report: `outputs/n37/n37_final_gate.json` and `docs/N37_FINAL_REPORT.md`.

## 2026-08-29 — N38 mechanism diagnostic blocked by frozen N37 artifact schema

- **Hypothesis/protocol:** N38 registered a `0.05` normalized target-state score margin, different top-two source public IDs, and a joint current-frame plus event+1 near-tie requirement before scanning. The only intended question was whether CCAM score changes can cross the existing Hungarian assignment boundary; checkpoint, candidates, embedding, solver, metrics and strict gate stayed frozen.
- **Stage A result:** The read-only audit found all 120 canonical N37 `(event, variant)` keys for 24 events and 24,000 future branch frames. The generated table has 24,120 unique rows (including current-frame placeholders), zero key/trace-alignment errors and zero runtime future-GT reads. Preserved N37 raw attempt-1 success artifacts provide lossless per-candidate audit for 105 artifacts; the remaining 15 artifacts are the five variants of three earlier failed ADD events and only have compact canonical traces.
- **Actionable root cause:** Every canonical N37 future trace explicitly excludes the event frame, so current-frame score/rank/Hungarian evidence is `0/120`. The compact fallback also lacks per-candidate IDs, scores, ranks and cost matrices. The detailed future-only scan reports provisional event+1 near-tie counts `M0/M1/M2/M3/M4=1/2/2/1/1` among the 21 detailed events, but the current-frame conjunct is unavailable; these were not selected or treated as N38 near-tie events.
- **Gate/decision:** N38 is `BLOCKED` at `N38-01` with `BLOCKED_INPUT_ARTIFACT_SCHEMA`; event selection, full-loop, M0–M4 replay, calibration, selector and decoder LoRA are `NOT_RUN`/`NOT_AUTHORIZED`. This is not a new scientific negative result and does not alter N37's frozen `FAIL_FUTURE_EFFECT` conclusion. The first N38 summary-count bug and the following `UnboundLocalError` execution failure are preserved with corrected rerun evidence; N36/N37 and `third_party/sam3` remain unchanged.
- **Keep/next:** Keep `outputs/n38/diagnostic/`, the failed-attempt evidence and Stage 02–05 statuses. A future continuation requires only a lossless event-frame/event+1 diagnostic sidecar for the same frozen 24×5 protocol; no threshold relaxation, event substitution or downstream training is authorized.

## N38R1 — Lossless diagnostic sidecar recovery (2026-08-29)

Hypothesis: the N38 Stage-A block was recoverable by replaying the frozen N37 24×5
protocol with an event-frame-inclusive, lossless diagnostic sidecar. R1 completed 120
event×variant artifacts over 24 events/21 sequences and 24,000 future frames with
184,140 finite candidate records, preserved mask hashes, complete mappings, and zero
runtime future-GT use. R2 completed 24,120 unique diagnostic rows after two preserved
generator repairs: event-frame row expansion was fixed, then dynamic public-state axes
were aligned by public ID rather than raw column index. Under the pre-registered M0,
memory-write=False baseline selection stream, only 1 event(s) met the
event-frame AND event+1 near-tie conjunction; fixed quotas (24 events, 16 sequences,
four per action) were not met. R3 therefore stopped before R4, with no future-effect
gate result and no authorization for calibration/selector/LoRA. Events remain labeled
simulated_from_gt. New evidence is under outputs/n38r1; N36/N37/N38 inputs remain
read-only. Minimal next step: recover/collect a protocol-approved near-tie candidate
pool without changing the frozen threshold or event protocol.

## 2026-08-29 — N39 weighted association-interface probe

- **Hypothesis/protocol:** N39 separated the internal `AppearanceMemory.human_weight` from the external `appearance_score_weight` (`lambda_assoc`) as possible causes of N37/N38R1's score-change-without-assignment-change behavior. The checkpoint, candidate stream, prefix, event manifest, future windows, Hungarian solver, metrics, bootstrap seed/repetitions and strict gate were frozen. The 24 N37 events over 21 sequences were reused exactly; all events remain `simulated_from_gt`, not historical human clicks.
- **Stage 1 scale result:** The CPU audit produced `869,797` candidate-level rows from `120` N38R1 artifacts, replayed `96` component matrices with maximum absolute difference `0.0`, and found zero causal violations and zero runtime future-GT reads. At the default scale, appearance delta median was about `.2213`, target-row `|delta|/margin` median `.0494` and p95 `.0909`, while assignment-margin median was `4.7091`; this supports a delta-smaller-than-margin interface diagnosis. The audit keeps machine-prototype, human-positive, negative, delta, fused and margin distributions by action, variant, horizon and target row.
- **Weight scan result:** The pre-registered values `{0,.25,.5,1,2,4,8}` were scanned separately for both parameters. The 42-worker smoke passed after one preserved worker-validator repair; the full scan passed `336/336` independent workers across `14` configurations. A first posthoc scorer run failed after computation because relative CLI paths were passed to `Path.relative_to(ROOT)`; the full traceback is preserved, path normalization passed targeted regression, and the same full input rerun completed successfully. No runtime/data failure or OOM occurred.
- **Future effect:** `lambda_assoc=0` removed appearance score and assignment changes. Higher lambda increased M2 H20 assignment-change rate to `.191667` at 4/8, and higher human weight increased it to `.166667` at 4/8. However, the largest observed M2 H20 utility mean was only about `+.030968` with sequence-cluster CI `[-.015175,.089059]`; the strict M2/M3/M4 lower-CI-greater-than-zero gate failed for every configuration. High-weight changes were heterogeneous (for lambda/human=4/8, M2 H20 had 44 correct versus 15 incorrect assignment changes in the posthoc classification), and no stable untouched-ID regression was detected.
- **Decision/keep:** Status is `COMPLETED_GATE_FAILED`, not resource BLOCKED. Score changes, assignment changes and correct assignment changes remain explicitly separate. Both internal and external weights can push some assignments across the boundary, but neither yields sequence-stable future benefit; candidate/base/association-boundary limitations remain plausible. Keep all N39 scale tables, frozen protocol, smoke/full manifests, worker artifacts, posthoc result and preserved failures. Calibration head, selector and decoder LoRA remain `NOT_AUTHORIZED`. The minimal next step is a protocol-approved real human event tape plus one frozen association-interface probe; do not use simulated events as historical clicks or bypass the gate. Final report: `docs/N39_FINAL_REPORT.md`.

## 2026-08-29 — N40 safe pause and N41 GT-controlled association-interface diagnosis

- **N40 state:** N40 remains safely paused at `PAUSED_WAITING_USER_DECISION` with zero real human event tape. Its `BLOCKED_INPUT_REAL_HUMAN_TAPE` and pause evidence are preserved. No synthetic event was created or imported during this continuation.
- **Hypothesis/protocol:** N41 separated the internal `AppearanceMemory.human_weight` from the external `lambda_assoc` and tested whether appearance evidence can cross the existing assignment boundary. N37's 24 events over 21 independent sequences, checkpoint, candidate streams, prefix/future windows, M0--M4 definitions, Hungarian evaluation and strict sequence-cluster gate were frozen. GT was allowed only for a controlled current-event ROI source and post-hoc scoring; all events remain `simulated_from_gt`, not real human evidence.
- **Stage N41-01:** Parameter transfer and causal-boundary audit passed. `lambda_assoc={0,1,8}` scaled appearance deltas without changing base/memory totals; `human_weight={1,4,8}` scaled the positive human term exactly; mapping, hard negatives, candidate axes and event-frame write invisibility passed. The pair scan contains `16,014` rows, with H100 positive appearance direction `0.633982`, `2,125` base-wrong pairs and `291` correctable at lambda 8, but `510` base-correct pairs pushed wrong by some scanned lambda.
- **Stage N41-02:** Frozen sources A (ideal GT ROI upper bound), B (frozen N37 current human-region path) and C (fixed corrupted ROI) completed a three-event smoke and `144/144` full runtime workers, yielding `720` variant results. Runtime future-GT use was false and all candidate-stream audits passed. A/B feature digests are equal for all 24 events, so B is not an independent real-human quality condition; C has median A--C cosine `0.5834461`.
- **Future effect:** At lambda 1, M2 H20 mean is `-0.0105888782` with CI `[-0.0317666345,0]`, 34/480 assignment changes and 0 correct/12 incorrect classified changes. At lambda 8, M2 H20 mean is `+0.0309682346` with CI `[-0.0151752800,+0.0890590526]`, 92/480 assignment changes and 44 correct/15 incorrect changes. M3/M4 and all other source/configuration gates also fail the strict lower-CI-greater-than-zero rule. Score change, assignment change and correct assignment change remain distinct.
- **Decision:** The parameter path is valid and scale/boundary evidence exists, but high-weight crossings are not sequence-stably beneficial and ideal/current source are indistinguishable. N41 therefore does not implement a new production fusion interface and does not authorize calibration head, selector or decoder LoRA. No GPU was used for N41 diagnostics; no checkpoint, candidate definition, Hungarian solver or metric changed.
- **Failure/repair record:** Wrong-environment import, scorer-count expectation, gate-classification, two source-replay validator schema checks, finalizer aggregate-key mismatch and one malformed read-only jq query were retained in `outputs/n41/attempts/`; each had a minimal non-scientific repair and targeted validation. No N41 OOM, runtime GT leak, candidate loss or production-formula modification occurred.
- **Keep/next:** Keep all N41 diagnostic tables, source manifests, worker/post-hoc artifacts, gate statuses and failure evidence. The smallest credible next step is external collection of a provenance-complete real human event tape, followed only by a separately frozen association-interface probe. Do not treat simulated events as human clicks or start downstream learning. Final report: `docs/N41_FINAL_REPORT.md`.

## 2026-08-30 — N42 isolated T1 association calibration probe

- **Hypothesis/protocol:** N42 tested whether a separately trained association/fusion calibration head could convert the mixed appearance signal identified by N41 into reliable assignment changes. The 24 frozen N37 events over 21 train/train_fold sequences, candidate stream, prefix/future windows, M0–M4 variants, Hungarian solver, metrics and sequence-cluster bootstrap were retained. Runtime future GT remained false; all events remain `simulated_from_gt`, and real human tape remains `0`.
- **Diagnosis:** N41 parameter transfer passed. N42 corrected the preserved A/B source-construction defect: the prior A/B source digests were equal for 24/24 events, while the corrected 72-feature source manifest has zero exact collisions and finite unit-norm features. Pair diagnostics found appearance direction correctness around `0.634–0.640`, but only `42/2943` H20, `99/7674` H50 and `291/15879` H100 pairs were base-wrong and correctable at lambda `<=8`; candidate/base score scale plus the assignment interface remain the primary bottleneck.
- **Training:** Mandatory T1 was actually trained in the isolated `outputs/n42` tree. The pairwise head used a frozen 23-D causal feature contract, sequence split train/validation/holdout, 11,012 materialized rows, 6 completed epochs, and a GPU0 smoke/full run. Loss/gradient finiteness and checkpoint save/reload passed. The resulting checkpoint is trained but not production-authorized.
- **Evaluation:** T0/T1 runtime replay completed `48/48` workers, `240/240` posthoc variant results, and all 100-frame future traces. T1 changed the future score interface, but T1/M2 H20 utility was `-0.0092653` with sequence-cluster CI lower `-0.0317666`; T1/M3/M4 were null at the aggregate gate. Holdout lower CIs were not strictly positive, so the future-effect gate is `FAIL_FUTURE_EFFECT`; no calibration promotion, selector, decoder LoRA, weight expansion, checkpoint replacement or threshold tuning was performed.
- **Isolation/keep:** Project pre-existing code hashes, configs, shared checkpoints, N39–N41 evidence, sibling MOT metadata and third-party SAM3 boundaries were preserved; the existing test suite passed `113/113`. Keep the T1 checkpoint as an isolated negative/diagnostic artifact, all preserved N42 failure attempts and posthoc traces, and the final gate/report. The smallest valid next step is external provenance-complete real-human tape collection through N40, not another blind weight scan. Final report: `docs/N42_FINAL_REPORT.md`.

## 2026-08-30 — N43 full candidate×public-ID calibration sidecar

- **Hypothesis/protocol:** N42's negative effect could be caused by applying an ordered-pair aggregate only to the human target public-ID column. N43 froze the N42/N37 24-event, 21-sequence candidate stream and built an independent full-cell audit and sidecar; no production MOT/OVMOT or third-party SAM3 code was modified.
- **Diagnosis:** 2424 audited frames contained 173793 candidate×public-ID cells. Target assigned recall@0.5 was `0.415338`, while the candidate Oracle ceiling was `0.751831`. N42's old interface changed only target-column cells. N43 added causal geometry/motion/margin/reliability/age fields derived from audit candidates and previous-frame native boxes, plus explicit immutable `-1e8` NONE dummies; target/event/GT/future outcome were not features.
- **Implementation/training:** The bounded sidecar applies `S_ij=B_ij+sigmoid(gate_ij)*A_ij+residual_ij` to every finite cell, preserves hard negatives, and bounds residual to ±0.5. Full dataset materialization produced `122303` rows (15115 positive, 107188 negative, 19334 ambiguous discarded, 30882 public-ID-GT-unavailable cells). Actual full training ran on GPU0 with seed `4242`, 12 completed epochs, best epoch `7`; checkpoint SHA-256 is `1fb7e6cdd7bd36521203684f2d9d5bc5371af101626a3007623bdd0bc823d6a3` and remains non-production.
- **Replay/result:** Same prefix/events/candidates replay completed 24/24 events, all M0–M4 variants and H20/H50/H100 after runtime validation. Runtime future GT was false and GT was loaded only post-hoc. Corrected sequence-cluster bootstrap is sequence-mean first, then equal sequence resampling (seed `4242`, `2000` replicates); the old event-weighted result is preserved separately. Corrected M2 identity utility is `-0.009573/-0.009451/-0.007485` at H20/H50/H100, with current corrected CI lower bounds `-0.032469/-0.032206/-0.024435`; assignment changes had zero correct changes in all three horizons. M1/M3/M4 were also slightly negative and their untouched-ID checks were not clean. Standard IDSW/IDF1/HOTA/AssA remain not computable from bounded event windows.
- **Risk repair:** The post-replay margin regression now uses explicit valid column indices and excludes the current column by index. Repeated-score, non-adjacent-valid-column, and all-hard-negative cases pass; the failed system-Python invocation (missing torch) and all prior failure artifacts remain preserved. The replay script now refuses to overwrite a result if the preserved legacy event-weighted artifact is missing.
- **Failure/repair record:** Two Stage-01 attempts exposed degenerate audit boxes; geometry/motion was made explicitly unavailable. Stage-02 exposed renamed matrix-key compatibility. The first Stage-04 run left five partial artifacts before an outer-session termination without captured exit code; that evidence is preserved and no OOM claim is made. Explicit NONE/public-ID mapping was repaired, then the full 24-event replay passed structural checks. Finalizer syntax/schema defects were caught by py_compile/runtime checks and repaired.
- **Gate/keep:** Real human tape remains unavailable and no real SAM3 full-loop was run. N43 is `COMPLETED_GATE_FAILED`: full-cell interface repair did not produce a positive future effect or clean untouched regression, so calibration and decoder LoRA remain unauthorized. Keep all `outputs/n43` stages, audit, dataset, checkpoint, corrected and legacy replay results, targeted regression, preserved failures, [outputs/n43/n43_final_gate.json](outputs/n43/n43_final_gate.json), and [docs/N43_FINAL_REPORT.md](docs/N43_FINAL_REPORT.md). Next action is one provenance-complete real-human tape/full-loop collection and rerun; do not tune thresholds or expand model capacity to bypass this gate.

## 2026-08-30 — N43 post-replay implementation-risk audit correction

- The requested post-replay `cell_features` regression was rerun with the project interpreter after a preserved system-Python failure (`ModuleNotFoundError: torch`). The three cases (duplicate valid scores with hard columns, non-adjacent valid-column exclusion, and all-hard-negative current cell) PASS. The minimal code fix keeps `valid_columns` as original column indices and excludes `column_index` by index before `best_other`; no replay artifact was silently rewritten.
- The pre-registered CI contract was rechecked against `scripts/run_n36_replay.py`: average events within each sequence first, then bootstrap sequence means with equal sequence weight. The corrected N43 result uses `equal_sequence_mean`; the earlier flattened event-weighted result remains at `outputs/n43/replay/paired_replay_results_legacy_event_weighted.json`. Corrected M2 lower CIs are `-0.032469/-0.032337/-0.025631` for H20/H50/H100; legacy lower CIs were `-0.030331/-0.032206/-0.024435`. Event utilities are unchanged.
- `scripts/n43_finalize.py` now enriches `outputs/n43/stage_04_status.json` with both audit checks and the full corrected-vs-legacy comparison, then regenerates Stage 05 and the report. Final status remains `N43_COMPLETED_GATE_FAILED`; no calibration head or decoder LoRA is authorized.

## 2026-08-30 — N44 assignment-aware structural experiment

- **Hypothesis/protocol:** N43's full-cell interface still applied a learned utility to every finite cell, while its target values were only `+/-0.5` cell labels. N44 first audited the frozen N43 matrices and then trained one isolated anti-symmetric candidate-vs-competitor head with explicit conservative near-tie, advantage and uncertainty gates. N43 inputs, production MOT/OVMOT, shared checkpoints and the unrelated four-GPU task were not modified.
- **Boundary diagnosis:** The read-only audit covered `2424` frames and `173793` candidate×public-ID cells. Baseline Hungarian had `11800` known correct assignments and `4583` known wrong assignments; `1551` wrong assignments had a positive candidate alternative. The offline candidate ceiling had `13604/16920=0.8040189` positive candidate-ID pairs. N43 changed all `173793` finite cells but only `29` assignments. Its dataset retained `15115` positive versus `107188` negative cells (negative/positive `7.0915`), making the cell target a classification utility rather than global assignment gain. The boundary is therefore nonempty, but N43's residual-every-cell/target-column semantics are mismatched to it.
- **Training:** N44 Stage 02 materialized `122303` cells, `29990` pair examples, `16776` frame×public-ID groups, `3280` no-positive abstain groups, and fixed train/validation/holdout sequence partitions. Actual full training ran on GPU0 with seed `4444`, AdamW, `31` epochs, best epoch `21`, and checkpoint `outputs/n44/training/n44_assignment_aware.pt` (SHA-256 `0b5e750f5d9569f71ae887595c1d88d4d625f120f8a3811f2598a852cf82348f`). The frozen gate is near-tie `2`, predicted advantage `0`, maximum calibrated pair uncertainty `2`, scale `1.1517618`; holdout was not used for selection.
- **Replay/result:** Same-prefix/same-event/same-candidate replay completed `24/24` simulated events, all M0–M4 and H20/H50/H100, with runtime `future_gt=false`; GT was loaded only after structural validation. N44 M2 utility is `-0.0092652684/-0.0093929460/-0.0074571786`, with equal-sequence CI lower bounds `-0.0317666345/-0.0322043864/-0.0255674694`. M2 assignment changes were `40/104/249`, correct `0/0/0`, incorrect `12/42/92`; no-change `407/1043/2072`. Untouched-ID regression remains failed. The sidecar selected only `14` proposals over `9600` write frames and never applied an unbounded residual; the structural experiment did not produce positive future effect.
- **Risk checks and real-input block:** The post-replay repeated-score/hard-negative regression passed after explicit valid-column index exclusion in `scripts/n43_full_matrix_common.py`; two checker-only schema/serialization failures remain under `outputs/n44/attempts/`. The pre-registered CI definition is sequence mean then equal-sequence bootstrap; the old event-weighted N43 result remains preserved. Three N40 feasibility checks found the sentinel tape `NOT_AVAILABLE` with zero events, N40 `BLOCKED_INPUT_REAL_HUMAN_TAPE`, and only a `simulated_from_gt` synthetic fallback. No old artifact was relabeled.
- **Decision/keep:** N44 is `N44_COMPLETED_GATE_FAILED`, not scientific completion: real human tape and real SAM3 full-loop are absent, M2/M3/M4 strict future-effect CIs are not positive, and untouched regression is not clean. Calibration and decoder LoRA remain unauthorized. Keep all Stage 01–05 outputs, N44 checkpoint/dataset/replay, blocked-input artifact, failure attempts, and N43 frozen evidence. Minimal next step is an external provenance-complete human UI export and candidate-complete tape followed by N40 validation and a real full-loop; no threshold/seed/metric/capacity bypass is justified.

## 2026-08-30 — N45 true N44 attribution repair

- **Hypothesis/protocol:** N44's `no_write -> write_plus_N44` comparison was not causal for N44 because it omitted the unchanged N42 write branch. N45 froze the N42 source, event/prefix/candidate stream, N44 checkpoint and all metrics/seeds, then materialized three aligned branches for all 24 events, M0–M4 and 100 future frames: `no_write`, original N42 `write_baseline`, and `write_plus_N44`. Runtime used `future_gt=false`; GT entered only after structural validation. N43/N44 artifacts remain untouched and N45 is isolated under `outputs/n45`.
- **Attribution diagnosis:** Runtime validation passed 36,000 branch trace rows, no frame gaps/duplicates, identical candidate native IDs/boxes/confidences, and exact write-vs-plus public-ID axes. Dynamic no-write/write public-ID universes were recorded rather than falsely treated as candidate loss. N44 proposal/application counts were 28 considered, 14 selected, 5 selected-but-no-assignment-change, 14 changed cells and 18 changed assignments; they are not efficacy counts. The true `write -> plus` M2 utility is exactly `0/0/0` at H20/H50/H100 with only neutral assignment changes `6/9/15`; M1/M3/M4 are negative with one incorrect and zero correct incremental changes. The separate M2 memory effect `no_write -> write` is `-0.0092653/-0.0093929/-0.0074572`; thus N44 is exercised but the structural hypothesis fails, rather than being `N44_NOT_EXERCISED`.
- **Repairs/failures retained:** A targeted regression found the N42 `-1` NONE encoding versus N44 explicit dummy columns and branch-local active public-ID axes; normalization/recording was repaired minimally. The frozen N44 checkpoint lacked an authorization field, so a hash-bound N45 manifest records `production_authorized=false` without editing weights. A prior N45 candidate-axis failure, public-ID-set failure, neutral-change schema failure, environment import failure and other failed attempts remain under `outputs/n45/attempts/`. The N44 Stage 02 contract source was corrected to state the frozen audit has zero hard-negative cells and code skips them; the old incorrect gate value remains only as legacy evidence. Stage 01 now names the corrected candidate ceiling denominator `13270/16383=0.8099859611` and retains the old total-cell rate explicitly as a legacy diagnostic.
- **Real-input feasibility/decision:** Three evidence-preserving N40 checks still find no external UI export or verifiable real tape: the N34 sentinel is unavailable with zero events, N40 reports `BLOCKED_INPUT_REAL_HUMAN_TAPE`, and inventory finds only a GT-derived synthetic fallback. No simulated event was relabeled. N45 gate is `N45_COMPLETED_GATE_FAILED`, with calibration and decoder LoRA not authorized. Keep the complete N45 per-event/per-frame attribution, equal-sequence bootstrap CIs (seed 4444, 2000 replicates), strict gate, blocker and failures. The research objective remains open; the minimal next step is external provenance-complete human tape plus real SAM3 full-loop, followed by a new hypothesis only if that evidence supports it.

## 2026-08-30 — N46 structural assignment diagnosis and attribution-contract audit

- **Contract audit:** Before diagnosis, the read-only N45 contract audit preserved `outputs/n46/attempts/n45_contract_audit_pre_fix.json`: the N44 negative-sampling comment overstated the implementation, N45 finalization hard-coded `three_branches=True`, and independent write-assignment recomputation was absent. The comment was minimally corrected to describe the actual two strongest baseline-score negatives; no appearance-negative sampler was invented. The frozen audit has zero hard-negative cells/examples, so hard-negative inclusion is not claimed. A legal branch-metadata schema failure in the first finalizer check is preserved at `outputs/n46/attempts/n45_finalizer_branch_metadata_schema_failure.json` and was repaired to require the three branches rather than exact metadata-key equality.
- **Attribution integrity:** New Stage 01 independently recomputed `hungarian_with_none(write_scores)` for 24 × 5 × 100 aligned frames and found zero mismatches. It verified exact event/frame manifests, unique native IDs, candidate row/public-ID alignment, direct `runtime_future_gt_used=false`, and GT-free runtime. The sidecar targeted regression and full integrity checker passed without rewriting N43/N44/N45; old N45 runtime hashes are recorded in `outputs/n46/n46_integrity_report.json`.
- **Diagnosis:** The completed N46 structural runtime has 12000 frames, 33 proposals considered, 17 selected, 6 selected-but-no-assignment-change, 17 changed cells and 22 assignment changes. Posthoc GT was opened only after runtime validation: 11605 available frames and 395 unavailable. There are 21818 oracle-desired pairs; 21790 are blocked by another public-ID owner. Required delta median is 4.401657 versus fixed +0.25. Proposal coverage is sparse, owner/column constraints are the dominant interface bottleneck, and score alignment is not the primary bottleneck on 115 clear offline proposal cells (25 positive/90 negative; Pearson 0.883128). Neutral changes are never classified as correct. The fixed lambda sensitivity set `{0,0.25,0.5,1,2,4,8}` is diagnostic only and was not used to select a gate.
- **Decision:** N45 true M2 increment remains exactly zero at H20/H50/H100 with neutral 6/9/15 changes; the separate M2 memory effect remains negative with 0 correct and 12/42/92 incorrect changes. N46 therefore records `STRUCTURAL_HYPOTHESIS_FAILED`, not `N44_NOT_EXERCISED`, and does not justify another training experiment. `outputs/n46/n46_final_gate.json` and `outputs/n46/stage_03_status.json` are `N46_COMPLETED_DIAGNOSTIC_GATE_FAILED`; calibration, decoder LoRA and production changes remain unauthorized.
- **Provenance/blocker:** All events remain `simulated_from_gt`. N40 still reports `BLOCKED_INPUT_REAL_HUMAN_TAPE`; the N34 sentinel is unavailable/empty and no external UI/annotator export or real SAM3 full-loop exists. The explicit N46 blocker is preserved at `outputs/n46/BLOCKED_INPUT_REAL_HUMAN_TAPE.json`. All partial/failed N46 attempts remain under `outputs/n46/attempts`; no old failure was deleted or relabeled. Research objective remains open pending provenance-complete real human tape and candidate-complete real SAM3 full-loop.

## 2026-08-30 — N45 attribution repair continuation and N46 repair2

- **First actionable N46 runtime defect:** the initial isolated N46 diagnosis applied the N44 sidecar to M0, which must remain the exact no-sidecar control. The failure is preserved at `outputs/n46/attempts/stage_02_m0_sidecar_applied_in_diagnosis.json`. The minimal M0 guard was applied only to the new N46 diagnosis path; repair2 was then rerun and reproduced the frozen N45 runtime application totals `28/14/5/14/18` (considered/selected/selected-no-assignment-change/changed-cells/changed-assignments).
- **N45 attribution defect:** an immutable comparison found `78` write-source frames where old N45 baseline mapping used raw `candidate_public_ids` outside the active `public_id_order`, while write-plus used assignment columns. The example and counts are preserved at `outputs/n46/attempts/n45_baseline_candidate_public_id_axis_mismatch.json`; the old `outputs/n45/replay/attribution_results.json` was not changed and is legacy/provisional evidence.
- **Normalized repair:** `scripts/n45_normalized_attribution_repair.py` maps no-write, write-baseline and write-plus uniformly through assignment columns and each branch's active public-ID axis, after GT-free runtime validation. The corrected isolated result is `outputs/n46/n45_attribution_repair/normalized_attribution_results.json`. For M2, the true write→plus increment is exactly `0/0/0` utility at H20/H50/H100, with assignment changes `1/2/2`, all neutral, and no correct or incorrect changes. The separately measured no-write→write memory effect remains negative `-0.009265268392/-0.009392946035/-0.007457178589` with incorrect changes `12/42/92`.
- **Validation:** the repaired targeted regression and full integrity checker both PASS at `outputs/n46/n45_attribution_repair/targeted_regression.json` and `outputs/n46/n45_attribution_repair/full_integrity.json`. Their earlier schema/provenance failures remain under `outputs/n46/attempts/`. The normalized final gate is `N45_ATTRIBUTION_REPAIR_COMPLETED_GATE_FAILED`; it does not authorize calibration, decoder LoRA or production changes. No new training was started. All evidence remains simulated-from-GT, with N40 real human tape and real SAM3 full-loop still blocked.
- **Final audit:** one first-pass read-only audit assertion used an over-specific report phrase and is preserved at `outputs/n46/attempts/final_readonly_audit_assertion_failure.json`; the corrected read-only audit PASSed without changing data, checkpoint, protocol or metrics.
- **N46 supersession:** repair2’s authoritative diagnostic gate/status are now isolated at `outputs/n46/diagnosis_final_repair2/final_gate.json` and `outputs/n46/diagnosis_final_repair2/stage_03_status.json`; the original `outputs/n46/n46_final_gate.json` remains untouched legacy evidence.

## 2026-08-30 — N47 isolated global candidate-to-public-ID assignment probe

- **Conclusion audit and hypothesis:** N46's “no further training” conclusion was not treated as absolute. The sparse local proposals, `21790/21818` owner-by-column blocked oracle pairs, fixed `+0.25` versus required-delta median `4.401657`, and negative M2 memory effect support a falsifiable structural probe: predict a candidate-level appearance logit for every finite candidate×public-ID cell and run one global Hungarian solve with explicit NONE and legal swaps. This is an isolated diagnostic hypothesis, not production authorization.
- **Protocol/implementation:** The frozen protocol is `outputs/n47_global_probe/probe_protocol.json`: N42 runtime/t0 source, N42 sequence-disjoint split, seed `4747`, fixed pairwise softplus ranking plus logit L2, no holdout selection, and causal current/past score, appearance/memory, confidence, age, rank and frame-offset features only. The global solver masks hard/NONE cells below the explicit dummy; the cheap swap/NONE smoke first exposed and then fixed a float32 tie. All failures are retained under `outputs/n47_global_probe/attempts/`.
- **Training:** The first CPU run built `611451` labelled cells and `404584` pairs but ended with only a partial epoch-2 checkpoint; it is preserved as `stage_03_cpu_partial_timeout.json` and `n47_global_fusion_probe_partial_epoch2.pt`. After the minimal device/tensor fix, the same frozen protocol completed actual GPU0 training: 7 epochs, best epoch 2, reload PASS, checkpoint SHA-256 `492d409c62b8a2af772df6e60df415ff8b7165b69ba8f66f61a918f10600adf1`, and `production_authorized=false`.
- **Replay/integrity:** Runtime and posthoc completed `24×5×100=12000` frames; runtime used direct `runtime_future_gt_used=false`, complete candidate rows/native IDs/axes, and loaded GT only after independent runtime validation. Full integrity passed with 39150 posthoc effect-frame records, independently recomputed Hungarian/NONE assignments, hard-negative preservation, equal-sequence bootstrap and unchanged N44 checkpoint.
- **Result:** The global structure was exercised, not a no-op: M2 write→global-plus had H20/H50/H100 identity utilities `0.0084729/0.0036257/-0.0113691`, correct changes `13/21/24`, incorrect changes `0/1/40`, and equal-sequence CI lower bounds `0.0003947/0.0000311/-0.0222698`. Thus the structure exposes short-horizon correct changes that N46's local proposal gate could not express, but fails robust H100 and untouched-ID gates; neutral changes are not called correct. The separate N42/N45 memory effect remains negative and is not credited to N47.
- **Gate/provenance:** N47 is `N47_COMPLETED_GATE_FAILED`; no threshold, seed, metric, LoRA or checkpoint replacement was used. All events remain `simulated_from_gt`, standard MOT/TrackEval is not computable from bounded windows, and N40 real human tape plus real SAM3 full-loop remain absent. Calibration, decoder LoRA and production interface changes remain unauthorized. Report: `docs/N47_FINAL_REPORT.md`; gate: `outputs/n47_global_probe/n47_final_gate.json`. The broader research objective remains open.
- **Final preservation audit:** `scripts/n47_final_audit.py` PASSed with 212/212 legacy N43/N44/N45 hashes stable, unchanged N44 checkpoint hash, complete N47 stage schemas and five retained N47 attempts. No production or shared artifact was modified.

## 2026-08-30 — N48 risk-aware 512-D global assignment diagnostic

- **Decision audit:** N47's “no further training” conclusion was tested rather than treated as absolute. The first actionable root cause was an unbounded 8-scalar global logit with no temporal/untouched-risk guard; N47 was not a complete 512-D appearance-memory result. Existing N36 machine embeddings provide a legal 512-D causal diagnostic input, while human interaction remains explicitly simulated.
- **Protocol/training:** N48 froze `outputs/n48/protocol.json`, seed `4848`, N42 sequence-disjoint train/validation/holdout split, 512-D candidate/memory fusion, bounded `0.25*tanh` residual, fixed uncertainty/global-margin gate, explicit NONE and global Hungarian. Actual GPU0 training completed 8 epochs, best validation epoch 1; checkpoint SHA-256 `ab26489371d4c9109392d27b8c1557a558002357c390bef03b093cdbc554ca49`, reloadable and `production_authorized=false`. No holdout or threshold tuning was used.
- **Runtime/replay:** GT-free runtime and posthoc replay completed `24×5×100=12000` frames. The runtime retained candidates, native IDs, boxes/confidences, 512-D vectors, memory provenance, active public-ID axes, matrices, gate reasons and assignment transitions. Independent integrity recomputed normalized Hungarian+NONE on 12000 frames and 24000 source future-trace frames; it passed candidate/axis, M0 no-op, hard-negative, unique-ID and future-GT checks. Sequence-cluster bootstrap uses equal sequence means with seed 4848 and 2000 replicates.
- **First repair evidence:** A local-vs-global margin contract mismatch, malformed targeted fixture, replay branch-frame schema omission, and integrity checker import/normalized-list errors were each preserved under `outputs/n48/attempts/`, minimally repaired, py-compiled, smoke-tested and rerun. The provisional malformed report/table and false CI status were preserved, then corrected. No N47/N46 artifact, production MOT/OVMOT code, shared checkpoint or external task was modified.
- **Result:** N48 separates memory from sidecar attribution. M2 memory effect H20/H50/H100 is `-0.009265268/-0.009392946/-0.007457179`, with `0/12/42/92` C/I/N at H100 decomposition (0 correct, 92 incorrect, 143 neutral). N48 incremental M2 utility is exactly `0/0/0` at H20/H50/H100, with `0` correct, `0` incorrect and `2/2/3` neutral changes; equal-sequence CIs are `[0,0]`. Selected score cells and proposals are retained separately and are not called assignment successes when Hungarian does not change. This is `N48_NOT_EXERCISED_GATE_FAILED`, not a positive structural result.
- **Real-input gate/next:** Three inventory checks still find no externally supplied UI/session/annotator tape; available files are GT-derived or machine candidate tapes. `outputs/n48/BLOCKED_INPUT_REAL_HUMAN_TAPE.json` records the exact blocker and required direct public ID, human-confirmed BOX/CLICK/CONFIRMED_MASK, lossless ROI digest, provenance timestamps, mapping and candidate-complete future rows. Real SAM3 full-loop and standard TrackEval remain unavailable. Keep the broader objective open; no calibration, decoder LoRA or production authorization. The unique minimum next step is external real human tape plus candidate-complete real SAM3 full-loop, followed by a new falsifiable experiment only if warranted.

## 2026-08-30 — N47 swap-metric repair1

- **First actionable root cause:** `scripts/n47_stage04_global_probe_replay.py` had named `sorted(write_public_ids) != sorted(plus_public_ids)` as `swap_changes`; this measured ID multiset changes, not pure row-assignment swaps. The old N47 runtime/status/result/report are preserved in `outputs/n47_global_probe/repair1_swap_metric/legacy_snapshot` with hashes, and the original root files remain unchanged.
- **Minimal repair and evidence:** The isolated replay now records `assignment_changes`, `id_set_changes` (the legacy multiset-change semantics, including NONE/new/removal), and `pure_swap_changes`. The registered pure-swap predicate requires equal non-NONE/full assignment multisets and at least two changed row mappings. Frozen replay used the same N47 checkpoint (`492d409c62b8a2af772df6e60df415ff8b7165b69ba8f66f61a918f10600adf1`), seed and N42 input, with no training. Counts are assignment/pure-swap/id-set: M1 `335/56/279`, M2 `455/64/391`, M3 `335/56/279`, M4 `375/39/336`; M0 is zero. The old M2 `391` is explicitly a mislabelled legacy ID-set count; corrected pure swaps are `64`.
- **Validation/failed attempts:** `py_compile`, targeted taxonomy regression, full `24×5×100=12000` runtime+posthoc replay, independent Hungarian-with-NONE/candidate/frame/native-ID/GT-provenance integrity, and finalization completed. The first direct-two-cycle classifier (`54/62/54/37`) and two checker/finalizer import/contract failures are retained under `outputs/n47_global_probe/repair1_swap_metric/attempts`; no failure was deleted or promoted to PASS.
- **Metric/gate interpretation:** Posthoc utility, target IoU, future identity error, re-correction, correct/incorrect/neutral/no-change decomposition and untouched-ID values are unchanged from legacy N47. The repair gate is `N47_SWAP_METRIC_REPAIR_COMPLETED_GATE_FAILED`; inherited H100/untouched gates and missing real human tape/full-loop remain failed. Events remain `simulated_from_gt`; calibration, decoder LoRA and production remain unauthorized. Report: `docs/N47_SWAP_METRIC_REPAIR_FINAL_REPORT.md`; gate: `outputs/n47_global_probe/repair1_swap_metric/n47_swap_metric_gate.json`. The broader scientific objective remains open.

## 2026-08-30 — N48-R1 protocol repair and complete isolated replay

- **First actionable root cause:** The frozen N48-R0 protocol declared pairwise softplus ranking plus weighted valid-cell BCE, uncertainty BCE and fixed residual L2, but `scripts/n48_stage03_train.py` omitted the cell-BCE term. R0 checkpoint `ab26489371d4c9109392d27b8c1557a558002357c390bef03b093cdbc554ca49` and all R0 status/results remain preserved and are explicitly classified as a protocol-mismatch diagnostic attempt.
- **Minimal repair/protocol:** Before retraining, the isolated amendment `outputs/n48/repair1/protocol_amendment.json` froze targets (IoU ≥0.5 positive, ≤0.3 negative, ambiguous/unavailable excluded), train counts 11942/105947, inverse-frequency weights `w_pos=4.935898509462402`, `w_neg=0.5563583678631768`, fixed `cell_BCE` coefficient 0.25, same seed 4848/split/data/8-epoch budget, no holdout selection, runtime GT false and production authorization false. Stage01 diagnosis was minimally repaired to enforce per-sequence closure, changed-row-only NONE accounting, frame/target naming separation, and invalid non-comparable oracle score-gap labeling; old JSONs remain under `outputs/n48/repair1/legacy_snapshots/`.
- **Training/replay:** Actual GPU0 R1 training completed 8 epochs with checkpoint `outputs/n48/repair1/training/n48_r1_risk_aware_512d_bce.pt`, SHA256 `9c0a222d699bbed3951aa897cbca9f068cb3f2b6ddf5485597619e39adbed149`; rank/BCE/uncertainty/L2 were logged, reload/smoke and targeted regression passed. The first replay attempt failed on the shared N48 checkpoint protocol allow-list and is preserved at `outputs/n48/repair1/attempts/replay_checkpoint_protocol_mismatch.json`; the minimal isolated-compatible loader repair was regression-tested. Full replay then completed 24 events × 5 variants × 100 frames = 12000 runtime frames, with independent integrity passing 12000 runtime and 24000 source future-trace frames, explicit NONE/Hungarian normalization, candidate completeness and runtime GT-free checks.
- **Result/gate:** R1 separates no-write→write memory effect from write-baseline→R1 increment. M2 incremental utility is exactly `0/0/0` at H20/H50/H100; assignment changes are `1/1/2`, correct `0/0/0`, incorrect `0/0/0`, neutral `1/1/2`, with equal-sequence bootstrap CIs `[0,0]`. M2 memory remains negative at H100 (`-0.007457179`, 0/92/143 correct/incorrect/neutral), and H100 untouched regression fails. The schema mismatch in the first status finalizer is preserved at `outputs/n48/repair1/attempts/stage02_finalize_schema_mismatch.json`; after the smallest reader-only fix, py_compile, smoke, reload and Stage01 targeted regression passed.
- **Provenance/next:** The strict gate is `N48_R1_NOT_EXERCISED_GATE_FAILED` at `outputs/n48/repair1/stage_05_status.json`; all events are `simulated_from_gt`, standard MOT/TrackEval is not computable, and `production_authorized=false`. `outputs/n48/repair1/BLOCKED_INPUT_REAL_HUMAN_TAPE.json` retains the N40 blocker: no external provenance-complete human tape or real SAM3 candidate-complete full-loop exists. The research objective remains open; the unique minimum next step is external real human tape plus real full-loop evidence. No calibration, decoder LoRA, production change, threshold/seed/metric tuning or further blind weighting is authorized.

## 2026-08-30 — N48-R1 repair2: corrected evaluate indices and one-step objective

- **First actionable root cause:** A read-only audit confirmed R1 passed `pair_split` labels into `evaluate` instead of complete `train_pairs`/`val_pairs` index sets. It also confirmed a second cell-only optimizer loop after pair updates, which was not the frozen single objective. R1 is now `PROVISIONAL_INVALID_TRAINING_SELECTION`; checkpoint, manifest, status, replay and report were preserved under `outputs/n48/repair1b/legacy_r1` with their hashes.
- **Frozen protocol:** `outputs/n48/repair1b/protocol_amendment_repair2.json` keeps seed 4848, the same data/split, coefficients `1/0.25/0.25/0.001`, train/validation/holdout pair counts `95536/9322/16050`, valid cell counts `117889/6207/11651`, fixed ascending index order and micro-batch 1024. All pair and cell gradients are accumulated before exactly one optimizer step per epoch; validation uses true validation indices and holdout is never selected.
- **Training/replay:** The first new smoke failed only because the regression harness lacked project-root import setup; it is preserved at `outputs/n48/repair1b/attempts/repair2_smoke_import_failure.json`. After the smallest harness fix, py_compile, deterministic smoke, split/objective targeted regression and reload passed. Actual GPU0 training completed 8 epochs, best epoch 8, checkpoint SHA256 `5e5a8a99e1ad24005c0e83046b8c10d467183d1b296095bee6ccb984dbeb988d`; all four loss components and one optimizer step per epoch are recorded.
- **Replay/gate:** Full replay completed `24×5×100=12000` runtime frames and independent integrity passed 12000 runtime plus 24000 source future-trace frames. No repair2 score cell was accepted; M2 write→repair2 incremental utility is `0/0/0` at H20/H50/H100 with assignment changes `0/0/0`, while the separate memory effect remains `-0.009265268/-0.009392946/-0.007457179` and H100 untouched regression fails. Gate: `N48_R1_REPAIR2_NOT_EXERCISED_GATE_FAILED`, with all events `simulated_from_gt`, real human tape/full-loop absent, and production/calibration/LoRA unauthorized.

## 2026-08-31 — N50 N49 read-only root-cause diagnosis

- **Hypothesis/coverage:** N49's strict zero-effect result was audited without changing N49. The 24 runtime event files were loaded one at a time; M1–M4 each cover 2400 frames and 172519 cells. The audit independently rebuilt explicit-NONE Hungarian, row-major candidate/memory flattening, source base scores, M0 no-op, checkpoint outputs and all stored gate reasons.
- **Failure evidence:** M2 has 89000 finite-base×valid-memory upstream proposals, but first rejection is memory-invalid `83519`, global-margin `66379`, absolute bounded residual `<0.05` `267`, uncertainty `>0.35` `22354`, accepted `0`; counts and proposal/selected totals match every cell matrix. All 2400 M2 plus matrices have zero score delta and unchanged solver assignments. Independent checkpoint reproduction errors are below `2.4e-7` raw and `6.0e-8` uncertainty.
- **Structural defect:** Frozen N49 copied N48 labels/features/splits/pair order and the assignment-critical mask is exactly row-aligned, but runtime scalar column 6 uses row-mean memory validity while training stores per-public-ID cell validity. This produces `166449/172519` aligned scalar mismatches. Holdout uncertainty AUC for label-0 risk is `0.5276` (validation `0.5042`) with weak calibration. Offline-only oracle signed-delta diagnosis finds finite candidate-row required-delta median `4.3162`; NONE-involved rows are separated and no counterfactual effect is claimed.
- **Decision/keep:** N50 status is `PASS_DIAGNOSTIC_NOT_EFFICACY`. Classification is A gate/risk scale failure confirmed, B not yet the sole cause and remains falsifiable, C interface/owner/NONE corruption refuted, plus a confirmed data/feature contract mismatch. N50 diagnosis JSON, script, logs (including preserved failed attempts), stage status and report are under `outputs/n50` and `docs/N50_DIAGNOSIS_REPORT.md`. N49/N48 evidence and all real-input blockers remain unchanged; calibration, decoder LoRA and production remain unauthorized. Evidence supports freezing an isolated N50 protocol before any new training.

## 2026-08-31 — N50 signed-advantage/risk isolation gate

- **Training/replay:** After freezing `outputs/n50/protocol.json`, the isolated N50 dataset repaired the runtime scalar contract and trained a signed score-advantage/risk head on GPU0 for 32 epochs, one optimizer step per epoch, best epoch 20, checkpoint `92769db9e5fd897711d42e27d01d9524707e5a5aff3cf502e681f76d97e79e3d`. The first training attempt had an index-type failure and the second exposed an invalid inherited `pair_split` length; both are preserved with minimal repairs and smoke/regression evidence.
- **Replay/gate:** GT-free runtime and posthoc completed 24×5×100=12000 frames. Independent integrity passed candidate/public-ID/row-major axes, base source, M0 no-op, explicit-NONE Hungarian, applied-score contract and runtime future-GT boundary. N50 M2 write→plus utility is exactly 0 at H20/H50/H100 with zero selected cells, score changes, assignment changes and correct changes. Strict gate: `N50_SIMULATED_GATE_FAILED`.
- **Root cause/next:** The new head's risk target marks nearly all non-target cells unsafe and the learned runtime risk is approximately 0.94, so the fixed `risk<=0.35` gate rejects all 89000 upstream proposals. This is a target prevalence/calibration failure, not evidence for threshold relaxation. Preserve all N50 artifacts and continue with read-only N51 target/risk/signed-advantage diagnosis only if evidence supports a new non-threshold protocol. Real human tape and real SAM3 full-loop remain hard blockers; no calibration/LoRA/production authorization.

## 2026-08-31 — N51 N50 risk-target diagnosis

- **Read-only result:** N51 scanned all 2400 N50 M2 frames and 172519 aligned cells. Safe target cells are `13450/172519=7.80%`; runtime risk is approximately `0.93` for both safe-target and non-target groups. Validation/holdout AUC for unsafe prediction is `0.3848/0.4192`, so the risk head is weak/reversed rather than merely narrowly above the fixed gate.
- **Decision:** The fixed gate admits zero cells, while counterfactual cutoff counts were diagnostic only and not used for selection. This supports a new target/loss contract (explicit cost-balanced or gate-aligned risk training) but does not support threshold relaxation. N51 status is `PASS_DIAGNOSTIC_NOT_EFFICACY`; N50 remains `N50_SIMULATED_GATE_FAILED`, and real human tape/real SAM3 full-loop remain hard blockers.

## 2026-08-31 — N51 alignment audit and N52 cosine-risk isolation

- **N51 alignment:** A fixed explicit-key audit covered `(event_id, variant=M2, frame, candidate_row, public_id)` for all `172519` cells. Key set/order, base scores, scalar features and all offline labels matched; candidate/memory errors were bounded by the declared float16 storage round-trip, and checkpoint→runtime plus float16-control re-inference had zero mismatches. The legacy `event_index` field is documented as a fixed 2400-frame ordinal; the frozen N51 dataset was not rewritten. Audit: `outputs/n51/stage_10_status.json` is `PASS_DATASET_RUNTIME_ALIGNED`.
- **N52 protocol/training:** N52 froze a new non-production protocol before implementation: monotonic risk `sigmoid(softplus(raw_scale)*(-cosine)+bias)`, fixed risk gate `<=0.35`, train-only class-balanced BCE, N42 split, seed `5252`, 64 full-batch CUDA epochs, one optimizer step per epoch, validation-only checkpoint selection and holdout-final evaluation. Fixtures passed; fresh cosine dataset and actual GPU0 training passed with checkpoint SHA256 `f097cf1fb640b54f8a3e4e6d106613a9630fdc3ed32948b03dcf66871d6c7805`.
- **N52 runtime/integrity:** GT-free `24x5x100=12000` replay and independent integrity passed. N52 risk opened the fixed gate: M2 had `451` selected cells and `50` assignment changes. All provenance remained simulated-from-GT and runtime future GT false.
- **N52 strict result/diagnosis:** Posthoc M2 H20/H50/H100 utilities were `0/0/0`; correct changes `0/0/0`, neutral changes `24/43/48`, and untouched regressions `65/90/95`. Of 451 selected cells, 233 were target-safe and 218 non-target-unsafe; selected target delta sign agreement was `0.97425` but MAE `2.25088`. The first actionable root cause is frozen N51 delta magnitude/non-target suppression, not risk-gate coverage. N52 is strict gate failed; no threshold/seed/metric/production bypass is authorized. Real human tape and real SAM3 full-loop remain absent.
- **Next hypothesis:** Freeze N53 to train a bounded target-delta calibrator with explicit target-cell oracle signed-advantage supervision and non-target-zero supervision, while reusing N52 cosine risk only as a frozen risk gate. Do not overwrite N52/N51 or claim simulated evidence as human tape.

## 2026-08-31 — N66 stage08 stream recovery root cause (recorded before repair)

- **First actionable root cause:** `outputs/n66/stage_08_diagnose_r1.py` uses `jq --stream`; path-only end-marker records (notably `M2 selected_action.reason`) contain only `[path]`. The parser at line 313 unconditionally reads `path_value[1]`, producing `IndexError`. Its exception cleanup then reads `stderr` before stdout is drained, allowing a second pipe deadlock; the prior diagnosis attempt consequently stopped at `pipe_wait`.
- **Required minimal repair:** Ignore stream records with `len(path_value)<2` before accessing the value, and make subprocess cleanup drain both pipes safely before `wait`. Keep all N66 failure evidence, protocol, checkpoint, runtime and strict-failure classification unchanged. Run a small runtime smoke followed by one single-process targeted regression over all 24 runtime files; no concurrent duplicate.
- **Provenance:** This is a diagnostic parser/cleanup repair only. `interaction_source=simulated_from_gt`, `not_real_human_evidence=true`, `runtime_future_gt_used=false`, `production_authorized=false`; no training has started.

## 2026-09-01 — N68 identity-scoped local association and margin interface

- **Hypothesis/protocol:** N67's simulated CCAM failure was decomposed before implementation. Stage01 found a physical target native candidate in all 30 accepted actions, but `29/30` native-to-public mapping mismatches and `1/30` wrong target scope. N68 then froze a 32-D target-conditioned local association sidecar with explicit known `public_id`, target-column-only bounded residual, global Hungarian/NONE retained, sequence-disjoint N42 split, seed `6801`, and runtime future GT false. No production module, checkpoint or N67 evidence was modified.
- **Training:** Actual T1 training was completed on isolated GPU1 (`MLP 32->64->32->1`, `4,225` parameters), best validation epoch 2, holdout AUC `0.7693398676`, checkpoint SHA256 `c068cc8ac87bbbc0d0c1e31639d961df119b51fd03235275ece2207ef0d7b9e2`. Dataset has `92,070` examples (`11,910` positive, `80,160` negative), 12,000 frames and 24 events. This is a sidecar training result, not production authorization.
- **Paired replay:** Stage02 and Stage03 both completed `24×5×100=12,000` runtime frames with candidate/public-ID/Hungarian integrity and `runtime_future_gt_used=false`; GT was loaded only for posthoc scoring. Learned local association changed scores on all frames but produced only `5` correct and `0` incorrect assignment changes, all in ADD_NEW_IDENTITY; H20/H50/H100 utility was `+0.002083/+0.000833/+0.000417`, but all sequence-cluster CI lower bounds were `0` and untouched changes were `18`. A pre-registered margin-aware target-column projection produced `0` correct changes and `0/0/0` utility, with `5` untouched changes.
- **Interpretation/gate:** Score change is real and rarely crosses the assignment boundary, but the effect is sparse, action-specific and not statistically strict or untouched-safe. This does not support more blind weight scaling, TACT, calibration, selector or decoder LoRA. `outputs/n68/n68_final_gate.json` is `BLOCKED`: all evidence is `simulated_from_gt`, `real_human_tape=false`, `real_sam3_full_loop=false`, and production remains unauthorized. The minimum next step is external provenance-complete real human tape plus a native/local/global-to-public mapping audit before another architecture.
- **Failures retained:** N68 preserves Stage00 self-inclusion, Stage01 schema/classification, Stage02 axis/environment/posthoc, Stage03 import, isolation import, and final-gate aggregate-frame-count failures under `outputs/n68/attempts/`; each was minimally repaired and targeted-regression tested. Isolation found `44/44` production/third-party hashes unchanged, `113` SAM3 unit tests passed, and `27` adjacent MOT tests passed. No external MOT/OVMOT write was made.

## 2026-09-01 — N69 target-conditioned association and protected assignment guard

- **Protocol and mapping:** N69 froze a mapping-first, target-conditioned association protocol over the read-only N37/N54 GT-simulated input: 24 events, 21 sequences, five upstream variants and 100 future frames per event/variant. Stage01 audited all `12,000` rows, resolved target scope on `11,910` available frames, retained `90` explicit target-candidate absences, and recorded `3,665` old N68 target-public conflicts. Candidate/frame integrity passed, but the frozen cache still lacks complete native→local→global→public provenance; formal mapping efficacy and production gates remain false.
- **Dataset/model:** The isolated dataset has `92,070` candidate examples (`11,910` positive, `80,160` negative) and raw candidate/human-anchor/target-memory/hard-negative 512-D inputs with audited context. An actual GPU0 run trained a `128,902`-parameter low-rank target-conditioned target/NONE scorer using sequence-disjoint splits and fixed seed `6901`; the corrected checkpoint has holdout AUC `0.850147`. No production module, checkpoint, candidate generator or Hungarian solver was changed.
- **Failures and repairs:** The default interpreter lacked torch; the project environment smoke then exposed a missing module-level torch binding; a temporal-pair loop used `pair_indices.size` and produced NaN; and the first training pass inverted the CE label/logit contract (`dataset 1=target`, replay logit 0=target). Each fact is retained under `outputs/n69/attempts/`; only the environment selection, binding, loop bound and CE-boundary label conversion were minimally repaired. The invalid-label checkpoint/replay is preserved separately and excluded from the corrected result.
- **Paired replay:** Corrected GT-free replay completed `24×5×100=12,000` future frames with identical candidate/public axes, target-column-only bounded updates, event-frame memory hidden, event+1 first visibility and `runtime_future_gt_used=false`. Score changed on `99.1083%` of frames, but assignment changed on only `0.1167%`: `10` target changes were correct, `0` incorrect, and `14` untouched candidate assignments changed. H20/H50/H100 raw event-variant utility was `0.00416667/0.00166667/0.00083333`, while sequence-cluster CI lower bounds were `0/0/0`.
- **Alternative:** Because every correct crossing also had an untouched change, Stage06 ran one isolated native-scoped protected-assignment guard. It completed the same `12,000` frames and rejected `14` proposals that displaced an already assigned native candidate. The guard had `0` correct, `0` incorrect and `0` assignment changes with zero untouched regression, demonstrating a safety/effect trade-off rather than a useful future-effect result. TACT, calibration, selector and decoder LoRA were not run.
- **Gate/isolation:** N69 final status is `BLOCKED_N69_STRICT_GATE_AND_PRODUCTION_EVIDENCE`; synthetic future-effect is `FAIL_FUTURE_EFFECT` and production evidence is blocked because all events are `simulated_from_gt`, `real_human_tape=false`, and `real_sam3_full_loop=false`. Production/third-party/configuration hashes, protected checkpoints, and N36–N68 evidence remain unchanged; `113` SAM3_InterMOT tests and `27` adjacent MOT/OVMOT tests pass in the canonical environment. Report: `docs/N69_FINAL_REPORT.md`; gate: `outputs/n69/n69_final_gate.json`.
- **Scientific interpretation/next step:** The result supports “the scorer can learn target evidence and alter scores” but not “appearance memory reliably improves identity association.” The main actionable bottleneck is the global assignment interface: row-wise target-column changes can displace untouched candidates, and conservative protection removes the crossing. The minimum next step is externally supplied provenance-complete real human tape plus candidate-complete real SAM3 full-loop and mapping audit; no production promotion or downstream learning is authorized.

## 2026-09-01 — N70 candidate×identity association and mapping repair

- **Hypothesis/protocol:** N70 tested whether repairing native→local→global→public provenance and training a candidate×identity scorer could make human-conditioned evidence cross the global Hungarian assignment boundary. The frozen input was the N37 24-event/21-sequence GT-simulated protocol with unchanged N36/N54 candidate stream, checkpoint, embedding, candidate order, solver and metrics. Runtime future GT remained false; all events remain `simulated_from_gt` and not real-human evidence.
- **Mapping/cache:** The N70 cache rehydration completed 12,000 event×variant×frame rows and 92,070 candidate rows with 0 duplicate/error rows and finite 512-D features. Native/local/global mapping is auditable for observed candidates. The 90 target-candidate-absent frames, 10 target-public-assignment-absent frames and 183 explicit unmatched public assignments were retained, never filled. The initial zero-area-box IoU false negative and later `int(None)` uniqueness failure are preserved under `outputs/N70/attempts/` and were minimally repaired.
- **Training:** Actual GPU training completed for both isolated branches on factual GPU 6: Branch A (128,902 parameters, 765 steps, holdout AUC 0.823025) and Branch B (99,730 parameters, 918 steps, holdout AUC 0.861854). Sequence-disjoint splits and validation-only checkpoint selection were respected. Three smoke harness failures and one final replay serialization failure remain preserved as evidence.
- **Replay/gate:** The full paired replay produced 24 event artifacts, 2,400 frame records, 12,000 event×variant×frame keys and 38,400 boundary rows; independent integrity found 0 duplicate/nonfinite/runtime-GT errors. Branch A changed scores on 93.0583% of rows but had 5 correct target crossings and 28 untouched changes; Branch B changed 78.0917% with the same 5 correct crossings and 15 untouched changes. H20/H50/H100 utility deltas were +0.002083/+0.000840/+0.000420 for both, but every sequence-cluster 95% CI lower bound was 0 and untouched regression failed. N70 is `FAIL_FUTURE_EFFECT`, not PASS.
- **Decision/next:** Engineering integrity is `PASS_WITH_EXPLICIT_LIMITATIONS`; calibration head, selector, decoder LoRA and production association remain unauthorized. The result strengthens the diagnosis that score motion rarely crosses the global assignment boundary and that crossings can move untouched IDs. The minimum next step is external provenance-complete real-human event tape and candidate-complete real SAM3 full-loop evidence; synthetic N70 events must not be relabeled as human evidence.

## 2026-09-01 — N71 global identity-association system probe started

- **Frozen start:** N70 remains read-only with `FAIL_FUTURE_EFFECT`; its 24-event/21-sequence GT-simulated stream, candidate absence, mapping absence and untouched-ID limitations are not reinterpreted. N71 uses `outputs/N71/` for small status/protocol evidence and `/data2/usr_for_deadline/SAM3_InterMOT_N71/` for re-creatable heavy artifacts.
- **Hypothesis:** N70's row-wise target-column residual is not a joint candidate×identity×NONE decision. N71 freezes an identity-scoped complete matrix scorer, explicit candidate-specific NONE columns and a short temporal protection diagnostic, while retaining an N70 reproduction control and a genuinely re-exported official-SAM3 candidate branch.
- **Constraints:** all interactions remain `simulated_from_gt`; runtime future GT is false; no N36--N70 evidence, production MOT/OVMOT path, shared checkpoint or `third_party/sam3` is modified. The N71 protocol, method-search record and isolation snapshot were written before experiments (`outputs/N71/protocol.json`, `outputs/N71/method_search.json`, `outputs/N71/stage_00_status.json`).
- **Next:** run CPU diagnosis and matrix/cache schema smoke, then the independent official SAM3 candidate smoke. Only after those pass will sequence-disjoint CUDA training and paired replay start; a raw positive with sequence-cluster CI lower bound 0 remains exploratory, not a gate pass.

## 2026-09-01 — N71 global identity-association system probe completed

- **Scope and isolation:** N71 kept N36–N70 evidence read-only and used `outputs/N71/` plus `/data2/usr_for_deadline/SAM3_InterMOT_N71/`. All 547 production Python source hashes captured before the run remained unchanged; no production or `third_party/sam3` file had a newer mtime after the snapshot. The project has no usable Git repository, so the preservation audit records file hashes and protected-root checks.
- **Stage 01 diagnosis:** N70's score-to-assignment boundary was independently separated from upstream candidate/mapping failures. Branch A/B had 23/15 assignment changes and 5/5 correct crossings despite 11167/9371 score-changing rows; 90 target-candidate-absent frames, 10 public-assignment-absent frames and 70 variant-axis mismatch frames were retained as separate evidence.
- **Candidate branch:** A legal official SAM3 branch (`max_num_objects=16`, `multiplex_count=16`, threshold 0.30, CPU video offload) completed 6 independent windows, 927 frames and 9333 candidate rows with no missing masks. Its public mapping bridge was unavailable and was not fabricated; posthoc recall was not used for runtime selection. The unsupported 32-capacity checkpoint-shape attempt remains preserved.
- **Training:** The true candidate×identity×NONE dataset had 12000 sequence-disjoint groups, 863427 cells, 91887 positive cells and 183 explicit NONE candidate cases. T1 global-matrix training completed on CUDA GPU1 with seed 7171; attempt 1 was excluded after a candidate-specific NONE-head semantic audit, then the minimal repaired head was smoke-tested and retrained as attempt 2. No calibration, selector, decoder LoRA or production change was authorized.
- **Replay:** GT-free runtime and posthoc completed 24 events × 5 memory variants × 100 future frames = 12000 variant-frame artifacts. The trained global matrix, temporal guard, and one pre-frozen standardized base+pair interface all changed finite scores (normalized branch 1.0) but produced zero assignment crossings, zero candidate-present improvements, zero treatment-induced wrong reassociations, zero untouched regressions, and H20/H50/H100 sequence-cluster utility CI `[0, 0]`. The first replay schema error, semantic wrong-reassociation repair, and normalized NONE-spread validator error remain preserved under `outputs/N71/attempts/`.
- **Scientific decision:** N71 is `COMPLETE_SYNTHETIC_GATE_FAIL_PRODUCTION_DEFERRED`, not a success. The result rejects raw score-scale normalization as sufficient on this frozen stream and does not support more blind weight scaling. Causal trimming was not run because no noncausal branch met its preregistered positive precondition. All interactions remain `simulated_from_gt`; real human tape and candidate-complete real SAM3 full-loop evidence remain absent. Final report: `docs/N71_FINAL_REPORT.md`; final gate: `outputs/N71/n71_final_gate.json`.

## 2026-09-02 — N72 mapping and real-human contract closure

- **Frozen scope:** N72 froze a new protocol, source/environment manifests and read-only N36–N71 evidence before work. No checkpoint, candidate definition, Hungarian solver, metric, third-party SAM3 file, historical output, replay, training, calibration, selector or LoRA was started or modified. New artifacts are isolated under `outputs/N72/`; large-workspace scaffolding is under `/data2/usr_for_deadline/SAM3_InterMOT_N72/`.
- **ID-axis diagnosis:** The audit reproduced the old stable-ID mutation problem: official `out_obj_ids` were overwritten by adapter-visible IDs before legacy export. The N71 official branch has `927` frames and `9,333` candidate rows, with `0` raw-native fields, `0` adapter-external fields and `9,333` explicit public-mapping-unavailable rows. N70's `92,070` legacy rows and `183` unmatched public assignments remain frozen context, not N72 five-axis evidence. The first N72 scanner import failure and incomplete smoke-root scan are preserved under `outputs/N72/attempts/`.
- **Minimal repair:** `PromptObjectObservation.raw_sam_object_id` now preserves only the official raw ID; the opt-in `include_raw_provenance=True` export exposes raw and adapter axes while the default export remains unchanged. The focused raw/export/mapping suite passed, the full project regression passed `143` tests, and the third-party SAM3 hash audit is unchanged (`591/591`). No public ID was fabricated.
- **Exact mapping result:** The N72 bridge accepts only authoritative exact sources and keeps `EXACT`, `UNMAPPED_NO_SOURCE`, `AMBIGUOUS_ONE_TO_MANY`, `COLLISION`, `AXIS_MISMATCH`, `STALE_MAPPING`, `CANDIDATE_ABSENT` and `PUBLIC_ASSIGNMENT_ABSENT` explicit. The complete N71 scan is `9,333/9,333 AXIS_MISMATCH` for the new five-axis contract because its public mapping is unavailable; no IoU, GT, appearance or future heuristic was used. N70's old four-axis boundary rows are not promoted.
- **Real-human path:** An independent external JSONL adapter, append-only recorder and strict validator were added. It requires direct human `public_id`, raw BOX/CLICK/CONFIRMED_MASK payload digest, session/annotator/timestamp/frame provenance, candidate-complete H20/H50/H100 ranges, exact mapping axes, correction-before-memory-write and event-frame read-hidden/event+1-first-visible evidence. It rejects GT/simulator fields, machine masks, candidate gaps, duplicate IDs, ambiguous mapping and nonzero runtime future GT. Verified real-human event count remains `0`; no synthetic record was created.
- **Regression/diagnostic:** The first 12-test valid-fixture failure was caused by an omitted required `runtime_future_gt_used=false` field in the toy fixture; it is preserved and the repaired focused suite passed `30/30`. Causal tests confirm event-frame output is unchanged after a write, first memory visibility is event+1 and hard negatives cannot be overridden. The N72 strict boundary diagnostic emitted `0` eligible rows: all N71 official rows lack public mapping. N70's `38,400` boundary rows are descriptive historical context only; no TrackEval was started.
- **Gate/next step:** N72 final status is `CANDIDATE_PROVENANCE_PASS_PUBLIC_MAPPING_BLOCKED`, with `research_gate=NOT_RUN_NO_REAL_HUMAN_TAPE`, `efficacy_status=NOT_ASSESSED`, and all downstream authorization flags false. The minimum next step is external provenance-complete human UI collection plus a candidate-complete exact public mapping export, followed by the documented validator; only then can a real full-loop/replay be considered. Report: `docs/N72_FINAL_REPORT.md`; machine gate: `outputs/N72/n72_final_status.json`.

## 2026-09-02 — N72R1 same-run public mapping and real-human runtime closure

- **Protocol and isolation:** N72R1 froze the N72 inputs and copied only code/config/test scaffolding into `/data2/usr_for_deadline/SAM3_InterMOT_N72R1/worktree`; historical N36–N72 outputs, checkpoints, and `third_party/sam3` remain read-only. The new protocol preserves the official checkpoint, candidate ordering, SAM3 backend, mapping semantics, no-runtime-GT rule, and no efficacy/training goal.
- **Contract implementation:** Candidate V2/UID V2, separate official raw/adapter/local/global/association/public axes, same-run assignment sidecars, explicit handover status, append-only/path-safe transaction logs, action-specific real-human V2 validation, allocator-backed ADD, server causal guard, standard-library ingestion UI, and focused toy tests were added. A StateManager PID is still association-local; no numeric PID-to-public bridge was inferred.
- **GPU structural evidence:** The first post-repair smoke failure was a validator false negative: legacy/V2 common-field comparison required raw float64/float32 and double-normalized feature arrays to be byte-identical (`1548` rows). The failure is preserved. The minimal canonical-boundary repair passed the same frozen `dancetrack0001` window: `160/160` frames, `1548/1548` candidate rows, zero equivalence failures, and zero runtime GT use. Six independent official-SAM3 windows then passed structural export: `927/927` frames, `9333` V2 rows, zero missing/duplicate/UID-collision/axis-mismatch rows, and same-run sidecar coverage `1.0`.
- **Mapping and real input:** The six-window export has `0` final public assignments and `9493` explicit `PUBLIC_ASSIGNMENT_ARTIFACT_ABSENT` rows because the active runtime has no audited public-authority resolver; cross-window handover is therefore unresolved rather than fabricated. The external UI/validator and a four-slot collection queue are ready, but real human event count remains `0`; all earlier `simulated_from_gt` records remain synthetic and were not imported.
- **Testing and decision:** Focused N72R1 tests pass `14/14` and full CPU regression passes `157/157` with three warnings. No replay, calibration, selector, LoRA, or production promotion was authorized. N72R1 remains structurally partial: the local next step is external direct human annotation plus an explicit same-run public-ID resolver, not more score/weight experiments.

## 2026-09-02 — N72R2 public-ID closure and autonomous GT-simulated loop

- **Frozen scope:** N72R2 kept N36--N72R1 evidence, checkpoint, candidate definition, metrics, Hungarian implementation and `third_party/sam3` read-only. New code and artifacts were isolated under `/data2/usr_for_deadline/SAM3_InterMOT_N72R2/`; all interaction evidence remains explicitly `simulated_from_gt`, and no real-human tape was created or imported.
- **Stage 01:** The same-run `TrackManager.final_mot_track_id` authority bridge and the exact candidate×public-ID+NONE solver completed the frozen 160-frame `dancetrack0001` smoke: `1548` candidate rows, `160/160` exact-solver frames, mapping coverage `1.0`, association-state IDs not treated as public IDs, and runtime future GT `false`. The first exact-solver smoke failure (`UnboundLocalError` in the frame writer counter) remains preserved before the targeted repair.
- **Stage 02 handover:** The initial independent session exposed only `7` next-session overlap tracks against `13` prior overlap tracks (`7/13=0.53846`). Three per-object seed/recovery attempts remained incomplete; their official seed failure, zero-recovery partial output and concept-fallback partial output are preserved. A distinct official multi-box rebind smoke used the latest prior runtime boxes at or before boundary frame `416` (older rows were used for objects not visible at that boundary): `13` requested, `11` observed on the persisted-box prompt, `10` after the official adapter's sanitized batch, `10` recovered, and `9/13=0.69231` overlap transactions. Three objects remained unresolved, including a same-frame duplicate-box authority ambiguity; no raw SAM-ID equality or future GT was used.
- **Gate:** Stage 02 failed candidate recall/public handover, so the fixed `1→2→6` ladder stopped at `1/6`. The causal simulated observer was locally contract-tested (`8` focused tests pass), but official six-action events, current-frame correction, public-ID memory, M0--M4/NO_WRITE replay, H20/H50/H100 scoring and strict future-effect gate were not run. There are therefore no N72R2 efficacy numbers and no authorization for calibration, selector, decoder LoRA or production changes.
- **Decision/next:** N72R2 is `BLOCKED_CANDIDATE_RECALL`, not a scientific effect pass. The next minimum step is a proven official multi-object/session rebind primitive that preserves every persisted candidate authority (including occluded/duplicate-box cases), followed by the unchanged handover gate. Public identities must not be recovered from raw IDs, geometry alone or future GT.

## 2026-09-02 — N72R3 persistent public identity across independent SAM sessions

- **Frozen contract:** N72R3 froze the N72R2 inputs, checkpoint, candidate definition, Hungarian/evaluation definitions, seeds and strict gates. The persistent identity contract is explicit: `public_id`, `lineage_id`, TrackManager track, association state, appearance memory and motion state belong to one sequence-lifetime identity; session-local candidate bindings may be cleared, and `ACTIVE`/`LOST` may change. No N36–N72R2 evidence or `third_party/sam3` file was modified.
- **Authority repair:** The authority audit preserved the N72R2 root cause: candidate-first/local association state was not a valid public authority and could accept conflicting public IDs. N72R3 introduced an immutable state→lineage→public binding, one outer persistent runtime/TrackManager, explicit NONE/LOST handling, session-boundary detachment and atomic backend→identity→memory transactions. The #1007 toy contract passed: candidate 17 in Session A, NONE/LOST at the boundary, and candidate 8 in Session B all resolve to public ID 1007 with stable lineage/track identity. This is structural evidence, not a scientific efficacy result.
- **Official current-frame path:** Six frozen `simulated_from_gt` events completed official SAM3 current-frame correction and 512-D OSNet feature writes: `6/6` correction and `6/6` finite memory writes. The correction occurs before memory write, the event frame cannot read the new memory, and first visibility is event+1. No real human tape was created; GT was isolated to the simulated oracle/posthoc paths.
- **Candidate/runtime gates:** The persistent baseline passed one-, two- and six-window structural checks with public restore coverage `1.0`, renumber `0`, lineage loss `0`, and runtime future GT `false`. The candidate-recall diagnostic is partial performance evidence: H20 overall recall `0.6923076923`, with AUTHORITATIVE_REASSIGN `0.8701298701` and RECOVER_IDENTITY `0.35`; only six eligible events/sequences were available against the preregistered `40/20` target, so no event duplication was used.
- **Paired effect result:** Six events × six variants × 101 frames completed with the frozen candidate stream and global Hungarian solver. M1/M2 changed target appearance scores but produced no assignment crossings. M3/M4 produced 20 H20 crossings, 15 correct and 0 incorrect, but identity-utility mean was only `0.00074865675` and the sequence-cluster 95% CI lower bound was `0`; H50/H100 lower bounds were also `0`. Protected-ID regression and runtime GT leakage were `0`. The strict result is `FAIL_FUTURE_EFFECT`, not PASS.
- **Mechanism diagnosis:** Five CPU-only rounds were completed from sealed artifacts. The exact target-row margin residual was about `7.57`, while appearance deltas at the same alternative column were much smaller (median approximately `0.145–0.685`); only `1–4/600` rows reached the boundary in the frozen probe. The evidence supports a combined candidate-recall and association decision-boundary bottleneck; it does not justify changing the checkpoint, candidate generation, Hungarian solver, threshold, calibration, selector or LoRA.
- **Final decision:** N72R3 is `N72R3_COMPLETE_EVIDENCE_EXHAUSTED_MECHANISM_BRANCHES_NO_STRICT_EFFECT`, with `research_gate=FAIL_FUTURE_EFFECT` and production authorization false. The minimum next step is more independent eligible current-frame events or provenance-complete real-human tape, prioritizing recover candidate recall, followed by the same frozen paired replay. Final report: `docs/N72R3_FINAL_REPORT.md`; machine gate: `outputs/N72R3/n72r3_final_gate.json`.

## 2026-09-02 — N72R4 persistent-state/full-loop closure

- **Scope and preservation:** N72R4 retained N72R3R1's exact-NONE solver, corrected identity-error sign, strict crossing taxonomy and sequence-cluster bootstrap. N36--N72R3R1 evidence, checkpoints, candidate definitions, metrics and `third_party/sam3` were not overwritten or modified. New CPU audit code is isolated in the N72R3R1 worktree and new machine artifacts are under `outputs/N72R4/`.
- **Persistent identity:** Stage 6 captured event-prestate at `t-1`; Stage 7 sourced association/public axes from persistent records rather than candidate index or raw SAM IDs; Stage 8 passed the six-event persistent structural probe. The intended #1007 contract remains structural: a SAM session boundary clears candidate binding and can mark the identity `LOST`, but does not change `public_id`, lineage, track, appearance state or motion state.
- **Official future path:** Stage 9 attempt2 completed the six-event official SAM3 paired future stream with prefix equivalence and `runtime_future_gt_used=false`; the earlier attempt-1 hot-start failure remains preserved. Stage 10 separated spatial correction from memory: M0 candidate recall was `0.7521368/0.6498316/0.5748299` at H20/H50/H100, while NO was `0.6923077/0.6835017/0.7108844`, so correction helped H20 but degraded H50/H100.
- **Memory result:** Stage 11 corrected-stream M0--M4 execution passed structurally but the strict future-effect gate failed. M1/M2 made no assignment changes. M3/M4 made `20/50/100` changes at H20/H50/H100, but had zero true-correct and zero true-incorrect crossings, with H20 missing rate `0.1709402` and sequence-cluster CI lower bound `0`. The historical N72R3 broad M3 labels were therefore not a valid identity effect.
- **Recovery and expansion:** Stage 13 accepted five track-centric proposals; H20/H50/H100 identity-error gain was exactly zero and no true crossings occurred. Stage 14 attempt4 froze a replay-independent policy of 40 events across 24 sequences (`8/8/8/16` by ADD/ATOMIC/REASSIGN/RECOVER) after excluding known atomic/materialization failures, but it was deliberately not executed after the repaired mechanism precondition failed. Earlier attempts, including the forbidden atomic-candidate selection, remain retained and are not adopted.
- **Final decision:** CPU Round 1 records `STOP_DOWNSTREAM_NO_SURVIVING_TRUE_M3_CROSSING`. Stage15 larger replay, Stage16 confirmation, Stage17 independent confirmation and Stage18 full-sequence TrackEval are not authorized/run; no synthetic sample expansion is used to manufacture confirmation. Final status is `M3_SIGNAL_WAS_SOLVER_ARTIFACT` with `research_gate=FAIL_FUTURE_EFFECT`; calibration, selector, decoder LoRA and production association remain unauthorized. The minimum next step is provenance-complete real-human tape plus authoritative public-ID/mapping evidence, or a newly frozen association-interface hypothesis before any further synthetic experiment.

## 2026-09-03 — N72R5 official full-loop structural closure

- **Scope and preservation:** N72R5 used only the frozen train/train_fold N72R5 event policy and N36 candidate tape. All events remain explicitly `simulated_from_gt`; no real-human tape was created, no future GT was read at runtime, and N36–N72R4 evidence plus `third_party/sam3` were not rewritten.
- **Mechanism routing:** The existing N72R5 CPU stages retained the separation between candidate absence and candidate-present public-decision errors. Image-grounded recovery produced no candidate-recall gain, TVC_V0 produced no correct crossing, and the feature-separability gate was negative; these are diagnostic findings, not efficacy passes.
- **Official execution:** The frozen 40-event policy covers 20 independent sequences with action counts `ADD=5`, `ATOMIC=8`, `AUTHORITATIVE_REASSIGN=14`, `RECOVER=13`. Stage07 attempt5 completed `40/40` events × `5/5` branches = `200/200` branch artifacts and `20,200/20,200` frame rows. The independent CPU audit reports zero duplicate/missing worker keys, zero frame/candidate integrity errors, and `runtime_future_gt_used=false`; attempts 1–4 and all earlier OOM/observation failure artifacts remain preserved.
- **Scientific boundary:** The 200 official artifacts contain `146,176` candidate rows with `0` authoritative public-ID assignments and `146,176` explicit unassigned rows. Thus exact public-ID association, posthoc identity effect, H20/H50/H100 effect scoring, calibration, selector, decoder LoRA and production promotion remain unauthorized. The structural result is `N72R5_STRUCTURAL_FULL_LOOP_PASS_EFFICACY_BLOCKED_NO_PUBLIC_MAPPING`, not a model success.
- **Decision/next:** Keep the final gate/report and the lossless Stage07 audit. The minimum next step is provenance-complete real-human event JSONL plus a same-run authoritative public-ID resolver/mapping that survives session boundaries, followed by the unchanged exact-association/future-effect protocol. Do not infer public IDs from native IDs, candidate indices or dataset GT IDs.

## 2026-09-04 — N72R5R1 exact public association and autonomous mechanism closure

- **Hypothesis/protocol:** Complete the missing persistent public-ID plus explicit-NONE association layer on the frozen N72R5 Stage07 stream, then test only preregistered mechanisms. Stage07, checkpoint, candidate order/definition, Hungarian solver, metrics, `max_num_objects=24`, 40-event policy, and sequence-cluster bootstrap (`seed=7202`, `repetitions=2000`) were reused without rerunning SAM3.
- **Structural result:** Stage08 completed `40/40` events, `200/200` branches, `20,200/20,200` frame rows and `146,176` candidate rows; Stage09 found no duplicate/missing/extra branch or frame error, coverage `1.0`, and runtime future GT `false`. A real cross-branch simulated-oracle contamination bug affecting four ADD events was repaired by isolated branch-local oracles; the old inconsistency and all earlier failures remain preserved.
- **Mechanism evidence:** The 40-event decision audit found `3,890` candidate-absent rows versus `9,342` candidate-present decision/solver errors. V0 required residual median/p90/max was `8.865210/9.382172/10.393968`, while observed residual was capped at `1.0`. Feature direction rates were anchor `0.740884`, prototype `0.624892`, temporal `0.197759`; the small sequence-disjoint TVC_V1 verifier reached holdout AUC `0.662822` and produced an incremental but unsafe local effect. The persistence probe changed `3,105` fused scores but zero B1 assignments across `4,040` frames; H100 target-public overwrite was `2,407/3,200`.
- **Effect/gate:** The corrected V0 Stage10 contains `585` metrics over `39` evaluable events and `19` sequence clusters. Primary B4−B0 H20/H50/H100 means were `-0.485982/-0.506111/-0.525982`, with lower CIs `-0.564105/-0.603379/-0.626231`; H20 correct/incorrect crossings were `5/17` and protected regression was `1,014`. The gate is `FAIL_FUTURE_EFFECT`, not PASS.
- **Decision/next:** Six evidence rounds do not support a production mechanism; status is `EXHAUSTED_PRE_REGISTERED_MECHANISMS_NO_EFFECT`. All calibration, selector, decoder LoRA, confirmation and production promotion flags remain false. All events remain `simulated_from_gt`; real human tape count is `0`. The minimum next step is provenance-complete real-human tape plus a newly frozen association-interface probe, not blind weight scaling or checkpoint changes. Final report: `docs/N72R5R1_FINAL_REPORT.md`.

## 2026-09-04 — N72R6 target-scoped correction, recovery and fallback closure

- **Protocol and implementation:** N72R6 froze the N72R5R1 B0 main stream and 40-event policy, then implemented an independent target-correction session, correction epoch, scope-aware native authority, target-exclusive public/NONE domain, human-ROI verification gate and a separately registered frozen-B0 target-main fallback. No checkpoint, Hungarian solver, metric, `third_party/sam3`, N36–N72R5R1 evidence or main B0 stream was modified. All interactions remain `simulated_from_gt`; real human tape remains `0`.
- **Execution:** The recovery stream was completed as `32/32` validated events over `18` sequences; the original official no-observation failure and retry facts remain preserved, while the legal recovery-miss representation completed the replacement. The fallback smoke passed `4/4` actions and the full fallback C0/C1 replay passed `32/32` events. The corrected structural audit reports `main_candidate_mutation=0`, target-domain/runtime-GT/public-axis violations `0`, and fallback UID audit mismatches `0`.
- **Effect:** Recovery-only C1 had H20 `-0.2054662` with CI `[-0.3455225, 0.1312918]` and target candidate recall `0.9878125`. The fixed human-anchor gate had H20 `-0.3599367` with CI `[-0.4591578, -0.0425281]`, reducing protected regression from `49` to `20` but reducing target candidate coverage. The target-scoped B0 fallback reduced the negative H20 mean to `-0.2478248`, but CI remained `[-0.3615559, 0.0262414]`; H50/H100 were `-0.4080980/-0.4398112`, protected regression remained `20`, and correct/incorrect crossings were `5/2`. The strict future-effect gate therefore remains `FAIL_FUTURE_EFFECT`.
- **Root cause:** The fallback replay has `831` accepted target-session rows, `374` legal frozen-B0 target-main fallback rows, and `1995` future frames with no source. Posthoc root-cause audit finds `2312` target-candidate-absent visible frames and `351` drift frames; among target-session rows, native-scope match and explicit target assignment are both `831/831`, with no nonfinite or nonpositive target-public base scores. This rules out native/geometry/solver rejection as the primary bottleneck and supports `TARGET_SESSION_CANDIDATE_PROPAGATION_AND_SPATIAL_QUALITY`.
- **Decision:** N72R6 is complete for the target-scoped mechanism routes but not a positive research result: `FAIL_FUTURE_EFFECT`, production/calibration/selector/decoder-LoRA authorization all false. C2/TVC was not run because the preregistered C1-positive prerequisite was not met; this result must not be interpreted as a TVC test. The minimum next step is a newly frozen target-session candidate-source/propagation quality probe on the unchanged B0 main stream, before C2/TVC, training or production promotion. Code was pushed to `codex/n72r6-target-scoped-correction` at commit `9bab1e4344796e43c035a87f192b70de48d5dded`.

## 2026-09-04 — N72R7 candidate-generator route and N72R8 deferred confirmation closed

- **Scope and preservation:** The N72R7 R5 route reused the frozen N72R5R1 B0/public-assignment stream, checkpoint, candidate definitions, Hungarian solver, metrics and H20/H50/H100 protocol. New candidate-generator and confirmation evidence stayed under `outputs/N72R7/`; N36–N72R6 evidence and `third_party/sam3` were not rewritten. All events remain explicitly `simulated_from_gt`; real human tape remains `0`.
- **R5 candidate source:** A fixed multi-query official SAM3 target-session re-query (center-shrink plus four fixed offsets) completed `32/32` events over `18` independent sequences with `4859` audited rows and no runtime GT. It increased H20 candidate recall to `0.9494290`. Against frozen B0, H20 identity-error reduction was `0.0522023` with CI `[0.0069329, 0.1301042]`, but H50/H100 reductions were `0.0160875`/`0.0099042` with lower CIs `-0.0055877`/`-0.0037119`. Against the current target-session baseline, R5 H20/H50/H100 identity-error reduction was `0/0/0`, so candidate presence did not yield incremental persistent identity effect.
- **Deferred confirmation:** The preregistered `dancetrack0020` ADD and `dancetrack0049` ATOMIC events used explicit public authority (`state=17→1016` and `1003/1004`), not GT/raw-ID inference. Independent target streams and D1/D2 replay completed `2/2` events, `101` frames each, with runtime audit errors `0` and `runtime_future_gt_used=false`. D2−D1 produced `0` treatment-induced assignment changes and `0` identity-error reduction at H20 (CI `[0,0]`); posthoc correct/wrong switch diagnostics were retained.
- **Failure preservation:** Protocol freeze attempt 1 (candidate columns incorrectly treated as live states), target smoke attempt 1 (missing protocol-level flag), target batch attempt 1 (`ModuleNotFoundError: torch` from the base interpreter), direct smoke-auditor misuse, R5 validator semantic mismatch and first posthoc import failure remain preserved. Minimal fixes were applied and the same inputs were rerun; no gate or metric was relaxed.
- **Decision:** N72R8 is `COMPLETE_CONFIRMATION_FAIL_FUTURE_EFFECT`. The evidence separates candidate presence from assignment change and durable future correctness; no calibration head, selector, decoder LoRA or production identity promotion is authorized. The final report is `docs/N72R8_CONFIRMATION_REPORT.md`, machine gate is `outputs/N72R7/n72r8_final_gate.json`, and the next defensible step is provenance-complete real-human tape or a separately frozen association-interface hypothesis—not more blind query/weight scaling.
