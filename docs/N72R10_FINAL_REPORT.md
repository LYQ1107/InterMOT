# InterMOT N72R10 Final Report

**Protocol:** `N72R10_TRUE_CLOSED_LOOP_REACQUISITION`
**Date:** 2026-09-05 (Asia/Shanghai)
**Branch:** `codex/n72r10-true-closed-loop`
**Baseline commit:** `669bdb5fe15368e21a49ca931a04201ce2cc4548`
**Research status:** `FAIL_N72R10_DEVELOPMENT_GATE`
**Research gate:** `FAIL_FUTURE_EFFECT_OR_DEVELOPMENT_READINESS`
**Terminal scientific success:** `false`

## Executive conclusion

N72R10 implemented and executed a genuine causal future-frame re-query session. After
an uncertainty trigger, a fresh official SAM3 session queried the current future frame,
optionally propagated the selected source, and changed the raw/native trajectory binding
while preserving the immutable public ID. The complete runtime and posthoc batch covered
32 frozen events across 18 sequences and passed its structural audit.

This is not a positive scientific or production result. Aggregate E1 temporal-memory
and isolated E2−E1 future-requery effects are positive at H20/H50/H100, but the gate
remains failed because:

1. the combined E1 path has protected-identity regression (`7/7/7` at H20/H50/H100);
2. the validation split has zero positive `FUTURE_FRAME_REQUERY` labels and only 13
   NONE examples, so a target-edge bridge cannot be selected without leakage;
3. the frozen event pool yields only 3,000 train and 200 validation examples, below the
   preregistered 30,000/5,000 target, and cannot be enlarged by duplication, horizon
   extension, or reuse of older static streams.

The runtime found a specific mechanism bottleneck: the frozen model selected 29 fresh
candidates with posthoc target IoU ≥ 0.50, but 21 of those were rejected by the unchanged
global public-ID solver. This is direct model-to-solver global-competition evidence, not
evidence that a new solver bridge will generalize. One event completed the full
trigger → fresh candidate → selection → assignment → raw/native rebinding → immutable
public ID → posthoc wrong-to-correct path.

No calibration head, selector, decoder LoRA, or production association promotion is
authorized. All interaction events remain explicitly `simulated_from_gt`; no real-human
tape was created or imported.

## Frozen boundaries and preservation

The following were kept unchanged:

- N36–N72R9 reports, gates, ledgers, candidate tapes and failure evidence;
- checkpoint, SAM3 backbone/embedding definition and candidate ranking;
- Hungarian solver and H20/H50/H100 evaluation definitions;
- runtime rule that future GT is forbidden;
- `third_party/sam3`;
- shared MOT/OVMOT configurations and checkpoints.

The old N72R9 `TEMPORAL_REQUERY` source was not rewritten. The new Stage 00 audit
classified it as a frozen/static stream. Only the new source `FUTURE_FRAME_REQUERY`
denotes a fresh current-frame SAM3 session.

## Stage completion

| Stage | Result | Main evidence |
|---|---|---|
| 00. Mechanism-label correction | PASS, historical label corrected | `outputs/N72R10/stage_00_status.json` |
| 01. Atomic protected-regression audit | PASS, historical posthoc collision identified | `outputs/N72R10/stage_01_status.json`, `atomic_regression_audit.json` |
| 02. Production pair-repair decision | PASS, no production pair guard authorized | `outputs/N72R10/stage_02_status.json` |
| 03. True future-frame re-query batch | PASS, 32/32 events | `outputs/N72R10/stage_03_status.json` |
| 04. Causal training corpus | PASS, corpus sealed but limited | `outputs/N72R10/stage_04_status.json` |
| 05. Training smoke | PASS, finite forward/backward/save/restore | `outputs/N72R10/stage_05_training_smoke_status.json` |
| 06. Isolated source-aware training | PASS as development probe | `outputs/N72R10/stage_06_training_status.json` |
| 07–08. Paired replay and audit | PASS, attempt 2, 32/32 events | `outputs/N72R10/stage_08_status_attempt_02.json` |
| 09. Future-effect/development gate | FAIL, no downstream promotion | `outputs/N72R10/stage_09_gate.json` |
| 10. Training-distribution audit | LIMITED, new causal interactions required | `outputs/N72R10/stage_10_training_distribution_audit.json` |
| 11. Root-cause classification | COMPLETE, bridge deferred | `outputs/N72R10/stage_11_root_cause_classification.json` |

The stale Stage 08 and aggregate files from replay attempt 1 remain preserved. Final
claims use only the complete attempt-2 artifacts:

- `outputs/N72R10/stage_07_replay_attempt_02/ccam_paired_replay_results.json`;
- `outputs/N72R10/stage_07_replay_attempt_02/runtime_audit.json`;
- `outputs/N72R10/stage_07_replay_attempt_02/event_metrics.jsonl`.

## True future-frame session

Each fresh session was created after the event and started at a causal trigger frame.
Its local frame zero mapped to the global trigger frame, and its window ended no later
than `event_frame + 100`. The query box came from causal predicted state, not a future
GT box. Candidate rows remained public-ID-free until association. The selected source
was re-run in its own active session and then closed.

The official runtime policy recorded in the valid smoke was:

```text
official_batched_grounding = true
official_batched_grounding_batch_size = 1
offload_video_to_cpu = true
offload_output_to_cpu_for_eval = true
offload_state_to_cpu = false
trim_past_non_cond_mem_for_eval = false
```

The state-offload field is explicitly false; no unsupported state-offload mechanism is
claimed. The implementation uses process/session isolation and pinned official controls.

### Structural audit

| Quantity | Count/result |
|---|---:|
| Events | 32 unique |
| Independent sequences | 18 |
| Action counts | ADD 4; ATOMIC 3; REASSIGN 14; RECOVER 11 |
| Future frames | 3,200 |
| Future candidate/nonempty rows | 3,020 |
| Missing frames | 0 |
| Duplicate candidate UIDs | 0 |
| Mapping failures | 0 |
| Public-ID changes | 0 |
| Event-frame memory-read violations | 0 |
| Future-boundary violations | 0 |
| Unclosed backends | 0 |
| Runtime future-GT reads | 0 |
| Deterministic mask→box repairs | 1 |

The batch used at most four independent processes, one per GPU, on GPU IDs `1,2,3,4`.
No fifth GPU was used.

## Failure and repair history

All failures remain under `outputs/N72R10/attempts/` and were not converted into passes:

1. The first true-session batch attempt failed on `dancetrack0033:154` because an
   official observation box had non-positive area.
2. The second attempt identified the precise condition: a non-empty official mask paired
   with a zero-area box.
3. The targeted third attempt derived only the box from that same official non-empty mask
   and retained the raw box in the audit. The full batch then passed.
4. Replay attempt 1 exposed a provenance-schema failure because `done.json` lacked
   `started_at_utc`. The failure remains retained; a provenance smoke passed after the
   timestamp fix, and the complete attempt-2 replay used the same frozen input.

The deterministic mask→box operation is an observation normalization repair, not a
synthetic candidate or metric substitution.

## Training distribution and actual training

The isolated model was trained after the causal replay corpus was sealed. It does not
modify SAM3, candidate generation, the Hungarian solver, or production association.
Dataset GT was used only offline to attach labels after causal feature construction;
every runtime row recorded `runtime_future_gt_used=false`.

### Frozen corpus

| Split | Events | Sequences | Examples | Future rows | Future rows selected as positive label |
|---|---:|---:|---:|---:|---:|
| Train | 30 | 16 | 3,000 | 2,833 | 380 |
| Validation | 2 | 2 | 200 | 187 | 0 |

The maximum causal future-frame examples from the existing frozen pool is 3,200. Source
row counts were:

```text
train:       MAIN_B0_CANDIDATE 23002,
             TARGET_SESSION_CURRENT_RAW 2961,
             FUTURE_FRAME_REQUERY 2833
validation:  MAIN_B0_CANDIDATE 1300,
             TARGET_SESSION_CURRENT_RAW 200,
             FUTURE_FRAME_REQUERY 187
```

Train labels were `2,662` highest-IoU target candidates, `68` target-not-visible rows
and `270` visible/no-candidate rows. Validation labels were `187` highest-IoU, `2`
target-not-visible and `11` visible/no-candidate. Validation source-specific FUTURE
positive coverage is empty and NONE accuracy is only `1/13 = 0.0769231`.

### Model and training configuration

```text
architecture: N72R10_source_conditioned_candidate_set_plus_causal_memory_temporal_context
candidate feature dim: 530
source feature dim: 5
temporal feature dim: 8
hidden dim/layers/heads: 96/1/4
trusted/distractor memory slots: 4/4
dropout: 0
seed: 7210
batch size: 128
learning rate: 5e-4
weight decay: 1e-4
pairwise weight/margin: 0.15/0.20
NONE example weight: 2.0 (frozen before training)
maximum epochs/patience: 40/8
device: GPU 1
epochs completed: 15
best epoch: 7
best validation loss: 0.66012655
```

Training smoke passed finite forward/backward and checkpoint save/restore. Formal
training completed, but the checkpoint is development-only:

- train accuracy: `0.7853333`, target-candidate accuracy `0.7776108`, NONE accuracy
  `0.8461538`;
- validation accuracy: `0.795`, target-candidate accuracy `0.8449198`, NONE accuracy
  `0.0769231`;
- validation FUTURE positive-label coverage: `0`.

### Why the corpus was not silently enlarged

The older 40-event pool overlaps N72R10 by 32 events. Its eight additional events have
old candidate tapes but no same-run N72R10 c0/c1/target stream and no explicit public
authority. Reusing them would mix static historical streams with the new causal
mechanism. Duplication, extending the 100-frame horizon and selecting from posthoc
outcomes were prohibited. The distribution audit therefore found zero additional valid
local events.

## Paired E0/E1/E2 replay

The complete attempt-2 replay used the same event, prefix, future candidate stream and
posthoc protocol:

```text
E0 = B0 baseline
E1 = TEMPORAL_CURRENT_V2
E2 = TRUE_CLOSED_LOOP_REQUERY_V2
```

The isolated module-2 comparison is **E2−E1**, not E2−E0.

| Comparison | Horizon | Identity-error reduction | 95% CI | Assignment changes | Correct / incorrect changes | Protected regressions |
|---|---:|---:|---:|---:|---:|---:|
| E1−E0 | H20 | 0.0880914 | [0.0402454, 0.2370541] | 123 | 57 / 3 | 7 |
| E1−E0 | H50 | 0.0392535 | [0.0110198, 0.1268395] | 176 | 72 / 11 | 7 |
| E1−E0 | H100 | 0.0287540 | [0.0070238, 0.0968725] | 267 | 105 / 15 | 7 |
| E2−E1 | H20 | 0.0163132 | [0.0016340, 0.0599673] | 44 | 10 / 0 | 2 |
| E2−E1 | H50 | 0.0141570 | [0.0011111, 0.0508511] | 69 | 24 / 2 | 3 |
| E2−E1 | H100 | 0.0099042 | [0.0000057, 0.0341041] | 94 | 34 / 3 | 6 |

Both aggregate E2−E1 lower-CI rows are above zero, but aggregate significance is not
the sole gate. Action-level evidence is weak: several action/horizon lower bounds are
zero or negative, and the E1 path violates the zero protected-regression requirement.
The replay had `9,696` runtime frame rows and `80,207` candidate rows; duplicate,
missing, partial, unavailable, mapping and runtime-GT audit errors were all zero.

## Closed-loop milestone and mechanism diagnosis

The posthoc milestone audit used GT only after runtime artifacts were sealed:

| Runtime quantity | Count |
|---|---:|
| Uncertainty triggers | 591 |
| Applied fresh sources | 591 |
| Fresh source candidates | 591 |
| Fresh candidates selected by frozen model | 33 |
| Selected candidates with target IoU ≥ 0.50 | 29 |
| Selected wrong candidates | 4 |
| Fresh source assigned to target public ID | 12 |
| Fresh source assigned with target IoU ≥ 0.50 | 8 |
| Fresh source assigned wrongly | 4 |
| Good selected candidates refused by global solver | 21 |
| Posthoc wrong-to-correct frame occurrences | 34 |
| Logical raw/native rebinding events with public ID stable | 295 |
| Complete end-to-end milestones | 1 |

The complete milestone was:

```text
event: n72r5-pool-n37-dancetrack0062-0291-add_new_identity-001
event frame: 291
fresh selection: frame 293
posthoc E1-wrong → E2-correct: frames 297–300
public identity: unchanged
raw/native binding: changed by target-session native scope
```

Root-cause classification:

- **Primary D — model-to-solver global competition:** 29 target-quality fresh
  selections versus 21 target-public solver refusals.
- **Secondary C — model/admission rejection:** 591 applied sources but only 33 selected;
  source-specific validation cannot distinguish the future positive class yet.
- **Secondary F — protected identity competition:** E1 has 7 regressions at every
  horizon, and E2 adds 2/3/6 incremental regressions.
- **Not A:** the uncertainty trigger did execute.
- **Not B:** fresh candidates existed at every applied trigger.
- **Not a clean E pass:** one full milestone exists, but it does not establish stable
  long-horizon generalization.

This rules out “the module was never called” as the only explanation. It does not
justify changing the global solver or adding an unvalidated target-edge bridge.

## Gate decision

| Gate condition | Result |
|---|---|
| True future-requery integrity | PASS |
| Replay integrity | PASS |
| E1 aggregate H20/H50/H100 lower CIs positive | PASS |
| E2−E1 aggregate lower CIs positive | PASS |
| Correct crossings exceed incorrect crossings, aggregate | PASS |
| E2 incremental protected regression no worse than E1 count | PASS |
| E1 protected regression equals zero | **FAIL** |
| Complete live milestone present | PASS |
| Validation has positive FUTURE_REQUERY labels | **FAIL** |
| Train/validation size reaches preregistered target | **FAIL** |
| Calibration/selector/decoder LoRA authorization | **NOT AUTHORIZED** |
| Production promotion | **NOT AUTHORIZED** |

The final result is a development-gate failure, not a failed implementation and not a
scientific success. The current evidence supports continuing only with a new lawful
causal interaction pool and a sequence-disjoint validation split containing positive
future-requery labels. An isolated target-edge bridge may be trained only after that
input exists and its selection rule is frozen.

## External method audit

The audit covered GitHub, OpenReview, arXiv, CVF Open Access and official project pages,
prioritizing 2025–2026 human-in-the-loop tracking, interactive tracking, online
appearance/memory, association, calibration and parameter-efficient adaptation. The
listed revisions were queried on 2026-09-05; no external code was copied here.

| Method | Paper/official page | Code/revision | Reusable idea and limitation here |
|---|---|---|---|
| SENTRY | [arXiv:2606.24449](https://arxiv.org/abs/2606.24449) | [HamadYA/SENTRY](https://github.com/HamadYA/SENTRY), `dd4486c7eeadd7e7022854e29e95e3101390ce65`, 2026-07-15 | Temporal/neighbor-aware candidate admission; design context only. |
| InteractTrack / IMAT | [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_Interactive_Tracking_A_Human-in-the-Loop_Paradigm_with_Memory-Augmented_Adaptation_CVPR_2026_paper.pdf) | [NorahGreen/InteractTrack](https://github.com/NorahGreen/InteractTrack), `5f149d4001a84c8b83129192057bf6dd820f71b3`, 2026-06-16 | Timestamped interaction and positive/negative memory provenance; no historical human tape exists here. |
| TCEI | [arXiv:2603.21629](https://arxiv.org/abs/2603.21629) | [1941Zpf/TCEI](https://github.com/1941Zpf/TCEI), `145d1b8431398156f8d9f854430e306fdee39eaa`, 2026-03-30 | Separates transient memory from accumulated calibration; source-accounting reference. |
| MOTIP | [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.pdf) | [GISer-WB/MOTIP-2](https://github.com/GISer-WB/MOTIP-2), `012856c1dc13b324064e79339ae71054518d1b5e`, 2025-03-23 | Identity-conditioned association context; solver is not replaced. |
| TrackTrack / STAR context | [TrackTrack CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.pdf) and [OpenReview](https://openreview.net/forum?id=fmCnNQjZrr) | [kamkyu94/TrackTrack](https://github.com/kamkyu94/TrackTrack), `ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da`, 2025-09-24 | Track-aware/global-association diagnosis; no code imported. |
| DAM4SAM | [official repository](https://github.com/jovanavidenovic/DAM4SAM) | `9c954504b39ebca4c412f207be0787c26bfac85a`, 2026-04-07 | Distractor-aware memory admission for SAM2; not used as a SAM3 API. |
| SAM2Long | [arXiv:2410.16268](https://arxiv.org/abs/2410.16268) | [Mark12Ding/SAM2Long](https://github.com/Mark12Ding/SAM2Long), `d70b50a7936fec55af201244ecde3d4433aff943`, 2026-08-14 | Long-video memory pruning/uncertainty reference; no SAM2 memory tree transplanted. |

The full source audit, including exclusions and dispositions, is in
`docs/N72R10_EXTERNAL_REFERENCE_AUDIT.md`.

## Isolation, tests and reproducibility

New implementation/scripts/tests are isolated to this InterMOT branch and
`outputs/N72R10/`. No file under `third_party/sam3` was modified. Focused validation
passed:

```text
py_compile: passed for changed N72R10 Python files
pytest tests/test_n72r10_future_requery_session.py: 7 passed
git diff --check: passed
```

The training checkpoint was written only under `outputs/N72R10/training/` and is not
imported by production association. Important artifact hashes:

```text
protocol (N72R9 frozen input): e012ecc3bd64fec4409fccd57d920f3690b8b1c4b5a2dd557e03e2bfb43ef0e9
stage_03 future audit: 0156ffc973ca54d6b0c3ea322e72b9efbc6e8f4cce7d5a9f611b0390e6308c03
stage_03 status: f26c74442fcf15ba7302e5941435eafb0411bf74293b6de7fd2bccb69ae39c45
replay runtime audit: 55f477b08d5c76e7410cb6c68f5c9ab1c1b0bed33edc0cee872589653ca8cb14
replay event metrics: 1051af89ece5e90bd7f5deaffe2e82c7f35781b1537df58d3c22f44d720cff32
replay result: bd3c79b510eeebacb18651fc1a646fff18c12051ad886c34b954937a81023284
stage_08 status: 396e67bb4cc05571373cc9ab2a6b412ead041275b3562c2ecb611e97af91279f
training corpus manifest: 0788900a9dfcc0f1eeefa4f989740d3d8499a44e7a0ffa692e92a7a388e679da
train.npz: 863c110574ec6a34e80a0c156035b4087c35e92449a4689869f7e3c3a1ee440d
validation.npz: d260726c98968b9da6178742e3c1636d10ac85a23f450c1314166af9ad0bbf4c
training checkpoint: 0e7f8f97b22da85b633797bf9756c75d4df83bd81b93b3efeca20ce8b3b5e294
training history: 3537e3b1d469e4db7cfde2c6adb9b6dce79dd04150c1a7831c91589eae5f2c3
final gate: 1ccb8634236e0a351ec4c604e702e63f26f97c87605b3ddf3f5a8fdcb6cf2d51
training-distribution audit: af716328fa6e76f905e386d05a580f5356468227e57f443ddde74085edb8ab33
```

## ICLR 2027 schedule and next minimum step

Using the frozen project schedule, the remaining calendar on 2026-09-05 is:

| Milestone | Date | Approximate time remaining |
|---|---|---:|
| Abstract deadline | 2026-09-18 AoE | 13 days |
| Full paper deadline | 2026-09-25 AoE | 20 days |

The next defensible experiment is not another blind weight scan. It is:

1. generate a larger same-run causal interaction pool with explicit public authority,
   c0/c1/target streams and fresh future sessions;
2. reserve sequence-disjoint validation sequences containing positive and negative
   `FUTURE_FRAME_REQUERY` examples;
3. train an isolated target-edge bridge with frozen rules and evaluate it once on that
   holdout, including protected IDs and H50/H100;
4. promote nothing unless the unchanged strict future-effect, protected-regression and
   leakage gates all pass.

Real-human evidence is a separate unmet requirement: the current 32 events are
`simulated_from_gt`, not historical human clicks. No current result should be described
as real-human validation.
