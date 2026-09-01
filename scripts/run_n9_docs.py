"""Generate N9 stage documentation and final report."""

import json
from pathlib import Path


ROOT = Path(".")
DOCS = ROOT / "docs"
DATA = ROOT / "outputs/n9"


def write(name, text):
    (DOCS / name).write_text(text, encoding="utf-8")


def doc_benchmark():
    return """# N9 Tracklet Relinking Benchmark

## Construction

- P0 (frozen SAM3 A0_v2) run on DanceTrack train (40 sequences) and reused on
  val (canonical P0).
- Per-frame decision episodes: for every GT identity at frame f, the positive
  row is the P0 row matched by Hungarian IoU>=0.5; negatives are all other P0
  rows at f; memory = the identity's matched frames before f (last 10 frames,
  max 60-frame gap).
- Total episodes: train 347,995 (40 seqs), val 18,796 (3 seqs used for
  sanity).
- Split: 30 train sequences / 10 calibration sequences, sequence-disjoint.
  Calibration = dancetrack0074, 0075, 0080, 0082, 0083, 0086, 0087, 0096,
  0098, 0099.

## Key observation

P0 tids are long-lived (each tid spans hundreds of frames), but the same GT
identity switches between tids on adjacent frames (e.g., gid0 tid 4->2->1->4
at frames 48-50 on dancetrack0004).  The N8 temporal-error stream therefore
partly reflects per-frame GT-box matching ambiguity in dense dance scenes, not
tracklet death.
"""


def doc_feasibility():
    return """# N9 Feature Feasibility (calibration: 255,741 positive episodes)

| Baseline | AUC | R@1 | R@5 |
| --- | ---: | ---: | ---: |
| ReID cosine | 0.916 | 0.925 | 0.997 |
| ReID + motion (w=1) | 0.996 | 0.990 | 1.000 |
| ReID + motion (w=2) | 0.996 | 0.990 | 1.000 |

Gap-stratified R@1 (ReID+motion): 0-5: 0.992, 6-15: 0.882, 16-30: 0.760,
31-60: 0.725.

Crowding-stratified R@1 (ReID+motion): <=4: 0.999, 5-8: 0.995, 9-12: 0.986,
>12: 0.982.

Conclusion: frozen ReID + motion is very strong on the same-frame decision
task; appearance alone degrades with gap length and crowding.
"""


def doc_training():
    return """# N9 Training Protocol

- Features: frozen OSNet x1_0 (Market1501-trained, 512-d), cached per
  (frame, tid) for all 65 sequences (40 train + 25 val).
- Pairwise MLP: concat(mem, row, motion) -> 256 -> logit; BCE.
- Set-level: SetAssociator (2-layer cross-attention, hidden 256) with
  pairwise motion logits; CE over memories/rows; Hungarian at inference.
- HCPIM: SetAssociator + human anchor blending (0.7 anchor + 0.3 adaptive) +
  negative constraints + correction-conditioned future objective
  (L_gain = max(0, CE_B - CE_A + 0.05), L_preserve = L1 logit consistency).
- Training: 30 train sequences, 2-5 epochs, lr 1e-3, AdamW, batch 512-1024;
  selection on the 10 calibration sequences only.
"""


def doc_calibration():
    return """# N9 Calibration Results

## Association accuracy (calibration decision episodes)

- Pairwise MLP cal accuracy: 100% (R@1 over same-frame negatives).
- Set-level cal R@1: 81.2% after 4 epochs (32,680 frame sets).

## Correction persistence (10 calibration sequences, B8)

| Variant | relinks | anchor relinks | retention t+1/t+3/t+5/t+10/t+30 | TTE median/mean |
| --- | ---: | ---: | --- | ---: |
| reid | 246 | 53 | 53.8 / 40.0 / 37.5 / 26.0 / 23.4 | 1 / 2.19 |
| pairwise | 154 | 22 | 53.8 / 40.0 / 37.5 / 26.0 / 23.4 | 1 / 2.19 |
| auto | 5 | 0 | 52.5 / 40.0 / 37.5 / 26.0 / 23.4 | 1 / 2.19 |
| proposed | 0 | 0 | 52.5 / 40.0 / 37.5 / 26.0 / 23.4 | 1 / 2.19 |

All variants show identical retention, i.e., the added relinking mechanism
does not change correction persistence; TTE median remains 1 frame.

## Calibration gate decision

FAIL_PERSISTENT_IDENTITY_MEMORY / FAIL_INTERACTION_SPECIFIC_GAIN:
human-conditioned memory does not improve retention over the identical AUTO
counterpart, and neither improves over N8's canonical-map behavior.
"""


def doc_three_seq():
    return """# N9 Three-Sequence Sanity (dancetrack0004/0005/0007)

The three-sequence runs below use the final trained models (pairwise, set,
hcpim) and the frozen ReID baseline, run with the same pipeline as a sanity
check after the calibration gate failed.

## Official TrackEval (post)

| Method | HOTA | AssA | MOTA | IDF1 | IDSW | Frag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | 67.924 | 58.891 | 85.327 | 74.272 | 49 | 519 |
| N8 B1-B8 | 67.902-67.922 | 58.84-58.89 | 85.30-85.38 | 74.23-74.27 | 52-55 | 521-524 |
| ReID B0-B8 | 67.90-67.91 | 58.84-58.88 | 85.30-85.38 | 74.22-74.26 | 52-58 | 520-525 |
| Pairwise B0-B8 | 67.90-67.91 | 58.84-58.88 | 85.31-85.36 | 74.22-74.26 | 51-57 | 520-525 |
| Auto B0-B8 | 67.90-67.91 | 58.84-58.87 | 85.30-85.36 | 74.22-74.26 | 53-59 | 520-526 |
| Proposed B0-B8 | 67.90-67.91 | 58.84-58.87 | 85.30-85.36 | 74.22-74.26 | 53-59 | 520-526 |

No baseline collapse with the conservative new-tid relinking; no measurable
improvement either.  Aggressive per-frame reassignment collapses identity
consistency (HOTA 27-57, IDSW 211-5339) and was discarded.

## Retention (B8, 24 accepted events)

All variants identical: t+1 75.0%, t+3 45.8%, t+5 33.3%, t+10 8.3%,
t+30 26.1%; TTE median 1.
"""


def doc_ablation():
    return """# N9 Ablations

## Association overlay strategies (3-seq)

| Strategy | HOTA | IDSW | Note |
| --- | ---: | ---: | --- |
| Aggressive per-frame reassignment (v1) | 27-57 | 211-5339 | memory drift / identity collapse |
| Conservative new-tid-only relinking (v2) | 67.90 | 52-59 | baseline preserved, no gain |

## Feature ablation (calibration decision task)

| Feature set | AUC | R@1 |
| --- | ---: | ---: |
| ReID only | 0.916 | 0.925 |
| ReID + motion | 0.996 | 0.990 |

## Module ablation (human-conditioning)

| Module | Effect |
| --- | --- |
| Human anchor bank | no measurable retention gain (anchor relinks 0-53) |
| Negative constraints | no measurable effect |
| Future identity loss / gain objective | implemented; no calibration gain |
| Non-intervention preservation | necessary: aggressive relinking collapses baseline |

## SAM native features

SAM_NATIVE_FEATURE_NOT_USABLE (see N9_SAM_FEATURE_AUDIT.md); ReID route used.
"""


def doc_persistence():
    return """# N9 Interaction Persistence Analysis

- Retention@1/3/5/10/30 after accepted corrections is identical across N8,
  ReID, pairwise, AUTO and proposed variants (calibration B8:
  53.8/40.0/37.5/26.0/23.4; three-seq B8: 75.0/45.8/33.3/8.3/26.1).
- Time-to-next-error median = 1 frame for all variants; mean 1.0-2.2.
- Anchor usage is negligible: 0-53 anchor-driven relinks per 10 calibration
  sequences.
- Conclusion: one human correction does not persist into future association;
  the bottleneck identified in N8 is not removed by the N9 learned memory.
"""


def doc_final():
    return """# N9 Final Report

## 1. Executive Summary

N9 built a full tracklet-relinking benchmark, froze and cached a pretrained
ReID encoder, implemented and trained pairwise / set-level / human-conditioned
association models, and integrated them into the N8 chronological
verified-error observer as a persistent identity memory.  The mechanism is
technically implemented and runs without baseline collapse under conservative
relinking, but it does not achieve the required scientific goal:

- One human correction still does not persist (retention identical to N8;
  time-to-next-error median 1 frame).
- Human-conditioned association does not beat its identical AUTO counterpart.
- SAM3 native features are not reliably extractable under the pinned API
  (SAM_NATIVE_FEATURE_NOT_USABLE).

Final status: **FAIL_PERSISTENT_IDENTITY_MEMORY / FAIL_INTERACTION_SPECIFIC_GAIN**.
The canonical 25-sequence evaluation was not run because the calibration gate
failed, per the stage protocol.

## 2. Starting Point from N8

N8 showed B1-B8 flat, unlimited dense correction strong (HOTA 80.57 on 25),
t+1 retention ~4-8%, TTE median 1.

## 3. Research Question

Can one human correction become persistent identity knowledge that future
tracklets automatically use?

## 4. AGENTS Research Workflow Compliance

AGENTS.md updated with the research-workflow rule; research_log.md maintained
(N9.0-N9.4); short plan written; experiments prioritized over planning;
only necessary sanity checks performed.

## 5. Recent 2025-2026 Literature and GitHub Audit

No directly applicable method found; conceptual references: FC-Track, MTT,
GTATrack, Seg2Track-SAM2, thermal identity-repair backend, DIPLOMAT
(see N9_RECENT_METHOD_GITHUB_AUDIT.md).

## 6. Why ReID Alone Is Not the Proposed Idea

ReID+motion reaches R@1 0.99 on the same-frame decision task; the proposed
idea is that human-confirmed anchors persist and condition future set
association.  The experiments show the conditioning adds no measurable
benefit on this backbone.

## 7. Training / Calibration / Validation Split

40 train sequences -> 30 train / 10 calibration (sequence-disjoint); val 25
reserved for final evaluation (not reached).

## 8. Tracklet Relinking Benchmark

347,995 train decision episodes; per-frame identity-memory vs current rows.

## 9. SAM3 Native Feature Audit

SAM_NATIVE_FEATURE_NOT_USABLE: OOM on full pass, zero-batch crash on limited
window, no retained per-object features after adapter propagate.

## 10. Pretrained ReID Feature Audit

OSNet x1_0 (Market1501, 512-d), frozen; cached for all 65 sequences.

## 11. Motion and Temporal Context

gap, memory age, crowd, predicted-IoU, center distance, box size, last frame,
tid-change flag.

## 12. Feature Feasibility Results

ReID: AUC 0.916 / R@1 0.925; ReID+motion: AUC 0.996 / R@1 0.990.

## 13. Pairwise Learned Baseline

Pairwise MLP trains to near-zero loss; calibration R@1 100%.

## 14. Automatic Set-Level Association Baseline

SetAssociator calibration R@1 81.2% (32,680 frame sets, 4 epochs).

## 15. Human-Confirmed Persistent Identity Memory

Implemented (anchor bank, adaptive memory, negative constraints); no
measurable retention gain.

## 16. Human Anchor vs Adaptive Machine Memory

Anchor usage 0-53 relinks per 10 sequences; retention identical with/without.

## 17. Positive and Negative Human Constraints

Implemented; no measurable effect.

## 18. Correction-Conditioned Set Association

Implemented with anchor-blended memory tokens; no calibration gain.

## 19. Training Objective

L = L_assoc + 0.5*(L_future + L_gain + 0.3*L_preserve) for HCPIM.

## 20. Future Identity Loss

Implemented over t+1..t+10 windows (3,972 anchor episodes).

## 21. Human Anchor Loss

Anchor contrastive blending in memory token; no separate contrastive head.

## 22. Non-Intervention Preservation

Necessary and effective: conservative new-tid-only relinking preserves
baseline; aggressive reassignment collapses (documented in ablations).

## 23. Interaction Gain Objective

Implemented as max(0, CE_B - CE_A + margin); no gain observed.

## 24. Training Details

AdamW lr 1e-3; pairwise 5 epochs; set 4 epochs; hcpim 2-3 epochs; hidden 256;
CPU training.

## 25. Calibration Results

See N9_CALIBRATION_RESULTS.md: retention identical across variants;
TTE median 1.

## 26. Correction Persistence

Not achieved: retention equals N8's canonical-map persistence.

## 27. Retention@1/3/5/10/30

Calibration B8: 53.8/40.0/37.5/26.0/23.4 (all variants identical).

## 28. Time-to-Next-Error

Median 1 frame; mean 1.0-2.2.

## 29. Tracklet Relinking Accuracy

ReID+motion R@1 0.99; set-level 0.81; pairwise 1.0 (calibration).

## 30. Gap-Stratified Analysis

R@1 drops 0.99 (0-5) -> 0.73 (31-60) for ReID+motion.

## 31. Crowd / Multi-Object Analysis

R@1 0.999 (<=4) -> 0.982 (>12) with motion; set-level does not beat pairwise
on the decision task.

## 32. Three-Sequence Sanity Gate

Pipeline runs, causality holds, no baseline collapse with conservative
relinking; no improvement; aggressive variant collapsed (discarded).

## 33. Frozen N9 Model

No model was frozen for final evaluation because the calibration gate failed;
`outputs/n9/n9_frozen.json` records what was tested and why.

## 34. Canonical 25-Sequence Results

NOT RUN (calibration gate failed; protocol requires stop before 25).

## 35. B1/B2/B4/B8 Results

Three-seq diagnostics only: HOTA 67.90-67.91 (flat vs P0/N8).

## 36. Pre vs Post

Pre/post nearly identical; interaction adds no measurable future effect.

## 37. N8 vs ReID vs AUTO vs Proposed

Statistically indistinguishable (HOTA spread <0.02 on 3-seq; retention
identical on calibration).

## 38. Ablation Study

See N9_ABLATIONS.md.

## 39. Feature Ablation

ReID-only vs ReID+motion: motion matters (R@1 0.925 -> 0.990).

## 40. Human-Conditioning Ablation

Human anchor / negative constraints / future loss: no measurable gain.

## 41. Cost-Performance Curve

Flat at sparse budgets (same as N8); not improved.

## 42. Statistical Significance

No significant differences between variants on the three-sequence diagnostic;
retention identical by construction of the measured events.

## 43. Failure Cases

- Aggressive per-frame reassignment: memory drift, IDSW 211-5339, HOTA 27-57.
- Conservative relinking: no opportunity (P0 tids rarely disappear) and no
  benefit when it fires.
- Anchors rarely usable (0-53 relinks per 10 sequences).

## 44. What Actually Improved

ReID+motion is a strong same-frame discriminator; conservative relinking
preserves baseline; benchmark and training pipeline work.

## 45. What Did Not Improve

Sparse-budget HOTA/AssA/IDF1, retention, TTE, interaction-specific gain.

## 46. Interaction-Specific Gain

None: proposed == auto == reid == n8.

## 47. Novelty Audit Against Recent Methods

No directly applicable 2025-2026 method; this work is best described as
"standard trainable MOT association" (category B) with a failed
human-conditioning attempt; it does not rise to an interaction-aware
persistent identity contribution on this backbone.

## 48. Whether the Method Is More Than SAM3+ReID

No: SAM3 features were unusable, and ReID+motion explains the association
ability; the human-conditioned memory did not add value.

## 49. Reproducibility

Scripts: run_n9_p0_train.py / orchestrator, run_n9_reid_features.py /
orchestrator, run_n9_build_benchmark.py, run_n9_feature_baselines.py,
run_n9_train.py, run_n9_real.py, run_n9_cpu_orchestrator.py,
run_n9_eval.py, run_n9_persistence_analysis.py.  Seeds: torch 42, numpy 42,
bootstrap 42.

## 50. Final Scientific Conclusion

On the frozen P0 backbone, one human correction does not become persistent
identity knowledge through the tested learned memory: the failure is not
scheduling but the backbone's association instability and the absence of a
stable tracklet boundary at which a relink can fire.  ReID+motion alone is a
strong but insufficient patch; it neither persists corrections nor improves
official metrics at sparse budgets.

## 51. Final Stage Status

**FAIL_PERSISTENT_IDENTITY_MEMORY / FAIL_INTERACTION_SPECIFIC_GAIN**.

## 52. Recommended Next Stage

Move to detection-side interaction (Human Query / new-target ADD with visual
propagation) or replace the frozen P0 association with a trainable tracker;
do not invest in interaction scheduling (N10) on this backbone.
"""


def main():
    write("N9_TRACKLET_RELINKING_BENCHMARK.md", doc_benchmark())
    write("N9_FEATURE_FEASIBILITY.md", doc_feasibility())
    write("N9_TRAINING_PROTOCOL.md", doc_training())
    write("N9_CALIBRATION_RESULTS.md", doc_calibration())
    write("N9_THREE_SEQUENCE_RESULTS.md", doc_three_seq())
    write("N9_ABLATIONS.md", doc_ablation())
    write("N9_INTERACTION_PERSISTENCE_ANALYSIS.md", doc_persistence())
    write("N9_FINAL_REPORT.md", doc_final())
    frozen = {
        "stage": "N9-FREEZE-RECORD",
        "date": "2026-08-08",
        "final_status": "FAIL_PERSISTENT_IDENTITY_MEMORY / FAIL_INTERACTION_SPECIFIC_GAIN",
        "route": "ReID + learned association overlay on frozen P0 backbone (N8 protocol)",
        "protocol": "N8 chronological verified-error budget (unchanged)",
        "p0_backbone": "outputs/n5/integrity/canonical_mot_results/b0 (val), outputs/n9/p0_train (train)",
        "features": {
            "sam3_native": "SAM_NATIVE_FEATURE_NOT_USABLE (see docs/N9_SAM_FEATURE_AUDIT.md)",
            "reid": "OSNet x1_0 Market1501 512-d frozen (outputs/n9/checkpoints/osnet_x1_0_market1501.pth)",
        },
        "models": {
            "pairwise": "outputs/n9/models/pairwise_mlp.pt",
            "set": "outputs/n9/models/set_associator.pt",
            "hcpim": "outputs/n9/models/hcpim.pt",
        },
        "split": {"train": 30, "calibration": 10, "val": "reserved (not run: gate failed)"},
        "calibration_gate": "FAIL: proposed == auto == n8 on retention (t+1 53.8%, TTE median 1)",
        "three_sequence_sanity": "no collapse with conservative relinking; no improvement; aggressive variant discarded",
        "canonical_25": "NOT_RUN (protocol stop condition)",
        "decision": "Do not proceed to N10 scheduling on this backbone; next stage: detection-side interaction or trainable tracker.",
    }
    (DATA / "n9_frozen.json").write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    gate = json.loads((ROOT / "outputs/stage_gate.json").read_text(encoding="utf-8"))
    gate["N8_FINAL_REPORT"] = "DONE"
    gate["N9_FEATURES"] = "DONE_REID_SAM_NOT_USABLE"
    gate["N9_BENCHMARK"] = "DONE"
    gate["N9_TRAINING"] = "DONE"
    gate["N9_CALIBRATION_GATE"] = "FAIL_PERSISTENT_IDENTITY_MEMORY"
    gate["N9_INTERACTION_SPECIFIC_GAIN"] = "FAIL_INTERACTION_SPECIFIC_GAIN"
    gate["N9_THREE_SEQUENCE_SANITY"] = "PASS_NO_COLLAPSE_NO_GAIN"
    gate["N9_CANONICAL_25"] = "NOT_RUN"
    gate["N9_FINAL_REPORT"] = "DONE"
    gate["TOTAL"] = "FAIL_PERSISTENT_IDENTITY_MEMORY"
    gate["notes"] = (
        "N9: human-confirmed persistent identity memory and correction-conditioned "
        "association implemented and trained, but retention identical to N8 "
        "(t+1 53.8% calibration B8; TTE median 1), proposed==auto, no sparse-budget "
        "gain; SAM native features not usable; canonical 25 not run per protocol.  "
        "Next: detection-side interaction / trainable tracker."
    )
    (ROOT / "outputs/stage_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    final = DOCS / "SAM3_INTERMOT_FINAL_REPORT.md"
    addendum = """

---

## N9 Addendum (2026-08-08)

Stage N9 (Human-Confirmed Persistent Identity Memory and Correction-
Conditioned Set Association) completed with **FAIL_PERSISTENT_IDENTITY_MEMORY /
FAIL_INTERACTION_SPECIFIC_GAIN**:

- Built a 347,995-episode tracklet-relinking benchmark on DanceTrack train;
  frozen ReID (OSNet Market1501) cached for all 65 sequences; SAM3 native
  features recorded SAM_NATIVE_FEATURE_NOT_USABLE.
- Trained pairwise (cal R@1 1.0), set-level (cal R@1 0.81), and
  human-conditioned (cal R@1 0.87) association models on a 30/10
  sequence-disjoint split.
- Calibration: retention identical across reid/pairwise/auto/proposed and
  equal to N8 (B8 t+1 53.8%, t+3 40%, t+5 37.5%; TTE median 1); proposed
  does not beat its identical AUTO counterpart.
- Three-seq final runs: no baseline collapse with conservative new-tid-only
  relinking (HOTA 67.89-67.91 vs P0 67.92) but no improvement; aggressive
  per-frame reassignment collapsed identity consistency and was discarded.
- Canonical 25-sequence evaluation NOT RUN per the stage's calibration-gate
  stop condition.

Full evidence: `docs/N9_FINAL_REPORT.md`.
"""
    if "## N9 Addendum" not in final.read_text(encoding="utf-8"):
        with final.open("a", encoding="utf-8") as f:
            f.write(addendum)
    print("n9 docs written")


if __name__ == "__main__":
    main()
