"""Build the immutable N68 machine-readable final gate and source audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/n68"
GATE = OUT / "n68_final_gate.json"
SOURCES = OUT / "research_sources.json"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def method_summary(method: dict[str, Any]) -> dict[str, Any]:
    return {
        key: method.get(key)
        for key in [
            "frame_count",
            "target_candidate_recall",
            "target_correct_rate",
            "future_identity_error_rate",
            "score_change_frame_rate",
            "assignment_change_rate",
            "correct_changes",
            "incorrect_changes",
            "neutral_changes",
            "untouched_assignment_changed_total",
            "untouched_regression_frame_rate",
            "re_correction_opportunity_proxy",
            "mean_target_rank",
            "mean_target_vs_distractor_margin",
            "none_predicted_count",
            "none_accuracy_posthoc",
        ]
    } | {
        "horizons": {
            str(h): {
                "utility": item.get("mean_utility_delta_raw_event_variant"),
                "ci95": item.get("sequence_cluster_bootstrap", {}).get("ci95"),
                "sequence_count": item.get("sequence_cluster_bootstrap", {}).get("sequence_count"),
                "seed": item.get("sequence_cluster_bootstrap", {}).get("seed"),
                "repetitions": item.get("sequence_cluster_bootstrap", {}).get("repetitions"),
            }
            for h, item in method.get("horizons", {}).items()
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rewrite-existing",
        action="store_true",
        help="Rewrite the N68 gate after preserving a prior validation artifact.",
    )
    args = parser.parse_args()
    stage_paths = {f"stage_{i:02d}": OUT / f"stage_{i:02d}_status.json" for i in range(0, 7)}
    stages = {name: read_json(path) for name, path in stage_paths.items()}
    stage02 = read_json(OUT / "replay/paired_replay_results.json")
    stage03 = read_json(OUT / "replay/stage03_paired_replay_results.json")
    stage02_runtime = read_json(OUT / "replay/runtime_status.json")
    stage03_runtime = read_json(OUT / "replay/stage03_runtime_status.json")
    isolation = read_json(OUT / "n68_isolation_regression.json")
    training = read_json(OUT / "training/n68_identity_local_head_training_manifest.json")
    dataset = read_json(OUT / "training/n68_local_association_dataset_manifest.json")

    method_sources = [
        {
            "name": "TACT",
            "year_status": "NeurIPS 2025",
            "paper_url": "https://openreview.net/forum?id=zFGdHL9pcD",
            "code_url": "https://github.com/NancyQuris/TACT",
            "observed_head": "56aeefa8b49df9ec9bc4fd309ea65824a415877d",
            "observed_head_date": "2025-12-27T10:00:09Z",
            "license": "MIT",
            "mechanism": "Identity/semantic-preserving augmentations, PCA over representation variation, projection trimming of samples and prototypes, moving-average prototype refinement.",
            "transferable": "Only as a separately gated representation ablation after a no-trimming branch proves future utility; treat nuisance directions as a hypothesis.",
            "not_transferable": "High variance is not automatically non-causal for DanceTrack clothing/body identity cues; it must not alter the frozen backbone or be used to bypass the gate.",
        },
        {
            "name": "HATReID-MOT",
            "year_status": "ECCV 2026 accepted; arXiv 2025",
            "paper_url": "https://arxiv.org/abs/2503.12562",
            "code_url": "https://github.com/MCG-NJU/HATReID-MOT",
            "observed_head": "3eb440c288bdc5e8548a49c43107f6543c74b264",
            "observed_head_date": "2026-07-23T06:16:23Z",
            "license": "Apache-2.0",
            "mechanism": "History-aware transformation searches a more discriminative ReID subspace using historical trajectory information.",
            "transferable": "A target-conditioned feature projection is a plausible isolated sidecar analogue.",
            "not_transferable": "The N68 sidecar must keep public_id supplied by the event and preserve global Hungarian; it cannot import a separate tracker or infer arbitrary IDs.",
        },
        {
            "name": "MOTIP",
            "year_status": "CVPR 2025",
            "paper_url": "https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.html",
            "code_url": "https://github.com/MCG-NJU/MOTIP",
            "observed_head": "ffc0e905ac196a603027eca8d18fb0dff48c8bcc",
            "observed_head_date": "2026-07-30T12:04:18Z",
            "license": "Apache-2.0",
            "mechanism": "Treats MOT as in-context ID prediction: trajectories with ID information condition direct current-detection ID decoding.",
            "transferable": "The identity-conditioned feature contract motivates conditioning on the known target public ID and competitors.",
            "not_transferable": "N68 deliberately retains the existing candidate stream and Hungarian/NONE solver; a direct arbitrary-ID decoder is out of scope and unsafe for known public IDs.",
        },
        {
            "name": "LA-MOTR",
            "year_status": "ICCV 2025",
            "paper_url": "https://openaccess.thecvf.com/content/ICCV2025/html/Wang_LA-MOTR_End-to-End_Multi-Object_Tracking_by_Learnable_Association_ICCV2025_paper.html",
            "code_url": "https://github.com/PenK1nG/LA-MOTR",
            "observed_head": None,
            "observed_head_date": None,
            "license": "Not verified: the named GitHub URL returned 404 during the 2026-09-01 audit",
            "mechanism": "Spatial-guided learnable association and tracklet updates in an end-to-end MOT model.",
            "transferable": "Only the general idea of spatially conditioned set association is relevant for a future, separately frozen probe.",
            "not_transferable": "No code was copied; replacing the frozen solver or candidate definition would confound N68.",
        },
        {
            "name": "InteractTrack / IMAT",
            "year_status": "CVPR 2026",
            "paper_url": "https://arxiv.org/abs/2604.01974",
            "code_url": "https://github.com/NorahGreen/InteractTrack",
            "observed_head": "5f149d4001a84c8b83129192057bf6dd820f71b3",
            "observed_head_date": "2026-06-16T18:37:04Z",
            "license": "No SPDX license reported by GitHub repository metadata",
            "mechanism": "Interactive tracking benchmark and memory-augmented adaptation driven by human feedback, with switch/interaction annotations.",
            "transferable": "Supports the need for provenance-complete event tapes and explicit feedback-to-memory evaluation.",
            "not_transferable": "N68 has no real tape; its GT-derived events cannot be relabeled as human interactions, and no external dataset was imported.",
        },
        {
            "name": "REMIND",
            "year_status": "arXiv 2026",
            "paper_url": "https://arxiv.org/abs/2607.09267",
            "code_url": "https://github.com/cvar-vision-dl/remind-reid-tracker",
            "observed_head": "f88ea1d5d81da0a8ed28b206df6d4dab48327342",
            "observed_head_date": "2026-08-12T09:28:13Z",
            "license": "MIT",
            "mechanism": "Per-object multi-prototype dual work/stable banks, part/background descriptors, neighbor context, robust update gating and explicit ambiguous/provisional tracks.",
            "transferable": "Multi-prototype and conservative update ideas are useful future ablations for memory contamination and long-horizon drift.",
            "not_transferable": "N68 tests a single target-conditioned local sidecar; adding all memory banks now would make attribution impossible and still cannot create human evidence.",
        },
        {
            "name": "Rethinking Memory Design in SAM-Based Visual Object Tracking",
            "year_status": "arXiv preprint 2025",
            "paper_url": "https://arxiv.org/abs/2512.22624",
            "code_url": "https://github.com/HamadYA/SAM3_Tracking_Zoo",
            "observed_head": "d37e4a975e480e6471b68b64eb6fd98c65a6b989",
            "observed_head_date": "2026-06-30T10:35:16Z",
            "license": "No SPDX license reported by GitHub repository metadata; repository NOTICE must be followed",
            "mechanism": "Controlled SAM3 adaptations of SAM2-era memory policies, including short-term appearance and long-term distractor-resolving memory.",
            "transferable": "Provides a principled future comparison for memory policy and distractor handling after the current association bottleneck is resolved.",
            "not_transferable": "N68 cannot change SAM3 checkpoint, candidate generation or memory definition; no code was imported.",
        },
        {
            "name": "DAM4SAM",
            "year_status": "CVPR 2025; IJCV 2026 extension",
            "paper_url": "https://arxiv.org/abs/2411.17576",
            "code_url": "https://github.com/jovanavidenovic/DAM4SAM",
            "observed_head": "9c954504b39ebca4c412f207be0787c26bfac85a",
            "observed_head_date": "2026-04-07T07:39:27Z",
            "license": "No SPDX license reported by GitHub repository metadata",
            "mechanism": "Distractor-aware memory and memory-management strategy for SAM2-based visual object tracking.",
            "transferable": "The explicit distractor-risk framing supports N68's hard-negative and untouched-ID audits.",
            "not_transferable": "It is a SAM2 VOT tracker and would change the frozen SAM3 candidate/memory stack; no code was copied.",
        },
        {
            "name": "TCEI",
            "year_status": "CVPR 2026",
            "paper_url": "https://openaccess.thecvf.com/content/CVPR2026/html/Guo_Dual-level_Adaptation_for_Multi-Object_Tracking_Building_Test-Time_Calibration_from_Experience_CVPR_2026_paper.html",
            "code_url": "https://github.com/1941Zpf/TCEI",
            "observed_head": "145d1b8431398156f8d9f854430e306fdee39eaa",
            "observed_head_date": "2026-03-30T02:04:24Z",
            "license": "No SPDX license reported by GitHub repository metadata",
            "mechanism": "Separates transient intuitive memory from accumulated experiential calibration for test-time MOT adaptation.",
            "transferable": "Suggests a future separation between short-lived target evidence and conservative long-term identity memory.",
            "not_transferable": "N68 is not authorized for calibration or test-time adaptation because the simulated future-effect and real-human gates fail.",
        },
    ]
    atomic_json(
        SOURCES,
        {
            "schema": "N68_RESEARCH_SOURCES_V1",
            "audited_at": "2026-09-01",
            "search_scope": "Official paper pages/OpenReview/arXiv and named official GitHub repositories; 2025-2026 prioritized as requested.",
            "sources": method_sources,
        },
    )

    frozen_paths = [
        OUT / "stage_00_readonly_audit.json",
        OUT / "stage_01_status.json",
        OUT / "stage_02_protocol.json",
        OUT / "stage_03_protocol.json",
        OUT / "training/n68_local_association_dataset_manifest.json",
        OUT / "training/n68_identity_local_head_training_manifest.json",
        OUT / "replay/runtime_status.json",
        OUT / "replay/stage03_runtime_status.json",
        OUT / "replay/paired_replay_results.json",
        OUT / "replay/stage03_paired_replay_results.json",
        OUT / "n68_isolation_regression.json",
    ]
    failure_artifacts = [
        OUT / "attempts/stage_00_initial_audit_failure.json",
        OUT / "attempts/stage_01_failure.json",
        OUT / "attempts/stage_01_failure_attempt1.json",
        OUT / "attempts/stage_01_classification_repair.json",
        OUT / "attempts/stage_02_local_association_failure_attempt1.json",
        OUT / "attempts/stage_02_local_association_failure_attempt2.json",
        OUT / "attempts/stage_02_posthoc_scoring_attempt1_preserved.json",
        OUT / "attempts/stage_03_margin_failure_attempt1.json",
        OUT / "attempts/isolation_import_failure_attempt1.json",
        OUT / "attempts/final_gate_validation_failure_attempt1.json",
    ]
    stage03_gate = stage03.get("gate_by_mode", {})
    learned = stage02["methods"]["LEARNED_LOCAL_ASSOCIATION"]
    margin = stage03["methods"]["LEARNED_MARGIN_AWARE_COLUMN"]
    gate = {
        "schema": "N68_FINAL_GATE_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "N68",
        "status": "BLOCKED",
        "status_detail": "ENGINEERING_COMPLETED_SIMULATED_EFFECT_GATE_FAILED_REAL_HUMAN_EVIDENCE_MISSING",
        "production_authorized": False,
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
        "runtime_future_gt_used": False,
        "decision": {
            "stage_02_trained": True,
            "stage_02_strict_gate": False,
            "stage_03_alternative_executed": True,
            "stage_03_strict_gate": False,
            "tact_status": "NOT_RUN_PREREQUISITE_NO_TRIMMING_GATE_FAILED",
            "calibration_status": "NOT_AUTHORIZED",
            "selector_status": "NOT_AUTHORIZED",
            "decoder_lora_status": "NOT_AUTHORIZED",
            "research_claim": "No evidence that either learned local association or margin-aware projection yields a robust future identity benefit under the frozen simulated protocol.",
        },
        "strict_checks": {
            "stage_02_runtime_complete": stage02_runtime.get("status") == "PASS_RUNTIME_REPLAY",
            "stage_03_runtime_complete": stage03_runtime.get("status") == "PASS_RUNTIME_REPLAY",
            "all_events_24": stage03_runtime.get("metrics", {}).get("event_count") == 24,
            "all_frames_12000": stage03_runtime.get("metrics", {}).get("frames") == 12000,
            "all_upstream_variants_5": stage03.get("variant_count") == 5,
            "runtime_gt_free": stage03.get("runtime_future_gt_used") is False,
            "candidate_recall_learned": learned.get("target_candidate_recall"),
            "learned_correct_gt_incorrect": learned.get("correct_changes", 0) > learned.get("incorrect_changes", 0),
            "learned_h20_lower_ci_gt_zero": learned["horizons"]["20"]["sequence_cluster_bootstrap"]["ci95"][0] > 0,
            "learned_h50_lower_ci_gt_zero": learned["horizons"]["50"]["sequence_cluster_bootstrap"]["ci95"][0] > 0,
            "learned_h100_lower_ci_gt_zero": learned["horizons"]["100"]["sequence_cluster_bootstrap"]["ci95"][0] > 0,
            "learned_untouched_safe": learned.get("untouched_regression_frame_rate", 0.0) == 0.0,
            "margin_correct_gt_incorrect": margin.get("correct_changes", 0) > margin.get("incorrect_changes", 0),
            "margin_h20_lower_ci_gt_zero": margin["horizons"]["20"]["sequence_cluster_bootstrap"]["ci95"][0] > 0,
            "margin_untouched_safe": margin.get("untouched_regression_frame_rate", 0.0) == 0.0,
            "real_human_tape_available": False,
            "real_full_loop_available": False,
            "production_hash_gate": isolation["interpretation"]["production_hash_gate"],
            "sam3_unit_gate": isolation["interpretation"]["sam3_unit_gate"],
            "mot_unit_gate": isolation["interpretation"]["mot_unit_gate"],
        },
        "stage_statuses": {name: data.get("status") for name, data in stages.items()},
        "stage_02_learned_summary": method_summary(learned),
        "stage_03_margin_summary": method_summary(margin),
        "training": {
            "manifest": str(OUT / "training/n68_identity_local_head_training_manifest.json"),
            "checkpoint": training.get("checkpoint"),
            "checkpoint_sha256": training.get("checkpoint_sha256"),
            "dataset": dataset.get("dataset"),
            "dataset_sha256": dataset.get("dataset_sha256"),
            "parameter_count": training.get("parameter_count"),
            "seed": training.get("seed"),
            "device": training.get("device"),
            "cuda_visible_devices": training.get("cuda_visible_devices"),
            "train_sequence_count": training.get("train_sequence_count"),
            "validation_sequence_count": training.get("validation_sequence_count"),
            "holdout_sequence_count": training.get("holdout_sequence_count"),
            "holdout_used_for_selection": False,
            "holdout_evaluated_once_after_selection": training.get("holdout_evaluated_once_after_selection"),
        },
        "isolation": isolation["interpretation"],
        "failure_artifacts": [
            {"path": str(path), "exists": path.is_file(), "sha256": sha256(path)}
            for path in failure_artifacts
        ],
        "artifact_hashes": [
            {"path": str(path), "exists": path.is_file(), "sha256": sha256(path)}
            for path in frozen_paths
        ],
        "method_sources": str(SOURCES),
        "minimal_next_step": "Obtain provenance-complete real human event tape with direct public_id and run real SAM3 full-loop; before any new architecture, audit native/local/global-to-public mapping and target-ID scope on those events.",
        "iclr_2027_calendar": {
            "audit_date": "2026-09-01",
            "abstract_deadline_aoe": "2026-09-18",
            "full_paper_deadline_aoe": "2026-09-25",
            "days_to_abstract_approx": 17,
            "days_to_full_paper_approx": 24,
        },
    }
    if GATE.exists() and not args.rewrite_existing:
        raise RuntimeError(f"refusing to overwrite existing final gate: {GATE}")
    atomic_json(GATE, gate)
    print(json.dumps({"status": gate["status"], "gate": str(GATE), "sources": str(SOURCES)}, sort_keys=True))


if __name__ == "__main__":
    main()
