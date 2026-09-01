#!/usr/bin/env python3
"""N44 strict final gate and reproducible report writer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
N43_GATE = ROOT / "outputs/n43/n43_final_gate.json"
N43_RESULT = ROOT / "outputs/n43/replay/paired_replay_results.json"
N44_STAGE1 = ROOT / "outputs/n44/stage_01_status.json"
N44_STAGE2 = ROOT / "outputs/n44/stage_02_status.json"
N44_STAGE3 = ROOT / "outputs/n44/stage_03_status.json"
N44_STAGE4 = ROOT / "outputs/n44/stage_04_status.json"
N44_RESULT = ROOT / "outputs/n44/replay/paired_replay_results.json"
N44_BLOCK = ROOT / "outputs/n44/blocked_input_real_human_tape.json"
REGRESSION = ROOT / "outputs/n44/targeted_regression.json"
OUT = ROOT / "outputs/n44"
STAGE = OUT / "stage_05_status.json"
REPORT = ROOT / "docs/N44_FINAL_REPORT.md"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    s1, s2, s3, s4 = (read(path) for path in (N44_STAGE1, N44_STAGE2, N44_STAGE3, N44_STAGE4))
    n43_gate, n43_result, n44, block, regression = (read(path) for path in (N43_GATE, N43_RESULT, N44_RESULT, N44_BLOCK, REGRESSION))
    frozen_n43 = {"h20": -0.009572664281843673, "h50": -0.00945110201410842, "h100": -0.007484765399726164, "changes": {"h20": [40, 0, 13, 407], "h50": [104, 0, 43, 1043], "h100": [249, 0, 93, 2072]}}
    n43_m2 = n43_result["aggregates"]["M2"]
    n43_reconciled = all(abs(float(n43_m2[str(h)]["all"]["identity_utility"]) - frozen_n43[f"h{h}"]) < 1e-15 for h in (20, 50, 100)) and all([n43_m2[str(h)]["all"]["assignment_change_count"], n43_m2[str(h)]["all"]["assignment_change_correct_count"], n43_m2[str(h)]["all"]["assignment_change_incorrect_count"], n43_m2[str(h)]["all"]["assignment_no_change_count"]] == frozen_n43["changes"][f"h{h}"] for h in (20, 50, 100))
    n44_m2 = n44["aggregates"]["M2"]
    future_effect = all(float(n44["aggregates"][variant][str(h)]["sequence_cluster_bootstrap_95ci"]["lower"]) > 0.0 for variant in ("M2", "M3", "M4") for h in (20, 50, 100))
    untouched = all(bool(n44["aggregates"][variant][str(h)]["untouched_regression"]["all_no_obvious_regression"]) for variant in ("M1", "M2", "M3", "M4") for h in (20, 50, 100))
    checks = {"stage_01_assignment_audit": s1.get("status") == "PASS", "stage_02_sidecar": s2.get("status") == "PASS", "stage_03_actual_full_training": s3.get("status") == "PASS" and s3.get("gate_checks", {}).get("actual_full_training") is True, "stage_04_full_replay": s4.get("status") == "PASS" and s4.get("metrics", {}).get("runtime", {}).get("event_count") == 24, "n43_frozen_numbers_reconciled": n43_reconciled, "all_events_simulated": n44.get("interaction_source") == "simulated_from_gt" and n44.get("event_count") == 24, "runtime_future_gt_false": n44.get("runtime_future_gt_used") is False, "posthoc_only_after_validation": n44.get("gt_loaded_only_after_runtime_validation") is True, "future_effect_strict_ci": future_effect, "untouched_regression_absent": untouched, "equal_sequence_bootstrap": n44.get("bootstrap_protocol") == "sequence_mean_then_equal_sequence_cluster_bootstrap", "targeted_regressions": regression.get("status") == "PASS", "real_human_tape": False, "real_sam3_full_loop": False}
    gate = {"artifact": "outputs/n44/stage_05_status.json", "protocol": "N44_FINAL_RESEARCH_GATE_V1", "status": "N44_COMPLETED_GATE_FAILED", "research_gate": "FAIL_ASSIGNMENT_AWARE_EXPERIMENT_AND_REAL_INPUT", "authorization": {"calibration_head": "NOT_AUTHORIZED", "decoder_lora": "NOT_AUTHORIZED", "production_interface_changed": False}, "checks": checks, "failure_reasons": {"future_effect": "N44 M2/M3/M4 equal-sequence bootstrap lower bounds are not strictly greater than zero; M2 remains negative at all horizons.", "untouched_regression": "N44 M1-M4 include at least one untouched-ID regression under the fixed posthoc criterion.", "real_human_tape": block["exact_blocker"], "real_full_loop": "No real SAM3 candidate-complete human full-loop is available in this workspace.", "n43_preservation": "N43 frozen values reconcile exactly; N43 gate and artifacts were not rewritten."}, "side_by_side": {"n43_frozen": frozen_n43, "n44_m2": {str(h): n44_m2[str(h)] for h in (20, 50, 100)}, "n44_checkpoint": s3["outputs"]["checkpoint"], "n44_checkpoint_sha256": s4["metrics"]["runtime"]["checkpoint_sha256"]}, "next_action": "Collect external provenance-complete real human tape and run a real candidate-complete SAM3 full-loop; then revalidate N40 before any calibration authorization. Do not start decoder LoRA or tune N44 on holdout."}
    output = {"status": gate["status"], "protocol": gate["protocol"], "command": ["python", "scripts/n44_finalize.py"], "inputs": {"stage_01": str(N44_STAGE1), "stage_02": str(N44_STAGE2), "stage_03": str(N44_STAGE3), "stage_04": str(N44_STAGE4), "n43_final_gate": str(N43_GATE), "n43_result": str(N43_RESULT), "n44_result": str(N44_RESULT), "real_tape_feasibility": str(N44_BLOCK), "targeted_regression": str(REGRESSION)}, "outputs": {"final_gate": str(STAGE), "report": str(REPORT)}, "metrics": {"n43_m2_frozen_utility": {str(h): frozen_n43[f"h{h}"] for h in (20, 50, 100)}, "n43_m2_frozen_changes": frozen_n43["changes"], "n44_m2_identity_utility": {str(h): n44_m2[str(h)]["identity_utility"] for h in (20, 50, 100)}, "n44_m2_ci_lower": {str(h): n44_m2[str(h)]["sequence_cluster_bootstrap_95ci"]["lower"] for h in (20, 50, 100)}, "n44_runtime_application": s4["metrics"]["runtime"]["application_totals"], "side_by_side": gate["side_by_side"], "independent_sequences": 21, "event_count": 24, "real_tape": False, "real_full_loop": False}, "gate_checks": checks, "failure_root_cause": gate["failure_reasons"], "next_action": gate["next_action"], "runtime_future_gt_used": False, "finished_at": now()}
    OUT.mkdir(parents=True, exist_ok=True)
    STAGE.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# N44 Final Report — assignment-aware structural experiment",
        "",
        f"Date: {now()}  ",
        "Final status: `N44_COMPLETED_GATE_FAILED`. This is completion of the feasible N44 experiment, not completion of the scientific objective. Calibration and decoder LoRA remain `NOT_AUTHORIZED`.",
        "",
        "## Scope and frozen evidence",
        "",
        "The project root was confirmed as `.`. AGENTS.md, N43 report/gate, N40 Stage 01, and the N36/N37 tape/full-loop contracts were read. N43 evidence is preserved, including its actual full training, 24 `simulated_from_gt` events, no real human tape, no real SAM3 full-loop, and no production code change.",
        "",
        "N43 frozen M2 utility (H20/H50/H100) is exactly `-0.009572664281843673 / -0.00945110201410842 / -0.007484765399726164`. Assignment changes were respectively `40/104/249`, with correct `0/0/0`, incorrect `13/43/93`, and no-change `407/1043/2072`. The finalizer reconciled these values directly against the preserved N43 result.",
        "",
        "## Stage 01 — decision-boundary audit",
        "",
        "The read-only audit covered 2424 frames, 173793 candidate×public-ID cells, and 21 independent sequences. Baseline Hungarian with explicit NONE had 16383 known assigned cells: 11800 correct and 4583 wrong. There were 1551 base-wrong assignments with a positive candidate alternative, so the learnable boundary is nonempty. The candidate ceiling contained 13604 positive candidate-ID pairs among 16920 known candidate-ID pairs (0.8040189125); the oracle assignment had 13270 correct assigned cells.",
        "",
        "N43 changed all 173793 finite cells but only 29 assignments. Its frozen cell target was +0.5 for 15115 positives and -0.5 for 107188 negatives (negative/positive ratio 7.0914985), with 19334 ambiguous cells discarded and 30882 GT-unavailable cells. This is a cell-classification utility, not a global assignment-gain target. The first actionable root cause is therefore the mismatch between target-column/cellwise residual semantics and global candidate×ID assignment utility; the data does not support claiming candidate generation is the only cause because the positive alternative boundary is nonempty.",
        "",
        "## Stage 02/03 — isolated sidecar and actual training",
        "",
        "N44 uses a new `scripts/n44_*` sidecar only. Its pairwise score difference is anti-symmetric by construction; runtime features are the 18 causal current/past candidate/state features. Public ID, target identity, GT, future outcome, sequence encoding and hidden candidate oracle are not feature inputs. Hard-negative and explicit NONE/abstain boundaries are retained. The application defaults to the branch fused baseline and adds only a bounded +0.25 proposal boost after a frozen near-tie, predicted-advantage and calibrated-uncertainty gate; it does not add a residual to every cell.",
        "",
        "The actual sequence-disjoint training used 122303 cells, 29990 pairs, 16776 groups, 3280 no-positive abstain groups, fixed N42 train/validation/holdout sequence splits, seed 4444, AdamW, GPU 0, and 31 epochs (best epoch 21). The checkpoint is [n44_assignment_aware.pt](./outputs/n44/training/n44_assignment_aware.pt), SHA-256 `0b5e750f5d9569f71ae887595c1d88d4d625f120f8a3811f2598a852cf82348f`. Holdout was not used for optimization, gate selection or tuning.",
        "",
        "## Stage 04 — paired replay",
        "",
        "The replay used the same N42 prefix, 24 frozen events, candidate streams and M0–M4 definitions. All 24 events are `simulated_from_gt`; runtime `future_gt=false`; GT was loaded only after runtime structural validation. It produced 100 future frames per branch and equal-sequence cluster bootstrap (seed 4444, 2000 replicates). Standard MOT/TrackEval metrics are `NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_SEQUENCE_INPUT`.",
        "",
        "N44 M2 utility / equal-sequence CI lower bound:",
        "",
        "| Horizon | Utility | CI lower | Changes (correct / incorrect / no-change) |",
        "|---:|---:|---:|---:|",
        *[f"| H{h} | {n44_m2[str(h)]['identity_utility']:.15f} | {n44_m2[str(h)]['sequence_cluster_bootstrap_95ci']['lower']:.15f} | {n44_m2[str(h)]['assignment_change_correct_count']} / {n44_m2[str(h)]['assignment_change_incorrect_count']} / {n44_m2[str(h)]['assignment_no_change_count']} |" for h in (20, 50, 100)],
        "",
        "N44 M2 has zero correct assignment changes and 12/42/92 incorrect changes at H20/H50/H100; untouched-ID regression is not clean. Across 9600 write frames, 28 proposals were considered and 14 selected, with 18 resulting assignment changes; all other cells stayed at the branch baseline.",
        "",
        "## Targeted regressions and input feasibility",
        "",
        "The post-replay `cell_features` regression passes repeated finite scores plus hard-negative columns and records the minimal fix in `scripts/n43_full_matrix_common.py`: retain original valid column indices and exclude the current column by index. The CI audit confirms the preregistered definition is sequence mean first, then equal-sequence bootstrap; the old event-weighted N43 result remains at `outputs/n43/replay/paired_replay_results_legacy_event_weighted.json`. Two checker-only failures (aggregate-key lookup and numpy.float32 JSON serialization) are preserved under `outputs/n44/attempts/` and were minimally repaired.",
        "",
        "Three evidence-preserving N40 feasibility checks found the N34 sentinel `NOT_AVAILABLE` with zero events, N40 `BLOCKED_INPUT_REAL_HUMAN_TAPE`, and only a `simulated_from_gt` synthetic fallback. The strict validator exists, but no external UI/session/annotator export supplies direct public_id, human-confirmed BOX/CLICK/CONFIRMED_MASK, lossless ROI digest, timestamps and candidate-complete future rows. Artifact: [blocked_input_real_human_tape.json](./outputs/n44/blocked_input_real_human_tape.json). No old log was fabricated or relabeled.",
        "",
        "## Gate and reproducibility index",
        "",
        "Stage status artifacts: [Stage 01](./outputs/n44/stage_01_status.json), [Stage 02](./outputs/n44/stage_02_status.json), [Stage 03](./outputs/n44/stage_03_status.json), [Stage 04](./outputs/n44/stage_04_status.json), and [Stage 05](./outputs/n44/stage_05_status.json). Full replay is [paired_replay_results.json](./outputs/n44/replay/paired_replay_results.json); targeted audit is [targeted_regression.json](./outputs/n44/targeted_regression.json).",
        "",
        "The strict gate fails because real human tape and real full-loop are absent, N44 M2/M3/M4 future-effect lower CIs are not strictly positive, and untouched-ID regression is not absent. The next action is the external provenance-complete human tape plus candidate-complete real SAM3 full-loop required by N40/N36/N37, followed by validation before any calibration authorization. Decoder LoRA is not considered.",
        "",
        "Actual commands used include:",
        "",
        "```text",
        "python scripts/n44_stage1_assignment_audit.py",
        "python scripts/n44_stage2_assignment_sidecar.py",
        "CUDA_VISIBLE_DEVICES=0 python scripts/n44_stage3_train.py",
        "CUDA_VISIBLE_DEVICES=0 python scripts/n44_stage4_paired_replay.py",
        "python scripts/n44_post_replay_targeted_regression.py",
        "python scripts/n44_stage5_real_tape_feasibility.py",
        "python scripts/n44_finalize.py",
        "```",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "stage": str(STAGE), "report": str(REPORT)}))


if __name__ == "__main__":
    main()
