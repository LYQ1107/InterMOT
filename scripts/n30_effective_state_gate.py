#!/usr/bin/env python3
"""Compute the N30-A/B effective-state gate from frozen formal artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _comparison(summary: dict[str, Any], name: str) -> dict[str, Any]:
    value = summary.get("comparisons", {}).get(name, {}).get("future_delivered_box_iou")
    if not isinstance(value, dict):
        raise ValueError(f"missing future IoU comparison: {name}")
    return value


def build_gate(ablation_path: Path, summary_path: Path, manifest_path: Path) -> dict[str, Any]:
    ablation = _read(ablation_path)
    summary = _read(summary_path)
    if ablation.get("status") != "PASS" or summary.get("status") != "PASS":
        raise ValueError("N30-B formal artifacts must both be PASS")
    if int(ablation.get("case_count_pass", 0)) < 10 or int(summary.get("case_count", 0)) < 10:
        raise ValueError("N30-B effective-state gate requires at least ten passing cases")
    if ablation.get("manifest_sha256") != summary.get("manifest_sha256"):
        raise ValueError("N30-B result/summary manifest hashes disagree")

    m1 = _comparison(summary, "M1_minus_M0")
    m2 = _comparison(summary, "M2_minus_M0")
    synergy = _comparison(summary, "M3_minus_max_M1_M2")
    m1_ci = m1.get("sequence_cluster_bootstrap_ci95")
    m2_ci = m2.get("sequence_cluster_bootstrap_ci95")
    synergy_ci = synergy.get("episode_bootstrap_ci95")
    if not all(isinstance(value, list) and len(value) == 2 for value in (m1_ci, m2_ci, synergy_ci)):
        raise ValueError("N30-B gate requires all three future-IoU confidence intervals")

    m1_stable = float(m1["mean"]) > 0.0 and float(m1_ci[0]) > 0.0
    m2_stable = float(m2["mean"]) > 0.0 and float(m2_ci[0]) > 0.0
    synergy_stable = float(synergy["mean"]) > 0.0 and float(synergy_ci[0]) > 0.0
    if m1_stable and not m2_stable and not synergy_stable:
        conclusion = "OFFICIAL_TRACKER_STATE_DOMINANT"
        rationale = "M1 has a positive sequence-cluster lower bound; M2 and M3 synergy do not."
    elif m2_stable and not m1_stable and not synergy_stable:
        conclusion = "B10_IDENTITY_MEMORY_DOMINANT"
        rationale = "M2 has a positive sequence-cluster lower bound; M1 and M3 synergy do not."
    elif synergy_stable:
        conclusion = "JOINT_SYNERGY"
        rationale = "M3 exceeds the stronger single write with a positive future-IoU lower bound."
    else:
        conclusion = "NO_STABLE_WRITE_BENEFIT_AFTER_DECOMPOSITION"
        rationale = "No permitted single or joint write comparison has a positive lower bound."

    return {
        "protocol": "N30-A/B-EFFECTIVE-STATE-GATE",
        "status": "PASS",
        "conclusion": conclusion,
        "writer_authorized": conclusion in {
            "OFFICIAL_TRACKER_STATE_DOMINANT",
            "B10_IDENTITY_MEMORY_DOMINANT",
            "JOINT_SYNERGY",
        },
        "rationale": rationale,
        "ablation": str(ablation_path),
        "summary": str(summary_path),
        "manifest": str(manifest_path),
        "manifest_sha256": str(ablation["manifest_sha256"]),
        "case_count": int(summary["case_count"]),
        "future_metric": "future_delivered_box_iou",
        "comparisons": {
            "M1_official_spatial_write_only_minus_M0": m1,
            "M2_b10_identity_write_only_minus_M0": m2,
            "M3_joint_minus_max_single_write": synergy,
        },
        "decision_rule": {
            "stable_positive": "mean > 0 and lower endpoint of the declared bootstrap interval > 0",
            "interval_for_M1_M2": "sequence_cluster_bootstrap_ci95",
            "interval_for_M3_synergy": "episode_bootstrap_ci95",
            "exactly_one_conclusion": True,
        },
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_selection": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", type=Path, default=ROOT / "outputs/n30/multi_identity_write_ablation.json")
    parser.add_argument("--summary", type=Path, default=ROOT / "outputs/n30/multi_identity_write_summary.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/n30/multi_identity_case_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/n30/effective_state_gate.json")
    args = parser.parse_args()
    gate = build_gate(args.ablation, args.summary, args.manifest)
    _write(args.output, gate)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
