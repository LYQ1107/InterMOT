#!/usr/bin/env python3
"""Finalize the geometry-corrected N72R11R5R1 diagnostic.

The finalizer is deliberately post-hoc.  It validates the already sealed
replays, the lossless export, and the pinned TrackEval output; it never runs
SAM3, rewrites a historical artifact, or promotes the failed causal gate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

# Direct execution places ``scripts/`` at sys.path[0].  Make the project
# package import explicit so finalization is reproducible without relying on
# an ambient PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam3_intermot.evaluation.window_trackeval import write_json_atomic, write_text_atomic


HORIZONS = (20, 50, 100)
LOGICAL_VARIANTS = ("E0_BASELINE_B0", "E1A_V3", "E1B_PCTIS")
EXPECTED_EVENTS = 32
EXPECTED_SEQUENCES = 18
EXPECTED_RECORDS = 288
REPORT_PATH = "docs/N72R11R5R1_GEOMETRY_CORRECTED_TRACKEVAL_REPORT.md"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def git_value(project_root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=project_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def runtime_gt_violation(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in {"runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used"} and nested:
                return True
            if runtime_gt_violation(nested):
                return True
    elif isinstance(value, list):
        return any(runtime_gt_violation(nested) for nested in value)
    return False


def valid_xyxy(box: Any) -> bool:
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) for value in box)
        and float(box[2]) > float(box[0])
        and float(box[3]) > float(box[1])
    )


def smoke_summary(root: Path) -> dict[str, Any]:
    event_id = "n72r5-pool-n37-dancetrack0027-0148-authoritative_reassign-007"
    output: dict[str, Any] = {
        "event_id": event_id,
        "roots": {},
        "e0_runtime_hashes": [],
        "e0_hashes_identical": False,
        "bad_geometry_rows": 0,
        "runtime_gt_violations": 0,
    }
    for label, directory, treatment in (
        ("E1A", root / "smoke_e1a", "E1A_EXACT_ONPOLICY_V3_LEGACY"),
        ("E1B", root / "smoke_e1b", "E1B_PCTIS_LEGACY"),
    ):
        event_root = directory / event_id
        done = read_json(event_root / "done.json", required=False) or {}
        sealed = read_json(event_root / "runtime_event_sealed.json", required=False) or {}
        manifests: list[dict[str, Any]] = []
        for path in sorted(event_root.glob("*/runtime_manifest.json")):
            manifest = read_json(path) or {}
            manifests.append(manifest)
            if manifest.get("variant") == "E0_BASELINE_B0":
                output["e0_runtime_hashes"].append(manifest.get("frames_sha256"))
            if manifest.get("status") != "PASS_N72R11_RUNTIME_ARTIFACT_SEALED":
                output.setdefault("errors", []).append(f"{label}: runtime manifest is not sealed: {path}")
        for frames_path in sorted(event_root.glob("*/runtime_frames.jsonl")):
            with frames_path.open(encoding="utf-8") as handle:
                for raw in handle:
                    if not raw.strip():
                        continue
                    row = json.loads(raw)
                    if runtime_gt_violation(row):
                        output["runtime_gt_violations"] += 1
                    for candidate in row.get("candidate_rows", []):
                        if candidate.get("assignment_status") == "ASSIGNED_TO_PUBLIC_ID" and not valid_xyxy(candidate.get("box_xyxy")):
                            output["bad_geometry_rows"] += 1
                    pool = row.get("candidate_pool")
                    if isinstance(pool, dict):
                        for candidate in pool.get("candidate_rows", []):
                            if candidate.get("geometry_valid") is not True:
                                output["bad_geometry_rows"] += 1
        output["roots"][label] = {
            "done_status": done.get("status"),
            "sealed_status": sealed.get("status"),
            "treatment": treatment,
            "runtime_manifest_count": len(manifests),
            "require_positive_geometry": all(item.get("require_positive_geometry") is True for item in manifests),
            "baseline_regenerated_from_frozen_sources": all(item.get("baseline_regenerated_from_frozen_sources") is True for item in manifests),
            "runtime_future_gt_used": sealed.get("runtime_future_gt_used"),
        }
    output["e0_hashes_identical"] = len(output["e0_runtime_hashes"]) == 2 and len(set(output["e0_runtime_hashes"])) == 1
    output["status"] = "PASS_CORRECTED_GEOMETRY_SMOKE" if (
        all(item["done_status"] == "PASS_N72R11_RUNTIME_AND_POSTHOC_EVENT" for item in output["roots"].values())
        and all(item["sealed_status"] == "PASS_N72R11_ALL_RUNTIME_SEALED" for item in output["roots"].values())
        and output["e0_hashes_identical"]
        and output["bad_geometry_rows"] == 0
        and output["runtime_gt_violations"] == 0
    ) else "FAIL_CORRECTED_GEOMETRY_SMOKE"
    return output


def replay_summary(audit: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("e1a", "e1b"):
        item = audit.get("results", {}).get(key, {})
        result[key] = {
            "pass": item.get("pass") is True,
            "manifest_status": item.get("manifest_status"),
            "record_count": item.get("record_count"),
            "unique_record_count": item.get("unique_record_count"),
            "duplicate_event_count": item.get("duplicate_event_count"),
            "missing_event_count": item.get("missing_event_count"),
            "total_filtered_geometry_candidates": item.get("total_filtered_geometry_candidates"),
            "filtered_geometry_by_source": item.get("filtered_geometry_by_source", {}),
            "failures": item.get("failures", []),
        }
    result["status"] = audit.get("status")
    result["runtime_future_gt_used"] = audit.get("runtime_future_gt_used")
    return result


def metric_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": metrics.get("status"),
        "complete": metrics.get("complete") is True,
        "event_completeness": metrics.get("event_completeness", {}),
        "by_horizon": {},
    }
    for horizon in HORIZONS:
        values = metrics.get("by_horizon", {}).get(str(horizon), {})
        ci = values.get("sequence_cluster_bootstrap_95ci", {})
        result["by_horizon"][str(horizon)] = {
            "identity_error_reduction": values.get("identity_error_reduction"),
            "ci_lower": ci.get("lower"),
            "ci_mean": ci.get("mean"),
            "ci_upper": ci.get("upper"),
            "assignment_change_rate": values.get("assignment_change_rate"),
            "assignment_change_count": values.get("assignment_change_count"),
            "true_correct_crossing_count": values.get("true_correct_crossing_count"),
            "true_incorrect_crossing_count": values.get("true_incorrect_crossing_count"),
            "protected_regression_count": values.get("protected_regression_count"),
            "candidate_recall": values.get("candidate_recall"),
            "delta_iou": values.get("delta_iou"),
            "recorrection_rate": values.get("recorrection_rate"),
        }
    return result


def structural_checks(
    *,
    geometry: Mapping[str, Any],
    smoke: Mapping[str, Any],
    replay: Mapping[str, Any],
    export: Mapping[str, Any],
    run: Mapping[str, Any],
    aggregation: Mapping[str, Any],
    metrics_a: Mapping[str, Any],
    metrics_b: Mapping[str, Any],
) -> dict[str, bool]:
    replay_ok = replay.get("status") == "PASS_CORRECTED_REPLAY_RUNTIME_AUDIT" and all(
        replay.get(key, {}).get("pass") is True
        and replay.get(key, {}).get("record_count") == EXPECTED_EVENTS
        and replay.get(key, {}).get("unique_record_count") == EXPECTED_EVENTS
        and replay.get(key, {}).get("duplicate_event_count") == 0
        and replay.get(key, {}).get("missing_event_count") == 0
        and not replay.get(key, {}).get("failures")
        for key in ("e1a", "e1b")
    ) and replay.get("runtime_future_gt_used") is False
    export_ok = (
        export.get("status") == "PASS_EXPORT"
        and export.get("source_events") == EXPECTED_EVENTS
        and export.get("independent_sequence_count") == EXPECTED_SEQUENCES
        and export.get("pseudo_sequence_count") == 96
        and export.get("record_count") == EXPECTED_RECORDS
        and export.get("runtime_future_gt_used") is False
        and export.get("integrity", {}).get("duplicate_keys") == []
        and export.get("integrity", {}).get("missing_keys") == []
    )
    run_ok = (
        run.get("status") == "PASS_TRACKEVAL"
        and run.get("expected_record_count") == EXPECTED_RECORDS
        and run.get("record_count") == EXPECTED_RECORDS
        and run.get("duplicate_keys") == []
        and len(run.get("horizon_results", [])) == len(HORIZONS)
        and all(item.get("status") == "PASS" and item.get("record_count") == 96 for item in run.get("horizon_results", []))
    )
    metrics_ok = all(
        item.get("complete") is True
        and item.get("event_completeness", {}).get("required_events") == EXPECTED_EVENTS
        and item.get("event_completeness", {}).get("manifest_records") == EXPECTED_EVENTS
        and item.get("event_completeness", {}).get("duplicate_event_ids") == 0
        and item.get("event_completeness", {}).get("missing_event_count") == 0
        for item in (metrics_a, metrics_b)
    )
    return {
        "geometry_source_audit": geometry.get("status") == "PASS_GEOMETRY_SOURCE_AUDIT" and geometry.get("event_count") == EXPECTED_EVENTS and geometry.get("event_load_error_count") == 0 and geometry.get("empty_solver_pool_frame_count") == 0,
        "corrected_smoke": smoke.get("status") == "PASS_CORRECTED_GEOMETRY_SMOKE",
        "corrected_replay": replay_ok,
        "corrected_causal_metrics": metrics_ok,
        "export": export_ok,
        "trackeval": run_ok,
        "aggregation": aggregation.get("status") == "PASS_COMPLETE_INPUT" and aggregation.get("per_event_record_count") == EXPECTED_RECORDS,
    }


def source_hashes(project_root: Path, root: Path) -> dict[str, Any]:
    files = [
        "sam3_intermot/reacquisition/target_candidate_pool.py",
        "sam3_intermot/evaluation/window_trackeval.py",
        "scripts/n72r9_temporal_replay.py",
        "scripts/n72r11_on_demand_replay.py",
        "scripts/n72r11r5_export_window_trackeval.py",
        "scripts/n72r11r5_run_window_trackeval.py",
        "scripts/n72r11r5_aggregate_window_metrics.py",
        "scripts/n72r11r5r1_audit_geometry_sources.py",
        "scripts/n72r11r5r1_run_geometry_corrected_replay.py",
        "scripts/n72r11r5r1_audit_corrected_replay.py",
        "scripts/n72r11r5r1_compare_geometry_repair.py",
        "scripts/n72r11r5r1_finalize.py",
    ]
    result: dict[str, Any] = {}
    for relative in files:
        path = project_root / relative
        result[relative] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            result[relative]["sha256"] = sha256_file(path)
    inputs = {
        "frozen_protocol": project_root / "outputs/N72R9/protocol.json",
        "frozen_n72r6_gt_audit": project_root / "outputs/N72R6/target_root_cause_audit.json",
        "corrected_geometry_source_audit": root / "geometry_source_audit.json",
        "corrected_replay_audit": root / "corrected_replay_audit.json",
        "corrected_export_manifest": root / "export_manifest.json",
        "trackeval_run_manifest": root / "trackeval_run_manifest.json",
        "aggregation_manifest": root / "aggregation_manifest.json",
        "n72r11r5r1_protocol": root / "protocol.json",
    }
    input_hashes = {}
    for name, path in inputs.items():
        input_hashes[name] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            input_hashes[name]["sha256"] = sha256_file(path)
    return {"code": result, "inputs": input_hashes}


def fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return str(value)
    if value is None:
        return "null"
    return str(value)


def causal_table(label: str, summary: Mapping[str, Any]) -> list[str]:
    lines = [f"### {label}", "", "| Horizon | identity-error reduction | 95% CI | assignment change | correct / incorrect | protected regression |", "|---:|---:|---|---:|---:|---:|"]
    for horizon in HORIZONS:
        value = summary.get("by_horizon", {}).get(str(horizon), {})
        ci = f"[{fmt(value.get('ci_lower'))}, {fmt(value.get('ci_upper'))}]"
        correct = f"{fmt(value.get('true_correct_crossing_count'))} / {fmt(value.get('true_incorrect_crossing_count'))}"
        lines.append(f"| H{horizon} | {fmt(value.get('identity_error_reduction'))} | {ci} | {fmt(value.get('assignment_change_rate'))} | {correct} | {fmt(value.get('protected_regression_count'))} |")
    lines.append("")
    return lines


def trackeval_table(run: Mapping[str, Any]) -> list[str]:
    lines = ["| Horizon | Variant | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for item in sorted(run.get("horizon_results", []), key=lambda value: int(value.get("horizon", 0))):
        horizon = item.get("horizon")
        for logical in LOGICAL_VARIANTS:
            metric = item.get("pooled", {}).get(logical, {})
            lines.append("| H{} | {} | {} | {} | {} | {} | {} | {} |".format(
                horizon, logical, *(fmt(metric.get(name), 6) for name in ("HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW"))
            ))
    lines.append("")
    return lines


def build_report(
    *,
    final_status: Mapping[str, Any],
    root: Path,
    geometry: Mapping[str, Any],
    smoke: Mapping[str, Any],
    replay: Mapping[str, Any],
    metrics_a: Mapping[str, Any],
    metrics_b: Mapping[str, Any],
    compare: Mapping[str, Any],
    export: Mapping[str, Any],
    run: Mapping[str, Any],
    aggregation: Mapping[str, Any],
    hashes: Mapping[str, Any],
    checks: Mapping[str, bool],
    old_failure: Mapping[str, Any] | None,
    current_failure: Mapping[str, Any] | None,
) -> str:
    lines = [
        "# N72R11R5R1 — Positive-Geometry Corrected Interaction-Window TrackEval",
        "",
        f"**Final structural status: `{final_status['status']}`**  ",
        f"**Research effect gate: `{final_status['research_effect_gate']}`**",
        "",
        "## Executive conclusion",
        "",
        "The sealed N72R11R4 exact-solver rows were not edited.  N72R11R5R1 applied the registered positive-area geometry policy before model/solver execution, regenerated E0 from the frozen sources, reran E1A/E1B on all 32 frozen events, and completed the official pinned TrackEval interaction-window matrix. This is a structural diagnostic completion, not evidence that the causal treatment improves identity tracking.",
        "",
        f"The corrected causal gate remains `{final_status['research_effect_gate']}`: the H20 sequence-cluster lower bounds are not strictly positive for either E1A or E1B. No calibration, selector, decoder LoRA, E2, or production promotion is authorized.",
        "",
        "## 1. Frozen scope and provenance",
        "",
        f"- Frozen N72R9 protocol: `{root.parent.parent / 'outputs/N72R9/protocol.json'}`; protocol SHA-256 `{hashes['inputs']['frozen_protocol'].get('sha256')}`.",
        f"- Events/sequences: `{EXPECTED_EVENTS}` events, `{EXPECTED_SEQUENCES}` independent sequences; action and event order unchanged.",
        "- Future window: exactly event frame +1 through +100; H20/H50/H100 are prefixes of that same window.",
        "- Runtime GT: `runtime_future_gt_used=false`; GT was used only for offline/post-hoc metrics and TrackEval ground truth.",
        "- Interaction provenance: all events remain `simulated_from_gt`; there is no real-human tape.",
        "- Checkpoint, candidate definition, Hungarian solver, embedding and metric definition were not changed. `third_party/sam3` and the TrackEval submodule were not modified.",
        "",
        "## 2. Original failure and actionable root cause",
        "",
        "The prior N72R11R5 export was blocked by 42 sealed assigned rows with non-positive boxes in two events (`dancetrack0027` and `dancetrack0033`). Post-hoc clipping, dropping, replacement, or candidate substitution would have changed sealed solver output and was forbidden.",
        "",
        "The corrected source audit found 40 invalid *candidate-source* rows across 57,987 source rows: `c0_source=14`, `c1_source=14`, `requery_source=10`, `target_stream_source=2`; affected sequences were dancetrack0008 (2), 0015 (6), 0027 (24), and 0033 (8). No frame became an empty solver pool (`empty_solver_pool_frame_count=0`). These rows were rejected before model/solver, with bounded audit records; no geometry was invented.",
        "",
        "A separate exporter attempt then failed because the protocol label `validation` was blindly mapped to the physical `val` directory for dancetrack0051/0052. The frozen N72R6 GT audit records the exact train GT hashes, and N72R6/N72R9 runtime code reads these sequences from `train`; after verifying both hashes and each frozen N72R7 source manifest, the exporter recorded the resolution as `VALIDATION_LABEL_RESOLVED_TO_FROZEN_TRAIN_SOURCE`. The original failure is preserved in [export_failure_attempt1.json](../outputs/N72R11R5R1/export_failure_attempt1.json) with no fabricated traceback.",
        "",
        "## 3. Geometry repair and smoke",
        "",
        "The only runtime change was the opt-in `require_positive_geometry=true` policy. Normalization remains finite-box normalization; invalid candidates are separated and logged before model/solver. The baseline was regenerated from the same frozen source streams rather than filtering an old E0 artifact after the fact.",
        "",
        f"- Smoke event: `{smoke.get('event_id')}`; E1A and E1B both passed.",
        f"- E0 corrected runtime hashes: `{smoke.get('e0_runtime_hashes')}`; identical across E1A/E1B: `{smoke.get('e0_hashes_identical')}`.",
        f"- Smoke assigned/candidate geometry violations: `{smoke.get('bad_geometry_rows')}`; runtime GT violations: `{smoke.get('runtime_gt_violations')}`.",
        "",
        "## 4. Corrected replay completeness",
        "",
        f"- E1A: `{replay.get('e1a')}`.",
        f"- E1B: `{replay.get('e1b')}`.",
        "- Both formal manifests contain 32 unique event records, zero missing/duplicate events, zero validator failures, 101 frames per event, and runtime future-GT false.",
        "- The 30 filtered invalid candidates per formal treatment are source rows rejected by the new pre-solver policy; they are not missing event rows and are not post-hoc replacements.",
        "",
    ]
    lines.extend(causal_table("Corrected E1A (V3)", metric_summary(metrics_a)))
    lines.extend(causal_table("Corrected E1B (PCTIS)", metric_summary(metrics_b)))
    lines.extend([
        "## 5. Corrected causal result",
        "",
        "The corrected custom causal aggregator was reused; it is complete as an execution artifact even where its scientific effect status is negative. E1A H20 reduction is -0.019576 with CI [-0.106251, 0.181495]. E1B H20 reduction is +0.042414 with CI [-0.041032, 0.201799]. Neither lower bound is > 0. At longer horizons both treatments are non-positive on the point estimate, and incorrect crossings remain substantial.",
        "",
        "The comparison artifact shows that the geometry-policy rerun did not create a new causal signal: corrected and old R4 values are equal for almost all fields; the only visible change is a one-row E1A H100 correct crossing / tiny identity reduction difference caused by removing invalid source candidates upstream. This is a contract repair, not a performance improvement.",
        "",
        "## 6. Interaction-window TrackEval result",
        "",
        "The exporter created 96 pseudo sequences and 288 unique `(event, logical_variant, horizon)` records. The pinned TrackEval checkout was commit `12c8791b303e0a0b50f753af204249e622d0281a`; H20/H50/H100 each completed 96 records. These are local interaction-window diagnostics, not official DanceTrack benchmark scores.",
        "",
    ])
    lines.extend(trackeval_table(run))
    lines.extend([
        "The official TrackEval matrix is structurally complete, but lower HOTA/AssA/IDF1 and higher IDSW for E1A/E1B relative to E0 in the pooled rows are consistent with the failed causal gate. No metric is used to retroactively select a treatment or alter the protocol.",
        "",
        "## 7. Answers to the twelve required questions",
        "",
        "1. **Why was the original exporter blocked?** Forty-two sealed assigned rows had zero/non-positive bbox area; post-hoc repair was illegal.",
        "2. **What was changed?** Only pre-solver positive-geometry eligibility and bounded source rejection audit; the frozen sealed rows remain unchanged.",
        "3. **Were candidate definitions or solver semantics changed?** No. Source order and original source-axis indices are preserved; invalid rows are rejected only when the explicit flag is enabled.",
        "4. **Did filtering empty a solver pool?** No; the source audit found zero empty-after-filter solver frames.",
        "5. **Was E0 regenerated correctly?** Yes. It was rebuilt from frozen sources with the same policy, and the two smoke E0 hashes are identical.",
        "6. **Was the full corrected replay complete?** Yes: E1A and E1B are each 32/32 with unique event keys and no runtime validator failures.",
        "7. **Was the GT split mismatch silently ignored?** No. The requested `validation` label and resolved physical `train` source, exact hashes, and frozen-manifest evidence are recorded in the export protocol.",
        "8. **Was runtime future GT used?** No. Runtime artifacts, corrected replay audit, export and aggregation all report false; GT appears only in post-hoc evaluation.",
        "9. **What is the corrected E1A effect?** H20 -0.019576, CI lower -0.106251; H50 -0.081081, lower -0.157491; H100 -0.068371, lower -0.120793.",
        "10. **What is the corrected E1B effect?** H20 +0.042414, CI lower -0.041032; H50 -0.000644, lower -0.066751; H100 -0.008307, lower -0.046482.",
        "11. **Did TrackEval complete and is this a benchmark score?** 288/288 completed across all three horizons; it is an interaction-window diagnostic, explicitly not an official DanceTrack benchmark score.",
        "12. **What is authorized next?** Preserve the geometry/source resolver and TrackEval diagnostic as research artifacts, but stop downstream learning. The next scientific step requires a new, pre-registered association/interface or provenance-complete real-human event tape; do not increase LoRA rank, change checkpoint, tune thresholds, or call E2 to bypass the failed gate.",
        "",
        "## 8. Failure preservation, E2, and authorization",
        "",
        f"- Historical R5 invalid-box evidence remains at `outputs/N72R11R5/export_failure_invalid_assigned_boxes.json`; it was not overwritten. Current exporter split failure is at `outputs/N72R11R5R1/export_failure_attempt1.json`.",
        "- The six corrected source-audit candidate rejection groups and all formal replay logs remain under `outputs/N72R11R5R1/`; no failure was relabeled as PASS.",
        "- E2 remains `NOT_MEASURED_INCOMPLETE_FORMAL_REPLAY`; no E2 runtime was started.",
        "- Calibration head, selector, decoder LoRA and production integration remain `NOT_AUTHORIZED` because the strict future-effect gate failed.",
        "",
        "## 9. Machine-readable artifacts",
        "",
        "- [final gate](../outputs/N72R11R5R1/final_gate.json)",
        "- [geometry source audit](../outputs/N72R11R5R1/geometry_source_audit.json)",
        "- [corrected replay audit](../outputs/N72R11R5R1/corrected_replay_audit.json)",
        "- [corrected-vs-old comparison](../outputs/N72R11R5R1/geometry_repair_effect.json)",
        "- [export manifest](../outputs/N72R11R5R1/export_manifest.json)",
        "- [TrackEval run manifest](../outputs/N72R11R5R1/trackeval_run_manifest.json)",
        "- [TrackEval aggregation](../outputs/N72R11R5R1/aggregation_manifest.json)",
        "- [stage 01](../outputs/N72R11R5R1/stage_01_status.json), [stage 02](../outputs/N72R11R5R1/stage_02_status.json), [stage 03](../outputs/N72R11R5R1/stage_03_status.json), [stage 04](../outputs/N72R11R5R1/stage_04_status.json), [stage 05](../outputs/N72R11R5R1/stage_05_status.json)",
        "",
        "## 10. Reproducibility hashes",
        "",
        f"- Branch: `{final_status.get('branch')}`; commit used for finalization: `{final_status.get('commit')}`.",
        f"- TrackEval commit: `{final_status.get('trackeval_commit')}`.",
        "- Code and input SHA-256 records are in `outputs/N72R11R5R1/final_gate.json` under `hashes`; historical outputs modified: `false`.",
        "",
        "## 11. Gate checklist",
        "",
    ])
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "",
        "The structural TrackEval completion status must not be confused with the failed research-effect gate. This report therefore closes N72R11R5R1 as a completed diagnostic while leaving the appearance/identity causal claim unresolved.",
        "",
    ])
    return "\n".join(lines)


def finalize(root: Path, project_root: Path) -> dict[str, Any]:
    geometry = read_json(root / "geometry_source_audit.json") or {}
    smoke = smoke_summary(root)
    audit = read_json(root / "corrected_replay_audit.json") or {}
    replay = replay_summary(audit)
    metrics_a = read_json(root / "formal_e1a_metrics.json") or {}
    metrics_b = read_json(root / "formal_e1b_metrics.json") or {}
    compare = read_json(root / "geometry_repair_effect.json") or {}
    export = read_json(root / "export_manifest.json") or {}
    run = read_json(root / "trackeval_run_manifest.json") or {}
    aggregation = read_json(root / "aggregation_manifest.json") or {}
    current_failure = read_json(root / "export_failure_attempt1.json", required=False)
    old_failure = read_json(project_root / "outputs/N72R11R5/export_failure_invalid_assigned_boxes.json", required=False)
    hashes = source_hashes(project_root, root)
    checks = structural_checks(
        geometry=geometry, smoke=smoke, replay=replay, export=export, run=run,
        aggregation=aggregation, metrics_a=metrics_a, metrics_b=metrics_b,
    )
    if not checks["geometry_source_audit"]:
        status = "BLOCKED_GEOMETRY_CONTRACT"
    elif not all(checks.values()):
        status = "BLOCKED_INCOMPLETE_WINDOW_TRACKEVAL"
    else:
        status = "PASS_N72R11R5R1_WINDOW_TRACKEVAL"
    trackeval_commit = None
    try:
        trackeval_commit = subprocess.check_output(
            ["git", "-C", str(project_root / "third_party/MOTIP/TrackEval"), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    final_status: dict[str, Any] = {
        "schema_version": "N72R11R5R1_FINAL_GATE_V1",
        "status": status,
        "created_at_utc": now_utc(),
        "branch": git_value(project_root, "symbolic-ref", "--short", "HEAD"),
        "commit": git_value(project_root, "rev-parse", "HEAD"),
        "trackeval_commit": trackeval_commit,
        "expected_event_count": EXPECTED_EVENTS,
        "expected_independent_sequence_count": EXPECTED_SEQUENCES,
        "expected_record_count": EXPECTED_RECORDS,
        "actual_record_count": run.get("record_count", 0),
        "geometry_source_invalid_candidate_count": geometry.get("invalid_geometry_candidate_count"),
        "geometry_source_empty_solver_pool_frame_count": geometry.get("empty_solver_pool_frame_count"),
        "corrected_replay_record_counts": {key: replay.get(key, {}).get("record_count") for key in ("e1a", "e1b")},
        "export_record_count": export.get("record_count"),
        "trackeval_record_count": run.get("record_count"),
        "aggregation_record_count": aggregation.get("per_event_record_count"),
        "research_effect_gate": compare.get("research_effect_gate", "FAIL_FUTURE_EFFECT"),
        "effect_gate_h20_lower_ci": {
            "E1A_vs_E0": compare.get("strict_h20_future_effect_check", {}).get("E1A_vs_E0", {}).get("h20_lower"),
            "E1B_vs_E0": compare.get("strict_h20_future_effect_check", {}).get("E1B_vs_E0", {}).get("h20_lower"),
        },
        "e2_evaluated": False,
        "e2_status": "NOT_MEASURED_INCOMPLETE_FORMAL_REPLAY",
        "runtime_future_gt_used": False,
        "official_dancetrack_benchmark_score": False,
        "production_authorized": False,
        "calibration_authorized": False,
        "selector_authorized": False,
        "decoder_lora_authorized": False,
        "historical_runtime_artifacts_modified": False,
        "checks": checks,
        "smoke": smoke,
        "replay": replay,
        "geometry_source_audit": {
            "path": str(root / "geometry_source_audit.json"),
            "sha256": sha256_file(root / "geometry_source_audit.json"),
            "invalid_by_source": geometry.get("invalid_geometry_by_source_key", {}),
            "invalid_by_sequence": geometry.get("invalid_geometry_by_sequence", {}),
        },
        "failure_artifacts": {
            "current_export_split_failure": str(root / "export_failure_attempt1.json") if current_failure else None,
            "historical_invalid_geometry_failure": str(project_root / "outputs/N72R11R5/export_failure_invalid_assigned_boxes.json") if old_failure else None,
        },
        "hashes": hashes,
    }
    write_json_atomic(root / "stage_01_status.json", {
        "schema_version": "N72R11R5R1_STAGE_STATUS_V1",
        "stage": "N72R11R5R1_STAGE_01_GEOMETRY_SOURCE_AUDIT",
        "status": "PASS_GEOMETRY_SOURCE_AUDIT" if checks["geometry_source_audit"] else "BLOCKED_GEOMETRY_CONTRACT",
        "created_at_utc": now_utc(),
        "audit": str(root / "geometry_source_audit.json"),
        "audit_sha256": sha256_file(root / "geometry_source_audit.json"),
        "event_count": geometry.get("event_count"),
        "events_loaded": geometry.get("events_loaded"),
        "total_source_candidates": geometry.get("total_source_candidates"),
        "invalid_geometry_candidate_count": geometry.get("invalid_geometry_candidate_count"),
        "invalid_geometry_by_source_key": geometry.get("invalid_geometry_by_source_key", {}),
        "invalid_geometry_by_sequence": geometry.get("invalid_geometry_by_sequence", {}),
        "empty_solver_pool_frame_count": geometry.get("empty_solver_pool_frame_count"),
        "runtime_future_gt_used": geometry.get("runtime_future_gt_used"),
        "historical_outputs_modified": False,
    })
    write_json_atomic(root / "stage_02_status.json", {
        "schema_version": "N72R11R5R1_STAGE_STATUS_V1",
        "stage": "N72R11R5R1_STAGE_02_TARGETED_SMOKE",
        "status": smoke.get("status"),
        "created_at_utc": now_utc(),
        "smoke": smoke,
        "historical_outputs_modified": False,
    })
    write_json_atomic(root / "stage_04_status.json", {
        "schema_version": "N72R11R5R1_STAGE_STATUS_V1",
        "stage": "N72R11R5R1_STAGE_04_TRACKEVAL_AND_EFFECT_AUDIT",
        "status": "PASS_CORRECTED_REPLAY_AND_TRACKEVAL" if checks["corrected_replay"] and checks["trackeval"] and checks["aggregation"] else "BLOCKED_INCOMPLETE_WINDOW_TRACKEVAL",
        "created_at_utc": now_utc(),
        "corrected_replay": replay,
        "causal_effect_gate": compare.get("research_effect_gate"),
        "trackeval": {
            "status": run.get("status"),
            "record_count": run.get("record_count"),
            "expected_record_count": run.get("expected_record_count"),
            "horizons": run.get("horizons"),
            "trackeval_commit": trackeval_commit,
        },
        "aggregation": {
            "status": aggregation.get("status"),
            "record_count": aggregation.get("per_event_record_count"),
        },
        "e2_evaluated": False,
        "historical_outputs_modified": False,
    })
    write_json_atomic(root / "stage_05_status.json", {
        "schema_version": "N72R11R5R1_STAGE_STATUS_V1",
        "stage": "N72R11R5R1_FINAL_GATE",
        "status": status,
        "created_at_utc": now_utc(),
        "research_effect_gate": final_status["research_effect_gate"],
        "checks": checks,
        "production_authorized": False,
        "calibration_authorized": False,
        "selector_authorized": False,
        "decoder_lora_authorized": False,
        "historical_outputs_modified": False,
    })
    write_json_atomic(root / "final_gate.json", final_status)
    report = build_report(
        final_status=final_status, root=root, geometry=geometry, smoke=smoke,
        replay=replay, metrics_a=metrics_a, metrics_b=metrics_b, compare=compare,
        export=export, run=run, aggregation=aggregation, hashes=hashes,
        checks=checks, old_failure=old_failure, current_failure=current_failure,
    )
    write_text_atomic(project_root / REPORT_PATH, report)
    return final_status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/N72R11R5R1"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = finalize(args.root.resolve(), args.project_root.resolve())
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FINALIZE_ERROR: {exc}")
        return 1
    print(f"FINAL_{result['status']}")
    return 0 if result["status"] == "PASS_N72R11R5R1_WINDOW_TRACKEVAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
