#!/usr/bin/env python
"""Finalize N72R11R5 without promoting partial TrackEval output.

The finalizer is deliberately conservative: a sealed-runtime geometry failure,
missing export manifest, incomplete horizon, or missing aggregate produces a
machine-readable BLOCKED status and a report with no invented metric values.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam3_intermot.evaluation.window_trackeval import (  # noqa: E402
    FROZEN_METRICS,
    HORIZONS,
    LOGICAL_VARIANTS,
    WindowTrackEvalError,
    load_frozen_sources,
    sha256_file,
    write_json_atomic,
    write_text_atomic,
)


REPORT_NAME = "N72R11R5_WINDOW_TRACKEVAL_REPORT.md"
EXPECTED_RECORDS = 32 * len(LOGICAL_VARIANTS) * len(HORIZONS)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WindowTrackEvalError(f"expected JSON object: {path}")
    return value


def _git_value(project_root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=project_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_facts(project_root: Path) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "project_root": str(project_root),
        "source_metrics": {},
        "frozen_protocol": None,
        "trackeval_commit": _git_value(project_root / "third_party/MOTIP/TrackEval", "rev-parse", "HEAD"),
    }
    for logical, relative in FROZEN_METRICS.items():
        path = project_root / relative
        entry: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            entry["sha256"] = sha256_file(path)
            try:
                metrics = json.loads(path.read_text(encoding="utf-8"))
                entry["complete"] = metrics.get("complete")
                entry["source_manifest"] = metrics.get("source_manifest")
                entry["source_manifest_sha256"] = metrics.get("source_manifest_sha256")
                entry["protocol"] = metrics.get("protocol")
                entry["protocol_sha256"] = metrics.get("protocol_sha256")
                if facts["frozen_protocol"] is None:
                    facts["frozen_protocol"] = {
                        "path": metrics.get("protocol"),
                        "sha256": metrics.get("protocol_sha256"),
                    }
            except (OSError, json.JSONDecodeError, AttributeError):
                entry["parse_error"] = True
        facts["source_metrics"][logical] = entry
    return facts


def _failure_summary(failure: dict[str, Any] | None) -> dict[str, Any]:
    if not failure:
        return {"present": False}
    rows = failure.get("invalid_rows")
    rows = rows if isinstance(rows, list) else []
    events: dict[str, int] = {}
    variants: dict[str, int] = {}
    frames: dict[str, list[int]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        event_id = str(row.get("event_id"))
        variant = str(row.get("sealed_runtime_variant"))
        events[event_id] = events.get(event_id, 0) + 1
        variants[variant] = variants.get(variant, 0) + 1
        frame = row.get("original_frame")
        if isinstance(frame, int):
            frames.setdefault(event_id, []).append(frame)
    return {
        "present": True,
        "path": str(Path(failure.get("path", "export_failure_invalid_assigned_boxes.json"))),
        "status": failure.get("status"),
        "reason": failure.get("reason"),
        "invalid_assigned_row_count": failure.get("invalid_assigned_row_count", len(rows)),
        "unique_event_logical_frame_public_count": failure.get("unique_event_logical_frame_public_count"),
        "events": events,
        "variants": variants,
        "frames": {event: sorted(set(values)) for event, values in frames.items()},
        "allowed_repairs": failure.get("allowed_repairs", []),
        "historical_runtime_artifacts_modified": failure.get("historical_runtime_artifacts_modified"),
    }


def _stage_payload(stage: str, status: str, **fields: Any) -> dict[str, Any]:
    return {
        "schema_version": "N72R11R5_STAGE_STATUS_V1",
        "stage": stage,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }


def _markdown_report(
    *, project_root: Path, final_status: dict[str, Any], source_facts: dict[str, Any],
    failure: dict[str, Any] | None, export: dict[str, Any] | None,
    run: dict[str, Any] | None, aggregation: dict[str, Any] | None,
) -> str:
    failure_info = _failure_summary(failure)
    lines = [
        "# N72R11R5 — Interaction-Window TrackEval Diagnostic",
        "",
        f"**Final status: `{final_status['status']}`**",
        "",
        "## Conclusion",
        "",
        "This post-hoc diagnostic did not alter SAM3, PCTIS, Hungarian association, public IDs, candidates, thresholds, runtime artifacts, or E2. It is blocked before a complete 32-event × 3-variant × 3-horizon TrackEval matrix can be evaluated.",
        "",
        "The blocked condition is an input-integrity failure in the sealed N72R11R4 solver output, not a TrackEval score and not evidence for or against the research hypothesis. No incomplete aggregate, imputed box, or official DanceTrack benchmark score is reported.",
        "",
        "## Scope and frozen inputs",
        "",
        "- Evaluation scope: `INTERACTION_WINDOW_DIAGNOSTIC`.",
        "- Each pseudo sequence is exactly `[event_frame+1, event_frame+H]`, remapped to frames `1..H`, for `H20/H50/H100`.",
        "- Variants: `E0_BASELINE_B0`, `E1A_V3`, `E1B_PCTIS`.",
        "- Expected matrix: 32 events, 18 independent original sequences, 96 pseudo windows, 288 per-event variant records.",
        "- TrackEval values, if available in a future valid run, must come only from the pinned official checkout. `official_dancetrack_benchmark_score=false` because these are local interaction windows.",
        "- All N72R11R4 events are `simulated_from_gt`; they are not historical real-human evidence.",
        "- E2 was not evaluated and remains `NOT_MEASURED_INCOMPLETE_FORMAL_REPLAY`.",
        "",
        "### Frozen input hashes",
        "",
    ]
    for logical, info in source_facts["source_metrics"].items():
        lines.append(f"- `{logical}` metrics: `{info.get('path')}`; SHA-256 `{info.get('sha256')}`.")
        lines.append(f"  - source manifest: `{info.get('source_manifest')}`; recorded SHA-256 `{info.get('source_manifest_sha256')}`.")
    lines.extend([
        f"- Frozen protocol: `{source_facts.get('frozen_protocol', {}).get('path') if isinstance(source_facts.get('frozen_protocol'), dict) else None}`; recorded SHA-256 `{source_facts.get('frozen_protocol', {}).get('sha256') if isinstance(source_facts.get('frozen_protocol'), dict) else None}`.",
        f"- Pinned TrackEval commit: `{source_facts.get('trackeval_commit')}`.",
        "",
        "## Execution record",
        "",
        "1. Export smoke for `n72r5-pool-n37-dancetrack0001-0296-authoritative_reassign-001` passed: 1 event, 3 horizons, 9 export records.",
        "2. The first exporter smoke failed before experiment logic because the direct script did not add the project root to `sys.path`; the failure is retained in `outputs/N72R11R5/smoke_attempt1/export_failure_attempt1.json`. The import-path repair was targeted and the same smoke then passed.",
        "3. TrackEval smoke attempt 1 failed because the pinned CLI parsed the default `SEQMAP_FILE=None` as a one-element list. The failure is retained in `outputs/N72R11R5/smoke_attempt1/trackeval_raw/h020/trackeval_failure_attempt1.json`.",
        "4. TrackEval smoke attempt 2 failed because the pinned CLI similarly parsed `OUTPUT_FOLDER` as a list. The failure is retained in `outputs/N72R11R5/smoke_attempt1/trackeval_raw/h020/trackeval_failure_attempt2.json`.",
        "5. The runner was repaired locally by using the supported `SEQ_INFO` path and a narrow compatibility unwrap for scalar CLI arguments; the identical H20 smoke then passed with 3/3 records. The pinned TrackEval submodule was not modified.",
        "6. The full exporter was run CPU-only in one blocking process. Its first failure and the subsequent lossless preflight regression are both retained; no full TrackEval run was started after the preflight found the same sealed invalid rows.",
        "",
        "## Actionable blocking evidence",
        "",
    ])
    if failure_info.get("present"):
        lines.extend([
            f"- Artifact: `outputs/N72R11R5/export_failure_invalid_assigned_boxes.json`.",
            f"- Status: `{failure_info.get('status')}`; reason: `{failure_info.get('reason')}`.",
            f"- Invalid exact-solver assigned rows: **{failure_info.get('invalid_assigned_row_count')}**; de-duplicated logical `(event, variant, frame, public_id)` rows: **{failure_info.get('unique_event_logical_frame_public_count')}**.",
            f"- Affected events: **{len(failure_info.get('events', {}))}** — `{', '.join(sorted(failure_info.get('events', {})))}`.",
            f"- Per sealed variant: `{failure_info.get('variants')}`.",
            "- Representative row: `dancetrack0027`, absolute frame `222` (relative H100 frame `74`), public ID `1007`, box `[571.0000228881836, 307.00000047683716, 571.0000228881836, 308.00000050105155]`; therefore `x2 == x1`.",
            "- The rows have `solver_status=ASSIGNED_TO_PUBLIC_ID`, explicit exact global solver authority, and no alternate box field. Clipping, dropping, replacing with a one-pixel box, or selecting a different candidate would change the sealed solver output and violate the protocol.",
            "- No historical runtime artifact was modified; `allowed_repairs=[]`.",
        ])
        lines.append("- Affected absolute frames by event:")
        for event_id, frames in sorted(failure_info.get("frames", {}).items()):
            lines.append(f"  - `{event_id}`: `{frames}`.")
    else:
        lines.append("- No export failure artifact was found; see machine-readable final status for the exact missing/incomplete input condition.")
    lines.extend([
        "",
        "## Machine-readable gate",
        "",
        f"- Export status: `{(export or {}).get('status', 'NOT_AVAILABLE')}`.",
        f"- TrackEval status: `{(run or {}).get('status', 'NOT_RUN')}`; records `{(run or {}).get('record_count', 0)}/{(run or {}).get('expected_record_count', EXPECTED_RECORDS)}`.",
        f"- Aggregation status: `{(aggregation or {}).get('status', 'NOT_RUN')}`.",
        f"- Final status: `{final_status['status']}`; metrics available: `{final_status.get('metrics_available')}`.",
        "- Because the complete gate did not pass, no pooled/paired TrackEval aggregate and no HOTA/AssA/DetA/IDF1/MOTA/IDSW claim is made here.",
        "",
        "## Questions required by the protocol",
        "",
        "- Q1 (PCTIS vs B0 association): not estimable in N72R11R5 because the complete official interaction-window matrix was blocked before aggregation.",
        "- Q2 (PCTIS vs V3): not estimable for the same reason.",
        "- Q3 (association vs detection): not estimable; the diagnostic cannot use the one-event smoke as a 32-event result.",
        "- Q4 (interpretation of the earlier `+0.042414` identity-error reduction): that N72R11R4 value remains a frozen legacy post-hoc identity metric, not an interaction-window TrackEval result. N72R11R5 does not reinterpret it or upgrade it to a benchmark claim.",
        "",
        "## Required next step",
        "",
        "The smallest protocol-preserving recovery is upstream: repair/re-seal the two affected N72R11R4 runtime windows so every exact solver-assigned future box has finite positive width and height, or obtain an explicitly authorized equivalent runtime regeneration. Then rerun the lossless exporter preflight and, only if it passes, the complete pinned TrackEval matrix. N72R11R5 itself must not alter those boxes or omit the rows.",
        "",
        "No calibration head, selector, decoder LoRA, E2 evaluation, threshold change, checkpoint change, or GT-ID remapping was performed or authorized by this blocked result.",
        "",
        "## Artifacts",
        "",
        "- Machine-readable final gate: `outputs/N72R11R5/FINAL_STATUS.json`.",
        "- Export failure: `outputs/N72R11R5/export_failure_invalid_assigned_boxes.json`.",
        "- Stage statuses: `outputs/N72R11R5/stage_01_status.json`, `outputs/N72R11R5/stage_02_status.json`, `outputs/N72R11R5/stage_03_status.json`.",
        "",
    ])
    return "\n".join(lines) + "\n"


def finalize(root: Path, project_root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    source_facts = _source_facts(project_root)
    failure = _read_json(root / "export_failure_invalid_assigned_boxes.json")
    export = _read_json(root / "export_manifest.json")
    run = _read_json(root / "trackeval_run_manifest.json")
    aggregation = _read_json(root / "aggregation_manifest.json")
    aggregation_blocked = _read_json(root / "aggregation_blocked.json")

    reasons: list[str] = []
    if failure and failure.get("status") != "PASS":
        reasons.append(f"sealed export failure: {failure.get('status')}")
    if export is None or export.get("status") != "PASS_EXPORT":
        reasons.append(f"export is incomplete: {(export or {}).get('status', 'MISSING')}")
    if run is None or run.get("status") != "PASS_TRACKEVAL":
        reasons.append(f"TrackEval is incomplete: {(run or {}).get('status', 'NOT_RUN')}")
    if aggregation is None or aggregation.get("status") != "PASS_COMPLETE_INPUT":
        reasons.append(f"aggregation is incomplete: {(aggregation or aggregation_blocked or {}).get('status', 'NOT_RUN')}")

    # Reuse the frozen-source loader as an additional read-only provenance
    # check.  It never runs SAM3 and it never changes the historical artifacts.
    source_validation: dict[str, Any]
    try:
        frozen = load_frozen_sources(project_root)
        source_validation = {
            "status": "PASS_FROZEN_INPUTS",
            "event_count": len(frozen["events"]),
            "independent_sequence_count": len({event["sequence"] for event in frozen["events"]}),
            "protocol_sha256": frozen.get("protocol_sha256"),
        }
    except (OSError, KeyError, TypeError, WindowTrackEvalError) as exc:
        source_validation = {"status": "ERROR_FROZEN_INPUTS", "error": str(exc)}
        reasons.append(f"frozen input validation error: {exc}")

    status = "PASS_COMPLETE_WINDOW_TRACKEVAL" if not reasons else "BLOCKED_INCOMPLETE_WINDOW_TRACKEVAL"
    final_status = {
        "schema_version": "N72R11R5_FINAL_STATUS_V1",
        "status": status,
        "evaluation_scope": "INTERACTION_WINDOW_DIAGNOSTIC",
        "official_dancetrack_benchmark_score": False,
        "expected_event_count": 32,
        "expected_independent_sequence_count": 18,
        "expected_record_count": EXPECTED_RECORDS,
        "actual_record_count": (run or {}).get("record_count", 0),
        "metrics_available": status == "PASS_COMPLETE_WINDOW_TRACKEVAL",
        "e2_evaluated": False,
        "e2_status": "NOT_MEASURED_INCOMPLETE_FORMAL_REPLAY",
        "runtime_future_gt_used": False,
        "reasons": reasons,
        "source_validation": source_validation,
        "source_facts": source_facts,
        "failure_artifact": str(root / "export_failure_invalid_assigned_boxes.json") if failure else None,
        "historical_runtime_artifacts_modified": False,
    }

    failure_status = (failure or {}).get("status")
    stage_01_status = "BLOCKED_INVALID_SEALED_ASSIGNED_BOXES" if failure_status == "BLOCKED_INVALID_SEALED_ASSIGNED_BOXES" else ((export or {}).get("status") or "BLOCKED_EXPORT_NOT_COMPLETE")
    write_json_atomic(
        root / "stage_01_status.json",
        _stage_payload(
            "N72R11R5_EXPORT",
            stage_01_status,
            expected_events=32,
            expected_windows=96,
            expected_records=288,
            export_status=(export or {}).get("status", "NOT_WRITTEN"),
            invalid_assigned_row_count=(failure or {}).get("invalid_assigned_row_count", 0),
            unique_invalid_logical_rows=(failure or {}).get("unique_event_logical_frame_public_count", 0),
            failure_artifact=str(root / "export_failure_invalid_assigned_boxes.json") if failure else None,
            historical_runtime_artifacts_modified=False,
            command="CUDA_VISIBLE_DEVICES= /home/lwr/anaconda3/envs/intermot/bin/python -u scripts/n72r11r5_export_window_trackeval.py --project-root . --output-root outputs/N72R11R5 --mode full",
        ),
    )
    write_json_atomic(
        root / "stage_02_status.json",
        _stage_payload(
            "N72R11R5_TRACKEVAL",
            "BLOCKED_EXPORT_GATE" if stage_01_status != "PASS_EXPORT" else ((run or {}).get("status") or "NOT_RUN"),
            smoke_status="PASS_H020_3_RECORDS",
            full_status=(run or {}).get("status", "NOT_RUN_EXPORT_BLOCKED"),
            expected_records=288,
            actual_records=(run or {}).get("record_count", 0),
            pinned_trackeval_commit=source_facts.get("trackeval_commit"),
            cuda_visible_devices="",
            trackeval_submodule_modified=False,
        ),
    )
    write_json_atomic(
        root / "stage_03_status.json",
        _stage_payload(
            "N72R11R5_AGGREGATION_AND_FINAL_GATE",
            "BLOCKED_INCOMPLETE_WINDOW_TRACKEVAL" if status != "PASS_COMPLETE_WINDOW_TRACKEVAL" else "PASS_COMPLETE_WINDOW_TRACKEVAL",
            aggregation_status=(aggregation or aggregation_blocked or {}).get("status", "NOT_RUN"),
            final_status=status,
            aggregate_files_written=bool(aggregation and aggregation.get("status") == "PASS_COMPLETE_INPUT"),
            reasons=reasons,
        ),
    )
    write_json_atomic(root / "FINAL_STATUS.json", final_status)
    report = _markdown_report(
        project_root=project_root,
        final_status=final_status,
        source_facts=source_facts,
        failure=failure,
        export=export,
        run=run,
        aggregation=aggregation or aggregation_blocked,
    )
    write_text_atomic(project_root / "docs" / REPORT_NAME, report)
    return final_status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/N72R11R5"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    try:
        result = finalize(args.root.resolve(), args.project_root.resolve())
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, WindowTrackEvalError) as exc:
        print(f"FINALIZE_ERROR: {exc}")
        return 1
    print(f"FINAL_{result['status']}")
    return 0 if result["status"] == "PASS_COMPLETE_WINDOW_TRACKEVAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
