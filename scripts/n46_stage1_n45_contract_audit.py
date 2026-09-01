#!/usr/bin/env python3
"""N46 Stage 01: read-only audit of the frozen N45 runtime contract.

This audit independently recomputes Hungarian-with-NONE from every frozen N42
write score matrix.  It never loads GT and never rewrites N45 artifacts.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n44_assignment_common import hungarian_with_none


EVENTS = ROOT / "outputs/n37/real_event_manifest.json"
N42 = ROOT / "outputs/n42/replay/runtime/t0"
N45_RUNTIME = ROOT / "outputs/n45/replay/runtime"
OUT = ROOT / "outputs/n46"
MISMATCHES = OUT / "replay/assignment_mismatches.json"
STAGE = OUT / "stage_01_status.json"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
BRANCHES = ("no_write", "write_baseline", "write_plus_n44")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(values: Any, public_id_count: int) -> list[int]:
    return [(-1 if int(value) >= public_id_count else int(value)) for value in values]


def candidate_signature(candidates: list[dict[str, Any]]) -> list[tuple[Any, Any, Any]]:
    return [(x.get("native_tid"), x.get("box"), x.get("confidence")) for x in candidates]


def slim_signature(row: dict[str, Any]) -> list[tuple[Any, Any, Any]]:
    return [(x.get("native_tid"), x.get("box"), x.get("confidence")) for x in row.get("rows", [])]


def future_flag_issues(value: Any, path: str = "") -> list[str]:
    issues: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            lower = key_text.lower()
            child_path = f"{path}/{key_text}"
            if lower in {"runtime_future_gt_used", "future_gt_used", "gt_loaded_posthoc", "gt_loaded_in_worker"} and child is not False:
                issues.append(f"{child_path}={child!r}; expected false")
            if lower == "future_gt_fields_sent" and child not in ([], None):
                issues.append(f"{child_path} is nonempty")
            if "future_gt_unused" in lower:
                issues.append(f"{child_path} uses reverse future-GT naming")
            issues.extend(future_flag_issues(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(future_flag_issues(child, f"{path}/{index}"))
    return issues


def expected_events() -> dict[str, dict[str, Any]]:
    payload = load(EVENTS)
    if payload.get("status") != "PASS" or len(payload.get("events", [])) != 24:
        raise RuntimeError("frozen N37 event manifest is invalid")
    return {str(item["event"]["event_id"]): item["event"] for item in payload["events"]}


def audit() -> dict[str, Any]:
    event_map = expected_events()
    runtime_files = sorted(N45_RUNTIME.glob("*.json"))
    mismatches: list[dict[str, Any]] = []
    failures: list[str] = []
    counts = {"source_traces_checked": 0, "runtime_traces_checked": 0, "source_candidate_rows_checked": 0, "runtime_candidate_rows_checked": 0}
    if len(runtime_files) != 24:
        failures.append(f"N45 runtime artifact count {len(runtime_files)} != 24")
    if {path.stem for path in runtime_files} != set(event_map):
        failures.append("N45 runtime artifact event IDs do not exactly match the frozen 24-event manifest")
    for event_id, event in sorted(event_map.items()):
        source_path = N42 / f"{event_id}.json"
        runtime_path = N45_RUNTIME / f"{event_id}.json"
        if not source_path.is_file() or not runtime_path.is_file():
            failures.append(f"missing source/runtime artifact {event_id}")
            continue
        source_payload = load(source_path)
        runtime_payload = load(runtime_path)
        if source_payload.get("runtime_boundary", {}).get("runtime_future_gt_used") is not False:
            failures.append(f"{event_id}: N42 source runtime future-GT flag is not false")
        failures.extend(f"{event_id}: {issue}" for issue in future_flag_issues(source_payload.get("runtime_boundary", {}), "runtime_boundary"))
        failures.extend(f"{event_id}: {issue}" for issue in future_flag_issues(runtime_payload.get("runtime_boundary", {}), "runtime_boundary"))
        source_variants = source_payload.get("variants", {})
        runtime_variants = runtime_payload.get("variants", {})
        if set(runtime_variants) != set(VARIANTS):
            failures.append(f"{event_id}: runtime variants mismatch")
            continue
        if set(source_variants) != set(VARIANTS):
            failures.append(f"{event_id}: N42 source variants mismatch")
            continue
        for variant in VARIANTS:
            source_branches = source_variants[variant].get("branches", {})
            runtime_entry = runtime_variants[variant]
            if set(source_branches) != {"memory_write=False", "memory_write=True"}:
                failures.append(f"{event_id}/{variant}: N42 source branches mismatch")
                continue
            if set(runtime_entry) != set(BRANCHES) | {"frame_attribution", "runtime_future_gt_used"}:
                failures.append(f"{event_id}/{variant}: N45 branch schema mismatch")
            no_source = source_branches["memory_write=False"]["future_trace"]
            write_source = source_branches["memory_write=True"]["future_trace"]
            no_runtime = runtime_entry.get("no_write", [])
            write_runtime = runtime_entry.get("write_baseline", [])
            plus_runtime = runtime_entry.get("write_plus_n44", [])
            for label, trace in (("source_no_write", no_source), ("source_write", write_source), ("runtime_no_write", no_runtime), ("runtime_write", write_runtime), ("runtime_plus", plus_runtime)):
                if not isinstance(trace, list) or len(trace) != 100:
                    failures.append(f"{event_id}/{variant}/{label}: trace length is not 100")
                    continue
                frames = [int(item["frame"]) for item in trace]
                if frames != list(range(frames[0], frames[0] + 100)):
                    failures.append(f"{event_id}/{variant}/{label}: duplicate/missing/non-contiguous frame")
                for item in trace:
                    runtime_like = label.startswith("runtime")
                    if runtime_like:
                        counts["runtime_traces_checked"] += 1
                        rows = item.get("rows", [])
                        counts["runtime_candidate_rows_checked"] += len(rows)
                        natives = [row.get("native_tid") for row in rows if row.get("native_tid") is not None]
                        if len(natives) != len(set(natives)):
                            failures.append(f"{event_id}/{variant}/{label}/{item.get('frame')}: duplicate native ID")
                        failures.extend(f"{event_id}/{variant}/{label}/{item.get('frame')}: {issue}" for issue in future_flag_issues(item, "row"))
                    else:
                        counts["source_traces_checked"] += 1
                        counts["source_candidate_rows_checked"] += len(item.get("candidate_audit", {}).get("candidates", []))
                        candidates = item.get("candidate_audit", {}).get("candidates", [])
                        natives = [row.get("native_tid") for row in candidates if row.get("native_tid") is not None]
                        if len(natives) != len(set(natives)):
                            failures.append(f"{event_id}/{variant}/{label}/{item.get('frame')}: duplicate native ID")
                        failures.extend(f"{event_id}/{variant}/{label}/{item.get('frame')}: {issue}" for issue in future_flag_issues(item.get("candidate_audit", {}), "candidate_audit"))
            if not (isinstance(no_source, list) and isinstance(write_source, list) and isinstance(no_runtime, list) and isinstance(write_runtime, list) and isinstance(plus_runtime, list)):
                continue
            for no_source_entry, write_source_entry, no_runtime_entry, write_runtime_entry, plus_runtime_entry in zip(no_source, write_source, no_runtime, write_runtime, plus_runtime):
                no_audit = no_source_entry["candidate_audit"]
                write_audit = write_source_entry["candidate_audit"]
                if candidate_signature(no_audit.get("candidates", [])) != candidate_signature(write_audit.get("candidates", [])):
                    failures.append(f"{event_id}/{variant}/{write_source_entry['frame']}: N42 no/write candidate rows differ")
                if candidate_signature(no_audit.get("candidates", [])) != slim_signature(no_runtime_entry) or candidate_signature(write_audit.get("candidates", [])) != slim_signature(write_runtime_entry):
                    failures.append(f"{event_id}/{variant}/{write_source_entry['frame']}: N45 runtime candidate rows differ from N42 source")
                if slim_signature(write_runtime_entry) != slim_signature(plus_runtime_entry):
                    failures.append(f"{event_id}/{variant}/{write_source_entry['frame']}: write/plus candidate rows differ")
                if write_runtime_entry.get("public_id_order") != plus_runtime_entry.get("public_id_order"):
                    failures.append(f"{event_id}/{variant}/{write_source_entry['frame']}: write/plus public-ID axis differs")
                pids = list(write_audit.get("public_id_order", []))
                scores = np.asarray(write_audit.get("fused_scores"), dtype=np.float32)
                recomputed = normalize(hungarian_with_none(scores), len(pids))
                original = normalize(write_audit.get("assignment_after_scope", []), len(pids))
                if recomputed != original:
                    mismatches.append({"event_id": event_id, "variant": variant, "frame": int(write_source_entry["frame"]), "public_id_count": len(pids), "original_n42_write_assignment": original, "recomputed_hungarian_with_none": recomputed})
                runtime_assignment = normalize(write_runtime_entry.get("assignment", []), len(pids))
                if runtime_assignment != original:
                    mismatches.append({"event_id": event_id, "variant": variant, "frame": int(write_source_entry["frame"]), "kind": "runtime_write_baseline_vs_source", "original_n42_write_assignment": original, "runtime_write_baseline_assignment": runtime_assignment})
    OUT.mkdir(parents=True, exist_ok=True)
    MISMATCHES.parent.mkdir(parents=True, exist_ok=True)
    mismatch_payload = {"schema": "N46_N45_WRITE_ASSIGNMENT_RECOMPUTE_V1", "status": "PASS" if not mismatches else "FAIL_ASSIGNMENT_RECOMPUTE", "mismatch_count": len(mismatches), "n44_increment_available": not mismatches, "mismatches": mismatches, "runtime_future_gt_used": False, "gt_loaded": False}
    MISMATCHES.write_text(json.dumps(mismatch_payload, indent=2) + "\n", encoding="utf-8")
    gate = {"runtime_artifacts_exactly_24": len(runtime_files) == 24 and {path.stem for path in runtime_files} == set(event_map), "variants_exactly_m0_m4": not any("variants mismatch" in failure for failure in failures), "each_variant_has_three_branches": not any("branch schema mismatch" in failure or "trace length is not 100" in failure for failure in failures), "write_plus_candidate_rows_equal": not any("write/plus candidate rows differ" in failure for failure in failures), "write_plus_public_id_axis_equal": not any("write/plus public-ID axis differs" in failure for failure in failures), "source_future_traces_100": not any("source_" in failure and "trace length is not 100" in failure for failure in failures), "runtime_future_traces_100": not any("runtime_" in failure and "trace length is not 100" in failure for failure in failures), "native_ids_unique": not any("duplicate native ID" in failure for failure in failures), "runtime_future_gt_false_direct": not any("future-GT flag" in failure or "future_gt_used" in failure or "reverse future-GT" in failure for failure in failures), "write_assignment_recompute_matches": not mismatches, "n44_increment_available": not mismatches}
    status = "PASS" if not failures and not mismatches else "FAIL"
    result = {"status": status, "protocol": "N46_STAGE_01_N45_CONTRACT_AUDIT_V1", "command": ["python", "scripts/n46_stage1_n45_contract_audit.py"], "inputs": {"n37_event_manifest": str(EVENTS), "n42_frozen_runtime": str(N42), "n45_frozen_runtime": str(N45_RUNTIME)}, "outputs": {"assignment_mismatches": str(MISMATCHES)}, "metrics": {"event_count": len(event_map), "runtime_artifact_count": len(runtime_files), **counts, "assignment_mismatch_count": len(mismatches), "n44_increment_available": not mismatches, "failures": failures[:100]}, "gate_checks": gate, "failure_root_cause": "No assignment mismatches are expected; any mismatch makes the N44 incremental attribution unavailable. This stage is read-only and does not load GT.", "next_action": "Run the N46 structural diagnosis only if the write assignment recomputation and all three-branch runtime contract checks pass.", "runtime_future_gt_used": False, "gt_loaded_posthoc": False, "finished_at": now()}
    STAGE.parent.mkdir(parents=True, exist_ok=True)
    STAGE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "assignment_mismatches": len(mismatches), "output": str(STAGE)}))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    audit()
