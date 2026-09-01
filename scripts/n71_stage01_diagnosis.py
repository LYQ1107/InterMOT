"""Independent CPU diagnosis of the frozen N70 replay boundary.

This reader never constructs a new event or changes N70.  It classifies the
stored row-wise residual against the stored per-frame assignment and keeps
candidate absence separate from association changes.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
N70_ARTIFACTS = ROOT / "outputs/N70/replay/event_artifacts"
OUT = ROOT / "outputs/N71/diagnosis"
STAGE = ROOT / "outputs/N71/stage_01_status.json"
N70_AUDIT = ROOT / "outputs/N70/replay/replay_integrity_audit.json"
BRANCH_KEYS = {"A": "branch_a", "B": "branch_b"}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(float(value)) != float("inf")


def main() -> None:
    files = sorted(Path(p) for p in glob.glob(str(N70_ARTIFACTS / "*.jsonl")))
    if len(files) != 24:
        raise RuntimeError(f"expected 24 frozen N70 event artifacts, found {len(files)}")
    totals: dict[str, Any] = {
        "frame_rows": 0,
        "events": set(),
        "sequences": set(),
        "actions": Counter(),
        "candidate_absent_frames": 0,
        "target_public_assignment_absent_frames": 0,
        "mapping_uncertain_frames": 0,
        "runtime_future_gt_true": 0,
        "variant_axis_mismatch_frames_retained": 0,
        "branches": {b: {"frame_rows": 0, "score_changed": 0, "assignment_changed": 0, "target_assignment_changed": 0, "correct_changes": 0, "incorrect_changes": 0, "neutral_changes": 0, "untouched_changed_frames": 0, "untouched_changed_total": 0, "candidate_absent_frames": 0, "target_present_frames": 0, "target_scope_resolved_frames": 0} for b in BRANCH_KEYS},
        "by_action": {b: defaultdict(lambda: {"frames": 0, "score_changed": 0, "assignment_changed": 0, "correct": 0, "incorrect": 0, "neutral": 0, "untouched_changed": 0, "candidate_absent": 0}) for b in BRANCH_KEYS},
    }
    examples: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, int]] = set()
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                frame = json.loads(raw)
                event_id = str(frame["event_id"])
                variant_names = frame.get("variants", {})
                frame_key = (event_id, str(frame["sequence"]), int(frame["frame"]))
                if frame_key in seen_keys:
                    raise RuntimeError(f"duplicate frozen N70 frame key {frame_key}")
                seen_keys.add(frame_key)
                totals["frame_rows"] += 1
                totals["events"].add(event_id)
                totals["sequences"].add(str(frame["sequence"]))
                totals["actions"][str(frame["action_type"])] += 1
                if frame.get("runtime_future_gt_used") is not False:
                    totals["runtime_future_gt_true"] += 1
                for variant, data in variant_names.items():
                    mapping = data.get("mapping_audit", {})
                    absent = not bool(mapping.get("target_candidate_present", False))
                    if absent:
                        totals["candidate_absent_frames"] += 1
                    if bool(mapping.get("target_public_assignment_absent", False)):
                        totals["target_public_assignment_absent_frames"] += 1
                    if bool(mapping.get("mapping_uncertain", False)):
                        totals["mapping_uncertain_frames"] += 1
                    base = data.get("assignment_columns")
                    target_row = mapping.get("target_row")
                    target_col = mapping.get("target_public_column")
                    for branch, key in BRANCH_KEYS.items():
                        item = data.get(key, {})
                        treated = item.get("assignment_columns")
                        sidecar = item.get("sidecar", {})
                        stat = totals["branches"][branch]
                        action_stat = totals["by_action"][branch][str(frame["action_type"])]
                        stat["frame_rows"] += 1
                        action_stat["frames"] += 1
                        score_changed = bool(float(sidecar.get("max_abs_score_delta", 0.0)) > 1e-12)
                        assignment_changed = bool(treated != base)
                        target_changed = bool(target_row is not None and target_row < len(base) and target_row < len(treated) and treated[target_row] != base[target_row])
                        baseline_correct = bool(target_row is not None and target_col is not None and int(base[target_row]) == int(target_col))
                        treated_correct = bool(target_row is not None and target_col is not None and int(treated[target_row]) == int(target_col))
                        utility = int(treated_correct) - int(baseline_correct) if not bool(mapping.get("mapping_uncertain", False)) else 0
                        untouched = 0
                        if isinstance(base, list) and isinstance(treated, list):
                            untouched = sum(1 for i, (left, right) in enumerate(zip(base, treated)) if i != target_row and left != right)
                        stat["score_changed"] += int(score_changed)
                        stat["assignment_changed"] += int(assignment_changed)
                        stat["target_assignment_changed"] += int(target_changed)
                        stat["untouched_changed_frames"] += int(untouched > 0)
                        stat["untouched_changed_total"] += int(untouched)
                        stat["candidate_absent_frames"] += int(absent)
                        stat["target_present_frames"] += int(not absent)
                        stat["target_scope_resolved_frames"] += int(bool(mapping.get("target_scope_resolved", False)))
                        action_stat["score_changed"] += int(score_changed)
                        action_stat["assignment_changed"] += int(assignment_changed)
                        action_stat["untouched_changed"] += int(untouched)
                        if utility > 0:
                            stat["correct_changes"] += 1; action_stat["correct"] += 1
                        elif utility < 0:
                            stat["incorrect_changes"] += 1; action_stat["incorrect"] += 1
                        elif assignment_changed and target_changed:
                            stat["neutral_changes"] += 1; action_stat["neutral"] += 1
                        if len(examples) < 48:
                            kind = None
                            if absent and not any(e.get("kind") == "TARGET_CANDIDATE_ABSENT" for e in examples): kind = "TARGET_CANDIDATE_ABSENT"
                            elif score_changed and not assignment_changed and not any(e.get("kind") == "SCORE_CHANGED_NO_CROSSING" for e in examples): kind = "SCORE_CHANGED_NO_CROSSING"
                            elif untouched > 0 and not any(e.get("kind") == "UNTOUCHED_COLLATERAL" for e in examples): kind = "UNTOUCHED_COLLATERAL"
                            elif utility > 0 and not any(e.get("kind") == "CORRECT_CROSSING" for e in examples): kind = "CORRECT_CROSSING"
                            if kind:
                                examples.append({"kind": kind, "event_id": event_id, "sequence": frame["sequence"], "frame": frame["frame"], "event_frame": frame["event_frame"], "variant": variant, "branch": branch, "action_type": frame["action_type"], "target_row": target_row, "target_col": target_col, "mapping_audit": mapping, "base_assignment_columns": base, "treated_assignment_columns": treated, "score_sidecar": {k: sidecar.get(k) for k in ("max_abs_score_delta", "score_cells_changed", "target_column", "target_column_only", "runtime_future_gt_used")}, "candidate_rows_mapping": data.get("candidate_rows_mapping", [])})
    if len(seen_keys) != 2400 or totals["frame_rows"] != 2400:
        raise RuntimeError(f"N70 frame audit expected 2400 unique rows, got {len(seen_keys)} / {totals['frame_rows']}")
    n70_audit = json.loads(N70_AUDIT.read_text(encoding="utf-8"))
    totals["variant_axis_mismatch_frames_retained"] = int(n70_audit.get("artifacts", {}).get("variant_axis_mismatch_frame_count", 0))
    totals["events"] = sorted(totals["events"])
    totals["sequences"] = sorted(totals["sequences"])
    totals["actions"] = dict(sorted(totals["actions"].items()))
    for branch in BRANCH_KEYS:
        for action, value in list(totals["by_action"][branch].items()):
            totals["by_action"][branch][action] = dict(value)
        totals["by_action"][branch] = dict(sorted(totals["by_action"][branch].items()))
    summary = {
        "schema": "N71_STAGE_01_N70_DIAGNOSIS_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"directory": str(N70_ARTIFACTS), "event_file_count": len(files), "n70_integrity_audit": str(N70_AUDIT), "n70_integrity_audit_sha256": sha256(N70_AUDIT)},
        "counts": totals,
        "root_cause_buckets": {
            "candidate_recall_or_absence": {"frames": totals["candidate_absent_frames"], "interpretation": "upstream candidate failure; not assigned to appearance/solver"},
            "public_assignment_absence": {"frames": totals["target_public_assignment_absent_frames"], "interpretation": "explicit frozen absence; no fabricated public mapping"},
            "rowwise_score_to_assignment_boundary": {"interpretation": "N70 changes many scores but only a small number of assignment rows; this is an interface/boundary diagnostic"},
            "untouched_collateral": {"interpretation": "candidate-row assignment changes outside the target row are measured separately"},
            "axis_mismatch": {"frames": totals["variant_axis_mismatch_frames_retained"], "interpretation": "retained N70 diagnostic, not credited as an effect"},
        },
        "provenance": {"interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "runtime_future_gt_used": False, "real_human_tape": False, "real_sam3_full_loop": False, "production_authorized": False},
    }
    atomic_json(OUT / "n70_root_cause_summary.json", summary)
    atomic_jsonl(OUT / "n70_root_cause_examples.jsonl", examples)
    status = {
        "schema": "N71_STAGE_01_STATUS_V1",
        "status": "PASS_N70_ROOT_CAUSE_DIAGNOSIS_CPU_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": summary["input"],
        "outputs": {"summary": str(OUT / "n70_root_cause_summary.json"), "examples": str(OUT / "n70_root_cause_examples.jsonl")},
        "metrics": {"frame_rows": totals["frame_rows"], "events": len(totals["events"]), "sequences": len(totals["sequences"]), "candidate_absent_frames": totals["candidate_absent_frames"], "target_public_assignment_absent_frames": totals["target_public_assignment_absent_frames"], "variant_axis_mismatch_frames_retained": totals["variant_axis_mismatch_frames_retained"], "branches": totals["branches"]},
        "first_actionable_conclusion": "N70's target-column residual is not a complete candidate-by-identity decision: score motion must be separated from target-correct assignment crossing, while target candidate absence and untouched collateral remain independent failure buckets.",
        "next_stage": "N71_STAGE_02_NEW_OFFICIAL_SAM3_CANDIDATE_BRANCH_AND_CACHE_AUDIT",
        "provenance": summary["provenance"],
    }
    atomic_json(STAGE, status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
