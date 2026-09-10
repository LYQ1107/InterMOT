#!/usr/bin/env python3
"""Audit accepted PCTIS assignment changes in the frozen 516-event corpus.

This is a metadata-only audit.  It never opens GT and does not re-run the
scorer.  The source metadata contains the already sealed selection and exact
solver results from N72R11R4; those are counted without relabelling them as
new experimental outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT / "outputs/N72R11R4/exact_onpolicy_v3")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/N72R13/intervention_opportunity_audit.json")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    manifest_path = source_root / "corpus_manifest.json"
    try:
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "PASS_N72R11R4_EXACT_SOLVER_ON_POLICY_CORPUS":
            raise RuntimeError(f"source corpus is not frozen PASS: {manifest.get('status')}")
        if int(manifest.get("event_count", -1)) != 516:
            raise RuntimeError(f"expected 516 source interactions, found {manifest.get('event_count')}")
        rows: list[dict[str, Any]] = []
        by_action: Counter[str] = Counter()
        by_split: Counter[str] = Counter()
        by_sequence: Counter[str] = Counter()
        by_source: Counter[str] = Counter()
        accepted = 0
        changed = 0
        opportunities = 0
        event_ids: set[str] = set()
        opportunity_rows: list[dict[str, Any]] = []
        for split in ("train", "validation"):
            metadata_path = source_root / f"{split}_metadata.jsonl"
            if not metadata_path.is_file():
                raise FileNotFoundError(metadata_path)
            with metadata_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise TypeError(f"{metadata_path}:{line_number} is not an object")
                    for flag in ("runtime_future_gt_used", "runtime_gt_read", "posthoc_gt_used", "public_id_inference"):
                        if row.get(flag) is not False:
                            raise RuntimeError(f"{metadata_path}:{line_number} violates {flag}")
                    event_id = str(row["event_id"])
                    event_ids.add(event_id)
                    selected = row.get("selection", {}).get("selected_candidate_uid")
                    selected_score = row.get("selection", {}).get("selected_score")
                    selected_margin = row.get("selection", {}).get("best_minus_second_margin")
                    if selected is not None:
                        _finite(selected_score, f"{event_id}:{row.get('frame')}:selected_score")
                        _finite(selected_margin, f"{event_id}:{row.get('frame')}:selected_margin")
                    base_uid = row.get("base_solver_target_uid")
                    assignment = row.get("assignment")
                    if not isinstance(assignment, Mapping):
                        raise RuntimeError(f"{event_id}:{row.get('frame')} missing assignment audit")
                    proposal_uid = assignment.get("target_assigned_candidate_uid")
                    accepted_here = selected is not None
                    changed_here = str(proposal_uid) != str(base_uid)
                    effective = bool(accepted_here and changed_here)
                    candidate_uids = [str(value) for value in row.get("candidate_uids", [])]
                    candidate_sources = [str(value) for value in row.get("candidate_sources", [])]
                    if len(candidate_uids) != len(candidate_sources) or len(candidate_uids) != len(set(candidate_uids)):
                        raise RuntimeError(f"{event_id}:{row.get('frame')} candidate axis is not unique")
                    source = None
                    if selected is not None and selected in candidate_uids:
                        source = candidate_sources[candidate_uids.index(selected)]
                    source = "UNKNOWN" if source is None else source
                    accepted += int(accepted_here)
                    changed += int(changed_here)
                    opportunities += int(effective)
                    by_action[str(row["action_type"])] += int(effective)
                    by_split[split] += int(effective)
                    by_sequence[str(row["sequence"])] += int(effective)
                    by_source[source] += int(effective)
                    if effective:
                        opportunity_rows.append({
                            "event_id": event_id,
                            "sequence": str(row["sequence"]),
                            "split": split,
                            "action_type": str(row["action_type"]),
                            "event_frame": int(row["event_frame"]),
                            "frame": int(row["frame"]),
                            "selected_candidate_uid": str(selected),
                            "selected_candidate_source": source,
                            "selected_score": float(selected_score),
                            "selected_margin": float(selected_margin),
                            "base_target_uid": None if base_uid is None else str(base_uid),
                            "proposal_target_uid": None if proposal_uid is None else str(proposal_uid),
                            "runtime_future_gt_used": False,
                            "runtime_gt_read": False,
                            "posthoc_gt_used": False,
                        })
                    rows.append({
                        "event_id": event_id,
                        "sequence": str(row["sequence"]),
                        "split": split,
                        "action_type": str(row["action_type"]),
                        "frame": int(row["frame"]),
                        "selection_accepted": accepted_here,
                        "assignment_changed": changed_here,
                        "effective_intervention_opportunity": effective,
                        "selected_candidate_source": source,
                    })
        if len(event_ids) != 516:
            raise RuntimeError(f"metadata event count mismatch: {len(event_ids)}")
        payload = {
            "schema_version": "N72R13_INTERVENTION_OPPORTUNITY_AUDIT_V1",
            "status": "PASS_N72R13_INTERVENTION_OPPORTUNITY_AUDIT",
            "created_at_utc": now_utc(),
            "source_corpus_manifest": str(manifest_path),
            "source_corpus_manifest_sha256": sha256_file(manifest_path),
            "source_corpus_status": manifest["status"],
            "event_count": len(event_ids),
            "total_frame_rows": len(rows),
            "selection_accepted_count": accepted,
            "assignment_changed_count": changed,
            "true_intervention_opportunity_count": opportunities,
            "definition": {
                "selection_accepted": "sealed PCTIS selection.selected_candidate_uid is non-null",
                "assignment_changed": "sealed proposal target UID differs from sealed BASE target UID",
                "true_opportunity": "selection_accepted AND assignment_changed",
                "future_gt_used_for_count": False,
            },
            "by_source": dict(sorted(by_source.items())),
            "by_action": dict(sorted(by_action.items())),
            "by_split": dict(sorted(by_split.items())),
            "by_sequence": dict(sorted(by_sequence.items())),
            "opportunities": opportunity_rows,
            "audit_rows_sha256": hashlib.sha256(
                json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            ).hexdigest(),
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "real_human_evidence": False,
            "not_real_human_evidence": True,
            "historical_outputs_modified": False,
        }
        atomic_json(output, payload)
        atomic_json(
            output.parent / "stage_02_status.json",
            {
                "schema_version": "N72R13_STAGE_STATUS_V1",
                "stage": "N72R13-02-INTERVENTION-OPPORTUNITY-AUDIT",
                "status": "PASS_N72R13_INTERVENTION_OPPORTUNITY_AUDIT",
                "audit": str(output),
                "event_count": len(event_ids),
                "total_frame_rows": len(rows),
                "selection_accepted_count": accepted,
                "assignment_changed_count": changed,
                "true_intervention_opportunity_count": opportunities,
                "created_at_utc": now_utc(),
                "runtime_future_gt_used": False,
                "real_human_evidence": False,
                "historical_outputs_modified": False,
            },
        )
        print(json.dumps({"status": payload["status"], "events": len(event_ids), "rows": len(rows), "opportunities": opportunities, "output": str(output)}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = output.with_name(output.stem + "_failure.json")
        payload = {
            "schema_version": "N72R13_INTERVENTION_OPPORTUNITY_AUDIT_FAILURE_V1",
            "status": "FAIL_N72R13_INTERVENTION_OPPORTUNITY_AUDIT",
            "created_at_utc": now_utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": __import__("traceback").format_exc(),
            "historical_outputs_modified": False,
        }
        atomic_json(failure, payload)
        print(json.dumps({"status": payload["status"], "failure": str(failure), "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
