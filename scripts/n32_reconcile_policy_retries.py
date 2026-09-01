#!/usr/bin/env python3
"""CPU-only reconciliation of the N32 policy-level retry artifacts.

The retry supervisor's ``PARTIAL`` summary is not itself a policy result.  This
script audits the frozen ``(episode_id, policy)`` manifest keys, excludes the
supervisor summary JSON, and derives a new artifact directory.  It changes no
model metric and never overwrites the raw retry artifacts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from scripts.n32_policy_semantics import (
    VISIBLE_UNDEFINED,
    drift_status,
    finite,
    policy_metric_issues,
    visible_h20_status,
    zero_visible_completion_class,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/n32/policy_rollouts/retry_manifest.json"
RETRY_DIR = ROOT / "outputs/n32/policy_rollouts/policy_retries_attempt2"
OUT_DIR = ROOT / "outputs/n32/policy_rollouts/policy_reconciled_attempt2"
AUDIT_OUT = ROOT / "outputs/n32/policy_rollouts/retry_semantic_reconciliation_attempt2.json"
SUPERVISOR_NAME = "retry_supervisor_attempt_2.json"
POLICIES = ("K0_KEEP_OLD", "K1_APPLY_ENSURE", "K2_PROMPT_THEN_RESTORE")
VISIBLE_REASONS = {
    "h20_mean_box_iou_visible_nonfinite",
    "h20_missing_prediction_rate_visible_nonfinite",
}
DEFAULT_SKIPPED_KEY = "n31_expanded_dancetrack0015:1122:6|K1_APPLY_ENSURE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_path(retry_dir: Path, episode_id: str, policy: str) -> Path:
    key = f"{episode_id}|{policy}".encode("utf-8")
    return retry_dir / f"{hashlib.sha256(key).hexdigest()}.json"


def _write_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _key(artifact: Mapping[str, Any]) -> tuple[str, str]:
    return str(artifact.get("episode_id", "")), str(artifact.get("policy", ""))


def _h20(policy_row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    metrics = policy_row.get("metrics")
    value = metrics.get("20") if isinstance(metrics, Mapping) else None
    return value if isinstance(value, Mapping) else None


def _classify(
    artifact: Mapping[str, Any],
    *,
    skipped_keys: set[str],
) -> tuple[str, str | None, list[str], dict[str, Any]]:
    """Return class, B subtype, issues, and semantic details for one artifact."""
    policy_row = artifact.get("policy_row")
    if not isinstance(policy_row, Mapping):
        return "A", None, ["policy_row_missing"], {}
    h20 = _h20(policy_row)
    details: dict[str, Any] = {
        "policy_status": policy_row.get("status"),
        "available": policy_row.get("available"),
        "reward_finite": finite(policy_row.get("reward")),
        "artifact_status": artifact.get("status"),
        "raw_strict_complete": artifact.get("strict_complete"),
        "raw_strict_failure_reasons": list(artifact.get("strict_failure_reasons") or []),
    }
    if h20 is not None:
        details["visible"] = visible_h20_status(h20)
        details["drift"] = drift_status(h20)
    base_issues = policy_metric_issues(policy_row)
    raw_reasons = list(artifact.get("strict_failure_reasons") or [])
    raw_strict_pass = (
        artifact.get("status") == "PASS"
        and artifact.get("strict_complete") is True
        and not raw_reasons
        and not base_issues
    )
    key_string = f"{artifact.get('episode_id', '')}|{artifact.get('policy', '')}"
    if raw_strict_pass:
        return (
            "SKIPPED_EXISTING_PASS" if key_string in skipped_keys else "RAW_STRICT_PASS",
            None,
            [],
            details,
        )

    b_class, b_issues = zero_visible_completion_class(policy_row)
    b_eligible = (
        b_class in {"B1", "B2"}
        and artifact.get("status") in {"PASS", "FAIL"}
        and bool(raw_reasons)
        and set(raw_reasons).issubset(VISIBLE_REASONS)
    )
    if b_eligible:
        return b_class, b_class, [], details

    issues = list(dict.fromkeys(raw_reasons + base_issues + b_issues))
    if not issues:
        issues = ["artifact_not_strict_complete"]
    return "A", None, issues, details


def _reconciled_artifact(
    artifact: Mapping[str, Any],
    *,
    source_path: Path,
    classification: str,
    subtype: str | None,
    issues: list[str],
    details: Mapping[str, Any],
) -> dict[str, Any]:
    derived = copy.deepcopy(dict(artifact))
    policy_row = derived.get("policy_row")
    if subtype in {"B1", "B2"} and isinstance(policy_row, dict):
        metrics = policy_row.get("metrics")
        h20 = metrics.get("20") if isinstance(metrics, dict) else None
        if isinstance(h20, dict):
            # Nulls are intentionally retained.  This only records why they
            # are valid; it does not synthesize a measurement.
            h20["visible_metric_status"] = VISIBLE_UNDEFINED
            h20["visible_metric_semantics"] = VISIBLE_UNDEFINED
        derived["status"] = "PASS"
        derived["strict_complete"] = True
        derived["strict_failure_reasons"] = [
            reason
            for reason in list(derived.get("strict_failure_reasons") or [])
            if reason not in VISIBLE_REASONS
        ]

    action_trace = policy_row.get("action_trace") if isinstance(policy_row, Mapping) else None
    derived["reconciliation"] = {
        "applied": subtype in {"B1", "B2"},
        "protocol": "N32-C-ZERO-VISIBLE-H20-SEMANTIC-RECONCILIATION",
        "classification": classification,
        "subtype": subtype,
        "source_artifact": str(source_path),
        "source_artifact_sha256": _sha256(source_path),
        "raw_status": artifact.get("status"),
        "raw_strict_complete": artifact.get("strict_complete"),
        "raw_strict_failure_reasons": list(artifact.get("strict_failure_reasons") or []),
        "policy_outcome_preserved": True,
        "action_trace_preserved": True,
        "action_trace_failure_preserved": None if not isinstance(action_trace, Mapping) else action_trace.get("failure"),
        "policy_failure_preserved": None if not isinstance(policy_row, Mapping) else policy_row.get("failure"),
        "reward_preserved": None if not isinstance(policy_row, Mapping) else policy_row.get("reward"),
        "null_visible_metrics_preserved": subtype in {"B1", "B2"},
        "issues_before_reconciliation": list(issues),
        "semantic_details": copy.deepcopy(dict(details)),
    }
    return derived


def run(
    *,
    manifest_path: Path = MANIFEST,
    retry_dir: Path = RETRY_DIR,
    output_dir: Path = OUT_DIR,
    audit_output: Path = AUDIT_OUT,
    skipped_keys: tuple[str, ...] = (DEFAULT_SKIPPED_KEY,),
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = list(manifest.get("items", []))
    expected_keys = [(str(item.get("episode_id")), str(item.get("policy"))) for item in items]
    expected_set = set(expected_keys)
    expected_key_strings = {f"{episode_id}|{policy}" for episode_id, policy in expected_set}
    skipped_set = set(skipped_keys)
    issues: list[str] = []
    if manifest.get("status") != "PASS" or manifest.get("retry_items_unique") is not True:
        issues.append("retry manifest is not a unique PASS artifact")
    if len(items) != 268 or len(expected_set) != 268:
        issues.append(f"retry manifest is not exactly 268 unique items: {len(items)}/{len(expected_set)}")
    if manifest.get("val25_read") is not False or manifest.get("test_labels_used") is not False or manifest.get("future_gt_used_for_selection") is not False:
        issues.append("retry manifest violates the blind/causal boundary")
    if not skipped_set.issubset(expected_key_strings):
        issues.append("configured skipped-existing key is not in the retry manifest")

    artifact_files = sorted(path for path in retry_dir.glob("*.json") if path.name != SUPERVISOR_NAME)
    excluded = retry_dir / SUPERVISOR_NAME
    if excluded.is_file():
        excluded_files = [str(excluded)]
    else:
        excluded_files = []
        issues.append(f"missing excluded supervisor summary: {excluded}")

    by_key: dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    invalid_files: list[str] = []
    for path in artifact_files:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            invalid_files.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        if not isinstance(value, dict):
            invalid_files.append(f"{path}: top-level JSON is not an object")
            continue
        by_key[_key(value)].append((path, value))
    if invalid_files:
        issues.extend(invalid_files)

    duplicate_keys = sorted(key for key, values in by_key.items() if key in expected_set and len(values) > 1)
    unexpected_keys = sorted(key for key in by_key if key not in expected_set)
    missing_keys = sorted(expected_set - set(by_key))
    if duplicate_keys:
        issues.append(f"duplicate retry artifact keys: {len(duplicate_keys)}")
    if unexpected_keys:
        issues.append(f"unexpected retry artifact keys: {len(unexpected_keys)}")
    if missing_keys:
        issues.append(f"missing retry artifact keys: {len(missing_keys)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    drift_counts = Counter()
    classifications: dict[tuple[str, str], dict[str, Any]] = {}
    b1_keys: list[str] = []
    b2_keys: list[str] = []
    a_keys: list[str] = []
    post_gate_issues: list[str] = []
    for key in expected_keys:
        key_string = f"{key[0]}|{key[1]}"
        records = by_key.get(key, [])
        if len(records) != 1:
            counts["MISSING_OR_DUPLICATE"] += 1
            classification = "A"
            subtype = None
            class_issues = ["artifact_missing"] if not records else ["duplicate_artifact_key"]
            details: dict[str, Any] = {}
            if not records:
                a_keys.append(key_string)
            classifications[key] = {"classification": classification, "subtype": subtype, "issues": class_issues}
            continue
        source_path, artifact = records[0]
        classification, subtype, class_issues, details = _classify(artifact, skipped_keys=skipped_set)
        counts[classification] += 1
        if subtype == "B1":
            b1_keys.append(key_string)
        elif subtype == "B2":
            b2_keys.append(key_string)
        elif classification == "A":
            a_keys.append(key_string)
        h20 = _h20(artifact.get("policy_row") or {}) if isinstance(artifact.get("policy_row"), Mapping) else None
        if h20 is not None:
            drift = drift_status(h20)
            drift_counts[drift["mask_area_drift_status"]] += 1
            drift_counts[drift["box_area_drift_status"]] += 1
        derived = _reconciled_artifact(
            artifact,
            source_path=source_path,
            classification=classification,
            subtype=subtype,
            issues=class_issues,
            details=details,
        )
        destination = _artifact_path(output_dir, key[0], key[1])
        _write_atomic(destination, derived)
        classifications[key] = {
            "classification": classification,
            "subtype": subtype,
            "source": str(source_path),
            "derived": str(destination),
            "issues": class_issues,
        }

        post_policy = derived.get("policy_row")
        if not isinstance(post_policy, Mapping):
            post_gate_issues.append(f"{key_string}: policy_row_missing_after_reconciliation")
            continue
        post_issues = policy_metric_issues(post_policy, require_explicit_visible_status=True)
        if derived.get("status") != "PASS" or derived.get("strict_complete") is not True or derived.get("strict_failure_reasons"):
            post_issues.append("derived_artifact_not_strict_pass")
        if post_issues:
            post_gate_issues.append(f"{key_string}: {','.join(dict.fromkeys(post_issues))}")

    if len(artifact_files) != 268:
        issues.append(f"policy artifact file count excluding supervisor is {len(artifact_files)}, expected 268")
    if len(set(expected_keys)) != len(expected_keys):
        issues.append("manifest key uniqueness failed")
    issues.extend(post_gate_issues)

    unavailable = 0
    not_run = 0
    partial = 0
    policy_status_counts = Counter()
    for values in by_key.values():
        if len(values) != 1:
            continue
        artifact = values[0][1]
        policy_row = artifact.get("policy_row") or {}
        status = policy_row.get("status")
        policy_status_counts[str(status)] += 1
        if policy_row.get("available") is not True:
            unavailable += 1
        if status == "NOT_RUN" or artifact.get("status") == "NOT_RUN":
            not_run += 1
        if status == "PARTIAL" or artifact.get("status") == "PARTIAL":
            partial += 1

    supervisor: dict[str, Any] = {}
    if excluded.is_file():
        try:
            supervisor = json.loads(excluded.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(f"unreadable supervisor summary: {type(exc).__name__}: {exc}")

    skipped_present = [key for key in skipped_set if key in expected_key_strings]
    skipped_valid = [
        key for key in skipped_present
        if classifications.get(tuple(key.split("|", 1)), {}).get("classification") == "SKIPPED_EXISTING_PASS"
    ]
    if len(skipped_valid) != len(skipped_set):
        issues.append(f"configured skipped-existing PASS count mismatch: {len(skipped_valid)}/{len(skipped_set)}")

    result = {
        "protocol": "N32-C-ZERO-VISIBLE-H20-SEMANTIC-RECONCILIATION",
        "status": "PASS" if not issues else "FAIL",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "input_retry_dir": str(retry_dir),
        "output_reconciled_dir": str(output_dir),
        "excluded_non_policy_artifacts": excluded_files,
        "manifest_item_count": len(items),
        "manifest_unique_key_count": len(expected_set),
        "policy_artifact_file_count_excluding_supervisor": len(artifact_files),
        "artifact_unique_key_count": len(set(by_key)),
        "duplicate_key_count": len(duplicate_keys),
        "duplicate_keys": [f"{key[0]}|{key[1]}" for key in duplicate_keys],
        "missing_key_count": len(missing_keys),
        "missing_keys": [f"{key[0]}|{key[1]}" for key in missing_keys],
        "unexpected_key_count": len(unexpected_keys),
        "unexpected_keys": [f"{key[0]}|{key[1]}" for key in unexpected_keys],
        "classification_counts": dict(counts),
        "raw_strict_pass_count": counts["RAW_STRICT_PASS"] + counts["SKIPPED_EXISTING_PASS"],
        "raw_strict_pass_non_skipped_count": counts["RAW_STRICT_PASS"],
        "skipped_existing_pass_count": counts["SKIPPED_EXISTING_PASS"],
        "b1_legal_zero_visible_count": counts["B1"],
        "b2_legal_zero_visible_count": counts["B2"],
        "real_a_count": counts["A"],
        "b1_keys": sorted(b1_keys),
        "b2_keys": sorted(b2_keys),
        "real_a_keys": sorted(a_keys),
        "unavailable_policy_count": unavailable,
        "not_run_policy_count": not_run,
        "partial_policy_count": partial,
        "policy_status_counts": dict(policy_status_counts),
        "legitimately_undefined_visible_window_count": counts["B1"] + counts["B2"],
        "legitimately_undefined_drift_metric_count": sum(
            count
            for status, count in drift_counts.items()
            if status.startswith("LEGITIMATELY_UNDEFINED_")
        ),
        "drift_status_counts": dict(drift_counts),
        "supervisor_summary": {
            "path": str(excluded),
            "status": supervisor.get("status"),
            "requested_item_count": supervisor.get("requested_item_count"),
            "completed_pass": supervisor.get("completed_pass"),
            "skipped_existing_pass": supervisor.get("skipped_existing_pass"),
            "failed_or_incomplete": supervisor.get("failed_or_incomplete"),
            "explicitly_not_policy_artifact": True,
        },
        "post_reconciliation_gate": {
            "expected_policy_rows": 268,
            "derived_policy_rows": len(expected_keys),
            "all_policy_rows_strict_pass": not post_gate_issues and len(expected_keys) == 268,
            "duplicate_count": len(duplicate_keys),
            "missing_count": len(missing_keys),
            "unavailable_count": unavailable,
            "not_run_count": not_run,
            "partial_count": partial,
            "real_a_count": counts["A"],
            "legally_undefined_visible_count": counts["B1"] + counts["B2"],
        },
        "issues": issues,
        "source_artifacts_untouched": True,
        "null_metrics_filled": False,
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_selection": False,
    }
    _write_atomic(audit_output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--retry-dir", type=Path, default=RETRY_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--audit-output", type=Path, default=AUDIT_OUT)
    parser.add_argument("--skipped-key", action="append", default=[DEFAULT_SKIPPED_KEY])
    args = parser.parse_args()
    result = run(
        manifest_path=args.manifest,
        retry_dir=args.retry_dir,
        output_dir=args.output_dir,
        audit_output=args.audit_output,
        skipped_keys=tuple(args.skipped_key),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
