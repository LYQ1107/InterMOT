#!/usr/bin/env python3
"""Merge sequence-sharded N32 policy JSONL and validate completeness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.n32_policy_semantics import (
    VISIBLE_UNDEFINED,
    drift_status,
    policy_metric_issues,
    visible_h20_status,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/n31/episode_manifest.json"
WORKERS = ROOT / "outputs/n32/policy_rollouts"
OUT = ROOT / "outputs/n32/policy_rollout_index.json"
POLICIES = ("K0_KEEP_OLD", "K1_APPLY_ENSURE", "K2_PROMPT_THEN_RESTORE")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(*, manifest_path: Path = MANIFEST, worker_dir: Path = WORKERS, worker_count: int = 4, output: Path = OUT) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = manifest.get("episodes", [])
    expected = {str(row["episode_id"]): row for row in episodes}
    if manifest.get("status") != "PASS" or len(expected) != 689 or manifest.get("val25_read") is not False or manifest.get("test_labels_used") is not False:
        raise ValueError("N31 expanded manifest is not the required frozen blind source")
    merged: dict[str, dict[str, Any]] = {}
    duplicate_episode_ids: list[str] = []
    worker_files: list[str] = []
    issues: list[str] = []
    for index in range(int(worker_count)):
        path = worker_dir / f"worker_{index:02d}.jsonl"
        worker_files.append(str(path))
        if not path.is_file():
            issues.append(f"missing worker output: {path}")
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                episode_id = str(row.get("episode_id", ""))
                if episode_id not in expected:
                    issues.append(f"unexpected episode {episode_id} in {path}:{line_number}")
                    continue
                if episode_id in merged:
                    issues.append(f"duplicate episode {episode_id}")
                    duplicate_episode_ids.append(episode_id)
                    continue
                merged[episode_id] = row
    missing = sorted(set(expected) - set(merged))
    issues.extend(f"missing episode {episode_id}" for episode_id in missing)
    grouped: list[dict[str, Any]] = []
    flat: list[dict[str, Any]] = []
    policy_issue_counts: dict[str, int] = {}
    complete_policy_row_count = 0
    unavailable_policy_row_count = 0
    not_run_policy_row_count = 0
    partial_policy_row_count = 0
    legitimately_undefined_visible_window_count = 0
    legitimately_undefined_drift_count = 0
    for episode_id in sorted(expected, key=lambda value: (str(expected[value]["sequence"]), int(expected[value]["correction_frame"]), value)):
        row = merged.get(episode_id)
        if row is None:
            continue
        policies = row.get("policies", {})
        if set(policies) != set(POLICIES):
            issues.append(f"episode {episode_id} does not contain exactly three policies: {sorted(policies)}")
        for name in POLICIES:
            policy = policies.get(name)
            if policy is None:
                issues.append(f"episode {episode_id} is missing policy {name}")
                continue
            policy_issues = policy_metric_issues(policy, require_explicit_visible_status=True)
            if policy.get("available") is not True:
                unavailable_policy_row_count += 1
            if policy.get("status") == "NOT_RUN":
                not_run_policy_row_count += 1
            if policy.get("status") == "PARTIAL":
                partial_policy_row_count += 1
            if not policy_issues:
                complete_policy_row_count += 1
            for reason in policy_issues:
                policy_issue_counts[reason] = policy_issue_counts.get(reason, 0) + 1
                issues.append(f"episode {episode_id} policy {name}: {reason}")
            metrics20 = (policy.get("metrics") or {}).get("20")
            if isinstance(metrics20, dict):
                visible = visible_h20_status(metrics20, require_explicit_undefined=True)
                if visible["status"] == VISIBLE_UNDEFINED and visible["valid"]:
                    legitimately_undefined_visible_window_count += 1
                drift = drift_status(metrics20)
                legitimately_undefined_drift_count += sum(
                    value.startswith("LEGITIMATELY_UNDEFINED_")
                    for value in (drift["mask_area_drift_status"], drift["box_area_drift_status"])
                )
            flat.append({
                "episode_id": episode_id,
                "sequence": str(row.get("sequence", expected[episode_id]["sequence"])),
                "learning_split": str(row.get("learning_split", expected[episode_id]["learning_split"])),
                "policy": name,
                "available": bool(policy.get("available", False)),
                "status": policy.get("status"),
                "feature_vector": policy.get("feature_vector", []),
                "feature_names": policy.get("feature_names", []),
                "temporal_feature_sequence": policy.get("temporal_feature_sequence", []),
                "metrics": policy.get("metrics", {}),
                "reward": policy.get("reward"),
                "action_trace": policy.get("action_trace", {}),
                "prompt_success": policy.get("prompt_success"),
                "fallback_used": policy.get("fallback_used"),
                "rollback_used": policy.get("rollback_used"),
                "mapping_valid": policy.get("mapping_valid"),
                "target_state_present": policy.get("target_state_present"),
                "protected_identity_regression": policy.get("protected_identity_regression"),
                "protected_identity_status": policy.get("protected_identity_status"),
                "current_raw_output_recorded": policy.get("current_raw_output_recorded"),
                "current_delivered_box": policy.get("current_delivered_box"),
                "future_frame_count": policy.get("future_frame_count"),
                "failure": policy.get("failure"),
                "elapsed_seconds": policy.get("elapsed_seconds"),
                "future_gt_used_for_selection": policy.get("future_gt_used_for_selection"),
                "future_gt_used_for_posthoc_evaluation": policy.get("future_gt_used_for_posthoc_evaluation"),
            })
        grouped.append(row)
    flat_keys = [(str(row.get("episode_id")), str(row.get("policy"))) for row in flat]
    duplicate_policy_keys = sorted({key for key in flat_keys if flat_keys.count(key) > 1})
    complete = (
        len(grouped) == len(expected)
        and len(flat) == len(expected) * len(POLICIES)
        and len(set(flat_keys)) == len(flat_keys)
        and complete_policy_row_count == len(expected) * len(POLICIES)
        and unavailable_policy_row_count == 0
        and not_run_policy_row_count == 0
        and partial_policy_row_count == 0
        and not issues
        and all(set(row.get("policies", {})) == set(POLICIES) for row in grouped)
    )
    result = {
        "protocol": "N32-C-POLICY-ROLLOUT-INDEX",
        "status": "PASS" if complete else "FAIL",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha(manifest_path),
        "worker_files": worker_files,
        "episode_count_expected": len(expected),
        "episode_count_merged": len(grouped),
        "policy_row_count_expected": len(expected) * len(POLICIES),
        "policy_row_count_merged": len(flat),
        "duplicate_policy_key_count": len(duplicate_policy_keys),
        "duplicate_policy_keys": [f"{key[0]}|{key[1]}" for key in duplicate_policy_keys],
        "complete_policy_row_count": complete_policy_row_count,
        "unavailable_policy_row_count": unavailable_policy_row_count,
        "not_run_policy_row_count": not_run_policy_row_count,
        "partial_policy_row_count": partial_policy_row_count,
        "policy_issue_counts": policy_issue_counts,
        "legitimately_undefined_visible_window_count": legitimately_undefined_visible_window_count,
        "legitimately_undefined_drift_count": legitimately_undefined_drift_count,
        "duplicate_episode_count": len(duplicate_episode_ids),
        "duplicate_episode_ids": sorted(set(duplicate_episode_ids)),
        "missing_episode_count": len(missing),
        "missing_episode_ids": missing,
        "issues": issues,
        "episodes": grouped,
        "rows": flat,
        "all_three_policies_per_episode": complete,
        "all_failures_retained": all("failure" in row or row.get("status") in {"PASS", "PARTIAL"} for row in grouped),
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_selection": False,
    }
    _write(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--worker-dir", type=Path, default=WORKERS)
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    result = run(manifest_path=args.manifest, worker_dir=args.worker_dir, worker_count=args.worker_count, output=args.output)
    print(json.dumps({key: result[key] for key in ("protocol", "status", "episode_count_expected", "episode_count_merged", "policy_row_count_merged", "missing_episode_count", "issues")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
