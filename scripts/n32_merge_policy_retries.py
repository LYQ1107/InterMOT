#!/usr/bin/env python3
"""Overlay successful policy retries without deleting original failures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.n32_policy_semantics import policy_metric_issues


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/n31/episode_manifest.json"
WORKER_DIR = ROOT / "outputs/n32/policy_rollouts"
RETRY_MANIFEST = WORKER_DIR / "retry_manifest.json"
RETRY_DIR = WORKER_DIR / "policy_retries"
MERGED_DIR = WORKER_DIR / "retry_merged_workers"
OUT = ROOT / "outputs/n32/retry_merge.json"
INDEX = ROOT / "outputs/n32/policy_rollout_index.json"
POLICIES = ("K0_KEEP_OLD", "K1_APPLY_ENSURE", "K2_PROMPT_THEN_RESTORE")


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _artifact_path(item: dict[str, Any], retry_dir: Path) -> Path:
    digest = hashlib.sha256(f"{item['episode_id']}|{item['policy']}".encode("utf-8")).hexdigest()
    return retry_dir / f"{digest}.json"


def _load_workers(worker_dir: Path, worker_count: int) -> tuple[dict[str, dict[str, Any]], dict[str, int], list[str]]:
    rows: dict[str, dict[str, Any]] = {}
    owners: dict[str, int] = {}
    issues: list[str] = []
    for worker_index in range(int(worker_count)):
        path = worker_dir / f"worker_{worker_index:02d}.jsonl"
        if not path.is_file():
            issues.append(f"missing source worker file: {path}")
            continue
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                episode_id = str(row.get("episode_id", ""))
                if not episode_id:
                    issues.append(f"empty episode_id: {path}:{line_number}")
                    continue
                if episode_id in rows:
                    issues.append(f"duplicate source episode_id: {episode_id}")
                    continue
                rows[episode_id] = row
                owners[episode_id] = worker_index
    return rows, owners, issues


def _load_retry_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("retry_items_unique") is not True:
        raise ValueError("retry manifest is not a unique PASS artifact")
    return list(payload.get("items", []))


def _compact_retry_evidence(
    artifact: dict[str, Any],
    item: dict[str, Any],
    *,
    retry_dir: Path,
    replaced: bool,
) -> dict[str, Any]:
    row = artifact.get("policy_row") or {}
    return {
        "episode_id": str(item["episode_id"]),
        "policy": str(item["policy"]),
        "artifact": str(_artifact_path(item, retry_dir)),
        "attempt": artifact.get("attempt"),
        "status": "REPLACED" if replaced else "FAILED_NOT_REPLACED",
        "retry_status": artifact.get("status"),
        "strict_complete": artifact.get("strict_complete"),
        "failure": artifact.get("failure"),
        "strict_failure_reasons": artifact.get("strict_failure_reasons", []),
        "retry_policy_status": row.get("status"),
        "retry_policy_available": row.get("available"),
        "retry_reward": row.get("reward"),
        "retry_h20": {
            key: ((row.get("metrics") or {}).get("20") or {}).get(key)
            for key in (
                "mean_box_iou_visible",
                "missing_prediction_rate_visible",
                "mask_area_drift",
                "mask_area_sample_count",
                "box_area_drift_proxy",
            )
        },
    }


def run(
    *,
    manifest_path: Path = MANIFEST,
    worker_dir: Path = WORKER_DIR,
    retry_manifest_path: Path = RETRY_MANIFEST,
    retry_dir: Path = RETRY_DIR,
    merged_dir: Path = MERGED_DIR,
    output: Path = OUT,
    index_output: Path = INDEX,
    worker_count: int = 4,
) -> dict[str, Any]:
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {str(row["episode_id"]): row for row in frozen.get("episodes", [])}
    issues: list[str] = []
    if frozen.get("status") != "PASS" or len(expected) != 689:
        issues.append("frozen manifest is not exactly 689 PASS episodes")
    source_rows, owners, source_issues = _load_workers(worker_dir, worker_count)
    issues.extend(source_issues)
    if set(source_rows) != set(expected):
        issues.append(f"source episode set mismatch: expected={len(expected)} actual={len(source_rows)}")
    retry_items = _load_retry_items(retry_manifest_path)
    item_keys = [(str(item["episode_id"]), str(item["policy"])) for item in retry_items]
    if len(set(item_keys)) != len(item_keys):
        issues.append("retry manifest contains duplicate (episode_id, policy) keys")
    item_by_key = dict(zip(item_keys, retry_items))
    artifacts: dict[tuple[str, str], dict[str, Any]] = {}
    missing_artifacts: list[tuple[str, str]] = []
    invalid_artifacts: list[tuple[str, str]] = []
    for key, item in item_by_key.items():
        path = _artifact_path(item, retry_dir)
        if not path.is_file():
            missing_artifacts.append(key)
            continue
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            invalid_artifacts.append(key)
            continue
        if (str(artifact.get("episode_id")), str(artifact.get("policy"))) != key:
            issues.append(f"retry artifact key mismatch: {path}")
            invalid_artifacts.append(key)
            continue
        artifacts[key] = artifact
    if missing_artifacts:
        issues.append(f"missing retry artifacts: {len(missing_artifacts)}")
    if invalid_artifacts:
        issues.append(f"invalid retry artifacts: {len(invalid_artifacts)}")

    merged_rows = {episode_id: copy.deepcopy(row) for episode_id, row in source_rows.items()}
    replaced = 0
    failed_not_replaced = 0
    for key, item in item_by_key.items():
        episode_id, policy_name = key
        row = merged_rows.get(episode_id)
        artifact = artifacts.get(key)
        if row is None or artifact is None:
            failed_not_replaced += 1
            continue
        policy_row = artifact.get("policy_row")
        strict_pass = (
            artifact.get("status") == "PASS"
            and artifact.get("strict_complete") is True
            and not artifact.get("strict_failure_reasons")
            and isinstance(policy_row, dict)
            and not policy_metric_issues(policy_row, require_explicit_visible_status=True)
        )
        row.setdefault("policy_retry_history", []).append(
            _compact_retry_evidence(artifact, item, retry_dir=retry_dir, replaced=strict_pass)
        )
        if strict_pass:
            row.setdefault("policies", {})[policy_name] = policy_row
            replaced += 1
        else:
            failed_not_replaced += 1

    # Write a derived worker set.  The original JSONL remains untouched, so
    # every pre-retry failure is recoverable and auditable by source line.
    merged_dir.mkdir(parents=True, exist_ok=True)
    for worker_index in range(int(worker_count)):
        path = merged_dir / f"worker_{worker_index:02d}.jsonl"
        rows = [
            merged_rows[episode_id]
            for episode_id in sorted(merged_rows)
            if owners.get(episode_id) == worker_index
        ]
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        temporary.replace(path)

    # Verify uniqueness before asking the canonical merger to consume the
    # derived workers.
    after_rows, _, after_issues = _load_workers(merged_dir, worker_count)
    issues.extend(after_issues)
    if set(after_rows) != set(expected) or len(after_rows) != 689:
        issues.append(f"post-retry derived worker set is not exactly 689 unique episodes: {len(after_rows)}")

    canonical: dict[str, Any] | None = None
    if not any("duplicate source" in issue for issue in issues):
        from scripts.n32_merge_policy_rollouts import run as merge_policy_rollouts

        canonical = merge_policy_rollouts(
            manifest_path=manifest_path,
            worker_dir=merged_dir,
            worker_count=worker_count,
            output=index_output,
        )

    result = {
        "protocol": "N32-C-POLICY-LEVEL-RETRY-MERGE",
        "status": "PASS" if canonical is not None and canonical.get("status") == "PASS" and not issues else "FAIL",
        "source_episode_count": len(source_rows),
        "source_duplicate_or_set_issues": source_issues,
        "retry_item_count": len(retry_items),
        "retry_artifact_count": len(artifacts),
        "missing_retry_artifact_count": len(missing_artifacts),
        "invalid_retry_artifact_count": len(invalid_artifacts),
        "successful_policy_replacements": replaced,
        "failed_or_unreplaced_policy_retries": failed_not_replaced,
        "derived_episode_count": len(after_rows),
        "derived_duplicate_count": max(0, sum(1 for _ in after_rows) - len(set(after_rows))),
        "merged_worker_dir": str(merged_dir),
        "canonical_index": None if canonical is None else {
            "path": str(index_output),
            "status": canonical.get("status"),
            "episode_count_merged": canonical.get("episode_count_merged"),
            "policy_row_count_merged": canonical.get("policy_row_count_merged"),
            "complete_policy_row_count": canonical.get("complete_policy_row_count"),
            "unavailable_policy_row_count": canonical.get("unavailable_policy_row_count"),
            "legitimately_undefined_drift_count": canonical.get("legitimately_undefined_drift_count"),
        },
        "issues": issues,
        "failure_evidence_preserved_in_original_workers": True,
        "val25_read": False,
        "test_labels_used": False,
        "future_gt_used_for_selection": False,
    }
    _write(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--worker-dir", type=Path, default=WORKER_DIR)
    parser.add_argument("--retry-manifest", type=Path, default=RETRY_MANIFEST)
    parser.add_argument("--retry-dir", type=Path, default=RETRY_DIR)
    parser.add_argument("--merged-dir", type=Path, default=MERGED_DIR)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--index-output", type=Path, default=INDEX)
    parser.add_argument("--worker-count", type=int, default=4)
    args = parser.parse_args()
    result = run(
        manifest_path=args.manifest,
        worker_dir=args.worker_dir,
        retry_manifest_path=args.retry_manifest,
        retry_dir=args.retry_dir,
        merged_dir=args.merged_dir,
        output=args.output,
        index_output=args.index_output,
        worker_count=args.worker_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
