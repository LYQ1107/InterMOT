#!/usr/bin/env python3
"""Build the N32 policy-level retry manifest from the persisted worker JSONL.

The retry unit is exactly ``(episode_id, policy)``.  A policy is retryable when
its persisted outcome is not a complete PASS, is unavailable, or lacks finite
reward/H20 values.  Successful policy rows are deliberately not included and
are therefore preserved by the retry/merge path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.n32_policy_semantics import (
    VISIBLE_UNDEFINED,
    drift_status as _drift_status,
    finite as _finite,
    visible_h20_status,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/n31/episode_manifest.json"
WORKER_DIR = ROOT / "outputs/n32/policy_rollouts"
OUT = WORKER_DIR / "retry_manifest.json"
POLICIES = ("K0_KEEP_OLD", "K1_APPLY_ENSURE", "K2_PROMPT_THEN_RESTORE")
H20_FIELDS = (
    "mean_box_iou_visible",
    "missing_prediction_rate_visible",
    "mask_area_drift",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_worker_rows(worker_dir: Path, worker_count: int) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    rows: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for worker_index in range(int(worker_count)):
        path = worker_dir / f"worker_{worker_index:02d}.jsonl"
        if not path.is_file():
            issues.append(f"missing worker file: {path}")
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    issues.append(f"invalid JSON {path}:{line_number}: {exc}")
                    continue
                episode_id = str(row.get("episode_id", ""))
                if not episode_id:
                    issues.append(f"empty episode_id {path}:{line_number}")
                    continue
                if episode_id in rows:
                    issues.append(f"duplicate episode_id {episode_id}: {sources[episode_id]['path']} and {path}:{line_number}")
                    continue
                rows[episode_id] = row
                sources[episode_id] = {
                    "worker_index": worker_index,
                    "path": str(path),
                    "line": line_number,
                }
    return rows, sources, issues


def _retry_reasons(policy: Mapping[str, Any] | None) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    reasons: list[str] = []
    if policy is None:
        return ["policy_missing"], {}, {}
    if policy.get("status") != "PASS":
        reasons.append("status_not_pass")
    if policy.get("available") is not True:
        reasons.append("available_not_true")
    if not _finite(policy.get("reward")):
        reasons.append("reward_nonfinite")
    h20 = (policy.get("metrics") or {}).get("20")
    if not isinstance(h20, Mapping):
        reasons.append("h20_missing")
        return reasons, {}, {}
    visible = visible_h20_status(h20)
    # A zero-visible H20 has a valid zero denominator: keep the two metrics as
    # JSON null and do not turn that semantic state into a retry request.
    if visible["status"] != VISIBLE_UNDEFINED or not visible["valid"]:
        reasons.extend(visible["reasons"])
    drift = _drift_status(h20)
    if drift["mask_area_drift_status"] == "NONFINITE_WITH_MASK_SAMPLES":
        reasons.append("h20_mask_area_drift_nonfinite_with_samples")
    elif drift["mask_area_drift_status"] == "MASK_SAMPLE_COUNT_UNDEFINED":
        reasons.append("h20_mask_area_sample_count_undefined")
    if drift["box_area_drift_status"] == "NONFINITE_WITH_BOX_SAMPLES":
        reasons.append("h20_box_area_drift_proxy_nonfinite_with_samples")
    return reasons, dict(h20), drift


def run(
    *,
    manifest_path: Path = MANIFEST,
    worker_dir: Path = WORKER_DIR,
    output: Path = OUT,
    worker_count: int = 4,
) -> dict[str, Any]:
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_rows = frozen.get("episodes", [])
    expected = {str(row["episode_id"]): row for row in expected_rows}
    issues: list[str] = []
    if frozen.get("status") != "PASS" or len(expected) != 689:
        issues.append("frozen N31 manifest is not exactly 689 PASS episodes")
    if frozen.get("val25_read") is not False or frozen.get("test_labels_used") is not False or frozen.get("future_gt_used_for_selection") is not False:
        issues.append("frozen N31 manifest violates blind/causal boundary")
    for episode in expected.values():
        sequence = str(episode.get("sequence", ""))
        split = str(episode.get("split", ""))
        if any(token in sequence.lower() for token in ("val", "test")) or any(token in split.lower() for token in ("val", "test")):
            issues.append(f"non-train episode in frozen manifest: {episode.get('episode_id')}")

    persisted, sources, worker_issues = _load_worker_rows(worker_dir, worker_count)
    issues.extend(worker_issues)
    unexpected = sorted(set(persisted) - set(expected))
    issues.extend(f"unexpected persisted episode: {episode_id}" for episode_id in unexpected)

    retry_items: list[dict[str, Any]] = []
    legitimately_undefined_drift: list[dict[str, Any]] = []
    legitimately_undefined_visible: list[dict[str, Any]] = []
    for episode_id in sorted(expected, key=lambda value: (str(expected[value].get("sequence", "")), int(expected[value].get("correction_frame", 0)), value)):
        episode = expected[episode_id]
        row = persisted.get(episode_id)
        policies = row.get("policies", {}) if row is not None else {}
        for policy_name in POLICIES:
            policy = policies.get(policy_name)
            reasons, h20, drift = _retry_reasons(policy)
            if h20 and (visible := visible_h20_status(h20))["status"] == VISIBLE_UNDEFINED and visible["valid"]:
                legitimately_undefined_visible.append({
                    "episode_id": episode_id,
                    "policy": policy_name,
                    "status": VISIBLE_UNDEFINED,
                    "evaluated_frame_count": visible["evaluated_frame_count"],
                    "visible_frame_count": visible["visible_frame_count"],
                    "absent_gt_frame_count": visible["absent_gt_frame_count"],
                })
            if drift.get("mask_area_drift_status") == "LEGITIMATELY_UNDEFINED_NO_MASK_SAMPLES":
                legitimately_undefined_drift.append({
                    "episode_id": episode_id,
                    "policy": policy_name,
                    "metric": "mask_area_drift",
                    "status": "LEGITIMATELY_UNDEFINED_NO_MASK_SAMPLES",
                    "sample_count": 0,
                })
            if drift.get("box_area_drift_status") == "LEGITIMATELY_UNDEFINED_NO_BOX_SAMPLES":
                legitimately_undefined_drift.append({
                    "episode_id": episode_id,
                    "policy": policy_name,
                    "metric": "box_area_drift_proxy",
                    "status": "LEGITIMATELY_UNDEFINED_NO_BOX_SAMPLES",
                    "sample_count": 0,
                })
            if not reasons:
                continue
            retry_items.append({
                "episode_id": episode_id,
                "policy": policy_name,
                "sequence": str(episode.get("sequence", "")),
                "sequence_path": str(episode.get("sequence_path", "")),
                "learning_split": str(episode.get("learning_split", "")),
                "split": str(episode.get("split", "")),
                "initialization_frame": int(episode["initialization_frame"]),
                "correction_frame": int(episode["correction_frame"]),
                "query_end": int(episode["query_end"]),
                "public_id": int(episode["public_id"]),
                "dataset_identity": int(episode["dataset_identity"]),
                "correction_box": list(episode["correction_box"]),
                "reason_codes": reasons,
                "source": sources.get(episode_id),
                "source_episode_status": None if row is None else row.get("status"),
                "source_policy_status": None if policy is None else policy.get("status"),
                "source_policy_available": None if policy is None else policy.get("available"),
                "source_reward": None if policy is None else policy.get("reward"),
                "source_h20": None if not h20 else {
                    field: h20.get(field)
                    for field in (
                        "mean_box_iou_visible",
                        "missing_prediction_rate_visible",
                        "mask_area_drift",
                        "mask_area_sample_count",
                        "box_area_drift_proxy",
                    )
                },
                "drift_status": drift,
                "source_failure": None if policy is None else policy.get("failure"),
            })

    counts = {policy: sum(item["policy"] == policy for item in retry_items) for policy in POLICIES}
    result = {
        "protocol": "N32-C-POLICY-LEVEL-RETRY-MANIFEST",
        "status": "PASS" if not issues and len(persisted) == len(expected) else "FAIL",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "worker_dir": str(worker_dir),
        "worker_count": int(worker_count),
        "episode_count_expected": len(expected),
        "episode_count_persisted": len(persisted),
        "unexpected_episode_count": len(unexpected),
        "retry_item_count": len(retry_items),
        "retry_item_counts_by_policy": counts,
        "retry_items_unique": len({(item["episode_id"], item["policy"]) for item in retry_items}) == len(retry_items),
        "legitimately_undefined_drift_count": len(legitimately_undefined_drift),
        "legitimately_undefined_drift": legitimately_undefined_drift,
        "legitimately_undefined_visible_count": len(legitimately_undefined_visible),
        "legitimately_undefined_visible": legitimately_undefined_visible,
        "successful_policy_rows_preserved": True,
        "issues": issues,
        "items": retry_items,
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
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--worker-count", type=int, default=4)
    args = parser.parse_args()
    result = run(
        manifest_path=args.manifest,
        worker_dir=args.worker_dir,
        output=args.output,
        worker_count=args.worker_count,
    )
    print(json.dumps({
        key: result[key]
        for key in (
            "protocol",
            "status",
            "episode_count_expected",
            "episode_count_persisted",
            "retry_item_count",
            "retry_item_counts_by_policy",
            "retry_items_unique",
            "legitimately_undefined_drift_count",
            "issues",
        )
    }, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" and result["retry_items_unique"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
