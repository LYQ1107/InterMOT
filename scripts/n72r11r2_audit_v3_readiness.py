#!/usr/bin/env python3
"""Audit the fixed N72R11 V3 readiness thresholds for a development corpus."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-npz", type=Path, required=True)
    parser.add_argument("--validation-metadata", type=Path, required=True)
    parser.add_argument("--self-rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np

    with np.load(args.validation_npz, allow_pickle=False) as loaded:
        candidate_counts = np.asarray(loaded["candidate_counts"], dtype=np.int64)
        labels = np.asarray(loaded["labels"], dtype=np.int64)
    metadata = [
        json.loads(line)
        for line in args.validation_metadata.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rollout = json.loads(args.self_rollout.read_text(encoding="utf-8"))
    predictions = {
        (str(row["event_id"]), int(row["frame"])): row
        for row in rollout.get("records", [])
    }
    if len(metadata) != len(labels) or len(predictions) != len(rollout.get("records", [])):
        raise RuntimeError("validation metadata/label/prediction cardinality is inconsistent")

    categories: dict[str, list[bool]] = defaultdict(list)
    missing_predictions: list[dict[str, Any]] = []
    action_categories: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(metadata):
        key = (str(row["event_id"]), int(row["frame"]))
        prediction = predictions.get(key)
        if prediction is None:
            missing_predictions.append({"event_id": key[0], "frame": key[1]})
            continue
        if not _finite(prediction.get("predicted_index")):
            raise RuntimeError(f"non-finite V3 prediction at {key}")
        candidate_count = int(candidate_counts[index])
        label_index = int(labels[index])
        label_kind = str(row["label_kind"])
        predicted_index = int(prediction["predicted_index"])
        correct = predicted_index == label_index if label_kind == "TARGET_CANDIDATE" else predicted_index == candidate_count
        if label_kind == "TARGET_CANDIDATE":
            category = "future_positive" if int(row["frame_horizon"]) > 0 else "target_candidate"
        elif label_kind == "NONE":
            category = "none"
        else:
            raise RuntimeError(f"unknown label kind at {key}: {label_kind}")
        categories[category].append(bool(correct))
        action_categories[str(row["action_type"])][category].append(bool(correct))

    if missing_predictions:
        raise RuntimeError(f"missing self-rollout predictions: {len(missing_predictions)}")

    def summarize(values: list[bool]) -> dict[str, Any]:
        return {
            "count": len(values),
            "correct": sum(values),
            "accuracy": (sum(values) / len(values)) if values else None,
        }

    summary = {key: summarize(value) for key, value in sorted(categories.items())}
    thresholds = {
        "target_candidate_accuracy_min": 0.80,
        "none_accuracy_min": 0.50,
        "future_positive_accuracy_min": 0.60,
        "future_positive_count_min": 50,
    }
    target = summary.get("target_candidate", {"accuracy": None, "count": 0})
    none = summary.get("none", {"accuracy": None, "count": 0})
    future = summary.get("future_positive", {"accuracy": None, "count": 0})
    gate = {
        "target_candidate_accuracy": target["accuracy"] is not None and target["accuracy"] >= thresholds["target_candidate_accuracy_min"],
        "none_accuracy": none["accuracy"] is not None and none["accuracy"] >= thresholds["none_accuracy_min"],
        "future_positive_accuracy": future["accuracy"] is not None and future["accuracy"] >= thresholds["future_positive_accuracy_min"],
        "future_positive_count": future["count"] >= thresholds["future_positive_count_min"],
    }
    result = {
        "schema_version": "N72R11R2_V3_READINESS_AUDIT_V1",
        "status": "PASS_V3_READINESS" if all(gate.values()) else "SOURCE_VALIDATION_UNDERCOVERED_OR_V3_NOT_READY",
        "resource_censored_development": True,
        "thresholds": thresholds,
        "gate": gate,
        "all_thresholds_pass": all(gate.values()),
        "summary": summary,
        "by_action": {
            action: {category: summarize(values) for category, values in sorted(by_category.items())}
            for action, by_category in sorted(action_categories.items())
        },
        "inputs": {
            "validation_npz": str(args.validation_npz.resolve()),
            "validation_npz_sha256": _sha256(args.validation_npz),
            "validation_metadata": str(args.validation_metadata.resolve()),
            "validation_metadata_sha256": _sha256(args.validation_metadata),
            "self_rollout": str(args.self_rollout.resolve()),
            "self_rollout_sha256": _sha256(args.self_rollout),
        },
        "runtime_future_gt_used": False,
        "gt_used_for_runtime_decision": False,
        "interaction_source": "simulated_from_gt",
        "not_real_human_evidence": True,
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["all_thresholds_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
