#!/usr/bin/env python3
"""Audit N72R11R3 V3 readiness with source-correct causal categories.

The R2 auditor called every positive future row ``future_positive``.  R3 keeps
the broad target-candidate gate, but reserves the future-requery gate for rows
whose offline label actually points to a FUTURE_FRAME_REQUERY candidate.
"""

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

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.reacquisition.temporal_state_policy import TEMPORAL_FEATURE_SCHEMA


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
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


def _summary(values: list[bool]) -> dict[str, Any]:
    return {"count": len(values), "correct": int(sum(values)), "accuracy": (sum(values) / len(values)) if values else None}


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
    metadata = [json.loads(line) for line in args.validation_metadata.read_text(encoding="utf-8").splitlines() if line.strip()]
    rollout = json.loads(args.self_rollout.read_text(encoding="utf-8"))
    records = list(rollout.get("records", []))
    predictions: dict[tuple[str, int], dict[str, Any]] = {}
    duplicate_predictions: list[tuple[str, int]] = []
    for row in records:
        key = (str(row["event_id"]), int(row["frame"]))
        if key in predictions:
            duplicate_predictions.append(key)
        predictions[key] = row
    if len(metadata) != len(labels):
        raise RuntimeError("validation metadata/label cardinality is inconsistent")
    if duplicate_predictions:
        raise RuntimeError(f"duplicate self-rollout predictions: {len(duplicate_predictions)}")

    categories: dict[str, list[bool]] = defaultdict(list)
    action_categories: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    missing: list[dict[str, Any]] = []
    source_counts: dict[str, int] = defaultdict(int)
    for index, row in enumerate(metadata):
        key = (str(row["event_id"]), int(row["frame"]))
        prediction = predictions.get(key)
        if prediction is None:
            missing.append({"event_id": key[0], "frame": key[1]})
            continue
        if not _finite(prediction.get("predicted_index")):
            raise RuntimeError(f"non-finite V3 prediction at {key}")
        candidate_count = int(candidate_counts[index])
        predicted_index = int(prediction["predicted_index"])
        label_index = int(labels[index])
        if predicted_index < 0 or predicted_index > candidate_count:
            raise RuntimeError(f"prediction outside candidate/NONE axis at {key}")
        label_kind = str(row["label_kind"])
        correct = predicted_index == label_index if label_kind == "TARGET_CANDIDATE" else predicted_index == candidate_count
        if label_kind not in {"TARGET_CANDIDATE", "NONE"}:
            raise RuntimeError(f"unknown label kind at {key}: {label_kind}")
        categories["target_candidate" if label_kind == "TARGET_CANDIDATE" else "none"].append(bool(correct))
        action_categories[str(row["action_type"])]["target_candidate" if label_kind == "TARGET_CANDIDATE" else "none"].append(bool(correct))
        if label_kind == "TARGET_CANDIDATE":
            sources = list(row.get("candidate_sources", []))
            if not 0 <= label_index < len(sources):
                raise RuntimeError(f"positive label source is outside candidate axis at {key}")
            source = str(sources[label_index])
            source_counts[source] += 1
            if source == "FUTURE_FRAME_REQUERY":
                categories["future_requery_positive"].append(bool(correct))
                action_categories[str(row["action_type"])]["future_requery_positive"].append(bool(correct))
    if missing:
        raise RuntimeError(f"missing self-rollout predictions: {len(missing)}")
    summary = {key: _summary(value) for key, value in sorted(categories.items())}
    thresholds = {
        "target_candidate_accuracy_min": 0.80,
        "none_accuracy_min": 0.50,
        "future_requery_positive_accuracy_min": 0.60,
        "future_requery_positive_count_min": 50,
    }
    target = summary.get("target_candidate", {"accuracy": None, "count": 0})
    none = summary.get("none", {"accuracy": None, "count": 0})
    future = summary.get("future_requery_positive", {"accuracy": None, "count": 0})
    gate = {
        "target_candidate_accuracy": target["accuracy"] is not None and target["accuracy"] >= thresholds["target_candidate_accuracy_min"],
        "none_accuracy": none["accuracy"] is not None and none["accuracy"] >= thresholds["none_accuracy_min"],
        "future_requery_positive_accuracy": future["accuracy"] is not None and future["accuracy"] >= thresholds["future_requery_positive_accuracy_min"],
        "future_requery_positive_count": future["count"] >= thresholds["future_requery_positive_count_min"],
    }
    result = {
        "schema_version": "N72R11R3_V3_READINESS_AUDIT_V1",
        "status": "PASS_V3_READINESS" if all(gate.values()) else "V3_NOT_READY_AFTER_STATE_ALIGNMENT",
        "resource_censored_development": True,
        "thresholds": thresholds,
        "gate": gate,
        "all_thresholds_pass": all(gate.values()),
        "summary": summary,
        "by_action": {action: {category: _summary(values) for category, values in sorted(value.items())} for action, value in sorted(action_categories.items())},
        "candidate_source_counts_for_positive_labels": dict(sorted(source_counts.items())),
        "temporal_feature_schema": list(TEMPORAL_FEATURE_SCHEMA),
        "inputs": {
            "validation_npz": str(args.validation_npz.resolve()), "validation_npz_sha256": _sha256(args.validation_npz),
            "validation_metadata": str(args.validation_metadata.resolve()), "validation_metadata_sha256": _sha256(args.validation_metadata),
            "self_rollout": str(args.self_rollout.resolve()), "self_rollout_sha256": _sha256(args.self_rollout),
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
