#!/usr/bin/env python3
"""Posthoc separability statistics for N72R15 state-edge components."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n72r11_on_demand_replay as replay  # noqa: E402


HORIZONS = (20, 50, 100)
IOU_THRESHOLD = 0.50
VARIANTS = ("E1I_HUMAN_RELATIVE_STATE", "E1J_TRUSTED_GLOBAL_RELATIVE_STATE")
COMPONENTS = ("appearance", "appearance_prototype", "appearance_positive", "appearance_negative", "motion", "native_continuity", "gap", "raw", "relative", "relative_delta")
MATRIX_KEYS = {
    "raw": "raw_evidence_matrix",
    "relative": "relative_evidence_matrix",
    "relative_delta": "relative_delta_matrix",
}
DEFAULT_MANIFEST = ROOT / "outputs/N72R15/formal/formal_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/N72R15/state_edge_component_diagnosis.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _iou(a: Any, b: Any) -> float:
    left, top = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    right, bottom = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _auc(values: list[tuple[float, bool]]) -> float | None:
    positive_count = sum(1 for _, label in values if label)
    negative_count = len(values) - positive_count
    if not positive_count or not negative_count:
        return None
    # Exact Mann–Whitney probability with tie handling, using one sorted
    # pass instead of the O(N²) positive×negative Cartesian product.
    ordered = sorted((float(score), bool(label)) for score, label in values)
    negatives_seen = 0
    wins = 0.0
    index = 0
    while index < len(ordered):
        score = ordered[index][0]
        group_positive = group_negative = 0
        while index < len(ordered) and ordered[index][0] == score:
            if ordered[index][1]:
                group_positive += 1
            else:
                group_negative += 1
            index += 1
        wins += group_positive * negatives_seen + 0.5 * group_positive * group_negative
        negatives_seen += group_negative
    return float(wins / (positive_count * negative_count))


def _stats(values: list[tuple[float, bool]]) -> dict[str, Any]:
    numbers = np.asarray([value for value, _ in values], dtype=np.float64)
    pos = [value for value, label in values if label]
    neg = [value for value, label in values if not label]
    def summary(items: list[float]) -> dict[str, Any]:
        if not items:
            return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None}
        array = np.asarray(items, dtype=np.float64)
        return {"count": len(items), "mean": float(np.mean(array)), "median": float(np.median(array)), "p25": float(np.quantile(array, 0.25)), "p75": float(np.quantile(array, 0.75))}
    return {
        "count": int(numbers.size),
        "positive_count": len(pos),
        "negative_count": len(neg),
        "all": summary(numbers.tolist()),
        "positive": summary(pos),
        "negative": summary(neg),
        "roc_auc_positive_vs_negative": _auc(values),
    }


def diagnose(manifest_path: Path = DEFAULT_MANIFEST, output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PASS_N72R15_FORMAL_REPLAY" or int(manifest.get("event_count", -1)) != 32:
        raise RuntimeError("formal manifest is not complete")
    protocol = read_json(ROOT / "outputs/N72R9/protocol.json")
    protocol_events = {str(item["event_id"]): dict(item) for item in protocol["source_event_selection"]["events"]}
    values: dict[str, dict[str, list[tuple[float, bool]]]] = defaultdict(lambda: defaultdict(list))
    by_action: dict[str, dict[str, dict[str, list[tuple[float, bool]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    by_horizon: dict[str, dict[str, dict[str, list[tuple[float, bool]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    record_count = 0
    positive_edge_count = 0
    negative_edge_count = 0
    for event_record in manifest["events"]:
        event_id = str(event_record["event_id"])
        event = protocol_events[event_id]
        inputs = replay._load_inputs(event, horizon=100)
        gt = replay.legacy._load_gt(str(event["sequence"]))
        protected = replay.legacy._protected_map(inputs["rows"]["c0_source"][int(event["event_frame"])], gt, int(event["event_frame"]), int(event["dataset_gt_id"]))
        source_event_manifest = read_json(Path(str(event["source_event_manifest"])))
        target_public_id = int(source_event_manifest["target_public_id"])
        public_to_gid = {target_public_id: int(event["dataset_gt_id"])}
        public_to_gid.update({int(public): int(gid) for gid, public in protected.items()})
        variant_paths = {str(item["variant"]): Path(str(item["frames"])) for item in event_record["variants"]}
        for variant in VARIANTS:
            rows = [json.loads(line) for line in variant_paths[variant].read_text(encoding="utf-8").splitlines() if line.strip()]
            for offset, row in enumerate(rows[1:], start=1):
                horizon = int(offset)
                edge = row.get("persistent_state_association", {})
                axes = [int(value) for value in edge.get("public_id_axis", row.get("score_audit", {}).get("public_id_axis", []))]
                candidates = row.get("candidate_rows", [])
                matrices = {
                    component: np.asarray(edge.get(MATRIX_KEYS.get(component, component), []), dtype=np.float64)
                    for component in COMPONENTS
                }
                if any(matrix.shape != (len(candidates), len(axes)) for matrix in matrices.values()):
                    raise RuntimeError(f"{event_id}/{variant}:{row.get('frame')}: component shape mismatch")
                if not axes:
                    continue
                if horizon > max(HORIZONS):
                    continue
                for public_index, public in enumerate(axes):
                    gid = public_to_gid.get(public)
                    if gid is None:
                        continue
                    gt_item = gt.get(int(row["frame"]), {}).get(gid)
                    if gt_item is None:
                        continue
                    for row_index, candidate in enumerate(candidates):
                        candidate_box = candidate.get("box_xyxy", candidate.get("box"))
                        positive = bool(_iou(candidate_box, gt_item["box"]) >= IOU_THRESHOLD)
                        key = f"{variant}/ALL"
                        for component in COMPONENTS:
                            value = float(matrices[component][row_index, public_index])
                            if not math.isfinite(value):
                                raise RuntimeError(f"non-finite {component} at {event_id}:{row.get('frame')}")
                            values[key][component].append((value, positive))
                            by_action[str(event["action_type"])][key][component].append((value, positive))
                            for cutoff in HORIZONS:
                                if horizon <= cutoff:
                                    by_horizon[str(cutoff)][key][component].append((value, positive))
                            record_count += 1
                            positive_edge_count += int(positive)
                            negative_edge_count += int(not positive)
    summary = {key: {component: _stats(items) for component, items in sorted(component_values.items())} for key, component_values in sorted(values.items())}
    action_summary = {action: {key: {component: _stats(items) for component, items in sorted(component_values.items())} for key, component_values in sorted(groups.items())} for action, groups in sorted(by_action.items())}
    horizon_summary = {horizon: {key: {component: _stats(items) for component, items in sorted(component_values.items())} for key, component_values in sorted(groups.items())} for horizon, groups in sorted(by_horizon.items())}
    recover_key_values = {component: _stats(items) for component, items in sorted(values.get("E1J_TRUSTED_GLOBAL_RELATIVE_STATE/ALL", {}).items())}
    output = {
        "schema_version": "N72R15_STATE_EDGE_COMPONENT_DIAGNOSIS_V1",
        "status": "PASS_N72R15_STATE_EDGE_COMPONENT_DIAGNOSIS",
        "created_at_utc": now_utc(),
        "source_formal_manifest": str(manifest_path),
        "source_formal_manifest_sha256": sha256_file(manifest_path),
        "iou_threshold": IOU_THRESHOLD,
        "components": list(COMPONENTS),
        "record_count": record_count,
        "positive_edge_count": positive_edge_count,
        "negative_edge_count": negative_edge_count,
        "summary": summary,
        "by_action": action_summary,
        "by_horizon": horizon_summary,
        "recover_global_all_horizon_summary": recover_key_values,
        "runtime_future_gt_used": False,
        "posthoc_gt_used": True,
        "diagnostic_only": True,
        "historical_outputs_modified": False,
    }
    atomic_json(output_path, output)
    return output


def main() -> int:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--formal-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = diagnose(args.formal_manifest.resolve(), args.output.resolve())
        print(json.dumps({"status": result["status"], "record_count": result["record_count"], "output": str(args.output.resolve())}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {"schema_version": "N72R15_FAILURE_V1", "status": "FAIL_N72R15_STATE_EDGE_COMPONENT_DIAGNOSIS", "error_type": type(exc).__name__, "error": str(exc), "created_at_utc": now_utc(), "runtime_future_gt_used": False}
        atomic_json(args.output.resolve().with_name("state_edge_component_diagnosis_failure.json"), failure)
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
