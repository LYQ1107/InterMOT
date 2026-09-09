#!/usr/bin/env python3
"""Build the N72R12 learned-safe-gate corpus from the frozen 516 interactions.

This is an offline label materializer.  It consumes the completed
N72R11R4 resource-censored corpus, which already contains the causal base and
PCTIS proposal solver audits.  No SAM3 process is started here.  GT is opened
only after the runtime row and CSI audit have been reconstructed, and is used
solely for the binary training label.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import os
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.counterfactual_safe_intervention import (  # noqa: E402
    build_counterfactual_audit,
)
from sam3_intermot.association.learned_safe_intervention import (  # noqa: E402
    SAFE_GATE_FEATURE_DIM,
    SAFE_GATE_FEATURE_SCHEMA,
    build_safe_gate_feature,
)
from scripts import n72r9_temporal_replay as legacy  # noqa: E402


SOURCE_MANIFEST = ROOT / "outputs/N72R11R4/exact_onpolicy_v3/corpus_manifest.json"
SOURCE_ROOT = ROOT / "outputs/N72R11R4/exact_onpolicy_v3"
RESOURCE_AUDIT = ROOT / "outputs/N72R11R2/resource_censoring_audit.json"
PROTOCOL = ROOT / "outputs/N72R9/protocol.json"
OUTPUT_ROOT = ROOT / "outputs/N72R12/gate_corpus"
IOU_THRESHOLD = 0.50
EXPECTED_INTERACTIONS = 516


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        if isinstance(content, bytes):
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write(path, json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
    )


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(temporary, **dict(arrays))
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            result.append(value)
    return result


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite")
    return result


def _solver_row(pool: Sequence[Mapping[str, Any]], solver: Mapping[str, Any], label: str) -> dict[str, Any]:
    rows = solver.get("assignment_rows")
    if not isinstance(rows, list):
        raise ValueError(f"{label} lacks assignment_rows")
    by_uid: dict[str, Mapping[str, Any]] = {}
    for item in rows:
        if not isinstance(item, Mapping) or item.get("candidate_uid") is None:
            raise ValueError(f"{label} contains an invalid assignment row")
        uid = str(item["candidate_uid"])
        if uid in by_uid:
            raise ValueError(f"{label} contains duplicate candidate UID {uid}")
        by_uid[uid] = item
    pool_uids = [str(item.get("candidate_uid")) for item in pool]
    if len(pool_uids) != len(set(pool_uids)) or set(pool_uids) != set(by_uid):
        raise ValueError(f"{label} does not cover the complete candidate axis")
    candidates: list[dict[str, Any]] = []
    for candidate in pool:
        item = dict(candidate)
        decision = by_uid[str(candidate["candidate_uid"])]
        public_id = decision.get("public_id")
        item["public_id"] = None if public_id is None else int(public_id)
        candidates.append(item)
    return {"candidate_rows": candidates, "runtime_future_gt_used": False, "runtime_gt_read": False}


def _base_event_id(event_id: str) -> str:
    return str(event_id).split(":secondary:", 1)[0]


def _safe_audit(
    row: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    row_index: int,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    candidate_pool = row.get("candidate_pool")
    if not isinstance(candidate_pool, Mapping):
        raise ValueError(f"{row.get('event_id')}: missing candidate_pool")
    pool = list(candidate_pool.get("candidate_rows", []))
    if not pool:
        raise ValueError(f"{row.get('event_id')}:{row.get('frame')}: empty candidate pool")
    score = row.get("score_audit")
    if not isinstance(score, Mapping):
        raise ValueError(f"{row.get('event_id')}:{row.get('frame')}: missing score audit")
    base_matrix = np.asarray(score.get("base_score_matrix", []), dtype=np.float64)
    proposal_matrix = np.asarray(score.get("fused_score_matrix", []), dtype=np.float64)
    public_axis = [int(value) for value in score.get("public_id_axis", [])]
    if base_matrix.shape != proposal_matrix.shape or base_matrix.shape != (len(pool), len(public_axis)):
        raise ValueError(f"{row.get('event_id')}:{row.get('frame')}: score/axis shape mismatch")
    if not np.isfinite(base_matrix).all() or not np.isfinite(proposal_matrix).all():
        raise ValueError(f"{row.get('event_id')}:{row.get('frame')}: non-finite solver matrix")
    base_solver = row.get("base_solver")
    proposal_solver = row.get("assignment", {}).get("solver") if isinstance(row.get("assignment"), Mapping) else None
    selection = row.get("selection")
    if not isinstance(base_solver, Mapping) or not isinstance(proposal_solver, Mapping) or not isinstance(selection, Mapping):
        raise ValueError(f"{row.get('event_id')}:{row.get('frame')}: incomplete solver/selection audit")
    motion = np.asarray(arrays["motion_iou"][row_index], dtype=np.float64).reshape(-1)[: len(pool)]
    if motion.shape != (len(pool),) or not np.isfinite(motion).all():
        raise ValueError(f"{row.get('event_id')}:{row.get('frame')}: motion_iou axis mismatch")
    model_values = {
        "legacy_best_other_scores": np.asarray(score.get("base_best_other_scores", []), dtype=np.float64),
        "motion_iou": motion,
        "causal_order": list(row.get("causal_order", [])),
        "base_assignment_margin": score.get("base_assignment_margin"),
    }
    audit = build_counterfactual_audit(
        target_public_id=int(row["target_public_id_for_offline_audit"]),
        pool=pool,
        base_matrix=base_matrix,
        proposed_matrix=proposal_matrix,
        public_axis=public_axis,
        base_solver=base_solver,
        proposal_solver=proposal_solver,
        selection=selection,
        model_values=model_values,
    )
    feature = build_safe_gate_feature(
        audit=audit,
        selection=selection,
        state_audit=row.get("temporal_state_before"),
        action=str(row["action_type"]),
        source=audit.get("selected_candidate_source"),
        candidate_count=len(pool),
        pool=pool,
    )
    return audit, feature, pool


def _protected_map_for(
    event_id: str,
    protocol_events: Mapping[str, Mapping[str, Any]],
    gt_cache: Mapping[str, Mapping[int, Mapping[int, Mapping[str, Any]]]],
    protected_cache: dict[tuple[str, int], dict[int, int]],
    target_gid: int,
    sequence: str,
) -> dict[int, int]:
    key = (_base_event_id(event_id), int(target_gid))
    if key in protected_cache:
        return protected_cache[key]
    event = protocol_events.get(key[0])
    if event is None:
        raise ValueError(f"metadata event is not in frozen N72R9 protocol: {key[0]}")
    inputs = legacy._load_rows(event)
    gt = gt_cache[sequence]
    protected = legacy._protected_map(
        inputs["rows"]["c0_source"][int(event["event_frame"])],
        gt,
        int(event["event_frame"]),
        int(target_gid),
    )
    protected_cache[key] = {int(gid): int(pid) for gid, pid in protected.items()}
    return protected_cache[key]


def _label(
    row: Mapping[str, Any],
    base_runtime: Mapping[str, Any],
    proposal_runtime: Mapping[str, Any],
    gt_cache: Mapping[str, Mapping[int, Mapping[int, Mapping[str, Any]]]],
    protocol_events: Mapping[str, Mapping[str, Any]],
    protected_cache: dict[tuple[str, int], dict[int, int]],
) -> tuple[int | None, dict[str, Any]]:
    sequence = str(row["sequence"])
    frame = int(row["frame"])
    target_gid = int(row["target_dataset_gt_id_for_offline_label"])
    target_public = int(row["target_public_id_for_offline_audit"])
    gt_frame = gt_cache[sequence].get(frame, {})
    target = gt_frame.get(target_gid)
    base_iou = 0.0
    proposal_iou = 0.0
    protected_regressions: list[int] = []
    if target is not None:
        base_iou = float(legacy._public_box_for_gt(base_runtime, target_public, target["box"])[0])
        proposal_iou = float(legacy._public_box_for_gt(proposal_runtime, target_public, target["box"])[0])
        protected = _protected_map_for(
            str(row["event_id"]), protocol_events, gt_cache, protected_cache, target_gid, sequence
        )
        for protected_gid, protected_public in protected.items():
            other = gt_frame.get(int(protected_gid))
            if other is None:
                continue
            base_other = legacy._public_box_for_gt(base_runtime, int(protected_public), other["box"])[0] >= IOU_THRESHOLD
            proposal_other = legacy._public_box_for_gt(proposal_runtime, int(protected_public), other["box"])[0] >= IOU_THRESHOLD
            if base_other and not proposal_other:
                protected_regressions.append(int(protected_gid))
    visible = target is not None
    if not visible:
        return None, {
            "included_in_training": False,
            "label_reason": "EXCLUDED_TARGET_NOT_VISIBLE",
            "target_visible": False,
            "base_target_correct": None,
            "proposal_target_correct": None,
            "protected_regression_count": None,
            "gt_used_for_offline_label": True,
        }
    base_correct = bool(base_iou >= IOU_THRESHOLD)
    proposal_correct = bool(proposal_iou >= IOU_THRESHOLD)
    apply = bool((not base_correct) and proposal_correct and not protected_regressions)
    return int(apply), {
        "included_in_training": True,
        "label_reason": "APPLY_SAFE" if apply else "KEEP_UNSAFE_OR_NO_GAIN",
        "target_visible": True,
        "base_target_correct": base_correct,
        "proposal_target_correct": proposal_correct,
        "base_target_iou": float(base_iou),
        "proposal_target_iou": float(proposal_iou),
        "protected_regression_count": len(protected_regressions),
        "protected_regression_gids": protected_regressions,
        "gt_used_for_offline_label": True,
    }


def _runtime_row_from_solver(pool: Sequence[Mapping[str, Any]], solver: Mapping[str, Any]) -> dict[str, Any]:
    return _solver_row(pool, solver, "solver")


def _protocol_events() -> dict[str, dict[str, Any]]:
    payload = read_json(PROTOCOL)
    events = payload.get("source_event_selection", {}).get("events", [])
    result = {str(item["event_id"]): dict(item) for item in events}
    if len(result) != 32:
        raise ValueError(f"frozen protocol must contain 32 events, found {len(result)}")
    return result


def main() -> int:
    started = now_utc()
    try:
        prior_failure = OUTPUT_ROOT / "corpus_failure.json"
        prior_attempt = OUTPUT_ROOT / "corpus_failure_attempt_01.json"
        if prior_failure.is_file() and not prior_attempt.is_file():
            atomic_write(prior_attempt, prior_failure.read_bytes())
        source_manifest = read_json(SOURCE_MANIFEST)
        resource_audit = read_json(RESOURCE_AUDIT)
        if source_manifest.get("status") != "PASS_N72R11R4_EXACT_SOLVER_ON_POLICY_CORPUS":
            raise RuntimeError(f"source corpus is not the exact PASS corpus: {source_manifest.get('status')}")
        if int(source_manifest.get("event_count", -1)) != EXPECTED_INTERACTIONS or not source_manifest.get("resource_censored_development"):
            raise RuntimeError("learned gate requires the frozen 516 resource-censored corpus")
        counts = resource_audit.get("counts", {})
        if counts.get("retained_executable") != EXPECTED_INTERACTIONS or counts.get("resource_censored_cuda_oom") != 11:
            raise RuntimeError(f"resource-censor audit mismatch: {counts}")
        protocol_events = _protocol_events()
        split_source = Path(str(source_manifest["source_corpus_manifest"]))
        if not split_source.is_file() or sha256_file(split_source) != str(source_manifest["source_corpus_manifest_sha256"]):
            raise RuntimeError("frozen source corpus split manifest hash mismatch")
        split_payload = read_json(split_source)
        train_sequences = [str(value) for value in split_payload["train_sequences"]]
        validation_sequences = [str(value) for value in split_payload["validation_sequences"]]
        if len(train_sequences) != 12 or len(validation_sequences) != 6 or set(train_sequences) & set(validation_sequences):
            raise RuntimeError("frozen sequence split is not 12 train / 6 validation")
        gt_cache = {sequence: legacy._load_gt(sequence) for sequence in train_sequences + validation_sequences}
        protected_cache: dict[tuple[str, int], dict[int, int]] = {}
        split_summaries: dict[str, Any] = {}
        all_source_rows = 0
        all_included = 0
        all_positive = 0
        for split in ("train", "validation"):
            metadata_path = SOURCE_ROOT / f"{split}_metadata.jsonl"
            npz_path = SOURCE_ROOT / f"{split}.npz"
            source_rows = read_jsonl(metadata_path)
            source_arrays = np.load(npz_path, allow_pickle=False)
            arrays = {key: source_arrays[key] for key in source_arrays.files}
            if len(source_rows) != int(arrays["motion_iou"].shape[0]):
                raise RuntimeError(f"{split}: metadata/array count mismatch")
            output_rows: list[dict[str, Any]] = []
            features: list[np.ndarray] = []
            labels: list[int] = []
            label_counts: Counter[str] = Counter()
            for index, row in enumerate(source_rows):
                for flag in ("runtime_future_gt_used", "runtime_gt_read", "public_id_inference"):
                    if row.get(flag) is not False:
                        raise RuntimeError(f"{split}:{index} violates {flag}")
                audit, feature, pool = _safe_audit(row, arrays, index)
                base_solver = row["base_solver"]
                proposal_solver = row["assignment"]["solver"]
                base_runtime = _runtime_row_from_solver(pool, base_solver)
                proposal_runtime = _runtime_row_from_solver(pool, proposal_solver)
                label, label_info = _label(
                    row, base_runtime, proposal_runtime, gt_cache, protocol_events, protected_cache
                )
                if label is not None:
                    features.append(feature)
                    labels.append(int(label))
                    all_included += 1
                    all_positive += int(label)
                label_counts[str(label_info["label_reason"])] += 1
                output_rows.append({
                    "sample_index": int(len(output_rows)),
                    "source_row_index": int(index),
                    "event_id": str(row["event_id"]),
                    "source_event_id": _base_event_id(str(row["event_id"])),
                    "sequence": str(row["sequence"]),
                    "split": split,
                    "action_type": str(row["action_type"]),
                    "event_frame": int(row["event_frame"]),
                    "frame": int(row["frame"]),
                    "frame_horizon": int(row["frame_horizon"]),
                    "candidate_count": len(pool),
                    "selected_candidate_uid": audit.get("selected_candidate_uid"),
                    "selected_candidate_source": audit.get("selected_candidate_source"),
                    "counterfactual_audit": audit,
                    "feature": feature.astype(float).tolist(),
                    **label_info,
                    "runtime_future_gt_used": False,
                    "runtime_gt_read": False,
                    "posthoc_gt_used": False,
                    "interaction_source": "simulated_from_gt",
                    "not_real_human_evidence": True,
                })
            feature_array = np.stack(features, axis=0).astype(np.float32) if features else np.zeros((0, SAFE_GATE_FEATURE_DIM), dtype=np.float32)
            label_array = np.asarray(labels, dtype=np.int64)
            if feature_array.shape[1:] != (SAFE_GATE_FEATURE_DIM,) or not np.isfinite(feature_array).all():
                raise RuntimeError(f"{split}: feature matrix is not finite {SAFE_GATE_FEATURE_DIM}-D")
            if label_array.shape != (feature_array.shape[0],) or not np.isin(label_array, [0, 1]).all():
                raise RuntimeError(f"{split}: labels are invalid")
            atomic_npz(OUTPUT_ROOT / f"{split}.npz", {"features": feature_array, "labels": label_array})
            atomic_jsonl(OUTPUT_ROOT / f"{split}_metadata.jsonl", output_rows)
            split_summaries[split] = {
                "source_metadata": str(metadata_path),
                "source_metadata_sha256": sha256_file(metadata_path),
                "source_npz": str(npz_path),
                "source_npz_sha256": sha256_file(npz_path),
                "source_row_count": len(source_rows),
                "included_count": int(feature_array.shape[0]),
                "excluded_count": int(len(source_rows) - feature_array.shape[0]),
                "positive_count": int(np.sum(label_array == 1)),
                "negative_count": int(np.sum(label_array == 0)),
                "event_count": len({str(row["event_id"]) for row in output_rows}),
                "sequence_count": len({str(row["sequence"]) for row in output_rows}),
                "action_counts": dict(sorted(Counter(str(row["action_type"]) for row in output_rows).items())),
                "label_counts": dict(sorted(label_counts.items())),
                "npz": str(OUTPUT_ROOT / f"{split}.npz"),
                "npz_sha256": sha256_file(OUTPUT_ROOT / f"{split}.npz"),
                "metadata": str(OUTPUT_ROOT / f"{split}_metadata.jsonl"),
                "metadata_sha256": sha256_file(OUTPUT_ROOT / f"{split}_metadata.jsonl"),
            }
            all_source_rows += len(source_rows)
        manifest = {
            "schema_version": "N72R12_SAFE_GATE_CORPUS_V1",
            "status": "PASS_N72R12_SAFE_GATE_CORPUS",
            "created_at_utc": now_utc(),
            "source_corpus_manifest": str(SOURCE_MANIFEST),
            "source_corpus_manifest_sha256": sha256_file(SOURCE_MANIFEST),
            "source_resource_censor_audit": str(RESOURCE_AUDIT),
            "source_resource_censor_audit_sha256": sha256_file(RESOURCE_AUDIT),
            "source_protocol": str(PROTOCOL),
            "source_protocol_sha256": sha256_file(PROTOCOL),
            "resource_censored_development": True,
            "required_executable_interactions": EXPECTED_INTERACTIONS,
            "source_row_count": all_source_rows,
            "included_sample_count": all_included,
            "positive_sample_count": all_positive,
            "negative_sample_count": all_included - all_positive,
            "feature_dim": SAFE_GATE_FEATURE_DIM,
            "feature_schema": list(SAFE_GATE_FEATURE_SCHEMA),
            "label_definition": "target_visible AND base_target_incorrect AND proposal_target_correct AND protected_regression_zero",
            "excluded_label": "EXCLUDED_TARGET_NOT_VISIBLE",
            "gt_used_only_offline_label_generation": True,
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "real_human_evidence": False,
            "not_real_human_evidence": True,
            "train_sequences": train_sequences,
            "validation_sequences": validation_sequences,
            "splits": split_summaries,
        }
        atomic_json(OUTPUT_ROOT / "manifest.json", manifest)
        print(json.dumps({"status": manifest["status"], "source_rows": all_source_rows, "samples": all_included, "positive": all_positive, "output": str(OUTPUT_ROOT / "manifest.json")}, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "N72R12_SAFE_GATE_CORPUS_FAILURE_V1",
            "status": "FAIL_N72R12_SAFE_GATE_CORPUS",
            "started_at_utc": started,
            "finished_at_utc": now_utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": __import__("traceback").format_exc(),
            "historical_outputs_modified": False,
        }
        failure_path = OUTPUT_ROOT / "corpus_failure.json"
        if failure_path.is_file():
            attempt = OUTPUT_ROOT / "corpus_failure_attempt_01.json"
            if attempt.is_file():
                attempt = OUTPUT_ROOT / "corpus_failure_attempt_02.json"
            atomic_write(attempt, failure_path.read_bytes())
        atomic_json(failure_path, failure)
        print(json.dumps({"status": failure["status"], "error": str(exc), "failure": str(failure_path)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
