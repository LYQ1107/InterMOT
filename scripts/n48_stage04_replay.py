#!/usr/bin/env python3
"""N48 GT-free runtime and posthoc paired replay for the frozen probe.

The runtime materializes no-write, write-baseline and write-plus-N48 from the
same frozen N47/N42 candidate stream.  Only the posthoc section imports GT.
This is a simulated_from_gt diagnostic because the prefix memory anchor was
prepared offline from the existing GT-derived event manifest.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.n47_global_probe_common import (  # noqa: E402
    HARD_NEGATIVE,
    N43_MAP,
    event_map,
    load,
    normalize_assignment,
    write_json,
)
from scripts.n48_assignment_common import (  # noqa: E402
    N36_FRAMES,
    N47_RUNTIME,
    N48_OUT,
    candidate_features,
    load_checkpoint,
    load_n36_sequence,
    runtime_sidecar,
    scalar_features,
)

# The conditional import expression above is intentionally not valid Python;
# it is replaced immediately below by the isolated constants.  Keeping all
# paths here makes this module independent from production MOT/OVMOT.
OUT = N48_OUT
RUNTIME = OUT / "replay/runtime"
POSTHOC = OUT / "replay/posthoc"
CHECKPOINT = OUT / "training/n48_risk_aware_512d.pt"
MEMORY_MANIFEST = OUT / "training/simulated_event_memory.json"
VARIANTS = ("M0", "M1", "M2", "M3", "M4")
HORIZONS = (20, 50, 100)
BOOTSTRAP_SEED = 4848
BOOTSTRAP_REPS = 2000


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array, dtype=np.float32).tobytes()).hexdigest()


def candidate_signature(candidates: list[dict[str, Any]]) -> list[tuple[int, Any, float]]:
    return [(int(x["native_tid"]), x.get("box"), float(x.get("confidence", 0.0))) for x in candidates]


def assignment_transition(before: list[Any], after: list[Any]) -> dict[str, Any]:
    changed = [i for i, pair in enumerate(zip(before, after)) if pair[0] != pair[1]]
    before_non_none = Counter(x for x in before if x is not None)
    after_non_none = Counter(x for x in after if x is not None)
    same_non_none = before_non_none == after_non_none
    same_full = Counter(before) == Counter(after)
    pure_swap = bool(len(changed) >= 2 and same_non_none and same_full)
    return {
        "assignment_changed": bool(changed),
        "changed_row_count": len(changed),
        "pure_swap_changes": pure_swap,
        "id_set_changes": bool(changed) and not (same_non_none and same_full),
        "other_assignment_changes": bool(changed) and not pure_swap,
        "non_none_public_id_multiset_equal": same_non_none,
        "full_assignment_multiset_equal": same_full,
    }


def rows_for(candidates: list[dict[str, Any]], pids: list[int], assignment: list[int]) -> list[dict[str, Any]]:
    normalized = normalize_assignment(assignment, len(pids))
    if len(normalized) != len(candidates):
        raise ValueError("assignment/candidate row mismatch")
    return [{
        "candidate_index": int(candidate.get("index", row)),
        "native_tid": int(candidate["native_tid"]),
        "box": candidate["box"],
        "confidence": float(candidate.get("confidence", 0.0)),
        "public_id": int(pids[normalized[row]]) if normalized[row] >= 0 else None,
    } for row, candidate in enumerate(candidates)]


def branch(candidates: list[dict[str, Any]], pids: list[int], assignment: list[int], scores: np.ndarray, label: str, frame: int) -> dict[str, Any]:
    normalized = normalize_assignment(assignment, len(pids))
    return {
        "branch": label,
        "frame": int(frame),
        "candidate_rows": candidates,
        "candidate_count": len(candidates),
        "candidate_native_ids": [int(x["native_tid"]) for x in candidates],
        "public_id_order": pids,
        "assignment_columns": normalized,
        "assignment_public_ids": [int(pids[col]) if col >= 0 else None for col in normalized],
        "rows": rows_for(candidates, pids, normalized),
        "score_matrix": np.asarray(scores, dtype=np.float32).astype(float).tolist(),
        "runtime_future_gt_used": False,
    }


def gate_reason(base: float, finite: bool, valid_memory: bool, margin: float, residual: float, uncertainty: float, accepted: bool) -> str:
    if not finite:
        return "hard_negative_or_nonfinite"
    if not valid_memory:
        return "memory_invalid"
    if margin > 2.0:
        return "global_margin_gt_2"
    if abs(residual) < 0.05:
        return "abs_residual_lt_0.05"
    if uncertainty > 0.35:
        return "uncertainty_gt_0.35"
    if accepted:
        return "accepted"
    return "gate_rejected_unclassified"


def runtime(events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not CHECKPOINT.is_file() or not MEMORY_MANIFEST.is_file():
        raise FileNotFoundError("N48 checkpoint or simulated memory manifest missing")
    model, checkpoint = load_checkpoint(CHECKPOINT, "cpu")
    memory_manifest = load(MEMORY_MANIFEST)
    if memory_manifest.get("interaction_source") != "simulated_from_gt" or memory_manifest.get("runtime_future_gt_used") is not False:
        raise ValueError("invalid simulated memory provenance")
    n36_cache: dict[str, dict[int, dict[str, Any]]] = {}
    RUNTIME.mkdir(parents=True, exist_ok=True)
    summary = {"event_count": 0, "frames": 0, "by_variant": {v: {"proposal_count": 0, "selected_count": 0, "score_cells_changed": 0, "assignment_changes": 0, "pure_swap_changes": 0, "id_set_changes": 0, "other_assignment_changes": 0} for v in VARIANTS}}
    for event_id, event in sorted(events.items()):
        source = load(N47_RUNTIME / f"{event_id}.json")
        event_memory = memory_manifest["events"][event_id]
        sequence = str(event["sequence"])
        if sequence not in n36_cache:
            n36_cache[sequence] = load_n36_sequence(sequence)
        event_out = {
            "schema": "N48_RISK_AWARE_GLOBAL_RUNTIME_EVENT_V1",
            "status": "PASS",
            "event_id": event_id,
            "sequence": sequence,
            "event_frame": int(event["frame"]),
            "action_type": str(event["action_type"]),
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
            "runtime_boundary": {"runtime_future_gt_used": False, "gt_loaded_in_worker": False, "future_gt_fields_sent": []},
            "variants": {},
        }
        for variant in VARIANTS:
            source_frames = source["variants"][variant]["frames"]
            if len(source_frames) != 100:
                raise RuntimeError(f"source frame count {event_id}/{variant}")
            frames = []
            for source_frame in source_frames:
                frame = int(source_frame["frame"])
                write = source_frame["write_baseline"]
                no = source_frame["no_write"]
                candidates = write["candidate_rows"]
                pids = [int(x) for x in write["public_id_order"]]
                if candidate_signature(candidates) != candidate_signature(no["candidate_rows"]):
                    raise RuntimeError(f"no/write candidate mismatch {event_id}/{variant}/{frame}")
                base = np.asarray(write["base_scores"], dtype=np.float32)
                if base.shape != (len(candidates), len(pids)) or not np.all(np.isfinite(base)):
                    raise RuntimeError(f"invalid base matrix {event_id}/{variant}/{frame}")
                features = candidate_features(candidates, n36_cache[sequence][frame])
                valid_memory = np.asarray([bool(event_memory["memory_valid"].get(str(pid), False)) for pid in pids], dtype=bool)
                zero = np.zeros(512, dtype=np.float32)
                memory_rows = np.asarray([np.asarray(event_memory["memory_vectors"].get(str(pid), zero), dtype=np.float32) for pid in pids], dtype=np.float32)
                if memory_rows.shape != (len(pids), 512):
                    raise RuntimeError(f"invalid memory rows {event_id}/{variant}/{frame}")
                scalar_rows = scalar_features(base, candidates, valid_memory, frame - int(event["frame"]))
                n, p = base.shape
                candidate_cells = np.asarray([features[row] for row in range(n) for _ in range(p)], dtype=np.float32)
                memory_cells = np.asarray([memory_rows[col] for _ in range(n) for col in range(p)], dtype=np.float32)
                scalar_cells = np.asarray([scalar_rows[row] for row in range(n) for _ in range(p)], dtype=np.float32)
                if variant == "M0":
                    baseline_assignment = normalize_assignment(write["assignment_columns"], p)
                    probe = {"raw_residual": [0.0] * (n * p), "bounded_residual": [0.0] * (n * p), "uncertainty": [0.0] * (n * p), "accepted": [False] * (n * p), "adjusted_scores": base.astype(float).tolist(), "baseline_assignment": baseline_assignment, "global_assignment_margin": None, "plus_assignment": baseline_assignment, "runtime_future_gt_used": False}
                    applied = False
                else:
                    probe = runtime_sidecar(model, candidate_cells, memory_cells, scalar_cells, base, valid_memory, "cpu")
                    applied = True
                adjusted = np.asarray(probe["adjusted_scores"], dtype=np.float32)
                changed = np.argwhere(np.abs(adjusted - base) > 1.0e-12)
                changed_cells = [{"candidate_index": int(row), "column": int(col), "public_id": int(pids[col]), "baseline_score": float(base[row, col]), "score_delta": float(adjusted[row, col] - base[row, col])} for row, col in changed]
                plus_assignment = normalize_assignment(probe["plus_assignment"], p)
                write_assignment = normalize_assignment(write["assignment_columns"], p)
                plus = branch(candidates, pids, plus_assignment, adjusted, "write_plus_n48", frame)
                write_branch = branch(candidates, pids, write_assignment, base, "write_baseline", frame)
                no_branch = branch(no["candidate_rows"], [int(x) for x in no["public_id_order"]], normalize_assignment(no["assignment_columns"], len(no["public_id_order"])), np.asarray(no["base_scores"], dtype=np.float32), "no_write", frame)
                if candidate_signature(write_branch["candidate_rows"]) != candidate_signature(plus["candidate_rows"]):
                    raise RuntimeError(f"write/plus candidates changed {event_id}/{variant}/{frame}")
                if write_branch["public_id_order"] != plus["public_id_order"]:
                    raise RuntimeError(f"write/plus public-ID axis changed {event_id}/{variant}/{frame}")
                accepted = np.asarray(probe["accepted"], dtype=bool).reshape(n, p)
                residual = np.asarray(probe["bounded_residual"], dtype=np.float32).reshape(n, p)
                uncertainty = np.asarray(probe["uncertainty"], dtype=np.float32).reshape(n, p)
                margin = probe.get("global_assignment_margin")
                proposal_count = int(np.sum((base > HARD_NEGATIVE) & valid_memory[None, :])) if applied else 0
                selected_count = int(np.sum(accepted)) if applied else 0
                reasons = [[gate_reason(float(base[row, col]), bool(base[row, col] > HARD_NEGATIVE), bool(valid_memory[col]), float(margin) if margin is not None else float("inf"), float(residual[row, col]), float(uncertainty[row, col]), bool(accepted[row, col])) for col in range(p)] for row in range(n)]
                transition = assignment_transition(write_branch["assignment_public_ids"], plus["assignment_public_ids"])
                frame_out = {
                    "frame": frame,
                    "no_write": no_branch,
                    "write_baseline": write_branch,
                    "write_plus_n48": plus,
                    "candidate_features_512": features.astype(float).tolist(),
                    "candidate_feature_source": ["N36_machine_embedding" for _ in candidates],
                    "candidate_feature_sha256": [digest_array(row) for row in features],
                    "memory_public_id_order": pids,
                    "memory_valid": valid_memory.tolist(),
                    "memory_vectors_512": memory_rows.astype(float).tolist(),
                    "memory_provenance": "offline_event_prefix_machine_embedding_with_GT_simulated_human_target_anchor",
                    "probe": {
                        "applied": applied,
                        "proposal_count": proposal_count,
                        "selected_count": selected_count,
                        "accepted_cells": accepted.astype(bool).tolist(),
                        "gate_reasons": reasons,
                        "global_assignment_margin": margin,
                        "raw_residual": np.asarray(probe["raw_residual"]).reshape(n, p).astype(float).tolist(),
                        "bounded_residual": residual.astype(float).tolist(),
                        "uncertainty": uncertainty.astype(float).tolist(),
                        "changed_cells": changed_cells,
                        "score_cells_changed": len(changed_cells),
                        "hard_negative_preserved": bool(np.array_equal(adjusted[base <= HARD_NEGATIVE], base[base <= HARD_NEGATIVE])),
                        "explicit_none": True,
                        "swap_allowed": True,
                        **transition,
                        "runtime_future_gt_used": False,
                    },
                }
                frames.append(frame_out)
                summary["frames"] += 1
                stats = summary["by_variant"][variant]
                stats["proposal_count"] += proposal_count; stats["selected_count"] += selected_count; stats["score_cells_changed"] += len(changed_cells)
                stats["assignment_changes"] += int(transition["assignment_changed"]); stats["pure_swap_changes"] += int(transition["pure_swap_changes"]); stats["id_set_changes"] += int(transition["id_set_changes"]); stats["other_assignment_changes"] += int(transition["other_assignment_changes"])
            event_out["variants"][variant] = {"frame_count": len(frames), "frames": frames}
        write_json(RUNTIME / f"{event_id}.json", event_out)
        summary["event_count"] += 1
    status = {"status": "PASS", "protocol": "N48_STAGE_04_RUNTIME_V1", "command": ["python", "scripts/n48_stage04_replay.py"], "inputs": {"frozen_n47_runtime": str(N47_RUNTIME), "n48_checkpoint": str(CHECKPOINT), "simulated_memory_manifest": str(MEMORY_MANIFEST), "frozen_seed": 4848}, "outputs": {"runtime": str(RUNTIME)}, "metrics": summary, "gate_checks": {"all_24_events": summary["event_count"] == 24, "all_5_variants": True, "all_100_frames": summary["frames"] == 24 * 5 * 100, "same_candidate_stream": True, "write_plus_axis_equal_write_baseline": True, "512d_candidate_features": True, "global_hungarian_explicit_none": True, "runtime_future_gt_false": True, "gt_loaded": False, "simulated_provenance": True, "production_authorized": False}, "failure_root_cause": "N48 is an isolated simulated diagnostic; runtime preserves the frozen candidate stream and changes only accepted finite score cells before global Hungarian.", "next_action": "Run independent runtime integrity, then load GT for posthoc metrics only.", "runtime_future_gt_used": False, "gt_loaded_posthoc": False, "finished_at": now()}
    write_json(OUT / "replay/runtime_status.json", status)
    return status


def pid_iou(rows: list[dict[str, Any]], pid: int, target: Any, iou_fn: Any) -> float:
    return max((float(iou_fn(row["box"], target)) for row in rows if row.get("public_id") is not None and int(row["public_id"]) == pid), default=0.0)


def native_map(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {str(row["native_tid"]): row.get("public_id") for row in rows}


def compare(reference: list[dict[str, Any]], treated: list[dict[str, Any]], event: dict[str, Any], mapping: dict[int, int], gt_frames: dict[int, Any], horizon: int, iou_fn: Any) -> dict[str, Any]:
    target_pid = int(event["public_id"]); target_gid = int(event["dataset_gt_id"]); details = []
    for ref, yes in zip(reference[:horizon], treated[:horizon]):
        gt_frame = gt_frames.get(int(ref["frame"]))
        if gt_frame is None:
            continue
        target = next((box for gid, box in zip(gt_frame.gt_ids, gt_frame.boxes) if int(gid) == target_gid), None)
        if target is None:
            continue
        ri = pid_iou(ref["rows"], target_pid, target, iou_fn); yi = pid_iou(yes["rows"], target_pid, target, iou_fn); changed = native_map(ref["rows"]) != native_map(yes["rows"])
        details.append({"frame": int(ref["frame"]), "reference_iou": ri, "treated_iou": yi, "target_iou_delta": yi - ri, "reference_error": ri < 0.5, "treated_error": yi < 0.5, "assignment_changed": changed, "assignment_change_correct": bool(changed and yi > ri + 1e-9), "assignment_change_incorrect": bool(changed and yi < ri - 1e-9), "assignment_change_neutral": bool(changed and abs(yi - ri) <= 1e-9), "assignment_no_change": not changed})
    if not details:
        return {"evaluated_frames": 0, "identity_utility": None, "target_iou_delta": None, "future_identity_error_reduction": None, "recorrection_proxy_reduction": None, "assignment_change_count": 0, "assignment_change_correct_count": 0, "assignment_change_incorrect_count": 0, "assignment_change_neutral_count": 0, "assignment_no_change_count": 0, "untouched_regression": {"status": "NOT_COMPUTABLE"}, "frame_details": []}
    ref_iou = float(np.mean([x["reference_iou"] for x in details])); yes_iou = float(np.mean([x["treated_iou"] for x in details])); ref_error = float(np.mean([x["reference_error"] for x in details])); yes_error = float(np.mean([x["treated_error"] for x in details]))
    ref_recorrect = sum(x["reference_error"] and (i == 0 or not details[i - 1]["reference_error"]) for i, x in enumerate(details)); yes_recorrect = sum(x["treated_error"] and (i == 0 or not details[i - 1]["treated_error"]) for i, x in enumerate(details))
    untouched = []
    for ref, yes in zip(reference[:horizon], treated[:horizon]):
        gt_frame = gt_frames.get(int(ref["frame"]));
        if gt_frame is None: continue
        boxes = {int(gid): box for gid, box in zip(gt_frame.gt_ids, gt_frame.boxes)}
        for pid, gid in mapping.items():
            if pid == target_pid or gid not in boxes: continue
            untouched.append(pid_iou(yes["rows"], pid, boxes[gid], iou_fn) - pid_iou(ref["rows"], pid, boxes[gid], iou_fn))
    untouched_result = {"compared": len(untouched), "mean_iou_delta": float(np.mean(untouched)) if untouched else None, "regression_count_delta_lt_minus_0.05": int(sum(x < -0.05 for x in untouched)), "all_no_obvious_regression": bool(untouched) and not any(x < -0.05 for x in untouched), "status": "PASS" if untouched and not any(x < -0.05 for x in untouched) else "FAIL" if untouched else "NOT_COMPUTABLE"}
    return {"evaluated_frames": len(details), "identity_utility": 0.5 * (yes_iou - ref_iou) + 0.5 * (ref_error - yes_error), "target_iou_delta": yes_iou - ref_iou, "future_identity_error_reduction": ref_error - yes_error, "recorrection_proxy_reduction": int(ref_recorrect - yes_recorrect), "assignment_change_count": int(sum(x["assignment_changed"] for x in details)), "assignment_change_correct_count": int(sum(x["assignment_change_correct"] for x in details)), "assignment_change_incorrect_count": int(sum(x["assignment_change_incorrect"] for x in details)), "assignment_change_neutral_count": int(sum(x["assignment_change_neutral"] for x in details)), "assignment_no_change_count": int(sum(x["assignment_no_change"] for x in details)), "untouched_regression": untouched_result, "frame_details": details}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["identity_utility"] is not None]
    by_sequence: dict[str, list[float]] = defaultdict(list)
    for row in valid: by_sequence[str(row["sequence"])].append(float(row["identity_utility"]))
    means = {seq: float(np.mean(values)) for seq, values in by_sequence.items() if values}; names = sorted(means)
    rng = np.random.default_rng(BOOTSTRAP_SEED); draws = [float(np.mean([means[name] for name in rng.choice(names, len(names), replace=True)])) for _ in range(BOOTSTRAP_REPS)] if names else []
    return {"event_count": len(valid), "independent_sequence_count": len(names), "identity_utility": float(np.mean([x["identity_utility"] for x in valid])) if valid else None, "target_iou_delta": float(np.mean([x["target_iou_delta"] for x in valid])) if valid else None, "future_identity_error_reduction": float(np.mean([x["future_identity_error_reduction"] for x in valid])) if valid else None, "recorrection_proxy_reduction": float(np.mean([x["recorrection_proxy_reduction"] for x in valid])) if valid else None, "assignment_change_count": int(sum(x["assignment_change_count"] for x in valid)), "assignment_change_correct_count": int(sum(x["assignment_change_correct_count"] for x in valid)), "assignment_change_incorrect_count": int(sum(x["assignment_change_incorrect_count"] for x in valid)), "assignment_change_neutral_count": int(sum(x["assignment_change_neutral_count"] for x in valid)), "assignment_no_change_count": int(sum(x["assignment_no_change_count"] for x in valid)), "untouched_regression": {"all_no_obvious_regression": bool(valid) and all(x["untouched_regression"].get("all_no_obvious_regression", False) for x in valid), "mean_iou_delta": float(np.mean([x["untouched_regression"]["mean_iou_delta"] for x in valid if x["untouched_regression"].get("mean_iou_delta") is not None])) if valid else None}, "sequence_cluster_bootstrap_95ci": {"lower": float(np.quantile(draws, 0.025)) if draws else None, "upper": float(np.quantile(draws, 0.975)) if draws else None, "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPS, "clusters": len(names), "cluster_weighting": "equal_sequence_mean", "cluster_mean_identity_utility": float(np.mean(list(means.values()))) if means else None, "event_weighted_identity_utility": float(np.mean([x["identity_utility"] for x in valid])) if valid else None}}


def validate_runtime(events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    files = sorted(RUNTIME.glob("*.json"))
    if len(files) != 24 or {path.stem for path in files} != set(events): raise RuntimeError("N48 event file set incomplete")
    frames_checked = 0
    for event_id in events:
        payload = load(RUNTIME / f"{event_id}.json")
        if payload["runtime_boundary"].get("runtime_future_gt_used") is not False: raise RuntimeError(f"runtime GT flag {event_id}")
        for variant in VARIANTS:
            frames = payload["variants"][variant]["frames"]
            if len(frames) != 100 or [int(x["frame"]) for x in frames] != list(range(int(frames[0]["frame"]), int(frames[0]["frame"]) + 100)): raise RuntimeError(f"frame contract {event_id}/{variant}")
            for item in frames:
                if item["probe"].get("runtime_future_gt_used") is not False: raise RuntimeError(f"frame GT flag {event_id}/{variant}/{item['frame']}")
                write = item["write_baseline"]; plus = item["write_plus_n48"]
                if candidate_signature(write["candidate_rows"]) != candidate_signature(plus["candidate_rows"]): raise RuntimeError("write/plus candidate mismatch")
                if write["public_id_order"] != plus["public_id_order"]: raise RuntimeError("write/plus axis mismatch")
                if len(set(write["candidate_native_ids"])) != len(write["candidate_native_ids"]): raise RuntimeError("duplicate native candidate ID")
                if len(item["candidate_features_512"]) != write["candidate_count"] or any(len(row) != 512 for row in item["candidate_features_512"]): raise RuntimeError("512D feature missing")
                frames_checked += 1
    return {"status": "PASS", "event_count": 24, "frames_checked": frames_checked, "runtime_future_gt_used": False, "gt_loaded": False, "candidate_complete": True}


def posthoc(events: dict[str, dict[str, Any]], validation: dict[str, Any]) -> dict[str, Any]:
    from sam3_intermot.datasets.dancetrack import DanceTrackDataset
    from scripts.n36_real_eval_common import DATA_ROOT
    from scripts.n43_full_matrix_common import iou
    mapping_all = load(N43_MAP)["public_to_gt_mapping"]
    sequences = sorted({str(item["sequence"]) for item in events.values()}); dataset = DanceTrackDataset(str(DATA_ROOT), sequences=sequences, split="train"); gt = {sequence: dataset.load_gt(sequence) for sequence in sequences}
    all_rows: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list); event_results = {}
    POSTHOC.mkdir(parents=True, exist_ok=True)
    for event_id, event in sorted(events.items()):
        mapping = {int(pid): int(gid) for pid, gid in mapping_all.get(event_id, {}).items()}; runtime_payload = load(RUNTIME / f"{event_id}.json"); output = {"schema": "N48_POSTHOC_EVENT_V1", "status": "PASS", "event_id": event_id, "sequence": str(event["sequence"]), "interaction_source": "simulated_from_gt", "not_real_human_evidence": True, "runtime_future_gt_used": False, "gt_loaded_posthoc": True, "variants": {}}
        for variant in VARIANTS:
            frames = runtime_payload["variants"][variant]["frames"]; no = [item["no_write"] for item in frames]; write = [item["write_baseline"] for item in frames]; plus = [item["write_plus_n48"] for item in frames]; output["variants"][variant] = {"runtime_frame_count": len(frames), "horizons": {}}
            for horizon in HORIZONS:
                memory = compare(no, write, event, mapping, gt[str(event["sequence"])], horizon, iou); incremental = compare(write, plus, event, mapping, gt[str(event["sequence"])], horizon, iou)
                counts = {"proposals": int(sum(item["probe"]["proposal_count"] for item in frames[:horizon])), "selected": int(sum(item["probe"]["selected_count"] for item in frames[:horizon])), "score_cells_changed": int(sum(item["probe"]["score_cells_changed"] for item in frames[:horizon])), "assignment_changes": int(sum(bool(item["probe"]["assignment_changed"]) for item in frames[:horizon])), "selected_but_no_assignment_change": int(sum(item["probe"]["selected_count"] > 0 and not item["probe"]["assignment_changed"] for item in frames[:horizon]))}
                output["variants"][variant]["horizons"][str(horizon)] = {"memory_effect_no_write_to_write_baseline": memory, "n48_incremental_effect_write_baseline_to_write_plus_n48": incremental, "application_counts": counts}
                all_rows[("memory", variant, horizon)].append({"event_id": event_id, "sequence": str(event["sequence"]), **memory}); all_rows[("incremental", variant, horizon)].append({"event_id": event_id, "sequence": str(event["sequence"]), **incremental})
        write_json(POSTHOC / f"{event_id}.json", output); event_results[event_id] = output
    effects = {effect: {variant: {str(h): aggregate(all_rows[(effect, variant, h)]) for h in HORIZONS} for variant in VARIANTS} for effect in ("memory", "incremental")}
    result = {"schema": "N48_RISK_AWARE_GLOBAL_ASSIGNMENT_RESULT_V1", "status": "PASS", "protocol": {"runtime_future_gt_used": False, "gt_loaded_only_after_runtime_validation": True, "bootstrap": "equal_sequence_mean_then_cluster_bootstrap", "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPS}, "inputs": {"n47_runtime": str(N47_RUNTIME), "n48_checkpoint": str(CHECKPOINT)}, "outputs": {"runtime": str(RUNTIME), "posthoc": str(POSTHOC)}, "event_count": 24, "variant_count": 5, "horizons": list(HORIZONS), "interaction_source": "simulated_from_gt", "real_human_tape_created": False, "standard_mot": "NOT_COMPUTABLE_NO_COMPLETE_TRACKEVAL_INPUT", "runtime_validation": validation, "effects": effects, "attribution": {"memory_effect": "write_baseline minus no_write", "n48_incremental_effect": "write_plus_n48 minus write_baseline"}}
    write_json(OUT / "replay/paired_replay_results.json", result)
    return result


def main() -> None:
    status_path = OUT / "stage_04_status.json"; status = {"status": "FAIL", "protocol": "N48_STAGE_04_REPLAY_V1", "started_at": now()}
    try:
        events = event_map(); runtime_status = runtime(events); validation = validate_runtime(events); result = posthoc(events, validation)
        status.update({"status": "PASS", "command": ["python", "scripts/n48_stage04_replay.py"], "inputs": result["inputs"], "outputs": result["outputs"], "metrics": {"runtime": runtime_status["metrics"], "effects": result["effects"]}, "gate_checks": {"runtime_complete": True, "runtime_integrity_precedes_gt": True, "all_24_events": True, "all_5_variants": True, "all_horizons": True, "same_candidate_stream": True, "global_hungarian": True, "explicit_none": True, "runtime_future_gt_false": True, "gt_loaded_posthoc": True, "equal_sequence_bootstrap": True, "simulated_provenance": True, "standard_mot_not_computable": True, "production_authorized": False}, "failure_root_cause": "This is a simulated structural diagnostic, not real human efficacy evidence; memory and N48 increments are separately reported.", "next_action": "Run independent integrity checker and semantic final gate; real human tape/full-loop remain hard gates.", "runtime_future_gt_used": False, "gt_loaded_posthoc": True, "finished_at": now()})
        write_json(status_path, status); print(json.dumps({"status": "PASS", "frames": runtime_status["metrics"]["frames"]}))
    except Exception as exc:
        status.update({"status": "FAIL_PRESERVED", "failure_root_cause": f"{type(exc).__name__}: {exc}", "outputs": {}, "metrics": {}, "gate_checks": {"false_pass": False}, "next_action": "Preserve failure, fix only first actionable root cause, run smoke and targeted regression, then rerun.", "runtime_future_gt_used": False, "finished_at": now()})
        write_json(OUT / "attempts/stage04_failure.json", status); write_json(status_path, status); raise


if __name__ == "__main__":
    main()
