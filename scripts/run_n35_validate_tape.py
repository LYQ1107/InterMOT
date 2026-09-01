#!/usr/bin/env python3
"""CPU-only integrity audit for the N35 candidate-complete tape.

This validator deliberately reads only the exported JSONL and its done/
manifest metadata.  It does not open DanceTrack annotations and therefore
cannot introduce future ground truth into the runtime tape.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import tempfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def finite_array(value: Any) -> bool:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(array)))


def decode_mask(payload: dict[str, Any]) -> np.ndarray:
    if payload.get("encoding") != "packbits_zlib_base64":
        raise ValueError("unsupported mask encoding")
    shape = tuple(int(item) for item in payload["shape"])
    if not shape or any(item <= 0 for item in shape):
        raise ValueError("invalid mask shape")
    packed = zlib.decompress(base64.b64decode(payload["data"].encode("ascii")))
    count = int(np.prod(shape))
    values = np.unpackbits(
        np.frombuffer(packed, dtype=np.uint8),
        bitorder=str(payload.get("bitorder", "little")),
        count=count,
    )
    array = values.reshape(shape).astype(bool)
    digest = hashlib.sha256(array.tobytes()).hexdigest()
    if digest != payload.get("sha256"):
        raise ValueError("mask sha256 mismatch")
    return array


def check_matrix(audit: dict[str, Any], field: str, n_state: int, n_candidate: int, errors: list[str]) -> None:
    value = audit.get(field)
    if not isinstance(value, list) or len(value) != n_state:
        errors.append(f"{field}_row_shape")
        return
    for row in value:
        if not isinstance(row, list) or len(row) != n_candidate or not finite_array(row):
            errors.append(f"{field}_shape_or_nonfinite")
            return


def validate_row(row: dict[str, Any], sequence: str, expected_frame: int, stats: dict[str, int]) -> list[str]:
    errors: list[str] = []
    if row.get("record_type") != "candidate_frame":
        errors.append("record_type")
    if row.get("sequence") != sequence:
        errors.append("sequence")
    if row.get("split") != "train/train_fold":
        errors.append("split")
    if int(row.get("frame", -1)) != expected_frame:
        errors.append("frame_order")
    if row.get("candidate_complete") is not True or row.get("candidate_set_complete") is not True:
        errors.append("candidate_complete")
    if row.get("runtime_future_gt_used") is not False or row.get("runtime_gt_read") is not False:
        errors.append("runtime_gt_leakage_flags")
    candidates = row.get("candidates")
    if not isinstance(candidates, list):
        return [*errors, "candidates_not_list"]
    candidate_indices = [item.get("candidate_index") for item in candidates if isinstance(item, dict)]
    if candidate_indices != list(range(len(candidates))):
        errors.append("candidate_order")
    native_ids: list[int] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            errors.append(f"candidate_{index}_not_object")
            continue
        if candidate.get("candidate_index") != index:
            errors.append(f"candidate_{index}_index")
        native_tid = candidate.get("native_tid")
        if not isinstance(native_tid, int) or isinstance(native_tid, bool):
            errors.append(f"candidate_{index}_native_tid")
        else:
            native_ids.append(native_tid)
        box = candidate.get("box")
        if not isinstance(box, list) or len(box) != 4 or not finite_array(box):
            errors.append(f"candidate_{index}_box")
        for scalar in (candidate.get("confidence"), candidate.get("presence_score")):
            if scalar is not None and not finite_number(scalar):
                errors.append(f"candidate_{index}_score")
        mask = candidate.get("mask")
        if not isinstance(mask, dict):
            errors.append(f"candidate_{index}_mask_missing")
        else:
            try:
                decode_mask(mask)
                stats["mask_decode_ok"] += 1
            except Exception as exc:
                errors.append(f"candidate_{index}_mask:{type(exc).__name__}")
        embedding = candidate.get("embedding")
        if not isinstance(embedding, list) or len(embedding) != 512 or not finite_array(embedding):
            errors.append(f"candidate_{index}_embedding")
        else:
            norm = float(np.linalg.norm(np.asarray(embedding, dtype=np.float32)))
            if not math.isfinite(norm) or abs(norm - 1.0) > 2e-3:
                errors.append(f"candidate_{index}_embedding_norm")
        if candidate.get("embedding_status") != "MACHINE_ROI_FALLBACK":
            errors.append(f"candidate_{index}_embedding_status")
        if candidate.get("feature_source") != "machine_roi_fallback":
            errors.append(f"candidate_{index}_feature_source")
        if "public_id" in candidate or "gt_id" in candidate or "dataset_identity" in candidate:
            errors.append(f"candidate_{index}_identity_leakage")
    if len(set(native_ids)) != len(native_ids):
        errors.append("duplicate_native_tid")

    audit = row.get("association_audit")
    if not isinstance(audit, dict):
        return [*errors, "association_audit_missing"]
    public_ids = audit.get("public_id_order")
    candidate_order = audit.get("candidate_order")
    if not isinstance(public_ids, list) or len(set(public_ids)) != len(public_ids):
        errors.append("public_id_order")
    if candidate_order != list(range(len(candidates))):
        errors.append("audit_candidate_order")
    n_state = len(public_ids) if isinstance(public_ids, list) else 0
    for field in (
        "public_id_score_matrix",
        "public_id_base_score_matrix",
        "public_id_appearance_score_matrix",
        "public_id_fused_score_matrix",
    ):
        check_matrix(audit, field, n_state, len(candidates), errors)
    mapping = audit.get("public_id_to_native_tid")
    if not isinstance(mapping, dict):
        errors.append("public_id_to_native_tid")
    else:
        if set(mapping) != {str(item) for item in public_ids}:
            errors.append("public_id_mapping_keys")
        mapped = [value for value in mapping.values() if value is not None]
        if len(set(mapped)) != len(mapped):
            errors.append("public_id_mapping_duplicate_native")
        if any(value is not None and value not in native_ids for value in mapped):
            errors.append("public_id_mapping_unknown_native")
    pairs = audit.get("assignment_pairs_after_scope")
    if not isinstance(pairs, list):
        errors.append("assignment_pairs_after_scope")
    else:
        pair_candidates = [item.get("candidate_index") for item in pairs if isinstance(item, dict)]
        if len(set(pair_candidates)) != len(pair_candidates):
            errors.append("assignment_duplicate_candidate")
        if any(item not in range(len(candidates)) for item in pair_candidates):
            errors.append("assignment_unknown_candidate")
        for item in pairs:
            if not isinstance(item, dict) or not finite_number(item.get("score")):
                errors.append("assignment_nonfinite_score")
                break
    stats["rows"] += 1
    stats["candidates"] += len(candidates)
    return errors


def validate(args: argparse.Namespace) -> dict[str, Any]:
    tape_root = args.tape_root.resolve()
    manifest_path = tape_root / "tape_manifest.json"
    errors: list[dict[str, Any]] = []
    stats = {
        "sequence_expected": 0,
        "sequence_pass": 0,
        "rows": 0,
        "candidates": 0,
        "mask_decode_ok": 0,
        "done_failures": 0,
    }
    if not manifest_path.is_file():
        result = {
            "protocol": "N35_CPU_TAPE_INTEGRITY_AUDIT",
            "status": "FAIL",
            "tape_root": str(tape_root),
            "errors": [{"scope": "manifest", "reasons": ["missing"]}],
            "stats": stats,
        }
        return result
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sequences = [str(item) for item in manifest.get("sequences", [])]
    stats["sequence_expected"] = len(sequences)
    if manifest.get("runtime_future_gt_used") is not False or manifest.get("runtime_gt_read") is not False:
        errors.append({"scope": "manifest", "reasons": ["runtime_gt_leakage_flags"]})
    if manifest.get("split") != "train/train_fold":
        errors.append({"scope": "manifest", "reasons": ["split"]})
    for sequence in sequences:
        done_path = tape_root / "done" / f"{sequence}.json"
        frame_path = tape_root / "frames" / f"{sequence}.jsonl"
        if not done_path.is_file():
            errors.append({"sequence": sequence, "scope": "done", "reasons": ["missing"]})
            continue
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("status") != "PASS":
            stats["done_failures"] += 1
            errors.append({"sequence": sequence, "scope": "done", "reasons": ["status_not_pass"]})
            continue
        if not frame_path.is_file():
            errors.append({"sequence": sequence, "scope": "frames", "reasons": ["missing"]})
            continue
        sequence_errors: list[str] = []
        row_count = 0
        sequence_candidate_count = 0
        with frame_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    sequence_errors.append(f"line_{line_number}:json:{exc.msg}")
                    continue
                row_errors = validate_row(row, sequence, row_count, stats)
                sequence_errors.extend(f"frame_{row_count}:{item}" for item in row_errors)
                if isinstance(row.get("candidates"), list):
                    sequence_candidate_count += len(row["candidates"])
                row_count += 1
        expected_count = int(done.get("frame_count", -1))
        if row_count != expected_count:
            sequence_errors.append(f"frame_count:{row_count}!={expected_count}")
        if int(done.get("candidate_count", -1)) != sequence_candidate_count:
            sequence_errors.append(
                f"candidate_count:{sequence_candidate_count}!={int(done.get('candidate_count', -1))}"
            )
        if sequence_errors:
            errors.append({"sequence": sequence, "scope": "frames", "reasons": sequence_errors[:64]})
        else:
            stats["sequence_pass"] += 1

    expected_frame_count = int(manifest.get("frame_count", -1))
    expected_candidate_count = int(manifest.get("candidate_count", -1))
    if stats["rows"] != expected_frame_count:
        errors.append({"scope": "manifest", "reasons": [f"frame_count:{stats['rows']}!={expected_frame_count}"]})
    if stats["candidates"] != expected_candidate_count:
        errors.append({"scope": "manifest", "reasons": [f"candidate_count:{stats['candidates']}!={expected_candidate_count}"]})
    if stats["sequence_pass"] != stats["sequence_expected"] or stats["done_failures"]:
        errors.append({"scope": "manifest", "reasons": ["sequence_coverage"]})
    status = "PASS" if not errors else "FAIL"
    return {
        "protocol": "N35_CPU_TAPE_INTEGRITY_AUDIT",
        "status": status,
        "tape_root": str(tape_root),
        "manifest": str(manifest_path),
        "candidate_complete": status == "PASS",
        "candidate_set_complete": status == "PASS",
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "stats": stats,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tape-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/n35/tape_integrity_audit.json")
    parser.add_argument("--stage-status", type=Path, default=ROOT / "outputs/n35/stage_02_status.json")
    args = parser.parse_args()
    result = validate(args)
    atomic_json(args.output.resolve(), result)
    stage = {
        "stage": "02_tape_export_and_integrity",
        "status": result["status"],
        "commands": [
            "run_n35_export_tape.py (real train/train_fold candidate export)",
            f"run_n35_validate_tape.py --tape-root {args.tape_root}",
        ],
        "artifacts": [str(args.output.resolve()), str((args.tape_root / "tape_manifest.json").resolve())],
        "errors": result.get("errors", []),
        "next_action": (
            "start four-GPU real train/train_fold export"
            if result["status"] == "PASS" and "smoke" in str(args.tape_root)
            else "continue N35 only after this integrity gate is PASS"
        ),
    }
    atomic_json(args.stage_status.resolve(), stage)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
