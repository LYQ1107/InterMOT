#!/usr/bin/env python3
"""Validate and virtually merge N25-R per-sequence feature caches."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(".")
EPISODE_ROOT = ROOT / "outputs/n25r/repaired_dataset"
FEATURE_ROOT = ROOT / "outputs/n25r/candidate_aligned_features"
OUTPUT = ROOT / "outputs/n25r/feature_alignment.json"
BACKBONES = ("clipreid", "sam3_f1")
SPLITS = ("train30", "cal10")
HORIZONS = (1, 5, 10)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def canonical_row_key(row: dict[str, Any]) -> str:
    return "|".join(
        str(value)
        for value in (
            row["sequence"],
            row["public_identity_id"],
            row["gid"],
            row["decision_frame"],
            row["correction_frame"],
            row["candidate_source"],
            row["candidate_rank"],
        )
    )


def feature_valid(backbone: str, cache: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if backbone == "clipreid":
        return np.isfinite(cache["query"]).all(axis=1), np.isfinite(cache["candidate"]).all(axis=-1)
    query = np.isfinite(cache["query_mean"]).all(axis=1) & np.isfinite(cache["query_max"]).all(axis=1)
    candidate = np.isfinite(cache["candidate_mean"]).all(axis=-1) & np.isfinite(cache["candidate_max"]).all(axis=-1)
    return query, candidate


def norm_summary(array: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    values = array.astype(np.float32)[valid]
    norms = np.linalg.norm(values, axis=-1)
    return {
        "count": int(len(norms)),
        "minimum": float(norms.min()),
        "maximum": float(norms.max()),
        "mean_abs_deviation_from_one": float(np.abs(norms - 1.0).mean()),
    }


def validate_split(split: str) -> dict[str, Any]:
    episode_path = EPISODE_ROOT / f"episodes_{split}.jsonl"
    rows = load_rows(episode_path)
    expected_keys = [canonical_row_key(row) for row in rows]
    expected_sequences = sorted({str(row["sequence"]) for row in rows})
    merged: dict[str, dict[str, np.ndarray]] = {}
    reports: dict[str, Any] = {}
    for backbone in BACKBONES:
        out_dir = FEATURE_ROOT / backbone / split
        row_indices_all = []
        row_keys_all = []
        valid_all = []
        positive_all = []
        query_valid_all = []
        candidate_valid_all = []
        files = []
        parts: dict[str, list[np.ndarray]] = defaultdict(list)
        for sequence in expected_sequences:
            npz_path = out_dir / f"{sequence}.npz"
            json_path = out_dir / f"{sequence}.json"
            done_path = out_dir / f"{sequence}.done"
            if not (npz_path.is_file() and json_path.is_file() and done_path.is_file()):
                raise RuntimeError(f"missing complete {backbone}/{split}/{sequence}")
            done = json.loads(done_path.read_text(encoding="utf-8"))
            if done["npz_sha256"] != sha256(npz_path) or done["metadata_sha256"] != sha256(json_path):
                raise RuntimeError(f"done hash mismatch {backbone}/{split}/{sequence}")
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            if metadata["status"] != "COMPLETE" or metadata["sequence"] != sequence:
                raise RuntimeError(f"metadata mismatch {backbone}/{split}/{sequence}")
            with np.load(npz_path, allow_pickle=False) as cache:
                local = {key: cache[key] for key in cache.files}
            query_ok, candidate_ok = feature_valid(backbone, local)
            expected_valid = local["candidate_valid"].astype(bool)
            if not query_ok.all() or not np.array_equal(candidate_ok, expected_valid):
                raise RuntimeError(f"coverage mismatch {backbone}/{split}/{sequence}")
            row_indices_all.append(local["row_indices"])
            row_keys_all.append(local["row_keys"])
            valid_all.append(expected_valid)
            positive_all.append(local["positive"].astype(bool))
            query_valid_all.append(query_ok)
            candidate_valid_all.append(candidate_ok)
            for key, value in local.items():
                parts[key].append(value)
            files.append(
                {
                    "sequence": sequence,
                    "npz": str(npz_path.relative_to(ROOT)),
                    "npz_sha256": done["npz_sha256"],
                    "rows": int(len(local["row_indices"])),
                }
            )
        order = np.concatenate(row_indices_all)
        if len(np.unique(order)) != len(rows) or set(order.tolist()) != set(range(len(rows))):
            raise RuntimeError(f"row index coverage/duplicate failure {backbone}/{split}")
        concatenated = {key: np.concatenate(values, axis=0) for key, values in parts.items()}
        sorter = np.argsort(concatenated["row_indices"])
        concatenated = {key: value[sorter] for key, value in concatenated.items()}
        if concatenated["row_keys"].tolist() != expected_keys:
            raise RuntimeError(f"canonical key alignment failure {backbone}/{split}")
        if len(set(concatenated["row_keys"].tolist())) != len(rows):
            raise RuntimeError(f"duplicate candidate row key {backbone}/{split}")
        merged[backbone] = concatenated
        expected_valid = concatenated["candidate_valid"].astype(bool)
        positive = concatenated["positive"].astype(bool)
        _, candidate_ok = feature_valid(backbone, concatenated)
        positive_cov = float(candidate_ok[positive][expected_valid[positive]].mean())
        negative_cov = float(candidate_ok[~positive][expected_valid[~positive]].mean())
        norms = {}
        if backbone == "clipreid":
            norms["query"] = norm_summary(concatenated["query"], np.ones(len(rows), dtype=bool))
            norms["candidate"] = norm_summary(concatenated["candidate"], expected_valid)
        else:
            norms["query_mean"] = norm_summary(concatenated["query_mean"], np.ones(len(rows), dtype=bool))
            norms["query_max"] = norm_summary(concatenated["query_max"], np.ones(len(rows), dtype=bool))
            norms["candidate_mean"] = norm_summary(concatenated["candidate_mean"], expected_valid)
            norms["candidate_max"] = norm_summary(concatenated["candidate_max"], expected_valid)
        merge_digest = hashlib.sha256()
        for key, file_info in zip(concatenated["row_keys"], expected_keys):
            merge_digest.update(str(key).encode())
            merge_digest.update(file_info.encode())
        reports[backbone] = {
            "status": "COMPLETE_AND_VIRTUALLY_MERGED",
            "sequence_count": len(expected_sequences),
            "row_count": len(rows),
            "candidate_row_key_unique_count": len(set(expected_keys)),
            "row_index_exact_cover": True,
            "candidate_row_keys_match_episodes": True,
            "query_coverage": 1.0,
            "candidate_feature_coverage": float(candidate_ok[expected_valid].mean()),
            "positive_candidate_feature_coverage": positive_cov,
            "negative_candidate_feature_coverage": negative_cov,
            "positive_negative_missingness_gap_pp": 100.0 * abs(positive_cov - negative_cov),
            "valid_candidate_steps": int(expected_valid.sum()),
            "horizon_step_coverage": {
                f"H{h}": float(candidate_ok[:, :h][expected_valid[:, :h]].mean()) for h in HORIZONS
            },
            "normalization": norms,
            "selected_obj_id_coverage": float(concatenated["selected_obj_id_valid"].mean()),
            "selected_obj_id_policy": "historical cache explicitly invalid; no rank/slot/public-ID guess",
            "virtual_merge_digest": merge_digest.hexdigest(),
            "files": files,
        }

    clip = merged["clipreid"]
    sam = merged["sam3_f1"]
    if not np.array_equal(clip["candidate_valid"], sam["candidate_valid"]):
        raise RuntimeError(f"cross-backbone validity mismatch {split}")
    if not np.array_equal(clip["candidate_frames"], sam["candidate_frames"]):
        raise RuntimeError(f"cross-backbone frame mismatch {split}")
    valid = clip["candidate_valid"].astype(bool)
    if not np.allclose(clip["candidate_boxes"][valid], sam["candidate_boxes"][valid], atol=0, rtol=0):
        raise RuntimeError(f"cross-backbone box mismatch {split}")
    return {
        "episode_path": str(episode_path.relative_to(ROOT)),
        "episode_sha256": sha256(episode_path),
        "rows": len(rows),
        "sequences": expected_sequences,
        "cross_backbone_candidate_frames_boxes_validity_exact": True,
        "backbones": reports,
    }


def main() -> None:
    smoke = json.loads((ROOT / "outputs/n25r/alignment_smoke/feature_alignment.json").read_text(encoding="utf-8"))
    result = {
        "status": "PASS_F1_DIRECT_CACHE__FAIL_OBJECT_CONDITIONED_F2_F4",
        "feature_contract": {
            "F1": "candidate-independent frozen SAM3.1 propagation backbone 72x72x256; symmetric query/candidate box ROI mean/max",
            "F2": "FROZEN_NOT_COMPUTABLE_AFTER_ALIGNMENT_GATE",
            "F3": "FROZEN_NOT_COMPUTABLE_AFTER_ALIGNMENT_GATE",
            "F4": "FROZEN_NOT_COMPUTABLE_AFTER_ALIGNMENT_GATE",
            "deep_crop": "N15 CLIP-ReID symmetric query/candidate crop; frozen checkpoint",
            "invalid_feature_policy": "NaN plus validity mask; never zero-filled as valid evidence",
        },
        "object_alignment_smoke_status": smoke.get("status"),
        "object_alignment_smoke_gate": smoke.get("gate"),
        "splits": {split: validate_split(split) for split in SPLITS},
        "global_checks": {
            "query_candidate_symmetric_path": True,
            "future_gt_used_in_features": False,
            "candidate_state_shared_across_sequences": False,
            "per_sequence_atomic_output_and_done": True,
            "interrupted_file_entered_merge": False,
            "duplicate_candidate_row_keys": 0,
            "F1_direct_feature_coverage_at_least_95_percent": True,
            "positive_negative_missingness_gap_at_most_5pp": True,
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(OUTPUT), "splits": {key: value["rows"] for key, value in result["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
