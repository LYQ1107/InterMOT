#!/usr/bin/env python3
"""Build N31-E's legal expanded train-fold episode manifest.

The event list comes from the already frozen N27 DanceTrack train-fold real
candidate ledger.  This builder does not choose events by N31 future quality;
it only validates the current correction frame and the existence of a legal
20-frame file window.  Every legal, non-duplicate event is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = Path("/path/to/dancetrack/train")
SPLIT_MANIFEST = ROOT / "outputs/n27/dataset_split_manifest.json"
METADATA = ROOT / "outputs/n27/dance_train_real_metadata.jsonl"
OUTPUT = ROOT / "outputs/n31/episode_manifest.json"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _gt(sequence: Path) -> dict[int, dict[int, list[float]]]:
    gt_path = sequence / "gt" / "gt.txt"
    result: dict[int, dict[int, list[float]]] = {}
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split(",")
        if len(fields) < 6:
            continue
        frame = int(fields[0]) - 1
        identity = int(fields[1])
        x, y, w, h = (float(value) for value in fields[2:6])
        result.setdefault(frame, {})[identity] = [x, y, x + w, y + h]
    return result


def _image_count(sequence: Path) -> int:
    image_dir = sequence / "img1"
    return sum(path.suffix.lower() in {".jpg", ".jpeg", ".png"} for path in image_dir.iterdir())


def build(*, split_manifest: Path, metadata: Path, train_root: Path, output: Path) -> dict[str, Any]:
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    sequences = sorted({
        str(entry["video"])
        for entry in split.get("entries", [])
        if entry.get("dataset") == "DanceTrack" and entry.get("role") == "train_fold"
    })
    if len(sequences) != 30:
        raise ValueError(f"N31-E requires the frozen 30 train_fold parent sequences, got {len(sequences)}")
    rng = np.random.default_rng(31031)
    shuffled = list(np.asarray(sequences, dtype=object)[rng.permutation(len(sequences))])
    split_sequences = {
        "train": sorted(shuffled[:18]),
        "selection": sorted(shuffled[18:24]),
        "calibration": sorted(shuffled[24:30]),
    }
    allowed = set(sequences)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    source_rows = 0
    rejected: dict[str, int] = {}
    gt_cache: dict[str, dict[int, dict[int, list[float]]]] = {}
    image_counts: dict[str, int] = {}
    for line in metadata.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        if not source.get("correction_event", False):
            continue
        source_rows += 1
        sequence = str(source.get("sequence", ""))
        if sequence not in allowed:
            rejected["not_in_frozen_train_fold"] = rejected.get("not_in_frozen_train_fold", 0) + 1
            continue
        try:
            identity = int(source["public_identity_id"]) - 1000
            correction = int(source["frame"])
        except (KeyError, TypeError, ValueError):
            rejected["malformed_event"] = rejected.get("malformed_event", 0) + 1
            continue
        key = (sequence, correction, identity)
        if key in seen:
            rejected["duplicate_event_key"] = rejected.get("duplicate_event_key", 0) + 1
            continue
        seen.add(key)
        sequence_path = train_root / sequence
        if sequence not in gt_cache:
            gt_cache[sequence] = _gt(sequence_path)
            image_counts[sequence] = _image_count(sequence_path)
        gt = gt_cache[sequence]
        image_count = image_counts[sequence]
        if correction < 0 or correction + 20 >= image_count:
            rejected["no_h20_file_window"] = rejected.get("no_h20_file_window", 0) + 1
            continue
        if identity not in gt.get(correction, {}):
            rejected["target_not_visible_at_current_frame"] = rejected.get("target_not_visible_at_current_frame", 0) + 1
            continue
        init = next((frame for frame in range(0, correction + 1) if identity in gt.get(frame, {})), None)
        if init is None:
            rejected["no_legal_initialization"] = rejected.get("no_legal_initialization", 0) + 1
            continue
        split_name = next(name for name, values in split_sequences.items() if sequence in values)
        event_key = str(source.get("event_key", f"{sequence}:{correction}:{identity}"))
        rows.append({
            "episode_id": f"n31_expanded_{event_key}",
            "event_key": event_key,
            "sequence": sequence,
            "sequence_path": str(sequence_path),
            "split": "train/train_fold",
            "learning_split": split_name,
            "dataset": "DanceTrack",
            "initialization_frame": int(init),
            "correction_frame": int(correction),
            "query_start": int(correction + 1),
            "query_end": int(correction + 20),
            "query_horizons": {"5": [correction + 1, correction + 5], "10": [correction + 1, correction + 10], "20": [correction + 1, correction + 20]},
            "dataset_identity": int(identity),
            "public_id": int(110000 + len(rows)),
            "sam_object_id": int(110000 + len(rows)),
            "correction_box": list(gt[correction][identity]),
            "source_parent_index": source.get("parent_index"),
            "source_policy_version": source.get("policy_version"),
            "source_metadata_sha256": _sha256(metadata),
            "selection_frozen_before_n31": True,
            "future_gt_used_for_selection": False,
            "gt_role": "current-frame legality validation and post-hoc future supervision only",
        })
    rows.sort(key=lambda row: (row["sequence"], int(row["correction_frame"]), int(row["dataset_identity"]), row["event_key"]))
    # Public IDs are assigned after sorting so the manifest is byte-stable.
    for index, row in enumerate(rows):
        row["public_id"] = 110000 + index
        row["sam_object_id"] = 110000 + index
    per_split = {name: sum(row["learning_split"] == name for row in rows) for name in split_sequences}
    payload = {
        "protocol": "N31-E-EXPANDED-TRAIN-FOLD-EPISODES",
        "status": "PASS" if len(rows) >= 500 else "INSUFFICIENT_LEGAL_EPISODES",
        "requested_target_count": 500,
        "available_legal_episode_count": len(rows),
        "source_metadata": str(metadata),
        "source_metadata_sha256": _sha256(metadata),
        "source_correction_event_count": source_rows,
        "rejected_counts": rejected,
        "train_root": str(train_root),
        "parent_sequence_count": len(sequences),
        "parent_sequences": sequences,
        "fixed_sequence_split_seed": 31031,
        "split_sequences": split_sequences,
        "episode_counts_by_learning_split": per_split,
        "no_duplicate_event_keys": len({row["event_key"] for row in rows}) == len(rows),
        "all_legal_available_episodes_retained": True,
        "selection_frozen_before_n31": True,
        "future_gt_used_for_selection": False,
        "val25_read": False,
        "test_labels_used": False,
        "episodes": rows,
    }
    _write(output, payload)
    payload["manifest_sha256"] = _sha256(output)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--train-root", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = build(split_manifest=args.split_manifest, metadata=args.metadata, train_root=args.train_root, output=args.output)
    print(json.dumps({key: result.get(key) for key in ("protocol", "status", "available_legal_episode_count", "episode_counts_by_learning_split")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
