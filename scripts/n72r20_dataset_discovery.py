#!/usr/bin/env python3
"""Audit existing DanceTrack roots without downloading or copying data."""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path("/data3/liuyeqiang/InterMOT")
OUT = REPO / "outputs" / "N72R20" / "dataset_discovery.json"
SEARCH_ROOTS = [Path("/data3"), Path("/data2"), Path("/data1"), Path("/home")]
SEARCH_COMMAND = "find /data3 /data2 /data1 /home -maxdepth 6 -type d \\( -iname '*dancetrack*' -o -iname 'dancetrack' \\) 2>/dev/null"
EXPECTED_COUNTS = {"train": 40, "val": 25}
SEQ_RE = re.compile(r"dancetrack\d+", re.IGNORECASE)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def run_find() -> list[str]:
    command = [
        "find",
        "/data3",
        "/data2",
        "/data1",
        "/home",
        "-maxdepth",
        "6",
        "-type",
        "d",
        "(",
        "-iname",
        "*dancetrack*",
        "-o",
        "-iname",
        "dancetrack",
        ")",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})


def ancestor_candidates(matches: list[str]) -> list[Path]:
    roots: set[Path] = set()
    for match in matches:
        path = Path(match)
        for level in range(0, 5):
            candidate = path
            for _ in range(level):
                candidate = candidate.parent
            roots.add(candidate)
    return sorted(roots)


def parse_seqinfo(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    try:
        for raw in path.read_text(errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    except OSError:
        return values
    return values


def sequence_audit(seq_dir: Path) -> dict[str, Any]:
    img_dir = seq_dir / "img1"
    gt_path = seq_dir / "gt" / "gt.txt"
    seqinfo_path = seq_dir / "seqinfo.ini"
    image_count = 0
    image_bytes = 0
    frame_numbers: list[int] = []
    all_bytes = 0
    regular_files = 0
    try:
        for root, _dirs, files in os.walk(seq_dir):
            for filename in files:
                path = Path(root) / filename
                try:
                    stat = path.stat()
                except OSError:
                    continue
                if not path.is_file():
                    continue
                regular_files += 1
                all_bytes += stat.st_size
                if path.parent == img_dir and path.suffix.lower() in IMAGE_SUFFIXES:
                    image_count += 1
                    image_bytes += stat.st_size
                    try:
                        frame_numbers.append(int(path.stem))
                    except ValueError:
                        pass
    except OSError:
        pass

    gt_lines = 0
    gt_first_frame: int | None = None
    gt_last_frame: int | None = None
    gt_readable = False
    if gt_path.is_file():
        try:
            with gt_path.open("r", errors="replace") as handle:
                for raw in handle:
                    if not raw.strip():
                        continue
                    gt_lines += 1
                    try:
                        frame = int(raw.split(",", 1)[0])
                    except (ValueError, IndexError):
                        continue
                    gt_first_frame = frame if gt_first_frame is None else min(gt_first_frame, frame)
                    gt_last_frame = frame if gt_last_frame is None else max(gt_last_frame, frame)
            gt_readable = True
        except OSError:
            gt_readable = False

    seqinfo = parse_seqinfo(seqinfo_path)
    expected_frames = None
    if seqinfo.get("seqLength", "").isdigit():
        expected_frames = int(seqinfo["seqLength"])
    frame_numbers.sort()
    return {
        "sequence": seq_dir.name,
        "path": str(seq_dir),
        "exists": seq_dir.is_dir(),
        "seqinfo_exists": seqinfo_path.is_file(),
        "seqinfo": seqinfo,
        "expected_frame_count": expected_frames,
        "image_count": image_count,
        "image_bytes": image_bytes,
        "image_min_frame": frame_numbers[0] if frame_numbers else None,
        "image_max_frame": frame_numbers[-1] if frame_numbers else None,
        "gt_path": str(gt_path),
        "gt_exists": gt_path.is_file(),
        "gt_readable": gt_readable,
        "gt_bytes": gt_path.stat().st_size if gt_path.is_file() else None,
        "gt_line_count": gt_lines,
        "gt_min_frame": gt_first_frame,
        "gt_max_frame": gt_last_frame,
        "regular_file_count": regular_files,
        "sequence_bytes": all_bytes,
        "complete": bool(
            seq_dir.is_dir()
            and img_dir.is_dir()
            and image_count > 0
            and seqinfo_path.is_file()
            and gt_path.is_file()
            and gt_readable
        ),
    }


def protocol_sequence_set(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return set()
    return {match.lower() for match in SEQ_RE.findall(text)}


def protocol_sources() -> dict[str, list[str]]:
    candidates = {
        "train": [
            "/data3/liuyeqiang/InterMOT_N72R17_assets/outputs/N72R17/protocol_train.json",
            "/data3/liuyeqiang/InterMOT_N72R16_assets/outputs/N72R16/protocol_train.json",
            "/data3/liuyeqiang/InterMOT/outputs/N72R17/protocol_train.json",
        ],
        "val": [
            "/data3/liuyeqiang/InterMOT_N72R17_assets/outputs/N72R17/protocol_val.json",
            "/data3/liuyeqiang/InterMOT_N72R16_assets/outputs/N72R16/protocol_val.json",
            "/data3/liuyeqiang/InterMOT/outputs/N72R17/protocol_val.json",
        ],
    }
    selected: dict[str, list[str]] = {}
    for split, paths in candidates.items():
        for raw_path in paths:
            path = Path(raw_path)
            if path.is_file():
                selected[split] = sorted(protocol_sequence_set(path))
                selected[f"{split}_source"] = [raw_path]
                break
        else:
            selected[split] = []
            selected[f"{split}_source"] = []
    return selected


def audit_root(root: Path, protocols: dict[str, list[str]]) -> dict[str, Any] | None:
    split_dirs = {split: root / split for split in EXPECTED_COUNTS if (root / split).is_dir()}
    if not split_dirs:
        return None
    split_records: dict[str, Any] = {}
    for split, split_dir in split_dirs.items():
        sequences = []
        try:
            children = sorted(path for path in split_dir.iterdir() if path.is_dir())
        except OSError:
            children = []
        for seq_dir in children:
            if SEQ_RE.fullmatch(seq_dir.name):
                sequences.append(sequence_audit(seq_dir))
        names = sorted(item["sequence"].lower() for item in sequences)
        expected = set(protocols.get(split, []))
        actual = set(names)
        split_records[split] = {
            "path": str(split_dir),
            "sequence_count": len(sequences),
            "expected_sequence_count": EXPECTED_COUNTS[split],
            "sequence_names": names,
            "missing_from_protocol": sorted(expected - actual),
            "extra_vs_protocol": sorted(actual - expected) if expected else [],
            "all_sequences_complete": all(item["complete"] for item in sequences)
            and len(sequences) == EXPECTED_COUNTS[split],
            "sequences": sequences,
        }
    for split in EXPECTED_COUNTS:
        split_records.setdefault(
            split,
            {
                "path": str(root / split),
                "sequence_count": 0,
                "expected_sequence_count": EXPECTED_COUNTS[split],
                "sequence_names": [],
                "missing_from_protocol": list(protocols.get(split, [])),
                "extra_vs_protocol": [],
                "all_sequences_complete": False,
                "sequences": [],
            },
        )
    complete = all(
        split_records[split]["sequence_count"] == EXPECTED_COUNTS[split]
        and split_records[split]["all_sequences_complete"]
        and not split_records[split]["missing_from_protocol"]
        for split in EXPECTED_COUNTS
    )
    lineage_evidence: list[str] = []
    if "InterMOT_N72R16_assets/dataset" in str(root):
        report = REPO / "docs" / "N72R16_FINAL_REPORT.md"
        manifest = REPO.parent / "InterMOT_N72R16_assets" / "outputs" / "N72R16" / "asset_manifest.json"
        if report.is_file() and "N72R16_NEW_ASSET_LINEAGE" in report.read_text(errors="replace"):
            lineage_evidence.append(str(report))
        if manifest.is_file():
            lineage_evidence.append(str(manifest))
    if lineage_evidence and complete:
        lineage_status = "N72R16_NEW_ASSET_LINEAGE"
    elif "InterMOT_N72R16_assets/dataset" in str(root):
        lineage_status = "N72R16_PATH_MATCH_BUT_INCOMPLETE_EVIDENCE"
    else:
        lineage_status = "UNKNOWN"
    ignored_splits = [split for split in ("test", "demo") if (root / split).is_dir()]
    return {
        "root": str(root),
        "train_val_complete_and_reusable": complete,
        "lineage_status": lineage_status,
        "lineage_evidence": lineage_evidence,
        "ignored_out_of_scope_splits_present": ignored_splits,
        "splits": split_records,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    matches = run_find()
    protocols = protocol_sources()
    audits: list[dict[str, Any]] = []
    seen_roots: set[str] = set()
    for root in ancestor_candidates(matches):
        if str(root) in seen_roots:
            continue
        seen_roots.add(str(root))
        audit = audit_root(root, protocols)
        if audit is not None:
            audits.append(audit)
    reusable = [item for item in audits if item["train_val_complete_and_reusable"]]
    preferred = sorted(
        reusable,
        key=lambda item: (
            item["lineage_status"] != "N72R16_NEW_ASSET_LINEAGE",
            item["root"],
        ),
    )
    recommended = preferred[0] if preferred else None
    result = {
        "stage": "N72R20",
        "audit": "dataset_discovery",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "search_command": SEARCH_COMMAND,
        "search_roots": [str(path) for path in SEARCH_ROOTS],
        "download_performed": False,
        "proxy_used": False,
        "allowed_dataset_scope": ["DanceTrack train", "DanceTrack val"],
        "forbidden_dataset_scope": [
            "DanceTrack test",
            "MOT17",
            "MOT20",
            "BDD100K",
            "TAO",
            "SportsMOT",
            "CrowdHuman",
            "other ReID datasets",
        ],
        "protocol_sources": protocols,
        "matched_paths": matches,
        "candidate_roots": audits,
        "recommended_root": recommended["root"] if recommended else None,
        "recommended_lineage_status": recommended["lineage_status"] if recommended else None,
        "DANCETRACK_ROOT": recommended["root"] if recommended else None,
        "reuse_existing_data": bool(recommended),
        "download_required": not bool(recommended),
        "test_downloaded": False,
        "conclusion": (
            "Complete N72R16-lineage DanceTrack train/val found; reuse in place and do not download."
            if recommended
            else "No complete reusable DanceTrack train/val root found; download decision remains pending source audit."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(OUT)
    print(json.dumps({"matched_paths": len(matches), "candidate_roots": len(audits), "recommended_root": result["recommended_root"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
