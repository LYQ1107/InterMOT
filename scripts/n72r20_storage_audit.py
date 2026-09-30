#!/usr/bin/env python3
"""Create the read-only N72R20 storage audit required before any generation."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path("/data3/liuyeqiang/InterMOT")
OUT = REPO / "outputs" / "N72R20"
DATASET_CANDIDATE = Path("/data3/liuyeqiang/InterMOT_N72R16_assets/dataset")


def command_output(argv: list[str]) -> str:
    result = subprocess.run(argv, text=True, capture_output=True, check=False)
    return result.stdout.strip()


def size_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return None
    result = subprocess.run(
        ["du", "-sb", "--", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return int(result.stdout.split()[0])
    except (ValueError, IndexError):
        return None


def record_size(label: str, path: str | Path) -> dict[str, Any]:
    target = Path(path)
    size = size_bytes(target)
    return {
        "label": label,
        "path": str(target),
        "exists": target.exists(),
        "bytes": size,
        "gib": round(size / (1024**3), 4) if size is not None else None,
    }


def find_named_checkpoints() -> list[dict[str, Any]]:
    command = [
        "find",
        "/data3",
        "/data2",
        "/data1",
        "/home",
        "-maxdepth",
        "8",
        "-type",
        "f",
        "-name",
        "sam3.1_multiplex.pt",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    records = []
    for line in result.stdout.splitlines():
        path = Path(line.strip())
        if path.is_file():
            records.append(record_size("SAM3 checkpoint candidate", path))
    return records


def find_partial_files() -> list[dict[str, Any]]:
    roots = [REPO, REPO.parent / "InterMOT_N72R16_assets", REPO.parent / "InterMOT_N72R17_assets"]
    records: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        command = [
            "find",
            str(root),
            "-xdev",
            "-maxdepth",
            "8",
            "-type",
            "f",
            "(",
            "-name",
            "*.partial",
            "-o",
            "-name",
            "*.part",
            "-o",
            "-name",
            "*.tmp",
            "-o",
            "-name",
            "*.crdownload",
            ")",
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        for line in result.stdout.splitlines():
            path = Path(line.strip())
            if path.is_file():
                records.append(record_size("partial or temporary file", path))
    return records


def output_large_files() -> list[dict[str, Any]]:
    roots = [
        REPO / "outputs",
        REPO.parent / "InterMOT_N72R16_assets" / "outputs",
        REPO.parent / "InterMOT_N72R17_assets" / "outputs",
    ]
    records: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        result = subprocess.run(
            ["find", str(root), "-xdev", "-type", "f", "-size", "+1G", "-print"],
            text=True,
            capture_output=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            path = Path(line.strip())
            if path.is_file():
                records.append(record_size("output file larger than 1 GiB", path))
    return records


def cache_records() -> list[dict[str, Any]]:
    explicit = [
        ("HuggingFace cache", REPO.parent / ".cache" / "huggingface"),
        ("Torch cache", REPO.parent / ".cache" / "torch"),
        ("pip cache", REPO.parent / ".cache" / "pip"),
        ("Codex temporary directory", REPO.parent / ".codex" / ".tmp"),
    ]
    records = [record_size(label, path) for label, path in explicit]
    pip_cache = command_output([sys.executable, "-m", "pip", "cache", "dir"])
    if pip_cache:
        pip_path = Path(pip_cache.splitlines()[-1].strip())
        if pip_path != explicit[2][1]:
            records.append(record_size("pip cache reported by pip", pip_path))
    return records


def mount_lines() -> list[str]:
    text = command_output(["df", "-hP"])
    return text.splitlines()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    records = [
        record_size("InterMOT repository", REPO),
        record_size("N72R16 assets", REPO.parent / "InterMOT_N72R16_assets"),
        record_size("N72R17 assets", REPO.parent / "InterMOT_N72R17_assets"),
        record_size("N72R18 outputs", REPO / "outputs" / "N72R18"),
        record_size("N72R19 outputs", REPO / "outputs" / "N72R19"),
        record_size("N72R19R1 outputs", REPO / "outputs" / "N72R19R1"),
        record_size("DanceTrack train", DATASET_CANDIDATE / "train"),
        record_size("DanceTrack val", DATASET_CANDIDATE / "val"),
        record_size("DanceTrack candidate root", DATASET_CANDIDATE),
        record_size("research_references", REPO.parent / "research_references"),
    ]
    records.extend(cache_records())
    checkpoints = find_named_checkpoints()
    partials = find_partial_files()
    large_outputs = output_large_files()
    records.extend(checkpoints)
    records.extend(partials)
    records.extend(large_outputs)
    records.sort(key=lambda item: item.get("bytes") or -1, reverse=True)

    df_h = mount_lines()
    audit = {
        "stage": "N72R20",
        "audit": "storage_audit_before",
        "timestamp_utc": now,
        "policy": {
            "normal_free_space_gib_minimum": 100,
            "hard_stop_free_space_gib": 80,
            "new_data_budget_gib_maximum": 60,
        },
        "df_h": df_h,
        "df_bytes": command_output(["df", "-P", "-B1"]).splitlines(),
        "sizing_records_descending": records,
        "checkpoint_candidates": checkpoints,
        "partial_or_temporary_files": partials,
        "output_files_over_1gib": large_outputs,
        "download_performed": False,
        "cleanup_performed": False,
        "notes": [
            "This is a read-only audit; no dataset, checkpoint, cache, or historical asset was deleted.",
            "DanceTrack candidate root is recorded for audit only and is not accepted until dataset discovery validates it.",
        ],
    }
    json_path = OUT / "storage_audit_before.json"
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")

    free_line = next((line for line in df_h if "/data3/liuyeqiang" in line), "not found")
    md_lines = [
        "# N72R20 Storage Audit Before",
        "",
        f"- Timestamp (UTC): `{now}`",
        f"- `/data3/liuyeqiang` filesystem: `{free_line}`",
        "- Download performed: `false`",
        "- Cleanup performed: `false`",
        "",
        "## Mounted filesystems",
        "",
        "```text",
        *df_h,
        "```",
        "",
        "## Sized paths and files (descending)",
        "",
        "| GiB | Exists | Label | Path |",
        "|---:|:---:|---|---|",
    ]
    for item in records:
        gib = "n/a" if item["gib"] is None else f"{item['gib']:.4f}"
        md_lines.append(
            f"| {gib} | {'yes' if item['exists'] else 'no'} | {item['label']} | `{item['path']}` |"
        )
    md_lines.extend(
        [
            "",
            "No cleanup is authorized by this audit. Any later cleanup must first appear in `cleanup_candidates.json` and `storage_cleanup_log.jsonl`.",
            "",
        ]
    )
    (OUT / "storage_audit_before.md").write_text("\n".join(md_lines))
    print(json_path)
    print(OUT / "storage_audit_before.md")


if __name__ == "__main__":
    main()
