#!/usr/bin/env python3
"""Freeze a metadata-only, untouched-sequence reservation for N72R9.

This module deliberately does not read annotations, image pixels beyond the
first-frame dimensions, replay scores, or post-treatment outcomes.  Historical
text is scanned only to determine whether a sequence name was referenced by a
previous experiment and to classify the reference conservatively.  The
reservation is sealed before any N72R9 training or replay is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT_DEFAULT = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "N72R9" / "confirmation_reservation"
SEQUENCE_RE = re.compile(r"\bdancetrack\d{4}\b", re.IGNORECASE)
FRAME_RE = re.compile(r"^(\d+)")
TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
    ".csv",
    ".toml",
}
SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "checkpoints",
    "checkpoint",
    "cache",
    "caches",
    "tmp",
    "temp",
    "envs",
    "venv",
    ".venv",
    ".tox",
    "site-packages",
    "third_party",
    "build",
    "dist",
    "mot_results",
    "rendered",
    "frames",
    "images",
    "videos",
    "video",
    "wandb",
    "tensorboard",
    "runs",
    "runtime",
    "replay",
    "event_prestate",
    "candidate_stream",
    "candidate_streams",
    "per_event",
    "event_artifacts",
    "posthoc_artifacts",
    "worker_artifacts",
    "failure_logs",
}
HISTORY_ROOTS = (
    PROJECT_ROOT,
    Path("/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT"),
    Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R2/worktree"),
    Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R3/worktree"),
    Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R3R1/worktree"),
    Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R5/worktree"),
    Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R5R1/worktree"),
)
MAX_SCANNED_BYTES = 4 * 1024 * 1024


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = _canonical_bytes(payload) + b"\n"
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _reference_categories(path: Path) -> list[str]:
    """Conservative path-only classification; never interprets result values."""

    text = path.as_posix().lower()
    categories: set[str] = set()
    if any(token in text for token in ("train", "training", "corpus", "calibration", "scorer", "decoder")):
        categories.add("training")
    if any(token in text for token in ("val", "validation", "holdout", "screening")):
        categories.add("validation")
    if any(token in text for token in ("confirmation", "fresh_confirmation", "independent_confirmation")):
        categories.add("confirmation")
    if any(
        token in text
        for token in (
            "replay",
            "effect",
            "mechanism",
            "candidate",
            "event",
            "posthoc",
            "root_cause",
            "oracle",
            "stage_",
            "n70",
            "n71",
            "n72",
        )
    ):
        categories.add("development")
    if any(token in text for token in ("posthoc", "score", "metric", "effect", "oracle")):
        categories.add("posthoc")
    return sorted(categories)


def _iter_text_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    # os.walk permits pruning before descending.  Path.rglob enumerates the
    # full directory tree first and made historical result/symlink trees
    # unnecessarily expensive during the first two audit attempts.
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        try:
            relative_current = current_path.relative_to(root).parts
        except ValueError:
            relative_current = current_path.parts
        if not relative_current:
            # The repository contains datasets, environments and historical
            # scratch trees which are not part of the N72 history audit.
            directory_names[:] = [
                name for name in directory_names if name in {"docs", "scripts", "outputs"}
            ]
        elif relative_current == ("outputs",):
            # N8 is represented by the conservative legacy-val marker below;
            # no need to enumerate its large rendered MOT result tree.
            directory_names[:] = [
                name for name in directory_names
                if name.lower().startswith(("n70", "n71", "n72"))
                and name.lower() != "n72r9"
            ]
        directory_names[:] = sorted(
            directory_name
            for directory_name in directory_names
            if directory_name not in SKIP_DIR_NAMES
            and not (Path(current) / directory_name).is_symlink()
        )
        for file_name in sorted(file_names):
            path = current_path / file_name
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                relative_parts = path.relative_to(root).parts
            except ValueError:
                relative_parts = path.parts
            path_text = path.as_posix().lower()
            file_text = path.name.lower()
            # Protocol/status/report files carry the authoritative split and
            # usage declarations.  Runtime rows are included only when their
            # own path names a sequence.  This avoids re-reading millions of
            # per-frame result rows while retaining conservative touch
            # evidence from event-directory names.
            root_level = len(relative_parts) == 1
            source_tree = bool(relative_parts and relative_parts[0] in {"docs", "scripts"})
            output_audit_file = bool(
                relative_parts
                and relative_parts[0] == "outputs"
                and any(token in file_text for token in ("protocol", "manifest", "status", "gate", "report", "registry", "selected"))
            )
            sequence_named = bool(SEQUENCE_RE.search(path_text))
            if not (root_level or source_tree or output_audit_file or sequence_named):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            # Large files are still yielded when their own path contains a
            # sequence name, so a result filename remains evidence without
            # forcing a multi-megabyte content read.
            if size <= MAX_SCANNED_BYTES or SEQUENCE_RE.search(str(path)):
                yield path


def _scan_historical_references(roots: Iterable[Path]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    references: dict[str, list[dict[str, Any]]] = {}
    scanned: list[str] = []
    skipped_roots: list[str] = []
    seen_realpaths: set[str] = set()
    for root in roots:
        root = root.resolve()
        if not root.exists():
            skipped_roots.append(str(root))
            continue
        for path in _iter_text_files(root):
            try:
                path_names = sorted({name.lower() for name in SEQUENCE_RE.findall(str(path))})
                realpath = str(path.resolve())
                if realpath in seen_realpaths:
                    continue
                seen_realpaths.add(realpath)
                # A result symlink's basename is itself historical evidence;
                # do not follow it into an unrelated tree.
                content_read = bool(not path.is_symlink() and path.stat().st_size <= MAX_SCANNED_BYTES)
                raw = path.read_bytes() if content_read else b""
            except (OSError, UnicodeError):
                continue
            names = sorted(
                set(path_names)
                | {name.lower() for name in SEQUENCE_RE.findall(raw.decode("utf-8", errors="ignore"))}
            )
            if not names:
                continue
            categories = _reference_categories(path)
            item = {
                "path": str(path),
                "sha256": _sha256_bytes(raw),
                "file_size_bytes": int(path.stat().st_size),
                "content_read": content_read,
                "categories": categories,
                "sequence_detection": "path_or_content_name_only_case_insensitive_regex",
            }
            scanned.append(str(path))
            for name in names:
                references.setdefault(name, []).append(item)
    for items in references.values():
        items.sort(key=lambda item: item["path"])
    legacy_full_val_eval_detected = any(
        (root.resolve() / "outputs" / "n8" / "eval_full25").exists()
        for root in roots
    )
    return references, {
        "roots": [str(Path(root).resolve()) for root in roots],
        "missing_roots": sorted(set(skipped_roots)),
        "files_with_sequence_references": len(scanned),
        "unique_scanned_files": len(seen_realpaths),
        "content_policy": "sequence names only; no JSON values, labels, scores, metrics, or outcomes parsed",
        "scan_scope": "root-level files, docs, scripts, named audit files, and sequence-named paths; result/render trees pruned",
        "additional_pruned_trees": sorted(
            SKIP_DIR_NAMES.intersection(
                {"runtime", "replay", "event_prestate", "candidate_stream", "candidate_streams", "per_event", "event_artifacts", "posthoc_artifacts", "worker_artifacts", "failure_logs"}
            )
        ),
        "posthoc_outcomes_used_for_selection": False,
        "legacy_full_val_eval_detected": legacy_full_val_eval_detected,
        "legacy_full_val_eval_policy": "if present, every val sequence is conservatively marked validation-used",
    }


def _first_frame_metadata(sequence_dir: Path) -> dict[str, Any]:
    image_dir = sequence_dir / "img1"
    frame_paths = []
    if image_dir.exists():
        for path in image_dir.iterdir():
            if not path.is_file():
                continue
            match = FRAME_RE.match(path.stem)
            if match and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                frame_paths.append((int(match.group(1)), path))
    frame_paths.sort(key=lambda item: (item[0], item[1].name))
    frame_ids = sorted({frame_id for frame_id, _ in frame_paths})
    dimensions = None
    if frame_paths:
        first_path = frame_paths[0][1]
        try:
            from PIL import Image  # type: ignore

            with Image.open(first_path) as image:
                dimensions = [int(image.width), int(image.height)]
        except Exception:
            dimensions = None
    return {
        "available": bool(frame_paths),
        "frame_count": len(frame_ids),
        "first_frame_id": frame_ids[0] if frame_ids else None,
        "last_frame_id": frame_ids[-1] if frame_ids else None,
        "frame_ids_contiguous": bool(frame_ids) and frame_ids == list(range(frame_ids[0], frame_ids[-1] + 1)),
        "dimensions_width_height": dimensions,
        "metadata_source": "img1 filenames and first image header only; no annotation read",
    }


def _build_sequence_records(
    data_root: Path,
    references: dict[str, list[dict[str, Any]]],
    history_scan: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in ("train", "val"):
        split_root = data_root / split
        if not split_root.exists():
            continue
        for sequence_dir in sorted(split_root.glob("dancetrack[0-9][0-9][0-9][0-9]")):
            sequence = sequence_dir.name.lower()
            metadata = _first_frame_metadata(sequence_dir)
            refs = references.get(sequence, [])
            categories = sorted({category for item in refs for category in item["categories"]})
            if split == "val" and history_scan.get("legacy_full_val_eval_detected"):
                categories = sorted(set(categories) | {"validation"})
            record = {
                "sequence": sequence,
                "split": split,
                "path": str(sequence_dir.resolve()),
                **metadata,
                "historical_reference_count": len(refs),
                "historical_reference_categories": categories,
                "ever_used_for_training": "training" in categories,
                "ever_used_for_validation": "validation" in categories,
                "ever_used_for_development": "development" in categories,
                "ever_used_for_posthoc": "posthoc" in categories,
                "ever_used_for_confirmation": "confirmation" in categories,
                "historical_references": refs,
            }
            record["eligible_for_fresh_confirmation"] = bool(
                metadata["available"]
                and not record["ever_used_for_training"]
                and not record["ever_used_for_validation"]
                and not record["ever_used_for_development"]
                and not record["ever_used_for_posthoc"]
                and not record["ever_used_for_confirmation"]
            )
            records.append(record)
    return records


def _reservation_payload(
    records: list[dict[str, Any]],
    history_scan: dict[str, Any],
    target_count: int,
    data_root: Path,
) -> dict[str, Any]:
    eligible = [record for record in records if record["eligible_for_fresh_confirmation"]]
    # Prefer never-referenced validation sequences; then never-referenced train
    # sequences.  This rule is fixed before any event frame or replay outcome.
    selected: list[dict[str, Any]] = []
    for split in ("val", "train"):
        selected.extend(record for record in eligible if record["split"] == split)
    selected = sorted(selected, key=lambda record: (0 if record["split"] == "val" else 1, record["sequence"]))
    selected = selected[: min(target_count, len(selected))]
    selected_names = [record["sequence"] for record in selected]
    if not selected:
        reservation_status = "BLOCKED_NO_FRESH_UNTOUCHED_CONFIRMATION_SEQUENCES"
        limitation = "All available train/val sequences have conservative historical-touch evidence; no fresh confirmation sequence can be reserved."
    elif len(selected) < target_count:
        reservation_status = "PASS_LIMITED_FRESH_RESERVATION_SEALED"
        limitation = f"Only {len(selected)} eligible sequences were available; protocol target was {target_count}."
    else:
        reservation_status = "PASS_FRESH_RESERVATION_SEALED"
        limitation = None
    registry_core = {
        "schema_version": "N72R9_TOUCHED_SEQUENCE_REGISTRY_V1",
        "selection_basis": "metadata_only_before_training_or_replay",
        "data_root": str(data_root.resolve()),
        "history_scan": history_scan,
        "sequences": records,
    }
    registry_hash = _sha256_bytes(_canonical_bytes(registry_core))
    protocol_core = {
        "schema_version": "N72R9_FRESH_CONFIRMATION_PROTOCOL_STUB_V1",
        "selection_rule": {
            "target_untouched_sequence_count": target_count,
            "priority": ["available untouched val", "available untouched train"],
            "order": "split priority then lexicographic sequence",
            "no_event_or_outcome_fields_used": True,
        },
        "registry_sha256": registry_hash,
        "reserved_sequences": selected_names,
        "future_outcome_fields_forbidden_until_runtime_sealed": [
            "future_identity_error",
            "H20",
            "H50",
            "H100",
            "IDSW",
            "IoU",
            "re_correction",
            "assignment_change",
            "replay_score",
        ],
        "runtime_contract": {
            "interaction_source": "simulated_from_gt_only_if_offline_event_materialization_is_later_authorized",
            "runtime_future_gt_used": False,
            "public_id_source": "direct_event_field_only",
            "event_frame_memory_read": False,
            "first_memory_visible_frame": "event_frame+1",
        },
    }
    protocol_hash = _sha256_bytes(_canonical_bytes(protocol_core))
    return {
        "registry_core": registry_core,
        "registry_sha256": registry_hash,
        "protocol_core": protocol_core,
        "protocol_sha256": protocol_hash,
        "selected": selected,
        "reservation_status": reservation_status,
        "limitation": limitation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--target-count", type=int, default=8)
    args = parser.parse_args()
    if args.target_count <= 0:
        raise SystemExit("--target-count must be positive")
    data_root = args.data_root.resolve()
    roots = list(dict.fromkeys(root.resolve() for root in HISTORY_ROOTS))
    references, history_scan = _scan_historical_references(roots)
    records = _build_sequence_records(data_root, references, history_scan)
    if not records:
        raise SystemExit(f"no train/val sequence metadata found under {data_root}")
    reservation = _reservation_payload(records, history_scan, args.target_count, data_root)

    registry = dict(reservation["registry_core"])
    registry["data_root"] = str(data_root)
    registry["registry_sha256"] = reservation["registry_sha256"]
    selected = [
        {
            "sequence": item["sequence"],
            "split": item["split"],
            "path": item["path"],
            "available": item["available"],
            "frame_count": item["frame_count"],
            "first_frame_id": item["first_frame_id"],
            "last_frame_id": item["last_frame_id"],
            "dimensions_width_height": item["dimensions_width_height"],
            "historical_reference_count": item["historical_reference_count"],
            "metadata_only": True,
        }
        for item in reservation["selected"]
    ]
    selected_payload = {
        "schema_version": "N72R9_FRESH_CONFIRMATION_SEQUENCE_LIST_V1",
        "status": reservation["reservation_status"],
        "registry_sha256": reservation["registry_sha256"],
        "protocol_sha256": reservation["protocol_sha256"],
        "selection_rule": reservation["protocol_core"]["selection_rule"],
        "sequences": selected,
        "no_event_frames_selected": True,
        "no_future_outcomes_read": True,
    }
    protocol = dict(reservation["protocol_core"])
    protocol["status"] = reservation["reservation_status"]
    protocol["limitation"] = reservation["limitation"]
    protocol["sealed_at_utc"] = datetime.now(timezone.utc).isoformat()
    status = {
        "schema_version": "N72R9_STAGE_01_RESERVATION_STATUS_V1",
        "stage": "N72R9_B1_B3_UNTOUCHED_CONFIRMATION_RESERVATION",
        "status": reservation["reservation_status"],
        "research_gate": "RESERVATION_ONLY_NO_TRAINING_OR_REPLAY",
        "data_root": str(data_root),
        "sequence_count": len(records),
        "eligible_sequence_count": sum(1 for record in records if record["eligible_for_fresh_confirmation"]),
        "reserved_sequence_count": len(selected),
        "reserved_sequences": [item["sequence"] for item in selected],
        "registry_sha256": reservation["registry_sha256"],
        "protocol_sha256": reservation["protocol_sha256"],
        "metadata_fields_used": ["sequence", "split", "frame_count", "availability", "dimensions"],
        "forbidden_before_reservation_seal": [
            "training",
            "validation",
            "root_cause_diagnosis",
            "posthoc_dev_analysis",
            "replay_outcome_selection",
        ],
        "runtime_future_gt_used": False,
        "next_step": "Only after this hash is sealed: source-aware corpus/training on non-reserved development sequences.",
        "limitation": reservation["limitation"],
    }
    protocol_correction_status = {
        "schema_version": "N72R9_STAGE_00_PROTOCOL_CORRECTION_STATUS_V1",
        "stage": "N72R8R1_PROTOCOL_CORRECTION_CARRIED_FORWARD",
        "status": "PASS_PROTOCOL_CORRECTION_CARRIED_FORWARD",
        "source": "outputs/N72R8R1/corrected_gate.json",
        "historical_gate_unchanged": True,
        "old_confirmation_reclassified_not_fresh": True,
        "new_gpu_or_replay_started": False,
        "next_stage": "N72R9_B1_B3_UNTOUCHED_CONFIRMATION_RESERVATION",
    }
    _atomic_json(OUTPUT_ROOT / "touched_sequence_registry.json", registry)
    _atomic_json(OUTPUT_ROOT / "fresh_confirmation_sequences.json", selected_payload)
    _atomic_json(OUTPUT_ROOT / "fresh_confirmation_protocol_stub.json", protocol)
    _atomic_json(OUTPUT_ROOT / "reservation_manifest.json", {
        "schema_version": "N72R9_RESERVATION_MANIFEST_V1",
        "status": reservation["reservation_status"],
        "registry": "touched_sequence_registry.json",
        "sequence_list": "fresh_confirmation_sequences.json",
        "protocol": "fresh_confirmation_protocol_stub.json",
        "registry_sha256": reservation["registry_sha256"],
        "protocol_sha256": reservation["protocol_sha256"],
        "reserved_sequence_count": len(selected),
        "reserved_sequences": [item["sequence"] for item in selected],
        "future_outcomes_used_for_selection": False,
        "selection_sealed": True,
    })
    _atomic_json(OUTPUT_ROOT.parent / "stage_00_protocol_correction_status.json", protocol_correction_status)
    _atomic_json(OUTPUT_ROOT.parent / "stage_01_confirmation_reservation_status.json", status)
    print(json.dumps({
        "status": reservation["reservation_status"],
        "sequence_count": len(records),
        "eligible_sequence_count": status["eligible_sequence_count"],
        "reserved_sequence_count": len(selected),
        "reserved_sequences": [item["sequence"] for item in selected],
        "registry_sha256": reservation["registry_sha256"],
        "protocol_sha256": reservation["protocol_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
