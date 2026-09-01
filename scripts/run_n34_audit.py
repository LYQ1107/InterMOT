#!/usr/bin/env python3
"""N34-0/N34-1 audit for the mechanism-first CCAM continuation.

The audit is deliberately conservative.  It reads only the authorized
DanceTrack train/train_fold data and source/artifact metadata.  In particular,
it never opens DanceTrack val/test content and it never treats an old
episode-level candidate cache as a complete per-frame SAM3 candidate tape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/path/to/dancetrack")
FORBIDDEN_SPLITS = {"val", "test", "validation", "testing"}
CALLER_TERMS = (
    "rollout_frame",
    "candidate_log",
    "apply_intervention",
    "human_event",
    "gt_box",
    "public_id",
    "ADD_NEW_IDENTITY",
    "AUTHORITATIVE_REASSIGN",
    "ATOMIC_ID_SWAP",
    "RECOVER_IDENTITY",
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def artifact_metadata(relative_paths: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for relative in relative_paths:
        path = ROOT / relative
        result[relative] = {
            "exists": path.exists(),
            "is_file": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256(path),
        }
    return result


def _caller_scan() -> dict[str, list[dict[str, Any]]]:
    matches: dict[str, list[dict[str, Any]]] = defaultdict(list)
    roots = [ROOT / "sam3_intermot", ROOT / "scripts", ROOT / "tests", ROOT / "docs"]
    for directory in roots:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".md", ".txt"}:
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for line_number, line in enumerate(lines, 1):
                for term in CALLER_TERMS:
                    if term in line:
                        matches[term].append(
                            {"file": safe_rel(path), "line": line_number, "text": line.strip()[:300]}
                        )
    return {key: value for key, value in sorted(matches.items())}


def _backend_capability_probe() -> dict[str, Any]:
    backend_path = ROOT / "sam3_intermot" / "backend" / "sam3_backend.py"
    output_path = ROOT / "sam3_intermot" / "backend" / "output_types.py"
    backend_text = backend_path.read_text(encoding="utf-8") if backend_path.is_file() else ""
    output_text = output_path.read_text(encoding="utf-8") if output_path.is_file() else ""
    output_fields = re.findall(r"^\s+(\w+):", output_text, flags=re.MULTILINE)
    return {
        "source_files": [safe_rel(backend_path), safe_rel(output_path)],
        "official_video_cpu_offload_argument_in_adapter": "offload_video_to_cpu=True" in backend_text,
        "async_loading_frames_argument_in_adapter": "async_loading_frames" in backend_text,
        "conditional_output_cpu_offload_flag_in_adapter": "offload_output_to_cpu_for_eval" in backend_text,
        "official_state_offload_to_cpu_supported_by_pinned_multiplex_init": False,
        "state_offload_reason": "adapter source records that pinned Sam3MultiplexTrackingWithInteractivity.init_state does not accept offload_state_to_cpu",
        "prompt_object_observation_fields": output_fields,
        "candidate_embedding_field_exported": any(field in output_fields for field in ("embedding", "feature", "decoder_token")),
        "candidate_score_matrix_exported": False,
        "per_frame_all_candidate_public_id_exported": False,
        "conclusion": "official adapter exposes active-object observations, not the N34 all-candidate public-ID tape contract",
    }


def _parse_gt(gt_path: Path) -> dict[str, Any]:
    per_frame: dict[int, set[int]] = defaultdict(set)
    unique_ids: set[int] = set()
    rows = 0
    malformed = 0
    if not gt_path.is_file():
        return {
            "exists": False,
            "rows": 0,
            "malformed_rows": 0,
            "unique_ids": [],
            "unique_id_count": 0,
            "max_simultaneous_ids": 0,
            "frames_with_2plus_ids": 0,
            "multi_id_frames": [],
        }
    with gt_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if not text:
                continue
            fields = text.split(",")
            if len(fields) < 6:
                malformed += 1
                continue
            try:
                frame = int(float(fields[0]))
                identity = int(float(fields[1]))
                float(fields[2])
                float(fields[3])
                float(fields[4])
                float(fields[5])
            except (TypeError, ValueError):
                malformed += 1
                continue
            if frame < 1 or identity < 0:
                malformed += 1
                continue
            rows += 1
            unique_ids.add(identity)
            per_frame[frame].add(identity)
    multi = sorted(frame for frame, ids in per_frame.items() if len(ids) >= 2)
    return {
        "exists": True,
        "rows": rows,
        "malformed_rows": malformed,
        "unique_ids": sorted(unique_ids),
        "unique_id_count": len(unique_ids),
        "max_simultaneous_ids": max((len(ids) for ids in per_frame.values()), default=0),
        "frames_with_2plus_ids": len(multi),
        "multi_id_frames": multi,
        "gt_frame_count": len(per_frame),
        "gt_min_frame": min(per_frame) if per_frame else None,
        "gt_max_frame": max(per_frame) if per_frame else None,
    }


def _image_frames(img_dir: Path) -> tuple[int, int | None, int | None]:
    if not img_dir.is_dir():
        return 0, None, None
    frame_numbers: list[int] = []
    pattern = re.compile(r"^(\d+)")
    for path in img_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        match = pattern.match(path.stem)
        if match:
            frame_numbers.append(int(match.group(1)))
    if not frame_numbers:
        return 0, None, None
    return len(frame_numbers), min(frame_numbers), max(frame_numbers)


def _candidate_cache_metadata(sequence: str) -> dict[str, Any]:
    sidecar = ROOT / "outputs" / "n25r" / "candidate_aligned_features" / "clipreid" / "train30" / f"{sequence}.json"
    result: dict[str, Any] = {
        "sidecar": safe_rel(sidecar),
        "exists": sidecar.is_file(),
        "status": "NOT_AVAILABLE",
        "feature_dim": None,
        "row_count": 0,
        "valid_candidate_steps": 0,
        "selected_obj_id_coverage": None,
        "candidate_feature_coverage": None,
        "candidate_competition_proxy": False,
        "candidate_cache_is_complete_tape": False,
        "reason": "no_train30_sidecar",
    }
    if not sidecar.is_file():
        return result
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["reason"] = f"sidecar_unreadable:{type(exc).__name__}"
        return result
    feature_definition = str(payload.get("feature_definition", ""))
    match = re.search(r"projectedCLS(\d+)", feature_definition)
    validation = payload.get("validation") if isinstance(payload.get("validation"), dict) else {}
    result.update(
        {
            "status": payload.get("status", "UNKNOWN"),
            "feature_dim": int(match.group(1)) + 768 if match else None,
            "feature_definition": feature_definition,
            "row_count": int(payload.get("row_count", 0) or 0),
            "valid_candidate_steps": int(validation.get("valid_candidate_steps", 0) or 0),
            "selected_obj_id_coverage": validation.get("selected_obj_id_coverage"),
            "candidate_feature_coverage": validation.get("candidate_feature_coverage"),
            "candidate_competition_proxy": bool(
                int(validation.get("valid_candidate_steps", 0) or 0)
                > int(payload.get("row_count", 0) or 0)
            ),
            "candidate_cache_is_complete_tape": False,
            "reason": "episode_window_cache_no_frame_complete_public_id_mapping",
            "source_sha256": sha256(sidecar),
        }
    )
    return result


def build_inventory(data_root: Path) -> list[dict[str, Any]]:
    train_root = data_root / "train"
    if train_root.name.lower() in FORBIDDEN_SPLITS:
        raise RuntimeError("refusing to inspect forbidden split")
    rows: list[dict[str, Any]] = []
    for sequence_dir in sorted(path for path in train_root.iterdir() if path.is_dir()):
        if sequence_dir.name.lower() in FORBIDDEN_SPLITS:
            continue
        image_count, image_min, image_max = _image_frames(sequence_dir / "img1")
        gt = _parse_gt(sequence_dir / "gt" / "gt.txt")
        multi_frames = gt.get("multi_id_frames", [])
        future_eligibility = {
            str(horizon): [
                int(frame)
                for frame in multi_frames
                if image_max is not None and int(frame) + horizon <= int(image_max)
            ]
            for horizon in (20, 50, 100)
        }
        cache = _candidate_cache_metadata(sequence_dir.name)
        criteria = {
            "at_least_two_simultaneous_ids": gt["max_simultaneous_ids"] >= 2,
            "candidate_competition_observed_in_reusable_cache": bool(cache["candidate_competition_proxy"]),
            "future_window_20": bool(future_eligibility["20"]),
            "future_window_50": bool(future_eligibility["50"]),
            "future_window_100": bool(future_eligibility["100"]),
        }
        rows.append(
            {
                "sequence": sequence_dir.name,
                "dataset": "DanceTrack",
                "split": "train/train_fold",
                "sequence_path": str(sequence_dir),
                "images": {
                    "directory": str(sequence_dir / "img1"),
                    "count": image_count,
                    "min_frame": image_min,
                    "max_frame": image_max,
                },
                "gt": gt,
                "future_window_eligibility": future_eligibility,
                "candidate_features": cache,
                "selection_criteria": criteria,
                "candidate_complete_for_n34": False,
                "candidate_complete_reason": (
                    "existing cache is episode-window/top-k aligned and has no valid public-ID mapping; "
                    "it does not enumerate every SAM3 candidate for every frame"
                ),
            }
        )
    return rows


def run(data_root: Path) -> dict[str, Any]:
    output_root = ROOT / "outputs" / "n34"
    output_root.mkdir(parents=True, exist_ok=True)
    relevant = [
        "AGENTS.md",
        "docs/N32_FINAL_REPORT.md",
        "docs/N33_MECHANISM_REPORT.md",
        "sam3_intermot/association/appearance_memory.py",
        "sam3_intermot/association/ccam_replay.py",
        "sam3_intermot/association/state_manager.py",
        "sam3_intermot/association/online_associator.py",
        "sam3_intermot/association/human_intervention.py",
        "sam3_intermot/association/rollout.py",
        "scripts/run_n33_ccam_ablation.py",
        "tests/test_n33_ccam.py",
        "outputs/n33/ccam_ablation.json",
        "outputs/n32/selector_feature_audit_attempt2.json",
        "outputs/n31/candidate_rollout_index.json",
        "outputs/n31/episode_manifest.json",
        "outputs/n25r/feature_alignment.json",
    ]
    caller_scan = _caller_scan()
    backend_probe = _backend_capability_probe()
    event_terms = {
        term: sum(1 for hit in caller_scan.get(term, []))
        for term in ("ADD_NEW_IDENTITY", "AUTHORITATIVE_REASSIGN", "ATOMIC_ID_SWAP", "RECOVER_IDENTITY")
    }
    audit = {
        "protocol": "N34_CCAM_MECHANISM_FIRST_AUDIT",
        "status": "PASS",
        "generated_by": safe_rel(Path(__file__)),
        "authorized_data_policy": {
            "allowed_root": str(data_root / "train"),
            "allowed_split": "train/train_fold",
            "forbidden_content_splits": sorted(FORBIDDEN_SPLITS),
            "val_test_content_opened": False,
        },
        "existing_artifacts": artifact_metadata(relevant),
        "caller_scan": caller_scan,
        "event_interface_presence": event_terms,
        "reusable_implementation": {
            "single_ccam_memory": True,
            "appearance_memory": "sam3_intermot/association/appearance_memory.py",
            "paired_replay": "sam3_intermot/association/ccam_replay.py",
            "candidate_audit": "StateManager._candidate_audit",
            "human_event_attachment": "StateManager.annotate_human_event / sam3_intermot/association/rollout.py",
            "legacy_m0_opt_in": True,
        },
        "backend_capability_probe": backend_probe,
        "known_missing_fields_before_n34": [
            "real candidate-complete per-frame tape with every SAM3 candidate",
            "public-ID candidate score matrices for every frame/event",
            "complete event-to-sequence transaction ledger for all event types",
            "real CCAM future-effect replay at H20/H50/H100",
            "identity feature coverage in N32 selector audit (reported as zero)",
        ],
        "identity_feature_constraint": {
            "n32_selector_identity_features_available_episode_count": 0,
            "identity_aware_selector_authorized": False,
            "temporal_geometry_only_fallback": "allowed only as an explicitly labeled fallback; not an identity-aware result",
        },
        "n31_n33_boundary": {
            "n31_candidate_rollout_is_selected_episode_policy_cache": True,
            "n25r_candidate_cache_has_feature_rows": True,
            "n25r_selected_obj_id_mapping_valid": False,
            "n34_must_not_promote_old_cache_to_candidate_complete": True,
        },
    }
    atomic_json(output_root / "audit_before_run.json", audit)
    atomic_json(output_root / "backend_capability_probe.json", backend_probe)

    inventory = build_inventory(data_root)
    inventory_payload = {
        "protocol": "N34_DANCETRACK_TRAIN_SEQUENCE_INVENTORY",
        "status": "PASS" if inventory else "NOT_AVAILABLE",
        "data_root": str(data_root / "train"),
        "split": "train/train_fold",
        "sequence_count": len(inventory),
        "real_multi_id_data": bool(
            any(row["gt"]["max_simultaneous_ids"] >= 2 for row in inventory)
        ),
        "candidate_complete_sequence_count": 0,
        "sequences": inventory,
    }
    atomic_json(output_root / "sequence_inventory.json", inventory_payload)

    selected = [
        row
        for row in inventory
        if all(
            (
                row["selection_criteria"]["at_least_two_simultaneous_ids"],
                row["selection_criteria"]["candidate_competition_observed_in_reusable_cache"],
                row["selection_criteria"]["future_window_20"],
                row["selection_criteria"]["future_window_50"],
                row["selection_criteria"]["future_window_100"],
            )
        )
    ]
    selected_payload = {
        "protocol": "N34_SELECTED_DANCETRACK_TRAIN_SEQUENCES",
        "status": "PASS" if selected else "NOT_AVAILABLE",
        "selection_rule": "max simultaneous IDs >= 2 and at least one multi-ID frame with H20/H50/H100 available",
        "candidate_competition_requirement": "reusable cache competition is recorded, but does not establish candidate completeness",
        "candidate_complete": False,
        "candidate_complete_reason": "no per-frame all-SAM3-candidate public-ID tape is available",
        "sequence_count": len(selected),
        "sequences": [
            {
                "sequence": row["sequence"],
                "max_simultaneous_ids": row["gt"]["max_simultaneous_ids"],
                "unique_id_count": row["gt"]["unique_id_count"],
                "frames_with_2plus_ids": row["gt"]["frames_with_2plus_ids"],
                "future_window_20_frames": row["future_window_eligibility"]["20"][:10],
                "future_window_50_frames": row["future_window_eligibility"]["50"][:10],
                "future_window_100_frames": row["future_window_eligibility"]["100"][:10],
                "candidate_features": row["candidate_features"],
                "candidate_complete": False,
            }
            for row in selected
        ],
    }
    atomic_json(output_root / "selected_sequences.json", selected_payload)
    stage = {
        "stage": "N34-0/N34-1",
        "status": "PASS" if inventory else "NOT_AVAILABLE",
        "commands": [
            f"{Path(os.environ.get('N34_PYTHON', 'python'))} scripts/run_n34_audit.py",
        ],
        "artifacts": [
            "outputs/n34/audit_before_run.json",
            "outputs/n34/sequence_inventory.json",
            "outputs/n34/selected_sequences.json",
            "outputs/n34/backend_capability_probe.json",
        ],
        "errors": [],
        "next_action": "Build an explicit real candidate-complete tape if an authorized per-frame SAM3 candidate source can be materialized; otherwise run synthetic fallback with NOT_AVAILABLE labeling.",
        "real_multi_id_data": inventory_payload["real_multi_id_data"],
        "candidate_complete_tape": False,
    }
    atomic_json(output_root / "stage_01_status.json", stage)
    return {"audit": audit, "inventory": inventory_payload, "selected": selected_payload, "stage": stage}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    result = run(args.data_root)
    print(
        json.dumps(
            {
                "audit": "outputs/n34/audit_before_run.json",
                "sequence_inventory": "outputs/n34/sequence_inventory.json",
                "selected_sequences": "outputs/n34/selected_sequences.json",
                "sequence_count": result["inventory"]["sequence_count"],
                "selected_count": result["selected"]["sequence_count"],
                "real_multi_id_data": result["inventory"]["real_multi_id_data"],
                "candidate_complete": result["selected"]["candidate_complete"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
