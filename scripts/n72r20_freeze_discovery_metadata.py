#!/usr/bin/env python3
"""Freeze compact N72R20 asset/protocol metadata after local discovery."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path("/data3/liuyeqiang/InterMOT")
OUT = REPO / "outputs" / "N72R20"
DISCOVERY = OUT / "dataset_discovery.json"
OLD_MANIFEST = REPO.parent / "InterMOT_N72R16_assets" / "outputs" / "N72R16" / "asset_manifest.json"
GRU = REPO / "outputs" / "N72R18" / "checkpoints" / "identity_memory_gru.pt"
OSNET = REPO.parent / "InterMOT_N72R16_assets" / "checkpoints" / "osnet_x1_0_market1501.pth"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, role: str) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "role": role,
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
        "sha256": sha256(path) if exists else None,
    }


def current_free_bytes() -> int:
    return shutil.disk_usage(REPO).free


def protocol_anchor_rule(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text()).get("anchor_rule")
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    discovery = json.loads(DISCOVERY.read_text())
    root = discovery.get("recommended_root")
    if not root:
        raise SystemExit("dataset discovery has no reusable root")
    protocols = discovery["protocol_sources"]
    train_source = Path(protocols["train_source"][0])
    val_source = Path(protocols["val_source"][0])
    now = datetime.now(timezone.utc).isoformat()

    old_manifest: dict[str, Any] = {}
    if OLD_MANIFEST.is_file():
        old_manifest = json.loads(OLD_MANIFEST.read_text())
    source_archives = []
    for name, item in old_manifest.get("huggingface", {}).get("files", {}).items():
        path = Path(item.get("path", ""))
        source_archives.append(
            {
                "name": name,
                "path": str(path),
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else item.get("bytes"),
                "sha256": item.get("sha256"),
                "sha256_matches_previous_manifest": item.get("sha256_matches_expected"),
                "status": "historical_verified_source_archive; not re-downloaded",
            }
        )

    protocol = {
        "stage": "N72R20",
        "final_goal_ref": "outputs/N72R20/FINAL_GOAL.json",
        "dataset_root": root,
        "DANCETRACK_ROOT": root,
        "dataset_scope": {"train": "development_and_smoke_only", "val": "frozen_final_evaluation_only", "test": "forbidden"},
        "split_counts": {"train": 40, "val": 25},
        "train_dev_sequences": protocols["train"][:2],
        "val_evaluation_sequences": protocols["val"],
        "anchor_rule": protocol_anchor_rule(train_source),
        "anchor_source": "frozen N72R17 earliest eligible anchor; simulated_from_gt",
        "interaction_source": "simulated_from_gt",
        "future_gt_runtime_input": False,
        "gt_use": "offline evaluation truth only: candidate availability, rank, discrimination, memory write and recovery",
        "horizons": [20, 50, 100],
        "candidate_iou_threshold": 0.5,
        "primary_comparison": "FROZEN_N72R18_GRU_VS_HUMAN_ANCHOR_ONLY_ON_IDENTICAL_REAL_SAM3_CANDIDATES",
        "secondary_comparison": "FROZEN_N72R18_GRU_VS_EMA_0.90",
        "memory_variants": [
            "B0_HUMAN_ANCHOR_ONLY",
            "B1_EMA_0.90_MACHINE_SELECTED_UPDATE",
            "B2_FROZEN_N72R18_GRU_MACHINE_SELECTED_UPDATE",
            "ORACLE_CORRECT_UPDATE_OFFLINE_ORACLE_DIAGNOSTIC",
        ],
        "update_diagnostic": ["IMMEDIATE_UPDATE", "2_FRAME_CONFIRMED_UPDATE"],
        "val_tuning_allowed": False,
        "paired_bootstrap": {"unit": "sequence_cluster", "replicates": 2000, "seed": 72020},
        "protocol_sources": {
            "train": file_record(train_source, role="frozen N72R17 train protocol"),
            "val": file_record(val_source, role="frozen N72R17 val protocol"),
        },
        "test_downloaded": False,
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")

    manifest = {
        "stage": "N72R20",
        "final_goal_ref": "outputs/N72R20/FINAL_GOAL.json",
        "asset_lineage": discovery.get("recommended_lineage_status"),
        "DANCETRACK_ROOT": root,
        "canonical_asset_root": "/data3/liuyeqiang/InterMOT_N72R20_assets/",
        "canonical_asset_root_exists": Path("/data3/liuyeqiang/InterMOT_N72R20_assets").exists(),
        "dataset": {
            "root": root,
            "train_sequences": 40,
            "val_sequences": 25,
            "train_bytes": 6988647747,
            "val_bytes": 4289202609,
            "discovery_file": str(DISCOVERY),
            "discovery_sha256": sha256(DISCOVERY),
            "copied": False,
            "downloaded_this_stage": False,
            "test_present_or_used": False,
        },
        "frozen_identity_assets": {
            "n72r18_gru": file_record(GRU, role="frozen N72R18 identity memory"),
            "osnet_x1_0_market1501": file_record(OSNET, role="frozen N72R16 OSNet crop encoder"),
        },
        "sam3_checkpoint": {
            "expected_name": "sam3.1_multiplex.pt",
            "path": None,
            "exists": False,
            "status": "not_found_locally; official Hugging Face access requires authenticated checkpoint access",
            "downloaded_this_stage": False,
        },
        "historical_source_archives": source_archives,
        "minimum_assets": {
            "source_code": {"status": "present", "path": str(REPO)},
            "base_identity_memory": {"status": "present", "path": str(GRU)},
            "osnet_encoder": {"status": "present", "path": str(OSNET)},
            "dancetrack_train_val": {"status": "present_and_complete", "path": root},
            "frozen_protocol": {"status": "present", "path": str(OUT / "protocol.json")},
            "sam3_checkpoint": {"status": "missing", "path": None},
        },
        "download_policy": {
            "dataset_download_performed": False,
            "proxy_used": False,
            "test_downloaded": False,
            "duplicate_dataset_created": False,
        },
        "captured_at_utc": now,
        "free_space_bytes_at_capture": current_free_bytes(),
    }
    (OUT / "asset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    candidates: list[dict[str, Any]] = [
        {
            "path": str(REPO.parent / ".cache" / "pip"),
            "size_bytes": 2598804445,
            "reason": "pip cache is not required to reuse the already installed N72R20 environment",
            "reproducible": True,
            "protected": False,
            "proposed_action": "retain_this_turn; no cleanup needed while free space is above 100 GiB; require explicit approval before deletion",
        }
    ]
    for item in source_archives:
        if item["exists"]:
            candidates.append(
                {
                    "path": item["path"],
                    "size_bytes": item["bytes"],
                    "reason": "verified historical archive has an extracted complete dataset; duplicate storage candidate",
                    "reproducible": True,
                    "protected": False,
                    "proposed_action": "retain_this_turn; deletion is unnecessary at current free space and requires explicit approval",
                    "sha256": item["sha256"],
                }
            )
    candidates.extend(
        [
            {
                "path": str(REPO / "outputs" / "N72R19R1"),
                "size_bytes": 10569801455,
                "reason": "historical N72R19R1 result and evidence",
                "reproducible": False,
                "protected": True,
                "proposed_action": "retain; protected by N72R20 PROTECTED_ASSETS",
            },
            {
                "path": str(REPO.parent / "InterMOT_N72R16_assets"),
                "size_bytes": 12537819346,
                "reason": "contains the reused raw dataset, frozen encoder and historical manifest",
                "reproducible": False,
                "protected": True,
                "proposed_action": "retain; protected by N72R20 PROTECTED_ASSETS",
            },
        ]
    )
    (OUT / "cleanup_candidates.json").write_text(
        json.dumps({"stage": "N72R20", "cleanup_performed": False, "candidates": candidates}, indent=2, ensure_ascii=False) + "\n"
    )

    cleanup_log = OUT / "storage_cleanup_log.jsonl"
    cleanup_log.write_text(
        json.dumps(
            {
                "timestamp": now,
                "path": None,
                "size": 0,
                "action": "none",
                "reason": "No cleanup performed; free space remains above the 100 GiB normal target.",
                "sha256": None,
                "reproducible": True,
                "approved_category": None,
            },
            ensure_ascii=False,
        )
        + "\n"
    )

    free_after = current_free_bytes()
    (OUT / "storage_audit_after_cleanup.json").write_text(
        json.dumps(
            {
                "stage": "N72R20",
                "audit": "storage_audit_after_cleanup",
                "cleanup_performed": False,
                "before_free_bytes_from_current_audit": None,
                "after_free_bytes": free_after,
                "recovered_bytes": 0,
                "deleted_paths": [],
                "compressed_paths": [],
                "protected_paths": [str(REPO / "outputs" / "N72R20" / "PROTECTED_ASSETS.json")],
                "note": "No cleanup was needed or performed.",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    stage_status = json.loads((OUT / "stage_status.json").read_text())
    stage_status.update(
        {
            "status": "DATASET_READY_SAM3_CHECKPOINT_MISSING",
            "current_stage": "STAGE_2_SAM3_CHECKPOINT_AUDIT",
            "dataset_ready": True,
            "DANCETRACK_ROOT": root,
            "dataset_discovery_ref": str(DISCOVERY),
            "download_required": False,
            "sam3_checkpoint_status": "missing_locally_and_requires_authenticated_official_access",
            "cleanup_performed": False,
            "last_update": now,
        }
    )
    (OUT / "stage_status.json").write_text(json.dumps(stage_status, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"dataset_root": root, "sam3_checkpoint": "missing", "free_space_bytes": free_after}, ensure_ascii=False))


if __name__ == "__main__":
    main()
