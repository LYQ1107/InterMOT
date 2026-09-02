"""Freeze N72R1 inputs without copying large historical artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SOURCE_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT")
N72R1_ROOT = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R1")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_tree_state(path: Path) -> dict[str, object]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "is_git_repository": completed.returncode == 0,
            "head": completed.stdout.strip() if completed.returncode == 0 else None,
            "stderr": completed.stderr.strip() if completed.returncode else None,
        }
    except OSError as exc:
        return {"is_git_repository": False, "head": None, "stderr": str(exc)}


def collect_files(root: Path, predicate) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not predicate(path):
            continue
        stat = path.stat()
        entries.append(
            {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(path),
            }
        )
    return entries


def checkpoint_inventory(root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        entries.append(
            {
                "path": str(path),
                "size": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


def main() -> None:
    status_root = N72R1_ROOT / "status"
    protocol_root = N72R1_ROOT / "protocol"
    audit_root = N72R1_ROOT / "audits"
    for directory in (status_root, protocol_root, audit_root):
        directory.mkdir(parents=True, exist_ok=True)

    key_paths = [
        SOURCE_ROOT / "docs/N72_FINAL_REPORT.md",
        SOURCE_ROOT / "outputs/N72/n72_final_status.json",
        SOURCE_ROOT / "outputs/N72/protocol.json",
        SOURCE_ROOT / "outputs/N72/preservation_audit.json",
        SOURCE_ROOT / "docs/N71_FINAL_REPORT.md",
        SOURCE_ROOT / "outputs/N71/n71_final_gate.json",
        SOURCE_ROOT / "docs/N70_FINAL_REPORT.md",
        SOURCE_ROOT / "research_log.md",
    ]
    key_files = []
    for path in key_paths:
        if path.is_file():
            key_files.append(
                {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
            )
        else:
            key_files.append({"path": str(path), "missing": True})

    protection = {
        "schema_version": "N72R1_PRE_RUN_PROTECTION_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(SOURCE_ROOT),
        "n72r1_root": str(N72R1_ROOT),
        "source_git": git_tree_state(SOURCE_ROOT),
        "protected_scope": {
            "python_source": collect_files(SOURCE_ROOT, lambda p: p.suffix == ".py"),
            "configs": collect_files(SOURCE_ROOT / "configs", lambda p: True),
            "tests": collect_files(SOURCE_ROOT / "tests", lambda p: p.suffix == ".py"),
            "third_party_sam3": collect_files(SOURCE_ROOT / "third_party/sam3", lambda p: True),
            "checkpoint_files": checkpoint_inventory(SOURCE_ROOT / "checkpoints"),
            "historical_attempts": collect_files(SOURCE_ROOT / "attempts", lambda p: True),
            "key_evidence": key_files,
        },
        "preservation_policy": {
            "source_root_modified": False,
            "historical_outputs_read_only": True,
            "third_party_sam3_modified": False,
            "checkpoint_modified": False,
            "large_outputs_copied": False,
            "n72r1_isolated_worktree": True,
        },
    }
    protection_path = audit_root / "pre_run_protection_manifest.json"
    protection_path.write_text(json.dumps(protection, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    protocol = {
        "schema_version": "N72R1_PROTOCOL_V1",
        "name": "N72R1_SAME_RUN_PUBLIC_MAPPING_AND_REAL_HUMAN_RUNTIME_CLOSURE",
        "source_root": str(SOURCE_ROOT),
        "isolated_root": str(N72R1_ROOT),
        "frozen_facts": {
            "n71_assignment_change": 0,
            "n71_interaction_source": "simulated_from_gt",
            "n72_raw_stable_axis_confusion": True,
            "n72_official_candidate_count": 9333,
            "n72_exact_public_mapping_rows": 0,
            "n72_real_human_event_count": 0,
            "n70_axis_mismatch": 70,
            "n70_target_candidate_absent": 90,
            "n70_public_assignment_absent": 10,
        },
        "non_goals": [
            "training",
            "calibration_head",
            "selector",
            "decoder_lora",
            "weight_or_threshold_scan",
            "checkpoint_replacement",
            "candidate_generator_change",
            "hungarian_solver_change",
            "future_gt_event_selection",
        ],
        "runtime_contract": {
            "y_pre_frozen_before_correction": True,
            "event_frame_memory_read": False,
            "first_memory_read_offset": 1,
            "runtime_future_gt_used": False,
            "append_only_artifacts": True,
        },
        "fixed_official_smoke": {
            "max_num_objects": 16,
            "multiplex_count": 16,
            "output_prob_thresh": 0.30,
            "offload_video_to_cpu": True,
            "window_frames": 160,
            "overlap_frames": 20,
            "gpu_count_max": 1,
        },
        "source_evidence_hashes": key_files,
        "protection_manifest": str(protection_path),
    }
    (protocol_root / "n72r1_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    status = {
        "schema_version": "N72R1_STAGE_STATUS_V1",
        "stage": "00",
        "status": "PASS_BASELINE_FROZEN",
        "source_root": str(SOURCE_ROOT),
        "isolated_root": str(N72R1_ROOT),
        "protection_manifest": str(protection_path),
        "protocol": str(protocol_root / "n72r1_protocol.json"),
        "source_git": protection["source_git"],
        "large_outputs_copied": False,
        "third_party_sam3_modified": False,
        "checkpoint_modified": False,
        "next_stage": "01",
        "notes": [
            "The source tree is not a Git repository; N72R1 uses an isolated rsync worktree.",
            "Checkpoint hashes are recorded without copying checkpoint bytes.",
            "Historical N70-N72 outputs are referenced read-only and are not copied into the worktree.",
        ],
        "platform": {"python": platform.python_version(), "uname": os.uname().sysname},
    }
    (status_root / "stage_00_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status["status"], "key_files": len(key_files), "worktree": str(N72R1_ROOT)}))


if __name__ == "__main__":
    main()
