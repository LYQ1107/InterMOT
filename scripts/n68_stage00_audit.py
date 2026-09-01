"""N68 Stage 00: immutable-input and resource audit.

This is a read-only audit of the workspace.  It does not import a dataset
reader, start a model, mutate a frozen N67 artifact, or inspect future GT.
The resulting JSON is a provenance record for the isolated N68 experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "n68"
AUDIT = OUT / "stage_00_readonly_audit.json"
STATUS = OUT / "stage_00_status.json"

FROZEN_INPUTS = [
    "AGENTS.md",
    "docs/PROJECT_STATE_AND_NEW_MONITOR_HANDOFF.md",
    "docs/N35_FINAL_REPORT.md",
    "docs/N36_LUNA_LONG_TASK_PROMPT.md",
    "outputs/n35/real_tape_failure_evidence.json",
    "outputs/n36/stage_01_status.json",
    "outputs/n36/stage_02_status.json",
    "outputs/n36/smoke_integrity_audit.json",
    "docs/N67_FINAL_REPORT.md",
    "outputs/n67/replay/paired_replay_results.json",
    "outputs/n67/replay/runtime_status.json",
    "outputs/n67/replay/stage_06_integrity.json",
    "docs/N67_NEXT_ID_PLAN_AND_LUNA_PROMPT.md",
]

CODE_ROOTS = [
    "sam3_intermot/association",
    "sam3_intermot/interaction",
    "sam3_intermot/tracking",
    "sam3_intermot/backend",
    "sam3_intermot/datasets",
    "sam3_intermot/evaluation",
    "scripts",
]

PRODUCTION_PATHS = [
    "sam3_intermot/backend",
    "sam3_intermot/tracking",
    "sam3_intermot/evaluation",
    "sam3_intermot/datasets",
    "third_party/sam3",
]

CHECKPOINTS = [
    "outputs/n66/training/n66r1_corrected_n51_cosine_risk.pt",
    "outputs/n67/training/n67r1_pairwise_crossing_action_magnitude.pt",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "is_file": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
    }


def command(args: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, check=False)
        return {
            "argv": args,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except OSError as exc:
        return {"argv": args, "returncode": None, "stdout": "", "stderr": repr(exc)}


def gpu_audit() -> dict[str, Any]:
    query = command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[dict[str, Any]] = []
    for line in query.get("stdout", "").splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 6:
            continue
        gpus.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_total_mib": int(fields[2]),
                "memory_used_mib": int(fields[3]),
                "memory_free_mib": int(fields[4]),
                "utilization_percent": int(fields[5]),
            }
        )
    apps = command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
    )
    return {"query": query, "gpus": gpus, "compute_apps": apps}


def disk_audit() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw in ("/data1", "/data2"):
        usage = shutil.disk_usage(raw)
        result[raw] = {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "free_gib": round(usage.free / (1024**3), 3),
        }
    return result


def max_experiment_number(excluded: set[int] | None = None) -> int:
    excluded = excluded or set()
    values: list[int] = []
    for parent in (ROOT / "outputs", ROOT / "docs"):
        if not parent.exists():
            continue
        for item in parent.iterdir():
            match = re.match(r"^(?:N|n)(\d+)", item.name)
            if match and int(match.group(1)) not in excluded:
                values.append(int(match.group(1)))
    return max(values, default=0)


def code_inventory() -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for root_name in CODE_ROOTS:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            rel = path.relative_to(ROOT).as_posix()
            files.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    aggregate = hashlib.sha256()
    for item in files:
        aggregate.update(f"{item['path']}:{item['sha256']}\n".encode())
    return {"roots": [str(ROOT / item) for item in CODE_ROOTS], "file_count": len(files), "files": files, "tree_sha256": aggregate.hexdigest()}


def main() -> None:
    # Capture these values before creating N68 itself; otherwise the audit
    # would count its own output as a pre-existing experiment.
    prior_n68_exists = OUT.exists() and any(OUT.iterdir())
    # The marker is created only by the first failed invocation of this
    # audit, so it lets a repaired rerun distinguish our own N68 directory
    # from an N68 directory that existed before this task began.
    self_created_marker = OUT / "attempts" / "stage_00_initial_audit_failure.json"
    if self_created_marker.exists():
        prior_n68_exists = False
    latest_existing = max_experiment_number({68})
    OUT.mkdir(parents=True, exist_ok=True)
    git = command(["git", "-C", str(ROOT), "status", "--short"])
    audit = {
        "schema": "N68_STAGE_00_READONLY_AUDIT_V1",
        "created_at_utc": now(),
        "project_root": str(ROOT),
        "latest_existing_experiment_number": latest_existing,
        "n68_output_preexisting": prior_n68_exists,
        "git": {"available": git.get("returncode") == 0, "command": git},
        "frozen_inputs": [file_record(item) for item in FROZEN_INPUTS],
        "checkpoints": [file_record(item) for item in CHECKPOINTS],
        "code_inventory": code_inventory(),
        "production_paths": [str(ROOT / item) for item in PRODUCTION_PATHS],
        "resources": {"gpu": gpu_audit(), "disk": disk_audit()},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
        "isolation": {
            "new_experiment": "N68",
            "new_evidence_root": str(OUT),
            "large_cache_root": "/path/to/cache/SAM3_InterMOT_n68",
            "frozen_outputs_modified": False,
            "mot_ovmot_production_modified": False,
            "third_party_sam3_modified": False,
            "git_metadata_available": git.get("returncode") == 0,
        },
    }
    AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    status = {
        "schema": "N68_STAGE_00_STATUS_V1",
        "status": "PASS_READONLY_AUDIT",
        "audit": str(AUDIT),
        "latest_existing_experiment_number": audit["latest_existing_experiment_number"],
        "selected_experiment": "N68",
        "git_status": audit["git"],
        "gpu": audit["resources"]["gpu"],
        "disk": audit["resources"]["disk"],
        "next_action": "Run N68 Stage 01 baseline/identity-scope audit without modifying N67 or production paths.",
        "interaction_source": "simulated_from_gt",
        "real_human_tape": False,
        "real_sam3_full_loop": False,
        "runtime_future_gt_used": False,
        "production_authorized": False,
    }
    STATUS.write_text(json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status["status"], "audit": str(AUDIT), "latest": audit["latest_existing_experiment_number"]}, sort_keys=True))


if __name__ == "__main__":
    main()
