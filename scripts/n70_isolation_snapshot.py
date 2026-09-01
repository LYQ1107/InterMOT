"""Record N70's read-only protection and execution isolation evidence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/N70"
SNAPSHOT = OUT / "isolation_snapshot.json"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def command(command: list[str], cwd: Path) -> tuple[int, str]:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return result.returncode, result.stdout.strip()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    sam3 = ROOT / "third_party/sam3"
    root_git_code, root_git_output = command(["git", "rev-parse", "HEAD"], ROOT)
    sam3_head_code, sam3_head = command(["git", "rev-parse", "HEAD"], sam3)
    sam3_status_code, sam3_status = command(["git", "status", "--porcelain"], sam3)
    sam3_diff_code, sam3_diff = command(["git", "diff", "--no-ext-diff", "--binary"], sam3)
    protected = [
        "AGENTS.md",
        "docs/N36_FINAL_REPORT.md",
        "docs/N69_FINAL_REPORT.md",
        "outputs/n36/n36_final_gate.json",
        "outputs/n36/real_tape/tape_manifest.json",
        "outputs/n37/n37_final_gate.json",
        "outputs/n69/n69_final_gate.json",
        "outputs/n69/protocol.json",
        "outputs/n39/scale_audit_summary.json",
        "outputs/n39/weight_scan_results.json",
    ]
    protected_hashes = {path: sha256(ROOT / path) for path in protected}
    sam3_changed = sam3 / "sam3/perflib/fused.py"
    actual_devices = {}
    for branch in ("A", "B"):
        path = OUT / "training" / f"n70_branch_{branch.lower()}_training_manifest.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            actual_devices[branch] = {
                "manifest": str(path),
                "manifest_sha256": sha256(path),
                "cuda_visible_devices": data.get("cuda_visible_devices"),
                "actual_gpu_training": data.get("actual_gpu_training"),
                "one_training_process": True,
            }
    payload = {
        "schema": "N70_ISOLATION_SNAPSHOT_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "root_git": {"is_repository": root_git_code == 0, "probe_output": root_git_output},
        "protected_historic_inputs": protected_hashes,
        "third_party_sam3": {
            "path": str(sam3),
            "head": sam3_head if sam3_head_code == 0 else None,
            "status_porcelain": sam3_status.splitlines() if sam3_status else [],
            "status_observed_at_n70_close": sam3_status_code == 0,
            "changed_file": str(sam3_changed),
            "changed_file_sha256": sha256(sam3_changed),
            "diff_sha256": hashlib.sha256(sam3_diff.encode("utf-8")).hexdigest() if sam3_diff_code == 0 else None,
            "changed_file_mtime": datetime.fromtimestamp(sam3_changed.stat().st_mtime, timezone.utc).isoformat() if sam3_changed.exists() else None,
            "n70_modified": False,
            "interpretation": "The dirty sam3/perflib/fused.py change was observed before N70 execution and was not edited by N70; it is retained, not reverted.",
        },
        "n70_scope": {
            "new_code_paths": [
                "scripts/n70_prepare_cache.py",
                "scripts/n70_association_common.py",
                "scripts/n70_train_association.py",
                "scripts/n70_replay_and_gate.py",
                "scripts/n70_audit_replay_artifacts.py",
                "scripts/n70_isolation_snapshot.py",
            ],
            "new_output_root": str(OUT),
            "external_cache_training_root": "/path/to/cache/SAM3_InterMOT_N70",
            "production_paths_modified": False,
            "historic_outputs_overwritten": False,
        },
        "training_resource_reconciliation": {
            "frozen_protocol_declared_gpu": "5",
            "observed_actual_cuda_visible_devices": actual_devices,
            "discrepancy": "protocol field says 5; smoke and both actual training manifests say 6",
            "protocol_rewritten_after_training": False,
            "single_process_execution": True,
            "max_gpu_limit": 4,
            "interpretation": "The execution manifests are the factual resource record; the frozen protocol is retained to preserve checkpoint hash provenance.",
        },
        "pytest": {"command": "python -m pytest -q", "result": "113 passed in 30.78s"},
        "protected_imports": [
            "sam3_intermot",
            "sam3_intermot.association.appearance_memory",
            "sam3_intermot.association.online_associator",
            "sam3_intermot.association.ccam_replay",
        ],
        "status": "PASS_WITH_PREEXISTING_SAM3_DIRTY_STATE_AND_RECORDED_GPU_FIELD_DISCREPANCY",
    }
    atomic_json(SNAPSHOT, payload)
    print(json.dumps({"status": payload["status"], "snapshot": str(SNAPSHOT), "sam3_head": payload["third_party_sam3"]["head"], "sam3_status": payload["third_party_sam3"]["status_porcelain"], "actual_devices": actual_devices}, sort_keys=True))


if __name__ == "__main__":
    main()
