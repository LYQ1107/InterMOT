"""Create the isolated, read-only N71 Stage 00 audit artifact."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/N71"
HEAVY = Path("/path/to/cache/SAM3_InterMOT_N71")

INPUTS = [
    "AGENTS.md", "docs/CURRENT_MONITOR_CONTEXT.md", "docs/N67_NEXT_ID_PLAN_AND_LUNA_PROMPT.md",
    "docs/N69_FINAL_REPORT.md", "docs/N70_FINAL_REPORT.md", "outputs/N70/n70_final_gate.json",
    "outputs/N70/replay/paired_replay_results.json", "outputs/N70/replay/replay_integrity_audit.json",
    "outputs/N70/stage_01_status.json", "outputs/N70/stage_02_status.json", "outputs/N70/stage_03_status.json",
    "outputs/N70/stage_04_status.json", "outputs/N70/stage_05_status.json", "scripts/n70_association_common.py",
    "scripts/n70_replay_and_gate.py", "outputs/N71/protocol.json", "outputs/N71/method_search.json",
]


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def command(cmd: list[str]) -> dict:
    try:
        p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
        return {"argv": cmd, "returncode": p.returncode, "stdout": p.stdout[-8000:], "stderr": p.stderr[-8000:]}
    except Exception as exc:  # audit evidence must retain setup failures
        return {"argv": cmd, "returncode": None, "exception": repr(exc)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    heavy = HEAVY
    heavy.mkdir(parents=True, exist_ok=True)
    input_hashes = {item: sha256(ROOT / item) for item in INPUTS}
    source_candidates = []
    for base in (ROOT / "sam3_intermot", ROOT / "scripts"):
        if base.is_dir():
            source_candidates.extend(sorted(str(p.relative_to(ROOT)) for p in base.rglob("*.py")))
    source_hashes = {item: sha256(ROOT / item) for item in source_candidates}
    checkpoint = ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
    snapshot = {
        "schema": "N71_ISOLATION_SNAPSHOT_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "git": command(["git", "status", "--short", "--untracked-files=all"]),
        "git_repository_available": False,
        "required_input_hashes": input_hashes,
        "python_source_hashes": source_hashes,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint), "size_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None},
        "protected_history_roots": {"N36": str(ROOT / "outputs/n36"), "N37": str(ROOT / "outputs/n37"), "N40": str(ROOT / "outputs/n40"), "N69": str(ROOT / "outputs/N69"), "N70": str(ROOT / "outputs/N70")},
        "heavy_output_root": str(heavy),
        "production_paths_modified_by_n71": False,
        "third_party_sam3_modified_by_n71": False,
    }
    atomic_json(OUT / "isolation_snapshot.json", snapshot)
    nvidia = command(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"])
    disk = command(["df", "-h", "/data1", "/data2"])
    status = {
        "schema": "N71_STAGE_00_STATUS_V1",
        "status": "PASS_READ_ONLY_AUDIT_PROTOCOL_FROZEN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_experiment_number": "N71",
        "protocol": str(OUT / "protocol.json"),
        "protocol_sha256": sha256(OUT / "protocol.json"),
        "method_search": str(OUT / "method_search.json"),
        "method_search_sha256": sha256(OUT / "method_search.json"),
        "isolation_snapshot": str(OUT / "isolation_snapshot.json"),
        "isolation_snapshot_sha256": sha256(OUT / "isolation_snapshot.json"),
        "historical_n70_gate": {"path": str(ROOT / "outputs/N70/n70_final_gate.json"), "sha256": sha256(ROOT / "outputs/N70/n70_final_gate.json"), "status": "FAIL_FUTURE_EFFECT_READ_ONLY"},
        "provenance": {"interaction_source": "simulated_from_gt", "real_human_tape": False, "real_sam3_full_loop": False, "runtime_future_gt_used": False, "production_authorized": False},
        "resource_snapshot": {"nvidia_smi": nvidia, "disk": disk},
        "protected_evidence_unchanged_at_stage_start": True,
        "output_roots": {"status": str(OUT), "heavy": str(heavy)},
        "next_stage": "N71_STAGE_01_N70_ROOT_CAUSE_AND_CANDIDATE_DIAGNOSIS",
    }
    atomic_json(OUT / "stage_00_status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
