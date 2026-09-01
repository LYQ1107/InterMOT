"""N69 Stage 00: immutable-input, resource, and isolation audit.

This file is deliberately a sidecar script.  It never imports or edits the
production association implementation, and it writes only to ``outputs/n69``.
The audit is run before any N69 output exists so the experiment-number check
cannot count its own files.
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
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "n69"
AUDIT = OUT / "stage_00_readonly_audit.json"
STATUS = OUT / "stage_00_status.json"
PROTOCOL = OUT / "protocol.json"

FROZEN_INPUTS = [
    "AGENTS.md",
    "docs/N68_FINAL_REPORT.md",
    "outputs/n68/n68_final_gate.json",
    "outputs/n68/stage_00_status.json",
    "outputs/n68/stage_01_status.json",
    "outputs/n68/stage_02_status.json",
    "outputs/n68/stage_02_dataset_status.json",
    "outputs/n68/stage_03_status.json",
    "outputs/n68/stage_04_status.json",
    "outputs/n68/stage_05_status.json",
    "outputs/n68/stage_06_status.json",
    "outputs/n68/replay/paired_replay_results.json",
    "outputs/n68/replay/stage03_paired_replay_results.json",
    "outputs/n68/replay/runtime_status.json",
    "outputs/n68/replay/stage03_runtime_status.json",
    "docs/N67_FINAL_REPORT.md",
    "docs/N67_NEXT_ID_PLAN_AND_LUNA_PROMPT.md",
    "outputs/n67/replay/paired_replay_results.json",
    "outputs/n67/replay/runtime_status.json",
    "outputs/n67/replay/stage_06_integrity.json",
    "outputs/n37/real_event_manifest.json",
    "outputs/n54/replay/runtime_status.json",
]

PROTECTED_ROOTS = [
    "sam3_intermot",
    "third_party/sam3",
    "configs",
    "environment",
]

PROTECTED_FILES = [
    "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt",
    "outputs/n52/training/n52_monotonic_cosine_risk.pt",
    "outputs/n54/training/n54r1_targetness.pt",
    "outputs/n67/training/n67r1_pairwise_crossing_action_magnitude.pt",
    "outputs/n68/training/n68_identity_local_head.pt",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dfd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def command(argv: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)
        return {
            "argv": argv,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except OSError as exc:
        return {"argv": argv, "returncode": None, "stdout": "", "stderr": repr(exc)}


def file_record(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "relative": relative,
        "exists": True,
        "is_file": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
    }


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


def tree_inventory(relative_root: str, suffixes: set[str] | None = None) -> dict[str, Any]:
    root = ROOT / relative_root
    files: list[dict[str, Any]] = []
    if root.exists():
        for path in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            if suffixes is not None and path.suffix.lower() not in suffixes:
                continue
            rel = path.relative_to(ROOT).as_posix()
            files.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)})
    aggregate = hashlib.sha256()
    for item in files:
        aggregate.update(f"{item['path']}:{item['sha256']}\n".encode())
    return {
        "root": str(root),
        "file_count": len(files),
        "files": files,
        "tree_sha256": aggregate.hexdigest(),
    }


def gpu_audit() -> dict[str, Any]:
    query = command([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    gpus: list[dict[str, Any]] = []
    for line in query.get("stdout", "").splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 6:
            continue
        try:
            gpus.append({
                "index": int(fields[0]),
                "name": fields[1],
                "memory_total_mib": int(fields[2]),
                "memory_used_mib": int(fields[3]),
                "memory_free_mib": int(fields[4]),
                "utilization_percent": int(fields[5]),
            })
        except ValueError:
            continue
    apps = command([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader",
    ])
    return {"query": query, "gpus": gpus, "compute_apps": apps}


def disk_audit() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw in ("/data1", "/data2"):
        try:
            usage = shutil.disk_usage(raw)
            result[raw] = {
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "free_gib": round(usage.free / (1024**3), 3),
            }
        except OSError as exc:
            result[raw] = {"error": repr(exc)}
    return result


def protocol_payload() -> dict[str, Any]:
    split_path = ROOT / "outputs/n68/stage_02_protocol.json"
    old = json.loads(split_path.read_text(encoding="utf-8"))
    split = old["sequence_split"]
    return {
        "schema": "N69_ID_SCOPED_TARGET_CONDITIONED_ASSOCIATION_PROTOCOL_V1",
        "status": "FROZEN_BEFORE_N69_MATERIALIZATION",
        "created_at_utc": now(),
        "experiment": "N69",
        "hypothesis": "A target-conditioned scorer using raw 512-D masked embeddings, historical target memory and explicit hard negatives can improve the known public-ID column without disturbing untouched IDs; this is a GT-simulated mechanism test, not production evidence.",
        "frozen_parent": {
            "n68_final_gate": str(ROOT / "outputs/n68/n68_final_gate.json"),
            "n68_learned_result": str(ROOT / "outputs/n68/replay/paired_replay_results.json"),
            "n68_margin_result": str(ROOT / "outputs/n68/replay/stage03_paired_replay_results.json"),
        },
        "data": {
            "event_manifest": str(ROOT / "outputs/n37/real_event_manifest.json"),
            "candidate_runtime": str(ROOT / "outputs/n54/replay/runtime"),
            "events": 24,
            "independent_sequences": 21,
            "future_frames_per_event_variant": 100,
            "upstream_variants": ["M0", "M1", "M2", "M3", "M4"],
            "action_types": ["ADD_NEW_IDENTITY", "AUTHORITATIVE_REASSIGN", "ATOMIC_ID_SWAP", "RECOVER_IDENTITY"],
            "sequence_split": split,
        },
        "model": {
            "name": "N69_LOW_RANK_TARGET_CONDITIONED_LISTWISE_NONE_SCORER",
            "candidate_embedding_dim": 512,
            "projection_dim": 64,
            "inputs": [
                "raw_candidate_embedding_512",
                "raw_human_anchor_512",
                "raw_target_memory_prototype_512",
                "raw_hard_negative_summary_512",
                "projected_candidate_anchor_product_64",
                "projected_candidate_anchor_absdiff_64",
                "projected_candidate_memory_product_64",
                "projected_candidate_memory_absdiff_64",
                "projected_candidate_hard_negative_product_64",
                "projected_candidate_hard_negative_absdiff_64",
                "audited_geometry_temporal_context",
            ],
            "shared_low_rank_projection": True,
            "output": ["target_logit", "none_logit"],
            "application": "bounded target-public-ID-column residual only; global Hungarian and explicit NONE remain frozen",
            "numeric_public_id_feature": False,
            "target_native_id_feature": False,
            "future_gt_feature": False,
        },
        "training": {
            "loss": [
                "candidate target-vs-none cross entropy",
                "within-frame hard-negative/listwise softplus ranking",
                "temporal consistency on adjacent target rows",
                "explicit NONE frame loss",
                "untouched/no-op residual penalty",
                "identity-scope consistency",
            ],
            "seed": 6901,
            "optimizer": "AdamW",
            "learning_rate": 0.0005,
            "weight_decay": 0.0001,
            "batch_size": 512,
            "max_epochs": 30,
            "early_stopping_patience": 5,
            "selection": "earliest minimum validation composite loss; holdout once after selection",
            "holdout_used_for_selection": False,
        },
        "runtime_boundary": {
            "event_frame_memory_read": False,
            "first_memory_visible_frame": "event_frame+1",
            "runtime_future_gt_used": False,
            "gt_loaded_only_offline_materialization_labels_and_posthoc": True,
        },
        "replay": {
            "methods": ["CURRENT_CCAM_BASELINE", "M0", "M1", "M2", "M3", "M4", "N69_TARGET_CONDITIONED"],
            "horizons": [20, 50, 100],
            "bootstrap": {"cluster": "sequence", "repetitions": 2000, "seed": 6908},
            "candidate_stream_changed": False,
            "hungarian_solver_changed": False,
            "checkpoint_changed": False,
            "threshold_scan": False,
            "seed_scan": False,
        },
        "provenance": {
            "interaction_source": "simulated_from_gt",
            "real_human_tape": False,
            "real_sam3_full_loop": False,
            "not_real_human_evidence": True,
            "production_authorized": False,
        },
    }


def main() -> None:
    # Check before creating the directory, and never count N69 itself.
    latest = max_experiment_number({69})
    preexisting = OUT.exists() and any(OUT.iterdir())
    if latest != 68:
        raise RuntimeError(f"expected N68 to be the latest completed experiment, found N{latest}")

    git_project = command(["git", "-C", str(ROOT), "status", "--short"])
    git_parent = command(["git", "-C", str(ROOT.parent / "InterMOT"), "status", "--short"])
    code_inventory = {
        "sam3_intermot": tree_inventory("sam3_intermot", {".py"}),
        "third_party_sam3": tree_inventory("third_party/sam3", {".py", ".toml", ".yaml", ".yml"}),
        "configs": tree_inventory("configs", None),
        "environment": tree_inventory("environment", None),
        "n67_n68_scripts": {
            "files": [
                file_record(path.relative_to(ROOT).as_posix())
                for path in sorted((ROOT / "scripts").glob("n6[789]*.py"))
            ]
        },
    }
    audit = {
        "schema": "N69_STAGE_00_READONLY_AUDIT_V1",
        "created_at_utc": now(),
        "project_root": str(ROOT),
        "latest_existing_experiment_number": latest,
        "n69_output_preexisting": preexisting,
        "frozen_inputs": [file_record(item) for item in FROZEN_INPUTS],
        "protected_files": [file_record(item) for item in PROTECTED_FILES],
        "code_inventory": code_inventory,
        "resources": {"gpu": gpu_audit(), "disk": disk_audit()},
        "git": {"project": git_project, "adjacent_InterMOT": git_parent},
        "runtime": {"python": sys.version, "platform": platform.platform(), "cwd": os.getcwd()},
        "isolation": {
            "new_experiment": "N69",
            "new_evidence_root": str(OUT),
            "large_cache_root": "/path/to/cache/SAM3_InterMOT_n69",
            "production_paths_modified_by_stage00": False,
            "third_party_sam3_modified_by_stage00": False,
            "n36_to_n68_evidence_modified_by_stage00": False,
            "mot_ovmot_locatemot_masa_modified_by_stage00": False,
        },
    }
    protocol = protocol_payload()
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(AUDIT, audit)
    atomic_json(PROTOCOL, protocol)
    status = {
        "schema": "N69_STAGE_00_STATUS_V1",
        "status": "PASS_READONLY_AUDIT_PROTOCOL_FROZEN",
        "audit": str(AUDIT),
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "latest_existing_experiment_number": latest,
        "selected_experiment": "N69",
        "frozen_input_count": len(FROZEN_INPUTS),
        "gpu_count": len(audit["resources"]["gpu"].get("gpus", [])),
        "isolation": audit["isolation"],
        "provenance": protocol["provenance"],
        "next_action": "Run N69 Stage 01 versioned mapping diagnosis and fixture gate; do not modify N68 or production paths.",
    }
    atomic_json(STATUS, status)
    print(json.dumps({"status": status["status"], "audit": str(AUDIT), "protocol": str(PROTOCOL), "latest": latest}, sort_keys=True))


if __name__ == "__main__":
    main()
