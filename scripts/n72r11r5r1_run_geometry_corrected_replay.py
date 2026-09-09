#!/usr/bin/env python3
"""Run the N72R11R5R1 positive-geometry corrected replay.

The runner reuses only the sealed N72R11R4 source manifests/checkpoints and
launches one independent child process per frozen N72R9 event.  E0 is rebuilt
from the frozen source axes with the opt-in geometry policy in every child;
the historical E0 rows are never posthoc-filtered.  A child failure remains in
its own ``attempts`` directory and in the parent manifest.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "outputs/N72R9/protocol.json"
R4_E1A_METRICS = ROOT / "outputs/N72R11R4/formal_e1a_metrics.json"
R4_E1B_METRICS = ROOT / "outputs/N72R11R4/formal_e1b_metrics.json"
R4_E1A_MANIFEST = ROOT / "outputs/N72R11R4/formal_e1a_attempt_01/exact_v3_replay_manifest_attempt_01.json"
R4_E1B_MANIFEST = ROOT / "outputs/N72R11R4/formal_e1b_attempt_01/exact_v3_replay_manifest_attempt_01.json"
DEFAULT_E1A_OUTPUT = ROOT / "outputs/N72R11R5R1/formal_e1a"
DEFAULT_E1B_OUTPUT = ROOT / "outputs/N72R11R5R1/formal_e1b"
VARIANTS = {
    "e1a": ("E1A_EXACT_ONPOLICY_V3_LEGACY", "v3", R4_E1A_METRICS, R4_E1A_MANIFEST),
    "e1b": ("E1B_PCTIS_LEGACY", "pctis", R4_E1B_METRICS, R4_E1B_MANIFEST),
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def resolve(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def frozen_events() -> list[dict[str, Any]]:
    payload = read_json(PROTOCOL)
    events = [dict(item) for item in payload.get("source_event_selection", {}).get("events", [])]
    ids = [str(item.get("event_id")) for item in events]
    if len(events) != 32 or len(set(ids)) != 32:
        raise RuntimeError(f"expected 32 unique frozen N72R9 events, found {len(events)}")
    for item in events:
        if item.get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"frozen event has runtime future GT enabled: {item.get('event_id')}")
        if item.get("interaction_source") != "simulated_from_gt" or item.get("not_real_human_evidence") is not True:
            raise RuntimeError(f"unexpected interaction provenance: {item.get('event_id')}")
        if len(item.get("future_window", [])) != 2:
            raise RuntimeError(f"malformed frozen future window: {item.get('event_id')}")
    return sorted(events, key=lambda item: str(item["event_id"]))


def _event_index(events: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(item["event_id"]): item for item in events}


def verify_frozen_config(kind: str, events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    treatment, scorer_kind, metrics_path, manifest_path = VARIANTS[kind]
    metrics = read_json(metrics_path)
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PASS_ALL_SELECTED" or int(manifest.get("event_count", -1)) != 32:
        raise RuntimeError(f"frozen R4 manifest is incomplete: {manifest_path}")
    if metrics.get("source_manifest") and resolve(metrics["source_manifest"]) != manifest_path:
        raise RuntimeError(f"metrics/source manifest mismatch for {kind}")
    if metrics.get("source_manifest_sha256") and str(metrics["source_manifest_sha256"]) != sha256_file(manifest_path):
        raise RuntimeError(f"frozen R4 manifest hash mismatch for {kind}")
    checkpoint = resolve(metrics.get("model_checkpoint"))
    manifest_checkpoint = resolve(manifest.get("model_checkpoint"))
    if checkpoint != manifest_checkpoint:
        raise RuntimeError(f"frozen checkpoint path mismatch for {kind}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = sha256_file(checkpoint)
    expected_sha = str(metrics.get("model_checkpoint_sha256"))
    manifest_sha = str(manifest.get("model_checkpoint_sha256"))
    if checkpoint_sha != expected_sha or checkpoint_sha != manifest_sha:
        raise RuntimeError(f"frozen checkpoint hash mismatch for {kind}")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 32:
        raise RuntimeError(f"frozen R4 record count is not 32: {manifest_path}")
    expected_events = _event_index(events)
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError(f"non-object frozen R4 record: {manifest_path}")
        event_id = str(record.get("event_id"))
        if event_id in seen or event_id not in expected_events:
            raise RuntimeError(f"frozen R4 event key mismatch: {event_id}")
        seen.add(event_id)
        if record.get("status") != "PASS":
            raise RuntimeError(f"frozen R4 event is not PASS: {event_id}")
        done_path = resolve(record.get("done"))
        done = read_json(done_path)
        if done.get("status") != "PASS_N72R11_RUNTIME_AND_POSTHOC_EVENT":
            raise RuntimeError(f"frozen R4 done is not sealed PASS: {done_path}")
        if resolve(done.get("model_checkpoint")) != checkpoint or str(done.get("scorer_kind")) != scorer_kind:
            raise RuntimeError(f"frozen R4 done config mismatch: {done_path}")
        if sha256_file(done_path) != str(record.get("done_sha256")):
            raise RuntimeError(f"frozen R4 done hash mismatch: {done_path}")
    if seen != set(expected_events):
        raise RuntimeError(f"frozen R4 manifest does not cover all events for {kind}")
    return {
        "kind": kind,
        "treatment_variant": treatment,
        "scorer_kind": scorer_kind,
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "model_checkpoint": str(checkpoint),
        "model_checkpoint_sha256": checkpoint_sha,
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
    }


def child_command(
    event_id: str,
    *,
    output_root: Path,
    config: Mapping[str, Any],
    device: str,
    smoke: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/n72r11_on_demand_replay.py"),
        "--event-id",
        str(event_id),
        "--output-root",
        str(output_root),
        "--device",
        str(device),
        "--model-checkpoint",
        str(config["model_checkpoint"]),
        "--scorer-kind",
        str(config["scorer_kind"]),
        "--horizon",
        "100",
        "--variants",
        str(config["treatment_variant"]),
        "--require-positive-geometry",
        "--rebuild-baseline-geometry",
    ]
    if smoke:
        command.append("--smoke")
    return command


def run(kind: str, *, output_root: Path, manifest_path: Path, device: str, smoke: bool, selected_ids: Sequence[str] | None) -> int:
    events = frozen_events()
    config = verify_frozen_config(kind, events)
    wanted = None if not selected_ids else {str(value) for value in selected_ids}
    event_ids = {str(item["event_id"]) for item in events}
    if wanted is not None and not wanted.issubset(event_ids):
        raise RuntimeError(f"requested IDs are not frozen N72R9 events: {sorted(wanted - event_ids)}")
    selected = [item for item in events if wanted is None or str(item["event_id"]) in wanted]
    output_root.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {
        str(item["event_id"]): {
            "event_id": str(item["event_id"]),
            "sequence": str(item["sequence"]),
            "action_type": str(item["action_type"]),
            "status": "NOT_RUN",
            "attempt": 1,
            "returncode": None,
        }
        for item in selected
    }

    def write_manifest(status: str) -> None:
        counts: dict[str, int] = {}
        for record in records.values():
            counts[str(record["status"])] = counts.get(str(record["status"]), 0) + 1
        atomic_json(
            manifest_path,
            {
                "schema_version": "N72R11R5R1_GEOMETRY_CORRECTED_REPLAY_MANIFEST_V1",
                "status": status,
                "created_at_utc": now_utc(),
                "protocol": config["protocol"],
                "protocol_sha256": config["protocol_sha256"],
                "source_r4_metrics": config["metrics_path"],
                "source_r4_metrics_sha256": config["metrics_sha256"],
                "source_manifest": config["source_manifest"],
                "source_manifest_sha256": config["source_manifest_sha256"],
                "model_checkpoint": config["model_checkpoint"],
                "model_checkpoint_sha256": config["model_checkpoint_sha256"],
                "model_kind": config["scorer_kind"],
                "scorer_kind": config["scorer_kind"],
                "attempt": 1,
                "device": str(device),
                "horizon": 100,
                "variants": ["E0_BASELINE_B0", config["treatment_variant"]],
                "geometry_policy": "POSITIVE_AREA_REQUIRED_BEFORE_MODEL_AND_SOLVER",
                "require_positive_geometry": True,
                "baseline_regenerated_from_frozen_sources": True,
                "runtime_future_gt_used": False,
                "interaction_source": "simulated_from_gt",
                "not_real_human_evidence": True,
                "execution": "one independent child per event; serial GPU ownership",
                "event_count": len(selected),
                "records": [records[event_id] for event_id in sorted(records)],
                "counts": counts,
                "selected_event_ids": sorted(records),
                "smoke": bool(smoke),
            },
        )

    write_manifest("RUNNING")
    for event in selected:
        event_id = str(event["event_id"])
        event_dir = output_root / event_id
        if (event_dir / "done.json").exists():
            records[event_id].update({"status": "FAIL_EXISTING_ARTIFACT", "error": str(event_dir / "done.json")})
            write_manifest("RUNNING")
            continue
        log_path = output_root / "logs" / f"{event_id}.attempt01.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = child_command(event_id, output_root=output_root, config=config, device=device, smoke=smoke)
        records[event_id].update(
            {
                "status": "RUNNING",
                "log": str(log_path),
                "command": command,
                "started_at_utc": now_utc(),
            }
        )
        write_manifest("RUNNING")
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("wb") as log_handle:
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        done_path = event_dir / "done.json"
        records[event_id].update(
            {
                "status": "PASS" if completed.returncode == 0 and done_path.is_file() else "FAIL_CHILD",
                "returncode": int(completed.returncode),
                "done": str(done_path) if done_path.is_file() else None,
                "done_sha256": sha256_file(done_path) if done_path.is_file() else None,
                "finished_at_utc": now_utc(),
            }
        )
        write_manifest("RUNNING")
    final_status = "PASS_ALL_SELECTED" if all(record["status"] == "PASS" for record in records.values()) else "PARTIAL_WITH_FAILURES"
    write_manifest(final_status)
    counts = {key: sum(record["status"] == key for record in records.values()) for key in sorted({str(record["status"]) for record in records.values()})}
    print(json.dumps({"status": final_status, "manifest": str(manifest_path), "counts": counts}, sort_keys=True))
    return 0 if final_status == "PASS_ALL_SELECTED" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--event-id", action="append", default=None)
    args = parser.parse_args()
    default_root = DEFAULT_E1A_OUTPUT if args.kind == "e1a" else DEFAULT_E1B_OUTPUT
    output_root = default_root if args.output_root is None else (args.output_root if args.output_root.is_absolute() else ROOT / args.output_root)
    default_manifest = ROOT / "outputs/N72R11R5R1" / ("smoke_e1a_manifest.json" if args.kind == "e1a" and args.smoke else "smoke_e1b_manifest.json" if args.kind == "e1b" and args.smoke else f"formal_{args.kind}_manifest.json")
    manifest_path = default_manifest if args.manifest is None else (args.manifest if args.manifest.is_absolute() else ROOT / args.manifest)
    return run(args.kind, output_root=output_root, manifest_path=manifest_path, device=str(args.device), smoke=bool(args.smoke), selected_ids=args.event_id)


if __name__ == "__main__":
    raise SystemExit(main())
