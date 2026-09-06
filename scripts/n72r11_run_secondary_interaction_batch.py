#!/usr/bin/env python3
"""Run the frozen N72R11 secondary-interaction schedule with bounded workers.

Each child owns one event and one physical GPU.  The supervisor never reuses a
SAM3 object between events and records every completion/failure in an atomic
manifest.  Existing completed event directories are skipped only after their
``done.json`` is read; no event artifact is overwritten.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "outputs/N72R11/secondary_event_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/N72R11/secondary_interactions_attempt_01"
DEFAULT_GPUS: tuple[int, ...] = ()
MAX_N72R11_GPUS = 4


@dataclass(frozen=True)
class GpuSnapshot:
    """Physical GPU state sampled immediately before a child launch."""

    gpu_id: int
    uuid: str | None
    total_mib: int
    used_mib: int
    free_mib: int
    utilization_percent: int
    external_compute_process_count: int
    external_compute_used_mib: int
    idle: bool


_compute_process_probe_supported = True


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_smi_int(value: str, *, default: int = 0) -> int:
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return int(default)
    try:
        return int(float(text))
    except ValueError:
        return int(default)


def query_gpu_snapshots(candidate_gpu_ids: Sequence[int]) -> list[GpuSnapshot]:
    """Read physical GPU and compute-app state from the real nvidia-smi CLI."""

    global _compute_process_probe_supported
    requested = {int(value) for value in candidate_gpu_ids}
    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    physical: dict[int, dict[str, Any]] = {}
    for line in gpu_result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 6:
            raise RuntimeError(f"nvidia-smi GPU row has unexpected schema: {line!r}")
        gpu_id = _parse_smi_int(fields[0], default=-1)
        if gpu_id < 0 or (requested and gpu_id not in requested):
            continue
        physical[gpu_id] = {
            "uuid": None if fields[1].upper() in {"N/A", "[N/A]"} else fields[1],
            "total_mib": _parse_smi_int(fields[2]),
            "used_mib": _parse_smi_int(fields[3]),
            "free_mib": _parse_smi_int(fields[4]),
            "utilization_percent": _parse_smi_int(fields[5]),
        }
    if not physical:
        scope = "all physical GPUs" if not requested else sorted(requested)
        raise RuntimeError(f"nvidia-smi returned no candidate GPUs for {scope}")

    apps_by_uuid: dict[str, list[int]] = {}
    try:
        app_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        # Some drivers expose GPU telemetry but not the compute-app query.  Do
        # not fail the batch; the manifest records the degraded probe.
        _compute_process_probe_supported = False
    else:
        _compute_process_probe_supported = True
        for line in app_result.stdout.splitlines():
            if not line.strip() or "no running processes" in line.lower():
                continue
            fields = [part.strip() for part in line.split(",")]
            if len(fields) < 3:
                continue
            uuid = fields[0]
            if uuid not in {value.get("uuid") for value in physical.values()}:
                continue
            apps_by_uuid.setdefault(uuid, []).append(_parse_smi_int(fields[2]))

    snapshots: list[GpuSnapshot] = []
    for gpu_id in sorted(physical):
        value = physical[gpu_id]
        used_apps = apps_by_uuid.get(str(value["uuid"]), [])
        snapshots.append(
            GpuSnapshot(
                gpu_id=int(gpu_id),
                uuid=value["uuid"],
                total_mib=int(value["total_mib"]),
                used_mib=int(value["used_mib"]),
                free_mib=int(value["free_mib"]),
                utilization_percent=int(value["utilization_percent"]),
                external_compute_process_count=len(used_apps),
                external_compute_used_mib=sum(used_apps),
                idle=bool(_compute_process_probe_supported and not used_apps),
            )
        )
    return snapshots


def choose_gpu(
    snapshots: Sequence[GpuSnapshot],
    *,
    n72r11_active_gpu_ids: Sequence[int],
) -> GpuSnapshot:
    """Choose an unoccupied-by-this-batch physical GPU by live telemetry."""

    active = {int(value) for value in n72r11_active_gpu_ids}
    candidates = [snapshot for snapshot in snapshots if snapshot.gpu_id not in active]
    if not candidates:
        raise RuntimeError("no physical GPU remains outside active N72R11 child slots")
    idle = [snapshot for snapshot in candidates if snapshot.idle]
    pool = idle if idle else candidates
    return sorted(
        pool,
        key=lambda snapshot: (
            -int(snapshot.free_mib),
            int(snapshot.utilization_percent),
            int(snapshot.external_compute_used_mib),
            int(snapshot.gpu_id),
        ),
    )[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
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
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_schedule(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("secondary event manifest must be an object")
    events = [dict(item) for item in payload.get("all_decisions", []) if item.get("status") == "ELIGIBLE"]
    events.sort(key=lambda item: (str(item.get("sequence")), int(item.get("secondary_frame", -1)), str(item.get("event_id"))))
    ids = [str(item.get("event_id")) for item in events]
    if len(ids) != len(set(ids)):
        raise RuntimeError("secondary schedule contains duplicate eligible event IDs")
    for item in events:
        if item.get("runtime_future_gt_used") is not False or item.get("interaction_source") != "simulated_from_gt":
            raise RuntimeError(f"invalid frozen schedule provenance: {item.get('event_id')}")
    return payload, events


def done_status(event_dir: Path) -> tuple[str, dict[str, Any] | None]:
    path = event_dir / "done.json"
    if not path.is_file():
        return "NOT_RUN", None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"INVALID_DONE:{type(exc).__name__}", None
    if not isinstance(value, dict):
        return "INVALID_DONE:non_object", None
    if value.get("status") == "PASS_N72R11_SECONDARY_INTERACTION" and value.get("runtime_future_gt_used") is False:
        return "SKIPPED_EXISTING_PASS", value
    return "EXISTING_NONPASS", value


def child_command(event_id: str, *, output_root: Path, attempt: int, device: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(ROOT / "scripts/n72r11_generate_secondary_interaction.py"),
        "--event-id",
        event_id,
        "--attempt",
        str(int(attempt)),
        "--device",
        device,
        "--output-root",
        str(output_root),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, default=SCHEDULE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="comma-separated physical GPU IDs; omit to discover all physical GPUs",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--event-id", action="append", default=None)
    parser.add_argument("--event-id-file", type=Path, default=None)
    args = parser.parse_args()
    schedule_path = args.schedule if args.schedule.is_absolute() else ROOT / args.schedule
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    if args.gpu_ids is None or not str(args.gpu_ids).strip():
        discovered = query_gpu_snapshots(DEFAULT_GPUS)
        gpu_ids = tuple(snapshot.gpu_id for snapshot in discovered)
    else:
        gpu_ids = tuple(int(value.strip()) for value in str(args.gpu_ids).split(",") if value.strip())
    if not gpu_ids:
        raise SystemExit("no candidate physical GPUs were discovered")
    if int(args.max_workers) < 1 or int(args.max_workers) > min(MAX_N72R11_GPUS, len(gpu_ids)):
        raise SystemExit(
            f"max-workers must be in 1..{min(MAX_N72R11_GPUS, len(gpu_ids))}"
        )
    if int(args.attempt) < 1 or int(args.start_index) < 0:
        raise SystemExit("attempt/start-index must be non-negative and attempt must be positive")
    payload, events = read_schedule(schedule_path)
    if args.event_id is not None and args.event_id_file is not None:
        raise SystemExit("use only one of --event-id and --event-id-file")
    requested_ids: list[str] | None = None
    if args.event_id is not None:
        requested_ids = [str(value) for value in args.event_id]
    elif args.event_id_file is not None:
        event_id_file = args.event_id_file if args.event_id_file.is_absolute() else ROOT / args.event_id_file
        value = json.loads(event_id_file.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("event_ids")
        if not isinstance(value, list):
            raise SystemExit("event-id-file must contain a JSON list or an object with event_ids")
        requested_ids = [str(item) for item in value]
    if requested_ids is not None:
        if not requested_ids or len(requested_ids) != len(set(requested_ids)):
            raise SystemExit("event-id selection must be non-empty and unique")
        event_by_id = {str(item["event_id"]): item for item in events}
        missing_ids = sorted(set(requested_ids) - set(event_by_id))
        if missing_ids:
            raise SystemExit(f"event-id selection is not in frozen schedule: {missing_ids}")
        # Preserve the frozen schedule order even if a retry file was written
        # by a different scanner ordering.
        selected = [item for item in events if str(item["event_id"]) in set(requested_ids)]
        start = 0
        selection_source = "explicit_event_ids"
    else:
        start = int(args.start_index)
        selected = events[start:]
        if args.limit is not None:
            selected = selected[: int(args.limit)]
        selection_source = "schedule_slice"
    if not selected:
        raise SystemExit("selected schedule is empty")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"batch_manifest_attempt_{int(args.attempt):02d}.json"
    log_root = output_root / "logs"
    records: dict[str, dict[str, Any]] = {}
    for event in selected:
        event_id = str(event["event_id"])
        state, done = done_status(output_root / event_id)
        records[event_id] = {
            "event_id": event_id,
            "sequence": str(event["sequence"]),
            "secondary_frame": int(event["secondary_frame"]),
            "action_type": str(event["action_type"]),
            "status": state,
            "attempt": int(args.attempt),
            "gpu_id": None,
            "gpu_snapshot_at_launch": None,
            "gpu_selection_reason": None,
            "external_compute_process_count": None,
            "external_compute_used_mib": None,
            "free_mib_at_launch": None,
            "used_mib_at_launch": None,
            "utilization_at_launch": None,
            "returncode": None,
            "done_sha256": None if done is None else sha256_file(output_root / event_id / "done.json"),
            "log": None,
        }

    def write_manifest(status: str) -> None:
        counts: dict[str, int] = {}
        for value in records.values():
            key = str(value["status"])
            counts[key] = counts.get(key, 0) + 1
        atomic_json(
            manifest_path,
            {
                "schema_version": "N72R11_SECONDARY_BATCH_MANIFEST_V1",
                "status": status,
                "created_at_utc": now_utc(),
                "schedule": str(schedule_path),
                "schedule_sha256": sha256_file(schedule_path),
                "schedule_status": payload.get("status"),
                "output_root": str(output_root),
                "attempt": int(args.attempt),
                "selected_start_index": start,
                "selected_count": len(selected),
                "selection_source": selection_source,
                "selected_event_ids": [str(event["event_id"]) for event in selected],
                "max_workers": int(args.max_workers),
                "gpu_ids": list(gpu_ids),
                "compute_process_probe_supported": bool(_compute_process_probe_supported),
                "runtime_future_gt_used": False,
                "interaction_source": "simulated_from_gt",
                "records": [records[event_id] for event_id in sorted(records)],
                "counts": counts,
            },
        )

    write_manifest("RUNNING")
    pending = [event for event in selected if records[str(event["event_id"])] ["status"] == "NOT_RUN"]
    # Key the live-slot decision by physical GPU, not by the current number of
    # active processes.  The latter reuses the last GPU as soon as an earlier
    # slot finishes, which can put multiple SAM3 children on one card.
    active: dict[int, tuple[dict[str, Any], subprocess.Popen[bytes], Any, Path, int]] = {}
    next_index = 0
    try:
        while next_index < len(pending) or active:
            while next_index < len(pending) and len(active) < int(args.max_workers):
                event = pending[next_index]
                next_index += 1
                event_id = str(event["event_id"])
                used_gpu_ids = {value[4] for value in active.values()}
                snapshots = query_gpu_snapshots(gpu_ids)
                selected_snapshot = choose_gpu(
                    snapshots,
                    n72r11_active_gpu_ids=used_gpu_ids,
                )
                gpu_id = int(selected_snapshot.gpu_id)
                selection_reason = (
                    "idle_free_mib_descending_utilization_ascending"
                    if selected_snapshot.idle
                    else "no_idle_gpu_free_mib_descending_utilization_external_memory"
                )
                log_path = log_root / f"{event_id}.attempt{int(args.attempt)}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("wb")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                env["PYTHONUNBUFFERED"] = "1"
                command = child_command(event_id, output_root=output_root, attempt=int(args.attempt), device="cuda:0")
                process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log_handle, stderr=subprocess.STDOUT)
                records[event_id].update(
                    {
                        "status": "RUNNING",
                        "gpu_id": gpu_id,
                        "gpu_snapshot_at_launch": asdict(selected_snapshot),
                        "gpu_selection_reason": selection_reason,
                        "external_compute_process_count": selected_snapshot.external_compute_process_count,
                        "external_compute_used_mib": selected_snapshot.external_compute_used_mib,
                        "free_mib_at_launch": selected_snapshot.free_mib,
                        "used_mib_at_launch": selected_snapshot.used_mib,
                        "utilization_at_launch": selected_snapshot.utilization_percent,
                        "log": str(log_path),
                        "command": command,
                        "started_at_utc": now_utc(),
                        "started_epoch": time.time(),
                    }
                )
                active[process.pid] = (event, process, log_handle, log_path, gpu_id)
                write_manifest("RUNNING")
            if not active:
                continue
            time.sleep(1.0)
            finished: list[int] = []
            for pid, (event, process, log_handle, log_path, gpu_id) in active.items():
                returncode = process.poll()
                if returncode is None:
                    continue
                event_id = str(event["event_id"])
                log_handle.flush()
                log_handle.close()
                state, done = done_status(output_root / event_id)
                if returncode == 0 and state == "SKIPPED_EXISTING_PASS":
                    state = "PASS"
                elif returncode != 0:
                    state = "FAIL_CHILD" if state == "NOT_RUN" else f"FAIL_CHILD_{state}"
                records[event_id].update(
                    {
                        "status": state,
                        "returncode": int(returncode),
                        "done_sha256": None if done is None else sha256_file(output_root / event_id / "done.json"),
                        "finished_at_utc": now_utc(),
                        "finished_epoch": time.time(),
                    }
                )
                finished.append(pid)
                write_manifest("RUNNING")
            for pid in finished:
                del active[pid]
    except BaseException:
        write_manifest("INTERRUPTED_WITH_EVIDENCE")
        for _, process, _, _, _ in active.values():
            # Do not terminate children from a supervisor exception; they own
            # their atomic artifacts and are allowed to finish naturally.
            _ = process
        raise
    final_status = "PASS_ALL_SELECTED" if all(str(value["status"]) in {"PASS", "SKIPPED_EXISTING_PASS"} for value in records.values()) else "PARTIAL_WITH_FAILURES"
    write_manifest(final_status)
    print(json.dumps({"status": final_status, "manifest": str(manifest_path), "counts": {key: sum(1 for value in records.values() if value["status"] == key) for key in sorted({str(value["status"]) for value in records.values()})}}, sort_keys=True))
    return 0 if final_status == "PASS_ALL_SELECTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
