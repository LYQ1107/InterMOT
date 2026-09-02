#!/usr/bin/env python3
"""Run the six frozen N71 structural-export windows for N72R1.

Each child process owns one SAM3 session and one frame range.  At most four
children are launched at once, one per explicitly selected GPU.  This stage
audits candidate provenance, same-run assignment joins, and mapping evidence;
it does not read GT or claim public-ID efficacy.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
N72R1_ROOT = Path("/data2/usr_for_deadline/SAM3_InterMOT_N72R1")
PLAN_PATH = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/SAM3_InterMOT/"
    "outputs/N71/candidate_branch/window_plan.json"
)
PYTHON = Path("/home/lwr/anaconda3/envs/intermot/bin/python")
RUNNER = ROOT / "scripts" / "n72r1_stage15_gpu_smoke.py"
GPU_IDS = (0, 1, 2, 3)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row is not an object")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_windows() -> list[dict[str, Any]]:
    payload = read_json(PLAN_PATH)
    windows = [dict(item) for item in payload.get("windows", [])]
    if len(windows) != 6:
        raise RuntimeError(f"N72R1 requires exactly six frozen windows, found {len(windows)}")
    required = {"window_id", "sequence", "frame_start", "frame_end", "runtime_future_gt_used"}
    for item in windows:
        missing = sorted(required - set(item))
        if missing:
            raise RuntimeError(f"frozen window missing fields {missing}: {item.get('window_id')}")
        if item.get("runtime_future_gt_used") is not False:
            raise RuntimeError(f"frozen window permits runtime GT: {item.get('window_id')}")
    return windows


def run_batch(windows: list[dict[str, Any]], batch_index: int, output_root: Path) -> list[dict[str, Any]]:
    children: list[tuple[dict[str, Any], int, subprocess.Popen[bytes], Any, Path]] = []
    for offset, window in enumerate(windows):
        gpu = GPU_IDS[offset]
        window_id = str(window["window_id"])
        window_root = output_root / "windows" / window_id
        window_root.mkdir(parents=True, exist_ok=True)
        status_path = window_root / "stage_15_status.json"
        log_path = window_root / "runner.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        command = [
            str(PYTHON),
            str(RUNNER),
            "--plan",
            str(PLAN_PATH),
            "--window-id",
            window_id,
            "--gpu",
            "0",
            "--output-root",
            str(window_root),
            "--status-path",
            str(status_path),
        ]
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        children.append((window, gpu, process, log_handle, log_path))

    results: list[dict[str, Any]] = []
    # wait() is deliberately used rather than polling.  The batch therefore
    # keeps one child per GPU and preserves each child's complete log.
    for window, gpu, process, log_handle, log_path in children:
        return_code = process.wait()
        log_handle.close()
        window_id = str(window["window_id"])
        window_root = output_root / "windows" / window_id
        done_path = window_root / "done.json"
        failure_path = window_root / "failure.json"
        result: dict[str, Any] = {
            "batch_index": batch_index,
            "gpu": gpu,
            "window_id": window_id,
            "sequence": str(window["sequence"]),
            "command_exit_code": int(return_code),
            "log_path": str(log_path),
            "output_root": str(window_root),
        }
        if return_code == 0 and done_path.is_file():
            result["status"] = "PASS"
            result["done"] = read_json(done_path)
        else:
            result["status"] = "FAIL"
            result["failure"] = read_json(failure_path) if failure_path.is_file() else {
                "error": "child exited without failure artifact",
                "command_exit_code": int(return_code),
            }
        atomic_json(window_root / "child_result.json", result)
        results.append(result)
    return results


def flatten_window_artifacts(results: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    candidate_rows: list[dict[str, Any]] = []
    legacy_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    sidecar_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    window_manifests: list[dict[str, Any]] = []
    audit_errors: list[dict[str, Any]] = []
    uid_counter: Counter[str] = Counter()
    frame_keys: Counter[tuple[str, int]] = Counter()
    sidecar_keys: Counter[tuple[str, int]] = Counter()
    public_absent = 0
    explicit_none = 0
    state_assignment_count = 0
    public_assignment_count = 0
    source_run_mismatch = 0
    axis_mismatch = 0
    expected_frame_count = 0

    for result in results:
        window_id = str(result["window_id"])
        window_root = Path(str(result["output_root"]))
        expected_frame_count += int(result.get("done", {}).get("frame_end", 0)) - int(result.get("done", {}).get("frame_start", 0)) + 1 if result["status"] == "PASS" else 0
        manifest = {
            "window_id": window_id,
            "sequence": result["sequence"],
            "status": result["status"],
            "command_exit_code": result["command_exit_code"],
            "output_root": str(window_root),
        }
        if result["status"] != "PASS":
            window_manifests.append(manifest)
            continue
        done = dict(result["done"])
        manifest.update({
            "frame_start": done.get("frame_start"),
            "frame_end": done.get("frame_end"),
            "frame_count": done.get("frame_count"),
            "candidate_row_count": done.get("candidate_row_count"),
            "legacy_v2_equivalence": done.get("legacy_v2_equivalence", {}),
            "runtime_future_gt_used": done.get("runtime_future_gt_used"),
            "public_mapping_status": done.get("public_mapping_status"),
        })
        try:
            candidates = read_jsonl(window_root / "candidate_v2.jsonl")
            legacy = read_jsonl(window_root / "legacy_candidates.jsonl")
            frames = read_jsonl(window_root / "candidate_frames.jsonl")
            sidecars = read_jsonl(window_root / "assignment_sidecar.jsonl")
            mappings = read_jsonl(window_root / "mapping_ledger.jsonl")
            equivalence = read_json(window_root / "equivalence_audit.json")
            authority = read_json(window_root / "public_authority_audit.json")
        except Exception as exc:
            audit_errors.append({"window_id": window_id, "code": "artifact_read_error", "error": repr(exc)})
            window_manifests.append(manifest)
            continue
        if equivalence.get("all_pass") is not True:
            audit_errors.append({"window_id": window_id, "code": "legacy_v2_equivalence_not_pass"})
        if authority.get("runtime_future_gt_used") is not False:
            audit_errors.append({"window_id": window_id, "code": "authority_runtime_gt_not_false"})
        for row in candidates:
            candidate_rows.append(row)
            uid = row.get("candidate_uid")
            if uid in (None, ""):
                axis_mismatch += 1
            else:
                uid_counter[str(uid)] += 1
            required = (
                "source_run_id", "session_id", "segment_id", "window_id", "chunk_id",
                "official_raw_sam_id", "adapter_external_id", "segment_local_id",
                "sequence_global_id", "candidate_uid", "candidate_uid_v2",
            )
            missing = [key for key in required if row.get(key) in (None, "")]
            if missing:
                audit_errors.append({"window_id": window_id, "frame_idx": row.get("frame_idx"), "code": "candidate_provenance_missing", "fields": missing})
            if row.get("runtime_future_gt_used") is not False:
                audit_errors.append({"window_id": window_id, "frame_idx": row.get("frame_idx"), "code": "candidate_runtime_gt_not_false"})
        for row in legacy:
            legacy_rows.append(row)
        for row in frames:
            frame_rows.append(row)
            frame_keys[(window_id, int(row["frame_idx"]))] += 1
            if row.get("runtime_future_gt_used") is not False:
                audit_errors.append({"window_id": window_id, "frame_idx": row.get("frame_idx"), "code": "frame_runtime_gt_not_false"})
        for row in sidecars:
            sidecar_rows.append(row)
            sidecar = dict(row.get("sidecar", {}))
            sidecar_keys[(window_id, int(row["frame_idx"]))] += 1
            if sidecar.get("runtime_future_gt_used") is not False or row.get("runtime_future_gt_used") is not False:
                audit_errors.append({"window_id": window_id, "frame_idx": row.get("frame_idx"), "code": "sidecar_runtime_gt_not_false"})
            public_absent += int(sum(1 for item in sidecar.get("public_assignment_rows", []) if item.get("public_assignment_status") == "PUBLIC_ASSIGNMENT_ARTIFACT_ABSENT"))
            explicit_none += int(sum(1 for item in sidecar.get("public_assignment_rows", []) if item.get("public_assignment_status") == "EXPLICIT_NONE"))
            state_assignment_count += int(sum(1 for item in sidecar.get("candidate_assignment_rows", []) if item.get("association_state_id") is not None))
            public_assignment_count += int(sum(1 for item in sidecar.get("candidate_assignment_rows", []) if item.get("public_id") is not None))
            source_run_mismatch += int(sidecar.get("source_run_mismatch_count", 0))
            if sidecar.get("integrity_status") != "EXACT":
                audit_errors.append({"window_id": window_id, "frame_idx": row.get("frame_idx"), "code": "sidecar_integrity_not_exact", "status": sidecar.get("integrity_status")})
        for row in mappings:
            mapping_rows.append(row)
        window_manifests.append(manifest)

    duplicate_uid_count = sum(max(0, count - 1) for count in uid_counter.values())
    duplicate_frame_count = sum(max(0, count - 1) for count in frame_keys.values())
    duplicate_sidecar_count = sum(max(0, count - 1) for count in sidecar_keys.values())
    missing_frame_count = 0
    for result in results:
        if result["status"] != "PASS":
            continue
        done = result["done"]
        expected = set(range(int(done["frame_start"]), int(done["frame_end"]) + 1))
        observed = {frame for (window_id, frame), count in frame_keys.items() if window_id == result["window_id"] and count > 0}
        missing_frame_count += len(expected - observed)

    atomic_jsonl(output_root / "candidate_v2.jsonl", candidate_rows)
    atomic_jsonl(output_root / "legacy_candidates.jsonl", legacy_rows)
    atomic_jsonl(output_root / "candidate_frames.jsonl", frame_rows)
    atomic_jsonl(output_root / "assignment_sidecar.jsonl", sidecar_rows)
    atomic_jsonl(output_root / "mapping_ledger.jsonl", mapping_rows)
    atomic_jsonl(output_root / "window_manifests.jsonl", window_manifests)

    integrity = {
        "schema_version": "N72R1_SIX_WINDOW_STRUCTURAL_INTEGRITY_V1",
        "window_count_expected": 6,
        "window_count_observed": len(results),
        "window_pass_count": sum(result["status"] == "PASS" for result in results),
        "window_fail_count": sum(result["status"] != "PASS" for result in results),
        "expected_frame_count_pass_windows": expected_frame_count,
        "observed_frame_record_count": len(frame_rows),
        "missing_frame_count": missing_frame_count,
        "duplicate_frame_count": duplicate_frame_count,
        "candidate_v2_row_count": len(candidate_rows),
        "legacy_candidate_row_count": sum(len(row.get("candidates", [])) for row in legacy_rows),
        "candidate_uid_collision_count": duplicate_uid_count,
        "raw_official_id_missing_count": sum(row.get("official_raw_sam_id") is None for row in candidate_rows),
        "adapter_id_missing_count": sum(row.get("adapter_external_id") is None for row in candidate_rows),
        "source_run_missing_count": sum(not row.get("source_run_id") for row in candidate_rows),
        "session_missing_count": sum(not row.get("session_id") for row in candidate_rows),
        "same_run_sidecar_frame_count": len(sidecar_rows),
        "same_run_join_coverage": (len(sidecar_rows) / len(frame_rows)) if frame_rows else 0.0,
        "association_state_assignment_count": state_assignment_count,
        "public_assignment_count": public_assignment_count,
        "final_public_mapping_coverage": (public_assignment_count / state_assignment_count) if state_assignment_count else 0.0,
        "explicit_none_count": explicit_none,
        "public_assignment_artifact_absent_count": public_absent,
        "source_run_mismatch_count": source_run_mismatch,
        "axis_mismatch_count": axis_mismatch,
        "candidate_absent_count": missing_frame_count,
        "target_candidate_absent_count": "NOT_COMPUTED_NO_RUNTIME_GT",
        "candidate_uid_unique": duplicate_uid_count == 0,
        "all_runtime_future_gt_false": not audit_errors and all(
            result.get("done", {}).get("runtime_future_gt_used") is False
            for result in results if result["status"] == "PASS"
        ),
        "public_authority_status": "BLOCKED_PUBLIC_AUTHORITY_NOT_IN_ACTIVE_RUNTIME",
        "legacy_export_compatible": not any(error.get("code") == "legacy_v2_equivalence_not_pass" for error in audit_errors),
        "cross_chunk_handover_status": "UNRESOLVED_SEPARATE_WINDOWS_NO_PUBLIC_BRIDGE",
        "audit_errors": audit_errors,
        "runtime_future_gt_used": False,
    }
    atomic_json(output_root / "integrity_audit.json", integrity)

    decomposition = {
        "schema_version": "N72R1_70_90_10_DECOMPOSITION_V1",
        "new_v2_structural_export": {
            "axis_mismatch": integrity["axis_mismatch_count"],
            "public_assignment_absent": integrity["public_assignment_artifact_absent_count"],
            "target_candidate_absent": "NOT_COMPUTED_NO_RUNTIME_GT",
        },
        "historical_n70_reference_read_only": {
            "axis_mismatch": 70,
            "target_candidate_absent": 90,
            "public_assignment_absent": 10,
            "note": "Historical N70 decomposition is reference-only; N72R1 does not rewrite it.",
        },
        "runtime_future_gt_used": False,
        "interpretation": "N72R1 structural export measures provenance/mapping availability; target absence cannot be inferred without runtime GT.",
    }
    atomic_json(output_root / "n70_70_90_10_decomposition.json", decomposition)
    return {"integrity": integrity, "decomposition": decomposition, "window_manifests": window_manifests}


def main() -> int:
    started = time.time()
    output_root = N72R1_ROOT / "six_window_export"
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        if not PLAN_PATH.is_file():
            raise FileNotFoundError(PLAN_PATH)
        if not PYTHON.is_file():
            raise FileNotFoundError(PYTHON)
        if not RUNNER.is_file():
            raise FileNotFoundError(RUNNER)
        windows = load_windows()
        results: list[dict[str, Any]] = []
        results.extend(run_batch(windows[:4], 0, output_root))
        results.extend(run_batch(windows[4:], 1, output_root))
        flattened = flatten_window_artifacts(results, output_root)
        integrity = flattened["integrity"]
        status = "PASS_SIX_WINDOW_STRUCTURAL_EXPORT_PUBLIC_MAPPING_BLOCKED"
        if integrity["window_fail_count"] or integrity["audit_errors"] or integrity["missing_frame_count"] or integrity["duplicate_frame_count"] or integrity["candidate_uid_collision_count"]:
            status = "BLOCKED_SIX_WINDOW_STRUCTURAL_INTEGRITY"
        payload = {
            "schema_version": "N72R1_STAGE_STATUS_V1",
            "stage": "N72R1-16",
            "status": status,
            "command": "scripts/n72r1_stage16_six_window_export.py",
            "plan_path": str(PLAN_PATH),
            "plan_sha256": sha256(PLAN_PATH),
            "gpu_ids": list(GPU_IDS),
            "process_isolation": "one_python_process_one_sam3_session_one_frame_range",
            "results": results,
            "integrity_audit": integrity,
            "decomposition_path": str(output_root / "n70_70_90_10_decomposition.json"),
            "research_efficacy": "NOT_RUN",
            "real_human_event_count": 0,
            "runtime_future_gt_used": False,
            "public_id_inferred": False,
            "next_minimum_action": "Acquire external real-human event tape and add an explicit same-run public authority resolver before any public-ID efficacy evaluation.",
            "elapsed_sec": time.time() - started,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(N72R1_ROOT / "status" / "stage_16_status.json", payload)
        print(json.dumps({"status": status, "windows": integrity["window_pass_count"], "frames": integrity["observed_frame_record_count"], "candidate_rows": integrity["candidate_v2_row_count"]}, sort_keys=True))
        return 0 if status.startswith("PASS_") else 1
    except Exception as exc:
        failure = {
            "schema_version": "N72R1_STAGE_STATUS_V1",
            "stage": "N72R1-16",
            "status": "BLOCKED_SIX_WINDOW_STRUCTURAL_EXPORT",
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_future_gt_used": False,
            "real_human_event_count": 0,
            "traceback": __import__("traceback").format_exc(),
            "next_minimum_action": "Preserve child artifacts and repair only the first actionable structural-export error.",
        }
        atomic_json(output_root / "failure.json", failure)
        atomic_json(N72R1_ROOT / "status" / "stage_16_status.json", failure)
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
