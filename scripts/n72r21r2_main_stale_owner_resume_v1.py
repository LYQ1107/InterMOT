"""Resume the currently stale MAIN owner without adopting its partial cell.

The frozen MAIN protocol and driver order remain unchanged. The old marker is
archived, its completed list is hydrated only after exact output-list checks,
and the single active cell is replayed in a fresh NAS recovery namespace.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from scripts import n72r21r2_main_policy as policy
from scripts import n72r21r2_main_policy_driver as driver
from scripts import n72r21r2_main_terminal_recovery_v1 as legacy
from scripts.n72r21r2_common import (
    ASSETS,
    OUT,
    PYTHON,
    ROOT,
    STORAGE_ROOT,
    GOAL,
    read_json,
    sha256,
    storage,
    utcnow,
    write_json,
)
from scripts.n72r21r2_idle_driver_recovery_v2 import routed_output


PREFIX = "audit/main_stale_owner_resume_v1"
PROTOCOL = OUT / (PREFIX + "/PROTOCOL.json")
FRESH = PREFIX + "/owner.json"
MARKER = policy.PREFIX + "/driver.json"
RECOVERY_BASE = STORAGE_ROOT / "main_stale_owner_resume_v1/main_policy_v1"
RECOVERY_LOGS = STORAGE_ROOT / "main_stale_owner_resume_v1_logs"
MODULE = "scripts.n72r21r2_main_stale_owner_resume_v1"
MODULES = (
    MODULE,
    "scripts.n72r21r2_main_policy_driver",
    "scripts.n72r21r2_main_policy",
    "scripts.n72r21r2_main_terminal_recovery_v1",
)


def current_marker():
    marker_path = OUT / MARKER
    marker = read_json(marker_path)
    active = marker.get("active")
    assert marker["stage"] == "N72R21R2" and marker["goal"] == GOAL
    assert marker["status"] == "ACTIVE_FINITE_REGISTERED_MAIN_OWN_POLICY_FULL_MOT"
    assert isinstance(active, dict) and active.get("action") == "runtime"
    legacy.dead_active(marker)
    legacy.no_live_scoped_workers(MODULES)
    assert marker["failed_retained"] == []
    assert marker["confirmation_authorized"] is False
    assert marker["protocol_sha256"] == sha256(policy.PROTOCOL)
    return marker_path, marker


def output_keys():
    return sorted(
        str(path.relative_to(OUT / policy.PREFIX / "results")).removesuffix(".json")
        for path in (OUT / policy.PREFIX / "results").rglob("*.json")
    )


def partial_receipts(marker):
    active = marker["active"]
    uid, sequence = active["key"].split("/")
    paths = [Path(active["log"])]
    paths += list((policy.BASE / "traces" / uid).glob(sequence + "*.jsonl.zst"))
    paths += list((policy.BASE / "trackers").glob(uid + "__click*/data/" + sequence + ".txt"))
    return {
        str(path): {"exists": path.exists(), "sha256": sha256(path) if path.is_file() else None}
        for path in paths
    }


def freeze():
    if PROTOCOL.exists() or (OUT / FRESH).exists():
        raise FileExistsError("Never overwrite the stale-owner recovery protocol")
    storage(6 << 30)
    marker_path, marker = current_marker()
    actual = output_keys()
    assert sorted(marker["completed"]) == actual
    active_key = marker["active"]["key"]
    assert active_key not in actual
    uid, sequence = active_key.split("/")
    for folder in ("runtime", "onset_runtime", "results"):
        assert not (OUT / policy.PREFIX / folder / uid / (sequence + ".json")).exists()
    archived = write_json(PREFIX + "/original_marker.json", marker)
    assert sha256(archived) == sha256(marker_path)
    value = {
        "stage": "N72R21R2",
        "goal": GOAL,
        "frozen": True,
        "utc": utcnow(),
        "marker": str(marker_path),
        "old_marker_sha256": sha256(marker_path),
        "original_marker_archive": str(archived),
        "protocol_sha256": sha256(policy.PROTOCOL),
        "active_key": active_key,
        "active_action": marker["active"]["action"],
        "completed_count": len(actual),
        "completed_output_keys_exact": True,
        "preserved_active_partial_receipts": partial_receipts(marker),
        "recovery_base": str(RECOVERY_BASE),
        "recovery_logs": str(RECOVERY_LOGS),
        "storage": storage(6 << 30),
        "termination_cause": "UNKNOWN_OWNER_ABSENT",
        "old_exit_code_observed": False,
        "unsealed_partial_adopted": False,
        "scientific_protocol_changed": False,
        "confirmation_authorized": False,
        "next_stage_authorized": False,
    }
    write_json(str(PROTOCOL.relative_to(OUT)), value)
    print(json.dumps({"protocol": str(PROTOCOL), "active_key": active_key, "completed": len(actual)}, indent=2))


def frozen():
    value = read_json(PROTOCOL)
    assert value["frozen"] and value["goal"] == GOAL
    assert value["protocol_sha256"] == sha256(policy.PROTOCOL)
    assert sha256(value["original_marker_archive"]) == value["old_marker_sha256"]
    return value


def operation(action, sequence, family, objective, seed, point):
    p = frozen()
    key = "MAIN__" + family + "__" + objective + "__seed" + str(seed) + "__" + point + "/" + sequence
    assert key == p["active_key"], key
    original = policy.BASE
    try:
        policy.BASE = RECOVERY_BASE
        if action == "evaluate":
            policy.evaluate(sequence, family, objective, seed, point)
        else:
            with legacy.runtime_file_guard():
                worker = policy.runtime if action == "runtime" else policy.onset_runtime
                worker(sequence, family, objective, seed, point)
    finally:
        policy.BASE = original


def run():
    p = frozen()
    marker_path, marker = current_marker()
    assert sha256(marker_path) == p["old_marker_sha256"]
    assert output_keys() == sorted(marker["completed"])
    if (OUT / FRESH).exists():
        raise FileExistsError("Never start a duplicate stale-owner recovery")
    saved = driver.OUT, driver.write_json, driver.run_cell, driver.ASSETS

    def mirrored(relative, value, **kwargs):
        if relative != MARKER:
            return write_json(relative, value, **kwargs)
        if not (OUT / FRESH).exists():
            old_path, old_value = current_marker()
            assert sha256(old_path) == p["old_marker_sha256"]
            assert old_value["active"]["key"] == p["active_key"]
            value.update(completed=list(p["completed_output_keys"] if "completed_output_keys" in p else marker["completed"]))
        value.update(
            stale_owner_recovery_protocol_sha256=sha256(PROTOCOL),
            original_marker_archive=p["original_marker_archive"],
            old_exit_code_not_observed_cause_UNKNOWN=True,
        )
        path = write_json(FRESH, value, **kwargs)
        write_json(relative, value, mutable=True)
        return path

    original_cell = driver.run_cell

    def recovered_cell(state, marker_value, job, point, sequence):
        uid = "MAIN__" + job["family"] + "__" + job["objective"] + "__seed" + str(job["seed"]) + "__" + point
        key = uid + "/" + sequence
        if key != p["active_key"]:
            return original_cell(state, marker_value, job, point, sequence)
        if key in state["completed"] or any(row["key"] == key for row in state["failed_retained"]):
            return
        log = RECOVERY_LOGS / uid / (sequence + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as handle:
            for action, folder in (("runtime", "runtime"), ("onsets", "onset_runtime"), ("evaluate", "results")):
                storage(32 << 20)
                if (OUT / policy.PREFIX / folder / uid / (sequence + ".json")).exists():
                    raise FileExistsError("Never adopt an unowned whole-cell seal")
                command = [
                    str(PYTHON), "-u", "-m", MODULE, action,
                    "--sequence", sequence, "--family", job["family"],
                    "--objective", job["objective"], "--seed", str(job["seed"]), "--point", point,
                ]
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                state["active"] = {"key": key, "action": action, "pid": process.pid, "log": str(log)}
                mirrored(MARKER, state, mutable=True)
                code = process.wait()
                if code:
                    state["failed_retained"].append({"key": key, "action": action, "returncode": code, "log": str(log)})
                    break
            else:
                state["completed"].append(key)
        state["active"] = None
        mirrored(MARKER, state, mutable=True)

    try:
        driver.OUT = routed_output(MARKER, FRESH)
        driver.write_json = mirrored
        driver.run_cell = recovered_cell
        driver.ASSETS = RECOVERY_LOGS
        driver.run()
    finally:
        driver.OUT, driver.write_json, driver.run_cell, driver.ASSETS = saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run", "runtime", "onsets", "evaluate"))
    parser.add_argument("--sequence")
    parser.add_argument("--family")
    parser.add_argument("--objective")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--point")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "run":
        run()
    else:
        operation(args.action, args.sequence, args.family, args.objective, args.seed, args.point)
