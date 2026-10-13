"""Exclusive recovery of the MAIN runtime interrupted by the October13 reboot.

Preserve all sealed cells and partial attempts. Only the interrupted whole cell
uses a fresh artifact BASE; the original driver/order/models/protocol are reused.
"""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess

from scripts import n72r21r2_main_terminal_recovery_v1 as old
from scripts.n72r21r2_common import (
    ROOT, OUT, ASSETS, PYTHON, GOAL, read_json, write_json, sha256, storage,
)

policy, driver = old.policy, old.driver
PREFIX = "audit/main_terminal_recovery_v2"
PROTOCOL = OUT / "audit/MAIN_TERMINAL_RECOVERY_PROTOCOL_V2.json"
MARKER = policy.PREFIX + "/driver.json"
FRESH = PREFIX + "/owner.json"
INTERRUPTED = "MAIN__DUAL_IDENTITY_STATE__L3_HARM_AWARE__seed730103__P0_CLAIM80/dancetrack0052"
RECOVERY_BASE = ASSETS / "main_terminal_recovery_v2/main_policy_v1"
MODULE = "scripts.n72r21r2_main_terminal_recovery_v2"
CODE = (MODULE.replace(".", "/") + ".py",
        "scripts/n72r21r2_main_terminal_recovery_v1.py",
        "scripts/n72r21r2_idle_driver_recovery_v2.py")
MODULES = (MODULE, "scripts.n72r21r2_main_policy_driver",
           "scripts.n72r21r2_main_policy", "scripts.n72r21r2_main_terminal_recovery_v1",
           "scripts.n72r21r2_main_storage_resume_v2", "scripts.n72r21r2_main_storage_resume_v3")


def exclusive():
    old.no_live_scoped_workers(MODULES)


def evidence():
    exclusive()
    path = OUT / MARKER
    marker = read_json(path)
    old.dead_active(marker)
    assert marker["active"]["key"] == INTERRUPTED
    assert marker["active"]["action"] == "runtime"
    assert marker["protocol_sha256"] == sha256(policy.PROTOCOL)
    actual = sorted(str(p.relative_to(OUT / policy.PREFIX / "results")).removesuffix(".json")
                    for p in (OUT / policy.PREFIX / "results").rglob("*.json"))
    assert len(marker["completed"]) == len(set(marker["completed"])) == 694
    assert sorted(marker["completed"]) == actual
    ev = old.Evidence()
    for i, key in enumerate(actual, 1):
        old.checked_cell(key, ev)
        if i % 24 == 0:
            print({"independently_checked_sealed_cells": i, "total": len(actual)}, flush=True)
    for group in marker["group_status"]:
        ev.json(OUT / policy.PREFIX / "selections" / (group + ".json"))
    uid, sequence = INTERRUPTED.split("/")
    for folder in ("runtime", "onset_runtime", "results"):
        assert not (OUT / policy.PREFIX / folder / uid / (sequence + ".json")).exists()
    partials = [Path(marker["active"]["log"])]
    partials += list((policy.BASE / "traces" / uid).glob(sequence + "*.jsonl.zst"))
    partials += list((policy.BASE / "trackers").glob(uid + "__click*/data/" + sequence + ".txt"))
    assert len(partials) > 1
    return {"old_marker": marker, "old_marker_sha256": sha256(path),
            "completed_evidence_sha256": ev.refs,
            "preserved_interrupted_files_sha256": {str(p): sha256(p) for p in partials}}


def freeze():
    if PROTOCOL.exists() or (OUT / FRESH).exists() or RECOVERY_BASE.exists():
        raise FileExistsError("Never overwrite recovery or create a duplicate owner")
    storage(6 << 30)
    row = evidence()
    archive = write_json(PREFIX + "/original_marker.json", row["old_marker"])
    assert sha256(archive) == row["old_marker_sha256"]
    write_json(str(PROTOCOL.relative_to(OUT)), {
        "stage": "N72R21R2", "goal": GOAL, "frozen": True,
        "source_sha256": {n: sha256(ROOT / n) for n in CODE},
        "original_policy_protocol_sha256": sha256(policy.PROTOCOL),
        "original_marker_archive": str(archive), **row,
        "interrupted_key": INTERRUPTED, "recovery_BASE": str(RECOVERY_BASE),
        "boot_id_observed": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "boot_time_observed": subprocess.check_output(["uptime", "-s"], text=True).strip(),
        "actual_old_exit_code_observed": False, "termination_cause": "UNKNOWN_REBOOT_OBSERVED",
        "scientific_policy_seed_point_order_weights_changed": False,
        "unsealed_click_adoption": False, "confirmation_authorized": False,
        "next_stage_authorized": False})
    print({"verified_sealed_MAIN_cells": 694, "protocol_sha256": sha256(PROTOCOL)}, flush=True)


def frozen():
    p = read_json(PROTOCOL)
    assert p["frozen"] and p["goal"] == GOAL and p["interrupted_key"] == INTERRUPTED
    assert p["source_sha256"] == {n: sha256(ROOT / n) for n in CODE}
    assert p["original_policy_protocol_sha256"] == sha256(policy.PROTOCOL)
    assert sha256(p["original_marker_archive"]) == p["old_marker_sha256"]
    assert all(sha256(path) == digest for path, digest in p["preserved_interrupted_files_sha256"].items())
    return p


@contextmanager
def recovery_paths():
    original = policy.BASE
    try:
        policy.BASE = RECOVERY_BASE
        yield
    finally:
        policy.BASE = original


def operation(action, sequence, family, objective, seed, point):
    frozen()
    key = "MAIN__" + family + "__" + objective + "__seed" + str(seed) + "__" + point + "/" + sequence
    if key != INTERRUPTED:
        raise ValueError("Only the registered interrupted cell routes BASE")
    with recovery_paths():
        if action == "evaluate":
            policy.evaluate(sequence, family, objective, seed, point)
        else:
            with old.runtime_file_guard():
                (policy.runtime if action == "runtime" else policy.onset_runtime)(
                    sequence, family, objective, seed, point)


def run():
    p = frozen()
    current = evidence()
    assert current == {k: p[k] for k in current}
    if (OUT / FRESH).exists():
        raise FileExistsError("Never duplicate recovery ownership")
    storage(6 << 30)
    saved = driver.OUT, driver.write_json, driver.run_cell, driver.ASSETS
    original_cell = driver.run_cell

    def mirrored(relative, value, **kwargs):
        if relative != MARKER:
            return write_json(relative, value, **kwargs)
        if not (OUT / FRESH).exists():
            exclusive()
            assert sha256(OUT / MARKER) == p["old_marker_sha256"]
            old.dead_active(read_json(OUT / MARKER))
            old.restored_state(value, p["old_marker"])
        value.update(terminal_recovery_v2_protocol_sha256=sha256(PROTOCOL),
                     original_marker_archive=p["original_marker_archive"], old_exit_code_not_observed=True)
        path = write_json(FRESH, value, **kwargs)
        write_json(MARKER, value, mutable=True)
        return path

    def recovered_cell(state, marker, job, point, sequence):
        uid = "MAIN__" + job["family"] + "__" + job["objective"] + "__seed" + str(job["seed"]) + "__" + point
        key = uid + "/" + sequence
        if key != INTERRUPTED:
            return original_cell(state, marker, job, point, sequence)
        if key in state["completed"] or any(r["key"] == key for r in state["failed_retained"]):
            return
        log = ASSETS / "main_terminal_recovery_v2/worker_logs" / uid / (sequence + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as handle:
            for action, folder in (("runtime", "runtime"), ("onsets", "onset_runtime"), ("evaluate", "results")):
                storage(32 << 20)
                if (OUT / policy.PREFIX / folder / uid / (sequence + ".json")).exists():
                    raise FileExistsError("Never adopt an unowned whole-cell seal")
                command = [str(PYTHON), "-u", "-m", MODULE, action,
                           "--sequence", sequence, "--family", job["family"],
                           "--objective", job["objective"], "--seed", str(job["seed"]), "--point", point]
                process = subprocess.Popen(command, cwd=ROOT,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                    stdout=handle, stderr=subprocess.STDOUT)
                state["active"] = {"key": key, "action": action, "pid": process.pid, "log": str(log)}
                mirrored(marker, state, mutable=True)
                code = process.wait()
                if code:
                    state["failed_retained"].append({"key": key, "action": action, "returncode": code, "log": str(log)})
                    break
            else:
                old.checked_cell(key, old.Evidence())
                state["completed"].append(key)
        state["active"] = None
        mirrored(marker, state, mutable=True)

    try:
        driver.OUT, driver.write_json, driver.run_cell, driver.ASSETS = (
            old.routed_output(MARKER, FRESH), mirrored, recovered_cell,
            ASSETS / "main_terminal_recovery_v2")
        driver.run()
    finally:
        driver.OUT, driver.write_json, driver.run_cell, driver.ASSETS = saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run", "runtime", "onsets", "evaluate"))
    for name in ("sequence", "family", "objective", "point"):
        parser.add_argument("--" + name)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.action in ("freeze", "run"):
        {"freeze": freeze, "run": run}[args.action]()
    else:
        operation(args.action, args.sequence, args.family, args.objective, args.seed, args.point)
