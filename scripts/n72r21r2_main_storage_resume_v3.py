"""Exclusive MAIN continuation after restored space, preserving all192 seals.

Original finite driver, model weights, sequences, points and scientific code
remain unchanged. Asset organization is only a verified same-inode path alias.
"""
import argparse
from scripts import n72r21r2_main_storage_resume_v2 as prior
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, GOAL, read_json, write_json, sha256, storage
from scripts.n72r21r2_storage_guard_recovery_v1 import dead_idle_storage_owner

PREFIX = "audit/main_storage_resume_v3"
PROTOCOL = OUT / "audit/MAIN_STORAGE_RESUME_PROTOCOL_V3.json"
READBACK = OUT / "audit/MAIN_ALL192_STORAGE_RESUME_ACTUAL_READBACK_V30.json"
MARKER = prior.MARKER
FRESH = PREFIX + "/owner.json"
OLD_LOG = ASSETS / "main_storage_resume_v2/driver.log"
OLD_OWNER = OUT / "audit/main_storage_resume_v2/owner.json"
SELF = "scripts/n72r21r2_main_storage_resume_v3.py"


def check_exclusive():
    prior.previous.no_live_scoped_workers((*prior.MODULES, "scripts.n72r21r2_main_storage_resume_v3"))


def checked_readback():
    check_exclusive()
    report = read_json(READBACK)
    assert report["goal"] == GOAL and not report["actual_new_RUNTIME_or_fit"]
    row = report["evidence"]
    marker = read_json(OUT / MARKER)
    dead_idle_storage_owner(marker, OLD_LOG)
    assert marker == row["old_marker"] and sha256(OUT / MARKER) == row["old_marker_sha256"]
    assert read_json(OLD_OWNER) == marker
    assert len(marker["completed"]) == len(set(marker["completed"])) == 192
    for section in ("completed_evidence_sha256", "preserved_logs_and_old_owner_sha256"):
        assert all(sha256(path) == digest for path, digest in row[section].items())
    assert prior.policy.BASE.samefile(report["original_logical_scientific_BASE"])
    assert str(prior.policy.BASE) == report["current_canonical_scientific_BASE"]
    return report


def freeze():
    if PROTOCOL.exists() or (OUT / FRESH).exists():
        raise FileExistsError("Never overwrite preparation or duplicate MAIN owner")
    storage(6 << 30)
    report = checked_readback()
    archived = write_json(PREFIX + "/old_marker_exact_archive.json", report["evidence"]["old_marker"])
    assert sha256(archived) == report["evidence"]["old_marker_sha256"]
    write_json(str(PROTOCOL.relative_to(OUT)), {
        "stage": "N72R21R2", "goal": GOAL, "frozen": True,
        "source_sha256": {SELF: sha256(ROOT / SELF)},
        "actual192_readback": {"path": str(READBACK), "sha256": sha256(READBACK)},
        "old_marker_archive": {"path": str(archived), "sha256": sha256(archived)},
        "old_owner": {"path": str(OLD_OWNER), "sha256": sha256(OLD_OWNER)},
        "original_policy_protocol_sha256": sha256(prior.policy.PROTOCOL),
        "only_change": "Exclusive new owner/log namespace; all192 checked cells and two completed groups hydrated into the original finite driver",
        "canonical_asset_path_alias_same_inode_only": True,
        "storage_floor_GiB": 60, "CPU_workers": 1, "CPU_threads": 1,
        "new_seed_threshold_sequence_fit_or_confirmation": False,
        "scientific_goal_complete": False, "next_stage_authorized": False})
    print({"verified_preserved_MAIN_cells": 192, "protocol_sha256": sha256(PROTOCOL)}, flush=True)


def run():
    p = read_json(PROTOCOL)
    assert p["frozen"] and p["goal"] == GOAL
    assert p["source_sha256"] == {SELF: sha256(ROOT / SELF)}
    for name in ("actual192_readback", "old_marker_archive", "old_owner"):
        assert sha256(p[name]["path"]) == p[name]["sha256"]
    assert sha256(prior.policy.PROTOCOL) == p["original_policy_protocol_sha256"]
    storage(6 << 30)
    report = checked_readback()
    if (OUT / FRESH).exists():
        raise FileExistsError("Never start a duplicate MAIN continuation")
    driver = prior.driver
    saved = driver.OUT, driver.write_json, driver.ASSETS

    def mirrored(relative, value, **kwargs):
        if relative != MARKER:
            return write_json(relative, value, **kwargs)
        if not (OUT / FRESH).exists():
            assert sha256(OUT / MARKER) == report["evidence"]["old_marker_sha256"]
            dead_idle_storage_owner(read_json(OUT / MARKER), OLD_LOG)
            prior.previous.restored_state(value, report["evidence"]["old_marker"])
        value.update(storage_resume_protocol_sha256=sha256(PROTOCOL),
                     old_marker_archive=p["old_marker_archive"], old_storage_stop_retained=True)
        path = write_json(FRESH, value, **kwargs)
        write_json(MARKER, value, mutable=True)
        return path

    try:
        driver.OUT = prior.previous.routed_output(MARKER, FRESH)
        driver.write_json, driver.ASSETS = mirrored, ASSETS / "main_storage_resume_v3"
        driver.run()
    finally:
        driver.OUT, driver.write_json, driver.ASSETS = saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run"))
    {"freeze": freeze, "run": run}[parser.parse_args().action]()
