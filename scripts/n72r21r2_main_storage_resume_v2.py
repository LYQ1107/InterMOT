"""Resume a proven storage-stopped MAIN owner, keeping all sealed cells.

Only owner/log namespaces change. Original driver, runtime, scientific BASE,
models, points, seeds, ordered splits and selection code stay frozen.
"""
import argparse
from scripts import n72r21r2_main_terminal_recovery_v1 as previous
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, GOAL, read_json, write_json, sha256, storage
from scripts.n72r21r2_storage_guard_recovery_v1 import dead_idle_storage_owner

policy, driver = previous.policy, previous.driver
PREFIX = "audit/main_storage_resume_v2"
PROTOCOL = OUT / "audit/MAIN_STORAGE_RESUME_PROTOCOL_V2.json"
MARKER = policy.PREFIX + "/driver.json"
FRESH = PREFIX + "/owner.json"
CODE = ("scripts/n72r21r2_main_storage_resume_v2.py",
        "scripts/n72r21r2_storage_guard_recovery_v1.py",
        "scripts/n72r21r2_main_terminal_recovery_v1.py",
        "scripts/n72r21r2_idle_driver_recovery_v2.py")
OLD_LOG = ASSETS / "main_terminal_recovery_v1_logs/driver.log"
MODULES = ("scripts.n72r21r2_main_policy_driver", "scripts.n72r21r2_main_policy",
           "scripts.n72r21r2_main_terminal_recovery_v1", "scripts.n72r21r2_main_storage_resume_v2")


def evidence():
    marker = read_json(OUT / MARKER)
    observation = dead_idle_storage_owner(marker, OLD_LOG)
    previous.no_live_scoped_workers(MODULES)
    assert not marker["failed_retained"]
    assert marker["protocol_sha256"] == sha256(policy.PROTOCOL)
    keys = marker["completed"]
    assert len(keys) == len(set(keys)) and keys
    # Guard occurred before the next cell log/child: no one-sided seal adoption.
    for folder in ("runtime", "onset_runtime", "results"):
        actual = {str(path.relative_to(OUT / policy.PREFIX / folder)).removesuffix(".json")
                  for path in (OUT / policy.PREFIX / folder).rglob("*.json")}
        assert actual == set(keys)
    ev = previous.Evidence()
    for key in keys:
        previous.checked_cell(key, ev)
    p = read_json(policy.PROTOCOL)
    for path in (OUT / policy.PREFIX / "selections").glob("*.json"):
        selection = ev.json(path)
        assert selection["protocol_sha256"] == ev.digest(policy.PROTOCOL)
        cells = {point: [] for point in p["points"]}
        for ref in selection["actual_INNER_result_refs"]:
            ev.artifact(ref)
            cell = ev.json(ref["path"])
            assert cell["sequence"] in p["split"]["inner"]
            assert cell["family"] == selection["family"] and cell["objective"] == selection["objective"]
            cells[cell["point"]].append(cell)
        chosen = policy.choose_inner_point(cells, p["seeds"], p["split"]["inner"])
        assert all(selection[k] == value for k, value in chosen.items())
    for group in marker["group_status"]:
        assert (OUT / policy.PREFIX / "selections" / (group + ".json")).exists()
    return {"old_marker": marker, "old_marker_sha256": sha256(OUT / MARKER),
            "completed_evidence_sha256": ev.refs, "storage_stop_observation": observation,
            "preserved_logs_and_old_owner_sha256": {
                str(OLD_LOG): sha256(OLD_LOG),
                str(OUT / previous.FRESH): sha256(OUT / previous.FRESH)}}


def freeze():
    if PROTOCOL.exists(): raise FileExistsError("Never overwrite recovery preparation")
    storage(256 << 20)
    row = evidence()
    archived = write_json(PREFIX + "/old_marker_exact_archive.json", row["old_marker"])
    assert sha256(archived) == row["old_marker_sha256"]
    assert not (OUT / FRESH).exists()
    write_json(str(PROTOCOL.relative_to(OUT)), {"stage": "N72R21R2", "goal": GOAL, "frozen": True,
        "source_sha256": {name: sha256(ROOT / name) for name in CODE},
        "original_policy_protocol_sha256": sha256(policy.PROTOCOL),
        "old_marker_archive": str(archived), **row,
        "only_change": "New owner and log namespace; original finite driver skips only fully verified completed cells. Existing INNER selection re-executes unchanged deterministic code and must retain identical sealed bytes.",
        "scientific_BASE_unchanged": str(policy.BASE), "storage_floor_GiB": 60,
        "CPU_workers": 1, "CPU_threads": 1, "new_fit_seed_threshold_or_sequence": False,
        "scientific_goal_complete": False, "confirmation_authorized": False, "next_stage_authorized": False})
    print({"proven_completed_MAIN_cells": len(row["old_marker"]["completed"]), "protocol_sha256": sha256(PROTOCOL)}, flush=True)


def validate():
    p = read_json(PROTOCOL)
    assert p["frozen"] and p["goal"] == GOAL
    assert p["source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
    assert p["original_policy_protocol_sha256"] == sha256(policy.PROTOCOL)
    previous.Evidence().code(read_json(policy.PROTOCOL)["source_sha256"])
    assert sha256(p["old_marker_archive"]) == p["old_marker_sha256"]
    assert str(policy.BASE) == p["scientific_BASE_unchanged"]
    assert all(sha256(path) == digest for path, digest in p["preserved_logs_and_old_owner_sha256"].items())
    return p


def run():
    p = validate()
    assert evidence() == {k: p[k] for k in evidence_keys()}
    if (OUT / FRESH).exists(): raise FileExistsError("Never duplicate resumed ownership")
    saved = driver.OUT, driver.write_json, driver.ASSETS
    def mirrored(relative, value, **kwargs):
        if relative != MARKER: return write_json(relative, value, **kwargs)
        if not (OUT / FRESH).exists():
            assert sha256(OUT / MARKER) == p["old_marker_sha256"]
            dead_idle_storage_owner(read_json(OUT / MARKER), OLD_LOG)
            previous.restored_state(value, p["old_marker"])
        value.update(storage_resume_protocol_sha256=sha256(PROTOCOL),
                     old_marker_archive=p["old_marker_archive"], old_storage_stop_retained=True)
        path = write_json(FRESH, value, **kwargs)
        write_json(MARKER, value, mutable=True)
        return path
    try:
        driver.OUT = previous.routed_output(MARKER, FRESH)
        driver.write_json, driver.ASSETS = mirrored, ASSETS / "main_storage_resume_v2"
        driver.run()
    finally:
        driver.OUT, driver.write_json, driver.ASSETS = saved


def evidence_keys():
    return ("old_marker", "old_marker_sha256", "completed_evidence_sha256",
            "storage_stop_observation", "preserved_logs_and_old_owner_sha256")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("freeze", "run")); args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.action]()
