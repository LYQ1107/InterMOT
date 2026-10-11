"""Resume only the two original storage-stopped controlled-state sources.

No new state curriculum, model-generated on-policy claim or scientific grid.
Old V4 owner, failed logs, all previous partials and completed sources remain.
"""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess

from scripts import n72r21r2_joint_state_terminal_resume_v4 as previous
from scripts.n72r21r2_main_terminal_recovery_v1 import no_live_scoped_workers
from scripts.n72r21r2_storage_guard_recovery_v1 import dead_idle_storage_owner, STORAGE_ERROR
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, GOAL, read_json, write_json, sha256, storage, development_sequence

original = previous.original
PREFIX = "on_policy/joint_state_storage_resume_v5"
PROTOCOL = OUT / "on_policy/JOINT_STATE_STORAGE_RESUME_PROTOCOL_V5.json"
SELF = "scripts/n72r21r2_joint_state_storage_resume_v5.py"
CODE = (SELF, "scripts/n72r21r2_storage_guard_recovery_v1.py", previous.SELF,
        "scripts/n72r21r2_main_terminal_recovery_v1.py")
OLD_MARKER = OUT / previous.PREFIX / "driver.json"
OLD_LOG = ASSETS / "joint_state_terminal_resume_v4_logs/driver.log"
MODULES = original.MODULES + ("scripts.n72r21r2_joint_state_terminal_resume_v4",
                              "scripts.n72r21r2_joint_state_storage_resume_v5")


def completed_evidence(sequence):
    paths = [OUT / prefix / "recovered_sequences" / (sequence + ".json")
             for prefix in ("on_policy/joint_state_terminal_resume_v3", previous.PREFIX)]
    found = [p for p in paths if p.exists()]
    if len(found) != 1: raise ValueError("Exactly one proven full source recovery required")
    receipt = read_json(found[0])
    saved = original.PREFIX
    try:
        original.PREFIX = str(found[0].parent.parent.relative_to(OUT))
        refs = previous.completed_evidence(sequence)
    finally:
        original.PREFIX = saved
    if found[0] == paths[1]:
        assert receipt["terminal_resume_v4_protocol_sha256"] == sha256(previous.PROTOCOL)
        assert receipt["terminal_resume_v4_source_sha256"] == sha256(ROOT / previous.SELF)
    return refs


def freeze():
    if PROTOCOL.exists(): raise FileExistsError("Preserve V5 preparation")
    previous.validate()
    marker = read_json(OLD_MARKER)
    observation = dead_idle_storage_owner(marker, OLD_LOG)
    no_live_scoped_workers(MODULES)
    assert marker["protocol_sha256"] == sha256(previous.PROTOCOL)
    # The actual child failure is observed; the old parent exit status is not.
    assert len(marker["failed_retained"]) == 1
    failure = marker["failed_retained"][0]
    assert failure["sequence"] == "dancetrack0052" and failure["actual_exit_code"] == 1
    text = Path(failure["log"]).read_text()
    assert STORAGE_ERROR in text and "Retain failed V4 operation: runtime" in text
    storage(256 << 20)
    registered = read_json(original.PROTOCOL)["sequences"]
    complete, refs = [], {}
    for sequence in registered:
        canonical = [OUT / "on_policy/joint_state_v1" / folder / (sequence + ".json")
                     for folder in ("runtime_sequences", "supervision")]
        if all(p.exists() for p in canonical):
            refs.update(completed_evidence(sequence)); complete.append(sequence)
        elif any(p.exists() for p in canonical):
            raise ValueError("One-sided installation must not be silently adopted")
    assert set(marker["completed"]) <= set(complete)
    remaining = previous.partition(registered, complete)
    assert remaining == ["dancetrack0052", "dancetrack0086"]
    for sequence in remaining:
        for folder in ("runtime_sequences", "supervision", "recovered_sequences"):
            assert not (OUT / previous.PREFIX / folder / (sequence + ".json")).exists()
    archived = write_json(PREFIX + "/old_marker_exact_archive.json", marker)
    assert sha256(archived) == sha256(OLD_MARKER)
    write_json(str(PROTOCOL.relative_to(OUT)), {"stage": "N72R21R2", "goal": GOAL, "frozen": True,
        "source_sha256": {name: sha256(ROOT / name) for name in CODE},
        "predecessor_protocol_sha256": sha256(previous.PROTOCOL),
        "unchanged_V3_scientific_resume_protocol_sha256": sha256(original.PROTOCOL),
        "old_marker_archive_path": str(archived), "old_marker_sha256": sha256(OLD_MARKER),
        "old_original_marker_path": str(OLD_MARKER), "storage_stop_observation": observation,
        "actual_old_failed_child_exit_code": 1, "actual_old_parent_exit_code": None,
        "preserved_failed_and_parent_logs_sha256": {str(OLD_LOG): sha256(OLD_LOG), failure["log"]: sha256(failure["log"])},
        "all_registered_sequences": registered, "proven_completed_sequences": complete,
        "proven_completed_evidence_sha256": refs, "remaining_sequences_original_order": remaining,
        "only_change": "Exclusive V5 owner and paths run unchanged V3/V2 scientific operations and install; skip only17 SHA-proven canonical recoveries, retaining the five original completed sources and all failures.",
        "CPU_workers": 1, "CPU_threads": 1, "storage_floor_GiB": 60,
        "actual_model_generated_on_policy": False, "scientific_goal_complete": False,
        "confirmation_authorized": False, "next_stage_authorized": False})
    print({"proven_prior_recoveries": len(complete), "remaining_original_sequences": remaining}, flush=True)


def validate():
    p = read_json(PROTOCOL)
    assert p["frozen"] and p["goal"] == GOAL
    assert p["source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
    assert p["predecessor_protocol_sha256"] == sha256(previous.PROTOCOL)
    assert p["unchanged_V3_scientific_resume_protocol_sha256"] == sha256(original.PROTOCOL)
    previous.validate()
    assert sha256(p["old_marker_archive_path"]) == sha256(OLD_MARKER) == p["old_marker_sha256"]
    for field in ("preserved_failed_and_parent_logs_sha256", "proven_completed_evidence_sha256"):
        assert all(sha256(path) == digest for path, digest in p[field].items())
    assert previous.partition(p["all_registered_sequences"], p["proven_completed_sequences"]) == p["remaining_sequences_original_order"]
    return p


@contextmanager
def routed():
    saved = original.PREFIX, original.ATTEMPT_ASSETS, original.write_json
    def annotated(relative, value, **kwargs):
        return write_json(relative, {**value, "storage_resume_v5_protocol_sha256": sha256(PROTOCOL),
            "storage_resume_v5_source_sha256": sha256(ROOT / SELF)}, **kwargs)
    try:
        original.PREFIX, original.ATTEMPT_ASSETS, original.write_json = PREFIX, ASSETS / "joint_state_storage_resume_v5", annotated
        yield
    finally:
        original.PREFIX, original.ATTEMPT_ASSETS, original.write_json = saved


def operation(sequence, action):
    p = validate(); development_sequence(sequence)
    if sequence not in p["remaining_sequences_original_order"] or action not in ("runtime", "label"):
        raise ValueError("Only original two remaining registered source operations")
    with routed(): original.operation(sequence, action)


def worker(sequence):
    p = validate()
    if sequence not in p["remaining_sequences_original_order"]: raise ValueError("Never rerun a proven source")
    for action in ("runtime", "label"):
        code = subprocess.call([str(PYTHON), "-u", "-m", "scripts.n72r21r2_joint_state_storage_resume_v5", "operation", "--sequence", sequence, "--operation", action], cwd=ROOT)
        if code: raise RuntimeError("Retain failed V5 operation: " + action)
    with routed(): original.install(sequence)


def run():
    p = validate(); marker = PREFIX + "/driver.json"
    no_live_scoped_workers(MODULES)
    dead_idle_storage_owner(read_json(OLD_MARKER), OLD_LOG)
    if (OUT / marker).exists(): raise FileExistsError("Never duplicate V5 source ownership")
    state = {"stage": "N72R21R2", "goal": GOAL, "pid": os.getpid(),
        "status": "ACTIVE_UNCHANGED_REMAINING_TWO_STORAGE_SOURCE_RECOVERIES", "protocol_sha256": sha256(PROTOCOL),
        "active": None, "completed": [], "failed_retained": [], "CPU_workers": 1,
        "previous_proven_complete": p["proven_completed_sequences"]}
    write_json(marker, state)
    for sequence in p["remaining_sequences_original_order"]:
        storage(256 << 20)
        log = ASSETS / "joint_state_storage_resume_v5_logs" / (sequence + ".log"); log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as handle:
            child = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_joint_state_storage_resume_v5", "worker", "--sequence", sequence], cwd=ROOT,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=handle, stderr=subprocess.STDOUT)
            state["active"] = {"sequence": sequence, "pid": child.pid, "log": str(log)}; write_json(marker, state, mutable=True)
            code = child.wait()
        if code: state["failed_retained"].append({"sequence": sequence, "actual_exit_code": code, "log": str(log)})
        else:
            assert (OUT / PREFIX / "recovered_sequences" / (sequence + ".json")).exists(); state["completed"].append(sequence)
        state["active"] = None; write_json(marker, state, mutable=True)
    state["status"] = "COMPLETE_REMAINING_SOURCE_ATTEMPTS_NOT_SCIENTIFIC_CLOSURE"; write_json(marker, state, mutable=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("freeze", "run", "worker", "operation")); parser.add_argument("--sequence"); parser.add_argument("--operation", choices=("runtime", "label")); args = parser.parse_args()
    {"freeze": freeze, "run": run, "worker": lambda: worker(args.sequence), "operation": lambda: operation(args.sequence, args.operation)}[args.action]()
