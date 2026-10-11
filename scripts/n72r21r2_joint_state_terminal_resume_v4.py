"""Resume a confirmed dead V3 owner; keep seven proven complete recoveries."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
from scripts import n72r21r2_joint_state_terminal_resume_v3 as original
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, GOAL, read_json, write_json, sha256, storage, utcnow, development_sequence

PREFIX = "on_policy/joint_state_terminal_resume_v4"
PROTOCOL = OUT / "on_policy/JOINT_STATE_TERMINAL_RESUME_PROTOCOL_V4.json"
SELF = "scripts/n72r21r2_joint_state_terminal_resume_v4.py"


def partition(registered, complete):
    if len(registered) != 19 or len(set(registered)) != 19 or len(set(complete)) != len(complete) or not set(complete) <= set(registered):
        raise ValueError("Exact original nineteen-video census and proven unique completion required")
    return [s for s in registered if s not in complete]


def completed_evidence(sequence):
    receipt_path = OUT / original.PREFIX / "recovered_sequences" / (sequence + ".json")
    receipt = read_json(receipt_path); refs = {}
    def checked(path):
        refs[str(path)] = sha256(path); return read_json(path)
    checked(receipt_path)
    assert receipt["protocol_sha256"] == sha256(original.PROTOCOL)
    for relative, digest in receipt["canonical_sha256"].items():
        canonical = checked(OUT / relative); assert refs[str(OUT / relative)] == digest
        raw_path = Path(canonical["successful_tensor_recovery_path"])
        raw = checked(raw_path); assert refs[str(raw_path)] == canonical["successful_tensor_recovery_sha256"]
        assert all(canonical[k] == v for k, v in raw.items())
        assert raw["terminal_resume_protocol_sha256"] == sha256(original.PROTOCOL)
        assert raw["terminal_resume_source_sha256"] == read_json(original.PROTOCOL)["source_sha256"]
        if "artifact" in raw:
            refs[raw["artifact"]["path"]] = sha256(raw["artifact"]["path"])
            assert refs[raw["artifact"]["path"]] == raw["artifact"]["sha256"]
        for row in raw.get("episodes", []):
            episode = checked(row["path"]); assert refs[row["path"]] == row["sha256"]
            refs[episode["artifact"]["path"]] = sha256(episode["artifact"]["path"])
            assert refs[episode["artifact"]["path"]] == episode["artifact"]["sha256"]
    return refs


def freeze():
    if PROTOCOL.exists(): raise FileExistsError("Preserve V4 preparation")
    previous = read_json(original.PROTOCOL); original.validate_frozen_sources(previous)
    marker_path = OUT / original.PREFIX / "driver.json"; marker = read_json(marker_path)
    assert marker["active"] and not marker["failed_retained"]
    for pid in (marker["pid"], marker["active"]["pid"]):
        assert type(pid) is int and pid > 0 and not Path("/proc", str(pid)).exists()
    assert not original.scoped_live_processes()
    storage(256 << 20)
    archive = write_json(PREFIX + "/old_driver_exact_archive.json", marker)
    assert sha256(archive) == sha256(marker_path)
    complete, refs = [], {}
    for sequence in previous["sequences"]:
        canonical = [OUT / "on_policy/joint_state_v1" / folder / (sequence + ".json") for folder in ("runtime_sequences", "supervision")]
        if all(path.exists() for path in canonical):
            refs.update(completed_evidence(sequence)); complete.append(sequence)
        elif any(path.exists() for path in canonical):
            raise ValueError("One-sided canonical installation needs explicit repair, never adoption")
    assert set(marker["completed"]) <= set(complete)
    remaining = partition(previous["sequences"], complete)
    write_json(str(PROTOCOL.relative_to(OUT)), {"stage": "N72R21R2", "goal": GOAL, "frozen": True,
        "source_sha256": {SELF: sha256(ROOT / SELF)}, "predecessor_protocol_sha256": sha256(original.PROTOCOL),
        "old_marker_archive_path": str(archive), "old_marker_sha256": sha256(archive),
        "actual_old_owner_exit_code_observed": False, "termination_cause": "UNKNOWN",
        "all_registered_sequences": previous["sequences"], "proven_completed_sequences": complete,
        "proven_completed_evidence_sha256": refs, "remaining_sequences_original_order": remaining,
        "only_change": "New owner and isolated V4 paths execute unchanged V3/V2 scientific runtime/labels/install. Skip only individually SHA-proven whole canonical recoveries, including0082 completed after last stale owner update. No partial click is adopted; old failures/partials/logs/markers unchanged.",
        "CPU_workers": 1, "CPU_threads": 1, "storage_floor_GiB": 60,
        "actual_model_generated_on_policy": False, "scientific_goal_complete": False, "next_stage_authorized": False})


def validate():
    p = read_json(PROTOCOL)
    assert p["frozen"] and p["goal"] == GOAL and p["source_sha256"] == {SELF: sha256(ROOT / SELF)}
    assert p["predecessor_protocol_sha256"] == sha256(original.PROTOCOL)
    original.validate_frozen_sources(read_json(original.PROTOCOL))
    assert sha256(p["old_marker_archive_path"]) == p["old_marker_sha256"]
    assert all(sha256(path) == digest for path, digest in p["proven_completed_evidence_sha256"].items())
    assert partition(p["all_registered_sequences"], p["proven_completed_sequences"]) == p["remaining_sequences_original_order"]
    return p


@contextmanager
def routed():
    old_prefix, old_assets, old_write = original.PREFIX, original.ATTEMPT_ASSETS, original.write_json
    def annotated(relative, value, **kwargs):
        return write_json(relative, {**value, "terminal_resume_v4_protocol_sha256": sha256(PROTOCOL),
            "terminal_resume_v4_source_sha256": sha256(ROOT / SELF)}, **kwargs)
    try:
        original.PREFIX, original.ATTEMPT_ASSETS, original.write_json = PREFIX, ASSETS / "joint_state_terminal_resume_v4", annotated
        yield
    finally:
        original.PREFIX, original.ATTEMPT_ASSETS, original.write_json = old_prefix, old_assets, old_write


def operation(sequence, action):
    p = validate(); development_sequence(sequence)
    if sequence not in p["remaining_sequences_original_order"] or action not in ("runtime", "label"):
        raise ValueError("Only original remaining registered operations")
    with routed(): original.operation(sequence, action)


def worker(sequence):
    p = validate()
    if sequence not in p["remaining_sequences_original_order"]: raise ValueError("Never rerun a proven complete source")
    for action in ("runtime", "label"):
        code = subprocess.call([str(PYTHON), "-u", "-m", "scripts.n72r21r2_joint_state_terminal_resume_v4", "operation", "--sequence", sequence, "--operation", action], cwd=ROOT)
        if code: raise RuntimeError("Retain failed V4 operation: " + action)
    with routed(): original.install(sequence)


def run():
    p = validate(); marker = PREFIX + "/driver.json"
    if (OUT / marker).exists() or original.scoped_live_processes(): raise FileExistsError("No duplicate source owner")
    state = {"stage": "N72R21R2", "goal": GOAL, "pid": os.getpid(), "status": "ACTIVE_UNCHANGED_REMAINING_REGISTERED_SOURCE_RECOVERIES",
        "protocol_sha256": sha256(PROTOCOL), "active": None, "completed": [], "failed_retained": [], "CPU_workers": 1,
        "previous_proven_complete": p["proven_completed_sequences"]}
    write_json(marker, state)
    for sequence in p["remaining_sequences_original_order"]:
        storage(256 << 20)
        log = ASSETS / "joint_state_terminal_resume_v4_logs" / (sequence + ".log"); log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as handle:
            child = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_joint_state_terminal_resume_v4", "worker", "--sequence", sequence], cwd=ROOT,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=handle, stderr=subprocess.STDOUT)
            state["active"] = {"sequence": sequence, "pid": child.pid, "log": str(log)}; write_json(marker, state, mutable=True)
            code = child.wait()
        if code: state["failed_retained"].append({"sequence": sequence, "actual_exit_code": code, "log": str(log)})
        else:
            assert (OUT / PREFIX / "recovered_sequences" / (sequence + ".json")).exists(); state["completed"].append(sequence)
        state["active"] = None; write_json(marker, state, mutable=True)
    state["status"] = "COMPLETE_REMAINING_REGISTERED_SOURCE_ATTEMPTS_NOT_SCIENTIFIC_CLOSURE"; write_json(marker, state, mutable=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("freeze", "run", "worker", "operation")); parser.add_argument("--sequence"); parser.add_argument("--operation", choices=("runtime", "label")); args = parser.parse_args()
    {"freeze": freeze, "run": run, "worker": lambda: worker(args.sequence), "operation": lambda: operation(args.sequence, args.operation)}[args.action]()
