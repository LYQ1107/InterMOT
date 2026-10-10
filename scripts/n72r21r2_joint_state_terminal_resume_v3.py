"""Recover a dead V2 owner in new paths; scientific replay remains frozen.

No old active marker, completed click, partial, or log is modified/adopted.
The old process exit status/cause is UNKNOWN unless independently observed.
V3 repeats the same nineteen registered V1-failed videos, not a new subset.
"""
import argparse
import os
from pathlib import Path
import subprocess

from scripts import n72r21r2_joint_state_tensor_v2 as tensor
from scripts.n72r21r2_common import (
    ROOT, OUT, ASSETS, PYTHON, GOAL, read_json, write_json, sha256,
    preregistration, development_sequence, storage, utcnow, append_log,
)

PREFIX = "on_policy/joint_state_terminal_resume_v3"
PROTOCOL = OUT / "on_policy/JOINT_STATE_TERMINAL_RESUME_PROTOCOL_V3.json"
TERMINAL = OUT / "on_policy/JOINT_STATE_V2_DEAD_OWNER_OBSERVATION_V1.json"
OLD_MARKER = OUT / "on_policy/joint_state_tensor_v2/driver.json"
ORIGINAL_MARKER = OUT / "on_policy/joint_state_v1/driver.json"
ATTEMPT_ASSETS = ASSETS / "joint_state_terminal_resume_v3"
SELF = "scripts/n72r21r2_joint_state_terminal_resume_v3.py"
MODULES = (
    "scripts.n72r21r2_joint_state_tensor_recovery_driver_v2",
    "scripts.n72r21r2_joint_state_tensor_v2",
    "scripts.n72r21r2_joint_state_curriculum",
    "scripts.n72r21r2_joint_state_terminal_resume_v3",
)


def process(pid, proc_root=Path("/proc")):
    """Never treat another cwd, reused PID, or a zombie as a live owner."""
    result = {"pid": pid, "present": False, "live_scoped_state_process": False}
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return result
    path = proc_root / str(pid)
    try:
        argv = path.joinpath("cmdline").read_bytes().split(b"\0")
        result["present"] = True
        if not any(m.encode() in argv for m in MODULES):
            return result
        status = path.joinpath("status").read_text()
        cwd = path.joinpath("cwd").resolve(strict=True)
    except FileNotFoundError:
        return result
    # Permission errors are not evidence of process absence.
    result["present"] = True
    scoped = cwd == ROOT.resolve() and any(m.encode() in argv for m in MODULES)
    state = next(line for line in status.splitlines() if line.startswith("State:"))
    result["live_scoped_state_process"] = scoped and state.split()[1] not in ("Z", "X")
    if scoped:
        result.update(exact_owned_argv=[a.decode() for a in argv if a], state=state)
    return result


def scoped_live_processes():
    values = []
    for directory in Path("/proc").iterdir():
        if directory.name.isdigit() and int(directory.name) != os.getpid():
            value = process(int(directory.name))
            if value["live_scoped_state_process"]:
                values.append(value)
    return values


def selected_sequences(original, old, split):
    if original["status"] == "ACTIVE" or old["completed"] or old["failed_retained"]:
        raise ValueError("This recovery is only for a dead first-attempt V2 owner")
    if old["status"] != "ACTIVE_TERMINAL_FAILURES_ONLY_RECOVERY" or not old.get("active"):
        raise ValueError("Retain the exact interrupted owner marker, not a guessed attempt")
    failed = {row["sequence"] for row in original["failed_retained"]}
    selected = [s for s in split["fit"] + split["inner"] if s in failed]
    if selected != old["sequences"] or len(selected) != 19:
        raise ValueError("All nineteen unchanged registered failures are required")
    if old["active"]["sequence"] != selected[0]:
        raise ValueError("Only the observed first interrupted video is covered")
    return selected


def preserved_inventory():
    paths = [OLD_MARKER, ORIGINAL_MARKER]
    for root in (ASSETS / "joint_state_tensor_recovery_v2_logs",
                 ASSETS / "joint_state_tensor_recovery_v2",
                 OUT / "on_policy/joint_state_tensor_v2/runtime"):
        if root.exists():
            paths.extend(p for p in root.rglob("*") if p.is_file())
    return [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(set(paths))]


def validate_frozen_sources(p):
    v2 = read_json(tensor.PROTOCOL)
    assert p["unchanged_tensor_V2_protocol_sha256"] == sha256(tensor.PROTOCOL)
    assert p["original_source_protocol_sha256"] == sha256(tensor.original.PROTOCOL)
    assert v2["wrapper_source_sha256"] == {s: sha256(ROOT / s) for s in tensor.CODE}
    assert v2["original_source_sha256"] == {s: sha256(ROOT / s) for s in tensor.original.CODE}
    assert p["source_sha256"] == {SELF: sha256(ROOT / SELF)}
    assert p["old_marker_sha256"] == sha256(OLD_MARKER)
    assert p["dead_owner_observation_sha256"] == sha256(TERMINAL)
    for row in read_json(TERMINAL)["preserved_old_evidence"]:
        assert sha256(row["path"]) == row["sha256"]


def freeze():
    if PROTOCOL.exists() or TERMINAL.exists():
        raise FileExistsError("Preserve prior resume preparation; inspect, do not overwrite")
    original, old = read_json(ORIGINAL_MARKER), read_json(OLD_MARKER)
    selected = selected_sequences(original, old, preregistration()["split"])
    assert old["original_terminal_receipt_sha256"] == sha256(ORIGINAL_MARKER)
    assert old["protocol_sha256"] == sha256(tensor.PROTOCOL)
    if scoped_live_processes():
        raise ValueError("An exact state recovery worker is still live; no duplicate owner")
    observed = [process(old["pid"]), process(old["active"]["pid"])]
    if any(p["live_scoped_state_process"] for p in observed):
        raise ValueError("Never replace a live owner/child")
    for sequence in selected:
        for folder in ("runtime_sequences", "supervision"):
            if (OUT / "on_policy/joint_state_v1" / folder / (sequence + ".json")).exists():
                raise FileExistsError("No implicit adoption of a completed original recovery")
    old_logs = []
    for row in original["failed_retained"]:
        text = Path(row["log"]).read_text()
        assert "full_tracker_state_before_sha256" in text and "AssertionError" in text
        old_logs.append({"sequence": row["sequence"], "path": row["log"], "sha256": sha256(row["log"])})
    preserved_original = []
    for folder in ("runtime_sequences", "supervision"):
        preserved_original.extend({"path": str(p), "sha256": sha256(p)} for p in (OUT / "on_policy/joint_state_v1" / folder).glob("*.json"))
    write_json(str(TERMINAL.relative_to(OUT)), {
        "stage": "N72R21R2", "goal": GOAL, "utc": utcnow(),
        "status": "OBSERVED_DEAD_V2_OWNER_AND_CHILD_WITH_INTERRUPTED_EVIDENCE",
        "process_observations": observed, "old_marker_sha256": sha256(OLD_MARKER),
        "actual_exit_code": None, "actual_exit_code_observed": False,
        "termination_cause": "UNKNOWN_NOT_INFERRED_FROM_STALE_MARKER_OR_MISSING_PROCESS",
        "old_session_handle": 20385, "old_handle_readback": "UNKNOWN_PROCESS_ID_AFTER_CONTEXT_HANDOFF",
        "preserved_old_evidence": preserved_inventory(), "old_failure_logs": old_logs,
        "preserved_completed_original_receipts": preserved_original,
        "old_completed_click_NOT_adopted_as_complete_video": True,
        "scientific_success": False, "next_stage_authorized": False,
    })
    p = {"stage": "N72R21R2", "goal": GOAL, "goal_file": "outputs/N72R21R2/FINAL_GOAL.json", "frozen": True,
         "sequences": selected, "old_marker_sha256": sha256(OLD_MARKER),
         "dead_owner_observation_sha256": sha256(TERMINAL),
         "unchanged_tensor_V2_protocol_sha256": sha256(tensor.PROTOCOL),
         "original_source_protocol_sha256": sha256(tensor.original.PROTOCOL),
         "source_sha256": {SELF: sha256(ROOT / SELF)}, "namespace": PREFIX,
         "assets_namespace": str(ATTEMPT_ASSETS.relative_to(ASSETS)), "CPU_workers": 1, "CPU_threads": 1,
         "reserve_GiB": 60, "max_counterfactual_workers_unchanged": 3,
         "only_change": "New owner and isolated evidence paths after observed owner/child absence. Execute the same frozen V2 runtime/label code on all19 original failures; redo, never adopt old partials.",
         "cases_frames_actions_clicks_features_labels_candidate_association_unchanged": True,
         "old_markers_logs_partials_completed_clicks_preserved": True,
         "actual_model_generated_on_policy": False, "scientific_success": False,
         "confirmation_or_next_stage_authorized": False}
    validate_frozen_sources(p)
    write_json(str(PROTOCOL.relative_to(OUT)), p)


def operation(sequence, name):
    p = read_json(PROTOCOL)
    validate_frozen_sources(p)
    if sequence not in p["sequences"] or name not in ("runtime", "label"):
        raise ValueError("Only unchanged registered recovery operations")
    saved = tensor.PREFIX, tensor.ASSETS, tensor.write_json
    old_original = {key: getattr(tensor.original, key) for key in ("OUT", "ASSETS", "write_json", "collect_case")}
    def annotated(relative, value, **kwargs):
        if relative.startswith(PREFIX + "/"):
            value = {**value, "terminal_resume_protocol_sha256": sha256(PROTOCOL),
                     "terminal_resume_source_sha256": p["source_sha256"],
                     "dead_V2_evidence_preserved_NOT_reclassified": True}
        return write_json(relative, value, **kwargs)
    try:
        tensor.PREFIX, tensor.ASSETS, tensor.write_json = PREFIX, ATTEMPT_ASSETS, annotated
        tensor.run(sequence, name)
    finally:
        tensor.PREFIX, tensor.ASSETS, tensor.write_json = saved
        for key, value in old_original.items():
            setattr(tensor.original, key, value)


def install(sequence):
    p = read_json(PROTOCOL)
    validate_frozen_sources(p)
    development_sequence(sequence)
    if sequence not in p["sequences"]:
        raise ValueError("No replacement/additional video")
    v2 = read_json(tensor.PROTOCOL)
    sources = {}
    for folder in ("runtime_sequences", "supervision"):
        path = OUT / PREFIX / folder / (sequence + ".json")
        value = read_json(path)
        assert value["protocol_sha256"] == p["original_source_protocol_sha256"]
        assert value["tensor_recovery_protocol_sha256"] == p["unchanged_tensor_V2_protocol_sha256"]
        assert value["tensor_recovery_wrapper_source_sha256"] == v2["wrapper_source_sha256"]
        assert value["source_sha256"] == v2["original_source_sha256"]
        assert value["terminal_resume_protocol_sha256"] == sha256(PROTOCOL)
        assert value["terminal_resume_source_sha256"] == p["source_sha256"]
        if "artifact" in value:
            assert sha256(value["artifact"]["path"]) == value["artifact"]["sha256"]
        if folder == "runtime_sequences":
            for row in value["episodes"]:
                assert sha256(row["path"]) == row["sha256"]
                episode = read_json(row["path"])
                assert sha256(episode["artifact"]["path"]) == episode["artifact"]["sha256"]
        canonical = "on_policy/joint_state_v1/" + folder + "/" + sequence + ".json"
        if (OUT / canonical).exists():
            raise FileExistsError("Never overwrite existing canonical state evidence")
        sources[canonical] = {**value, "successful_tensor_recovery_path": str(path),
            "successful_tensor_recovery_sha256": sha256(path), "canonicalization_source_sha256": sha256(__file__),
            "original_terminal_driver_path": str(ORIGINAL_MARKER), "original_terminal_driver_sha256": sha256(ORIGINAL_MARKER),
            "dead_V2_owner_observation_path": str(TERMINAL), "dead_V2_owner_observation_sha256": sha256(TERMINAL),
            "original_V1_failure_retained": True, "source_case_frames_actions_clicks_and_labels_unchanged": True,
            "NOT_scientific_success": True}
    # Check both receipts and all artifacts BEFORE installing either canonical.
    for relative, value in sources.items():
        write_json(relative, value)
    for row in read_json(TERMINAL)["preserved_completed_original_receipts"]:
        assert sha256(row["path"]) == row["sha256"]
    write_json(PREFIX + "/recovered_sequences/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "status": "COMPLETE_SAME_FROZEN_V2_REPLAY_NEW_TERMINAL_RESUME_PATHS",
        "protocol_sha256": sha256(PROTOCOL), "canonical_sha256": {relative: sha256(OUT / relative) for relative in sources},
        "old_failures_partials_and_completed_originals_preserved": True, "scientific_success": False,
        "next_stage_authorized": False})


def worker(sequence):
    # A V3 worker is a new attempt; no output/partial from any older attempt is adopted.
    for name in ("runtime", "label"):
        result = subprocess.run([str(PYTHON), "-u", "-m", "scripts.n72r21r2_joint_state_terminal_resume_v3",
                                 "--operation", name, "--sequence", sequence], cwd=ROOT, check=False)
        if result.returncode:
            raise RuntimeError("Preserve failed V3 " + name + "/" + sequence)
    install(sequence)


def run():
    p = read_json(PROTOCOL)
    validate_frozen_sources(p)
    marker = PREFIX + "/driver.json"
    if (OUT / marker).exists() or scoped_live_processes():
        raise FileExistsError("Inspect ownership; never start a duplicate/re-adopt attempt")
    state = {"stage": "N72R21R2", "pid": os.getpid(), "status": "ACTIVE_UNCHANGED_ALL19_TERMINAL_STATE_RECOVERIES",
             "protocol_sha256": sha256(PROTOCOL), "source_sha256": p["source_sha256"], "CPU_workers": 1,
             "active": None, "completed": [], "failed_retained": []}
    write_json(marker, state)
    for sequence in p["sequences"]:
        storage(256 << 20)
        log = ASSETS / "joint_state_terminal_resume_v3_logs" / (sequence + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as handle:
            child = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_joint_state_terminal_resume_v3",
                                      "--worker", "--sequence", sequence], cwd=ROOT,
                                     env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                                     stdout=handle, stderr=subprocess.STDOUT)
            state["active"] = {"sequence": sequence, "pid": child.pid, "log": str(log)}
            write_json(marker, state, mutable=True)
            code = child.wait()
        if code:
            state["failed_retained"].append({"sequence": sequence, "actual_exit_code": code, "actual_exit_code_observed": True, "log": str(log)})
        else:
            assert (OUT / PREFIX / "recovered_sequences" / (sequence + ".json")).exists()
            state["completed"].append(sequence)
        state["active"] = None
        write_json(marker, state, mutable=True)
        print({"actual_same_protocol_recoveries": len(state["completed"]), "failed_retained": len(state["failed_retained"])}, flush=True)
    state["status"] = "COMPLETE_UNCHANGED_ALL19_STATE_RECOVERIES" if not state["failed_retained"] else "AVAILABLE_STATE_RECOVERIES_WITH_FAILURES_RETAINED"
    write_json(marker, state, mutable=True)
    append_log("M7_TERMINAL_STATE_RESUME_V3_FINISHED", completed=state["completed"], failed=state["failed_retained"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--worker", action="store_true")
    group.add_argument("--operation", choices=("runtime", "label"))
    parser.add_argument("--sequence")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.run:
        run()
    elif args.worker:
        worker(args.sequence)
    else:
        operation(args.sequence, args.operation)
