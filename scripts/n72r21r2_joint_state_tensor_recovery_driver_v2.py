"""One versioned tensor-proof recovery owner after all original24 attempts."""
import argparse
import os
import subprocess
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, sha256, preregistration, development_sequence, append_log, storage
from scripts.n72r21r2_joint_state_tensor_v2 import PROTOCOL, PREFIX, CODE


def install(sequence):
    development_sequence(sequence)
    original_path = OUT / "on_policy/joint_state_v1/driver.json"
    original = read_json(original_path)
    assert original["status"] != "ACTIVE"
    failed = next(r for r in original["failed_retained"] if r["sequence"] == sequence)
    log = failed["log"]
    assert "full_tracker_state_before_sha256" in open(log).read() and "AssertionError" in open(log).read()
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    partials = []
    for event in init["inputs"]:
        partial = ASSETS / "joint_state_curriculum_v1/runtime" / (event["episode_uid"] + ".jsonl.zst")
        if partial.exists():
            partials.append({"path": str(partial), "sha256": sha256(partial), "bytes": partial.stat().st_size})
    p = read_json(PROTOCOL)
    for folder in ("runtime_sequences", "supervision"):
        source_path = OUT / PREFIX / folder / (sequence + ".json")
        source = read_json(source_path)
        assert source["protocol_sha256"] == p["original_source_protocol_sha256"]
        assert source["tensor_recovery_protocol_sha256"] == sha256(PROTOCOL)
        assert source["tensor_recovery_wrapper_source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
        assert source["source_sha256"] == p["original_source_sha256"]
        if "artifact" in source:
            assert sha256(source["artifact"]["path"]) == source["artifact"]["sha256"]
        if folder == "runtime_sequences":
            for reference in source["episodes"]:
                assert sha256(reference["path"]) == reference["sha256"]
                episode = read_json(reference["path"])
                assert sha256(episode["artifact"]["path"]) == episode["artifact"]["sha256"]
        canonical = "on_policy/joint_state_v1/" + folder + "/" + sequence + ".json"
        if (OUT / canonical).exists():
            raise FileExistsError("Never overwrite completed/canonical original state evidence")
        write_json(canonical, {**source, "successful_tensor_recovery_path": str(source_path), "successful_tensor_recovery_sha256": sha256(source_path),
            "canonicalization_source_sha256": sha256(__file__), "original_terminal_driver_path": str(original_path),
            "original_terminal_driver_sha256": sha256(original_path), "original_failed_log": {"path": log, "sha256": sha256(log)},
            "original_failed_partials_preserved": partials, "original_V1_failure_retained": True,
            "source_case_frames_actions_clicks_and_labels_unchanged": True, "NOT_scientific_success": True})
    write_json("on_policy/joint_state_tensor_v2/recovered_sequences/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "status": "COMPLETE_ACTUAL_NULL_AWARE_TENSOR_STATE_RECOVERY",
        "runtime_canonical_sha256": sha256(OUT / "on_policy/joint_state_v1/runtime_sequences" / (sequence + ".json")),
        "supervision_canonical_sha256": sha256(OUT / "on_policy/joint_state_v1/supervision" / (sequence + ".json")),
        "original_V1_failure_retained": True, "next_stage_authorized": False})


def worker(sequence):
    for operation, folder in (("runtime", "runtime_sequences"), ("label", "supervision")):
        if not (OUT / PREFIX / folder / (sequence + ".json")).exists():
            result = subprocess.run([str(PYTHON), "-u", "-m", "scripts.n72r21r2_joint_state_tensor_v2", operation, "--sequence", sequence], cwd=ROOT, check=False)
            if result.returncode:
                raise RuntimeError("Preserve failed V2 tensor recovery: " + operation + "/" + sequence)
    install(sequence)


def run():
    marker = PREFIX + "/driver.json"
    if (OUT / marker).exists():
        raise FileExistsError("Inspect recovery owner before duplicate launch")
    state = {"stage": "N72R21R2", "pid": os.getpid(), "status": "WAITING_ORIGINAL_ALL24_TERMINAL_ATTEMPTS",
             "protocol_sha256": sha256(PROTOCOL), "source_sha256": sha256(__file__), "CPU_workers": 1,
             "completed": [], "failed_retained": [], "active": None, "preserve_max3_counterfactual_workers": True}
    write_json(marker, state)
    old_path = OUT / "on_policy/joint_state_v1/driver.json"
    while read_json(old_path)["status"] == "ACTIVE":
        time.sleep(2)
    original = read_json(old_path)
    permitted = preregistration()["split"]["fit"] + preregistration()["split"]["inner"]
    selected = [s for s in permitted if s in {r["sequence"] for r in original["failed_retained"]}]
    state.update(status="ACTIVE_TERMINAL_FAILURES_ONLY_RECOVERY", original_terminal_receipt_sha256=sha256(old_path), sequences=selected)
    for sequence in selected:
        storage(256 << 20)
        log = ASSETS / "joint_state_tensor_recovery_v2_logs" / (sequence + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as handle:
            process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_joint_state_tensor_recovery_driver_v2", "--worker", sequence], cwd=ROOT,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=handle, stderr=subprocess.STDOUT)
            state["active"] = {"sequence": sequence, "pid": process.pid, "log": str(log)}
            write_json(marker, state, mutable=True)
            code = process.wait()
        if code:
            state["failed_retained"].append({"sequence": sequence, "returncode": code, "log": str(log)})
        else:
            state["completed"].append(sequence)
        state["active"] = None
        write_json(marker, state, mutable=True)
        print({"actual_tensor_recovery_complete": len(state["completed"]), "failed_retained": state["failed_retained"]}, flush=True)
    state.update(status="COMPLETE_TERMINAL_TENSOR_FAILURE_RECOVERIES" if not state["failed_retained"] else "ACTUAL_RECOVERIES_AVAILABLE_FAILURES_RETAINED", active=None)
    write_json(marker, state, mutable=True)
    append_log("M7_NULL_TENSOR_RECOVERY_DRIVER_FINISHED", completed=state["completed"], failed=state["failed_retained"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    args = parser.parse_args()
    worker(args.worker) if args.worker else run()
