"""One CPU pipeline: all24 bank sources ->6 fits -> actual zero-authority memory."""
import os
import subprocess
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, sha256, storage, preregistration, append_log
from scripts.n72r21r2_memory_risk_data import PROTOCOL


def all_sources_ready(sequences):
    return all((OUT / "memory/risk_v1/supervision" / (s + ".json")).exists()
               and (OUT / "memory/M_A/results" / (s + ".json")).exists() for s in sequences)


def launch(module, args, log, state, marker, phase, identity):
    storage(64 << 20)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as handle:
        process = subprocess.Popen([str(PYTHON), "-u", "-m", module, *args], cwd=ROOT,
                                   env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=handle, stderr=subprocess.STDOUT)
        state["active"] = {"phase": phase, "identity": identity, "pid": process.pid, "log": str(log)}
        write_json(marker, state, mutable=True)
        code = process.wait()
    if code:
        state["failed_retained"].append({"phase": phase, "identity": identity, "returncode": code, "log": str(log)})
    state["active"] = None
    write_json(marker, state, mutable=True)
    return code == 0


def run():
    marker = "memory/risk_v1/driver.json"
    if (OUT / marker).exists():
        raise FileExistsError("Inspect actual memory-risk ownership before duplicate launch")
    p = read_json(PROTOCOL)
    split = preregistration()["split"]
    sequences = split["fit"] + split["inner"]
    state = {"stage": "N72R21R2", "status": "COLLECTING_ALL24_ACTUAL_TEACHER_BANK_INPUTS", "pid": os.getpid(),
             "source_sha256": sha256(__file__), "protocol_sha256": sha256(PROTOCOL), "active": None,
             "data_completed": [], "fit_completed": [], "runtime_completed": [], "failed_retained": [],
             "CPU_workers": 1, "no_association_authority_or_automatic_M_B": True}
    write_json(marker, state)
    last = 0.
    while not all_sources_ready(sequences):
        state["data_completed"] = [s for s in sequences if (OUT / "memory/risk_v1/supervision" / (s + ".json")).exists()]
        failed = {r["identity"] for r in state["failed_retained"] if r["phase"] in ("collect", "label")}
        ready = [s for s in sequences if s not in state["data_completed"] and s not in failed
                 and (OUT / "memory/M_A/results" / (s + ".json")).exists()]
        if ready:
            sequence = ready[0]
            for action, folder in (("collect", "runtime_inputs"), ("label", "supervision")):
                if (OUT / "memory/risk_v1" / folder / (sequence + ".json")).exists():
                    continue
                log = ASSETS / "memory_risk_driver_v1_logs" / (sequence + "__" + action + ".log")
                if not launch("scripts.n72r21r2_memory_risk_data", [action, "--sequence", sequence], log, state, marker, action, sequence):
                    break
            continue
        if failed:
            state.update(status="ACTUAL_SOURCE_FAILURES_RETAINED_NO_SUBSET_FITS", active=None)
            write_json(marker, state, mutable=True)
            return
        if time.monotonic() - last > 20:
            write_json(marker, state, mutable=True)
            print({"memory_risk_driver": state["status"], "data_ready": len(state["data_completed"]), "required": 24}, flush=True)
            last = time.monotonic()
        time.sleep(2)
    state["status"] = "ACTUAL_ALL24_FIT_ONLY_CURRENT_WRITE_RISK_OPTIMIZATION"
    for family in p["families"]:
        for seed in p["seeds"]:
            uid = family + "__seed" + str(seed)
            path = OUT / "memory/risk_v1/fits" / (uid + ".json")
            if path.exists():
                raise FileExistsError("Do not silently replace/adopt current writer fits")
            log = ASSETS / "memory_risk_driver_v1_logs" / (uid + "__fit.log")
            if launch("scripts.n72r21r2_train_memory_risk", ["--family", family, "--seed", str(seed)], log, state, marker, "fit", uid):
                record = read_json(path)
                if "checkpoint_path" in record:
                    state["fit_completed"].append(uid)
                else:
                    state["failed_retained"].append({"phase": "fit", "identity": uid, "reason": record["status"]})
    if len(state["fit_completed"]) != 6:
        state.update(status="ACTUAL_INCOMPLETE_OR_FAILED_FITS_NO_INVENTED_OWN_MEMORY_RUNTIME", active=None)
        write_json(marker, state, mutable=True)
        return
    state["status"] = "ACTUAL_OWN_BANK_ZERO_AUTHORITY_RUNTIME_AND_OFFLINE_EVALUATION"
    for sequence in sequences:
        success = True
        for action, folder in (("runtime", "runtime_sequences"), ("evaluate", "results")):
            path = OUT / "memory/risk_v1" / folder / (sequence + ".json")
            if path.exists():
                assert read_json(path)["protocol_sha256"] == sha256(PROTOCOL)
                continue
            log = ASSETS / "memory_risk_driver_v1_logs" / (sequence + "__" + action + ".log")
            if not launch("scripts.n72r21r2_memory_risk_runtime", [action, "--sequence", sequence], log, state, marker, action, sequence):
                success = False
                break
        if success:
            state["runtime_completed"].append(sequence)
        write_json(marker, state, mutable=True)
    state.update(status="COMPLETE_ALL24_ACTUAL_CURRENT_RISK_MEMORY_G4_SUMMARY_PENDING" if not state["failed_retained"] else "ACTUAL_AVAILABLE_CASES_COMPLETE_FAILURES_RETAINED", active=None)
    write_json(marker, state, mutable=True)
    append_log("M9_FRESH_CURRENT_WRITE_RISK_PIPELINE_FINISHED", fits=len(state["fit_completed"]), runtime_videos=len(state["runtime_completed"]), failed=state["failed_retained"])


if __name__ == "__main__":
    run()
