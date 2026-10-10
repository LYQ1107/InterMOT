"""Wait for all24 new source receipts, then actual fixed12 FIT-only fits."""
import os
import subprocess
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, preregistration, sha256, storage, append_log


def run():
    marker = "availability/open_set_fit_driver_v1.json"
    if (OUT / marker).exists():
        raise FileExistsError("Inspect existing actual fit ownership before another driver")
    prereg = preregistration()
    sequences = prereg["split"]["fit"] + prereg["split"]["inner"]
    protocol_path = OUT / "availability/CURRENT_AXIS_FITS_PROTOCOL_V1.json"
    p = read_json(protocol_path)
    jobs = [(family, seed) for family in p["families"] for seed in p["seeds"]]
    state = {"stage": "N72R21R2", "pid": os.getpid(), "status": "WAITING_ALL24_CURRENT_AXIS_SOURCE_RECEIPTS",
             "source_sha256": sha256(__file__), "protocol_sha256": sha256(protocol_path), "CPU_workers": 1,
             "completed": [], "failed_retained": [], "active": None}
    write_json(marker, state)
    last = 0.
    while not all((OUT / "availability/current_axis_v1/supervision" / (s + ".json")).exists() for s in sequences):
        if time.monotonic() - last > 20:
            state["data_ready_count"] = sum((OUT / "availability/current_axis_v1/supervision" / (s + ".json")).exists() for s in sequences)
            write_json(marker, state, mutable=True)
            print({"fresh_open_set_fit_driver": state["status"], "ready": state["data_ready_count"]}, flush=True)
            last = time.monotonic()
        source = read_json(OUT / "availability/open_set_driver_v1.json")
        if source["status"] != "ACTIVE" and source["failed_retained"]:
            state.update(status="SOURCE_DATA_FAILED_RETAINED_NOT_MAIN_FITS", source_failures=source["failed_retained"])
            write_json(marker, state, mutable=True)
            return
        time.sleep(2)
    state["status"] = "ACTIVE_FIXED_FRESH_FIT_ONLY_OPTIMIZATION"
    for family, seed in jobs:
        uid = family + "__seed" + str(seed)
        destination = OUT / "availability/current_axis_fits" / (uid + ".json")
        if destination.exists():
            record = read_json(destination)
            assert record["protocol_sha256"] == sha256(protocol_path)
            assert sha256(record["checkpoint_path"]) == record["checkpoint_sha256"]
            state["completed"].append(uid)
            continue
        storage(64 << 20)
        log = ASSETS / "open_set_fit_driver_v1_logs" / (uid + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as handle:
            process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_train_open_set", "--family", family, "--seed", str(seed)], cwd=ROOT,
                                       env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=handle, stderr=subprocess.STDOUT)
            state["active"] = {"experiment": uid, "pid": process.pid, "log": str(log)}
            write_json(marker, state, mutable=True)
            code = process.wait()
        if code:
            state["failed_retained"].append({"experiment": uid, "returncode": code, "log": str(log)})
        else:
            assert destination.exists()
            state["completed"].append(uid)
        state["active"] = None
        write_json(marker, state, mutable=True)
        print({"actual_fresh_open_set_fits_completed": len(state["completed"]), "failed_retained": state["failed_retained"]}, flush=True)
    if not (OUT / "availability/SIMPLE_CURRENT_AXIS_CALIBRATION_V1.json").exists():
        log = ASSETS / "open_set_fit_driver_v1_logs/SIMPLE_CALIBRATION.log"
        with log.open("x") as handle:
            result = subprocess.run([str(PYTHON), "-u", "-m", "scripts.n72r21r2_train_open_set", "--simple"], cwd=ROOT,
                                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            state["failed_retained"].append({"experiment": "SIMPLE_CALIBRATION", "returncode": result.returncode, "log": str(log)})
    state.update(status="COMPLETE_ACTUAL_CURRENT_AXIS_FITS_NOT_MOT_SUCCESS" if not state["failed_retained"] else "ACTUAL_AVAILABLE_FITS_COMPLETE_FAILURES_RETAINED", active=None)
    write_json(marker, state, mutable=True)
    append_log("M8_FRESH_FIXED_FITS_DRIVER_COMPLETE", models=len(state["completed"]), failed=state["failed_retained"])


if __name__ == "__main__":
    run()
