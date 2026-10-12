"""Resume the original nine state-fit dispositions, not nine optimizations.

The three sealed BASELINE zero-benefit failures remain immutable; only the
six original TREATMENT/MIXED seeds execute the unchanged training entrypoint.
Controlled-state training is not model-generated on-policy training.
"""
import argparse
import os
from pathlib import Path
import subprocess
from scripts import n72r21r2_train_joint_state as study
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, GOAL, read_json, write_json, sha256, storage, utcnow
from scripts.n72r21r2_storage_guard_recovery_v1 import dead_idle_storage_owner
from scripts.n72r21r2_main_terminal_recovery_v1 import no_live_scoped_workers

SELF = "scripts/n72r21r2_state_fit_storage_resume_v3.py"
PREFIX = "audit/state_fit_storage_resume_v3"
MARKER = "on_policy/state_source_fits_v1/driver.json"
FRESH = PREFIX + "/owner.json"
PROTOCOL = OUT / "audit/STATE_FIT_STORAGE_RESUME_PROTOCOL_V3.json"
SUPPORT = OUT / "on_policy/state_source_support_v1/ACTUAL_READBACK.json"
OLD_LOG = ASSETS / "idle_driver_terminal_recovery_v2_logs/STATE_FIT.log"


def valid_no_fit(row, seed, protocol_sha):
    return (row.get("status") == "FAIL_MEASURED_STATE_SOURCE_SUPERVISION_DIVERSITY"
            and row.get("mode") == "BASELINE_STATE" and row.get("seed") == seed
            and row.get("optimizer_steps") == 0 and row.get("FIT_rows") == 517
            and row.get("INNER_rows") == 1212
            and row.get("FIT_target_counts") == {"benefit": 0, "risk": 336}
            and row.get("protocol_sha256") == protocol_sha)


def validate_old():
    no_live_scoped_workers(("scripts.n72r21r2_state_fit_storage_resume_v3",
                           "scripts.n72r21r2_joint_state_fit_driver", "scripts.n72r21r2_train_joint_state"))
    old = read_json(OUT / MARKER)
    observed = dead_idle_storage_owner(old, OLD_LOG)
    assert old["completed"] == [] and old["failed_retained"] == []
    p = read_json(study.PROTOCOL)
    assert p["source_sha256"] == {name: sha256(ROOT / name) for name in study.CODE}
    support = read_json(SUPPORT)
    assert support["original_protocol_sha256"] == sha256(study.PROTOCOL)
    refs = {str(SUPPORT): sha256(SUPPORT), str(OUT / MARKER): sha256(OUT / MARKER),
            str(OLD_LOG): sha256(OLD_LOG), str(study.PROTOCOL): sha256(study.PROTOCOL)}
    source = read_json(OUT / "on_policy/ALL24_CONTROLLED_SOURCE_ACTUAL_CANONICAL_READBACK_V1.json")
    assert source["actual_verified_sequences"] == 24
    refs.update(source["evidence_sha256"])
    assert all(sha256(path) == digest for path, digest in refs.items())
    baseline = []
    for ref in support["original_baseline_failure_receipts"]:
        assert sha256(ref["path"]) == ref["sha256"]
        row = read_json(ref["path"])
        assert valid_no_fit(row, ref["seed"], sha256(study.PROTOCOL))
        assert not (ASSETS / "training/state_source_fits_v1" / ("BASELINE_STATE__seed" + str(ref["seed"]))).exists()
        refs[ref["path"]] = ref["sha256"]
        baseline.append(ref)
    assert [r["seed"] for r in baseline] == p["seeds"]
    jobs = [{"mode": mode, "seed": seed} for mode in p["modes"] if mode != "BASELINE_STATE" for seed in p["seeds"]]
    assert len(jobs) == 6
    for job in jobs:
        uid = job["mode"] + "__seed" + str(job["seed"])
        assert not (OUT / "on_policy/state_source_fits_v1" / (uid + ".json")).exists()
        assert not (ASSETS / "training/state_source_fits_v1" / uid).exists()
    return {"old_marker": old, "storage_stop": observed, "references": refs,
            "baseline_measured_no_fit": baseline, "remaining_original_jobs": jobs}


def freeze():
    if PROTOCOL.exists() or (OUT / FRESH).exists():
        raise FileExistsError("Preserve preparation and exclusive ownership")
    storage(32 << 20)
    row = validate_old()
    archive = write_json(PREFIX + "/old_marker_exact_archive.json", row["old_marker"])
    assert sha256(archive) == sha256(OUT / MARKER)
    write_json(str(PROTOCOL.relative_to(OUT)), {"stage": "N72R21R2", "goal": GOAL,
        "frozen": True, "source_sha256": {SELF: sha256(ROOT / SELF)},
        "original_fit_protocol_sha256": sha256(study.PROTOCOL), **row,
        "old_marker_archive": str(archive), "CPU_workers": 1, "CPU_threads": 1,
        "storage_floor_GiB": 60, "new_model_seed_threshold_or_data": False,
        "actual_model_generated_on_policy_training": False, "confirmation_authorized": False})
    print({"preserved_baseline_no_fit": 3, "remaining_original_fits": 6}, flush=True)


def run():
    p = read_json(PROTOCOL)
    assert p["frozen"] and p["goal"] == GOAL and p["source_sha256"] == {SELF: sha256(ROOT / SELF)}
    assert p["original_fit_protocol_sha256"] == sha256(study.PROTOCOL)
    assert validate_old() == {key: p[key] for key in ("old_marker", "storage_stop", "references", "baseline_measured_no_fit", "remaining_original_jobs")}
    if (OUT / FRESH).exists():
        raise FileExistsError("Never duplicate state-source fit owner")
    state = {"stage": "N72R21R2", "goal": GOAL, "pid": os.getpid(), "CPU_workers": 1,
        "status": "ACTIVE_FIT_ONLY_PAIRED_STATE_SOURCE_CONTRAST", "active": None,
        "completed": [], "failed_retained": [], "protocol_sha256": sha256(study.PROTOCOL),
        "recovery_protocol_sha256": sha256(PROTOCOL),
        "baseline_measured_no_fit_dispositions_not_models": p["baseline_measured_no_fit"],
        "actual_model_generated_on_policy_training": False, "confirmation_authorized": False}
    def save():
        write_json(FRESH, state, mutable=(OUT / FRESH).exists())
        write_json(MARKER, state, mutable=True)
    save()
    for job in p["remaining_original_jobs"]:
        storage(32 << 20)
        uid = job["mode"] + "__seed" + str(job["seed"])
        destination = OUT / "on_policy/state_source_fits_v1" / (uid + ".json")
        if destination.exists():
            raise FileExistsError("No implicit refit or seal adoption")
        log = ASSETS / "state_fit_storage_resume_v3" / (uid + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [str(PYTHON), "-u", "-m", "scripts.n72r21r2_train_joint_state", "--mode", job["mode"], "--seed", str(job["seed"])]
        with log.open("x") as handle:
            child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}, stdout=handle, stderr=subprocess.STDOUT)
            state["active"] = {"experiment_uid": uid, "pid": child.pid, "log": str(log)}
            save()
            code = child.wait()
        if code:
            state["failed_retained"].append({"experiment_uid": uid, "returncode": code, "log": str(log)})
        else:
            receipt = read_json(destination)
            assert receipt["mode"] == job["mode"] and receipt["seed"] == job["seed"]
            assert receipt["protocol_sha256"] == sha256(study.PROTOCOL)
            if receipt["status"] == "COMPLETE_ACTUAL_STATE_SOURCE_CONTRAST_UNCALIBRATED_NOT_DEPLOYABLE":
                assert receipt["nonzero_gradient_steps"] > 0 and receipt["changed_weight_elements"] > 0
                assert sha256(receipt["checkpoint_path"]) == receipt["checkpoint_sha256"]
                state["completed"].append(uid)
            else:
                state["failed_retained"].append({"experiment_uid": uid, "status": receipt["status"], "returncode": code, "log": str(log)})
        state["active"] = None
        save()
        print({"actual_original_state_fits": len(state["completed"]), "new_failed_attempts": state["failed_retained"]}, flush=True)
    state["status"] = "ACTUAL_AVAILABLE_FITS_FAILURES_RETAINED" if state["failed_retained"] else "COMPLETE_REGISTERED_STATE_SOURCE_ATTEMPTS_NOT_ON_POLICY_CLOSURE"
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run"))
    {"freeze": freeze, "run": run}[parser.parse_args().action]()
