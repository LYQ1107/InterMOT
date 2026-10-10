"""All24 prerequisite gate, then the already registered42 actual main fits.

No ready-subset optimization and no historical scene substitution. The
independent memory/open-set pipeline is not a prerequisite or an authority
qualification shortcut. Main fits remain uncalibrated development weights.
"""
import os
import subprocess
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, sha256, preregistration, storage, append_log
from sam3_intermot.one_click.event_authority_learning import OBJECTIVES

PREREQUISITES = ("data/candidate_integrity", "mot/baseline_results", "events/counterfactual_sequences",
                 "events/current_context_sequences", "events/label_audit", "events/trajectory_utility_audit", "simple/results")


def all_required_ready(sequences):
    return all((OUT / folder / (sequence + ".json")).exists() for sequence in sequences for folder in PREREQUISITES)


def freeze():
    p = read_json(OUT / "training/EVENT_AUTHORITY_PROTOCOL_V1.json")
    jobs = [(family, p["primary_family_objective"], seed) for family in p["families"] for seed in p["seeds"]]
    jobs += [("SMALL_MLP", objective, seed) for objective in OBJECTIVES if objective != p["primary_family_objective"] for seed in p["seeds"]]
    assert len(jobs) == len(set(jobs)) == 42
    write_json("training/MAIN_EXECUTION_PROTOCOL_V1.json", {
        "stage": "N72R21R2", "goal_file": "outputs/N72R21R2/FINAL_GOAL.json", "frozen_before_any_main_fit": True,
        "event_fit_protocol_sha256": sha256(OUT / "training/EVENT_AUTHORITY_PROTOCOL_V1.json"),
        "source_sha256": sha256(__file__), "jobs": [{"family": f, "objective": o, "seed": s} for f, o, s in jobs],
        "required_source_receipts_each_of_all24_videos": list(PREREQUISITES),
        "initialization_failures": "All failed clips/videos are retained, without replacement. Zero-row video is not optimizer evidence or successful identity recognition.",
        "recovered_context": "Canonical label receipt for a terminal failed V1 attempt may reference actual V2 raw-anchor replay; all successful V2, original failure and partial SHA evidence remain immutable. This changes no CF arm or definition.",
        "CPU_workers": 1, "CPU_threads_per_fit": 1, "no_uncalibrated_MOT_authority": True,
        "RELATIONAL_or_confirmation_automatic": False, "not_on_policy_or_main_MOT_completion": True})


def verify_prerequisites(sequences):
    assert all_required_ready(sequences)
    receipts = []
    for sequence in sequences:
        labels_path = OUT / "events/label_audit" / (sequence + ".json")
        labels = read_json(labels_path)
        assert sha256(labels["artifact"]["path"]) == labels["artifact"]["sha256"]
        assert labels["all_registered_sequence_runtime_verified_before_GT_labels"]
        if "successful_recovery_receipt_path" in labels:
            assert sha256(labels["successful_recovery_receipt_path"]) == labels["successful_recovery_receipt_sha256"]
            assert sha256(labels["verified_tensor_context_receipt_path"]) == labels["verified_tensor_context_receipt_sha256"]
            assert labels["canonicalization_source_sha256"] == sha256(ROOT / "scripts/n72r21r2_context_recovery_driver.py")
            assert labels["wrapper_source_sha256"] == sha256(ROOT / "scripts/n72r21r2_label_counterfactual_v2.py")
            assert labels["exact_context_v2_source_sha256"] == sha256(ROOT / "scripts/n72r21r2_event_context_v2.py")
            assert labels["not_a_reclassification_of_V1_failure"]
        receipt = {folder: sha256(OUT / folder / (sequence + ".json")) for folder in PREREQUISITES}
        init = read_json(OUT / "data/initialization" / (sequence + ".json"))
        receipts.append({"sequence": sequence, "artifact_receipts_sha256": receipt,
                         "valid_clicks": sum(not e["initialization_failure"] for e in init["inputs"]),
                         "failed_clicks_not_replaced": sum(e["initialization_failure"] for e in init["inputs"])})
    return receipts


def run():
    marker = "training/main_fit_driver_v1.json"
    if (OUT / marker).exists():
        raise FileExistsError("Inspect actual main fit ownership before another driver")
    freeze()
    protocol_path = OUT / "training/MAIN_EXECUTION_PROTOCOL_V1.json"
    p = read_json(protocol_path)
    prereg = preregistration()
    sequences = prereg["split"]["fit"] + prereg["split"]["inner"]
    state = {"stage": "N72R21R2", "status": "WAITING_COMPLETE_ALL24_FRESH_CF_CONTROLS_ACTUAL_UTILITY",
             "pid": os.getpid(), "source_sha256": sha256(__file__), "protocol_sha256": sha256(protocol_path),
             "active": None, "completed": [], "failed_retained": [], "CPU_workers": 1}
    write_json(marker, state)
    last = 0.
    while not all_required_ready(sequences):
        if time.monotonic() - last > 20:
            state["ready_videos_all_prerequisites"] = [s for s in sequences if all_required_ready([s])]
            write_json(marker, state, mutable=True)
            print({"main_fit_driver": state["status"], "ready": len(state["ready_videos_all_prerequisites"]), "required": 24}, flush=True)
            last = time.monotonic()
        time.sleep(2)
    source = verify_prerequisites(sequences)
    write_json("training/ALL24_MAIN_SOURCE_MANIFEST_V1.json", {
        "stage": "N72R21R2", "status": "ALL24_ACTUAL_MAIN_PREREQUISITES_VERIFIED_BEFORE_FIRST_OPTIMIZER_STEP",
        "protocol_sha256": sha256(protocol_path), "receipts": source, "no_ready_subset_or_old_scene_substitution": True})
    state["status"] = "ACTIVE_REGISTERED_MAIN_FIT_ONLY_OPTIMIZATION"
    for job in p["jobs"]:
        f, o, s = job["family"], job["objective"], job["seed"]
        uid = "MAIN__" + f + "__" + o + "__seed" + str(s)
        destination = OUT / "training/event_authority" / (uid + ".json")
        if destination.exists():
            raise FileExistsError("Do not silently adopt or replace a main fit outside this driver's ownership")
        storage(32 << 20)
        log = ASSETS / "main_fit_driver_v1_logs" / (uid + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as handle:
            process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_train_event_authority", "--family", f, "--objective", o, "--seed", str(s)], cwd=ROOT,
                                       env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=handle, stderr=subprocess.STDOUT)
            state["active"] = {"experiment_uid": uid, "pid": process.pid, "log": str(log)}
            write_json(marker, state, mutable=True)
            code = process.wait()
        if code:
            state["failed_retained"].append({"experiment_uid": uid, "returncode": code, "log": str(log)})
        else:
            assert destination.exists()
            state["completed"].append(uid)
        state["active"] = None
        write_json(marker, state, mutable=True)
        print({"actual_main_fits_complete": len(state["completed"]), "actual_failures_retained": len(state["failed_retained"]), "not_MOT_success": True}, flush=True)
    state.update(status="COMPLETE_ACTUAL_UNCALIBRATED_MAIN_FITS_ONLY" if not state["failed_retained"] else "ACTUAL_FITS_AVAILABLE_CASES_COMPLETE_FAILURES_RETAINED", active=None)
    write_json(marker, state, mutable=True)
    append_log("M5_M6_REGISTERED_MAIN_FIT_DRIVER_FINISHED", completed=len(state["completed"]), failed=state["failed_retained"])


if __name__ == "__main__":
    run()
