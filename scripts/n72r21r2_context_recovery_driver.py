"""Recover only terminal V1-context label failures, without modifying tapes.

The original driver's failed receipt remains failed. A previously absent
canonical label receipt explicitly references the successful V2 attempt;
neither a V1 partial nor any completed V1 label receipt can be overwritten.
"""
import argparse
import os
from pathlib import Path
import subprocess
import time
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, read_json, write_json, sha256, preregistration, development_sequence, append_log


def install_verified_recovery(sequence):
    development_sequence(sequence)
    source_path = OUT / "events/label_audit_v2" / (sequence + ".json")
    source = read_json(source_path)
    assert source["wrapper_source_sha256"] == sha256(ROOT / "scripts/n72r21r2_label_counterfactual_v2.py")
    assert source["exact_context_v2_source_sha256"] == sha256(ROOT / "scripts/n72r21r2_event_context_v2.py")
    assert source["source_sha256"] == sha256(ROOT / "scripts/n72r21r2_label_counterfactual.py")
    assert sha256(source["artifact"]["path"]) == source["artifact"]["sha256"]
    context_path = OUT / "events/current_context_sequences_v2" / (sequence + ".json")
    context = read_json(context_path)
    assert context["actual_event_contexts"] == context["actual_tensor_inclusive_CF_matching_contexts"] == source["events"]
    assert context["original_raw_sealed_click_preserved"] and context["original_context_V1_not_overwritten"]
    partial_path = ASSETS / "offline_supervision_cf_v1" / (sequence + ".jsonl.zst")
    failed_log = ASSETS / "event_driver_v1_logs" / (sequence + ".log")
    failed = read_json(OUT / "events/event_driver_v1.json")["failed_retained"]
    assert sequence in {r["sequence"] for r in failed}
    assert "n72r21r2_label_counterfactual" in failed_log.read_text() and "AssertionError" in failed_log.read_text()
    record = {**source, "status": "COMPLETE_ACTUAL_LABELS_RECOVERED_WITH_EXACT_RAW_ANCHOR_CONTEXT_V2",
              "canonicalization_source_sha256": sha256(__file__),
              "successful_recovery_receipt_path": str(source_path), "successful_recovery_receipt_sha256": sha256(source_path),
              "verified_tensor_context_receipt_path": str(context_path), "verified_tensor_context_receipt_sha256": sha256(context_path),
              "original_failed_label_log": {"path": str(failed_log), "sha256": sha256(failed_log)},
              "original_failed_partial_preserved": {"path": str(partial_path), "sha256": sha256(partial_path), "bytes": partial_path.stat().st_size},
              "not_a_reclassification_of_V1_failure": True, "original_CF_tapes_and_sampling_not_rerun_or_changed": True}
    # Missing file only: completed or partial JSON evidence is never replaced.
    return write_json("events/label_audit/" + sequence + ".json", record)


def worker(sequence):
    development_sequence(sequence)
    for module, prefix in (("scripts.n72r21r2_event_context_v2", "events/current_context_sequences_v2"),
                           ("scripts.n72r21r2_label_counterfactual_v2", "events/label_audit_v2")):
        if not (OUT / prefix / (sequence + ".json")).exists():
            result = subprocess.run([str(PYTHON), "-u", "-m", module, "--sequence", sequence], cwd=ROOT, check=False)
            if result.returncode:
                raise RuntimeError("Preserve failed exact-context recovery: " + module)
    canonical = OUT / "events/label_audit" / (sequence + ".json")
    if not canonical.exists():
        install_verified_recovery(sequence)
    else:
        assert read_json(canonical)["canonicalization_source_sha256"] == sha256(__file__)
    window = OUT / "events/window_trackeval_results" / (sequence + ".json")
    if not window.exists():
        result = subprocess.run([str(PYTHON), "-u", "-m", "scripts.n72r21r2_window_trackeval", "--sequence", sequence], cwd=ROOT, check=False)
        if result.returncode:
            raise RuntimeError("Preserve failed actual recovered window TrackEval")
    write_json("events/recovered_sequences_v2/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "status": "COMPLETE_VERIFIED_CONTEXT_REPAIR_ACTUAL_LABELS_AND_WINDOW_METRICS",
        "canonical_label_audit_sha256": sha256(canonical), "window_metric_receipt_sha256": sha256(window),
        "source_sha256": sha256(__file__), "original_driver_failure_retained": True, "not_scientific_success": True})
    append_log("M3_EXACT_CONTEXT_RECOVERY_COMPLETE", sequence=sequence)


def run():
    protocol = preregistration()
    permitted = protocol["split"]["fit"] + protocol["split"]["inner"]
    marker = "events/context_recovery_driver_v2.json"
    if (OUT / marker).exists():
        raise FileExistsError("Inspect actual ownership before another recovery driver")
    state = {"stage": "N72R21R2", "pid": os.getpid(), "status": "ACTIVE", "CPU_workers": 1,
             "source_sha256": sha256(__file__), "completed": [], "failed_retained": [], "active": None}
    write_json(marker, state)
    child, last = None, 0.
    while True:
        if child and child["process"].poll() is not None:
            child["handle"].close()
            if child["process"].returncode:
                state["failed_retained"].append({"sequence": child["sequence"], "returncode": child["process"].returncode, "log": child["log"]})
            child = None
        original = read_json(OUT / "events/event_driver_v1.json")
        completed = [s for s in permitted if (OUT / "events/recovered_sequences_v2" / (s + ".json")).exists()]
        rejected = {r["sequence"] for r in state["failed_retained"]}
        if child is None:
            selected = next((r["sequence"] for r in original["failed_retained"] if r["sequence"] in permitted and r["sequence"] not in completed and r["sequence"] not in rejected), None)
            if selected:
                log = ASSETS / "context_recovery_driver_v2_logs" / (selected + ".log")
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = log.open("x")
                process = subprocess.Popen([str(PYTHON), "-u", "-m", "scripts.n72r21r2_context_recovery_driver", "--worker", selected], cwd=ROOT,
                                           env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=handle, stderr=subprocess.STDOUT)
                child = {"sequence": selected, "process": process, "handle": handle, "log": str(log)}
        if time.monotonic() - last > 20:
            state.update(completed=completed, active=None if child is None else {"sequence": child["sequence"], "pid": child["process"].pid, "log": child["log"]})
            write_json(marker, state, mutable=True)
            print({"exact_context_recovery_driver": "ACTIVE", "completed": completed, "active": state["active"], "failed": state["failed_retained"]}, flush=True)
            last = time.monotonic()
        if original["status"] != "ACTIVE" and child is None:
            break
        time.sleep(2)
    state.update(status="COMPLETE" if not rejected else "RECOVERABLE_CASES_COMPLETE_FAILURES_RETAINED", active=None)
    write_json(marker, state, mutable=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    args = parser.parse_args()
    worker(args.worker) if args.worker else run()
