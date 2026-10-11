"""Resume original six risk writers/all24 videos after proven owner death.

No fitting, policy change or unsealed partial adoption. Original runtime skips
SHA-verified sealed cases; ten missing cases use a new process-local ASSETS.
A verified symlink tracker view permits actual mixed old/new whole-video CLI.
"""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
from scripts import n72r21r2_memory_risk_runtime as policy
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, GOAL, read_json, write_json, sha256, storage, preregistration, append_log
from scripts.n72r21r2_main_terminal_recovery_v1 import Evidence, dead_active, no_live_scoped_workers
from scripts.n72r21r2_events import artifact
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard

PREFIX = "audit/memory_terminal_recovery_v1"
PROTOCOL = OUT / "audit/MEMORY_TERMINAL_RECOVERY_PROTOCOL_V1.json"
SELF = "scripts/n72r21r2_memory_terminal_recovery_v1.py"
CODE = (SELF, "scripts/n72r21r2_main_terminal_recovery_v1.py")
MARKER = "memory/risk_v1/driver.json"
FRESH = PREFIX + "/owner.json"
INTERRUPTED = "dancetrack0016"
RECOVERY_ASSETS = ASSETS / "memory_terminal_recovery_v1"


def expected_grid(events, cases):
    if len({e["episode_uid"] for e in events}) != len(events) or len(set(cases)) != len(cases):
        raise ValueError("Unique valid sole-click/case axes required")
    return {case + "/" + e["episode_uid"] for case in cases for e in events}


def checked_fits(ev, p):
    registered = list(policy.cases(p))
    fits = {}
    for case, point, path, fit in registered:
        ev.json(path); ev.code(fit["source_sha256"])
        assert fit["checkpoint_sha256"] == ev.digest(fit["checkpoint_path"])
        assert fit["nonzero_gradient_steps"] > 0 and fit["changed_weight_elements"] > 0
        assert fit["association_authority"] is False
        assert fit["protocol_sha256"] == ev.digest(policy.PROTOCOL)
        assert len(fit["source_receipts"]) == 24
        for source in fit["source_receipts"]:
            assert source["sequence"] in p["split"][source["split"]]
            source_path = OUT / "memory/risk_v1/supervision" / (source["sequence"] + ".json")
            receipt = ev.json(source_path)
            assert ev.digest(source_path) == source["receipt_sha256"]
            assert receipt["artifact"]["sha256"] == source["data_sha256"]
            ev.artifact(receipt["artifact"])
        fits[case] = (path, fit)
    return registered, fits


def checked_case(sequence, case, event, fit_entry, p, ev, *, raw=False):
    path = OUT / "memory/risk_v1/runtime" / case / (event["episode_uid"] + ".json")
    seal = ev.json(path); fit_path, fit = fit_entry
    assert seal["sequence"] == sequence and seal["case"] == case and seal["event"] == event
    assert seal["protocol_sha256"] == ev.digest(policy.PROTOCOL) and seal["source_sha256"] == p["source_sha256"]
    assert seal["fit_receipt_sha256"] == ev.digest(fit_path) and seal["checkpoint_sha256"] == fit["checkpoint_sha256"]
    integrity = ev.json(OUT / "data/candidate_integrity" / (sequence + ".json"))
    assert seal["candidate_index_sha256"] == integrity["index_sha256"] == ev.digest(integrity["index_path"])
    assert seal["actual_GT_file_guard"] and seal["all_C0_outputs_semantic_states_trajectory_bytes_AA"]
    assert seal["all_full_tensor_tracker_states_AA_vs_same_M_A_FROZEN"]
    assert seal["counts"]["actual_original_frames"] == event["frames"]
    for ref in seal["artifacts"]: ev.artifact(ref)
    base = ev.json(OUT / "mot/baseline_runtime/CLICK_C0" / (event["episode_uid"] + ".json"))
    frozen = ev.json(OUT / "memory/M_A/runtime/FROZEN" / (event["episode_uid"] + ".json"))
    for receipt in (base, frozen):
        for ref in receipt["artifacts"]: ev.artifact(ref)
    assert seal["identity_checkpoint_sha256"] == frozen["identity_checkpoint_sha256"]
    assert next(a["sha256"] for a in seal["artifacts"] if a["kind"] == "trajectory") == next(a["sha256"] for a in base["artifacts"] if a["kind"] == "trajectory")
    if raw:
        rows = read_zstd_jsonl(artifact(seal, "trace"))
        baseline = read_zstd_jsonl(artifact(base, "trace"))
        reference = read_zstd_jsonl(artifact(frozen, "trace"))
        assert len(rows) == len(baseline) == len(reference) == event["frames"]
        writes = 0
        for frame, (row, old, same) in enumerate(zip(rows, baseline, reference, strict=True)):
            assert row["frame"] == old["frame"] == same["frame"] == frame
            assert row["association_authority_zero"] and row["identity_score_before_current_write"]
            assert all(row[k] == old[k] for k in ("outputs", "target_uid", "state_before", "state_after"))
            assert all(row[k] == same[k] for k in ("full_tracker_state_before_sha256", "full_tracker_state_after_sha256"))
            writes += bool(row["joint_identity_memory_write"])
            if row["joint_identity_memory_write"]:
                assert row["joint_memory_write_candidate_uid"] == row["target_uid"] and row["target_uid"] is not None
        assert writes == seal["counts"].get("actual_accepted_writes", 0)
    return seal


def checked_video(sequence, p, fits, ev):
    from scripts.n72r21r2_memory_frontier import checked_result
    source_path = OUT / "memory/risk_v1/runtime_sequences" / (sequence + ".json")
    result_path = OUT / "memory/risk_v1/results" / (sequence + ".json")
    source, result = ev.json(source_path), ev.json(result_path)
    assert result["whole_video_runtime_receipt_sha256"] == ev.digest(source_path)
    assert source["source_sha256"] == result["source_sha256"] == p["source_sha256"]
    assert source["protocol_sha256"] == result["protocol_sha256"] == ev.digest(policy.PROTOCOL)
    assert source["all6_heads_all_seeds_both_points"] == list(fits)
    init = ev.json(OUT / "data/initialization" / (sequence + ".json"))
    events = [e for e in init["inputs"] if not e["initialization_failure"]]
    assert set(result["identity_and_write_results"]) == expected_grid(events, list(fits))
    for case in fits:
        for event in events: checked_case(sequence, case, event, fits[case], p, ev)
    checked_result(sequence, "fresh_risk")  # Original exposure/provenance/all9 checks; offline truth only.
    ev.pinned(result["receipt"], result["all_nine_actual_metrics"],
        [case + "__click" + str(event["slot"]) for case in fits for event in events], sequence)
    return result


def evidence():
    marker_path = OUT / MARKER; marker = read_json(marker_path); dead_active(marker)
    no_live_scoped_workers(("scripts.n72r21r2_memory_risk_driver", "scripts.n72r21r2_memory_risk_runtime", "scripts.n72r21r2_memory_terminal_recovery_v1"))
    assert marker["active"]["identity"] == INTERRUPTED and marker["active"]["phase"] == "runtime"
    ev = Evidence(); p = ev.json(policy.PROTOCOL); ev.code(p["source_sha256"])
    assert marker["protocol_sha256"] == ev.digest(policy.PROTOCOL)
    assert p["split"] == preregistration()["split"]
    registered, fits = checked_fits(ev, p)
    assert sorted(marker["fit_completed"]) == sorted(set(case.rsplit("__", 1)[0] for case in fits))
    actual = sorted(path.stem for path in (OUT / "memory/risk_v1/results").glob("*.json"))
    assert len(marker["runtime_completed"]) == len(set(marker["runtime_completed"])) and sorted(marker["runtime_completed"]) == actual
    assert sorted(path.stem for path in (OUT / "memory/risk_v1/runtime_sequences").glob("*.json")) == actual
    for sequence in actual: checked_video(sequence, p, fits, ev)
    sequences = p["split"]["fit"] + p["split"]["inner"]
    assert set(actual) <= set(sequences) and INTERRUPTED not in actual
    init = ev.json(OUT / "data/initialization" / (INTERRUPTED + ".json"))
    ev.artifact({"path": init["anchor_path"], "sha256": init["anchor_sha256"]})
    events = {e["episode_uid"]: e for e in init["inputs"] if not e["initialization_failure"]}
    seals = sorted((OUT / "memory/risk_v1/runtime").rglob(INTERRUPTED + "*.json"))
    keys, links = [], []
    for path in seals:
        case, episode = path.parent.name, path.stem
        assert case in fits and episode in events
        seal = checked_case(INTERRUPTED, case, events[episode], fits[case], p, ev, raw=True)
        keys.append(case + "/" + episode)
        trajectory = next(ref for ref in seal["artifacts"] if ref["kind"] == "trajectory")
        links.append({"path": str(RECOVERY_ASSETS / "memory_risk_v1/trackers" / seal["tracker_name"] / "data" / (INTERRUPTED + ".txt")),
                      "target": trajectory["path"], "sha256": trajectory["sha256"]})
    grid = expected_grid(list(events.values()), list(fits))
    assert len(keys) == len(set(keys)) and set(keys) < grid
    # Do not route/retry a second unowned partial video without its own diagnosis.
    for sequence in sequences:
        if sequence not in actual and sequence != INTERRUPTED:
            assert not list((OUT / "memory/risk_v1/runtime").rglob(sequence + "*.json"))
            for folder in ("traces", "trackers"):
                assert not list((ASSETS / "memory_risk_v1" / folder).rglob(sequence + "*"))
    partials = [Path(marker["active"]["log"])]
    partials += list((ASSETS / "memory_risk_v1/traces").rglob(INTERRUPTED + "*.jsonl.zst"))
    partials += list((ASSETS / "memory_risk_v1/trackers").rglob(INTERRUPTED + ".txt"))
    return {"old_marker": marker, "old_marker_sha256": sha256(marker_path), "completed_evidence_sha256": ev.refs,
        "preserved_interrupted_files_sha256": {str(path): sha256(path) for path in partials},
        "registered_sequences": sequences, "sealed_interrupted_case_keys": keys,
        "missing_interrupted_case_keys": sorted(grid - set(keys)), "tracker_links": links,
        "raw_interrupted_full_frame_tensor_cases_checked": len(keys)}


def freeze():
    if PROTOCOL.exists(): raise FileExistsError("Preserve frozen memory recovery")
    storage(256 << 20); row = evidence()
    assert not RECOVERY_ASSETS.exists() and not (OUT / FRESH).exists()
    archive = write_json(PREFIX + "/original_marker.json", row["old_marker"])
    assert sha256(archive) == row["old_marker_sha256"]
    write_json(str(PROTOCOL.relative_to(OUT)), {"stage": "N72R21R2", "goal": GOAL, "frozen": True,
        "source_sha256": {name: sha256(ROOT / name) for name in CODE},
        "original_policy_protocol_sha256": sha256(policy.PROTOCOL), "original_marker_archive": str(archive), **row,
        "actual_old_exit_code_observed": False, "termination_cause": "UNKNOWN", "recovery_ASSETS": str(RECOVERY_ASSETS),
        "new_fit_or_scientific_policy_seed_point_weights_changed": False, "unsealed_partial_adoption": False,
        "association_authority": False, "confirmation_authorized": False, "next_stage_authorized": False})
    print({"sealed_full_videos": len(row["old_marker"]["runtime_completed"]), "sealed_interrupted_cases": len(row["sealed_interrupted_case_keys"]),
           "missing_cases": len(row["missing_interrupted_case_keys"]), "protocol_sha256": sha256(PROTOCOL)}, flush=True)


def frozen():
    p = read_json(PROTOCOL)
    assert p["frozen"] and p["goal"] == GOAL and p["source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
    assert p["original_policy_protocol_sha256"] == sha256(policy.PROTOCOL)
    assert sha256(p["original_marker_archive"]) == p["old_marker_sha256"]
    assert all(sha256(path) == digest for path, digest in p["preserved_interrupted_files_sha256"].items())
    return p


def tracker_view(links, *, create):
    for row in links:
        path, target = Path(row["path"]), Path(row["target"])
        assert path.is_relative_to(RECOVERY_ASSETS / "memory_risk_v1/trackers")
        assert target.resolve().is_relative_to(ASSETS.resolve()) and sha256(target) == row["sha256"]
        if create:
            if path.exists() or path.is_symlink(): raise FileExistsError("Never adopt/overwrite an unowned tracker view")
            path.parent.mkdir(parents=True, exist_ok=True)
            assert path.parent.resolve().is_relative_to(RECOVERY_ASSETS.resolve())
            path.symlink_to(target)
        assert path.is_symlink() and path.resolve() == target.resolve() and sha256(path) == row["sha256"]


@contextmanager
def recovery_paths():
    previous = policy.ASSETS
    try:
        policy.ASSETS = RECOVERY_ASSETS
        yield
    finally:
        policy.ASSETS = previous


def operation(action, sequence):
    p = frozen()
    if sequence != INTERRUPTED: raise ValueError("Only the diagnosed partial sequence routes assets")
    tracker_view(p["tracker_links"], create=False)
    with recovery_paths():
        if action == "runtime":
            with runtime_file_guard(): policy.runtime(sequence)
        else: policy.evaluate(sequence)


def run():
    p = frozen(); current = evidence()
    assert current == {key: p[key] for key in current}
    if (OUT / FRESH).exists(): raise FileExistsError("Never duplicate a memory recovery owner")
    state = {**p["old_marker"], "goal": GOAL, "pid": os.getpid(), "active": None,
        "status": "ACTIVE_ORIGINAL_ALL24_MEMORY_RUNTIME_NO_REFITTING", "runtime_completed": list(p["old_marker"]["runtime_completed"]),
        "recovery_protocol_sha256": sha256(PROTOCOL), "original_marker_archive": p["original_marker_archive"],
        "old_exit_not_observed_cause_UNKNOWN": True}
    def save():
        write_json(FRESH, state, mutable=True); write_json(MARKER, state, mutable=True)
    assert sha256(OUT / MARKER) == p["old_marker_sha256"]; dead_active(read_json(OUT / MARKER))
    # Establish exclusive fresh ownership before creating its scoped tracker view.
    write_json(FRESH, state); write_json(MARKER, state, mutable=True)
    tracker_view(p["tracker_links"], create=True)
    for sequence in p["registered_sequences"]:
        if sequence in state["runtime_completed"]: continue
        success = True
        for action, folder in (("runtime", "runtime_sequences"), ("evaluate", "results")):
            if (OUT / "memory/risk_v1" / folder / (sequence + ".json")).exists():
                raise FileExistsError("No implicit whole-sequence adoption after owner interruption")
            storage(64 << 20)
            log = ASSETS / "memory_terminal_recovery_v1_logs" / (sequence + "__" + action + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            module = "scripts.n72r21r2_memory_terminal_recovery_v1" if sequence == INTERRUPTED else "scripts.n72r21r2_memory_risk_runtime"
            with log.open("x") as handle:
                process = subprocess.Popen([str(PYTHON), "-u", "-m", module, action, "--sequence", sequence], cwd=ROOT,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=handle, stderr=subprocess.STDOUT)
                state["active"] = {"phase": action, "identity": sequence, "pid": process.pid, "log": str(log)}; save()
                code = process.wait()
            state["active"] = None
            if code:
                state["failed_retained"].append({"phase": action, "identity": sequence, "returncode": code, "log": str(log)})
                save(); success = False; break
            save()
        if success:
            ev = Evidence(); original = ev.json(policy.PROTOCOL); _, fits = checked_fits(ev, original)
            checked_video(sequence, original, fits, ev)
            state["runtime_completed"].append(sequence); save()
            print({"actual_memory_full_videos": len(state["runtime_completed"]), "sequence": sequence}, flush=True)
    state["status"] = "COMPLETE_ALL24_ACTUAL_CURRENT_RISK_MEMORY_G4_SUMMARY_PENDING" if not state["failed_retained"] else "ACTUAL_AVAILABLE_CASES_COMPLETE_FAILURES_RETAINED"
    save(); append_log("ACTUAL_MEMORY_OWNER_RECOVERY_FINISHED", videos=len(state["runtime_completed"]), failures=state["failed_retained"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("freeze", "run", "runtime", "evaluate"))
    parser.add_argument("--sequence"); args = parser.parse_args()
    if args.action == "freeze": freeze()
    elif args.action == "run": run()
    else: operation(args.action, args.sequence)
