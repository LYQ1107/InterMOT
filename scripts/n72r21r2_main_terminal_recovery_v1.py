"""Restore a proven dead MAIN owner without adopting its unsealed click files.

Original scientific modules remain frozen. Only the interrupted whole cell uses
an exclusive new BASE; all sealed cells, partial files and original logs remain.
The finite original driver/order/grid/three seeds and shared selection are used.
"""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
from scripts import n72r21r2_main_policy as policy
from scripts import n72r21r2_main_policy_driver as driver
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, PYTHON, GOAL, read_json, write_json, sha256, storage
from scripts.n72r21r2_idle_driver_recovery_v2 import routed_output
from scripts.n72r20r3r2r3_pipeline import parse_trackeval, trackeval_summary
from sam3_intermot.evaluation.learned_policy_evidence import METRICS
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard

PREFIX = "audit/main_terminal_recovery_v1"
PROTOCOL = OUT / "audit/MAIN_TERMINAL_RECOVERY_PROTOCOL_V1.json"
SELF = "scripts/n72r21r2_main_terminal_recovery_v1.py"
MARKER = policy.PREFIX + "/driver.json"
FRESH = PREFIX + "/owner.json"
INTERRUPTED = "MAIN__LOGISTIC_RISK__L3_HARM_AWARE__seed730103__P0_CLAIM80/dancetrack0016"
RECOVERY_BASE = ASSETS / "main_terminal_recovery_v1/main_policy_v1"


def dead_active(marker):
    active = marker.get("active")
    if not isinstance(active, dict) or marker.get("failed_retained"):
        raise RuntimeError("Only the explicitly diagnosed interrupted runtime is recoverable")
    for pid in (marker.get("pid"), active.get("pid")):
        if type(pid) is not int or pid <= 0 or Path("/proc", str(pid)).exists():
            raise RuntimeError("Live, reused or invalid owner/child PID forbids recovery")


def no_live_scoped_workers(modules):
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().decode().split("\0")
            cwd = (proc / "cwd").resolve()
        except (OSError, UnicodeError):
            continue
        if cwd == ROOT and any(module in argv for module in modules):
            raise RuntimeError("Another scoped live worker exists; never duplicate or terminate it")


class Evidence:
    def __init__(self):
        self.refs = {}
        self.values = {}

    def digest(self, path):
        key = str(path)
        if key not in self.refs:
            self.refs[key] = sha256(path)
        return self.refs[key]

    def json(self, path):
        key = str(path)
        self.digest(path)
        if key not in self.values:
            self.values[key] = read_json(path)
        return self.values[key]

    def artifact(self, ref):
        assert self.digest(ref["path"]) == ref["sha256"]

    def code(self, refs):
        assert all(self.digest(ROOT / name) == value for name, value in refs.items())

    def pinned(self, receipt, metrics, names, sequence):
        assert set(metrics) == set(names)
        if not names:
            assert receipt["status"] == "NOT_RUN_NO_VALID_INITIALIZATION"
            return
        assert receipt["returncode"] == 0
        assert self.digest(receipt["log_path"]) == receipt["log_sha256"]
        command = receipt["command"]
        assert command[command.index("--SPLIT_TO_EVAL") + 1] == "train"
        directory = Path(command[command.index("--OUTPUT_FOLDER") + 1])
        for name in names:
            raw = parse_trackeval(directory, name, [sequence])
            assert raw["csv_files"] and sequence in raw["per_sequence"]
            parsed = trackeval_summary(raw)
            assert all(parsed[k] == metrics[name][k] for k in METRICS)
            for path in raw["csv_files"]: self.digest(path)


def checked_cell(key, evidence):
    uid, sequence = key.split("/")
    base = OUT / policy.PREFIX
    runtime_path = base / "runtime" / uid / (sequence + ".json")
    onset_path = base / "onset_runtime" / uid / (sequence + ".json")
    result = evidence.json(base / "results" / uid / (sequence + ".json"))
    runtime, onset = evidence.json(runtime_path), evidence.json(onset_path)
    p = evidence.json(policy.PROTOCOL); evidence.code(p["source_sha256"])
    assert all(r["protocol_sha256"] == evidence.digest(policy.PROTOCOL) for r in (result, runtime, onset))
    assert runtime["source_freeze"] == p["source_sha256"]
    assert result["runtime_seal_sha256"] == onset["runtime_seal_sha256"] == evidence.digest(runtime_path)
    assert result["onset_runtime_seal_sha256"] == evidence.digest(onset_path)
    assert runtime["experiment_uid"] == result["experiment_uid"] == onset["experiment_uid"] == uid
    assert runtime["sequence"] == result["sequence"] == sequence
    job = {k: result[k] for k in ("family", "objective", "seed")}
    assert job in p["jobs"] and result["point"] in p["points"]
    assert uid == "MAIN__" + job["family"] + "__" + job["objective"] + "__seed" + str(job["seed"]) + "__" + result["point"]
    fit_path = OUT / "training/event_authority" / (uid.rsplit("__", 1)[0] + ".json")
    fit = evidence.json(fit_path); evidence.code(fit["source_freeze"])
    assert fit["nonzero_gradient_steps"] > 0 and sum(fit["changed_weight_elements_by_tensor"].values()) > 0
    assert not fit["pilot_not_main_fit_or_independent_confirmation"]
    assert runtime["fit_record_sha256"] == evidence.digest(fit_path)
    assert runtime["checkpoint_sha256"] == result["checkpoint_sha256"] == fit["checkpoint_sha256"] == evidence.digest(fit["checkpoint_path"])
    assert all(runtime[k] == result[k] == job[k] == fit[k] for k in job)
    assert runtime["actual_GT_file_guard"] and not runtime["runtime_future_GT_read"]
    assert runtime["sole_click_full_global_original_hard_negatives"] and onset["GT_guard_actual"]
    init_path = OUT / "data/initialization" / (sequence + ".json")
    init = evidence.json(init_path)
    assert runtime["initialization_sha256"] == evidence.digest(init_path)
    evidence.artifact({"path": init["anchor_path"], "sha256": init["anchor_sha256"]})
    integrity = evidence.json(OUT / "data/candidate_integrity" / (sequence + ".json"))
    assert runtime["candidate_index_sha256"] == init["candidate_index_sha256"] == integrity["index_sha256"] == evidence.digest(integrity["index_path"])
    valid = {e["episode_uid"]: e for e in init["inputs"] if not e["initialization_failure"]}
    assert {e["episode_uid"] for e in runtime["episodes"]} == {e["episode_uid"] for e in onset["entries"]} == set(valid)
    assert result["valid_initializations"] == len(valid) == len(runtime["episodes"])
    assert result["initialization_failures_not_replaced"] == sum(bool(e["initialization_failure"]) for e in init["inputs"])
    baseline = evidence.json(OUT / "mot/baseline_results" / (sequence + ".json"))["actual_nine_metrics"]
    for episode, entry in zip(runtime["episodes"], onset["entries"], strict=True):
        assert episode["episode_uid"] == entry["episode_uid"] and episode["event"] == valid[episode["episode_uid"]]
        assert episode["memory_writes"] == 0
        for ref in episode["artifacts"]: evidence.artifact(ref)
        for branch in entry["branches"]:
            assert set(branch["artifacts"]) == {"OWN_KEEP", "ACTUAL_CURRENT_THEN_KEEP"}
            for ref in branch["artifacts"].values(): evidence.artifact(ref)
        if episode["effective_direct_decisions_NOT_independent_onsets"] == 0:
            old = evidence.json(OUT / "mot/baseline_runtime/CLICK_C0" / (episode["episode_uid"] + ".json"))
            trajectory = next(a for a in episode["artifacts"] if a["kind"] == "trajectory")
            original = next(a for a in old["artifacts"] if a["kind"] == "trajectory")
            evidence.artifact(original); assert trajectory["sha256"] == original["sha256"]
            assert episode["zero_action_prefix_tensor_C0_AA_frames"] == episode["event"]["frames"]
            metrics = result["all_nine_metrics"][episode["tracker_name"]]
            reference = baseline["CLICK_C0__click" + str(episode["event"]["slot"])]
            assert all(metrics[k] == reference[k] for k in METRICS)
    evidence.pinned(result["actual_pinned_full_original_video_TrackEval"], result["all_nine_metrics"],
                    [e["tracker_name"] for e in runtime["episodes"]], sequence)
    if sequence in p["split"]["fit"]:
        selection = evidence.json(base / "selections" / (job["family"] + "__" + job["objective"] + ".json"))
        assert selection["selected_point"] == result["point"] and selection["protocol_sha256"] == evidence.digest(policy.PROTOCOL)
    return result


def evidence():
    marker_path = OUT / MARKER; marker = read_json(marker_path); dead_active(marker)
    no_live_scoped_workers(("scripts.n72r21r2_main_policy_driver", "scripts.n72r21r2_main_policy", "scripts.n72r21r2_main_terminal_recovery_v1"))
    assert marker["active"]["key"] == INTERRUPTED and marker["active"]["action"] == "runtime"
    assert marker["protocol_sha256"] == sha256(policy.PROTOCOL)
    ev = Evidence()
    actual = sorted(str(path.relative_to(OUT / policy.PREFIX / "results")).removesuffix(".json")
                    for path in (OUT / policy.PREFIX / "results").rglob("*.json"))
    assert len(marker["completed"]) == len(set(marker["completed"])) and sorted(marker["completed"]) == actual
    for key in actual: checked_cell(key, ev)
    assert subprocess.check_output(["git", "-C", str(ROOT.parent / "InterMOT/third_party/MOTIP/TrackEval"), "rev-parse", "HEAD"], text=True).strip() == "12c8791b303e0a0b50f753af204249e622d0281a"
    for group in marker["group_status"]:
        ev.json(OUT / policy.PREFIX / "selections" / (group + ".json"))
    uid, sequence = INTERRUPTED.split("/")
    for folder in ("runtime", "onset_runtime", "results"):
        assert not (OUT / policy.PREFIX / folder / uid / (sequence + ".json")).exists()
    partials = [Path(marker["active"]["log"])]
    partials += list((policy.BASE / "traces" / uid).glob(sequence + "*.jsonl.zst"))
    partials += list((policy.BASE / "trackers").glob(uid + "__click*/data/" + sequence + ".txt"))
    assert len(partials) > 1
    return {"old_marker": marker, "old_marker_sha256": sha256(marker_path),
            "completed_evidence_sha256": ev.refs, "preserved_interrupted_files_sha256": {str(path): sha256(path) for path in partials}}


def freeze():
    if PROTOCOL.exists(): raise FileExistsError("Preserve frozen MAIN recovery")
    storage(256 << 20); row = evidence()
    archive = write_json(PREFIX + "/original_marker.json", row["old_marker"])
    assert sha256(archive) == row["old_marker_sha256"]
    assert not RECOVERY_BASE.exists() and not (OUT / FRESH).exists()
    write_json(str(PROTOCOL.relative_to(OUT)), {"stage": "N72R21R2", "goal": GOAL, "frozen": True,
        "source_sha256": {SELF: sha256(ROOT / SELF)}, "original_policy_protocol_sha256": sha256(policy.PROTOCOL),
        "original_marker_archive": str(archive), **row, "interrupted_key": INTERRUPTED,
        "recovery_BASE": str(RECOVERY_BASE), "actual_old_exit_code_observed": False, "termination_cause": "UNKNOWN",
        "scientific_policy_seed_point_order_weights_changed": False, "unsealed_click_adoption": False,
        "confirmation_authorized": False, "next_stage_authorized": False})
    print({"proven_completed_MAIN_cells": len(row["old_marker"]["completed"]), "protocol_sha256": sha256(PROTOCOL)}, flush=True)


def frozen():
    p = read_json(PROTOCOL)
    assert p["frozen"] and p["goal"] == GOAL and p["source_sha256"] == {SELF: sha256(ROOT / SELF)}
    assert p["original_policy_protocol_sha256"] == sha256(policy.PROTOCOL)
    assert sha256(p["original_marker_archive"]) == p["old_marker_sha256"]
    assert all(sha256(path) == value for path, value in p["preserved_interrupted_files_sha256"].items())
    return p


@contextmanager
def recovery_paths():
    original = policy.BASE
    try:
        policy.BASE = RECOVERY_BASE
        yield
    finally:
        policy.BASE = original


def operation(action, sequence, family, objective, seed, point):
    frozen()
    key = "MAIN__" + family + "__" + objective + "__seed" + str(seed) + "__" + point + "/" + sequence
    if key != INTERRUPTED: raise ValueError("Only one registered interrupted cell routes BASE")
    with recovery_paths():
        if action == "evaluate": policy.evaluate(sequence, family, objective, seed, point)
        else:
            with runtime_file_guard():
                worker = policy.runtime if action == "runtime" else policy.onset_runtime
                worker(sequence, family, objective, seed, point)


def restored_state(value, old):
    if value["completed"] or value["group_status"] or value["failed_retained"]:
        raise RuntimeError("Only the original empty initial state may be hydrated")
    # Mutate the actual object used by the original driver, not merely its JSON.
    value.update(completed=list(old["completed"]), group_status=dict(old["group_status"]))
    return value


def run():
    p = frozen(); current = evidence()
    assert current == {k: p[k] for k in current}
    if (OUT / FRESH).exists(): raise FileExistsError("Never duplicate recovery ownership")
    saved = driver.OUT, driver.write_json, driver.run_cell
    original_cell = driver.run_cell
    def mirrored(relative, value, **kwargs):
        if relative != MARKER: return write_json(relative, value, **kwargs)
        if not (OUT / FRESH).exists():
            assert sha256(OUT / MARKER) == p["old_marker_sha256"]
            dead_active(read_json(OUT / MARKER)); restored_state(value, p["old_marker"])
        value.update(recovery_protocol_sha256=sha256(PROTOCOL), original_marker_archive=p["original_marker_archive"],
                     old_exit_not_observed_cause_UNKNOWN=True)
        path = write_json(FRESH, value, **kwargs); write_json(MARKER, value, mutable=True)
        return path
    def recovered_cell(state, marker, job, point, sequence):
        uid = "MAIN__" + job["family"] + "__" + job["objective"] + "__seed" + str(job["seed"]) + "__" + point
        key = uid + "/" + sequence
        if key != INTERRUPTED: return original_cell(state, marker, job, point, sequence)
        if key in state["completed"] or any(r["key"] == key for r in state["failed_retained"]): return
        log = ASSETS / "main_terminal_recovery_v1_logs" / uid / (sequence + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as handle:
            for action, folder in (("runtime", "runtime"), ("onsets", "onset_runtime"), ("evaluate", "results")):
                storage(32 << 20)
                if (OUT / policy.PREFIX / folder / uid / (sequence + ".json")).exists():
                    raise FileExistsError("Unowned whole-cell seal cannot be implicitly adopted")
                command = [str(PYTHON), "-u", "-m", "scripts.n72r21r2_main_terminal_recovery_v1", action,
                    "--sequence", sequence, "--family", job["family"], "--objective", job["objective"],
                    "--seed", str(job["seed"]), "--point", point]
                process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, stdout=handle, stderr=subprocess.STDOUT)
                state["active"] = {"key": key, "action": action, "pid": process.pid, "log": str(log)}
                mirrored(marker, state, mutable=True); code = process.wait()
                if code:
                    state["failed_retained"].append({"key": key, "action": action, "returncode": code, "log": str(log)})
                    break
            else:
                checked_cell(key, Evidence()); state["completed"].append(key)
        state["active"] = None; mirrored(marker, state, mutable=True)
    try:
        driver.OUT, driver.write_json, driver.run_cell = routed_output(MARKER, FRESH), mirrored, recovered_cell
        driver.run()
    finally:
        driver.OUT, driver.write_json, driver.run_cell = saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("freeze", "run", "runtime", "onsets", "evaluate"))
    for name in ("sequence", "family", "objective", "point"): parser.add_argument("--" + name)
    parser.add_argument("--seed", type=int); args = parser.parse_args()
    if args.action == "freeze": freeze()
    elif args.action == "run": run()
    else: operation(args.action, args.sequence, args.family, args.objective, args.seed, args.point)
