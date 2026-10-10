"""Recover two verified dead IDLE owners; original workers/code stay frozen.

No completed cells, failures or logs are removed. Only mutable ownership
metadata is mirrored after an exact original-marker archive and a fresh
dead-owner check. All original data paths, ordering, seeds and gates remain.
"""
import argparse
import importlib
from pathlib import Path
from scripts.n72r21r2_common import (ROOT, OUT, ASSETS, GOAL, read_json,
    write_json, sha256, storage, utcnow)

PROTOCOL = OUT / "audit/IDLE_DRIVER_TERMINAL_RECOVERY_PROTOCOL_V1.json"
KINDS = {
    "EXACT_CF": ("scripts.n72r21r2_exact_cf_verifier_driver", "availability/exact_cf_v1/driver.json",
                 "availability/EXACT_CF_VERIFIER_PROTOCOL_V1.json"),
    "STATE_POLICY": ("scripts.n72r21r2_state_policy_driver", "mot/state_policy_v1/driver.json",
                     "on_policy/STATE_POLICY_PROTOCOL_V1.json"),
}


def process_exists(pid):
    # Any extant process with this PID blocks recovery, even if reused/foreign.
    return type(pid) is int and pid > 0 and Path("/proc", str(pid)).exists()


def check_idle_dead(marker):
    pid = marker.get("pid")
    if (type(pid) is not int or pid <= 0 or process_exists(pid)
            or marker.get("active") is not None or marker.get("failed_retained")):
        raise RuntimeError("Recovery is only for confirmed missing IDLE owners with no failures/active child")


def evidence(kind):
    module, marker_name, policy_name = KINDS[kind]
    marker_path = OUT / marker_name
    marker = read_json(marker_path); check_idle_dead(marker)
    policy = read_json(OUT / policy_name)
    assert marker["protocol_sha256"] == sha256(OUT / policy_name)
    assert all(sha256(ROOT / p) == digest for p, digest in policy["source_sha256"].items())
    refs = []
    if kind == "EXACT_CF":
        complete = sorted(p.stem for p in (OUT / "availability/exact_cf_v1/results").glob("dancetrack*.json"))
        assert sorted(marker["completed"]) == complete
        assert sorted(p.stem for p in (OUT / "availability/exact_cf_v1/predictions").glob("dancetrack*.json")) == complete
        for sequence in complete:
            pred_path = OUT / "availability/exact_cf_v1/predictions" / (sequence + ".json")
            result_path = OUT / "availability/exact_cf_v1/results" / (sequence + ".json")
            pred, result = read_json(pred_path), read_json(result_path)
            assert pred["protocol_sha256"] == result["protocol_sha256"] == sha256(OUT / policy_name)
            assert sha256(pred["artifact"]["path"]) == pred["artifact"]["sha256"]
            assert result["score_receipt_sha256"] == sha256(pred_path)
            assert result["actual_CF_labels_sha256"] == sha256(OUT / "events/label_audit" / (sequence + ".json"))
            assert result["exact_axis_labels_sha256"] == sha256(OUT / "on_policy/joint_state_v1/supervision" / (sequence + ".json"))
            assert pred["source_sequence_sha256"] == sha256(OUT / "on_policy/joint_state_v1/runtime_sequences" / (sequence + ".json"))
            for path in (pred_path, result_path, Path(pred["artifact"]["path"]), ASSETS / "exact_cf_verifier_driver_v1_logs" / (sequence + ".log")):
                refs.append({"path": str(path), "sha256": sha256(path)})
        assert sorted(p.name.removesuffix(".jsonl.zst") for p in (ASSETS / "exact_cf_verifier_v1/predictions").glob("*.jsonl.zst")) == complete
        assert sorted(p.stem for p in (ASSETS / "exact_cf_verifier_driver_v1_logs").glob("*.log")) == complete
    else:
        assert marker["completed"] == [] and marker["group_status"] == {}
        assert marker["status"] == "WAITING_ALL24_CONTROLLED_SOURCE_AND_ALL3_REGISTERED_STATE_WEIGHTS"
        for folder in ("runtime", "onset_runtime", "results", "selections"):
            assert not any(path.is_file() for path in (OUT / "mot/state_policy_v1" / folder).rglob("*"))
        assert not list((ASSETS / "state_policy_v1_logs").rglob("*.log"))
        assert not any(path.is_file() for path in (ASSETS / "state_policy_v1").rglob("*"))
    return {"kind": kind, "module": module, "marker_name": marker_name,
        "old_marker_sha256": sha256(marker_path), "old_marker": marker,
        "policy_protocol": policy_name, "policy_protocol_sha256": sha256(OUT / policy_name),
        "existing_completed_evidence_preserved": refs,
        "actual_original_process_exit_code_observed": False, "termination_cause": "UNKNOWN",
        "dead_owner_confirmed_from_absent_proc_not_timeout": True}


def freeze():
    if PROTOCOL.exists():
        raise FileExistsError("Preserve the original recovery protocol")
    storage(32 << 20)
    rows = {kind: evidence(kind) for kind in KINDS}
    for kind, row in rows.items():
        name = "audit/idle_driver_terminal_recovery_v1/original_markers/" + kind + ".json"
        path = write_json(name, row["old_marker"])
        assert sha256(path) == row["old_marker_sha256"], "Exact original marker bytes must be preserved"
        row["original_marker_archive"] = {"path": str(path), "sha256": sha256(path)}
    write_json("audit/IDLE_DRIVER_TERMINAL_RECOVERY_PROTOCOL_V1.json", {
        "stage": "N72R21R2", "goal": GOAL, "utc": utcnow(), "frozen": True,
        "source_sha256": sha256(__file__), "owners": rows,
        "recovery": "Original unchanged run() after confirmed dead-idle preflight; new exclusive V2 ownership marker plus original mutable-marker mirror. Preserve all8 completed exact-CF results/score artifacts/logs, no existing state cells, no partial retry. Original workers, source, data paths, order, policies, weights, seeds, operating points and 60GiB storage floor unchanged.",
        "not_scientific_failure_or_new_experiment_or_on_policy_training": True,
        "next_stage_authorized": False})


def run(kind):
    p = read_json(PROTOCOL); row = p["owners"][kind]
    assert p["frozen"] and p["goal"] == GOAL and p["source_sha256"] == sha256(__file__)
    current = evidence(kind)
    for key in ("old_marker_sha256", "policy_protocol_sha256", "existing_completed_evidence_preserved"):
        assert current[key] == row[key]
    assert sha256(row["original_marker_archive"]["path"]) == row["old_marker_sha256"]
    new_marker = "audit/idle_driver_terminal_recovery_v1/owners/" + kind + ".json"
    if (OUT / new_marker).exists():
        raise FileExistsError("Inspect recovered owner before any duplicate launch")
    original = importlib.import_module(row["module"])
    # Only original marker existence check routes to the fresh namespace.
    # All scientific data/result paths remain byte-identical original paths.
    class RoutedOutput(type(OUT)):
        def __truediv__(self, key):
            return Path(super().__truediv__(new_marker if str(key) == row["marker_name"] else key))
    saved_out, saved_write = original.OUT, original.write_json
    def mirrored(relative, value, **kwargs):
        if relative != row["marker_name"]:
            return write_json(relative, value, **kwargs)
        updated = {**value, "goal": GOAL, "recovery_protocol_sha256": sha256(PROTOCOL),
                   "original_marker_archive": row["original_marker_archive"],
                   "old_exit_code_not_observed_cause_UNKNOWN": True}
        if not (OUT / new_marker).exists():
            assert sha256(OUT / relative) == row["old_marker_sha256"]
            check_idle_dead(read_json(OUT / relative))
        path = write_json(new_marker, updated, **kwargs)
        write_json(relative, updated, mutable=True)
        return path
    try:
        original.OUT, original.write_json = RoutedOutput(str(OUT)), mirrored
        original.run()
    finally:
        original.OUT, original.write_json = saved_out, saved_write


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--kind", choices=KINDS)
    args = parser.parse_args()
    if args.action == "run" and args.kind is None:
        parser.error("--kind is required for run")
    freeze() if args.action == "freeze" else run(args.kind)
