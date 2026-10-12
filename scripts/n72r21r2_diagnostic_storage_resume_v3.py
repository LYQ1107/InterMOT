"""Finish original exact-CF / pure-margin queues after verified storage stops.

Original scientific workers and all completed artifacts stay unchanged. Only
the previous storage-only score failure with NO partial artifact is retried;
its original log/marker remains sealed in the recovery preparation.
"""
import argparse
import importlib
import os
from pathlib import Path
from scripts import n72r21r2_idle_driver_recovery_v2 as prior
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, GOAL, read_json, write_json, sha256, storage
from scripts.n72r21r2_storage_guard_recovery_v1 import dead_idle_storage_owner, STORAGE_ERROR
from scripts.n72r21r2_main_terminal_recovery_v1 import Evidence, no_live_scoped_workers

SELF = "scripts/n72r21r2_diagnostic_storage_resume_v3.py"
KINDS = {k: prior.KINDS[k] for k in ("EXACT_CF", "PURE_MARGIN")}


def protocol_path(kind):
    return OUT / ("audit/DIAGNOSTIC_STORAGE_RESUME_V3_" + kind + ".json")


def fresh_marker(kind):
    return "audit/diagnostic_storage_resume_v3/owners/" + kind + ".json"


def same_kind_worker(argv, kind):
    if "scripts.n72r21r2_diagnostic_storage_resume_v3" not in argv:
        return False
    if "--kind" not in argv:
        return True
    index = argv.index("--kind") + 1
    return index >= len(argv) or argv[index] == kind


def no_partial_score(kind, sequence):
    if kind == "EXACT_CF":
        receipt_root = OUT / "availability/exact_cf_v1"
        artifact_root = ASSETS / "exact_cf_verifier_v1/predictions"
        folders = ("predictions", "results")
    else:
        receipt_root = OUT / "availability/pure_margin_v1"
        artifact_root = ASSETS / "open_set_pure_margin_v1/exact_CF"
        folders = ("CF_predictions", "CF_results")
    return (not any((receipt_root / folder / (sequence + ".json")).exists() for folder in folders)
            and not any(artifact_root.glob(sequence + "*")))


def evidence(kind):
    module, marker_name, policy_name = KINDS[kind]
    no_live_scoped_workers((module,))
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().decode().split("\0")
            cwd = (proc / "cwd").resolve()
        except (OSError, UnicodeError):
            continue
        if cwd == ROOT and same_kind_worker(argv, kind):
            raise RuntimeError("Live diagnostic owner of this kind must never be duplicated")
    old = read_json(OUT / marker_name)
    log = ASSETS / "idle_driver_terminal_recovery_v2_logs" / (kind + ".log")
    observation = dead_idle_storage_owner(old, log)
    ev = Evidence()
    policy_path = OUT / policy_name
    policy = ev.json(policy_path)
    ev.code(policy["source_sha256"])
    assert old["protocol_sha256"] == ev.digest(policy_path)
    ev.digest(OUT / marker_name)
    ev.digest(OUT / "audit/idle_driver_terminal_recovery_v2/owners" / (kind + ".json"))
    ev.digest(log)
    if kind == "PURE_MARGIN":
        assert not old["failed_retained"]
        current = prior.evidence(kind)
        for ref in current["existing_completed_evidence_preserved"]:
            ev.artifact(ref)
        assert len(old["completed"]) == 23
    else:
        assert old["failed_retained"] == [{"action": "score", "log": "/data3/liuyeqiang/InterMOT_N72R21R2_assets/exact_cf_verifier_driver_v1_logs/dancetrack0075.log", "returncode": 1, "sequence": "dancetrack0075"}]
        failure_log = Path(old["failed_retained"][0]["log"])
        text = failure_log.read_text()
        assert "Traceback (most recent call last):" in text and text.rstrip().endswith(STORAGE_ERROR)
        ev.digest(failure_log)
        assert no_partial_score(kind, "dancetrack0075")
        base = OUT / "availability/exact_cf_v1"
        complete = sorted(p.stem for p in (base / "results").glob("dancetrack*.json"))
        assert len(complete) == 22 and sorted(old["completed"]) == complete
        assert sorted(p.stem for p in (base / "predictions").glob("dancetrack*.json")) == complete
        for sequence in complete:
            pred_path, result_path = base / "predictions" / (sequence + ".json"), base / "results" / (sequence + ".json")
            pred, result = ev.json(pred_path), ev.json(result_path)
            assert pred["protocol_sha256"] == result["protocol_sha256"] == ev.digest(policy_path)
            assert result["score_receipt_sha256"] == ev.digest(pred_path)
            assert pred["source_sequence_sha256"] == ev.digest(OUT / "on_policy/joint_state_v1/runtime_sequences" / (sequence + ".json"))
            assert result["actual_CF_labels_sha256"] == ev.digest(OUT / "events/label_audit" / (sequence + ".json"))
            assert result["exact_axis_labels_sha256"] == ev.digest(OUT / "on_policy/joint_state_v1/supervision" / (sequence + ".json"))
            ev.artifact(pred["artifact"])
    return {"old_marker": old, "storage_stop": observation, "references": ev.refs,
            "retried_storage_failure_only_no_partial_adoption": kind == "EXACT_CF"}


def freeze(kind):
    protocol = protocol_path(kind)
    if protocol.exists() or (OUT / fresh_marker(kind)).exists():
        raise FileExistsError("Never overwrite preparation or duplicate owner")
    storage(32 << 20)
    row = evidence(kind)
    archive = write_json("audit/diagnostic_storage_resume_v3/old_markers/" + kind + ".json", row["old_marker"])
    assert sha256(archive) == sha256(OUT / KINDS[kind][1])
    write_json(str(protocol.relative_to(OUT)), {"stage": "N72R21R2", "goal": GOAL,
        "frozen": True, "kind": kind, "source_sha256": {SELF: sha256(ROOT / SELF)},
        **row, "old_marker_archive": str(archive), "CPU_workers": 1, "storage_floor_GiB": 60,
        "new_model_seed_data_point_or_association_authority": False})
    print({"kind": kind, "verified_completed": len(row["old_marker"]["completed"])}, flush=True)


def run(kind):
    p = read_json(protocol_path(kind))
    assert p["frozen"] and p["goal"] == GOAL and p["kind"] == kind
    assert p["source_sha256"] == {SELF: sha256(ROOT / SELF)}
    assert evidence(kind) == {k: p[k] for k in ("old_marker", "storage_stop", "references", "retried_storage_failure_only_no_partial_adoption")}
    fresh = fresh_marker(kind)
    if (OUT / fresh).exists():
        raise FileExistsError("Never duplicate recovered diagnostic ownership")
    original = importlib.import_module(KINDS[kind][0])
    marker = KINDS[kind][1]
    saved = original.OUT, original.write_json, original.ASSETS
    def mirrored(relative, value, **kwargs):
        if relative != marker:
            return write_json(relative, value, **kwargs)
        if not (OUT / fresh).exists():
            assert sha256(OUT / marker) == p["references"][str(OUT / marker)]
            dead_idle_storage_owner(read_json(OUT / marker), ASSETS / "idle_driver_terminal_recovery_v2_logs" / (kind + ".log"))
            value["completed"] = list(p["old_marker"]["completed"])
        value.update(recovery_protocol_sha256=sha256(protocol_path(kind)),
                     prior_storage_attempt_failures_retained=p["old_marker"]["failed_retained"],
                     old_marker_archive=p["old_marker_archive"])
        path = write_json(fresh, value, **kwargs)
        write_json(marker, value, mutable=True)
        return path
    try:
        original.OUT = prior.routed_output(marker, fresh)
        original.write_json, original.ASSETS = mirrored, ASSETS / "diagnostic_storage_resume_v3" / kind
        original.run()
    finally:
        original.OUT, original.write_json, original.ASSETS = saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--kind", choices=KINDS, required=True)
    args = parser.parse_args()
    (freeze if args.action == "freeze" else run)(args.kind)
