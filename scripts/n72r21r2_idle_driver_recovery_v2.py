"""Restore four confirmed dead idle owners, preserving every original receipt."""
import argparse
import importlib
from pathlib import Path
from scripts import n72r21r2_idle_driver_recovery_v1 as legacy
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, GOAL, read_json, write_json, sha256, storage

PROTOCOL = OUT / "audit/IDLE_DRIVER_TERMINAL_RECOVERY_PROTOCOL_V2.json"
KINDS = {**legacy.KINDS,
    "PURE_MARGIN": ("scripts.n72r21r2_pure_margin_driver", "availability/pure_margin_v1/driver.json", "availability/PURE_MARGIN_PROTOCOL_V1.json"),
    "STATE_FIT": ("scripts.n72r21r2_joint_state_fit_driver", "on_policy/state_source_fits_v1/driver.json", "on_policy/STATE_SOURCE_FITS_PROTOCOL_V1.json")}


def routed_output(marker, fresh):
    class Routed(type(OUT)):
        def __truediv__(self, key):
            return Path(super().__truediv__(fresh if str(key) == marker else key))
    return Routed(str(OUT))


def evidence(kind):
    if kind in legacy.KINDS:
        return legacy.evidence(kind)
    module, marker_name, policy_name = KINDS[kind]
    marker_path = OUT / marker_name; marker = read_json(marker_path)
    legacy.check_idle_dead(marker)
    policy_path = OUT / policy_name; policy = read_json(policy_path)
    assert marker["protocol_sha256"] == sha256(policy_path)
    assert all(sha256(ROOT / n) == h for n, h in policy["source_sha256"].items())
    refs = []
    if kind == "STATE_FIT":
        assert marker["status"] == "WAITING_ALL24_CONTROLLED_STATE_SOURCES" and marker["completed"] == []
        assert not list((OUT / "on_policy/state_source_fits_v1").glob("*STATE__seed*.json"))
        assert not list((ASSETS / "state_source_fit_driver_v1_logs").glob("*.log"))
    else:
        complete = sorted(p.stem for p in (OUT / "availability/pure_margin_v1/CF_results").glob("*.json"))
        assert sorted(marker["completed"]) == complete
        assert sorted(p.stem for p in (OUT / "availability/pure_margin_v1/CF_predictions").glob("*.json")) == complete
        for sequence in complete:
            pred_path = OUT / "availability/pure_margin_v1/CF_predictions" / (sequence + ".json")
            result_path = OUT / "availability/pure_margin_v1/CF_results" / (sequence + ".json")
            pred, result = read_json(pred_path), read_json(result_path)
            assert pred["protocol_sha256"] == result["protocol_sha256"] == sha256(policy_path)
            assert result["prediction_receipt_sha256"] == sha256(pred_path)
            assert result["CF_labels_receipt_sha256"] == sha256(OUT / "events/label_audit" / (sequence + ".json"))
            assert pred["source_sequence_sha256"] == sha256(OUT / "on_policy/joint_state_v1/runtime_sequences" / (sequence + ".json"))
            assert pred["selected_points_sha256"] == sha256(OUT / "availability/PURE_MARGIN_SELECTED_POINTS_V1.json")
            assert sha256(pred["artifact"]["path"]) == pred["artifact"]["sha256"]
            for path in (pred_path, result_path, Path(pred["artifact"]["path"]), ASSETS / "pure_margin_v1_logs" / (sequence + ".log")):
                refs.append({"path": str(path), "sha256": sha256(path)})
        assert sorted(p.name.removesuffix(".jsonl.zst") for p in (ASSETS / "open_set_pure_margin_v1/exact_CF").glob("*.jsonl.zst")) == complete
        assert sorted(p.stem for p in (ASSETS / "pure_margin_v1_logs").glob("*.log")) == complete
    return {"kind": kind, "module": module, "marker_name": marker_name,
        "old_marker_sha256": sha256(marker_path), "old_marker": marker,
        "policy_protocol": policy_name, "policy_protocol_sha256": sha256(policy_path),
        "existing_completed_evidence_preserved": refs,
        "actual_original_process_exit_code_observed": False, "termination_cause": "UNKNOWN",
        "dead_owner_confirmed_from_absent_proc_not_timeout": True}


def freeze():
    if PROTOCOL.exists(): raise FileExistsError("Preserve V2 recovery protocol")
    storage(32 << 20)
    rows = {kind: evidence(kind) for kind in KINDS}
    for kind, row in rows.items():
        path = write_json("audit/idle_driver_terminal_recovery_v2/original_markers/" + kind + ".json", row["old_marker"])
        assert sha256(path) == row["old_marker_sha256"]
        row["original_marker_archive"] = {"path": str(path), "sha256": sha256(path)}
    write_json(str(PROTOCOL.relative_to(OUT)), {"stage": "N72R21R2", "goal": GOAL, "frozen": True,
        "source_sha256": {n: sha256(ROOT / n) for n in ("scripts/n72r21r2_idle_driver_recovery_v1.py", "scripts/n72r21r2_idle_driver_recovery_v2.py")},
        "owners": rows, "recovery": "Original unchanged idle run loops, fresh exclusive ownership marker and archived canonical mirror. Exact/pure twelve completed sources retained; STATE fit/policy wait for ALL24 with zero existing fits/cells. No active/failed/partial recovery, new policy, seed, threshold or data path.",
        "actual_process_exit_code_or_cause_NOT_inferred": True, "next_stage_authorized": False})


def run(kind):
    p = read_json(PROTOCOL); row = p["owners"][kind]
    assert p["frozen"] and p["goal"] == GOAL
    assert all(sha256(ROOT / n) == h for n, h in p["source_sha256"].items())
    current = evidence(kind)
    for key in ("old_marker_sha256", "policy_protocol_sha256", "existing_completed_evidence_preserved"):
        assert current[key] == row[key]
    assert sha256(row["original_marker_archive"]["path"]) == row["old_marker_sha256"]
    fresh = "audit/idle_driver_terminal_recovery_v2/owners/" + kind + ".json"
    if (OUT / fresh).exists(): raise FileExistsError("Never start a duplicate V2 owner")
    original = importlib.import_module(row["module"])
    saved_out, saved_write = original.OUT, original.write_json
    def mirrored(relative, value, **kwargs):
        if relative != row["marker_name"]: return write_json(relative, value, **kwargs)
        updated = {**value, "goal": GOAL, "recovery_protocol_sha256": sha256(PROTOCOL),
            "original_marker_archive": row["original_marker_archive"], "old_exit_code_not_observed_cause_UNKNOWN": True}
        if not (OUT / fresh).exists():
            assert sha256(OUT / relative) == row["old_marker_sha256"]
            legacy.check_idle_dead(read_json(OUT / relative))
        path = write_json(fresh, updated, **kwargs)
        write_json(relative, updated, mutable=True)
        return path
    try:
        original.OUT, original.write_json = routed_output(row["marker_name"], fresh), mirrored
        original.run()
    finally:
        original.OUT, original.write_json = saved_out, saved_write


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("freeze", "run")); parser.add_argument("--kind", choices=KINDS)
    args = parser.parse_args()
    if args.action == "run" and args.kind is None: parser.error("--kind required")
    freeze() if args.action == "freeze" else run(args.kind)
