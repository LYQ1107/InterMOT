"""Null-aware original tensor evidence; unchanged controlled-state protocol.

V2 CF's primitive future KEEP rows have a null before fingerprint, while
their after fingerprint is real. At t>source, the previous sealed after is
the exact before reference. Never compare a real hash to JSON null or call
missing pilot fingerprints an equality proof. No association changes.
"""
import argparse
from pathlib import Path
from scripts import n72r21r2_joint_state_curriculum as original
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, read_json, write_json, sha256, preregistration, development_sequence
from sam3_intermot.one_click.matched_event_observer import causal_bridge_fingerprint
from sam3_intermot.one_click.causal_state_fingerprint import full_tracker_fingerprint
from sam3_intermot.one_click.joint_intervention_primitives import prepare_proposal, commit_action
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl

PROTOCOL = OUT / "on_policy/JOINT_STATE_TENSOR_RECOVERY_PROTOCOL_V2.json"
CODE = ("scripts/n72r21r2_joint_state_tensor_v2.py", "scripts/n72r21r2_joint_state_tensor_recovery_driver_v2.py")
PREFIX = "on_policy/joint_state_tensor_v2"


def reference_checks(row, previous, before, after):
    direct = row.get("full_tracker_state_before_sha256")
    prior_after = None if previous is None else previous.get("full_tracker_state_after_sha256")
    expected_after = row.get("full_tracker_state_after_sha256")
    if direct is not None and before != direct:
        raise AssertionError("Actual source prestate differs from direct sealed tensor evidence")
    if prior_after is not None and before != prior_after:
        raise AssertionError("Actual source prestate differs from previous sealed after tensor evidence")
    if expected_after is not None and after != expected_after:
        raise AssertionError("Actual source poststate differs from sealed tensor evidence")
    return {"direct_before_checked": direct is not None, "previous_after_before_checked": prior_after is not None,
            "after_checked": expected_after is not None, "before_available": direct is not None or prior_after is not None,
            "null_before_never_counted_as_hash_equality": True}


def freeze():
    base = OUT / "on_policy/JOINT_STATE_SOURCE_PROTOCOL_V1.json"
    p = read_json(base)
    assert p["source_sha256"] == {name: sha256(ROOT / name) for name in original.CODE}
    write_json("on_policy/JOINT_STATE_TENSOR_RECOVERY_PROTOCOL_V2.json", {
        "stage": "N72R21R2", "goal": preregistration()["goal"], "goal_file": "outputs/N72R21R2/FINAL_GOAL.json", "frozen": True,
        "original_source_protocol_sha256": sha256(base), "original_source_sha256": p["source_sha256"],
        "wrapper_source_sha256": {name: sha256(ROOT / name) for name in CODE},
        "only_change": "Validate nonnull tensor evidence. Future prestate is independently checked against previous sealed poststate; every available poststate still checked. Source case/frame/action/anchor/offset/features/labels are unchanged.",
        "missing_pilot_tensor": "UNAVAILABLE, never zero-change or tensor equality proof",
        "namespace": PREFIX, "assets_namespace": "joint_state_tensor_recovery_v2", "CPU_workers": 1,
        "driver": "Only terminal failed V1 videos, and only after original V1 driver finishes all24 attempts, preserving CF max3-worker budget and preventing completed+failed duplicate counting. No original receipt/partial/log overwrite.",
        "canonicalization": "Previously missing runtime/supervision receipts explicitly link successful V2 and failed V1 evidence; successful V1 receipts never overwritten or reclassified",
        "model_generated_on_policy_or_scientific_success": False})


def route(relative):
    for suffix in ("runtime_sequences", "runtime", "supervision"):
        old = "on_policy/joint_state_v1/" + suffix
        if relative == old or relative.startswith(old + "/"):
            return relative.replace("on_policy/joint_state_v1", PREFIX, 1)
    return relative


class RoutedOutput(type(OUT)):
    def __truediv__(self, key):
        return Path(super().__truediv__(route(str(key))))


def collect_case(bridge, observer, frames, case, emit):
    source_frame, source_case = case["source_frame"], case["source_case"]
    assert sha256(case["source_trace"]["path"]) == case["source_trace"]["sha256"]
    reference = read_zstd_jsonl(Path(case["source_trace"]["path"]))
    assert sha256(case["source_KEEP_trace"]["path"]) == case["source_KEEP_trace"]["sha256"]
    keep_reference = read_zstd_jsonl(Path(case["source_KEEP_trace"]["path"]))
    parent, history = bridge.clone(), observer.clone()
    core_before, observer_before = causal_bridge_fingerprint(bridge), observer.state_fingerprint()
    trace, arms, actual_effects, tensor_checks = [], 0, [], 0
    proof_counts = {k: 0 for k in ("direct_before_checked", "previous_after_before_checked", "after_checked", "before_available")}
    for payload, rows in frames[source_frame:source_frame + max(original.OFFSETS) + 1]:
        f = int(payload["frame"])
        if f - source_frame in original.OFFSETS:
            arms += original.nested_arms(parent, history, frames, f, source_case, source_frame, keep_reference, emit)
        observation = history.observe(parent, f, rows)
        prepared = prepare_proposal(parent, f, rows)
        before = full_tracker_fingerprint(parent)
        index = f - source_frame
        expected = reference[index]
        result = commit_action(parent, f, rows, prepared, original.action(expected["selected_action"]))
        for key in ("outputs", "target_uid", "state_before", "state_after", "selected_action"):
            assert result[key] == expected[key], (source_case, f, key)
        checks = reference_checks(expected, reference[index - 1] if index else None, before, full_tracker_fingerprint(parent))
        for name in proof_counts:
            proof_counts[name] += int(checks[name])
        tensor_checks += checks["before_available"] and checks["after_checked"]
        if result["outputs"] != observation["own_KEEP_outputs"]:
            actual_effects.append(f)
        history.accept_commit(parent, f, observation, result)
        assert not result["joint_identity_memory_write"] and not parent.identity.bank
        trace.append(original.full_compact(result, parent, before))
    emit({"record_type": "SOURCE_PREFIX", "source_case": source_case, "source_frame": source_frame,
          "own_actual_source_prefix": trace, "effective_source_action_frames": actual_effects,
          "matched_original_source_tensor_frames": tensor_checks, "tensor_evidence_check_counts": proof_counts,
          "null_before_handled_without_weakening_available_evidence": True,
          "source_history_not_model_generated_on_policy": True, "actual_no_write_bank": True})
    assert core_before == causal_bridge_fingerprint(bridge) and observer_before == observer.state_fingerprint()
    return arms, len(actual_effects), tensor_checks


def run(sequence, operation):
    development_sequence(sequence)
    p = read_json(PROTOCOL)
    assert p["frozen"] and p["wrapper_source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
    assert p["original_source_sha256"] == {name: sha256(ROOT / name) for name in original.CODE}
    assert p["original_source_protocol_sha256"] == sha256(original.PROTOCOL)
    original.OUT, original.ASSETS = RoutedOutput(str(OUT)), ASSETS / "joint_state_tensor_recovery_v2"
    original.collect_case = collect_case

    def routed_write(relative, value, **kwargs):
        destination = route(relative)
        if destination != relative:
            value = {**value, "tensor_recovery_protocol_sha256": sha256(PROTOCOL),
                     "tensor_recovery_wrapper_source_sha256": p["wrapper_source_sha256"],
                     "original_failed_V1_attempt_NOT_reclassified": True}
        return write_json(destination, value, **kwargs)

    original.write_json = routed_write
    if operation == "runtime":
        with runtime_file_guard():
            original.runtime(sequence)
    elif operation == "label":
        original.label(sequence)
    else:
        raise ValueError("Only versioned runtime and separate offline labels")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "runtime", "label"))
    parser.add_argument("--sequence")
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run(args.sequence, args.action)
