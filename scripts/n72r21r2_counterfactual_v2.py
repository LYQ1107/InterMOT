"""Unchanged CF actions with stronger tensor-inclusive state evidence.

V1 completed pilot tapes remain immutable. V2 is used for subsequent fresh
videos; it adds proof fields, not a different intervention or continuation.
"""
import argparse
from pathlib import Path
from scripts import n72r21r2_counterfactual as original
from scripts.n72r21r2_common import ROOT, write_json, sha256
from sam3_intermot.one_click.causal_state_fingerprint import full_tracker_fingerprint, fingerprint
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard


def run(sequence):
    compact_v1, commit_v1, collect_v1 = original.compact, original.commit_action, original.collect_event
    original.CODE += ["scripts/n72r21r2_counterfactual_v2.py", "sam3_intermot/one_click/causal_state_fingerprint.py"]

    def commit(bridge, *args):
        before = full_tracker_fingerprint(bridge)
        result = commit_v1(bridge, *args)
        result["full_tracker_state_before_sha256"] = before
        return result

    def compact(result, bridge):
        state = bridge.tracker.states[bridge.tracker.target_public]
        return compact_v1(result, bridge) | {
            "full_tracker_state_before_sha256": result.get("full_tracker_state_before_sha256"),
            "full_tracker_state_after_sha256": full_tracker_fingerprint(bridge),
            "target_motion_state": {"last_box": state.last_box.tolist(), "velocity": state.velocity.tolist(),
                                    "last_seen_frame": state.last_seen_frame},
            "target_prototype_sha256": fingerprint(state.prototype)}

    def collect(bridge, frames, frame, episode, *args):
        before = full_tracker_fingerprint(bridge)
        result = collect_v1(bridge, frames, frame, episode, *args)
        assert full_tracker_fingerprint(bridge) == before, "Future arm changed generating source tensor state"
        write_json("events/tensor_state_isolation_v2/" + episode + "/frame" + str(frame) + ".json", {
            "stage": "N72R21R2", "full_tensor_inclusive_source_prestate_sha256": before,
            "source_full_tensor_state_equal_after_all_future_arms": True, "source_sha256": sha256(Path(__file__)),
            "label_or_GT_used": False})
        return result

    original.commit_action, original.compact, original.collect_event = commit, compact, collect
    with runtime_file_guard():
        original.runtime(sequence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    run(parser.parse_args().sequence)
