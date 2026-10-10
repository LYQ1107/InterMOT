"""Identical labels after exact raw-anchor V2 context; retain failed V1 tape.

Only the verified context source and new output namespace change. Actual CF
arms, component definitions, GT, action/window sampling and TrackEval do not.
"""
import argparse
from scripts import n72r21r2_label_counterfactual as original
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, read_json, write_json, sha256, development_sequence


def verified(sequence):
    development_sequence(sequence)
    sequence_path = OUT / "events/counterfactual_sequences" / (sequence + ".json")
    sequence = read_json(sequence_path)
    assert sequence["source_C0_shadow_states_outputs_AA_after_all_future_branches"] and sequence["actual_forbidden_GT_file_guard"]
    assert all(sha256(ROOT / p) == h for p, h in sequence["source_freeze"].items())
    plan_path = OUT / "events/counterfactual_plans" / (sequence["sequence"] + ".json")
    assert sha256(plan_path) == sequence["plan_sha256"]
    context_sequence = read_json(OUT / "events/current_context_sequences_v2" / (sequence["sequence"] + ".json"))
    assert context_sequence["CF_sequence_seal_sha256"] == sha256(sequence_path)
    assert context_sequence["source_sha256"] == sha256(ROOT / "scripts/n72r21r2_event_context_v2.py")
    plan = read_json(plan_path)
    events = []
    for episode in plan["episodes"]:
        for frame in episode["frames"]:
            path = OUT / "events/counterfactual_seals" / episode["episode_uid"] / ("frame" + str(frame) + ".json")
            seal = read_json(path)
            assert seal["source_freeze"] == sequence["source_freeze"] and seal["plan_sha256"] == sha256(plan_path)
            assert seal["all_branches_same_starting_tracker_state"] and seal["future_state_from_own_branch_only"]
            assert all(sha256(a["path"]) == a["sha256"] for a in seal["artifacts"])
            context_path = OUT / "events/current_context_v2" / episode["episode_uid"] / ("frame" + str(frame) + ".json")
            context = read_json(context_path)
            assert context["CF_seal_sha256"] == sha256(path) and not context["runtime_GT_or_future_truth_used"]
            assert context["source_sha256"] == sha256(ROOT / "scripts/n72r21r2_event_context_v2.py")
            assert context["sealed_original_raw_anchor_used_without_extra_normalization"]
            events.append((path, seal, context_path, context))
    assert len(events) == sequence["actual_event_positions"] == context_sequence["actual_event_contexts"]
    return events, plan


def run(sequence):
    original.verified = verified
    original.ASSETS = ASSETS / "recovered_event_labels_v2"
    original_write = write_json

    def route(relative, value, **kwargs):
        if relative.startswith("events/label_audit/"):
            relative = relative.replace("events/label_audit/", "events/label_audit_v2/", 1)
            value = {**value, "wrapper_source_sha256": sha256(__file__),
                     "exact_context_v2_source_sha256": sha256(ROOT / "scripts/n72r21r2_event_context_v2.py"),
                     "original_failed_V1_evidence_preserved": True, "actual_CF_arms_and_label_definitions_unchanged": True}
        return original_write(relative, value, **kwargs)

    original.write_json = route
    original.run(sequence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    run(parser.parse_args().sequence)
