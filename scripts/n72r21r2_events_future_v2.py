"""Version baseline auxiliary future windows to t+1..t+H, preserving V1.

V1 current-inclusive observations are archived as such, never CF rewards.
Event sampling/classes do not change. Actual CF action rewards remain pending.
"""
import argparse
from pathlib import Path
from scripts.n72r21r2_common import ROOT, OUT, TRAIN, sha256, read_json, write_json, development_sequence, append_log
from scripts.n72r21r2_baseline import checked_frames
from scripts.n72r21r2_events import artifact
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
from sam3_intermot.evaluation.causal_identity_windows import future_window
from sam3_intermot.evaluation.causal_identity_events import candidate_identity_outcome


def run(sequence):
    development_sequence(sequence)
    original_path = OUT / "events/corpus" / (sequence + ".json")
    corpus = read_json(original_path)
    frames, _ = checked_frames(sequence)
    truth = read_json(OUT / "data/initialization_truth" / (sequence + ".json"))
    assert sha256(TRAIN / sequence / "gt/gt.txt") == truth["GT_sha256"]
    identities = {r["episode_uid"]: r["target_gt_identity"] for r in truth["labels"]}
    gt = dancetrack_annotations(TRAIN / sequence)
    matching = {int(p["frame"]): strict_candidate_matching(rows, gt.get(int(p["frame"]), [])) for p, rows in frames}
    for episode in corpus["episodes"]:
        if not episode["events"] if "events" in episode else False:
            continue
        target = identities[episode["episode_uid"]]
        seal = read_json(OUT / "mot/baseline_runtime/CLICK_C0" / (episode["episode_uid"] + ".json"))
        trace = read_zstd_jsonl(artifact(seal, "trace"))
        observations = [{"frame": r["frame"], "outcome": candidate_identity_outcome(r["target_uid"], matching[r["frame"]], target)} for r in trace]
        for event in episode["event_probe_rows"]:
            labels = event["offline_supervision_labels"]
            labels["CURRENT_INCLUSIVE_BASELINE_WINDOWS_V1_ARCHIVE_NOT_CF_REWARD"] = labels.pop("actual_C0_future_observations_not_CF_action_rewards")
            future = {}
            for horizon in (1, 5, 20, 50, 100):
                window, complete = future_window(observations, event["frame"], horizon)
                future["H" + str(horizon)] = {"complete": complete, "observed_frames": len(window), "future_offsets": [1, horizon],
                                              "C0_correct": sum(r["outcome"] == "TARGET" for r in window),
                                              "C0_verified_OTHER": sum(r["outcome"] == "VERIFIED_OTHER" for r in window),
                                              "C0_UNKNOWN": sum(r["outcome"] == "UNKNOWN" for r in window)}
            labels["actual_C0_future_t_plus_1_through_H_not_CF_action_rewards"] = future
    corpus.update(original_V1_corpus_sha256=sha256(original_path), source_sha256=sha256(Path(__file__)),
                  future_boundary_definition="Current t supervised separately; future k=1..H inclusive, original frames",
                  window_utility_source_sha256=sha256(ROOT / "sam3_intermot/evaluation/causal_identity_windows.py"),
                  event_sampling_class_strata_unchanged=True, scientific_gate_unchanged=True)
    write_json("events/corpus_v2/" + sequence + ".json", corpus)
    append_log("M2_BASELINE_AUXILIARY_FUTURE_WINDOWS_V2_COMPLETE", sequence=sequence, action_CF_labels="PENDING")
    print({"sequence": sequence, "future_boundary": "t+1..t+H", "CF_action_labels": "PENDING"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    args = parser.parse_args()
    run(args.sequence)
