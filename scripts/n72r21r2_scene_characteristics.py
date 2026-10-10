"""M10 descriptive TRAIN-only original scene audit after runtime seals."""
import argparse
from pathlib import Path
import numpy as np
from scripts.n72r21r2_common import ROOT, OUT, TRAIN, GOAL, preregistration, read_json, write_json, sha256, append_log
from scripts.n72r21r2_baseline import checked_frames
from scripts.n72r21r2_density_tables import checked_density
from scripts.n72r21r2_events import artifact
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
from sam3_intermot.evaluation.scene_difficulty import describe_video, describe_episode

PROTOCOL = OUT / "data/SCENE_CHARACTERISTICS_PROTOCOL_V1.json"
CODE = ("scripts/n72r21r2_scene_characteristics.py", "sam3_intermot/evaluation/scene_difficulty.py",
        "scripts/n72r21r2_density_tables.py", "scripts/n72r21r2_baseline.py", "sam3_intermot/one_click/datasets.py")


def freeze():
    p = preregistration()
    write_json("data/SCENE_CHARACTERISTICS_PROTOCOL_V1.json", {"stage": "N72R21R2", "goal": GOAL,
        "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"), "frozen": True,
        "source_sha256": {name: sha256(ROOT / name) for name in CODE}, "development_split": {k: p["split"][k] for k in ("fit", "inner")},
        "scope": "Descriptive all24 FIT16/INNER8 only, original complete frozen candidates/GT axis and actual sealed C0 episodes. No new rollout, density relabeling, method/sequence selection, fitting or threshold tuning.",
        "occlusion": "Report actual GT visibility values and whether constant. GT-box intersection/min-area threshold0.5 is a fixed geometric overlap PROXY, never physical occlusion truth.",
        "ownership": "Separate positive assigned target public ID, assigned another public ID and UNASSIGNED; unassigned is not competitor-owned.",
        "reappearance": "Observed false-to-true return after earlier physical/strict-candidate presence. No independent causal-root claim.",
        "similarity": "Cosine of original sealed raw sole-human anchor and frozen current candidates. Hard competing identities are strict matched VERIFIED_OTHER; UNKNOWN is separate, not invented GT-negative identity. Diagnostic only, not a new identity representation goal or MOT gate.",
        "aggregation": "Per-video full-axis GT descriptors; per-click full postclick observations retained inside video, no frame/event independence claims. Frozen pre-effect candidate density remains unchanged.",
        "runtime_GT_access": False, "offline_GT_truth_only": True, "confirmation_authorized": False})
    print({"scene_protocol_sha256": sha256(PROTOCOL)}, flush=True)


def run():
    p, protocol = preregistration(), read_json(PROTOCOL)
    assert protocol["frozen"] and protocol["source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
    assert protocol["preregistration_sha256"] == sha256(OUT / "PREREGISTRATION.json")
    census, refs = [], []
    for sequence in p["split"]["fit"] + p["split"]["inner"]:
        role = "FIT" if sequence in p["split"]["fit"] else "INNER"
        init_path = OUT / "data/initialization" / (sequence + ".json")
        truth_path = OUT / "data/initialization_truth" / (sequence + ".json")
        base_result_path = OUT / "mot/baseline_results" / (sequence + ".json")
        init, truth, result = read_json(init_path), read_json(truth_path), read_json(base_result_path)
        density = checked_density(sequence)
        assert sha256(TRAIN / sequence / "gt/gt.txt") == truth["GT_sha256"]
        assert sha256(init["anchor_path"]) == init["anchor_sha256"]
        frames, index_sha = checked_frames(sequence)
        assert index_sha == init["candidate_index_sha256"] == result["candidate_index_sha256"]
        labels = {row["episode_uid"]: row["target_gt_identity"] for row in truth["labels"]}
        gt = dancetrack_annotations(TRAIN / sequence)  # OFFLINE ONLY: actual online C0 already sealed.
        matches = {f: strict_candidate_matching(rows, gt.get(f, [])) for f, (_, rows) in enumerate(frames)}
        anchors = np.load(init["anchor_path"], mmap_mode="r", allow_pickle=False)
        episodes = []
        for event in init["inputs"]:
            if event["initialization_failure"]:
                episodes.append({"episode_uid": event["episode_uid"], "status": "INITIALIZATION_FAILURE_RETAINED_NO_REPLACEMENT_NO_EPISODE_RESULT"})
                continue
            seal_path = OUT / "mot/baseline_runtime/CLICK_C0" / (event["episode_uid"] + ".json")
            seal = read_json(seal_path)
            assert seal["candidate_index_sha256"] == index_sha
            assert not seal["GT_runtime_read"] and seal["full_global_unique_ownership"]
            assert seal["source_sha256"] == sha256(ROOT / "scripts/n72r21r2_baseline.py")
            trace_path = artifact(seal, "trace")
            trace = read_zstd_jsonl(trace_path)
            record = describe_episode(frames, trace, matches, gt, anchors[event["anchor_index"]], event, labels[event["episode_uid"]])
            record["C0_runtime_seal_sha256"] = sha256(seal_path)
            episodes.append(record)
            refs.extend({"path": str(path), "sha256": sha256(path)} for path in (seal_path, trace_path))
        census.append({"sequence": sequence, "role": role, "original_frozen_candidate_density": density,
            "original_complete_video": describe_video(gt, len(frames)), "actual_sole_click_descriptors": episodes,
            "strict_matching_IOU": .5, "GT_sha256": truth["GT_sha256"]})
        refs.extend({"path": str(path), "sha256": sha256(path)} for path in (init_path, truth_path, base_result_path))
        print({"described_original_scene": sequence, "role": role, "valid_clicks": sum(e["status"].startswith("OFFLINE") for e in episodes)}, flush=True)
    write_json("data/SCENE_CHARACTERISTICS_OFFLINE_V1.json", {"stage": "N72R21R2", "goal": GOAL,
        "status": "COMPLETE_ALL24_OFFLINE_DESCRIPTIVE_ORIGINAL_SCENES_NOT_POLICY_CLOSURE",
        "protocol_sha256": sha256(PROTOCOL), "source_sha256": protocol["source_sha256"],
        "source_receipts": refs, "all24_census": census, "GT_only_after_C0_runtime_sealed": True,
        "new_runtime_rollouts_or_optimizer_steps": 0, "CONFIRM_VAL_TEST_SOT_accessed": False,
        "density_rule_changed": False, "occlusion_truth_NOT_inferred_from_box_overlap": True,
        "scientific_success": None, "next_association_stage_authorized": False})
    append_log("M10_ALL24_OFFLINE_ORIGINAL_SCENE_CHARACTERISTICS", protocol_sha256=sha256(PROTOCOL), videos=24, confirmation_authorized=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args(); freeze() if args.action == "freeze" else run()
