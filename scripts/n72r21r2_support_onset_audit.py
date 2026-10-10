"""All actual support-pilot action onsets, sealed before independent labels.

Own-policy prefix -> cloned same prestate -> own KEEP vs actual-now-then-KEEP.
Nonoverlapping windows are correlation controls, never independent G1 events.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, TRAIN, read_json, write_json, sha256, append_log
from scripts.n72r21r2_support_ablation_pilot import inputs, BASE, PROTOCOL as SOURCE_PROTOCOL, PREFIX as SOURCE_PREFIX
from scripts.n72r21r2_baseline import checked_frames, make_actor
from scripts.n72r21r2_event_context_v2 import sealed_click
from scripts.n72r21r2_simple_onset_audit import compact_full, save_rows
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.authority_support_ablation import SupportAblationBridge
from sam3_intermot.one_click.event_authority_runtime import EventAuthorityPredictor
from sam3_intermot.one_click.learned_onset_branches import onset_pair, chronological_nonoverlapping_candidates, complete_learned_state_fingerprint
from sam3_intermot.one_click.causal_state_fingerprint import full_tracker_fingerprint
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard

PROTOCOL = OUT / "training/SUPPORT_ONSET_AUDIT_PROTOCOL_V1.json"
PREFIX = "training/support_onset_v1"
CODE = ("scripts/n72r21r2_support_onset_audit.py", "scripts/n72r21r2_support_onset_driver.py",
        "sam3_intermot/one_click/learned_onset_branches.py", "scripts/n72r21r2_simple_onset_audit.py")


def freeze():
    p = read_json(SOURCE_PROTOCOL)
    write_json("training/SUPPORT_ONSET_AUDIT_PROTOCOL_V1.json", {
        "stage": "N72R21R2", "goal_file": "outputs/N72R21R2/FINAL_GOAL.json", "frozen": True,
        "support_policy_protocol_sha256": sha256(SOURCE_PROTOCOL), "sequences": p["sequences"], "seeds": p["seeds"], "policies": p["policies"],
        "source_sha256": {name: sha256(ROOT / name) for name in CODE},
        "runtime": "Audit EVERY actually effective original decision, exact own raw-anchor whole-video prefix, own cloned KEEP vs actual current then own KEEP k1..100. Full tracker and controller-history/pending clone equality, source core tensor equality, raw output/action equality; parent state never replaced.",
        "zero_actions": "Explicit NOT_RUN_NO_EFFECTIVE_ACTION, zero causal pairs and nonvacuity false; not a safety PASS",
        "correlation": "Also mark chronological nonoverlapping candidate windows, not proof of independent errors/roots. Never count all affected frames/seeds/windows as independent G1.",
        "labels": "Read GT only after whole cell runtime seal; target/current/future and every non-target harm/write component; original same-prestate public-origin proxy is not global IDF1",
        "resource": "One CPU worker; nonzero branch generation waits until all24 original CF sequences are sealed to preserve max3-CF-worker budget. Empty cases may be sealed immediately.",
        "not_confirmation_or_G1_G2_qualification": True})


def verify(sequence, seed, policy):
    p = read_json(PROTOCOL)
    assert p["source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
    assert p["support_policy_protocol_sha256"] == sha256(SOURCE_PROTOCOL)
    source, uid, fit = inputs(sequence, seed, policy)
    path = OUT / SOURCE_PREFIX / "runtime" / uid / (sequence + ".json")
    seal = read_json(path)
    assert seal["source_freeze"] == source["source_sha256"] and seal["protocol_sha256"] == sha256(SOURCE_PROTOCOL)
    assert all(sha256(a["path"]) == a["sha256"] for e in seal["episodes"] for a in e["artifacts"])
    return p, uid, fit, path, seal


def runtime(sequence, seed, policy):
    torch.set_num_threads(1)
    _, uid, fit, source_path, seal = verify(sequence, seed, policy)
    entries, frames = [], None
    for e in seal["episodes"]:
        ref = next(a for a in e["artifacts"] if a["kind"] == "trace")
        observed = read_zstd_jsonl(Path(ref["path"]))
        positions = [r["frame"] for r in observed if r["authority"].get("effective_assignment_change")]
        assert len(positions) == e["effective_direct_decisions_NOT_independent_onsets"]
        item = {"episode_uid": e["episode_uid"], "event": e["event"], "trace": ref,
                "all_effective_decision_frames": positions, "nonvacuity": bool(positions),
                "status": "ACTUAL_OWN_POLICY_ONSET_PAIRS" if positions else "NOT_RUN_NO_EFFECTIVE_ACTION", "branches": []}
        entries.append(item)
        if not positions:
            continue
        source = read_json(SOURCE_PROTOCOL)
        init = read_json(OUT / "data/initialization" / (sequence + ".json"))
        assert sha256(init["anchor_path"]) == init["anchor_sha256"]
        anchor = np.array(np.load(init["anchor_path"], mmap_mode="r")[e["event"]["anchor_index"]], np.float32)
        actor, _ = make_actor(e["event"], anchor)
        predictor = EventAuthorityPredictor(torch.load(fit["checkpoint_path"], map_location="cpu", weights_only=False), development_diagnostic=True)
        bridge = SupportAblationBridge(sealed_click(e["event"], anchor), actor, predictor=predictor,
                                      point=source["point"], frames=e["event"]["frames"], support_policy=policy)
        bridge.configure_fps(e["event"]["fps"])
        if frames is None:
            frames, _ = checked_frames(sequence)
        nonoverlapping = chronological_nonoverlapping_candidates(positions)
        for payload, candidates in frames:
            frame = int(payload["frame"])
            assert full_tracker_fingerprint(bridge) == observed[frame]["full_tracker_state_before_sha256"]
            if frame in positions:
                before = complete_learned_state_fingerprint(bridge)
                branches = onset_pair(bridge, frames, frame, observed[frame], compact_full)
                artifacts = {}
                for name, rows in branches.items():
                    path = ASSETS / "support_onsets_v1" / uid / e["episode_uid"] / ("frame" + str(frame)) / (name + ".jsonl.zst")
                    save_rows(path, rows)
                    artifacts[name] = {"path": str(path), "sha256": sha256(path), "frames": len(rows)}
                assert complete_learned_state_fingerprint(bridge) == before
                item["branches"].append({"frame": frame, "complete_learned_prestate_sha256": before,
                    "public_axis": sorted(bridge.tracker.states), "artifacts": artifacts,
                    "chronological_nonoverlapping_candidate_onset": frame in nonoverlapping})
            result = bridge.step(frame, candidates)
            assert result["outputs"] == observed[frame]["outputs"] and result["selected_action"] == observed[frame]["selected_action"]
            assert full_tracker_fingerprint(bridge) == observed[frame]["full_tracker_state_after_sha256"]
        item["actual_complete_controller_clone_and_own_prefix_core_tensor_AA"] = True
    write_json(PREFIX + "/runtime/" + uid + "/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "experiment_uid": uid, "protocol_sha256": sha256(PROTOCOL),
        "source_runtime_receipt_sha256": sha256(source_path), "source_sha256": {name: sha256(ROOT / name) for name in CODE},
        "entries": entries, "all_cell_runtime_sealed_before_GT": True, "actual_GT_file_guard": True,
        "all_initialization_failures_retained": seal["initialization_failures_not_replaced"], "not_G1_qualification": True})


def label(sequence, seed, policy):
    from sam3_intermot.one_click.datasets import dancetrack_annotations, strict_candidate_matching
    from sam3_intermot.evaluation.safe_intervention_events import assignment_map
    from sam3_intermot.evaluation.event_causal_labels import label_actual_branch
    _, uid, _, _, _ = verify(sequence, seed, policy)
    path = OUT / PREFIX / "runtime" / uid / (sequence + ".json")
    seal = read_json(path)
    assert seal["protocol_sha256"] == sha256(PROTOCOL)
    assert all(sha256(a["path"]) == a["sha256"] for e in seal["entries"] for b in e["branches"] for a in b["artifacts"].values())
    entries, gt, matched, targets = [], None, None, None
    for item in seal["entries"]:
        summaries = []
        if item["branches"]:
            if matched is None:
                frames, _ = checked_frames(sequence)
                gt = dancetrack_annotations(TRAIN / sequence)
                matched = {p["frame"]: strict_candidate_matching(rows, gt.get(p["frame"], [])) for p, rows in frames}
                targets = {r["episode_uid"]: r["target_gt_identity"] for r in read_json(OUT / "data/initialization_truth" / (sequence + ".json"))["labels"]}
            target = targets[item["episode_uid"]]
            observed = read_zstd_jsonl(Path(item["trace"]["path"]))
            origins, cursor = {}, 0
            for branch in item["branches"]:
                start = branch["frame"]
                while cursor < start:
                    for public, candidate in assignment_map(observed[cursor]).items():
                        identity = matched[cursor][candidate]
                        if identity is not None:
                            origins.setdefault(public, identity)
                    cursor += 1
                active = {public: identity for public, identity in origins.items() if public in branch["public_axis"]}
                rows = {name: read_zstd_jsonl(Path(a["path"])) for name, a in branch["artifacts"].items()}
                visible = {f: any(a["identity"] == target for a in gt.get(f, [])) for f in range(start, min(start + 101, len(frames)))}
                labels = label_actual_branch(rows["ACTUAL_CURRENT_THEN_KEEP"], rows["OWN_KEEP"], matched, target, active, visible)
                summaries.append({"frame": start, "chronological_nonoverlapping_candidate_onset": branch["chronological_nonoverlapping_candidate_onset"],
                    "labels": labels, "branch_seals": branch["artifacts"], "adaptive_future_not_used_as_one_shot_reward": True})
        entries.append({"episode_uid": item["episode_uid"], "effective_decisions": len(item["all_effective_decision_frames"]),
                        "nonvacuity": item["nonvacuity"], "status": item["status"], "one_shot_diagnostics": summaries})
    write_json(PREFIX + "/labels/" + uid + "/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "experiment_uid": uid, "protocol_sha256": sha256(PROTOCOL),
        "runtime_seal_sha256": sha256(path), "entries": entries, "zero_action_not_safety_PASS": True,
        "G1_independent_origins_NOT_inferred_from_windows_or_seeds": True, "full_policy_G1_qualified": False,
        "scientific_success": None})
    append_log("SUPPORT_PILOT_OWN_PREFIX_ONSET_LABELS", model=uid, sequence=sequence, actions=sum(e["effective_decisions"] for e in entries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "runtime", "label"))
    parser.add_argument("--sequence")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--policy")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "runtime":
        with runtime_file_guard():
            runtime(args.sequence, args.seed, args.policy)
    else:
        label(args.sequence, args.seed, args.policy)
