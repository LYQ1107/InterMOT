"""Matched raw-anchor FIT pilot: ablate hand filters, freeze existing models.

No additional fit, no best-seed choice, no retuned score/confirmation threshold,
and no confirmation permission. Evaluation uses all nine actual pinned metrics.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from scripts import n72r21r2_learned_pilot as legacy
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, read_json, write_json, sha256, storage, preregistration, append_log
from scripts.n72r21r2_baseline import checked_frames, make_actor
from scripts.n72r21r2_event_context_v2 import sealed_click
from scripts.n72r21r2_events import artifact
from scripts.n72r20r4_run_causal_tracker import trajectory_text
from sam3_intermot.one_click.authority_support_ablation import POLICIES, SupportAblationBridge
from sam3_intermot.one_click.event_authority_runtime import EventAuthorityPredictor
from sam3_intermot.one_click.causal_state_fingerprint import full_tracker_fingerprint, serial
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard

PREFIX = "training/support_ablation_v1"
PROTOCOL = OUT / "training/SUPPORT_ABLATION_PILOT_PROTOCOL_V1.json"
BASE = ASSETS / "support_ablation_v1/learned_pilot_v1"
FAMILY = "LOGISTIC_RISK"
CODE = tuple(dict.fromkeys((*legacy.CODE, "scripts/n72r21r2_support_ablation_pilot.py",
        "scripts/n72r21r2_support_ablation_driver.py", "sam3_intermot/one_click/authority_support_ablation.py",
        "scripts/n72r21r2_event_context_v2.py")))


def freeze():
    old = read_json(legacy.PROTOCOL)
    audit = read_json(OUT / "training/authority_support_v1/latest.json")
    assert sha256(audit["path"]) == audit["sha256"]
    support = read_json(audit["path"])
    assert support["role_totals"]["fit"]["actual_videos"] == 16
    assert support["role_totals"]["fit"]["hypothetical_confirmation_satisfied_ideal_model_pass_event_frames"] == 0
    # A fixed existing non-scalar linear model sees global causal features.
    # All three registered seeds are kept, without selecting pilot winners.
    models = []
    for seed in old["seeds"]:
        uid = legacy.identity(FAMILY, seed)
        fit_path = OUT / "training/event_authority" / (uid + ".json")
        fit = read_json(fit_path)
        assert fit["pilot_not_main_fit_or_independent_confirmation"] and sha256(fit["checkpoint_path"]) == fit["checkpoint_sha256"]
        models.append({"family": FAMILY, "seed": seed, "experiment_uid": uid,
                       "fit_record_sha256": sha256(fit_path), "checkpoint_sha256": fit["checkpoint_sha256"]})
    write_json("training/SUPPORT_ABLATION_PILOT_PROTOCOL_V1.json", {
        "stage": "N72R21R2", "goal_file": "outputs/N72R21R2/FINAL_GOAL.json", "frozen": True,
        "sequences": old["sequences"], "families": [FAMILY], "seeds": old["seeds"],
        "policies": list(POLICIES), "point": old["point"], "models": models,
        "existing_pilot_point_protocol_sha256": sha256(legacy.PROTOCOL),
        "FIT_only_support_trigger_snapshot": audit, "no_INNER_or_confirmation_used_for_trigger_or_pilot": True,
        "source_sha256": {p: sha256(ROOT / p) for p in CODE},
        "scope": "36 real own-full-video FIT pilot cells: 3 fixed seeds x4 named hand-filter variants x3 original FIT pilot videos. Same raw sealed click in all variants; old normalized-anchor evidence is not silently adopted as matched baseline.",
        "unchanged": "Identity model/memory/candidates/current feature vectors/learned weights/thresholds/branch ranking/current-history/full solver/hard negatives/feedback/pinned evaluation",
        "changed": "Only explicitly named authority hard-filter vetoes, in an independent process-local diagnostic wrapper",
        "no_GT_feature_or_force_candidate_or_target_only_trajectory_patch": True,
        "nonzero_actions_require_actual_own_prefix_harm_onset_audit_before_G1": True,
        "no_fits_or_scientific_success_or_next_stage_authorization": True,
        "CPU_workers": 1, "reserve_GiB": 60, "estimated_extra_assets_GiB_upper": .5})


def inputs(sequence, seed, policy):
    p = read_json(PROTOCOL)
    assert p["source_sha256"] == {name: sha256(ROOT / name) for name in CODE}
    if sequence not in p["sequences"] or seed not in p["seeds"] or policy not in p["policies"]:
        raise ValueError("Only frozen FIT support-ablation cells may run")
    assert set(p["sequences"]).issubset(preregistration()["split"]["fit"])
    model = next(m for m in p["models"] if m["seed"] == seed)
    fit_path = OUT / "training/event_authority" / (model["experiment_uid"] + ".json")
    assert sha256(fit_path) == model["fit_record_sha256"]
    fit = read_json(fit_path)
    assert sha256(fit["checkpoint_path"]) == model["checkpoint_sha256"]
    assert fit["source_freeze"] == {name: sha256(ROOT / name) for name in fit["source_freeze"]}
    return p, model["experiment_uid"] + "__" + policy, fit


def runtime(sequence, seed, policy):
    torch.set_num_threads(1)
    p, uid, fit = inputs(sequence, seed, policy)
    saved = torch.load(fit["checkpoint_path"], map_location="cpu", weights_only=False)
    predictor = EventAuthorityPredictor(saved, development_diagnostic=True)
    init = read_json(OUT / "data/initialization" / (sequence + ".json"))
    assert sha256(init["anchor_path"]) == init["anchor_sha256"]
    anchors = np.load(init["anchor_path"], mmap_mode="r")
    frames, index_sha = checked_frames(sequence)
    receipts = []
    storage(128 << 20)
    for event in init["inputs"]:
        if event["initialization_failure"]:
            continue
        anchor = np.array(anchors[event["anchor_index"]], np.float32)
        actor, _ = make_actor(event, anchor)
        bridge = SupportAblationBridge(sealed_click(event, anchor), actor, predictor=predictor,
                                      point=p["point"], frames=len(frames), support_policy=policy)
        bridge.configure_fps(event["fps"])
        name = uid + "__click" + str(event["slot"])
        trace_path = BASE / "traces" / uid / (event["episode_uid"] + ".jsonl.zst")
        tracker_path = BASE / "trackers" / name / "data" / (sequence + ".txt")
        for path in (trace_path, tracker_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError("Preserve original and partial support-ablation evidence")
        direct, writes, began = 0, 0, time.monotonic()
        with trace_path.open("xb") as handle, tracker_path.open("x") as tracker:
            proc = subprocess.Popen(["zstd", "-q", "-c", "-T1", "-3"], stdin=subprocess.PIPE, stdout=handle)
            try:
                for payload, rows in frames:
                    f = int(payload["frame"])
                    before = full_tracker_fingerprint(bridge)
                    result = bridge.step(f, rows)
                    result.update(full_tracker_state_before_sha256=before,
                                  full_tracker_state_after_sha256=full_tracker_fingerprint(bridge))
                    direct += bool(result["authority"].get("effective_assignment_change"))
                    writes += bool(result["joint_identity_memory_write"])
                    proc.stdin.write((json.dumps(serial(result), sort_keys=True, allow_nan=False) + "\n").encode())
                    tracker.write(trajectory_text([result]))
                proc.stdin.close()
                assert proc.wait() == 0
            finally:
                if proc.poll() is None:
                    proc.terminate(); proc.wait()
        assert writes == 0 and not actor.bank, "Authority diagnostic keeps P0 memory frozen"
        if direct == 0:
            c0 = read_json(OUT / "mot/baseline_runtime/CLICK_C0" / (event["episode_uid"] + ".json"))
            assert sha256(tracker_path) == sha256(artifact(c0, "trajectory"))
        receipts.append({"episode_uid": event["episode_uid"], "event": event, "tracker_name": name,
            "effective_direct_decisions_NOT_independent_onsets": direct, "memory_writes": writes,
            "seconds": time.monotonic() - began,
            "artifacts": [{"kind": k, "path": str(path), "sha256": sha256(path)}
                          for k, path in (("trace", trace_path), ("trajectory", tracker_path))]})
        print({"support_ablation_actual_episode": event["episode_uid"], "model_policy": uid,
               "effective_decisions_NOT_independent_onsets": direct}, flush=True)
    write_json(PREFIX + "/runtime/" + uid + "/" + sequence + ".json", {
        "stage": "N72R21R2", "sequence": sequence, "experiment_uid": uid,
        "status": "COMPLETE_ACTUAL_OWN_POLICY_UNQUALIFIED_FIT_SUPPORT_ABLATION",
        "checkpoint_sha256": fit["checkpoint_sha256"], "source_freeze": p["source_sha256"],
        "protocol_sha256": sha256(PROTOCOL), "candidate_index_sha256": index_sha,
        "episodes": receipts, "initialization_failures_not_replaced": sum(e["initialization_failure"] for e in init["inputs"]),
        "actual_GT_file_guard": True, "same_raw_sealed_click_all_variants": True,
        "policy_variant": policy, "pilot_not_main_fit_or_generalization": True, "scientific_success": None})


def route(relative):
    for old, new in (("training/pilot_policy_runtime", PREFIX + "/runtime"),
                     ("training/pilot_policy_results", PREFIX + "/results")):
        if relative == old or relative.startswith(old + "/"):
            return relative.replace(old, new, 1)
    return relative


class RoutedOutput(type(OUT)):
    def __truediv__(self, key):
        return Path(super().__truediv__(route(str(key))))


def evaluate(sequence, seed, policy):
    inputs(sequence, seed, policy)
    previous = {k: getattr(legacy, k) for k in ("OUT", "ASSETS", "PROTOCOL", "CODE", "inputs", "write_json")}
    def routed_write(relative, value, **kwargs):
        return write_json(route(relative), {**value, "policy_variant": policy,
                          "support_ablation_is_UNQUALIFIED_TRAIN_diagnostic": True}, **kwargs)
    try:
        legacy.OUT, legacy.ASSETS, legacy.PROTOCOL, legacy.CODE = RoutedOutput(str(OUT)), ASSETS / "support_ablation_v1", PROTOCOL, CODE
        legacy.inputs = lambda sequence, family, seed: inputs(sequence, seed, policy)
        legacy.write_json = routed_write
        legacy.evaluate(sequence, FAMILY, seed)
    finally:
        for name, value in previous.items():
            setattr(legacy, name, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "runtime", "evaluate"))
    parser.add_argument("--sequence")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--policy", choices=POLICIES)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "runtime":
        with runtime_file_guard():
            runtime(args.sequence, args.seed, args.policy)
    else:
        evaluate(args.sequence, args.seed, args.policy)
