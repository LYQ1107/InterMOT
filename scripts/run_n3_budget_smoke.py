#!/usr/bin/env python
"""N3 three-sequence budget smoke (A0/A1/A2/A3) with official TrackEval."""

import json
import gc
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from sam3_intermot.backend.sam3_backend import Sam3Backend
from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.evaluation.mot_export import export_mot_file, validate_mot_file
from sam3_intermot.identity.lineage import IdentityLineageRegistry
from sam3_intermot.identity.registry import ObjectIdentityRegistry
from sam3_intermot.interaction.actions import ActionType
from sam3_intermot.interaction.simulator import SimulatedInteractionDriver, SimulatorConfig
from sam3_intermot.tracking.track_manager import TrackManager
from sam3_intermot.utils.io import atomic_write_json, write_csv


ROOT = Path(__file__).resolve().parents[1]
SEQS = os.environ.get("N3_SEQS", "dancetrack0004 dancetrack0005 dancetrack0007").split()
BUDGETS = [int(x) for x in os.environ.get("N3_BUDGETS", "0 1 2 5").split()]
N4_CONFIG = os.environ.get("N4_CONFIG", "")
OUT_ROOT = Path(os.environ.get("N4_OUT_DIR", ROOT / "outputs" / "n3_smoke"))


def _n4_settings():
    if N4_CONFIG == "R1_C":
        return {ActionType.CORRECT}, True, True, True, False, False, 0.0
    if N4_CONFIG == "R1_CD":
        return {ActionType.CORRECT, ActionType.DELETE}, True, True, True, False, False, 0.0
    if N4_CONFIG == "R1_CA":
        return {ActionType.CORRECT, ActionType.ADD}, True, True, True, False, False, 0.0
    if N4_CONFIG == "R1_CR":
        return {ActionType.CORRECT, ActionType.REASSIGN}, True, True, True, False, False, 0.0
    if N4_CONFIG == "R1_FULL_NOGUARD":
        return {ActionType.ADD, ActionType.CORRECT, ActionType.REASSIGN, ActionType.DELETE}, True, True, True, False, False, 0.0
    if N4_CONFIG == "R2_G1":
        return {ActionType.CORRECT, ActionType.ADD}, True, False, True, True, False, 0.4
    if N4_CONFIG == "R2_G2":
        return {ActionType.CORRECT, ActionType.ADD}, True, False, True, True, False, 0.2
    if N4_CONFIG == "R2_G3":
        return {ActionType.CORRECT, ActionType.ADD}, True, False, True, True, True, 0.4
    if N4_CONFIG == "R2_G4":
        return {ActionType.CORRECT, ActionType.ADD}, True, False, True, True, True, 0.2
    return {ActionType.ADD, ActionType.CORRECT, ActionType.REASSIGN, ActionType.DELETE}, True, True, True, True, True, 0.0


class RealAutoDriver(SimulatedInteractionDriver):
    def __init__(self, backend, manager, lineages, config, propagated, num_frames, registry):
        super().__init__(backend, manager, lineages, config)
        self.propagated = propagated
        self.num_frames = num_frames
        self.registry = registry

    def _automatic_step(self, frame_idx):
        observations = self.propagated.get(frame_idx, [])
        if frame_idx % 200 == 0:
            self.registry.unbind_all_for_window()
        for obs in observations:
            if obs.sam_object_id not in self.backend._objects:
                self.backend._objects[obs.sam_object_id] = {
                    "box": self.backend._sanitize_box(obs.box_xyxy).copy(),
                    "human_box": obs.box_xyxy.copy(),
                    "frame": frame_idx,
                    "source": "concept_detection",
                }
        seen = set()
        for obs in observations:
            if obs.sam_object_id in seen:
                continue
            seen.add(obs.sam_object_id)
            if (
                obs.sam_object_id in self.manager._tombstones
                and frame_idx - self.manager._tombstones[obs.sam_object_id]
                < self.manager.lifecycle.tombstone_cooldown_frames
            ):
                continue
            self.registry.register_auto_object(frame_idx, obs)
        matched = {o.sam_object_id for o in observations}
        for track in self.manager.active_tracks():
            if track.sam_object_id is not None and track.sam_object_id not in matched:
                self.manager.mark_missed(track.mot_track_id, frame_idx)


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    data_root = Path(cfg["dataset"]["root"])
    out = OUT_ROOT
    (out / "mot_results").mkdir(parents=True, exist_ok=True)
    dataset = DanceTrackDataset(str(data_root), sequences=SEQS, split=cfg["dataset"]["split"])
    all_summaries = {}
    for budget in BUDGETS:
        method = "b%d" % budget
        mot_dir = out / "mot_results" / method
        mot_dir.mkdir(parents=True, exist_ok=True)
        budget_summary = []
        for seq in SEQS:
            backend = Sam3Backend(
                checkpoint_path=cfg["backend"]["checkpoint_path"],
                max_num_objects=cfg["backend"]["max_num_objects"],
                multiplex_count=cfg["backend"]["multiplex_count"],
                use_fa3=False,
                use_rope_real=True,
                compile=False,
                warm_up=False,
                async_loading_frames=False,
            )
            manager = TrackManager()
            lineages = IdentityLineageRegistry()
            registry = ObjectIdentityRegistry(manager, lineages)
            try:
                video = data_root / cfg["dataset"]["split"] / seq / "img1"
                backend.start_video(str(video))
                num_frames = len(list(video.glob("*.jpg")))
                propagated = {}
                for start in range(0, num_frames, 200):
                    end = min(start + 199, num_frames - 1)
                    dets = backend.detect_concept(start, "person")
                    propagated[start] = list(dets)
                    prop = backend.propagate(
                        start, end, start_frame_index=start, keep_masks=False
                    )
                    for f, obs_list in prop.items():
                        propagated.setdefault(f, [])
                        propagated[f] = list(obs_list)
                    backend._output_cache.clear()
                    torch.cuda.empty_cache()
                gt = dataset.load_gt(seq)
                enabled_actions, lineage_add, soft_delete, atomic_reassign, abstention, guard, utility = _n4_settings()
                if budget == 0:
                    enabled_actions = set()
                config = SimulatorConfig(
                    enabled_actions=enabled_actions,
                    budget_per_100_frames=budget,
                    detection_interval=1000,
                    match_iou_threshold=0.1,
                    enable_lineage_aware_add=lineage_add,
                    enable_soft_delete=soft_delete,
                    enable_atomic_reassign=atomic_reassign,
                    enable_abstention=abstention,
                    enable_guard=guard,
                    utility_threshold=utility,
                )
                driver = RealAutoDriver(backend, manager, lineages, config, propagated, num_frames, registry)
                driver.ctx._next_sam_object_id = 1000
                summary = driver.run(gt, num_frames)
                outputs_by_frame = {
                    f: list(m.items())
                    for f, m in manager._outputs.items()
                    if m
                }
                mot_path = mot_dir / (seq + ".txt")
                export_mot_file(mot_path, outputs_by_frame)
                violations = validate_mot_file(mot_path, num_frames=num_frames)
                inv = manager.invariant_violations()
                by_type = {}
                for r in summary["results"]:
                    by_type.setdefault(r.action_type, {"accepted": 0, "rejected": 0, "rolled_back": 0})
                    by_type[r.action_type]["accepted" if r.accepted else "rejected"] += 1
                    if r.rolled_back:
                        by_type[r.action_type]["rolled_back"] += 1
                entry = {
                    "sequence": seq,
                    "budget": budget,
                    "num_rows": sum(len(v) for v in outputs_by_frame.values()),
                    "validation_violations": violations,
                    "invariant_violations": inv,
                    "by_type": by_type,
                    "leakage": vars(summary["leakage"]),
                }
                budget_summary.append(entry)
                with (out / ("events_%s_%s.jsonl" % (method, seq))).open(
                    "w", encoding="utf-8"
                ) as f:
                    for r in summary["results"]:
                        f.write(
                            json.dumps(
                                {
                                    "sequence": seq,
                                    "budget": budget,
                                    "action_id": r.action_id,
                                    "action_type": r.action_type,
                                    "frame_idx": r.frame_idx,
                                    "accepted": r.accepted,
                                    "rolled_back": r.rolled_back,
                                    "reason": r.reason,
                                    "new_track_id": r.new_track_id,
                                    "new_sam_object_id": r.new_sam_object_id,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                print(json.dumps(entry, ensure_ascii=False))
                with (out / ("summary_%s_%s.json" % (method, seq))).open(
                    "w", encoding="utf-8"
                ) as f:
                    json.dump(entry, f, indent=2, ensure_ascii=False)
            finally:
                backend.close()
                del backend
                gc.collect()
                torch.cuda.empty_cache()
        all_summaries[method] = budget_summary
    if os.environ.get("N3_SKIP_TRACKEVAL") != "1":
        _run_trackeval(out)


def _run_trackeval(out):
    gt = Path("/path/to/dancetrack/val")
    seqmap = out / "seqmap.txt"
    seqmap.write_text("name\n" + "\n".join(SEQS) + "\n", encoding="utf-8")
    trackers = sorted(
        p.name
        for p in (out / "mot_results").iterdir()
        if p.is_dir() and any(p.glob("*.txt"))
    )
    cmd = [
        "python",
        str(Path("./third_party/MOTIP/TrackEval/scripts/run_mot_challenge.py")),
        "--GT_FOLDER", str(gt),
        "--TRACKERS_FOLDER", str(out / "mot_results"),
        "--TRACKERS_TO_EVAL", *trackers,
        "--TRACKER_SUB_FOLDER", "",
        "--OUTPUT_SUB_FOLDER", "",
        "--SEQMAP_FILE", str(seqmap),
        "--BENCHMARK", "DanceTrack",
        "--SPLIT_TO_EVAL", "val",
        "--SKIP_SPLIT_FOL", "True",
        "--DO_PREPROC", "False",
        "--CLASSES_TO_EVAL", "pedestrian",
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--USE_PARALLEL", "False",
        "--PLOT_CURVES", "False",
        "--PRINT_RESULTS", "True",
        "--PRINT_ONLY_COMBINED", "False",
        "--OUTPUT_SUMMARY", "True",
        "--OUTPUT_DETAILED", "True",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out / "trackeval.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    print("N3_TRACKEVAL_RC", proc.returncode)
    print(proc.stdout[-5000:])


if __name__ == "__main__":
    main()
