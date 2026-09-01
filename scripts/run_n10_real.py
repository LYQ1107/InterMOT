#!/usr/bin/env python
"""N10 online association runner (CPU): AUTO / HUMAN rollouts."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from sam3_intermot.association.human_intervention import HumanFeatureExtractor
from sam3_intermot.association.observation_tape import load_tape, tape_rows_by_frame
from sam3_intermot.association.online_associator import PairwiseMLP, SetAssociator
from sam3_intermot.association.rollout import N10Rollout
from sam3_intermot.association.state_manager import StateManagerConfig
from sam3_intermot.datasets.dancetrack import DanceTrackDataset


ROOT = Path(".")
DT = Path("/path/to/dancetrack")
SEQ = os.environ["N10_SEQ"]
SPLIT = os.environ.get("N10_SPLIT", "val")
BUDGET = int(os.environ.get("N10_BUDGET", "0"))
MODE = os.environ.get("N10_MODE", "auto")
APPLY = int(os.environ.get("N10_APPLY", "1"))
VARIANT = os.environ.get("N10_VARIANT", "pairwise")
MODEL_PATH = Path(os.environ.get("N10_MODEL_PATH", ROOT / "outputs/n10/models/n10_pairwise_mlp.pt"))
THRESHOLD = float(os.environ.get("N10_THRESHOLD", "0.0"))
MAX_LOST = int(os.environ.get("N10_MAX_LOST", "90"))
NATIVE_BONUS = float(os.environ.get("N10_NATIVE_BONUS", "3.0"))
OUT_DIR = Path(os.environ.get("N10_OUT_DIR", ROOT / "outputs/n10" / "real" / f"{VARIANT}_{MODE}_b{BUDGET}" / SEQ))
P0_SOURCE = (
    ROOT / "outputs/n9/p0_train" if SPLIT == "train" else ROOT / "outputs/n5/integrity/canonical_mot_results/b0"
)
CKPT = ROOT / "outputs/n9/checkpoints/osnet_x1_0_market1501.pth"


def export_mot(path: Path, rows_by_frame: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for f in sorted(rows_by_frame):
        for pid, box in sorted(rows_by_frame[f], key=lambda kv: kv[0]):
            x1, y1, x2, y2 = np.asarray(box, dtype=float)
            lines.append(
                f"{f+1},{int(pid)},{x1:.2f},{y1:.2f},{max(0.0, x2-x1):.2f},"
                f"{max(0.0, y2-y1):.2f},1.000,-1,-1,-1"
            )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def load_model(variant: str):
    if variant not in ("pairwise", "set"):
        return None
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model not found: {MODEL_PATH}")
    sd = torch.load(MODEL_PATH, map_location="cpu")
    if variant == "pairwise":
        model = PairwiseMLP()
    else:
        d = int(sd["layers.0.in_proj_bias"].shape[0]) // 3
        model = SetAssociator(d=d)
    model.load_state_dict(sd)
    model.eval()
    return model


def main() -> None:
    dataset = DanceTrackDataset(str(DT), sequences=[SEQ], split=SPLIT)
    num_frames = dataset.num_frames(SEQ)
    gt_frames = dataset.load_gt(SEQ)
    if VARIANT == "p0":
        src = P0_SOURCE / f"{SEQ}.txt"
        for sub in ("pre_mot", "post_mot"):
            dst = OUT_DIR / sub / f"{SEQ}.txt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        summary = {
            "sequence": SEQ,
            "variant": VARIANT,
            "mode": "p0",
            "budget": 0,
            "num_frames": num_frames,
            "copied": True,
        }
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        write_jsonl(OUT_DIR / "verified_errors.jsonl", [])
        write_jsonl(OUT_DIR / "interaction_events.jsonl", [])
        print(json.dumps(summary, ensure_ascii=False))
        return
    tape = load_tape(ROOT / "outputs/n10/tapes" / f"{SEQ}.npz")
    obs_by_frame = tape_rows_by_frame(tape)
    model = load_model(VARIANT)
    cfg = StateManagerConfig(
        score_threshold=THRESHOLD,
        max_lost_gap=MAX_LOST,
        native_bonus=NATIVE_BONUS,
        variant=VARIANT,
    )
    extractor = HumanFeatureExtractor(CKPT) if MODE == "human" else None
    rollout = N10Rollout(
        sequence=SEQ,
        num_frames=num_frames,
        gt_frames=gt_frames,
        model=model,
        manager_cfg=cfg,
        mode=MODE,
        budget=BUDGET,
        apply_interventions=bool(APPLY),
        seq_dir=DT / SPLIT / SEQ / "img1",
        feature_extractor=extractor,
    )
    rollout.run(obs_by_frame)
    export_mot(OUT_DIR / "pre_mot" / f"{SEQ}.txt", rollout.pre_rows)
    export_mot(OUT_DIR / "post_mot" / f"{SEQ}.txt", rollout.post_rows)
    write_jsonl(OUT_DIR / "verified_errors.jsonl", rollout.verified_errors)
    write_jsonl(OUT_DIR / "interaction_events.jsonl", rollout.interaction_events)
    write_jsonl(OUT_DIR / "intervention_log.jsonl", rollout.intervention_log)
    write_jsonl(OUT_DIR / "state_hashes.jsonl", rollout.state_hashes)
    summary = rollout.summary()
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
