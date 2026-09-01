#!/usr/bin/env python
"""N11 local human intervention runner (CPU)."""

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
SEQ = os.environ["N11_SEQ"]
SPLIT = os.environ.get("N11_SPLIT", "train")
BUDGET = int(os.environ.get("N11_BUDGET", "0"))
MODE = os.environ.get("N11_MODE", "auto")
APPLY = int(os.environ.get("N11_APPLY", "1"))
BASE = os.environ.get("N11_BASE", "pairwise")
MODEL_PATH = Path(os.environ.get("N11_MODEL_PATH", ROOT / "outputs/n10/models/n10_pairwise_mlp.pt"))
THRESHOLD = float(os.environ.get("N11_THRESHOLD", "-5.0" if BASE == "pairwise" else "1.8"))
MAX_LOST = int(os.environ.get("N11_MAX_LOST", "90"))
LOCAL = int(os.environ.get("N11_LOCAL", "0"))
SCOPE_FRAMES = int(os.environ.get("N11_SCOPE_FRAMES", "10"))
NATIVE_CF = os.environ.get("N11_NATIVE_CONSTRAINT_FRAMES", "")
AUTHORITY = os.environ.get("N11_AUTHORITY", "permanent")
AUTHORITY_HARD = int(os.environ.get("N11_AUTHORITY_HARD", "1"))
AUTHORITY_DECAY = int(os.environ.get("N11_AUTHORITY_DECAY", "8"))
FREEZE_MACHINE = int(os.environ.get("N11_FREEZE_MACHINE", "0"))
OUT_DIR = Path(os.environ.get("N11_OUT_DIR", ROOT / "outputs/n11" / "real" / f"{BASE}_{MODE}_b{BUDGET}" / SEQ))
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


def load_model(base: str):
    if base not in ("pairwise", "set"):
        return None
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model not found: {MODEL_PATH}")
    sd = torch.load(MODEL_PATH, map_location="cpu")
    if base == "pairwise":
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
    tape = load_tape(ROOT / "outputs/n10/tapes" / f"{SEQ}.npz")
    obs_by_frame = tape_rows_by_frame(tape)
    model = load_model(BASE)
    native_cf = None if NATIVE_CF == "" else int(NATIVE_CF)
    cfg = StateManagerConfig(
        score_threshold=THRESHOLD,
        max_lost_gap=MAX_LOST,
        variant=BASE,
        use_local_scope=bool(LOCAL),
        scope_frames=SCOPE_FRAMES,
        native_constraint_frames=native_cf,
        authority_mode=AUTHORITY,
        authority_hard_frames=AUTHORITY_HARD,
        authority_decay_frames=AUTHORITY_DECAY,
        freeze_machine_in_scope=bool(FREEZE_MACHINE),
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
    summary["config"] = {
        "base": BASE,
        "local": bool(LOCAL),
        "scope_frames": SCOPE_FRAMES,
        "native_constraint_frames": native_cf,
        "authority_mode": AUTHORITY,
        "authority_hard": AUTHORITY_HARD,
        "authority_decay": AUTHORITY_DECAY,
        "freeze_machine_in_scope": bool(FREEZE_MACHINE),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
