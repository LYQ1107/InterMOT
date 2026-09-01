#!/usr/bin/env python
"""N9 observer runner (CPU) on frozen P0 backbone with ReID association."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from sam3_intermot.datasets.dancetrack import DanceTrackDataset
from sam3_intermot.interaction.continuous_observer import read_mot_rows
from sam3_intermot.interaction.n9_observer import N9Config, N9Observer
from sam3_intermot.n9.models import PairwiseMLP, SetAssociator


ROOT = Path(__file__).resolve().parents[1]
SEQ = os.environ["N9_SEQ"]
BUDGET = int(os.environ.get("N9_BUDGET", "0"))
VARIANT = os.environ.get("N9_VARIANT", "n8")
SPLIT = os.environ.get("N9_SPLIT", "val")
OUT_DIR = Path(os.environ.get("N9_OUT_DIR", ROOT / "outputs/n9" / "tmp"))
P0_SOURCE = Path(
    os.environ.get(
        "N9_P0_SOURCE",
        str(ROOT / "outputs/n9/p0_train")
        if SPLIT == "train"
        else str(ROOT / "outputs/n5/integrity/canonical_mot_results/b0"),
    )
)
FEAT_DIR = Path(os.environ.get("N9_FEAT_DIR", ROOT / "outputs/n9/features"))
MODEL_PATH = Path(os.environ.get("N9_MODEL_PATH", ""))
FRAME_LIMIT = int(os.environ.get("N9_FRAMES", "0")) or None


def export_mot(path: Path, rows_by_frame: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for f in sorted(rows_by_frame):
        for pid, box in sorted(rows_by_frame[f], key=lambda kv: kv[0]):
            x1, y1, x2, y2 = np.asarray(box, dtype=float)
            lines.append(
                f"{f+1},{pid},{x1:.2f},{y1:.2f},{max(0.0, x2-x1):.2f},"
                f"{max(0.0, y2-y1):.2f},1.000,-1,-1,-1"
            )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def load_model(variant: str):
    if variant == "n8" or variant == "reid":
        return None
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model not found: {MODEL_PATH}")
    if variant == "pairwise":
        model = PairwiseMLP()
    else:
        model = SetAssociator(d=256)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def main() -> None:
    dataset = DanceTrackDataset(
        str(Path("/path/to/dancetrack")),
        sequences=[SEQ],
        split=SPLIT,
    )
    num_frames = dataset.num_frames(SEQ)
    gt_frames = dataset.load_gt(SEQ)
    backbone: dict = {}
    all_rows = read_mot_rows(P0_SOURCE / f"{SEQ}.txt")
    for f in range(num_frames):
        backbone[f] = list(all_rows.get(f, []))
    if FRAME_LIMIT is not None:
        num_frames = min(num_frames, FRAME_LIMIT)
        gt_frames = {f: v for f, v in gt_frames.items() if f < num_frames}
        backbone = {f: v for f, v in backbone.items() if f < num_frames}
    feat_cache = {}
    npz = FEAT_DIR / f"{SEQ}.npz"
    if npz.exists():
        d = np.load(npz)
        for f, t, fv in zip(d["frame"], d["tid"], d["feat"]):
            if f < num_frames:
                feat_cache[(int(f), int(t))] = fv.astype(np.float32)
    cfg = N9Config(
        budget=BUDGET,
        sequence=SEQ,
        variant=VARIANT,
        model_path=str(MODEL_PATH) if MODEL_PATH.exists() else None,
        use_human_anchor=VARIANT in ("reid", "pairwise", "proposed"),
        use_negative_constraints=VARIANT in ("pairwise", "proposed"),
        relink_threshold=0.0 if VARIANT in ("pairwise", "auto", "proposed") else 0.35,
        min_similarity=0.35 if VARIANT == "reid" else 0.0,
    )
    model = load_model(VARIANT)
    obs = N9Observer(
        backbone,
        gt_frames,
        num_frames,
        cfg,
        sequence=SEQ,
        feat_cache=feat_cache,
        model=model,
    )
    obs.run()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if VARIANT == "n8" and BUDGET == 0:
        for sub in ("pre_mot", "post_mot"):
            dst = OUT_DIR / sub / f"{SEQ}.txt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = P0_SOURCE / f"{SEQ}.txt"
            if FRAME_LIMIT is None:
                shutil.copyfile(src, dst)
            else:
                dst.write_text(
                    "".join(
                        line
                        for line in src.read_text(encoding="utf-8").splitlines(keepends=True)
                        if line.strip() and int(line.split(",")[0]) <= FRAME_LIMIT
                    ),
                    encoding="utf-8",
                )
    else:
        export_mot(OUT_DIR / "pre_mot" / f"{SEQ}.txt", obs.pre_rows)
        export_mot(OUT_DIR / "post_mot" / f"{SEQ}.txt", obs.post_rows)
    write_jsonl(OUT_DIR / "verified_errors.jsonl", obs.verified_errors)
    write_jsonl(OUT_DIR / "interaction_events.jsonl", obs.interaction_events)
    write_jsonl(OUT_DIR / "relink_events.jsonl", obs.relink_events)
    write_jsonl(OUT_DIR / "system_state_hashes.jsonl", obs.state_hashes)
    by_type = {}
    for e in obs.verified_errors:
        by_type[e["event_type"]] = by_type.get(e["event_type"], 0) + 1
    summary = {
        "sequence": SEQ,
        "variant": VARIANT,
        "budget": BUDGET,
        "num_frames": num_frames,
        "accepted_count": obs.accepted_count,
        "events_by_type": by_type,
        "auto_relink_count": obs.auto_relink_count,
        "anchor_usage": obs.anchor_usage,
        "invariant_violations": obs.invariant_violations,
        "gt_audit": obs.gt_audit,
        "namespace_violations": obs.ns.violations(),
        "canonical_map_size": len(obs.canonical_map),
        "memories": len(obs.id_memory),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
