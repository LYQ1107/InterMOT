#!/usr/bin/env python3
"""Register the N24 causal temporal dataset contract.

N21 already produced the real all-candidate SAM3 shadow feature cache used by
the project. N24 consumes that cache through a 20-step masked interface: the
first eight observed steps are retained and later positions are explicit
padding, while the dedicated N24 horizon-20 shadow subset supplies an
independent long-history evaluation set. This script writes a compact
manifest/statistics artifact rather than duplicating the multi-gigabyte NPZs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(".")


def npz_stats(path: Path):
    z = np.load(path)
    by_att = defaultdict(list)
    lengths = []
    for i, att in enumerate(z["att"]):
        key = str(att)
        by_att[key].append(i)
        lengths.append(int(z["vis_mask"][i].sum()))
    group_y = []
    for idxs in by_att.values():
        y = 0
        for i in idxs:
            if int(z["label"][i]):
                y = int(z["rank"][i])
                break
        group_y.append(y)
    return {
        "path": str(path), "candidate_rows": len(z["att"]),
        "groups": len(by_att), "complete_top5_groups": sum(
            len(v) == 5 for v in by_att.values()),
        "none_groups": int(sum(y == 0 for y in group_y)),
        "positive_group_label_counts": {
            str(k): int(v) for k, v in Counter(group_y).items()
        },
        "history_storage_shape": list(z["vis"].shape),
        "mean_valid_steps": float(np.asarray(lengths).mean()),
        "max_valid_steps": int(max(lengths) if lengths else 0),
        "feature_dim": int(z["vis"].shape[-1]),
        "label_policy": "offline final shadow-frame identity label; GT not in features",
    }


def shadow_stats(path: Path):
    rows = 0
    groups = set()
    lengths = []
    for fp in sorted(path.glob("*.jsonl")):
        with fp.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows += 1
                groups.add((row["sequence"], int(row["frame"]), int(row["gid"])))
                lengths.append(int(row.get("traj_len", 0)))
    return {
        "path": str(path), "candidate_rows": rows,
        "groups": len(groups), "mean_traj_len": float(np.mean(lengths)) if lengths else 0.0,
        "max_traj_len": int(max(lengths) if lengths else 0),
        "purpose": "sequence-disjoint cal10 long-history diagnostic; not used for training",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/n24/n24_temporal_dataset_manifest.json")
    ap.add_argument("--shadow20-dir", default="outputs/n20/n24_diag_shadow20")
    args = ap.parse_args()
    splits = {}
    for split in ("train30", "cal10"):
        splits[split] = npz_stats(
            ROOT / "outputs/n21/tracklet_identity_dataset" / f"{split}.npz")
    shadow_dir = Path(args.shadow20_dir)
    if not shadow_dir.is_absolute():
        shadow_dir = ROOT / shadow_dir
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "N24 causal temporal identity dataset",
        "interface": {"max_history": 20, "candidate_count": 5,
                       "feature": "GFN 2048 + R0 2048 per causal shadow frame",
                       "mask": "1 only when the frozen gallery matches a shadow box at IoU >= 0.5",
                       "root": "human-root GFN/R0 query; correction updates are applied only after a frame",
                       "gt_usage": "labels/post-hoc only; no GT box enters candidate construction or features"},
        "splits": splits,
        "shadow20": shadow_stats(shadow_dir) if shadow_dir.exists() else {"path": str(shadow_dir), "missing": True},
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print("N24_DATASET_MANIFEST_DONE", flush=True)


if __name__ == "__main__":
    main()
