"""Shared feature schema for the N19 memory-write policy (Writer V0)."""

import numpy as np

FEATURES = [
    "gfn_sim_human_root", "r0_sim_human_root",
    "gfn_sim_oracle_last", "gfn_sim_oracle_max", "r0_sim_oracle_max",
    "gfn_sim_heur_last", "gfn_sim_heur_max", "gfn_margin_h",
    "det_score", "box_area", "temporal_iou", "center_delta",
    "consecutive_delivered", "missing_streak", "crowd",
    "overlap_max", "nearest_det_distance", "heur_memory_age",
    "oracle_memory_age", "candidate_age", "slots_oracle_count",
    "slots_heur_count",
]
SOURCE_FLAGS = ["p0_tid", "p0", "react", "none"]


def feature_names():
    return FEATURES + [f"src_{s}" for s in SOURCE_FLAGS]


def to_feature_vec(r):
    x = []
    for k in FEATURES:
        v = r.get(k)
        x.append(0.0 if str(v) in ("", "nan", "None") else float(v))
    src = r.get("source", "")
    for s in SOURCE_FLAGS:
        x.append(1.0 if src == s else 0.0)
    return np.asarray(x, dtype=np.float32)
