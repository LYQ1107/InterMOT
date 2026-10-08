"""Exact R4 score decomposition, including its float32 rounding order."""
from __future__ import annotations
import numpy as np
from .online_associator import native_same, native_scope_of, predicted_iou, score_matrix_pairwise


def observations(rows):
    return [{"feat": r["feature"], "box": np.asarray(r["box_xyxy"]),
             "native_tid": int(r["native_tid"]), "native_scope": r.get("native_scope"),
             "native_age": int(r.get("native_age", 0)), "has_feat": True} for r in rows]


def decompose_scores(states, rows, frame):
    obs = observations(rows); shape = (len(rows), len(states))
    parts = {k: np.zeros(shape, dtype=np.float64) for k in
             ("appearance", "predicted_iou", "native_core", "gap", "native_bonus", "positive_bonus")}
    hard = np.zeros(shape, dtype=bool); exact = np.zeros(shape, dtype=np.float32)
    for i, o in enumerate(obs):
        for j, s in enumerate(states):
            same = native_same(s, o)
            parts["appearance"][i,j] = 1.5 * float(np.dot(o["feat"], s.effective_feat()))
            parts["predicted_iou"][i,j] = predicted_iou(s, o["box"], frame)
            parts["native_core"][i,j] = .5 * same
            parts["gap"][i,j] = -.1 * min(1., max(0, frame-s.last_seen_frame)/200.)
            exact[i,j] = sum(parts[k][i,j] for k in ("appearance", "predicted_iou", "native_core", "gap"))
            if same:
                parts["native_bonus"][i,j] = 3.; exact[i,j] += 3.
            hard[i,j] = s.has_negative(o["native_tid"], frame, native_scope_of(o))
            if hard[i,j]: exact[i,j] = -1e9
            elif s.has_positive(o["native_tid"], frame, native_scope_of(o)):
                parts["positive_bonus"][i,j] = 5.; exact[i,j] += 5.
    reference = score_matrix_pairwise(states, obs, frame, None,
        reid_weights={"sim": 1.5, "iou": 1., "native": .5, "gap": .1}, native_bonus=3., positive_bonus=5.)
    if not np.array_equal(exact, reference): raise RuntimeError("R4 float32 decomposition is not exact")
    mathematical = sum(parts.values())
    parts["rounding_delta"] = np.where(hard, 0., exact.astype(float)-mathematical)
    parts["hard_negative_override"] = np.where(hard, -1e9-mathematical, 0.)
    parts["hard_mask"] = hard; parts["scores"] = exact.astype(np.float64)
    return parts
