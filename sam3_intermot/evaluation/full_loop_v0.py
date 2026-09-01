"""N18 FULL_LOOP_V0: causal TRACK -> LOST -> RECOVER -> VERIFY ->
REACTIVATE -> TRACK over one sequence.

Pure Python/numpy core; the GPU parts (GFN recovery, verifier, SAM3
reactivation) are injected as callables so this module stays testable.

Protocol:
- Human authority H_i is seeded from the GT box at the identity's first
  appearance (GT is used only to generate the human input at that frame).
- ACTIVE delivery follows frozen P0 AUTO rows by causal IoU/motion, or the
  stored SAM3 reactivation trajectory when available.
- Rule lost trigger: ``lost_streak`` consecutive frames with no delivered
  box (plus a weak-score streak for P0-sourced rows).
- LOST runs recovery (GFN top-1) and the deployed logistic verifier.
- ACCEPT reactivates SAM3 in an isolated session and binds the new internal
  trajectory to the same canonical public id; other identities are untouched.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


def iou(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size != 4 or b.size != 4:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-9)


@dataclass
class LoopConfig:
    lost_streak: int = 3
    retry_interval: int = 5
    follow_iou: float = 0.3
    bind_iou: float = 0.5
    weak_score: float = 0.5
    weak_streak_threshold: int = 3
    reactivation_horizon: int = 120
    verifier_threshold: float = 0.6
    use_two_frame: bool = False
    accept_threshold: float = 0.4
    confirm_threshold: float = 0.3
    confirm_iou: float = 0.5
    anchor_policy: str = "first"
    anchor_score: float = 0.5
    anchor_continuity_iou: float = 0.5
    memory_health_threshold: float = 0.5
    memory_matched_iou: float = 0.3
    react_stable_frames: int = 3
    memory_k: int = 2
    write_fn: Optional[Callable] = None
    enable_recovery: bool = True
    enable_reactivation: bool = True
    # N20 shadow-mode: speculative tracklets verified over several frames
    # before the public-ID commit. Off by default (N18/N19 behavior intact).
    shadow_mode: bool = False
    shadow_horizon: int = 8
    shadow_timeout: int = 30
    on_correction: Optional[Callable] = None
    # on_correction(seq, frame, gid, public_id, corrected_box|None):
    # called at the correction frame after the (wrong) output is observed.
    # corrected_box is the human-provided box (GT is read only to generate
    # this human input). May perform causal online updates; must not affect
    # the current frame.


@dataclass
class Identity:
    uid: int
    public_id: int
    gid: int
    anchor_box: np.ndarray
    anchor_frame: int
    trusted_box: Optional[np.ndarray] = None
    trusted_frame: int = -1
    bound_tid: Optional[int] = None
    state: str = "ACTIVE"
    last_box: Optional[np.ndarray] = None
    last_frame: int = -1
    missing_streak: int = 0
    weak_streak: int = 0
    lost_started: Optional[int] = None
    last_retry: int = -10**9
    react_traj: Dict[int, np.ndarray] = field(default_factory=dict)
    react_consecutive: int = 0
    pending_box: Optional[np.ndarray] = None
    pending_frame: int = -1
    recoveries: int = 0
    accepts: int = 0
    memory_slots: List[tuple] = field(default_factory=list)
    memory_emb: List[np.ndarray] = field(default_factory=list)
    heur_slots: List[tuple] = field(default_factory=list)
    delivered_streak: int = 0
    prev_delivered_box: Optional[np.ndarray] = None


@dataclass
class P0Row:
    tid: int
    box: np.ndarray
    score: float


def run_full_loop(
    seq: str,
    gt_frames: Dict[int, object],      # frame -> GTFrame(boxes, gt_ids)
    p0_rows: Dict[int, List[P0Row]],
    num_frames: int,
    cfg: LoopConfig,
    recovery_fn: Callable[[str, int, np.ndarray, int], Optional[dict]],
    verifier_fn: Callable[[dict], float],
    reactivate_fn: Callable[[str, int, np.ndarray], Dict[int, np.ndarray]],
    health_fn: Optional[Callable[[str, int, int, np.ndarray],
                                 Optional[float]]] = None,
    shadow_start_fn: Optional[Callable[[str, int, int, np.ndarray, int],
                                       Optional[str]]] = None,
    shadow_step_fn: Optional[Callable[[str, int, int, str, int],
                                      Optional[dict]]] = None,
) -> dict:
    """Run the loop; returns the full trace and metrics."""
    # ---- seed human identities at first appearance (GT = human input only)
    identities: Dict[int, Identity] = {}
    first_appearance: Dict[int, int] = {}
    for f in range(num_frames):
        gf = gt_frames.get(f)
        if gf is None:
            continue
        for gid, box in zip(gf.gt_ids, gf.boxes):
            if gid not in first_appearance:
                first_appearance[gid] = f
    for gid, f in first_appearance.items():
        gf = gt_frames[f]
        box = np.asarray(gf.boxes[gf.gt_ids.index(gid)], dtype=float)
        uid = gid + 1
        ident = Identity(
            uid=uid, public_id=1000 + gid, gid=gid, anchor_box=box.copy(),
            anchor_frame=f, last_box=box.copy(), last_frame=f,
            trusted_box=box.copy(), trusted_frame=f)
        if cfg.anchor_policy == "learned":
            # Human root is the authoritative initial memory slot
            ident.memory_slots.append((f, box.copy()))
        # bind to the frozen P0 track that best matches the human anchor
        bound_row, bound_iou = None, cfg.bind_iou
        for row in p0_rows.get(f, []):
            v = iou(row.box, box)
            if v >= bound_iou:
                bound_iou, bound_row = v, row
        if bound_row is not None:
            ident.bound_tid = bound_row.tid
        identities[gid] = ident

    trace: List[dict] = []
    transactions: List[dict] = []
    lost_episodes: List[dict] = []
    # per-frame delivered output, public_id -> box
    delivered_history: List[Dict[int, np.ndarray]] = []
    pending_shadows: Dict[tuple, dict] = {}
    for f in range(num_frames):
        gf = gt_frames.get(f)
        rows = p0_rows.get(f, [])
        frame_delivered: Dict[int, Optional[np.ndarray]] = {}
        frame_source: Dict[int, str] = {}
        frame_delivery_score: Dict[int, Optional[float]] = {}
        frame_prev_iou: Dict[int, Optional[float]] = {}
        # ACTIVE delivery: follow the bound P0 track id first (track
        # continuity from the frozen AUTO output), then bridge short gaps
        # with a one-to-one greedy IoU match. A single P0 row can never be
        # claimed by two public identities.
        rows_by_tid: Dict[int, List[tuple]] = {}
        for r, row in enumerate(rows):
            rows_by_tid.setdefault(row.tid, []).append((r, row))
        assigned_row: set = set()
        tid_assign: Dict[int, P0Row] = {}
        tid_claimants: Dict[int, List[int]] = {}
        for gid, ident in identities.items():
            if ident.react_traj.get(f) is not None:
                continue
            if ident.bound_tid is not None and ident.bound_tid in rows_by_tid:
                tid_claimants.setdefault(ident.bound_tid, []).append(gid)
        for tid, gids in tid_claimants.items():
            row_idx, row = max(rows_by_tid[tid],
                               key=lambda e: e[1].score)
            if len(gids) == 1:
                tid_assign[gids[0]] = row
            else:
                winner = max(gids, key=lambda g:
                             iou(row.box, identities[g].last_box))
                tid_assign[winner] = row
            assigned_row.add(row_idx)
        match_candidates = []
        for gid, ident in identities.items():
            if ident.react_traj.get(f) is not None or gid in tid_assign:
                continue
            for r, row in enumerate(rows):
                if r in assigned_row:
                    continue
                v = iou(row.box, ident.last_box)
                if v >= cfg.follow_iou:
                    match_candidates.append((v, gid, r))
        match_candidates.sort(key=lambda t: -t[0])
        assign: Dict[int, P0Row] = {}
        for v, gid, r in match_candidates:
            if gid in tid_assign or gid in assign or r in assigned_row:
                continue
            assigned_row.add(r)
            assign[gid] = rows[r]
            if v >= cfg.bind_iou:
                identities[gid].bound_tid = rows[r].tid

        for gid, ident in identities.items():
            delivered = None
            source = "none"
            delivery_score = None
            if f in ident.react_traj and ident.react_traj[f] is not None:
                delivered = ident.react_traj[f]
                source = "react"
            elif gid in tid_assign:
                delivered = tid_assign[gid].box.copy()
                source = "p0_tid"
                delivery_score = tid_assign[gid].score
            elif gid in assign:
                row = assign[gid]
                delivered = row.box.copy()
                source = "p0"
                delivery_score = row.score
                ident.weak_streak = ident.weak_streak + 1 \
                    if row.score < cfg.weak_score else 0
            else:
                ident.weak_streak += 1
            if delivered is not None:
                prev_box = ident.last_box
                ident.prev_delivered_box = prev_box
                ident.last_box = delivered.copy()
                ident.last_frame = f
                ident.missing_streak = 0
                ident.delivered_streak += 1
                frame_prev_iou[ident.public_id] = iou(delivered, prev_box) \
                    if prev_box is not None else None
                frame_delivery_score[ident.public_id] = delivery_score
                if source == "react":
                    ident.react_consecutive += 1
                else:
                    ident.react_consecutive = 0
                if cfg.anchor_policy == "trusted":
                    if source in ("p0_tid", "p0") and delivery_score is not None \
                            and delivery_score >= cfg.anchor_score \
                            and iou(delivered, ident.trusted_box) >= \
                            cfg.anchor_continuity_iou:
                        ident.trusted_box = delivered.copy()
                        ident.trusted_frame = f
                    elif source == "react" and \
                            ident.react_consecutive >= cfg.react_stable_frames \
                            and prev_box is not None \
                            and iou(delivered, prev_box) >= \
                            cfg.anchor_continuity_iou:
                        ident.trusted_box = delivered.copy()
                        ident.trusted_frame = f
                elif cfg.anchor_policy == "verified" and \
                        source in ("p0_tid", "p0"):
                    health = health_fn(seq, f, gid, delivered) \
                        if health_fn is not None else None
                    if health is not None and \
                            health >= cfg.memory_health_threshold and \
                            prev_box is not None and \
                            iou(delivered, prev_box) >= \
                            cfg.anchor_continuity_iou:
                        ident.trusted_box = delivered.copy()
                        ident.trusted_frame = f
                elif cfg.anchor_policy == "oracle":
                    # offline upper-bound diagnostic: write when the current
                    # delivered observation is GT-correct (causal, <= t)
                    tgt = None
                    if gf is not None and gid in gf.gt_ids:
                        tgt = np.asarray(
                            gf.boxes[gf.gt_ids.index(gid)], dtype=float)
                    if tgt is not None and \
                            iou(delivered, tgt) >= 0.5:
                        ident.trusted_box = delivered.copy()
                        ident.trusted_frame = f
                elif cfg.anchor_policy == "learned" and \
                        cfg.write_fn is not None:
                    # learned verified write: the callable returns a score;
                    # a slot is written when it clears the policy threshold
                    wscore = float(cfg.write_fn(
                        seq, f, gid, delivered, source, delivery_score,
                        ident))
                    if wscore >= cfg.anchor_score:
                        ident.memory_slots.append(
                            (f, delivered.copy()))
                        if len(ident.memory_slots) > cfg.memory_k:
                            ident.memory_slots.pop(0)
                if ident.state in ("LOST", "REACTIVATED", "RECOVERY",
                                   "UNCERTAIN"):
                    ident.state = "ACTIVE"
            else:
                ident.react_consecutive = 0
                ident.missing_streak += 1
                ident.delivered_streak = 0
                if ident.state == "ACTIVE":
                    ident.state = "UNCERTAIN"
                if ident.missing_streak >= cfg.lost_streak and \
                        ident.state == "UNCERTAIN":
                    ident.state = "LOST"
                    ident.lost_started = f - cfg.lost_streak + 1
            frame_delivered[ident.public_id] = delivered
            frame_source[ident.public_id] = source

        # ---- recovery + verification + reactivation for LOST identities
        for gid, ident in identities.items():
            if not cfg.enable_recovery:
                continue
            if ident.state != "LOST":
                continue
            if cfg.shadow_mode and any(k[0] == gid
                                       for k in pending_shadows):
                continue
            if ident.pending_box is None and \
                    f - ident.last_retry < cfg.retry_interval:
                continue
            ident.last_retry = f
            ident.recoveries += 1
            qbox = ident.trusted_box if cfg.anchor_policy in ("trusted",
                                                              "verified",
                                                              "oracle") \
                else ident.anchor_box
            qframe = ident.trusted_frame if cfg.anchor_policy in (
                "trusted", "verified", "oracle") \
                else ident.anchor_frame
            if cfg.anchor_policy == "learned" and ident.memory_slots:
                qframe, qbox = ident.memory_slots[-1]
            rec = recovery_fn(seq, f, qbox, qframe, gid)
            if cfg.shadow_mode and shadow_start_fn is not None:
                sid = shadow_start_fn(seq, f, gid, qbox, qframe)
                if sid is not None:
                    pending_shadows[(gid, f)] = {
                        "sid": sid, "start_frame": f, "elapsed": 0,
                        "anchor_frame": qframe,
                    }
                    transactions.append({
                        "sequence": seq, "frame": f, "gid": gid,
                        "public_id": ident.public_id,
                        "anchor_frame": qframe,
                        "recovery_box": None,
                        "verifier_score": None,
                        "accepted": False,
                        "reactivated": False,
                        "state_after": "LOST",
                        "shadow_started": True,
                        "shadow_id": sid,
                        "shadow_event": "START",
                    })
                    continue
            if rec is None:
                transactions.append({
                    "sequence": seq, "frame": f, "gid": gid,
                    "public_id": ident.public_id,
                    "anchor_frame": qframe,
                    "recovery_box": None,
                    "verifier_score": None,
                    "accepted": False,
                    "reactivated": False,
                    "state_after": ident.state,
                })
                continue
            prob = None
            accepted = False
            reactivated = False
            traj = {}
            chosen_box = None
            if rec is not None:
                candidates = rec.get("candidates") or [rec]
                if cfg.use_two_frame and ident.pending_box is not None:
                    for cand in candidates:
                        cand = dict(cand)
                        p = float(verifier_fn(cand))
                        if iou(np.asarray(cand["box"], dtype=float),
                               ident.pending_box) >= cfg.confirm_iou and \
                                p >= cfg.confirm_threshold:
                            prob = p
                            chosen_box = cand["box"]
                            accepted = True
                            break
                    if not accepted:
                        ident.pending_box = None
                        ident.pending_frame = -1
                elif cfg.use_two_frame:
                    scored = []
                    for cand in candidates:
                        cand = dict(cand)
                        p = float(verifier_fn(cand))
                        if p >= cfg.accept_threshold:
                            motion = iou(
                                np.asarray(cand["box"], dtype=float),
                                ident.last_box) if ident.last_box is not None \
                                else 0.0
                            scored.append((p, motion, cand))
                    if scored:
                        scored.sort(key=lambda x: (-x[0], -x[1]))
                        prob, _, chosen = scored[0]
                        chosen_box = chosen["box"]
                        ident.pending_box = np.asarray(chosen_box,
                                                       dtype=float).copy()
                        ident.pending_frame = f
                else:
                    scored = []
                    for cand in candidates:
                        cand = dict(cand)
                        p = float(verifier_fn(cand))
                        if p >= cfg.verifier_threshold:
                            motion = iou(
                                np.asarray(cand["box"], dtype=float),
                                ident.last_box) if ident.last_box is not None \
                                else 0.0
                            scored.append((p, motion, cand))
                    if scored:
                        scored.sort(key=lambda x: (-x[0], -x[1]))
                        prob, _, chosen = scored[0]
                        chosen_box = chosen["box"]
                        accepted = True
                if accepted and cfg.enable_reactivation:
                    traj = reactivate_fn(seq, f, chosen_box)
                    if traj:
                        ident.react_traj = {
                            k: (None if v is None
                                else np.asarray(v, dtype=float))
                            for k, v in traj.items()
                        }
                        ident.accepts += 1
                        ident.state = "REACTIVATED"
                        ident.missing_streak = 0
                        ident.lost_started = None
                        reactivated = True
                        # rebind the P0 tid to the recovery box so delivery
                        # continues correctly after the reactivation horizon
                        rebind, rebind_iou = None, 0.5
                        for row in rows:
                            v = iou(row.box, chosen_box)
                            if v >= rebind_iou:
                                rebind_iou, rebind = v, row
                        ident.bound_tid = rebind.tid if rebind is not None \
                            else None
                        ident.pending_box = None
                        ident.pending_frame = -1
                elif accepted and not cfg.enable_reactivation:
                    # GFN-rank-only ablation: record the verifier verdict but
                    # do not reactivate; identity stays LOST and retries
                    pass
            transactions.append({
                "sequence": seq, "frame": f, "gid": gid,
                "public_id": ident.public_id,
                "anchor_frame": qframe,
                "recovery_box": None if chosen_box is None
                else [round(float(x), 2) for x in chosen_box],
                "verifier_score": None if prob is None else round(prob, 4),
                "accepted": accepted,
                "reactivated": reactivated,
                "state_after": ident.state,
            })
            # failure taxonomy needs the GT verdict of the candidate
            gt_now = gf
            target = None
            if gt_now is not None and gid in gt_now.gt_ids:
                target = np.asarray(
                    gt_now.boxes[gt_now.gt_ids.index(gid)], dtype=float)
            lost_episodes.append({
                "sequence": seq, "frame": f, "gid": gid,
                "recovered_iou": round(float(iou(chosen_box, target)), 4)
                if target is not None else None,
                "verifier_score": None if prob is None else round(prob, 4),
                "accepted": accepted,
            })

        # ---- N20 shadow-step: causally advance pending shadow hypotheses
        if cfg.shadow_mode and shadow_step_fn is not None:
            for key, pend in list(pending_shadows.items()):
                gid, sf = key
                ident = identities[gid]
                if ident.state != "LOST":
                    del pending_shadows[key]
                    continue
                pend["elapsed"] += 1
                res = shadow_step_fn(seq, f, gid, pend["sid"],
                                     pend["elapsed"])
                if res is None:
                    continue
                verdict = res.get("verdict", "PENDING")
                if verdict == "ACCEPT":
                    traj = res.get("traj") or {}
                    if traj:
                        ident.react_traj = {
                            k: (None if v is None
                                else np.asarray(v, dtype=float))
                            for k, v in traj.items()
                        }
                        ident.accepts += 1
                        ident.state = "REACTIVATED"
                        ident.missing_streak = 0
                        ident.lost_started = None
                        commit_f = int(res.get("commit_frame", f))
                        transactions.append({
                            "sequence": seq, "frame": sf, "gid": gid,
                            "public_id": ident.public_id,
                            "anchor_frame": pend.get("anchor_frame"),
                            "recovery_box": None,
                            "verifier_score": None,
                            "accepted": True,
                            "reactivated": True,
                            "state_after": "REACTIVATED",
                            "shadow_started": True,
                            "shadow_id": pend["sid"],
                            "shadow_commit": True,
                            "shadow_verdict": "ACCEPT",
                            "commit_frame": commit_f,
                            "shadow_event": "VERDICT",
                        })
                        rebind, rebind_iou = None, 0.5
                        for row in rows:
                            v = iou(row.box, np.asarray(
                                res.get("box", traj.get(commit_f)),
                                dtype=float))
                            if v >= rebind_iou:
                                rebind_iou, rebind = v, row
                        ident.bound_tid = rebind.tid if rebind is not None \
                            else None
                        b = ident.react_traj.get(f)
                        if b is not None:
                            frame_delivered[ident.public_id] = b.copy()
                            frame_source[ident.public_id] = "shadow_commit"
                    del pending_shadows[key]
                elif verdict in ("REJECT", "TIMEOUT"):
                    transactions.append({
                        "sequence": seq, "frame": sf, "gid": gid,
                        "public_id": ident.public_id,
                        "anchor_frame": pend.get("anchor_frame"),
                        "recovery_box": None,
                        "verifier_score": None,
                        "accepted": False,
                        "reactivated": False,
                        "state_after": ident.state,
                        "shadow_started": True,
                        "shadow_id": pend["sid"],
                        "shadow_commit": False,
                        "shadow_verdict": verdict,
                        "shadow_event": "VERDICT",
                    })
                    del pending_shadows[key]

        # duplicate public-id check
        dup_flag = False
        pub_boxes = [(pid, box) for pid, box in frame_delivered.items()
                     if box is not None]
        for i in range(len(pub_boxes)):
            for j in range(i + 1, len(pub_boxes)):
                if iou(pub_boxes[i][1], pub_boxes[j][1]) > 0.95:
                    dup_flag = True
                    break
            if dup_flag:
                break

        # GT evaluation of this frame (post-hoc metric; GT read after Y_pre)
        ev = []
        for gid, ident in identities.items():
            target = None
            if gf is not None and gid in gf.gt_ids:
                target = np.asarray(gf.boxes[gf.gt_ids.index(gid)], dtype=float)
            box = frame_delivered.get(ident.public_id)
            correct = (target is not None and box is not None
                       and iou(box, target) >= 0.5)
            needs_correction = target is not None and not correct
            if needs_correction and cfg.on_correction is not None:
                cfg.on_correction(seq, f, gid, ident.public_id, target)
            ev.append({
                "frame": f, "gid": gid, "public_id": ident.public_id,
                "state": ident.state,
                "source": frame_source.get(ident.public_id, "none"),
                "gt_present": int(target is not None),
                "delivered": int(box is not None),
                "correct": int(correct),
                "needs_correction": int(needs_correction),
                "duplicate_flag": int(dup_flag),
                "delivery_score": frame_delivery_score.get(
                    ident.public_id),
                "delivery_iou_prev": frame_prev_iou.get(ident.public_id),
                "delivered_box": None if box is None
                else [round(float(x), 2) for x in box],
            })
            trace.append(ev[-1])
        delivered_history.append(frame_delivered)

    return {
        "identities": identities,
        "trace": trace,
        "transactions": transactions,
        "lost_episodes": lost_episodes,
        "delivered_history": delivered_history,
    }


def aggregate_metrics(seq: str, result: dict, cfg: LoopConfig) -> dict:
    """Aggregate FULL_LOOP_V0 metrics from the trace."""
    trace = result["trace"]
    tx = result["transactions"]
    ep = result["lost_episodes"]
    identities = result["identities"]
    # re-correction probability per identity after the human anchor frame
    by_gid: Dict[int, List[dict]] = {}
    for e in trace:
        by_gid.setdefault(e["gid"], []).append(e)
    recorr = {}
    for gid, evs in by_gid.items():
        ident = identities[gid]
        base = ident.anchor_frame
        den = sum(1 for e in evs if e["frame"] > base and e["gt_present"])
        if den == 0:
            continue
        num = sum(1 for e in evs
                  if e["frame"] > base and e["gt_present"]
                  and e["needs_correction"])
        recorr[gid] = num / den
    accepted = [t for t in tx if t.get("reactivated")]
    verifier_accepts = sum(1 for t in tx if t.get("accepted"))
    # post-reactivation retention at horizons
    ret = {}
    for h in (1, 3, 5, 10, 30, 60, 120):
        n = hit = 0
        for t in accepted:
            f0 = t.get("commit_frame", t["frame"])
            ident = identities[t["gid"]]
            # use trace entries for gt presence/correctness
            for e in trace:
                if e["frame"] == f0 + h and e["gid"] == t["gid"]:
                    if e["gt_present"]:
                        n += 1
                        hit += e["correct"]
        ret[h] = hit / n if n else None
    attempts = len([t for t in tx
                    if t.get("shadow_event") != "VERDICT"])
    a = len(accepted)
    return {
        "sequence": seq,
        "n_identities": len(identities),
        "frames": len(trace) // max(1, len(identities)),
        "recovery_attempts": attempts,
        "accepted_recoveries": a,
        "accept_rate": a / attempts if attempts else None,
        "verifier_accepts": verifier_accepts,
        "lost_episodes": len(ep),
        "mean_recorrection_prob": float(np.mean(list(recorr.values())))
        if recorr else None,
        "retention_1": ret.get(1), "retention_3": ret.get(3),
        "retention_5": ret.get(5), "retention_10": ret.get(10),
        "retention_30": ret.get(30), "retention_60": ret.get(60),
        "retention_120": ret.get(120),
        "shadow_commits": sum(1 for t in accepted if t.get("shadow_commit")),
        "shadow_timeouts": sum(1 for t in tx if t.get("shadow_verdict")
                               in ("REJECT", "TIMEOUT")),
        "shadow_lost_frames": sum(
            int(t.get("commit_frame", t["frame"])) - int(t["frame"])
            for t in accepted if t.get("shadow_commit")),
    }
