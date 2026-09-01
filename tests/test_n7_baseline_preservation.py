"""N7-F toy / synthetic tests for Route A (full-state rehydration)."""

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sam3_intermot.backend.mock_backend import MockBackend
from sam3_intermot.interaction.n6_observer import N6Config
from sam3_intermot.interaction.n7_real_observer import N7RealObserver
from sam3_intermot.interaction.simulator import GTFrame


ROOT = Path(__file__).resolve().parents[1]
N_PEOPLE = 5


def _people_boxes(num_frames: int, n: int = N_PEOPLE):
    boxes_by_frame = {}
    for f in range(num_frames):
        frame = []
        for i in range(n):
            x0 = 100 + i * 220
            y0 = 100 + (i % 3) * 120
            x = min(x0 + f * 3, 1150)
            frame.append(np.asarray([x, y0, x + 60, y0 + 160], dtype=float))
        boxes_by_frame[f] = frame
    return boxes_by_frame


class _FrameBackend(MockBackend):
    """Mock backend whose concept detection sees the per-frame object set."""

    def __init__(self, boxes_by_frame, seed=3):
        super().__init__(seed=seed, velocity_scale=1.0)
        self.boxes_by_frame = boxes_by_frame

    def start_video(self, video_source):
        sid = super().start_video(video_source)
        # real SAM 3.1 native object ids are 0-based; registry track ids are
        # 1-based, so the integer collision is reproduced in tests.
        self._next_object_id = 0
        return sid

    def _make_object(self, frame_idx, box, source, verified):
        obj = super()._make_object(frame_idx, box, source, verified)
        # GT in the tests moves +3 px/frame; make the mock move with it so
        # restart prompts stay aligned with detections (as real SAM does).
        obj["velocity"] = np.asarray([3.0, 0.0, 3.0, 0.0], dtype=float)
        return obj

    def detect_concept(self, frame_idx, text_prompt):
        self._concept_boxes[text_prompt] = [
            np.asarray(b, float) for b in self.boxes_by_frame.get(frame_idx, [])
        ]
        # advance existing objects to their current positions before matching
        # (real SAM objects persist across windows; the base mock does not)
        for obj in self._objects.values():
            if obj["prompt_frame"] is not None and frame_idx > obj["prompt_frame"]:
                obj["prompt_box"] = obj["prompt_box"] + (
                    frame_idx - obj["prompt_frame"]
                ) * obj["velocity"]
                obj["prompt_frame"] = frame_idx
        return super().detect_concept(frame_idx, text_prompt)


def _run(num_frames=40, budget=1, segment_len=10, window_len=10, n=N_PEOPLE, boxes=None, backend=None):
    boxes = boxes or _people_boxes(num_frames, n)
    backend = backend or _FrameBackend(boxes)
    gt = {
        f: GTFrame(
            boxes=[np.asarray(b, float) for b in boxes[f]],
            gt_ids=list(range(1, len(boxes[f]) + 1)),
        )
        for f in range(num_frames)
    }
    obs = N7RealObserver(
        backend,
        "mock://video",
        gt,
        num_frames,
        N6Config(protocol="p4", budget=budget),
        sequence="toy",
        segment_len=segment_len,
        window_len=window_len,
    )
    obs.run()
    return obs, gt


def _ids_for_gt(obs, gt, n=N_PEOPLE):
    mapping = {}
    for f in range(len(gt)):
        gb = [np.asarray(b, float) for b in gt[f].boxes]
        po = obs.post_rows.get(f, [])
        for gi, gid in enumerate(gt[f].gt_ids):
            best = None
            best_iou = 0.5
            for pid, box in po:
                iou = _iou(gb[gi], box)
                if iou > best_iou:
                    best_iou = iou
                    best = pid
            mapping.setdefault(gid, set()).add(best)
    return mapping


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def test_n7_t1_edit_one_keeps_others_public_ids():
    b0, gt = _run(budget=0)
    b1, _ = _run(budget=1)
    m0 = _ids_for_gt(b0, gt)
    m1 = _ids_for_gt(b1, gt)
    edited = min(m1.keys(), key=lambda g: len(m1[g]))
    for gid in m0:
        if gid == edited:
            continue
        assert m1[gid] == m0[gid], f"non-edited GT {gid} public id changed"


def test_n7_t2_recover_keeps_others():
    obs, _ = _run(budget=2)
    uids = list(obs.ns.user_to_lineage)
    assert len(uids) >= 2
    before = {u: obs.ns.public_id_for(u) for u in uids}
    obs.ns.recover(uids[0])
    after = {u: obs.ns.public_id_for(u) for u in uids}
    for u in uids[1:]:
        assert after[u] == before[u]
    assert after[uids[0]] == before[uids[0]]


def test_n7_t3_reassign_keeps_others():
    obs, gt = _run(budget=2)
    uids = list(obs.ns.user_to_lineage)
    tid = max(obs.auto_registry.manager._tracks)  # a still-automatic identity
    before = {u: obs.ns.public_id_for(u) for u in uids}
    obs.track_to_uid[tid] = uids[0]
    after = {u: obs.ns.public_id_for(u) for u in uids}
    for u in uids[1:]:
        assert after[u] == before[u]


def test_n7_t4_swap_keeps_others():
    obs, _ = _run(budget=2)
    uids = list(obs.ns.user_to_lineage)
    tids = sorted(obs.auto_registry.manager._tracks)
    ua = obs.track_to_uid.get(tids[0])
    ub = obs.track_to_uid.get(tids[1])
    if ua is not None:
        obs.track_to_uid[tids[1]] = ua
    if ub is not None:
        obs.track_to_uid[tids[0]] = ub
    before = {u: obs.ns.public_id_for(u) for u in uids}
    uc = obs.track_to_uid.get(tids[0])
    ud = obs.track_to_uid.get(tids[2])
    if uc is not None:
        obs.track_to_uid[tids[2]] = uc
    if ud is not None:
        obs.track_to_uid[tids[0]] = ud
    after = {u: obs.ns.public_id_for(u) for u in uids}
    for u in uids[2:]:
        assert after[u] == before[u]


def test_n7_t5_active_count_consistent_across_restarts():
    obs, _ = _run(num_frames=40, budget=1, segment_len=10)
    assert len(obs.auto_registry.manager.tracks) == N_PEOPLE
    assert obs.window_count >= 3
    assert obs.interaction_restarts == 1
    assert obs.rehydrated_prompts > 0  # full-state rehydration re-prompts autos


def test_n7_t6_rehydration_uses_system_prediction_not_gt():
    boxes = _people_boxes(40)
    # drop person 3 at the interaction-triggered restart frame 1
    boxes[1] = [b for i, b in enumerate(boxes[1]) if i != 2]
    backend = _FrameBackend(boxes)
    gt = {
        f: GTFrame(
            boxes=[np.asarray(b, float) for b in boxes[f]],
            gt_ids=list(range(1, len(boxes[f]) + 1)),
        )
        for f in range(40)
    }
    obs2 = N7RealObserver(
        backend,
        "mock://video",
        gt,
        40,
        N6Config(protocol="p4", budget=1),
        sequence="toy",
        segment_len=10,
        window_len=10,
    )
    rec_prompts = []
    orig_add_box = backend.add_box

    def rec_add_box(frame_idx, object_id, box_xyxy):
        rec_prompts.append((frame_idx, object_id, np.asarray(box_xyxy).copy()))
        return orig_add_box(frame_idx, object_id, box_xyxy)

    backend.add_box = rec_add_box
    obs2.run()
    assert obs2.rehydrated_prompts > 0
    assert len(obs2.auto_registry.manager.tracks) == N_PEOPLE
    prev0 = {pid: np.asarray(b) for pid, b in obs2.pre_rows.get(0, [])}
    assert 3 in prev0
    # a rehydration prompt for person 3 must use the previous system box,
    # never the (absent) GT box
    system_box = prev0[3]
    rehyd = [
        np.asarray(b)
        for f, oid, b in rec_prompts
        if f == 1 and np.allclose(b, system_box)
    ]
    assert rehyd, "no rehydration prompt used the system prediction box"
    assert not np.allclose(system_box, [543.0, 340.0, 603.0, 500.0])


def test_n7_t7_budget_exhausted_no_state_change():
    obs, _ = _run(num_frames=40, budget=1, segment_len=10)
    assert obs.accepted_count == 1
    add_events = [e for e in obs.events if e["action_type"] == "ADD_NEW_IDENTITY"]
    assert len(add_events) == 1
    for f in range(1, 40):
        pre = {pid: tuple(np.round(b, 4)) for pid, b in obs.pre_rows.get(f, [])}
        post = {pid: tuple(np.round(b, 4)) for pid, b in obs.post_rows.get(f, [])}
        assert pre == post, f"frame {f}: post changed without accepted action"


def test_n7_t8_one_interaction_does_not_lose_others():
    obs, _ = _run(num_frames=40, budget=1, segment_len=10)
    for f in (25, 30, 35):
        rows = dict(obs.post_rows.get(f, []))
        assert len(rows) == N_PEOPLE, f"frame {f}: lost identities {rows}"


def test_n7_t9_no_duplicate_rows():
    obs, _ = _run(num_frames=40, budget=1, segment_len=10)
    for f in range(40):
        ids = [pid for pid, _ in obs.post_rows.get(f, [])]
        assert len(ids) == len(set(ids)), f"duplicate public ids at frame {f}"


def test_n7_t10_public_allocation_only_add_new():
    obs, _ = _run(num_frames=40, budget=2, segment_len=10)
    assert obs.ns.allocator.allocations_by_action == {"ADD_NEW_IDENTITY": 2}
    assert obs.ns.allocator.allocations_total == obs.accepted_count


class _FailingResetBackend(_FrameBackend):
    def __init__(self, boxes_by_frame):
        super().__init__(boxes_by_frame)
        self.reset_calls = 0

    def reset_session(self):
        self.reset_calls += 1
        if self.reset_calls == 1:
            raise RuntimeError("injected reset failure")
        super().reset_session()


def test_n7_t11_restart_rollback():
    boxes = _people_boxes(40)
    backend = _FailingResetBackend(boxes)
    gt = {
        f: GTFrame(boxes=[np.asarray(b, float) for b in boxes[f]], gt_ids=list(range(1, N_PEOPLE + 1)))
        for f in range(40)
    }
    obs = N7RealObserver(
        backend,
        "mock://video",
        gt,
        40,
        N6Config(protocol="p4", budget=1),
        sequence="toy",
        segment_len=10,
    )
    with pytest.raises(RuntimeError, match="injected reset failure"):
        obs.run()
    assert obs.rollback_count == 1
    assert obs.segment_id == 1
    assert len(obs.auto_registry.manager.tracks) == N_PEOPLE


def _write_mot(path, rows):
    lines = []
    for f, pid, box in rows:
        x1, y1, x2, y2 = box
        lines.append(
            f"{f+1},{pid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},1.0,-1,-1,-1"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gt(path, rows):
    lines = []
    for f, gid, box in rows:
        x1, y1, x2, y2 = box
        lines.append(
            f"{f+1},{gid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},1,1,1"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _trackeval_idsw(tmp_path, tracker_rows, gt_rows, name):
    gt_dir = tmp_path / "gt"
    tr_dir = tmp_path / "trackers"
    seq_dir = gt_dir / "seq"
    (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
    (tr_dir / name / "data").mkdir(parents=True, exist_ok=True)
    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\nname=seq\nimDir=img1\nframeRate=20\nseqLength=100\n"
        "imWidth=1920\nimHeight=1080\nimExt=.jpg\n",
        encoding="utf-8",
    )
    _write_gt(seq_dir / "gt" / "gt.txt", gt_rows)
    _write_mot(tr_dir / name / "data" / "seq.txt", tracker_rows)
    seqmap = tmp_path / "seqmap.txt"
    seqmap.write_text("name\nseq\n", encoding="utf-8")
    proc = subprocess.run(
        [
            "python",
            "./third_party/MOTIP/TrackEval/scripts/run_mot_challenge.py",
            "--GT_FOLDER", str(gt_dir),
            "--TRACKERS_FOLDER", str(tr_dir),
            "--TRACKERS_TO_EVAL", name,
            "--TRACKER_SUB_FOLDER", "data",
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
        ],
        capture_output=True,
        text=True,
    )
    text = proc.stdout + proc.stderr
    assert proc.returncode == 0, text[-3000:]
    clear = re.search(rf"CLEAR: {name}-pedestrian.*?\nCOMBINED\s+(.*)", text, re.S).group(1).split()
    return int(float(clear[12]))


def test_n7_t12_correction_adds_no_idsw_to_others(tmp_path):
    gt_rows = []
    for f in range(60):
        gt_rows.append((f, 1, [10 + f, 20, 30 + f, 50]))
        gt_rows.append((f, 2, [60 + f, 20, 80 + f, 50]))
        gt_rows.append((f, 3, [110 + f, 20, 130 + f, 50]))
    good = []
    for f in range(60):
        good.append((f, 1, [10 + f, 20, 30 + f, 50]))
        good.append((f, 2 if f >= 25 else 9, [60 + f, 20, 80 + f, 50]))
        good.append((f, 3, [110 + f, 20, 130 + f, 50]))
    bad = []
    for f in range(60):
        bad.append((f, 1 if f < 25 else 1001, [10 + f, 20, 30 + f, 50]))
        bad.append((f, 2 if f >= 25 else 9, [60 + f, 20, 80 + f, 50]))
        bad.append((f, 3, [110 + f, 20, 130 + f, 50]))
    idsw_good = _trackeval_idsw(tmp_path / "good", good, gt_rows, "good")
    idsw_bad = _trackeval_idsw(tmp_path / "bad", bad, gt_rows, "bad")
    assert idsw_good == 1  # only the corrected target
    assert idsw_bad >= 2   # collateral id change creates extra IDSW
