"""N8-D CPU toy tests T1-T15 (temporal-error observer semantics)."""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sam3_intermot.interaction.n8_temporal_observer import (
    EventType,
    N8Config,
    N8TemporalObserver,
)
from sam3_intermot.interaction.simulator import GTFrame


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _box(x=0.0, y=0.0, w=10.0, h=20.0):
    return np.asarray([x, y, x + w, y + h], dtype=float)


def _gt_frame(gid, box):
    return GTFrame(boxes=[np.asarray(box, dtype=float)], gt_ids=[gid])


def _run(backbone, gt, num_frames, budget=1):
    obs = N8TemporalObserver(
        backbone, gt, num_frames, N8Config(budget=budget), sequence="toy"
    )
    obs.run()
    return obs


def test_n8_t1_first_public_id_differs_from_gt_no_interaction():
    bb = {0: [(37, _box())]}
    gt = {0: _gt_frame(4, _box())}
    obs = _run(bb, gt, 1, budget=1)
    assert obs.accepted_count == 0
    assert obs.verified_errors[0]["event_type"] == EventType.FIRST_APPEARANCE_MATCHED
    assert obs.verified_errors[0]["interaction_required"] is False


def test_n8_t2_stable_public_id_zero_interactions():
    bb = {f: [(37, _box(x=f))] for f in range(3)}
    gt = {f: _gt_frame(1, _box(x=f)) for f in range(3)}
    obs = _run(bb, gt, 3, budget=2)
    assert obs.accepted_count == 0
    assert obs.verified_errors[0]["event_type"] == EventType.FIRST_APPEARANCE_MATCHED


def test_n8_t3_id_break_detected_and_corrected():
    bb = {0: [(37, _box(x=0))], 1: [(37, _box(x=1))], 2: [(52, _box(x=2))]}
    gt = {f: _gt_frame(1, _box(x=f)) for f in range(3)}
    obs = _run(bb, gt, 3, budget=1)
    break_events = [e for e in obs.verified_errors if e["event_type"] == EventType.TEMPORAL_ID_BREAK]
    assert len(break_events) == 1
    assert break_events[0]["accepted"] is True
    assert obs.canonical_map == {52: 37}
    pids = {pid for pid, _ in obs.post_rows[2]}
    assert 37 in pids and 52 not in pids


def test_n8_t4_recoverable_miss_detected_and_corrected():
    bb = {0: [(37, _box(x=0))], 1: [], 2: []}
    gt = {f: _gt_frame(1, _box(x=f)) for f in range(3)}
    obs = _run(bb, gt, 3, budget=1)
    recover = [e for e in obs.verified_errors if e["event_type"] == EventType.RECOVERABLE_MISS]
    assert len(recover) == 2
    assert recover[0]["accepted"] is True
    assert obs.post_rows[1][0][0] == 37
    assert np.allclose(obs.post_rows[1][0][1], _box(x=1))


def test_n8_t5_new_matched_first_appearance_zero_interaction():
    bb = {0: [(37, _box())]}
    gt = {0: _gt_frame(4, _box())}
    obs = _run(bb, gt, 1, budget=1)
    assert obs.accepted_count == 0
    assert obs.verified_errors[0]["event_type"] == EventType.FIRST_APPEARANCE_MATCHED
    assert obs.verified_errors[0]["action_type"] == "NONE"


def test_n8_t6_new_miss_adds_identity():
    bb = {0: []}
    gt = {0: _gt_frame(9, _box())}
    obs = _run(bb, gt, 1, budget=1)
    assert obs.accepted_count == 1
    add = [e for e in obs.verified_errors if e["event_type"] == EventType.TRUE_MISS_NEW]
    assert len(add) == 1 and add[0]["accepted"] is True
    pid, box = obs.post_rows[0][0]
    assert pid >= 1000
    assert np.allclose(box, _box())


def test_n8_t7_atomic_swap_corrected():
    bb = {
        0: [(37, _box(x=0)), (52, _box(x=50))],
        1: [(52, _box(x=1)), (37, _box(x=51))],
    }
    gt = {
        0: GTFrame(boxes=[_box(x=0), _box(x=50)], gt_ids=[1, 2]),
        1: GTFrame(boxes=[_box(x=1), _box(x=51)], gt_ids=[1, 2]),
    }
    obs = _run(bb, gt, 2, budget=1)
    swap = [e for e in obs.verified_errors if e["event_type"] == EventType.TEMPORAL_ID_SWAP]
    assert len(swap) == 1 and swap[0]["accepted"] is True
    assert obs.canonical_map == {37: 52, 52: 37}
    by_box = {round(float(b[0]), 2): pid for pid, b in obs.post_rows[1]}
    assert by_box[1.0] == 37 and by_box[51.0] == 52


def test_n8_t8_budget_exhausted_memory_updates_system_unchanged():
    bb = {0: [(37, _box(x=0))], 1: [(52, _box(x=1))], 2: [(53, _box(x=2))]}
    gt = {f: _gt_frame(1, _box(x=f)) for f in range(3)}
    obs = _run(bb, gt, 3, budget=1)
    breaks = [e for e in obs.verified_errors if e["event_type"] == EventType.TEMPORAL_ID_BREAK]
    assert len(breaks) == 2
    assert breaks[0]["accepted"] is True
    assert breaks[1]["accepted"] is False
    assert breaks[1]["reason"] == "BUDGET_EXHAUSTED"
    assert breaks[1]["system_state_hash_before"] == breaks[1]["system_state_hash_after"]
    rec = obs.memory.records[1]
    assert rec["last_observed_public_id"] == 53
    assert rec["canonical_public_id"] == 37
    assert obs.canonical_map == {52: 37}


def test_n8_t9_error_observation_never_overwrites_canonical():
    bb = {0: [(37, _box(x=0))], 1: [(52, _box(x=1))]}
    gt = {f: _gt_frame(1, _box(x=f)) for f in range(2)}
    obs = _run(bb, gt, 2, budget=0)
    rec = obs.memory.records[1]
    assert rec["canonical_public_id"] == 37
    assert rec["last_observed_public_id"] == 52


def test_n8_t10_gt_numeric_mismatch_is_not_error():
    bb = {0: [(37, _box())], 1: [(37, _box(x=1))]}
    gt = {0: _gt_frame(4, _box()), 1: _gt_frame(4, _box(x=1))}
    obs = _run(bb, gt, 2, budget=1)
    assert obs.accepted_count == 0
    types = {e["event_type"] for e in obs.verified_errors}
    assert EventType.TEMPORAL_ID_BREAK not in types


def test_n8_t11_b0_byte_identical_p0(tmp_path):
    dt_root = tmp_path / "dt"
    seq_dir = dt_root / "val" / "toy"
    (seq_dir / "img1").mkdir(parents=True)
    (seq_dir / "gt").mkdir(parents=True)
    for f in range(3):
        (seq_dir / "img1" / f"{f:06d}.jpg").write_bytes(b"")
    (seq_dir / "gt" / "gt.txt").write_text(
        "1,1,0.00,0.00,10.00,20.00,1,1,1\n"
        "2,1,1.00,0.00,10.00,20.00,1,1,1\n"
        "3,1,2.00,0.00,10.00,20.00,1,1,1\n",
        encoding="utf-8",
    )
    p0 = tmp_path / "p0"
    p0.mkdir()
    p0_bytes = (
        "1,1,0.00,0.00,10.00,20.00,0.926,-1,-1,-1\n"
        "2,1,1.00,0.00,10.00,20.00,0.926,-1,-1,-1\n"
        "3,1,2.00,0.00,10.00,20.00,0.926,-1,-1,-1\n"
    ).encode("utf-8")
    (p0 / "toy.txt").write_bytes(p0_bytes)
    out = tmp_path / "out"
    env = dict(os.environ)
    env.update(
        {
            "N8_SEQ": "toy",
            "N8_BUDGET": "0",
            "N8_OUT_DIR": str(out),
            "N8_P0_SOURCE": str(p0),
            "N8_DATASET_ROOT": str(dt_root),
            "PYTHONPATH": str(ROOT),
        }
    )
    proc = subprocess.run(
        [PY, str(ROOT / "scripts/run_n8_real.py")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out / "post_mot" / "toy.txt").read_bytes() == p0_bytes
    assert (out / "pre_mot" / "toy.txt").read_bytes() == p0_bytes


def test_n8_t12_same_frame_priority():
    bb = {
        0: [(37, _box(x=0)), (52, _box(x=50))],
        1: [(9, _box(x=1))],
    }
    gt = {
        0: GTFrame(boxes=[_box(x=0), _box(x=50)], gt_ids=[1, 2]),
        1: GTFrame(boxes=[_box(x=1), _box(x=51), _box(x=100)], gt_ids=[1, 2, 3]),
    }
    obs = _run(bb, gt, 2, budget=1)
    assert obs.accepted_count == 1
    assert obs.interaction_events[0]["event_type"] == EventType.TEMPORAL_ID_BREAK
    frame1_types = {e["event_type"] for e in obs.verified_errors if e["frame"] == 2}
    assert EventType.RECOVERABLE_MISS in frame1_types
    assert EventType.TRUE_MISS_NEW in frame1_types


def test_n8_t13_unaccepted_event_zero_system_mutation():
    bb = {0: [(37, _box(x=0))], 1: [(52, _box(x=1))]}
    gt = {f: _gt_frame(1, _box(x=f)) for f in range(2)}
    obs = _run(bb, gt, 2, budget=0)
    brk = [e for e in obs.verified_errors if e["event_type"] == EventType.TEMPORAL_ID_BREAK][0]
    assert brk["accepted"] is False
    assert brk["system_state_hash_before"] == brk["system_state_hash_after"]
    assert obs.canonical_map == {}
    assert obs.ns.violations() == []
    assert obs.gt_audit["system_mutation_without_accepted_action"] == 0


def test_n8_t14_reassign_persistent_canonical():
    bb = {
        0: [(37, _box(x=0))],
        1: [(37, _box(x=1))],
        2: [(52, _box(x=2))],
        3: [(52, _box(x=3))],
    }
    gt = {f: _gt_frame(1, _box(x=f)) for f in range(4)}
    obs = _run(bb, gt, 4, budget=1)
    assert obs.canonical_map == {52: 37}
    post_pids = {pid for pid, _ in obs.post_rows[3]}
    pre_pids = {pid for pid, _ in obs.pre_rows[3]}
    assert 37 in post_pids and 52 not in post_pids
    assert 37 in pre_pids and 52 not in pre_pids


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
        lines.append(f"{f+1},{gid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},1,1,1")
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
            PY,
            "./"
            "third_party/MOTIP/TrackEval/scripts/run_mot_challenge.py",
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


def test_n8_t15_trackeval_idsw_drops_after_correction(tmp_path):
    n = 60
    gt_rows = []
    for f in range(n):
        gt_rows.append((f, 1, _box(x=10 + f)))
        gt_rows.append((f, 2, _box(x=60 + f)))
    backbone = {}
    for f in range(n):
        a_pid = 1 if f < 30 else 52
        backbone[f] = [(a_pid, _box(x=10 + f)), (2, _box(x=60 + f))]
    gt = {
        f: GTFrame(boxes=[_box(x=10 + f), _box(x=60 + f)], gt_ids=[1, 2])
        for f in range(n)
    }
    obs = _run(backbone, gt, n, budget=1)
    assert obs.accepted_count == 1
    post_rows = []
    for f in range(n):
        for pid, box in obs.post_rows[f]:
            post_rows.append((f, pid, box))
    bad_rows = []
    for f in range(n):
        for tid, box in backbone[f]:
            bad_rows.append((f, tid, box))
    idsw_bad = _trackeval_idsw(tmp_path / "bad", bad_rows, gt_rows, "bad")
    idsw_good = _trackeval_idsw(tmp_path / "good", post_rows, gt_rows, "good")
    assert idsw_bad >= 1
    assert idsw_good < idsw_bad
