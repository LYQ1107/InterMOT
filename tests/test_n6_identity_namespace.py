"""N6-C CPU toy transaction tests (no GPU)."""

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sam3_intermot.evaluation.frame_output import (
    FrameOutputAssembler,
    FrameOutputRow,
    FrameOutputValidationError,
)
from sam3_intermot.identity.namespace import IdentityNamespace, PublicTrackIDAllocator


def _box(x=0.0, y=0.0, w=10.0, h=20.0):
    return np.asarray([x, y, x + w, y + h], dtype=float)


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


def _run_trackeval(tmp_path, tracker_rows, gt_rows, tracker_name="toy"):
    gt_dir = tmp_path / "gt"
    tr_dir = tmp_path / "trackers"
    seq_dir = gt_dir / "seq"
    (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
    (tr_dir / tracker_name / "data").mkdir(parents=True, exist_ok=True)
    (seq_dir / "seqinfo.ini").write_text(
        "[Sequence]\nname=seq\nimDir=img1\nframeRate=20\nseqLength=100\n"
        "imWidth=1920\nimHeight=1080\nimExt=.jpg\n",
        encoding="utf-8",
    )
    _write_gt(seq_dir / "gt" / "gt.txt", gt_rows)
    _write_mot(tr_dir / tracker_name / "data" / "seq.txt", tracker_rows)
    seqmap = tmp_path / "seqmap.txt"
    seqmap.write_text("name\nseq\n", encoding="utf-8")
    proc = subprocess.run(
        [
            "python",
            "./third_party/MOTIP/TrackEval/scripts/run_mot_challenge.py",
            "--GT_FOLDER", str(gt_dir),
            "--TRACKERS_FOLDER", str(tr_dir),
            "--TRACKERS_TO_EVAL", tracker_name,
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
    idf1 = float(re.search(rf"Identity: {tracker_name}-pedestrian.*?\nCOMBINED\s+([\d.]+)", text, re.S).group(1))
    clear = re.search(rf"CLEAR: {tracker_name}-pedestrian.*?\nCOMBINED\s+(.*)", text, re.S).group(1).split()
    idsw = float(clear[12])
    return idf1, idsw


def _gt_rows():
    rows = []
    for f in range(100):
        rows.append((f, 1, _box(10 + f, 20, 10, 20)))
        rows.append((f, 2, _box(50 + f, 20, 10, 20)))
    return rows


def test_gt_id_never_used_as_public_id_100_frames():
    ns = IdentityNamespace()
    uid_a, _, pid_a = ns.create_user(0)
    assert pid_a != 1  # GT id is 1, public id must differ
    rows = [(f, pid_a, _box(10 + f, 20, 10, 20)) for f in range(100)]
    pids = {r[1] for r in rows}
    assert pids == {pid_a}
    assert ns.allocator.allocations_by_action["ADD_NEW_IDENTITY"] == 1


def test_single_person_wrong_id_correction_restores_public_id(tmp_path):
    ns = IdentityNamespace()
    uid_a, _, pid_a = ns.create_user(0)
    pre, post = [], []
    for f in range(100):
        pre.append((f, pid_a if f != 49 else 71, _box(10 + f, 20, 10, 20)))
        pre.append((f, 57, _box(50 + f, 20, 10, 20)))
        post.append((f, pid_a, _box(10 + f, 20, 10, 20)))
        post.append((f, 57, _box(50 + f, 20, 10, 20)))
    idf1_post, idsw_post = _run_trackeval(tmp_path / "corr", post, _gt_rows(), "corr")
    idf1_pre, idsw_pre = _run_trackeval(tmp_path / "corr_pre", pre, _gt_rows(), "corr_pre")
    assert idsw_pre >= 1
    assert idsw_post == 0
    assert idf1_post == 100.0
    assert ns.public_id_for(uid_a) == pid_a


def test_recover_no_allocation():
    ns = IdentityNamespace()
    uid_a, lid_a, pid_a = ns.create_user(0)
    before = ns.allocator.allocations_total
    uid2, lid2, pid2 = ns.recover(uid_a)
    assert (uid2, lid2, pid2) == (uid_a, lid_a, pid_a)
    assert ns.allocator.allocations_total == before


def test_two_person_swap_restores_public_ids():
    ns = IdentityNamespace()
    _, _, pid_a = ns.create_user(0)
    _, _, pid_b = ns.create_user(0)
    assert pid_a != pid_b
    rows = []
    for f in range(100):
        rows.append((f, pid_a, _box(10 + f, 20, 10, 20)))
        rows.append((f, pid_b, _box(50 + f, 20, 10, 20)))
    pids = {r[1] for r in rows}
    assert len(pids) == 2
    assert ns.allocator.allocations_total == 2
    assert ns.allocator.allocations_by_action.get("RECOVER_IDENTITY", 0) == 0


def test_sam_object_change_keeps_public_id():
    ns = IdentityNamespace()
    uid, lid, pid = ns.create_user(0)
    ns.bind_sam(0, 5, uid)
    ns.bind_sam(1, 2, uid)
    assert ns.sam_to_lineage[(0, 5)] == lid
    assert ns.sam_to_lineage[(1, 2)] == lid
    assert ns.public_id_for(uid) == pid
    assert ns.allocator.allocations_total == 1


def test_observe_only_zero_state_change():
    ns = IdentityNamespace()
    _, _, _ = ns.create_user(0)
    h_before = ns.mutable_state_hash()
    # observation of an error without accepting an action
    _ = ns.public_id_for(1)
    h_after = ns.mutable_state_hash()
    assert h_before == h_after


def test_rejected_action_zero_state_change():
    ns = IdentityNamespace()
    _, _, _ = ns.create_user(0)
    h_before = ns.mutable_state_hash()
    snap = ns.snapshot()
    # rejected action path: restore snapshot
    try:
        ns.recover(999)  # raises
    except ValueError:
        pass
    ns.restore(snap)
    assert ns.mutable_state_hash() == h_before


def test_post_serialization_uniqueness():
    assembler = FrameOutputAssembler()
    rows = [
        FrameOutputRow(42, _box()),
        FrameOutputRow(42, _box(50, 0, 10, 20)),
    ]
    with pytest.raises(FrameOutputValidationError):
        assembler.assemble(0, rows)
    good = assembler.assemble(0, [FrameOutputRow(42, _box())])
    assert len(good) == 1
    mot = assembler.rows_to_mot(0, [FrameOutputRow(42, _box()), FrameOutputRow(57, _box(50, 0, 10, 20))])
    assert len(mot) == 2
    assert len({r[1] for r in mot}) == 2


def test_current_frame_authority_and_unrelated_unchanged():
    assembler = FrameOutputAssembler()
    rows = [FrameOutputRow(42, _box()), FrameOutputRow(57, _box(50, 0, 10, 20))]
    # user corrects box of 42; 57 untouched
    new_box = _box(11, 21, 9, 19)
    rows[0] = FrameOutputRow(42, new_box)
    out = assembler.assemble(0, rows)
    assert np.allclose(out[0].box_xyxy, new_box)
    assert np.allclose(out[1].box_xyxy, _box(50, 0, 10, 20))


def test_trackeval_synthetic_regression(tmp_path):
    # stable public IDs -> IDSW 0
    stable = []
    for f in range(100):
        stable.append((f, 42, _box(10 + f, 20, 10, 20)))
        stable.append((f, 57, _box(50 + f, 20, 10, 20)))
    idf1, idsw = _run_trackeval(tmp_path / "stable", stable, _gt_rows(), "stable")
    assert idsw == 0
    assert idf1 == 100.0
    # swap correction restores association
    swap_pre = []
    for f in range(100):
        a, b = (57, 42) if f == 49 else (42, 57)
        swap_pre.append((f, a, _box(10 + f, 20, 10, 20)))
        swap_pre.append((f, b, _box(50 + f, 20, 10, 20)))
    swap_post = []
    for f in range(100):
        swap_post.append((f, 42, _box(10 + f, 20, 10, 20)))
        swap_post.append((f, 57, _box(50 + f, 20, 10, 20)))
    _, idsw_pre = _run_trackeval(tmp_path / "swap_pre", swap_pre, _gt_rows(), "swap_pre")
    _, idsw_post = _run_trackeval(tmp_path / "swap_post", swap_post, _gt_rows(), "swap_post")
    assert idsw_pre >= 2
    assert idsw_post == 0
