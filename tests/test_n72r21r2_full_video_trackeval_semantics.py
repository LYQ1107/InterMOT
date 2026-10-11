"""Pinned TrackEval synthetic full-video unit fixture, never research output."""
import csv
import json
import subprocess
import pytest
from scripts.n72r21r2_common import ROOT
from scripts.n72r20r3r2r3_pipeline import run_trackeval_many, parse_trackeval, trackeval_summary


def unit_sequence(root, name, frames):
    sequence = root / name
    (sequence / "gt").mkdir(parents=True)
    (sequence / "seqinfo.ini").write_text(
        f"[Sequence]\nname={name}\nimDir=img1\nframeRate=20\nseqLength={frames}\nimWidth=100\nimHeight=100\nimExt=.jpg\n")
    (sequence / "gt/gt.txt").write_text("".join(f"{f},1,10,10,20,40,1,1,1\n" for f in range(1, frames + 1)))


def test_actual_pinned_CLI_full_video_includes_original_tail_and_all_sequences(tmp_path):
    pinned = ROOT / "third_party/MOTIP/TrackEval"
    assert subprocess.check_output(["git", "-C", str(pinned), "rev-parse", "HEAD"], text=True).strip() == "12c8791b303e0a0b50f753af204249e622d0281a"
    gt, trackers, evaluation = tmp_path / "gt", tmp_path / "trackers", tmp_path / "evaluation"
    sequences = {"unitA": 125, "unitB": 7}
    for name, frames in sequences.items():
        unit_sequence(gt, name, frames)
        for tracker in ("FULL", "TRUNCATED_H100"):
            path = trackers / tracker / "data" / (name + ".txt")
            path.parent.mkdir(parents=True, exist_ok=True)
            length = min(frames, 100) if tracker == "TRUNCATED_H100" else frames
            path.write_text("".join(f"{f},100001,10,10,20,40,1,-1,-1,-1\n" for f in range(1, length + 1)))
    seqmap = tmp_path / "seqmap.txt"
    seqmap.write_text("name\n" + "\n".join(sequences) + "\n")
    receipt = run_trackeval_many(trackers, evaluation, ["FULL", "TRUNCATED_H100"], seqmap,
                                gt_split="train", gt_folder=gt)
    assert receipt["returncode"] == 0
    assert receipt["command"][receipt["command"].index("--GT_FOLDER") + 1] == str(gt)
    assert receipt["command"][receipt["command"].index("--SEQMAP_FILE") + 1] == str(seqmap)
    full = trackeval_summary(parse_trackeval(evaluation, "FULL", sequences))
    short = trackeval_summary(parse_trackeval(evaluation, "TRUNCATED_H100", sequences))
    assert set(full["per_sequence"]) == set(short["per_sequence"]) == set(sequences)
    for metric in ("HOTA", "AssA", "DetA", "LocA", "IDF1", "MOTA"):
        assert full[metric] == pytest.approx(1.)
    for metric in ("IDSW", "FP", "FN"): assert full[metric] == 0
    assert short["FN"] == short["per_sequence"]["unitA"]["FN"] == 25
    assert short["per_sequence"]["unitB"]["FN"] == 0
    assert short["HOTA"] < full["HOTA"] and short["IDF1"] < full["IDF1"]
    assert short["per_sequence"]["unitA"]["HOTA"] < short["per_sequence"]["unitB"]["HOTA"]


@pytest.mark.parametrize("sequence_header", ["seq", "Sequence", "sequence"])
def test_nine_metric_CSV_numeric_units_and_per_sequence_not_combined_imputation(tmp_path, sequence_header):
    root = tmp_path / "unit"
    root.mkdir()
    columns = [sequence_header, "HOTA___AUC", "AssA___AUC", "DetA___AUC", "LocA___AUC", "IDF1", "IDSW", "MOTA", "CLR_FP", "CLR_FN"]
    with (root / "pedestrian_detailed.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerow(["unitA", .625, .5, .75, .9, .7, 12, -.1, 20, 30])
        writer.writerow(["unitB", .8, .6, .9, .95, .85, 2, .4, 3, 4])
        writer.writerow(["COMBINED", .7, .55, .8, .92, .77, 14, .2, 23, 34])
    parsed = trackeval_summary(parse_trackeval(tmp_path, "unit", ["unitA", "unitB", "missing"]))
    assert parsed["HOTA"] == .7 and parsed["IDSW"] == 14 and parsed["FP"] == 23 and parsed["FN"] == 34
    assert parsed["per_sequence"]["unitA"] == dict(HOTA=.625, AssA=.5, DetA=.75, LocA=.9, IDF1=.7, IDSW=12., MOTA=-.1, FP=20., FN=30.)
    assert "missing" not in parsed["per_sequence"]
    assert parsed["per_sequence"]["unitA"]["HOTA"] != parsed["HOTA"]


def test_missing_or_nonnumeric_metric_remains_unavailable_not_zero(tmp_path):
    root = tmp_path / "unit"
    root.mkdir()
    (root / "pedestrian_detailed.csv").write_text("seq,HOTA___AUC,IDSW\nCOMBINED,,not-a-number\nunitA,,not-a-number\n")
    actual = trackeval_summary(parse_trackeval(tmp_path, "unit", ["unitA"]))
    assert actual["HOTA"] is None and actual["IDSW"] is None and actual["IDF1"] is None
    assert actual["per_sequence"]["unitA"]["HOTA"] is None
