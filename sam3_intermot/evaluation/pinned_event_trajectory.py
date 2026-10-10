"""Offline event trajectory utility using the actual pinned TrackEval API.

This is not a target-identity count proxy. The caller supplies sealed global
MOT outputs and original-frame GT; official preprocessing and metrics do the
matching. Full-policy evaluation remains the separate pinned CLI experiment.
"""
from contextlib import contextmanager
from pathlib import Path
import importlib
import subprocess
import sys
import numpy as np

PINNED_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"
METRIC_NAMES = ("HOTA", "AssA", "DetA", "LocA", "IDF1", "IDSW", "MOTA", "FP", "FN")


@contextmanager
def numpy_legacy_types():
    """Same deprecated type aliases as the existing CLI wrapper, scoped here."""
    added = []
    for name, value in (("float", float), ("int", int), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, value)
            added.append(name)
    try:
        yield
    finally:
        for name in added:
            delattr(np, name)


def official_module(checkout):
    checkout = Path(checkout).resolve()
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if commit != PINNED_COMMIT:
        raise ValueError("Unpinned event trajectory evaluator")
    sys.path.insert(0, str(checkout))
    try:
        module = importlib.import_module("trackeval")
    finally:
        sys.path.pop(0)
    if not Path(module.__file__).resolve().is_relative_to(checkout):
        raise ValueError("A different TrackEval module is already imported")
    return module


def tracker_arrays(rows):
    """Byte-export-equivalent rounded MOT geometry, unmodified public IDs."""
    values = {key: [] for key in ("tracker_ids", "tracker_classes", "tracker_dets", "tracker_confidences")}
    for row in rows:
        outputs = sorted(row["outputs"], key=lambda o: o["public_id"])
        if len({o["public_id"] for o in outputs}) != len(outputs) or len({o["candidate_uid"] for o in outputs}) != len(outputs):
            raise ValueError("Full global unique ownership required")
        boxes = []
        for out in outputs:
            x1, y1, x2, y2 = out["box_xyxy"]
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Invalid MOT box")
            boxes.append([float(f"{v:.4f}") for v in (x1, y1, max(1., x2 - x1), max(1., y2 - y1))])
        values["tracker_ids"].append(np.asarray([o["public_id"] for o in outputs], dtype=int))
        values["tracker_classes"].append(np.full(len(outputs), -1, dtype=int))
        values["tracker_dets"].append(np.asarray(boxes, dtype=float).reshape(-1, 4))
        values["tracker_confidences"].append(np.asarray([float(f"{o['confidence']:.6f}") for o in outputs]))
    return values


def metric_summary(result):
    hota, clear, identity = result["HOTA"], result["CLEAR"], result["Identity"]
    return {**{k: float(np.mean(hota[k])) for k in ("HOTA", "AssA", "DetA", "LocA")},
            "IDF1": float(identity["IDF1"]), "IDSW": int(clear["IDSW"]), "MOTA": float(clear["MOTA"]),
            "FP": int(clear["CLR_FP"]), "FN": int(clear["CLR_FN"])}


class PinnedEventEvaluator:
    def __init__(self, checkout, *, gt_root, tracker_root, reference_tracker, sequence, frames):
        self.official = official_module(checkout)
        config = {"GT_FOLDER": str(gt_root), "TRACKERS_FOLDER": str(tracker_root),
                  "TRACKERS_TO_EVAL": [reference_tracker], "SEQ_INFO": {sequence: frames},
                  "PRINT_CONFIG": False, "BENCHMARK": "DanceTrack", "SPLIT_TO_EVAL": "train",
                  "SKIP_SPLIT_FOL": True, "DO_PREPROC": False, "CLASSES_TO_EVAL": ["pedestrian"]}
        self.dataset = self.official.datasets.MotChallenge2DBox(config)
        with numpy_legacy_types():
            self.gt = self.dataset._load_raw_file(reference_tracker, sequence, is_gt=True)
        self.sequence, self.frames = sequence, frames
        self.metrics = [self.official.metrics.HOTA(), self.official.metrics.CLEAR({"PRINT_CONFIG": False}),
                        self.official.metrics.Identity({"PRINT_CONFIG": False})]

    def evaluate(self, rows, *, first_frame, length):
        if len(rows) != length or [r["frame"] for r in rows] != list(range(first_frame, first_frame + length)):
            raise ValueError("Exact original-frame window required, no padded rewards")
        if first_frame < 0 or first_frame + length > self.frames:
            raise ValueError("Incomplete original future window")
        gt = {key: value[first_frame:first_frame + length] for key, value in self.gt.items() if isinstance(value, list)}
        raw = {**gt, **tracker_arrays(rows), "num_timesteps": length, "seq": self.sequence}
        raw["similarity_scores"] = [self.dataset._calculate_similarities(g, p) for g, p in zip(raw["gt_dets"], raw["tracker_dets"], strict=True)]
        with numpy_legacy_types():
            processed = self.dataset.get_preprocessed_seq_data(raw, "pedestrian")
            results = {m.get_name(): m.eval_sequence(processed) for m in self.metrics}
        return metric_summary(results)
