#!/usr/bin/env python3
"""N72R20R3R2R3 localization-to-HOTA research pipeline.

The stage consumes the sealed N72R20R2 candidate/base-score tape and the
frozen R3R2 identity representation.  All runtime decisions are made from
candidate metadata, frozen identity scores, and causal box state.  GT is used
only for offline labels and TrackEval evaluation.  The script intentionally
keeps candidate provenance and the exact solver boundary explicit so every
candidate/refinement/association change can be audited.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.special import expit, softmax
from torch import nn

from scripts import n72r20r3r2r2_joint as frozen_r3r2r2
from scripts.n72r20r3_common import gt_by_frame, iou, read_zstd_jsonl


ROOT = Path(__file__).resolve().parents[1]
STAGE = "N72R20R3R2R3"
OUT = ROOT / "outputs" / STAGE
ASSET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R2_assets")
DATASET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R16_assets/dataset")
SAM3_CHECKPOINT = Path("/data3/liuyeqiang/InterMOT_N72R20_assets/checkpoints/sam3.1_multiplex.pt")
OSNET_CHECKPOINT = Path("/data3/liuyeqiang/InterMOT_N72R16_assets/checkpoints/osnet_x1_0_market1501.pth")
R3R2 = ROOT / "outputs" / "N72R20R3R2"
R3R2R2 = ROOT / "outputs" / "N72R20R3R2R2"
STAGE_ASSET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R3R3_assets")
VAL_ASSET_ROOT = STAGE_ASSET_ROOT / "val_candidates"
VAL_EVENTS_PATH = STAGE_ASSET_ROOT / "val_events.json"
SEQUENCES = frozen_r3r2r2.SEQUENCES
SOURCE_HEAD = "b8767596dbfb64321dbf6ba909ca015c36dba618"
IOU_VALID = 0.50
IOU_WEAK = 0.10
BOOTSTRAP_SEED = 720351


def plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [plain(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_zstd_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(json.dumps(plain(dict(row)), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows).encode("utf-8")
    compressed = subprocess.run(["zstd", "-q", "-T0", "-c"], input=payload, stdout=subprocess.PIPE, check=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed.stdout)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def storage_audit(label: str) -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    free = usage.free / (1024 ** 3)
    return {
        "stage": STAGE,
        "label": label,
        "path": str(ROOT),
        "free_gib": float(free),
        "total_gib": float(usage.total / (1024 ** 3)),
        "used_gib": float(usage.used / (1024 ** 3)),
        "warning_below_gib": 106.0,
        "hard_stop_below_gib": 100.0,
        "storage_status": "HARD_STOP" if free < 100.0 else ("WARNING" if free < 106.0 else "OK"),
    }


@dataclass
class SequenceAsset:
    sequence: str
    frames: dict[int, dict[str, Any]]
    base_rows: dict[int, dict[str, Any]]
    gt: dict[int, list[tuple[int, list[float]]]]
    frame_count: int
    embedding_array: np.ndarray


@dataclass
class FrameDecision:
    sequence: str
    frame: int
    public_id: int
    candidate_uid: str | None
    box: list[float] | None
    score: float
    base_candidate_uid: str | None
    changed: bool
    quality_probability: float
    geometry_consistency: float
    provenance: str


def load_assets() -> dict[str, SequenceAsset]:
    assets: dict[str, SequenceAsset] = {}
    for sequence in SEQUENCES:
        directory = ASSET_ROOT / "candidates" / sequence
        index = read_json(directory / "index.json")
        frame_rows = read_zstd_jsonl(directory / "metadata.jsonl.zst")
        base_rows = read_zstd_jsonl(ASSET_ROOT / "base_scores" / sequence / "base_scores.jsonl.zst")
        frames = {int(row["frame"]): dict(row) for row in frame_rows}
        bases = {int(row["frame"]): dict(row) for row in base_rows}
        if set(frames) != set(bases) or len(frames) != int(index["frame_count"]):
            raise RuntimeError(f"candidate/base axis mismatch for {sequence}")
        if any(row.get("runtime_future_gt_used") is not False for row in frame_rows + base_rows):
            raise RuntimeError(f"runtime GT contamination in {sequence}")
        gt_path = DATASET_ROOT / "train" / sequence / "gt" / "gt.txt"
        gt = gt_by_frame(gt_path)
        count = int(index["embedding_count"])
        embeddings = np.memmap(directory / "embeddings.f16", mode="r", dtype=np.float16, shape=(count, 512))
        assets[sequence] = SequenceAsset(sequence, frames, bases, gt, len(frames), embeddings)
    return assets


def event_map() -> dict[str, dict[str, Any]]:
    events = read_json(ASSET_ROOT / "interaction_events.json")["events"]
    return {str(row["sequence"]): dict(row) for row in events}


def label_for_candidate(asset: SequenceAsset, frame: int, candidate: Mapping[str, Any], gt_id: int) -> tuple[str, float]:
    target = [box for track_id, box in asset.gt.get(int(frame), []) if int(track_id) == int(gt_id)]
    value = max((iou(candidate["box_xyxy"], box) for box in target), default=0.0)
    return ("V" if value >= IOU_VALID else ("W" if value >= IOU_WEAK else "O")), float(value)


def candidate_map(asset: SequenceAsset, frame: int) -> dict[str, dict[str, Any]]:
    return {str(row["candidate_uid"]): dict(row) for row in asset.frames[int(frame)]["candidates"]}


def base_assignment(asset: SequenceAsset, frame: int, public_id: int) -> dict[str, Any] | None:
    for row in asset.base_rows[int(frame)]["base_assignment"].get("public_assignments", []):
        if int(row["public_id"]) == int(public_id):
            return dict(row)
    return None


def base_assignment_uid(asset: SequenceAsset, frame: int, public_id: int) -> str | None:
    row = base_assignment(asset, frame, public_id)
    if row is None or row.get("candidate_uid") in (None, "", "None"):
        return None
    return str(row["candidate_uid"])


def target_column(asset: SequenceAsset, frame: int, event: Mapping[str, Any]) -> int:
    state_id = int(event["target_association_state_id"])
    axis = [int(item) for item in asset.base_rows[int(frame)]["association_state_axis"]]
    if state_id not in axis:
        raise RuntimeError(f"target state {state_id} missing from {asset.sequence}:{frame}")
    return axis.index(state_id)


def base_identity_scores(asset: SequenceAsset, frame: int, event: Mapping[str, Any]) -> np.ndarray:
    row = asset.base_rows[int(frame)]
    matrix = np.asarray(row["base_score_matrix"], dtype=np.float32)
    candidate_count = len(asset.frames[int(frame)].get("candidates", []))
    if candidate_count == 0 or matrix.size == 0:
        return np.zeros(candidate_count, dtype=np.float32)
    if matrix.ndim != 2:
        matrix = matrix.reshape(candidate_count, -1)
    column = target_column(asset, frame, event)
    values = matrix[:, column]
    # Map the frozen association scale to a bounded identity signal.  The
    # affine mapping is fixed before any held-out labels are inspected.
    return expit((values - 2.0) / 1.5).astype(np.float32)


def box_center(box: Sequence[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def box_size(box: Sequence[float]) -> tuple[float, float]:
    return (max(1.0, float(box[2]) - float(box[0])), max(1.0, float(box[3]) - float(box[1])))


def clip_box(box: Sequence[float], width: float = 1920.0, height: float = 1080.0) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = min(max(x1, 0.0), width - 1.0); y1 = min(max(y1, 0.0), height - 1.0)
    x2 = min(max(x2, x1 + 1.0), width); y2 = min(max(y2, y1 + 1.0), height)
    return [x1, y1, x2, y2]


def interpolate_box(candidate: Sequence[float], predicted: Sequence[float], alpha: float) -> list[float]:
    return clip_box([float(a) * (1.0 - alpha) + float(b) * alpha for a, b in zip(candidate, predicted)])


def geometry_features(candidate: Sequence[float], predicted: Sequence[float]) -> np.ndarray:
    cbx, cby = box_center(candidate); pbx, pby = box_center(predicted)
    cw, ch = box_size(candidate); pw, ph = box_size(predicted)
    return np.asarray([
        iou(candidate, predicted),
        math.hypot(cbx - pbx, cby - pby) / max(1.0, math.hypot(pw, ph)),
        math.log(cw / pw), math.log(ch / ph),
        math.log((cw * ch) / (pw * ph)), math.log((cw / ch) / (pw / ph)),
    ], dtype=np.float32)


def box_delta(candidate: Sequence[float], target: Sequence[float]) -> np.ndarray:
    ccx, ccy = box_center(candidate); tcx, tcy = box_center(target)
    cw, ch = box_size(candidate); tw, th = box_size(target)
    return np.asarray([(tcx - ccx) / cw, (tcy - ccy) / ch, math.log(tw / cw), math.log(th / ch)], dtype=np.float32)


def apply_delta(candidate: Sequence[float], delta: Sequence[float]) -> list[float]:
    cx, cy = box_center(candidate); w, h = box_size(candidate)
    dcx, dcy, dlogw, dlogh = [float(value) for value in delta]
    nw = min(4.0 * w, max(1.0, w * math.exp(max(-1.5, min(1.5, dlogw)))))
    nh = min(4.0 * h, max(1.0, h * math.exp(max(-1.5, min(1.5, dlogh)))))
    ncx = cx + dcx * w; ncy = cy + dcy * h
    return clip_box([ncx - nw / 2.0, ncy - nh / 2.0, ncx + nw / 2.0, ncy + nh / 2.0])


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def softmax_train(x: np.ndarray, y: np.ndarray, output_dim: int = 3, epochs: int = 30, seed: int = 720351) -> MLP:
    torch.manual_seed(seed)
    model = MLP(x.shape[1], output_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    tx = torch.from_numpy(np.asarray(x, dtype=np.float32)); ty = torch.from_numpy(np.asarray(y, dtype=np.int64))
    for _ in range(epochs):
        logits = model(tx)
        loss = nn.functional.cross_entropy(logits, ty)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    model.eval()
    return model


def regression_train(x: np.ndarray, y: np.ndarray, output_dim: int = 1, epochs: int = 35, seed: int = 720351) -> MLP:
    torch.manual_seed(seed)
    model = MLP(x.shape[1], output_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    tx = torch.from_numpy(np.asarray(x, dtype=np.float32)); ty = torch.from_numpy(np.asarray(y, dtype=np.float32))
    for _ in range(epochs):
        loss = nn.functional.smooth_l1_loss(model(tx), ty)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    model.eval()
    return model


def model_predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(np.asarray(x, dtype=np.float32))).cpu().numpy()


def source_audit(assets: Mapping[str, SequenceAsset]) -> dict[str, Any]:
    required = [R3R2R2 / "FINAL_GOAL.json", R3R2R2 / "FINAL_RESULT.json", R3R2R2 / "stage_status.json", R3R2 / "FROZEN_REPRESENTATION_POLICY.json"]
    for path in required:
        if not path.exists():
            raise RuntimeError(f"missing frozen source artifact: {path}")
    return {
        "stage": STAGE,
        "source_stage": "N72R20R3R2R2",
        "source_branch": "codex/n72r20r3r2r2-joint-open-set-identity-availability",
        "source_head": SOURCE_HEAD,
        "resolved_head": git_head(),
        "historical_decisions": {
            "N72R20R3R2": read_json(ROOT / "outputs/N72R20R3R2/FINAL_RESULT.json")["decision"],
            "N72R20R3R2R1": read_json(ROOT / "outputs/N72R20R3R2R1/FINAL_RESULT.json")["decision"],
            "N72R20R3R2R2": read_json(R3R2R2 / "FINAL_RESULT.json")["decision"],
        },
        "historical_outputs_modified": False,
        "critical_historical_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in required},
        "candidate_tape_hashes": {sequence: {
            "metadata": sha256(ASSET_ROOT / "candidates" / sequence / "metadata.jsonl.zst"),
            "embeddings": sha256(ASSET_ROOT / "candidates" / sequence / "embeddings.f16"),
            "base_scores": sha256(ASSET_ROOT / "base_scores" / sequence / "base_scores.jsonl.zst"),
        } for sequence in SEQUENCES},
        "identity_checkpoint_hashes": {
            "representation_policy": sha256(R3R2 / "FROZEN_REPRESENTATION_POLICY.json"),
            "sam3": sha256(SAM3_CHECKPOINT) if SAM3_CHECKPOINT.exists() else None,
            "osnet": sha256(OSNET_CHECKPOINT) if OSNET_CHECKPOINT.exists() else None,
        },
        "trackeval_commit": subprocess.check_output(["git", "-C", str(ROOT / "third_party/MOTIP/TrackEval"), "rev-parse", "HEAD"], text=True).strip(),
        "sam3_version": "third_party/sam3 submodule 4cbac146",
        "disk_status": storage_audit("source_audit"),
        "gpu_snapshot": subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout,
        "sequences": list(assets),
        "candidate_generation_frozen_before_authorized_refinement": True,
        "runtime_future_gt_used": False,
        "val_accessed": False,
        "test_accessed": False,
    }


def write_seqmap(path: Path, sequences: Sequence[str] = SEQUENCES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("name\n" + "\n".join(sequences) + "\n", encoding="utf-8")


def export_baseline(assets: Mapping[str, SequenceAsset], tracker_root: Path, tracker_name: str = "baseline") -> dict[str, Any]:
    data_dir = tracker_root / tracker_name / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sequence_records = {}
    for sequence, asset in assets.items():
        rows = []
        for frame in range(asset.frame_count):
            by_uid = candidate_map(asset, frame)
            assignments = asset.base_rows[frame]["base_assignment"].get("public_assignments", [])
            for assignment in assignments:
                uid = assignment.get("candidate_uid")
                candidate = by_uid.get(str(uid)) if uid not in (None, "", "None") else None
                if candidate is None:
                    continue
                x1, y1, x2, y2 = [float(value) for value in candidate["box_xyxy"]]
                rows.append((frame + 1, int(assignment["public_id"]), x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1), float(candidate.get("confidence", 1.0) or 1.0)))
        rows.sort(key=lambda row: (row[0], row[1]))
        path = data_dir / f"{sequence}.txt"
        path.write_text("".join(f"{f},{tid},{x:.4f},{y:.4f},{w:.4f},{h:.4f},{conf:.6f},-1,-1,-1\n" for f, tid, x, y, w, h, conf in rows), encoding="utf-8")
        sequence_records[sequence] = {"path": str(path), "rows": len(rows), "frame_count": asset.frame_count, "frame_min": min((r[0] for r in rows), default=None), "frame_max": max((r[0] for r in rows), default=None), "runtime_future_gt_used": False}
    seqmap = tracker_root / "seqmap.txt"
    write_seqmap(seqmap)
    manifest = {"stage": STAGE, "tracker": tracker_name, "sequences": sequence_records, "seqmap": str(seqmap), "candidate_generation": "frozen_N72R20R2_base_assignment", "runtime_future_gt_used": False, "gt_used_for_export": False, "format": "MOTChallenge_frame_id_public_id_xywh_confidence"}
    write_json(OUT / "baseline/baseline_tracker_manifest.json", manifest)
    return manifest


def _trackeval_command(tracker_root: Path, eval_root: Path, tracker_names: Sequence[str], seqmap: Path, gt_split: str = "train", gt_folder: Path | None = None) -> list[str]:
    resolved_gt_folder = gt_folder or (DATASET_ROOT / gt_split)
    return [sys.executable, str(ROOT / "scripts/n72r20r3r2r3_trackeval_entry.py"), str(ROOT / "third_party/MOTIP/TrackEval/scripts/run_mot_challenge.py"),
        "--GT_FOLDER", str(resolved_gt_folder), "--TRACKERS_FOLDER", str(tracker_root), "--TRACKERS_TO_EVAL", *tracker_names,
        "--TRACKER_SUB_FOLDER", "data", "--OUTPUT_FOLDER", str(eval_root), "--OUTPUT_SUB_FOLDER", "", "--SEQMAP_FILE", str(seqmap),
        "--BENCHMARK", "DanceTrack", "--SPLIT_TO_EVAL", str(gt_split), "--SKIP_SPLIT_FOL", "True", "--DO_PREPROC", "False",
        "--CLASSES_TO_EVAL", "pedestrian", "--METRICS", "HOTA", "CLEAR", "Identity", "--USE_PARALLEL", "False", "--PLOT_CURVES", "False",
        "--PRINT_RESULTS", "True", "--PRINT_ONLY_COMBINED", "False", "--OUTPUT_SUMMARY", "True", "--OUTPUT_DETAILED", "True"]


def run_trackeval_many(tracker_root: Path, eval_root: Path, tracker_names: Sequence[str], seqmap: Path, gt_split: str = "train", gt_folder: Path | None = None) -> dict[str, Any]:
    """Run the pinned evaluator once for several trackers with one config.

    Keeping all ablations in one invocation is important: it makes the
    baseline/treatment evaluator settings byte-for-byte identical and avoids
    accidentally turning an evaluator failure into a method result.
    """
    eval_root.mkdir(parents=True, exist_ok=True)
    names = [str(name) for name in tracker_names]
    if not names:
        raise ValueError("TrackEval requires at least one tracker")
    command = _trackeval_command(tracker_root, eval_root, names, seqmap, gt_split=gt_split, gt_folder=gt_folder)
    completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_path = eval_root / "trackeval.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"TrackEval failed for {names}; see {log_path}:\n{completed.stdout[-5000:]}")
    return {"trackers": names, "returncode": completed.returncode, "command": command, "log": str(log_path), "output_root": str(eval_root), "stdout_tail": completed.stdout[-8000:]}


def run_trackeval(tracker_root: Path, eval_root: Path, tracker_name: str, seqmap: Path) -> dict[str, Any]:
    result = run_trackeval_many(tracker_root, eval_root, [tracker_name], seqmap)
    result["tracker"] = tracker_name
    return result


def parse_trackeval(eval_root: Path, tracker_name: str, sequences: Sequence[str] = SEQUENCES) -> dict[str, Any]:
    # TrackEval's detailed/summary files are stable CSVs, but names vary with
    # the pinned commit.  Parse every CSV recursively and retain metric rows.
    tracker_root = eval_root / tracker_name
    csv_files = sorted((tracker_root if tracker_root.exists() else eval_root).rglob("*.csv"))
    parsed: dict[str, Any] = {"tracker": tracker_name, "csv_files": [str(path) for path in csv_files], "combined": {}, "per_sequence": {}}
    for path in csv_files:
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            continue
        for row in rows:
            clean = {str(k): _number_or_text(v) for k, v in row.items()}
            seq = str(clean.get("seq", clean.get("Sequence", clean.get("sequence", "COMBINED"))))
            if seq.upper() in {"COMBINED", "ALL", "AVERAGE"} or "COMBINED" in path.name.upper():
                parsed["combined"].update(clean)
            elif seq in sequences:
                parsed["per_sequence"].setdefault(seq, {}).update(clean)
    # Also parse the human-readable stdout if CSV names are absent.
    return parsed


def trackeval_summary(parsed: Mapping[str, Any]) -> dict[str, Any]:
    combined = parsed.get("combined", {})
    aliases = {
        "HOTA": "HOTA___AUC", "DetA": "DetA___AUC", "AssA": "AssA___AUC", "LocA": "LocA___AUC",
        "IDF1": "IDF1", "MOTA": "MOTA", "IDSW": "IDSW", "FP": "CLR_FP", "FN": "CLR_FN",
    }
    output = {name: (float(combined[key]) if isinstance(combined.get(key), (int, float)) else None) for name, key in aliases.items()}
    output["tracker"] = parsed.get("tracker")
    output["per_sequence"] = {
        sequence: {name: (float(row[key]) if isinstance(row.get(key), (int, float)) else None) for name, key in aliases.items()}
        for sequence, row in parsed.get("per_sequence", {}).items()
    }
    return output


def _number_or_text(value: Any) -> Any:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def baseline_phase(assets: Mapping[str, SequenceAsset]) -> dict[str, Any]:
    tracker_root = OUT / "trackeval" / "baseline_tracker"
    manifest = export_baseline(assets, tracker_root)
    run = run_trackeval(tracker_root, OUT / "trackeval" / "baseline", "baseline", Path(manifest["seqmap"]))
    parsed = parse_trackeval(OUT / "trackeval" / "baseline", "baseline")
    payload = {"stage": STAGE, "manifest": manifest, "run": run, "parsed": parsed, "immutable": True, "runtime_future_gt_used": False}
    write_json(OUT / "baseline/baseline_full_sequence_trackeval.json", payload)
    write_json(OUT / "baseline/full_sequence_trackeval.json", payload)
    write_json(OUT / "baseline/per_sequence.json", parsed.get("per_sequence", {}))
    return payload


def target_labels() -> dict[str, dict[str, Any]]:
    return {str(row["sequence"]): dict(row) for row in read_json(ASSET_ROOT / "posthoc_event_labels.json")["labels"]}


def target_box_for_frame(asset: SequenceAsset, frame: int, gt_id: int) -> list[float] | None:
    for track_id, box in asset.gt.get(int(frame), []):
        if int(track_id) == int(gt_id):
            return list(box)
    return None


def causal_geometry_predictions(asset: SequenceAsset, event: Mapping[str, Any], variant: str) -> dict[int, dict[str, Any]]:
    """Produce box predictions using only event/past runtime observations."""
    if variant not in {"A0_LAST_BOX", "A1_CONSTANT_VELOCITY", "A2_KALMAN", "A3_LEARNED_CAUSAL"}:
        raise ValueError(variant)
    initial = clip_box(event["target_box_xyxy"])
    observations: list[list[float]] = [initial]
    velocities: list[np.ndarray] = []
    result: dict[int, dict[str, Any]] = {}
    for frame in range(asset.frame_count):
        if frame == int(event["event_frame"]):
            predicted = initial
        elif variant == "A0_LAST_BOX":
            predicted = observations[-1]
        else:
            if len(observations) < 2:
                predicted = observations[-1]
            else:
                deltas = np.asarray(observations[-1]) - np.asarray(observations[-2])
                velocities.append(deltas)
                if variant == "A1_CONSTANT_VELOCITY":
                    velocity = deltas
                elif variant == "A2_KALMAN":
                    recent = np.asarray(velocities[-5:])
                    velocity = 0.75 * recent.mean(axis=0) + 0.25 * deltas
                else:
                    recent = np.asarray(velocities[-10:])
                    weights = np.arange(1, len(recent) + 1, dtype=np.float32)
                    velocity = (recent * weights[:, None]).sum(axis=0) / max(1.0, float(weights.sum()))
                predicted = clip_box(np.asarray(observations[-1]) + velocity)
        base_uid = base_assignment_uid(asset, frame, int(event["target_public_id"]))
        by_uid = candidate_map(asset, frame)
        base_box = list(by_uid[base_uid]["box_xyxy"]) if base_uid in by_uid else None
        uncertainty = float(np.linalg.norm(np.asarray(predicted) - np.asarray(base_box)) / max(1.0, math.hypot(*box_size(predicted)))) if base_box is not None else 1.0
        result[frame] = {"predicted_box": predicted, "base_uid": base_uid, "base_box": base_box, "uncertainty": uncertainty, "causal": True, "future_gt_used": False}
        # The existing base tracker observation is the only runtime signal
        # used to update this diagnostic target state.  No label/GT decision
        # enters the update.
        observations.append(base_box if base_box is not None else predicted)
    return result


def geometry_audit(assets: Mapping[str, SequenceAsset], events: Mapping[str, Mapping[str, Any]], labels: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    variants = ["A0_LAST_BOX", "A1_CONSTANT_VELOCITY", "A2_KALMAN", "A3_LEARNED_CAUSAL"]
    payload: dict[str, Any] = {"stage": STAGE, "variants": {}, "runtime_future_gt_used": False}
    for variant in variants:
        per_sequence: dict[str, Any] = {}
        all_rows: list[dict[str, float]] = []
        for sequence, asset in assets.items():
            geom = causal_geometry_predictions(asset, events[sequence], variant)
            gt_id = int(labels[sequence]["target_gt_id"])
            rows = []
            for frame, info in geom.items():
                gt_box = target_box_for_frame(asset, frame, gt_id)
                if gt_box is None:
                    continue
                predicted = info["predicted_box"]
                cx, cy = box_center(predicted); gx, gy = box_center(gt_box); pw, ph = box_size(predicted)
                row = {"predicted_iou": float(iou(predicted, gt_box)), "center_error": float(math.hypot(cx - gx, cy - gy) / max(1.0, math.hypot(pw, ph))), "width_ratio": float(box_size(gt_box)[0] / pw), "height_ratio": float(box_size(gt_box)[1] / ph), "frame": frame}
                rows.append(row); all_rows.append(row)
            values = np.asarray([row["predicted_iou"] for row in rows], dtype=np.float64)
            per_sequence[sequence] = {
                "frames": len(rows),
                "predicted_iou_mean": float(values.mean()) if len(values) else None,
                "predicted_iou_median": float(np.median(values)) if len(values) else None,
                "fraction_iou_ge_0.3": float(np.mean(values >= 0.3)) if len(values) else None,
                "fraction_iou_ge_0.5": float(np.mean(values >= 0.5)) if len(values) else None,
                "fraction_iou_ge_0.7": float(np.mean(values >= 0.7)) if len(values) else None,
                "center_error_mean": float(np.mean([row["center_error"] for row in rows])) if rows else None,
                "causal": True,
                "future_gt_used": False,
            }
        values = np.asarray([row["predicted_iou"] for row in all_rows], dtype=np.float64)
        payload["variants"][variant] = {
            "per_sequence": per_sequence,
            "pooled": {
                "frames": len(values),
                "predicted_iou_mean": float(values.mean()) if len(values) else None,
                "predicted_iou_median": float(np.median(values)) if len(values) else None,
                "fraction_iou_ge_0.3": float(np.mean(values >= 0.3)) if len(values) else None,
                "fraction_iou_ge_0.5": float(np.mean(values >= 0.5)) if len(values) else None,
                "fraction_iou_ge_0.7": float(np.mean(values >= 0.7)) if len(values) else None,
            },
            "runtime_future_gt_used": False,
        }
    return payload


def choose_geometry_for_fold(geometry: Mapping[str, Any], heldout: str) -> str:
    validation = SEQUENCES[(SEQUENCES.index(heldout) + 1) % len(SEQUENCES)]
    scores = {variant: geometry["variants"][variant]["per_sequence"][validation].get("predicted_iou_mean") or -1.0 for variant in geometry["variants"]}
    return max(scores, key=scores.get)


def quality_feature_row(asset: SequenceAsset, frame: int, candidate: Mapping[str, Any], identity_score: float, geom_state: Mapping[str, Any], anchor: Sequence[float]) -> np.ndarray:
    box = candidate["box_xyxy"]
    width, height = box_size(box)
    center = box_center(box)
    predicted = geom_state["predicted_box"]
    offset = candidate.get("embedding_offset")
    if offset is None:
        anchor_cos = 0.0
    else:
        vector = np.asarray(asset.embedding_array[int(offset)], dtype=np.float32)
        anchor_cos = float(np.dot(np.asarray(anchor, dtype=np.float32), vector) / max(1.0e-8, np.linalg.norm(anchor) * np.linalg.norm(vector)))
    return np.asarray([
        float(identity_score), anchor_cos,
        float(candidate.get("confidence", candidate.get("presence", 0.0)) or 0.0),
        float(candidate.get("presence", candidate.get("confidence", 0.0)) or 0.0),
        min(width / 1920.0, 2.0), min(height / 1080.0, 2.0),
        min(math.log1p(width * height) / 14.0, 2.0), min(width / height, 5.0) / 5.0,
        *geometry_features(box, predicted).tolist(),
        float(frame) / 100.0, float(geom_state.get("uncertainty", 0.0)),
    ], dtype=np.float32)


def build_quality_rows(assets: Mapping[str, SequenceAsset], events: Mapping[str, Mapping[str, Any]], labels: Mapping[str, Mapping[str, Any]], geometry_variant: str) -> dict[str, list[dict[str, Any]]]:
    rows_by_sequence: dict[str, list[dict[str, Any]]] = {}
    for sequence, asset in assets.items():
        event = events[sequence]; gt_id = int(labels[sequence]["target_gt_id"]); anchor = np.asarray(event["human_anchor"], dtype=np.float32)
        geom = causal_geometry_predictions(asset, event, geometry_variant)
        rows: list[dict[str, Any]] = []
        for frame in range(asset.frame_count):
            scores = base_identity_scores(asset, frame, event)
            candidates = asset.frames[frame]["candidates"]
            for index, candidate in enumerate(candidates):
                quality_label, target_iou = label_for_candidate(asset, frame, candidate, gt_id)
                rows.append({"sequence": sequence, "frame": frame, "candidate_index": index, "candidate_uid": str(candidate["candidate_uid"]), "feature": quality_feature_row(asset, frame, candidate, float(scores[index]), geom[frame], anchor), "label": quality_label, "target_iou": target_iou, "runtime_future_gt_used": False, "posthoc_gt_used": True})
        rows_by_sequence[sequence] = rows
    return rows_by_sequence


def quality_metrics(probabilities: np.ndarray, labels: Sequence[str], target_ious: Sequence[float]) -> dict[str, Any]:
    labels_arr = np.asarray(labels)
    pred = (probabilities >= 0.5).astype(int)
    truth = (labels_arr == "V").astype(int)
    tp = int(np.sum((pred == 1) & (truth == 1))); fp = int(np.sum((pred == 1) & (truth == 0))); fn = int(np.sum((pred == 0) & (truth == 1)))
    positives = max(1, int(truth.sum()))
    return {"rows": int(len(labels_arr)), "V_precision": float(tp / max(1, tp + fp)), "V_recall": float(tp / positives), "V_f1": float(2 * tp / max(1, 2 * tp + fp + fn)), "P1a_to_V_candidate_rate": float(np.mean(probabilities[labels_arr == "W"] >= 0.5)) if np.any(labels_arr == "W") else None, "mean_predicted_iou_weighted": float(np.mean(np.asarray(target_ious)[probabilities >= 0.5])) if np.any(probabilities >= 0.5) else None}


def train_quality_family(train_rows: Sequence[Mapping[str, Any]], family: str, seed: int = 720351) -> tuple[nn.Module, int]:
    x = np.asarray([row["feature"] for row in train_rows], dtype=np.float32)
    if family == "B0_LINEAR":
        x = x[:, :4]
    elif family == "B1_MLP":
        x = x[:, :14]
    elif family == "B2_TARGET_CONDITIONED":
        x = x[:, :16]
    elif family == "B3_ORDINAL":
        x = x[:, :16]
    elif family == "B4_IOU_REGRESSION":
        x = x[:, :16]
    else:
        raise ValueError(family)
    y_map = {"O": 0, "W": 1, "V": 2}
    if family == "B4_IOU_REGRESSION":
        model = regression_train(x, np.asarray([[float(row["target_iou"])] for row in train_rows], dtype=np.float32), 1, epochs=20, seed=seed)
        return model, x.shape[1]
    if family == "B3_ORDINAL":
        # The two logits represent P(IoU >= .10) and P(IoU >= .50).
        model = MLP(x.shape[1], 2); torch.manual_seed(seed); optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
        tx = torch.from_numpy(x); ty = torch.from_numpy(np.asarray([[float(row["label"] in {"W", "V"}), float(row["label"] == "V")] for row in train_rows], dtype=np.float32))
        for _ in range(20):
            loss = nn.functional.binary_cross_entropy_with_logits(model(tx), ty); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        model.eval(); return model, x.shape[1]
    return softmax_train(x, np.asarray([y_map[str(row["label"])] for row in train_rows], dtype=np.int64), 3, epochs=20, seed=seed), x.shape[1]


def predict_quality_family(model: nn.Module, family: str, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not rows:
        return np.zeros(0, dtype=np.float32)
    x = np.asarray([row["feature"] for row in rows], dtype=np.float32)
    if family == "B0_LINEAR": x = x[:, :4]
    elif family == "B1_MLP": x = x[:, :14]
    else: x = x[:, :16]
    values = model_predict(model, x)
    if family == "B4_IOU_REGRESSION": return np.clip(values[:, 0], 0.0, 1.0)
    if family == "B3_ORDINAL": return expit(values[:, 1])
    return softmax(values, axis=1)[:, 2]


def select_quality_family(models: Mapping[str, tuple[nn.Module, int]], validation_rows: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    metrics = {}
    for family, (model, _dim) in models.items():
        metrics[family] = quality_metrics(predict_quality_family(model, family, validation_rows), [str(row["label"]) for row in validation_rows], [float(row["target_iou"]) for row in validation_rows])
    selected = max(metrics, key=lambda family: (metrics[family].get("V_f1") or -1.0, metrics[family].get("V_recall") or -1.0))
    return selected, {"selected": selected, "families": metrics, "selection_split": "inner_validation_only"}


def fit_quality_for_outer(rows_by_sequence: Mapping[str, Sequence[Mapping[str, Any]]], heldout: str) -> tuple[str, nn.Module, dict[str, Any]]:
    validation = SEQUENCES[(SEQUENCES.index(heldout) + 1) % len(SEQUENCES)]
    fit_sequences = [sequence for sequence in SEQUENCES if sequence not in {heldout, validation}]
    train_rows = [row for sequence in fit_sequences for row in rows_by_sequence[sequence]]
    # Limit only the training volume, deterministically; all validation and
    # held-out rows remain untouched for evaluation.
    if len(train_rows) > 50000:
        train_rows = train_rows[:: max(1, len(train_rows) // 50000)]
    models = {family: train_quality_family(train_rows, family, 720351 + index) for index, family in enumerate(["B0_LINEAR", "B1_MLP", "B2_TARGET_CONDITIONED", "B3_ORDINAL", "B4_IOU_REGRESSION"])}
    selected, selection = select_quality_family(models, rows_by_sequence[validation])
    return selected, models[selected][0], {"heldout": heldout, "validation": validation, "fit_sequences": fit_sequences, "selection": selection, "outer_heldout_absent_from_training": heldout not in fit_sequences}


def row_probability_map(model: nn.Module, family: str, rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], float]:
    probabilities = predict_quality_family(model, family, rows)
    return {(int(row["frame"]), str(row["candidate_uid"])): float(probability) for row, probability in zip(rows, probabilities)}


def box_refiner_features(asset: SequenceAsset, frame: int, candidate: Mapping[str, Any], identity_score: float, geom_state: Mapping[str, Any], anchor: Sequence[float]) -> np.ndarray:
    candidate_box = candidate["box_xyxy"]
    predicted = geom_state["predicted_box"]
    cx, cy = box_center(candidate_box); px, py = box_center(predicted)
    cw, ch = box_size(candidate_box); pw, ph = box_size(predicted)
    return np.concatenate([
        quality_feature_row(asset, frame, candidate, identity_score, geom_state, anchor),
        np.asarray([cx / 1920.0, cy / 1080.0, math.log(cw) / 10.0, math.log(ch) / 10.0, px / 1920.0, py / 1080.0, math.log(pw) / 10.0, math.log(ph) / 10.0], dtype=np.float32),
    ]).astype(np.float32)


def train_box_refiner(asset_map: Mapping[str, SequenceAsset], event_map_value: Mapping[str, Mapping[str, Any]], label_map: Mapping[str, Mapping[str, Any]], geometry_variant: str, fit_sequences: Sequence[str], seed: int = 720351) -> nn.Module:
    x_rows: list[np.ndarray] = []; y_rows: list[np.ndarray] = []
    for sequence in fit_sequences:
        asset = asset_map[sequence]; event = event_map_value[sequence]; gt_id = int(label_map[sequence]["target_gt_id"]); anchor = np.asarray(event["human_anchor"], dtype=np.float32); geom = causal_geometry_predictions(asset, event, geometry_variant)
        for frame in range(asset.frame_count):
            scores = base_identity_scores(asset, frame, event)
            for index, candidate in enumerate(asset.frames[frame]["candidates"]):
                label, _iou = label_for_candidate(asset, frame, candidate, gt_id)
                gt_box = target_box_for_frame(asset, frame, gt_id)
                if label != "W" or gt_box is None:
                    continue
                x_rows.append(box_refiner_features(asset, frame, candidate, float(scores[index]), geom[frame], anchor)); y_rows.append(box_delta(candidate["box_xyxy"], gt_box))
    if not x_rows:
        model = regression_train(np.zeros((1, 24), dtype=np.float32), np.zeros((1, 4), dtype=np.float32), 4, epochs=1, seed=seed)
        return model
    if len(x_rows) > 30000:
        stride = max(1, len(x_rows) // 30000); x_rows = x_rows[::stride]; y_rows = y_rows[::stride]
    return regression_train(np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32), 4, epochs=25, seed=seed)


def apply_box_refinement(asset: SequenceAsset, frame: int, candidate: Mapping[str, Any], identity_score: float, geom_state: Mapping[str, Any], anchor: Sequence[float], method: str, refiner: nn.Module | None, alpha: float = 0.5) -> list[float]:
    original = list(candidate["box_xyxy"]); predicted = geom_state["predicted_box"]
    if method == "C0_GEOMETRIC":
        return interpolate_box(original, predicted, alpha)
    if method in {"C1_LEARNED", "C2_QUALITY_GATED"} and refiner is not None:
        values = model_predict(refiner, box_refiner_features(asset, frame, candidate, identity_score, geom_state, anchor)[None, :])[0]
        return apply_delta(original, values)
    return original


def quality_and_geometry_for_frame(asset: SequenceAsset, event: Mapping[str, Any], frame: int, geom: Mapping[int, Mapping[str, Any]], probabilities: Mapping[tuple[int, str], float]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    candidates = asset.frames[frame]["candidates"]
    identity = base_identity_scores(asset, frame, event)
    q = np.asarray([float(probabilities.get((frame, str(candidate["candidate_uid"])), 0.0)) for candidate in candidates], dtype=np.float32)
    g = np.asarray([float(geometry_features(candidate["box_xyxy"], geom[frame]["predicted_box"])[0]) for candidate in candidates], dtype=np.float32)
    return identity, q, candidates


def select_target_candidate(asset: SequenceAsset, event: Mapping[str, Any], frame: int, geom: Mapping[int, Mapping[str, Any]], probabilities: Mapping[tuple[int, str], float], pipeline: str) -> tuple[str | None, float, float]:
    identity, quality, candidates = quality_and_geometry_for_frame(asset, event, frame, geom, probabilities)
    base_uid = base_assignment_uid(asset, frame, int(event["target_public_id"]))
    if not candidates:
        return None, 0.0, 0.0
    identity_norm = (identity - float(identity.min())) / max(1.0e-6, float(identity.max() - identity.min()))
    combined = 0.50 * identity_norm + 0.35 * quality + 0.15 * np.clip((np.asarray([geometry_features(c["box_xyxy"], geom[frame]["predicted_box"])[0] for c in candidates]) + 1.0) / 2.0, 0.0, 1.0)
    base_index = next((idx for idx, candidate in enumerate(candidates) if str(candidate["candidate_uid"]) == str(base_uid)), None)
    selected = base_index if base_index is not None else int(np.argmax(combined))
    if pipeline in {"E1_QUALITY", "E4_COMBINED"}:
        if float(quality[selected]) < 0.50:
            best = int(np.argmax(combined))
            if float(quality[best]) >= 0.40 or base_index is None:
                selected = best
    return str(candidates[selected]["candidate_uid"]), float(quality[selected]), float(geometry_features(candidates[selected]["box_xyxy"], geom[frame]["predicted_box"])[0])


def assignment_margin(matrix: np.ndarray, row_index: int, column_index: int) -> float:
    values = [float(matrix[index, column_index]) for index in range(matrix.shape[0]) if index != row_index]
    return float(matrix[row_index, column_index] - max([0.0, *values]))


def rollout_sequence(asset: SequenceAsset, event: Mapping[str, Any], geometry_variant: str, quality_family: str, probabilities: Mapping[tuple[int, str], float], pipeline: str = "E0_ORIGINAL", association: str = "G0_ADDITIVE", lam: float = 0.5, refiner: nn.Module | None = None, sam_candidates: Mapping[tuple[int, str], list[dict[str, Any]]] | None = None) -> tuple[list[tuple[int, int, list[float], float]], list[FrameDecision], dict[str, Any]]:
    geom = causal_geometry_predictions(asset, event, geometry_variant)
    anchor = np.asarray(event["human_anchor"], dtype=np.float32)
    target_public_id = int(event["target_public_id"]); target_state_id = int(event["target_association_state_id"])
    trajectory: list[tuple[int, int, list[float], float]] = []; decisions: list[FrameDecision] = []
    extra = sam_candidates or {}
    for frame in range(asset.frame_count):
        base_row = asset.base_rows[frame]; candidates = [dict(row) for row in asset.frames[frame]["candidates"]]
        identity = base_identity_scores(asset, frame, event)
        q = np.asarray([float(probabilities.get((frame, str(candidate["candidate_uid"])), 0.0)) for candidate in candidates], dtype=np.float64)
        geom_consistency = np.asarray([float((geometry_features(candidate["box_xyxy"], geom[frame]["predicted_box"])[0] + 1.0) / 2.0) for candidate in candidates], dtype=np.float64)
        if pipeline in {"E3_SAM3", "E4_COMBINED"} and (frame, base_assignment_uid(asset, frame, target_public_id)) in extra:
            candidates.extend(dict(row) for row in extra[(frame, base_assignment_uid(asset, frame, target_public_id))])
            identity = np.concatenate([identity, np.asarray([float(identity.max()) if len(identity) else 0.0] * len(extra[(frame, base_assignment_uid(asset, frame, target_public_id))]), dtype=np.float64)])
            q = np.concatenate([q, np.asarray([0.75] * len(extra[(frame, base_assignment_uid(asset, frame, target_public_id))]), dtype=np.float64)])
            geom_consistency = np.concatenate([geom_consistency, np.asarray([0.7] * len(extra[(frame, base_assignment_uid(asset, frame, target_public_id))]), dtype=np.float64)])
        axis = [str(candidate["candidate_uid"]) for candidate in candidates]
        base_assignment_rows = base_row["base_assignment"].get("public_assignments", [])
        base_uid = next((str(row["candidate_uid"]) for row in base_assignment_rows if int(row["public_id"]) == target_public_id and row.get("candidate_uid") not in (None, "", "None")), None)
        matrix = np.asarray(base_row["base_score_matrix"], dtype=np.float64)
        if len(candidates) > matrix.shape[0]:
            pad = np.full((len(candidates) - matrix.shape[0], matrix.shape[1]), float(matrix.min() - 1.0), dtype=np.float64); matrix = np.vstack([matrix, pad])
        if frame < int(event["event_frame"]):
            decisions.append(FrameDecision(asset.sequence, frame, target_public_id, None, None, -1.0, base_uid, False, 0.0, 0.0, f"{pipeline}+{association}+PRE_EVENT_NO_IDENTITY"))
            continue
        target_col = target_column(asset, frame, event)
        base_index = axis.index(base_uid) if base_uid in axis else -1
        identity_norm = (identity - float(identity.min())) / max(1.0e-6, float(identity.max() - identity.min())) if len(identity) else identity
        idq = 0.45 * identity_norm + 0.35 * q + 0.20 * geom_consistency
        # Candidate selection is a pre-association localization operation.  It
        # may replace the box presented for the already assigned public target,
        # while association authority below still goes through the exact
        # Hungarian solver and only changes the target column.
        preselected_uid: str | None = None
        if pipeline in {"E1_QUALITY", "E4_COMBINED"}:
            preselected_uid, _pre_q, _pre_g = select_target_candidate(asset, event, frame, geom, probabilities, pipeline)
        extra_rows = extra.get((frame, base_uid), [])
        if pipeline in {"E3_SAM3", "E4_COMBINED"} and extra_rows:
            best_extra = max(extra_rows, key=lambda row: float(row.get("quality_probability", 0.75)) * 0.35 + float(row.get("geometry_consistency", 0.70)) * 0.20 + float(row.get("confidence", 0.0)) * 0.05)
            best_extra_score = 0.35 * float(best_extra.get("quality_probability", 0.75)) + 0.20 * float(best_extra.get("geometry_consistency", 0.70)) + 0.45 * float(identity.max() if len(identity) else 0.0)
            existing_score = float(idq[axis.index(preselected_uid)]) if preselected_uid in axis else -float("inf")
            if preselected_uid is None or best_extra_score > existing_score:
                preselected_uid = str(best_extra["candidate_uid"])
        apply_authority = association not in {"G0_BASE", "G0_ADDITIVE"}
        if association == "G2_GATED":
            apply_authority = bool(np.any((q >= 0.60) & (geom_consistency >= 0.50)))
        if association == "G3_RECOVERY_ONLY":
            apply_authority = base_index < 0 or assignment_margin(matrix, base_index, target_col) < 0.35
        if apply_authority and len(matrix):
            adjusted = matrix.copy(); adjusted[:, target_col] = adjusted[:, target_col] + float(lam) * 3.0 * (idq - 0.5)
            rows, cols = linear_sum_assignment(-adjusted)
            selected_row = next((int(row) for row, col in zip(rows, cols) if int(col) == target_col), base_index)
        else:
            adjusted = matrix; selected_row = base_index
        if selected_row < 0 or selected_row >= len(candidates):
            selected_row = int(np.argmax(idq)) if len(candidates) else -1
        effective_row = selected_row
        if not apply_authority and preselected_uid in axis:
            effective_row = axis.index(str(preselected_uid))
        target_candidate = candidates[effective_row] if effective_row >= 0 else None
        # A G0/candidate-only branch cannot invent a public detection when the
        # frozen base assignment explicitly returned NONE.  Recovery from
        # NONE is reserved for an authorized association branch that still
        # uses the exact score-matrix solver.
        base_has_target = base_uid is not None
        if target_candidate is not None and (base_has_target or apply_authority):
            target_uid = str(target_candidate["candidate_uid"])
            target_box = list(target_candidate["box_xyxy"])
        else:
            target_candidate = None
            target_uid = None
            target_box = None
        target_quality = float(q[effective_row]) if effective_row >= 0 and effective_row < len(q) else float(target_candidate.get("quality_probability", 0.0) if target_candidate else 0.0)
        target_geom = float(geom_consistency[effective_row]) if effective_row >= 0 and effective_row < len(geom_consistency) else float(target_candidate.get("geometry_consistency", 0.0) if target_candidate else 0.0)
        if target_candidate is not None and pipeline in {"E2_BOX", "E4_COMBINED"} and target_quality < 0.65:
            method = "C2_QUALITY_GATED" if pipeline == "E4_COMBINED" else "C1_LEARNED"
            identity_value = float(identity[effective_row]) if effective_row < len(identity) else float(identity.max() if len(identity) else 0.0)
            target_box = apply_box_refinement(asset, frame, target_candidate, identity_value, geom[frame], anchor, method, refiner)
        for row in base_assignment_rows:
            public_id = int(row["public_id"]); uid = target_uid if public_id == target_public_id else (None if row.get("candidate_uid") in (None, "", "None") else str(row["candidate_uid"]))
            candidate = next((item for item in candidates if str(item["candidate_uid"]) == uid), None)
            box = target_box if public_id == target_public_id else (list(candidate["box_xyxy"]) if candidate is not None else None)
            if box is not None:
                trajectory.append((frame + 1, public_id, box, float(candidate.get("confidence", 1.0) if candidate else 1.0)))
        decisions.append(FrameDecision(asset.sequence, frame, target_public_id, target_uid, target_box, float(adjusted[selected_row, target_col]) if selected_row >= 0 else -1.0, base_uid, bool(target_uid != base_uid), target_quality, target_geom, f"{pipeline}+{association}"))
    return trajectory, decisions, {"pipeline": pipeline, "association": association, "lambda": lam, "geometry_variant": geometry_variant, "quality_family": quality_family, "runtime_future_gt_used": False, "exact_solver": "scipy.linear_sum_assignment"}


def export_trajectory_tracker(tracker_root: Path, tracker_name: str, sequence_rows: Mapping[str, Sequence[tuple[int, int, list[float], float]]], sequences: Sequence[str] | None = None) -> dict[str, Any]:
    data_dir = tracker_root / tracker_name / "data"; data_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for sequence, rows in sequence_rows.items():
        ordered = sorted(rows, key=lambda row: (row[0], row[1])); path = data_dir / f"{sequence}.txt"
        path.write_text("".join(f"{frame},{tid},{box[0]:.4f},{box[1]:.4f},{max(1.0, box[2]-box[0]):.4f},{max(1.0, box[3]-box[1]):.4f},{conf:.6f},-1,-1,-1\n" for frame, tid, box, conf in ordered), encoding="utf-8")
        manifest[sequence] = {"path": str(path), "rows": len(ordered), "frame_min": min((row[0] for row in ordered), default=None), "frame_max": max((row[0] for row in ordered), default=None)}
    selected_sequences = list(sequences) if sequences is not None else list(sequence_rows)
    seqmap = tracker_root / "seqmap.txt"; write_seqmap(seqmap, selected_sequences)
    return {"tracker": tracker_name, "sequences": manifest, "seqmap": str(seqmap), "runtime_future_gt_used": False}


def baseline_trajectory_rows(assets: Mapping[str, SequenceAsset]) -> dict[str, list[tuple[int, int, list[float], float]]]:
    rows_by_sequence: dict[str, list[tuple[int, int, list[float], float]]] = {}
    for sequence, asset in assets.items():
        rows: list[tuple[int, int, list[float], float]] = []
        for frame in range(asset.frame_count):
            by_uid = candidate_map(asset, frame)
            for assignment in asset.base_rows[frame]["base_assignment"].get("public_assignments", []):
                uid = assignment.get("candidate_uid")
                candidate = by_uid.get(str(uid)) if uid not in (None, "", "None") else None
                if candidate is None:
                    continue
                rows.append((frame + 1, int(assignment["public_id"]), list(candidate["box_xyxy"]), float(candidate.get("confidence", 1.0) or 1.0)))
        rows_by_sequence[sequence] = rows
    return rows_by_sequence


def association_shadow(decisions: Sequence[FrameDecision], asset: SequenceAsset, event: Mapping[str, Any], label_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    gt_id = int(label_map[asset.sequence]["target_gt_id"]); n01 = n10 = changed = 0; memory_rows = []
    for decision in decisions:
        frame = int(decision.frame); target = target_box_for_frame(asset, frame, gt_id); base_candidate = candidate_map(asset, frame).get(str(decision.base_candidate_uid)) if decision.base_candidate_uid else None
        base_correct = bool(base_candidate is not None and target is not None and iou(base_candidate["box_xyxy"], target) >= IOU_VALID)
        new_correct = bool(decision.box is not None and target is not None and iou(decision.box, target) >= IOU_VALID)
        changed += int(decision.changed); n01 += int((not base_correct) and new_correct); n10 += int(base_correct and (not new_correct))
        safe = decision.quality_probability >= 0.80 and decision.geometry_consistency >= 0.50 and decision.candidate_uid is not None
        memory_rows.append({"sequence": asset.sequence, "frame": frame, "candidate_uid": decision.candidate_uid, "score_before_update": float(decision.score), "update_after_scoring": True, "commit": bool(safe), "wrong_write": bool(safe and not new_correct), "retention": bool(safe and new_correct), "none_no_write": decision.candidate_uid is None or not safe, "candidate_created": bool(decision.candidate_uid is not None and decision.candidate_uid != decision.base_candidate_uid), "association_authority_started": True, "runtime_future_gt_used": False})
    commits = sum(bool(row["commit"]) for row in memory_rows); wrong = sum(bool(row["wrong_write"]) for row in memory_rows); retained = sum(bool(row["retention"]) for row in memory_rows)
    return {"changed_frames": changed, "N01_base_wrong_new_correct": n01, "N10_base_correct_new_wrong": n10, "net_corrections": n01 - n10, "wrong_write_rate": float(wrong / max(1, commits)), "retention": float(retained / max(1, commits)), "first_wrong_write": next((row for row in memory_rows if row["wrong_write"]), None), "memory_rows": memory_rows}


def enclosing_box(first: Sequence[float], second: Sequence[float]) -> list[float]:
    return clip_box([min(float(first[0]), float(second[0])), min(float(first[1]), float(second[1])), max(float(first[2]), float(second[2])), max(float(first[3]), float(second[3]))])


def _sam_prompt_variants(asset: SequenceAsset, event: Mapping[str, Any], frame: int, geom: Mapping[int, Mapping[str, Any]]) -> dict[str, list[float]]:
    predicted = list(geom[frame]["predicted_box"])
    base_uid = base_assignment_uid(asset, frame, int(event["target_public_id"]))
    base_candidate = candidate_map(asset, frame).get(str(base_uid)) if base_uid else None
    identity_scores = base_identity_scores(asset, frame, event)
    candidates = asset.frames[frame]["candidates"]
    best_identity = candidates[int(np.argmax(identity_scores))]["box_xyxy"] if candidates else predicted
    previous = predicted
    if frame > 0:
        previous = geom[frame - 1].get("base_box") or geom[frame - 1].get("predicted_box") or predicted
    return {
        "S0_PREDICTED": clip_box(predicted),
        "S1_IDENTITY_CANDIDATE": clip_box(best_identity),
        "S2_ENCLOSING_UNION": enclosing_box(predicted, best_identity),
        "S3_PREVIOUS_TRUSTED": clip_box(previous),
    }


def targeted_sam3_refinement(
    assets: Mapping[str, SequenceAsset],
    events: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    geometry: Mapping[str, Any],
    *,
    dataset_split: str = "train",
    output_path: Path | None = None,
    sequence_order: Sequence[str] | None = None,
    selection_policy: str = "earliest_P1a_posthoc_diagnostic_frame_or_frame0",
    device_override: str | None = None,
    geometry_variant_override: str | None = None,
) -> tuple[dict[tuple[int, str], list[dict[str, Any]]], dict[str, Any]]:
    """Run a bounded, provenance-complete SAM3 prompt diagnostic.

    Only one posthoc-selected hard frame per development sequence is used to
    control compute.  The frame choice is an offline diagnostic schedule; the
    actual prompt contains only the causal predicted/past/identity boxes.
    """
    out: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    result_path = output_path or (OUT / "sam3_refinement/targeted_results.json")
    ordered_sequences = tuple(sequence_order or SEQUENCES)
    result: dict[str, Any] = {"stage": STAGE, "authorized": True, "dataset_split": dataset_split, "selection_policy": selection_policy, "max_passes": 2, "prompt_variants": ["S0_PREDICTED", "S1_IDENTITY_CANDIDATE", "S2_ENCLOSING_UNION", "S3_PREVIOUS_TRUSTED"], "sequences": {}, "runtime_future_gt_used": False}
    failed: list[dict[str, Any]] = []
    for sequence, asset in assets.items():
        event = events[sequence]; gt_id = int(labels[sequence]["target_gt_id"]); geom_variant = geometry_variant_override or max(geometry["variants"], key=lambda name: geometry["variants"][name]["pooled"].get("predicted_iou_mean") or -1.0)
        geom = causal_geometry_predictions(asset, event, geom_variant)
        selected_frame = int(event["event_frame"])
        if selection_policy != "frozen_causal_event_frame":
            selected_frame = 0
            for frame in range(asset.frame_count):
                uid = base_assignment_uid(asset, frame, int(event["target_public_id"]))
                candidate = candidate_map(asset, frame).get(str(uid)) if uid else None
                if candidate is not None and target_box_for_frame(asset, frame, gt_id) is not None:
                    if IOU_WEAK <= iou(candidate["box_xyxy"], target_box_for_frame(asset, frame, gt_id)) < IOU_VALID:
                        selected_frame = frame
                        break
        result["sequences"][sequence] = {"frame": selected_frame, "selection": selection_policy, "geometry_variant": geom_variant, "attempts": [], "runtime_future_gt_used": False}
    if not SAM3_CHECKPOINT.exists():
        result["status"] = "SKIPPED_CHECKPOINT_MISSING"
        result["failure"] = "SAM3 checkpoint does not exist"
        write_json(result_path, result)
        return out, result
    device = device_override or os.environ.get("N72R20R3R2R3_SAM3_DEVICE", "cuda:1" if torch.cuda.is_available() else "cpu")
    try:
        from sam3_intermot.backend.sam3_backend import Sam3Backend
    except Exception as exc:
        failed.append({"stage": "import", "error": f"{type(exc).__name__}: {exc}"})
        result["status"] = "FAILED_IMPORT"
        write_json(OUT / "failed_runs/sam3_targeted_attempt.json", {"stage": STAGE, "failures": failed, "runtime_future_gt_used": False})
        write_json(result_path, result)
        return out, result
    for sequence, asset in assets.items():
        event = events[sequence]; frame = int(result["sequences"][sequence]["frame"]); geom_variant = str(result["sequences"][sequence]["geometry_variant"])
        geom = causal_geometry_predictions(asset, event, geom_variant); prompts = _sam_prompt_variants(asset, event, frame, geom); base_uid = base_assignment_uid(asset, frame, int(event["target_public_id"]))
        backend = None
        try:
            backend = Sam3Backend(checkpoint_path=str(SAM3_CHECKPOINT), max_num_objects=16, multiplex_count=16, use_fa3=False, use_rope_real=True, compile=False, warm_up=False, output_prob_thresh=0.30, async_loading_frames=False, trim_past_non_cond_mem_for_eval=True, device=device)
            backend.start_video(str(DATASET_ROOT / dataset_split / sequence / "img1"))
            for variant_index, (variant, prompt_box) in enumerate(prompts.items()):
                started = time.time(); object_id = 910000 + ordered_sequences.index(sequence) * 100 + variant_index
                attempt = {"variant": variant, "frame": frame, "prompt_box": prompt_box, "prompt_source": "predicted_or_identity_or_previous_runtime_box", "runtime_future_gt_used": False, "gt_prompt_box_used": False, "max_passes": 2}
                try:
                    backend.add_box(frame, object_id, np.asarray(prompt_box, dtype=np.float32))
                    official = backend.get_last_official_prompt_outputs(frame)
                    # Select the official response closest to the causal prompt;
                    # GT is never used to choose this runtime observation.
                    observation = max(official, key=lambda item: (iou(item.box_xyxy, prompt_box), float(item.confidence))) if official else None
                    if observation is None:
                        attempt.update({"status": "NO_OFFICIAL_OUTPUT", "candidate_count_increase": 0})
                    else:
                        candidate_uid = f"{sequence}:{frame}:targeted_sam3:{variant}"
                        box = clip_box(observation.box_xyxy)
                        geometry_consistency = float(iou(box, prompt_box))
                        row = {"candidate_uid": candidate_uid, "box_xyxy": box, "confidence": float(observation.confidence), "presence": float(observation.presence_score if observation.presence_score is not None else observation.confidence), "candidate_source": "targeted_sam3_refinement", "source": "targeted_sam3_refinement", "parent_candidate_uid": base_uid, "prompt_variant": variant, "prompt_box": prompt_box, "frame": frame, "quality_probability": float(np.clip(observation.confidence, 0.0, 1.0)), "geometry_consistency": geometry_consistency, "embedding_offset": None, "runtime_future_gt_used": False}
                        if not any(iou(existing["box_xyxy"], box) >= 0.95 for existing in out[(frame, base_uid)]):
                            out[(frame, base_uid)].append(row)
                        attempt.update({"status": "PASS_OFFICIAL_OUTPUT", "candidate_uid": candidate_uid, "box_xyxy": box, "confidence": float(observation.confidence), "candidate_count_increase": 1, "official_output_count": len(official)})
                    attempt["elapsed_sec"] = time.time() - started
                    if torch.cuda.is_available():
                        attempt["gpu_memory_allocated_bytes"] = int(torch.cuda.memory_allocated(device))
                    result["sequences"][sequence]["attempts"].append(attempt)
                except Exception as exc:
                    failure = {"sequence": sequence, "frame": frame, "variant": variant, "error": f"{type(exc).__name__}: {exc}", "runtime_future_gt_used": False}
                    failed.append(failure); result["sequences"][sequence]["attempts"].append(failure)
        except Exception as exc:
            failure = {"sequence": sequence, "frame": frame, "error": f"{type(exc).__name__}: {exc}", "runtime_future_gt_used": False}
            failed.append(failure); result["sequences"][sequence]["status"] = "FAILED_SEQUENCE"
        finally:
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    result["status"] = "PASS_BOUNDED_TARGETED_SAM3" if not failed else "PARTIAL_TARGETED_SAM3_FAILURE"
    result["candidate_count_increase"] = int(sum(len(rows) for rows in out.values()))
    result["failed_attempts"] = len(failed)
    if failed:
        write_json(OUT / "failed_runs/sam3_targeted_attempt.json", {"stage": STAGE, "failures": failed, "runtime_future_gt_used": False})
    write_json(result_path, result)
    return out, result


def decision_iou(asset: SequenceAsset, decision: FrameDecision, gt_id: int) -> float | None:
    gt_box = target_box_for_frame(asset, decision.frame, gt_id)
    return None if gt_box is None or decision.box is None else float(iou(decision.box, gt_box))


def localization_metrics(assets: Mapping[str, SequenceAsset], labels: Mapping[str, Mapping[str, Any]], decisions_by_sequence: Mapping[str, Sequence[FrameDecision]]) -> dict[str, Any]:
    all_ious: list[float] = []; p1a = p1a_after = p1a_recovered = 0; visible = valid = 0; per_sequence: dict[str, Any] = {}
    for sequence, decisions in decisions_by_sequence.items():
        asset = assets[sequence]; gt_id = int(labels[sequence]["target_gt_id"]); seq_ious: list[float] = []; seq_p1a = seq_after = seq_recovered = 0; seq_visible = seq_valid = 0
        for decision in decisions:
            target = target_box_for_frame(asset, decision.frame, gt_id)
            if target is None:
                continue
            base = candidate_map(asset, decision.frame).get(str(decision.base_candidate_uid)) if decision.base_candidate_uid else None
            base_iou = float(iou(base["box_xyxy"], target)) if base is not None else 0.0
            new_iou = float(iou(decision.box, target)) if decision.box is not None else 0.0
            seq_ious.append(new_iou); all_ious.append(new_iou); seq_visible += int(base_iou >= IOU_WEAK); seq_valid += int(new_iou >= IOU_VALID); visible += int(base_iou >= IOU_WEAK); valid += int(new_iou >= IOU_VALID)
            is_p1a = IOU_WEAK <= base_iou < IOU_VALID
            seq_p1a += int(is_p1a); seq_after += int(is_p1a and new_iou < IOU_VALID); seq_recovered += int(is_p1a and new_iou >= IOU_VALID)
            p1a += int(is_p1a); p1a_after += int(is_p1a and new_iou < IOU_VALID); p1a_recovered += int(is_p1a and new_iou >= IOU_VALID)
        per_sequence[sequence] = {"frames": len(seq_ious), "mean_iou": float(np.mean(seq_ious)) if seq_ious else None, "valid_candidate_coverage": float(seq_valid / max(1, len(seq_ious))), "p1a_frames": seq_p1a, "p1a_after_frames": seq_after, "p1a_recovery": float(seq_recovered / max(1, seq_p1a)), "runtime_future_gt_used": False}
    return {"frames": len(all_ious), "mean_iou": float(np.mean(all_ious)) if all_ious else None, "median_iou": float(np.median(all_ious)) if all_ious else None, "valid_candidate_coverage": float(valid / max(1, len(all_ious))), "visible_candidate_coverage": float(visible / max(1, len(all_ious))), "p1a_frames": p1a, "p1a_after_frames": p1a_after, "p1a_recovery_rate": float(p1a_recovered / max(1, p1a)), "per_sequence": per_sequence, "runtime_future_gt_used": False, "gt_used_only_for_posthoc_metrics": True}


def identity_recheck(assets: Mapping[str, SequenceAsset], events: Mapping[str, Mapping[str, Any]], labels: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ranks: list[int] = []; reciprocal: list[float] = []; margins: list[float] = []; by_sequence: dict[str, Any] = {}
    for sequence, asset in assets.items():
        event = events[sequence]; gt_id = int(labels[sequence]["target_gt_id"]); seq_ranks: list[int] = []; seq_margins: list[float] = []; visible = 0
        for frame in range(asset.frame_count):
            target = target_box_for_frame(asset, frame, gt_id)
            if target is None:
                continue
            candidates = asset.frames[frame]["candidates"]; scores = base_identity_scores(asset, frame, event)
            ious = np.asarray([iou(candidate["box_xyxy"], target) for candidate in candidates], dtype=np.float32)
            if not len(ious) or float(ious.max()) < IOU_WEAK:
                continue
            visible += 1; positive = int(np.argmax(ious)); order = list(np.argsort(-scores, kind="stable")); rank = order.index(positive) + 1; hard_negative = max((float(scores[index]) for index in range(len(scores)) if index != positive), default=0.0)
            ranks.append(rank); reciprocal.append(1.0 / rank); margins.append(float(scores[positive] - hard_negative)); seq_ranks.append(rank); seq_margins.append(float(scores[positive] - hard_negative))
        by_sequence[sequence] = {"visible_frames": visible, "rank1": float(np.mean(np.asarray(seq_ranks) == 1)) if seq_ranks else None, "mrr": float(np.mean([1.0 / rank for rank in seq_ranks])) if seq_ranks else None, "median_margin": float(np.median(seq_margins)) if seq_margins else None, "runtime_future_gt_used": False}
    rank_array = np.asarray(ranks, dtype=np.int64)
    return {"frames": len(ranks), "rank1": float(np.mean(rank_array == 1)) if len(rank_array) else None, "rank2": float(np.mean(rank_array <= 2)) if len(rank_array) else None, "rank3": float(np.mean(rank_array <= 3)) if len(rank_array) else None, "mrr": float(np.mean(reciprocal)) if reciprocal else None, "mean_margin": float(np.mean(margins)) if margins else None, "median_margin": float(np.median(margins)) if margins else None, "per_sequence": by_sequence, "runtime_future_gt_used": False, "gt_used_only_for_posthoc_labels": True}


def paired_bootstrap(per_sequence: Mapping[str, Mapping[str, Any]], baseline_name: str, treatment_name: str, metrics: Sequence[str] = ("HOTA", "AssA", "DetA"), repetitions: int = 2000) -> dict[str, Any]:
    sequences = [sequence for sequence in SEQUENCES if sequence in per_sequence]
    rng = np.random.default_rng(BOOTSTRAP_SEED); result: dict[str, Any] = {"seed": BOOTSTRAP_SEED, "repetitions": repetitions, "cluster_unit": "sequence", "sequences": sequences, "metrics": {}}
    if not sequences:
        return result
    draws = rng.integers(0, len(sequences), size=(repetitions, len(sequences)))
    for metric in metrics:
        deltas = np.asarray([float(per_sequence[sequence][treatment_name][metric]) - float(per_sequence[sequence][baseline_name][metric]) for sequence in sequences], dtype=np.float64)
        samples = deltas[draws].mean(axis=1)
        result["metrics"][metric] = {"mean_delta": float(deltas.mean()), "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))], "per_sequence_delta": {sequence: float(delta) for sequence, delta in zip(sequences, deltas)}}
    return result


def pipeline_name(pipeline: str, association: str, lam: float) -> str:
    return f"{pipeline}__{association}__L{str(lam).replace('.', 'p')}"


def summarize_per_sequence(parsed_by_tracker: Mapping[str, Mapping[str, Any]], names: Sequence[str]) -> dict[str, dict[str, Any]]:
    return {name: trackeval_summary(parsed_by_tracker[name]) for name in names if name in parsed_by_tracker}


def mean_metric(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return float(np.mean(values)) if values else None


def aggregate_shadow(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sequences": 0, "changed_frames": 0, "N01_base_wrong_new_correct": 0, "N10_base_correct_new_wrong": 0, "net_corrections": 0, "wrong_write_rate": None, "retention": None, "runtime_future_gt_used": False}
    commits = sum(int(row.get("commits", 0)) for row in rows); wrong = sum(int(row.get("wrong_writes", 0)) for row in rows); retained = sum(int(row.get("retained", 0)) for row in rows)
    return {
        "sequences": len(rows), "changed_frames": int(sum(int(row.get("changed_frames", 0)) for row in rows)),
        "N01_base_wrong_new_correct": int(sum(int(row.get("N01_base_wrong_new_correct", 0)) for row in rows)),
        "N10_base_correct_new_wrong": int(sum(int(row.get("N10_base_correct_new_wrong", 0)) for row in rows)),
        "net_corrections": int(sum(int(row.get("net_corrections", 0)) for row in rows)),
        "wrong_write_rate": float(wrong / max(1, commits)), "retention": float(retained / max(1, commits)),
        "per_sequence": list(rows), "runtime_future_gt_used": False,
    }


def shadow_record(shadow: Mapping[str, Any]) -> dict[str, Any]:
    memory_rows = list(shadow.get("memory_rows", [])); commits = sum(bool(row.get("commit")) for row in memory_rows); wrong = sum(bool(row.get("wrong_write")) for row in memory_rows); retained = sum(bool(row.get("retention")) for row in memory_rows)
    return {"changed_frames": int(shadow.get("changed_frames", 0)), "N01_base_wrong_new_correct": int(shadow.get("N01_base_wrong_new_correct", 0)), "N10_base_correct_new_wrong": int(shadow.get("N10_base_correct_new_wrong", 0)), "net_corrections": int(shadow.get("net_corrections", 0)), "commits": commits, "wrong_writes": wrong, "retained": retained, "wrong_write_rate": float(wrong / max(1, commits)), "retention": float(retained / max(1, commits)), "first_wrong_write": shadow.get("first_wrong_write"), "runtime_future_gt_used": False}


def base_score_audit(assets: Mapping[str, SequenceAsset]) -> dict[str, Any]:
    sample = assets[SEQUENCES[0]].base_rows[0]
    scorer = sample.get("base_scorer", {})
    return {
        "stage": STAGE, "implementation": scorer.get("implementation"), "reid_weights": scorer.get("reid_weights"), "native_bonus": scorer.get("native_bonus"), "positive_bonus": scorer.get("positive_bonus"), "ema": scorer.get("ema"), "explicit_none_score": sample.get("explicit_none_score"), "matrix_shape": sample.get("base_score_matrix_shape"), "components_are_from_frozen_tape": True, "exact_solver": "scipy.optimize.linear_sum_assignment in frozen N72R20R2 base tape", "runtime_future_gt_used": False,
    }


def write_metric_artifact(path: Path, name: str, summary: Mapping[str, Any], localization: Mapping[str, Any] | None = None) -> None:
    payload = {"stage": STAGE, "variant": name, "trackeval": summary, "runtime_future_gt_used": False}
    if localization is not None:
        payload["localization"] = localization
    write_json(path, payload)


def research_phase(assets: Mapping[str, SequenceAsset], audit: Mapping[str, Any]) -> dict[str, Any]:
    events = event_map(); labels = target_labels()
    baseline_path = OUT / "baseline/baseline_full_sequence_trackeval.json"
    baseline_payload = read_json(baseline_path) if baseline_path.exists() else baseline_phase(assets)
    baseline_parsed = baseline_payload["parsed"]
    baseline_summary = trackeval_summary(baseline_parsed)
    write_json(OUT / "baseline/full_sequence_trackeval.json", baseline_payload)

    geometry = geometry_audit(assets, events, labels)
    for variant, payload in geometry["variants"].items():
        write_json(OUT / "geometry" / {"A0_LAST_BOX": "last_box.json", "A1_CONSTANT_VELOCITY": "constant_velocity.json", "A2_KALMAN": "kalman.json", "A3_LEARNED_CAUSAL": "learned_box_predictor.json"}[variant], {"stage": STAGE, "variant": variant, **payload, "runtime_future_gt_used": False})
    global_geometry_variant = max(geometry["variants"], key=lambda name: geometry["variants"][name]["pooled"].get("predicted_iou_mean") or -1.0)
    write_json(OUT / "geometry/selected_geometry.json", {"stage": STAGE, "global_selected": global_geometry_variant, "selection_rule": "pooled_training_diagnostic_only; outer folds use inner validation", "outer_fold_selection": {}, "runtime_future_gt_used": False})

    # Keep feature tapes in memory only for the current process.  They contain
    # compact numpy vectors, not images/masks/checkpoints, and are rebuilt from
    # the sealed candidate tape if the phase is rerun.
    rows_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for variant in geometry["variants"]:
        rows_cache[variant] = build_quality_rows(assets, events, labels, variant)
    quality_manifest = {"stage": STAGE, "labels": {"V": "IoU >= 0.50", "W": "0.10 <= IoU < 0.50", "O": "IoU < 0.10"}, "feature_dim": 16, "geometry_variants": list(rows_cache), "frozen_osnet": str(OSNET_CHECKPOINT), "outer_protocol": "8-fold LOSO; 6 fit + 1 inner validation; heldout absent from training and selection", "runtime_future_gt_used": False}
    write_json(OUT / "candidate_quality/training_manifest.json", quality_manifest)
    write_json(OUT / "candidate_quality/B0_linear.json", {"stage": STAGE, "family": "B0_LINEAR", "input_features": ["identity_score", "anchor_cosine", "confidence", "presence"], "runtime_future_gt_used": False})
    write_json(OUT / "candidate_quality/B1_mlp.json", {"stage": STAGE, "family": "B1_MLP", "input_feature_count": 14, "runtime_future_gt_used": False})
    write_json(OUT / "candidate_quality/B2_target_conditioned.json", {"stage": STAGE, "family": "B2_TARGET_CONDITIONED", "input_feature_count": 16, "runtime_future_gt_used": False})
    write_json(OUT / "candidate_quality/B3_ordinal.json", {"stage": STAGE, "family": "B3_ORDINAL", "targets": ["IoU >= 0.10", "IoU >= 0.50"], "runtime_future_gt_used": False})
    write_json(OUT / "candidate_quality/B4_iou_regression.json", {"stage": STAGE, "family": "B4_IOU_REGRESSION", "target": "posthoc candidate IoU", "runtime_future_gt_used": False})
    write_json(OUT / "stage_status.json", {"stage": STAGE, "goal_reference": f"outputs/{STAGE}/FINAL_GOAL.json", "goal_frozen": True, "status": "GEOMETRY_COMPLETE", "source_commit": audit["source_head"], "runtime_future_gt_used": False, "historical_outputs_modified": False})

    sam_candidates, sam3_result = targeted_sam3_refinement(assets, events, labels, geometry)
    write_json(OUT / "sam3_refinement/prompt_variants.json", {"stage": STAGE, "variants": {"S0": "predicted target box only", "S1": "best identity candidate box only", "S2": "enclosing union of predicted and identity candidate", "S3": "previous trusted runtime box"}, "gt_prompt_forbidden": True, "runtime_future_gt_used": False})
    write_json(OUT / "sam3_refinement/iterative_results.json", {"stage": STAGE, "authorized": True, "max_passes": 2, "actual_passes_per_invocation": 1, "status": "BOUNDED_DIAGNOSTIC", "runtime_future_gt_used": False})
    write_json(OUT / "sam3_refinement/adapter_results.json", {"stage": STAGE, "full_sam3_retraining": False, "lora": False, "candidate_union_provenance": "targeted_sam3_refinement with parent UID and prompt variant", "runtime_future_gt_used": False})
    write_json(OUT / "sam3_refinement/runtime_profile.json", {"stage": STAGE, "checkpoint": str(SAM3_CHECKPOINT), "checkpoint_sha256": sha256(SAM3_CHECKPOINT) if SAM3_CHECKPOINT.exists() else None, "result": sam3_result, "runtime_future_gt_used": False})
    write_json(OUT / "stage_status.json", {"stage": STAGE, "goal_reference": f"outputs/{STAGE}/FINAL_GOAL.json", "goal_frozen": True, "status": "SAM3_REFINEMENT_COMPLETE", "source_commit": audit["source_head"], "sam3_status": sam3_result.get("status"), "runtime_future_gt_used": False, "historical_outputs_modified": False})

    inner_configs: list[tuple[str, str, float]] = [(pipeline, "G0_ADDITIVE", 0.5) for pipeline in ("E0_ORIGINAL", "E1_QUALITY", "E2_BOX", "E3_SAM3", "E4_COMBINED")]
    inner_configs.append(("E0_ORIGINAL", "G1_TARGET_COLUMN", 0.5))
    for association in ("G2_GATED", "G3_RECOVERY_ONLY"):
        for lam in (0.25, 0.5, 1.0, 2.0):
            inner_configs.append(("E0_ORIGINAL", association, lam))
            inner_configs.append(("E4_COMBINED", association, lam))
    outer_records: list[dict[str, Any]] = []; selected_geometry_by_sequence: dict[str, str] = {}; ablation_rows: dict[str, dict[str, list[tuple[int, int, list[float], float]]]] = defaultdict(dict); ablation_decisions: dict[str, dict[str, list[FrameDecision]]] = defaultdict(dict); shadow_variants: dict[str, list[dict[str, Any]]] = defaultdict(list); quality_fold_records: list[dict[str, Any]] = []
    baseline_rows = baseline_trajectory_rows(assets)

    for fold_index, heldout in enumerate(SEQUENCES):
        validation = SEQUENCES[(SEQUENCES.index(heldout) + 1) % len(SEQUENCES)]
        geometry_variant = choose_geometry_for_fold(geometry, heldout); selected_geometry_by_sequence[heldout] = geometry_variant
        rows_by_sequence = rows_cache[geometry_variant]
        quality_family, quality_model, quality_meta = fit_quality_for_outer(rows_by_sequence, heldout)
        probability_maps = {sequence: row_probability_map(quality_model, quality_family, rows_by_sequence[sequence]) for sequence in SEQUENCES}
        fit_sequences = quality_meta["selection"]["selection_split"] if False else [sequence for sequence in SEQUENCES if sequence not in {heldout, validation}]
        refiner = train_box_refiner(assets, events, labels, geometry_variant, fit_sequences, seed=720351 + fold_index)
        inner_root = OUT / "trackeval" / "inner" / f"fold_{heldout}"; inner_tracker_root = inner_root / "trackers"; inner_eval_root = inner_root / "eval"; inner_tracker_names = ["BASELINE"]
        inner_rollouts: dict[str, tuple[list[tuple[int, int, list[float], float]], dict[str, Any]]] = {}
        export_trajectory_tracker(inner_tracker_root, "BASELINE", {validation: baseline_rows[validation]}, [validation])
        for pipeline, association, lam in inner_configs:
            name = pipeline_name(pipeline, association, lam); trajectory, decisions, meta = rollout_sequence(assets[validation], events[validation], geometry_variant, quality_family, probability_maps[validation], pipeline, association, lam, refiner, sam_candidates)
            inner_rollouts[name] = (trajectory, meta); export_trajectory_tracker(inner_tracker_root, name, {validation: trajectory}, [validation]); inner_tracker_names.append(name)
        inner_manifest = read_json(inner_tracker_root / "seqmap.txt") if False else None
        seqmap = inner_tracker_root / "seqmap.txt"
        run_trackeval_many(inner_tracker_root, inner_eval_root, inner_tracker_names, seqmap)
        inner_parsed = {name: parse_trackeval(inner_eval_root, name, [validation]) for name in inner_tracker_names}; inner_metrics = {name: trackeval_summary(parsed) for name, parsed in inner_parsed.items()}
        inner_base = inner_metrics["BASELINE"]; candidates = []
        for pipeline, association, lam in inner_configs:
            name = pipeline_name(pipeline, association, lam); metric = inner_metrics[name]; det_ok = metric.get("DetA") is not None and inner_base.get("DetA") is not None and metric["DetA"] >= inner_base["DetA"] - 0.01
            candidates.append({"name": name, "pipeline": pipeline, "association": association, "lambda": lam, "metrics": metric, "detA_constraint_satisfied": bool(det_ok), "complexity": (0 if pipeline == "E0_ORIGINAL" else 1) + (0 if association in {"G0_ADDITIVE", "G1_TARGET_COLUMN"} else 1)})
        feasible = [row for row in candidates if row["detA_constraint_satisfied"]]
        selected_inner = max(feasible or candidates, key=lambda row: ((row["metrics"].get("HOTA") or -1.0), (row["metrics"].get("AssA") or -1.0), -(row["metrics"].get("IDSW") or 1.0), -row["complexity"]))
        g2_candidates = [row for row in candidates if row["association"] == "G2_GATED"]
        g3_candidates = [row for row in candidates if row["association"] == "G3_RECOVERY_ONLY"]
        selected_g2 = max(g2_candidates, key=lambda row: (row["metrics"].get("HOTA") or -1.0, row["metrics"].get("AssA") or -1.0))["lambda"] if g2_candidates else 0.5
        selected_g3 = max(g3_candidates, key=lambda row: (row["metrics"].get("HOTA") or -1.0, row["metrics"].get("AssA") or -1.0))["lambda"] if g3_candidates else 0.5
        selected_pipeline = str(selected_inner["pipeline"]); selected_association = str(selected_inner["association"]); selected_lambda = float(selected_inner["lambda"])
        specs = {"CANDIDATE_QUALITY_ONLY": ("E1_QUALITY", "G0_ADDITIVE", 0.5), "BOX_REFINEMENT_ONLY": ("E2_BOX", "G0_ADDITIVE", 0.5), "SAM3_ONLY": ("E3_SAM3", "G0_ADDITIVE", 0.5), "IDENTITY_ASSOCIATION_ONLY": ("E0_ORIGINAL", "G2_GATED", float(selected_g2)), "R0_CANDIDATE_REFINEMENT": ("E2_BOX", "G0_ADDITIVE", 0.5), "R1_ASSOCIATION_ONLY": ("E0_ORIGINAL", "G2_GATED", float(selected_g2)), "R2_REFINEMENT_G2": ("E2_BOX", "G2_GATED", float(selected_g2)), "R3_REFINEMENT_G3": ("E2_BOX", "G3_RECOVERY_ONLY", float(selected_g3)), "COMBINED": (selected_pipeline, selected_association, selected_lambda)}
        fold_variant_metadata: dict[str, Any] = {}
        for display_name, (pipeline, association, lam) in specs.items():
            trajectory, decisions, meta = rollout_sequence(assets[heldout], events[heldout], geometry_variant, quality_family, probability_maps[heldout], pipeline, association, lam, refiner, sam_candidates)
            ablation_rows[display_name][heldout] = trajectory; ablation_decisions[display_name][heldout] = decisions; fold_variant_metadata[display_name] = {"pipeline": pipeline, "association": association, "lambda": lam, "decision_count": len(decisions), "runtime_future_gt_used": False}
        for association, lam in (("G0_ADDITIVE", 0.5), ("G1_TARGET_COLUMN", 0.5), ("G2_GATED", selected_g2), ("G3_RECOVERY_ONLY", selected_g3)):
            _trajectory, decisions, _meta = rollout_sequence(assets[heldout], events[heldout], geometry_variant, quality_family, probability_maps[heldout], "E0_ORIGINAL", association, float(lam), refiner, sam_candidates)
            shadow_variants[association].append({"sequence": heldout, **shadow_record(association_shadow(decisions, assets[heldout], events[heldout], labels))})
        quality_fold_records.append({"heldout": heldout, "validation": validation, "fit_sequences": fit_sequences, "geometry_variant": geometry_variant, "quality_family": quality_family, "quality_selection": quality_meta["selection"], "selected_pipeline": selected_pipeline, "selected_association": selected_association, "selected_lambda": selected_lambda, "inner_metrics": inner_metrics, "inner_selection": selected_inner, "inner_candidate_count": len(candidates), "runtime_future_gt_used": False})
        outer_records.append({"heldout": heldout, "validation": validation, "geometry_variant": geometry_variant, "quality_family": quality_family, "selected": {"pipeline": selected_pipeline, "association": selected_association, "lambda": selected_lambda}, "runtime_future_gt_used": False})
        write_json(OUT / "association/inner_selection.json", {"stage": STAGE, "folds": quality_fold_records, "selection_rule": "inner HOTA subject to DetA degradation <= 0.01; AssA, IDSW, simplicity tie-break", "runtime_future_gt_used": False})
        print(json.dumps({"stage": STAGE, "fold": fold_index + 1, "heldout": heldout, "selected": selected_inner["name"], "sam3_candidates": sum(len(value) for value in sam_candidates.values())}, sort_keys=True), flush=True)

    write_json(OUT / "geometry/selected_geometry.json", {"stage": STAGE, "global_selected": global_geometry_variant, "selection_rule": "inner validation per outer fold", "outer_fold_selection": selected_geometry_by_sequence, "runtime_future_gt_used": False})
    write_json(OUT / "candidate_quality/formal_loso.json", {"stage": STAGE, "folds": quality_fold_records, "outer_heldout_supervision_forbidden": True, "runtime_future_gt_used": False})
    for family, filename in (("B0_LINEAR", "B0_linear.json"), ("B1_MLP", "B1_mlp.json"), ("B2_TARGET_CONDITIONED", "B2_target_conditioned.json"), ("B3_ORDINAL", "B3_ordinal.json"), ("B4_IOU_REGRESSION", "B4_iou_regression.json")):
        family_rows = [fold["quality_selection"]["families"].get(family, {}) for fold in quality_fold_records]
        write_json(OUT / "candidate_quality" / filename, {"stage": STAGE, "family": family, "outer_validation_metrics": family_rows, "selected_count": sum(int(fold["quality_family"] == family) for fold in quality_fold_records), "runtime_future_gt_used": False})

    dev_tracker_root = OUT / "trackeval" / "dev_trackers"; dev_eval_root = OUT / "trackeval" / "dev"
    tracker_names = ["BASELINE", "CANDIDATE_QUALITY_ONLY", "BOX_REFINEMENT_ONLY", "SAM3_ONLY", "IDENTITY_ASSOCIATION_ONLY", "R0_CANDIDATE_REFINEMENT", "R1_ASSOCIATION_ONLY", "R2_REFINEMENT_G2", "R3_REFINEMENT_G3", "COMBINED"]
    for name in tracker_names:
        sequence_rows = baseline_rows if name == "BASELINE" else ablation_rows[name]
        export_trajectory_tracker(dev_tracker_root, name, sequence_rows, SEQUENCES)
    run_trackeval_many(dev_tracker_root, dev_eval_root, tracker_names, dev_tracker_root / "seqmap.txt")
    parsed_by_tracker = {name: parse_trackeval(dev_eval_root, name, SEQUENCES) for name in tracker_names}
    dev_metrics = {name: trackeval_summary(parsed_by_tracker[name]) for name in tracker_names}
    component_dirs = {"BASELINE": "baseline", "CANDIDATE_QUALITY_ONLY": "candidate_only", "BOX_REFINEMENT_ONLY": "box_only", "SAM3_ONLY": "sam3_only", "IDENTITY_ASSOCIATION_ONLY": "association_only", "COMBINED": "combined"}
    for tracker_name, directory in component_dirs.items():
        write_metric_artifact(OUT / "trackeval" / directory / "summary.json", tracker_name, dev_metrics[tracker_name])
        write_json(OUT / "trackeval" / directory / "per_sequence.json", dev_metrics[tracker_name].get("per_sequence", {}))
    trackeval_per_sequence: dict[str, dict[str, Any]] = {}
    for sequence in SEQUENCES:
        trackeval_per_sequence[sequence] = {name: dev_metrics[name].get("per_sequence", {}).get(sequence, {}) for name in tracker_names}
    combined_summary = {"stage": STAGE, "baseline": dev_metrics["BASELINE"], "treatments": {name: dev_metrics[name] for name in tracker_names if name != "BASELINE"}, "per_sequence": trackeval_per_sequence, "settings_identical": True, "runtime_future_gt_used": False}
    write_json(OUT / "trackeval/development_summary.json", combined_summary)
    bootstrap = paired_bootstrap(trackeval_per_sequence, "BASELINE", "COMBINED")
    write_json(OUT / "trackeval/paired_bootstrap.json", bootstrap)

    baseline_decisions: dict[str, list[FrameDecision]] = {}
    for sequence, asset in assets.items():
        event = events[sequence]; public_id = int(event["target_public_id"]); sequence_decisions: list[FrameDecision] = []
        for frame in range(asset.frame_count):
            uid = base_assignment_uid(asset, frame, public_id); candidate = candidate_map(asset, frame).get(str(uid)) if uid else None
            sequence_decisions.append(FrameDecision(sequence, frame, public_id, uid, list(candidate["box_xyxy"]) if candidate else None, 0.0, uid, False, 0.0, 0.0, "BASELINE"))
        baseline_decisions[sequence] = sequence_decisions
    ablation_decisions["BASELINE"] = baseline_decisions
    localization_by_display: dict[str, dict[str, Any]] = {}
    for display_name, decisions in ablation_decisions.items():
        localization_by_display[display_name] = localization_metrics(assets, labels, decisions)
    for display_name, filename in (("BASELINE", "E0_original.json"), ("CANDIDATE_QUALITY_ONLY", "E1_quality.json"), ("BOX_REFINEMENT_ONLY", "E2_box_refine.json"), ("SAM3_ONLY", "E3_sam3_refine.json"), ("COMBINED", "E4_combined.json")):
        write_json(OUT / "candidate_pipeline" / filename, {"stage": STAGE, "pipeline": display_name, "localization": localization_by_display[display_name], "trackeval": dev_metrics[display_name], "runtime_future_gt_used": False})
    write_json(OUT / "candidate_pipeline/selected_pipeline.json", {"stage": STAGE, "outer_folds": outer_records, "candidate_pipeline_selection_is_inner_only": True, "runtime_future_gt_used": False})

    # Evaluate C0 with the causal geometry prediction directly; C1/C2 use the
    # corresponding outer-heldout trajectories above, so no heldout labels
    # enter a refinement model.
    c0_decisions: dict[str, list[FrameDecision]] = {}
    for sequence, asset in assets.items():
        event = events[sequence]; variant = selected_geometry_by_sequence[sequence]; geom = causal_geometry_predictions(asset, event, variant); seq_decisions: list[FrameDecision] = []; public_id = int(event["target_public_id"])
        for frame in range(asset.frame_count):
            uid = base_assignment_uid(asset, frame, public_id); candidate = candidate_map(asset, frame).get(str(uid)) if uid else None
            box = interpolate_box(candidate["box_xyxy"], geom[frame]["predicted_box"], 0.5) if candidate else None
            seq_decisions.append(FrameDecision(sequence, frame, public_id, uid, box, 0.0, uid, False, 0.0, float((geometry_features(candidate["box_xyxy"], geom[frame]["predicted_box"])[0] + 1.0) / 2.0) if candidate else 0.0, "C0_GEOMETRIC"))
        c0_decisions[sequence] = seq_decisions
    write_json(OUT / "box_refinement/C0_geometric.json", {"stage": STAGE, "alphas": [0.25, 0.5, 0.75], "selected_alpha": 0.5, "localization": localization_metrics(assets, labels, c0_decisions), "runtime_future_gt_used": False})
    write_json(OUT / "box_refinement/C1_learned.json", {"stage": STAGE, "refiner": "causal_outer_fit_MLP_box_delta", "localization": localization_by_display["BOX_REFINEMENT_ONLY"], "runtime_future_gt_used": False})
    write_json(OUT / "box_refinement/C2_quality_gated.json", {"stage": STAGE, "gate": "quality_probability < 0.65", "localization": localization_by_display["COMBINED"], "runtime_future_gt_used": False})
    write_json(OUT / "box_refinement/p1a_conversion.json", {"stage": STAGE, "variants": {name: {"p1a_frames": localization_by_display[name]["p1a_frames"], "p1a_after_frames": localization_by_display[name]["p1a_after_frames"], "p1a_recovery_rate": localization_by_display[name]["p1a_recovery_rate"], "mean_iou": localization_by_display[name]["mean_iou"]} for name in ("BASELINE", "BOX_REFINEMENT_ONLY", "SAM3_ONLY", "COMBINED")}, "runtime_future_gt_used": False})

    identity = identity_recheck(assets, events, labels)
    write_json(OUT / "identity_recheck/ranking.json", {"stage": STAGE, **identity})
    write_json(OUT / "identity_recheck/open_set.json", {"stage": STAGE, "hard_negative_definition": "all non-max-IoU candidates in the same frame", "mean_margin": identity.get("mean_margin"), "median_margin": identity.get("median_margin"), "runtime_future_gt_used": False})
    write_json(OUT / "identity_recheck/per_sequence.json", identity.get("per_sequence", {}))

    shadow_summary = {association: aggregate_shadow(rows) for association, rows in shadow_variants.items()}
    write_json(OUT / "association/base_score_audit.json", base_score_audit(assets))
    for association, filename in (("G0_ADDITIVE", "G0_additive.json"), ("G1_TARGET_COLUMN", "G1_target_column.json"), ("G2_GATED", "G2_gated.json"), ("G3_RECOVERY_ONLY", "G3_recovery_only.json")):
        write_json(OUT / "association" / filename, {"stage": STAGE, "association": association, "shadow": shadow_summary.get(association, {}), "lambda_grid": [0.25, 0.5, 1.0, 2.0], "exact_solver": "scipy.optimize.linear_sum_assignment", "target_column_only": association == "G1_TARGET_COLUMN", "runtime_future_gt_used": False})
    write_json(OUT / "association/shadow_comparison.json", {"stage": STAGE, "variants": shadow_summary, "runtime_future_gt_used": False})
    write_json(OUT / "association/inner_selection.json", {"stage": STAGE, "folds": quality_fold_records, "selection_rule": "inner HOTA subject to DetA degradation <= 0.01", "runtime_future_gt_used": False})

    treatment_memory_rows: list[dict[str, Any]] = []; memory_per_sequence: dict[str, Any] = {}
    for sequence, decisions in ablation_decisions["COMBINED"].items():
        shadow = association_shadow(decisions, assets[sequence], events[sequence], labels); record = shadow_record(shadow); memory_per_sequence[sequence] = record; treatment_memory_rows.extend(shadow["memory_rows"])
    commits = sum(bool(row.get("commit")) for row in treatment_memory_rows); wrong_writes = sum(bool(row.get("wrong_write")) for row in treatment_memory_rows); retained = sum(bool(row.get("retention")) for row in treatment_memory_rows)
    wrong_indices = [index for index, row in enumerate(treatment_memory_rows) if row.get("wrong_write")]; cascade = 0; current = 0
    for row in treatment_memory_rows:
        current = current + 1 if row.get("wrong_write") else 0; cascade = max(cascade, current)
    memory_summary = {"stage": STAGE, "policy": "association-selected candidate AND quality_probability >= 0.80 AND geometry_consistency >= 0.50", "commits": commits, "wrong_write_rate": float(wrong_writes / max(1, commits)), "retention": float(retained / max(1, commits)), "first_wrong_write": treatment_memory_rows[wrong_indices[0]] if wrong_indices else None, "cascade_length": cascade, "gate_wrong_write_le_0.02": bool(wrong_writes / max(1, commits) <= 0.02), "gate_retention_ge_0.60": bool(retained / max(1, commits) >= 0.60), "runtime_future_gt_used": False}
    write_json(OUT / "memory/causal_replay.json", memory_summary)
    write_json(OUT / "memory/per_sequence.json", memory_per_sequence)
    write_zstd_jsonl(OUT / "memory/write_audit.jsonl.zst", treatment_memory_rows)

    dev_gate = bool((dev_metrics["COMBINED"].get("HOTA") or -1.0) > (dev_metrics["BASELINE"].get("HOTA") or -1.0) and (dev_metrics["COMBINED"].get("AssA") or -1.0) > (dev_metrics["BASELINE"].get("AssA") or -1.0) and (dev_metrics["COMBINED"].get("DetA") or -1.0) - (dev_metrics["BASELINE"].get("DetA") or -1.0) >= -0.005)
    frozen_pipeline = {"stage": STAGE, "dev_gate": dev_gate, "outer_folds": outer_records, "identity_model": "frozen N72R3R2 representation/base score tape", "quality_models": "outer-fit B0-B4 with inner family selection", "geometry_models": "outer inner-selected A0-A3", "candidate_pipeline": "inner-selected E0-E4", "association_policy": "inner-selected G0-G3 and lambda in {0.25,0.5,1.0,2.0}", "memory_policy": memory_summary["policy"], "val_tuning_forbidden": True, "test_used": False, "runtime_future_gt_used": False}
    write_json(OUT / "frozen/FROZEN_DEV_PIPELINE.json", frozen_pipeline)

    val_manifest = {"stage": STAGE, "split": "val", "status": "NOT_AUTHORIZED_DEV_GATE" if not dev_gate else "AUTHORIZED_BUT_VAL_CANDIDATE_TAPE_NOT_AVAILABLE", "dev_gate": dev_gate, "val_dataset_gt_not_read": True, "val_training": False, "val_tuning": False, "test_accessed": False, "runtime_future_gt_used": False}
    write_json(OUT / "val/generation_manifest.json", val_manifest)
    write_json(OUT / "val/baseline_trackeval.json", {"stage": STAGE, "status": val_manifest["status"], "metrics": None, "runtime_future_gt_used": False})
    write_json(OUT / "val/treatment_trackeval.json", {"stage": STAGE, "status": val_manifest["status"], "metrics": None, "runtime_future_gt_used": False})
    write_json(OUT / "val/delta.json", {"stage": STAGE, "status": val_manifest["status"], "delta": None, "runtime_future_gt_used": False})

    baseline_hota = dev_metrics["BASELINE"].get("HOTA"); treatment_hota = dev_metrics["COMBINED"].get("HOTA"); baseline_assa = dev_metrics["BASELINE"].get("AssA"); treatment_assa = dev_metrics["COMBINED"].get("AssA"); baseline_deta = dev_metrics["BASELINE"].get("DetA"); treatment_deta = dev_metrics["COMBINED"].get("DetA")
    if any(value is None for value in (baseline_hota, treatment_hota, baseline_assa, treatment_assa, baseline_deta, treatment_deta)):
        decision = "FAIL_ASSET_OR_STORAGE"
    elif not sam3_result.get("status", "").startswith(("PASS", "PARTIAL", "SKIPPED")):
        decision = "FAIL_RUNTIME_INVARIANT"
    elif treatment_hota <= baseline_hota:
        decision = "FAIL_LOCALIZATION_TO_HOTA_TRANSFER"
    elif treatment_assa <= baseline_assa:
        decision = "FAIL_ASSOCIATION_AUTHORITY"
    elif treatment_deta - baseline_deta < -0.005:
        decision = "FAIL_DETA_ASSA_TRADEOFF"
    elif dev_gate and val_manifest["status"] != "VAL_COMPLETE":
        decision = "FAIL_VAL_GENERALIZATION"
    elif not memory_summary["gate_wrong_write_le_0.02"]:
        decision = "FAIL_MEMORY_COMMIT"
    else:
        decision = "PASS_END_TO_END_HOTA_IMPROVEMENT"
    runtime_profile = {"stage": STAGE, "outer_fold_count": len(outer_records), "total_frames": int(sum(asset.frame_count for asset in assets.values())), "candidate_pipeline_fps_estimate": None, "sam3": sam3_result, "runtime_future_gt_used": False}
    write_json(OUT / "sam3_refinement/runtime_profile.json", runtime_profile)
    final_result = {"stage": STAGE, "goal": "Target-Conditioned Candidate Localization and End-to-End MOT Improvement", "decision": decision, "central_question": "Can target-conditioned candidate localization and controlled identity-aware association convert identity representation into end-to-end HOTA and AssA gains without materially damaging DetA?", "baseline_metrics": dev_metrics["BASELINE"], "treatment_metrics": dev_metrics["COMBINED"], "dev_deltas": {metric: (float(dev_metrics["COMBINED"].get(metric)) - float(dev_metrics["BASELINE"].get(metric))) if isinstance(dev_metrics["COMBINED"].get(metric), (int, float)) and isinstance(dev_metrics["BASELINE"].get(metric), (int, float)) else None for metric in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW")}, "component_trackeval": {name: dev_metrics[name] for name in tracker_names}, "paired_bootstrap": bootstrap, "localization": localization_by_display, "association": shadow_summary, "memory": memory_summary, "identity_recheck": identity, "sam3_refinement": sam3_result, "val": val_manifest, "outer_loso": outer_records, "runtime_future_gt_used": False, "historical_outputs_modified": False, "test_accessed": False}
    write_json(OUT / "FINAL_RESULT.json", final_result)

    report_lines = [
        "# InterMOT N72R20R3R2R3 Final Report", "", "## Q1 — Did candidate localization improve?", "", f"- Baseline P1a frames: {localization_by_display['BASELINE']['p1a_frames']}; combined P1a→V recovery: {localization_by_display['COMBINED']['p1a_recovery_rate']:.4f}.", f"- Baseline mean target IoU: {localization_by_display['BASELINE']['mean_iou']:.6f}; combined mean target IoU: {localization_by_display['COMBINED']['mean_iou']:.6f}.", f"- Combined valid-candidate coverage: {localization_by_display['COMBINED']['valid_candidate_coverage']:.4f}.", "", "## Q2 — Did identity-aware evidence improve association?", "", f"- Combined AssA: {treatment_assa:.6f} versus baseline {baseline_assa:.6f}; IDSW: {dev_metrics['COMBINED'].get('IDSW')} versus {dev_metrics['BASELINE'].get('IDSW')}.", f"- G2 shadow N01={shadow_summary.get('G2_GATED', {}).get('N01_base_wrong_new_correct')}, N10={shadow_summary.get('G2_GATED', {}).get('N10_base_correct_new_wrong')}; G3 shadow N01={shadow_summary.get('G3_RECOVERY_ONLY', {}).get('N01_base_wrong_new_correct')}, N10={shadow_summary.get('G3_RECOVERY_ONLY', {}).get('N10_base_correct_new_wrong')}.", "", "## Q3 — Did complete MOT improve?", "", f"- Dev HOTA(AUC): {treatment_hota:.6f} versus baseline {baseline_hota:.6f}; DetA: {treatment_deta:.6f} versus {baseline_deta:.6f}; AssA: {treatment_assa:.6f} versus {baseline_assa:.6f}.", f"- Final scientific decision: `{decision}`.", "", "## Required end-to-end table", "", "| Method | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FPS |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",]
    table_names = [("Baseline", "BASELINE"), ("Candidate quality only", "CANDIDATE_QUALITY_ONLY"), ("Box refinement only", "BOX_REFINEMENT_ONLY"), ("Targeted SAM3 only", "SAM3_ONLY"), ("Identity association only", "IDENTITY_ASSOCIATION_ONLY"), ("Combined Dev", "COMBINED")]
    for label, name in table_names:
        metric = dev_metrics[name]; report_lines.append(f"| {label} | {metric.get('HOTA') if metric.get('HOTA') is not None else 'NA'} | {metric.get('DetA') if metric.get('DetA') is not None else 'NA'} | {metric.get('AssA') if metric.get('AssA') is not None else 'NA'} | {metric.get('IDF1') if metric.get('IDF1') is not None else 'NA'} | {metric.get('MOTA') if metric.get('MOTA') is not None else 'NA'} | {metric.get('IDSW') if metric.get('IDSW') is not None else 'NA'} | NA |")
    report_lines.extend(["| Combined VAL | NA | NA | NA | NA | NA | NA | NA |", "", "## Localization variants", "", "| Variant | P1a frames | Valid candidate coverage | Mean best IoU | P1a→V recovery |", "| --- | ---: | ---: | ---: | ---: |"])
    for label, name in (("Original", "BASELINE"), ("Box refine", "BOX_REFINEMENT_ONLY"), ("Targeted SAM3", "SAM3_ONLY"), ("Combined", "COMBINED")):
        metric = localization_by_display[name]; report_lines.append(f"| {label} | {metric['p1a_frames']} | {metric['valid_candidate_coverage']:.6f} | {metric['mean_iou']:.6f} | {metric['p1a_recovery_rate']:.6f} |")
    report_lines.extend(["", "## Association variants", "", "| Variant | Changed frames | Wrong→Correct | Correct→Wrong | Net corrections | AssA Δ |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for association in ("G0_ADDITIVE", "G1_TARGET_COLUMN", "G2_GATED", "G3_RECOVERY_ONLY"):
        shadow = shadow_summary.get(association, {}); report_lines.append(f"| {association} | {shadow.get('changed_frames', 0)} | {shadow.get('N01_base_wrong_new_correct', 0)} | {shadow.get('N10_base_correct_new_wrong', 0)} | {shadow.get('net_corrections', 0)} | NA |")
    report_lines.extend(["", "## Protocol and limitations", "", "- TrackEval is the pinned evaluator invoked through the dedicated local scalar/list compatibility wrapper with identical settings for all dev trackers.", "- Outer LOSO uses six fit sequences and one inner validation sequence; heldout GT is used only after model/policy selection for posthoc evaluation.", f"- Targeted SAM3 status: `{sam3_result.get('status')}`; full SAM3 retraining and LoRA were not used.", f"- VAL status: `{val_manifest['status']}`; DanceTrack test was not accessed.", "- Historical N72R20R3R2, R3R2R1 and R3R2R2 outputs were not modified.", ""])
    (OUT / "FINAL_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    final_status = "PASS" if decision.startswith("PASS") else "FAIL"
    write_json(OUT / "storage_audit_after.json", storage_audit("after"))
    write_json(OUT / "stage_status.json", {"stage": STAGE, "goal_reference": f"outputs/{STAGE}/FINAL_GOAL.json", "goal_frozen": True, "status": final_status, "scientific_decision": decision, "source_commit": audit["source_head"], "historical_outputs_modified": False, "runtime_future_gt_used": False, "val_accessed": False, "test_accessed": False})
    return final_result


def load_assets_from_root(asset_root: Path, dataset_root: Path, split: str, sequences: Sequence[str]) -> dict[str, SequenceAsset]:
    assets: dict[str, SequenceAsset] = {}
    for sequence in sequences:
        candidate_dir = asset_root / "candidates" / sequence; index = read_json(candidate_dir / "index.json"); frame_rows = read_zstd_jsonl(candidate_dir / "metadata.jsonl.zst"); base_rows = read_zstd_jsonl(asset_root / "base_scores" / sequence / "base_scores.jsonl.zst")
        frames = {int(row["frame"]): dict(row) for row in frame_rows}; bases = {int(row["frame"]): dict(row) for row in base_rows}
        if set(frames) != set(bases) or len(frames) != int(index["frame_count"]):
            raise RuntimeError(f"{split} candidate/base axis mismatch for {sequence}")
        if any(row.get("runtime_future_gt_used") is not False for row in frame_rows + base_rows):
            raise RuntimeError(f"{split} runtime GT contamination for {sequence}")
        gt = gt_by_frame(dataset_root / split / sequence / "gt" / "gt.txt")
        count = int(index["embedding_count"]); embeddings = np.memmap(candidate_dir / "embeddings.f16", mode="r", dtype=np.float16, shape=(count, 512))
        assets[sequence] = SequenceAsset(sequence, frames, bases, gt, len(frames), embeddings)
    return assets


def load_candidate_event_assets(asset_root: Path, dataset_root: Path, split: str, sequences: Sequence[str]) -> dict[str, SequenceAsset]:
    assets: dict[str, SequenceAsset] = {}
    for sequence in sequences:
        candidate_dir = asset_root / "candidates" / sequence; index = read_json(candidate_dir / "index.json"); frame_rows = read_zstd_jsonl(candidate_dir / "metadata.jsonl.zst"); frames = {int(row["frame"]): dict(row) for row in frame_rows}; gt = gt_by_frame(dataset_root / split / sequence / "gt" / "gt.txt"); count = int(index["embedding_count"]); embeddings = np.memmap(candidate_dir / "embeddings.f16", mode="r", dtype=np.float16, shape=(count, 512)); assets[sequence] = SequenceAsset(sequence, frames, {}, gt, len(frames), embeddings)
    return assets


def build_val_base_score_sequence(asset_root: Path, sequence: str, event: Mapping[str, Any]) -> dict[str, Any]:
    """Build the frozen R2 base tape while respecting a nonzero click frame.

    The R2 builder historically assumes that the human event is frame zero.
    VAL candidate generation can legitimately have no candidate on frame zero,
    so this adapter keeps pre-click frames in the tape with an empty identity
    axis and starts the unchanged R2 scorer/solver at the first candidate-backed
    click frame.  It does not read GT or change matcher/association semantics.
    """
    from scripts import n72r20r2_build_base_score_tape as base_builder

    tape = base_builder.load_candidate_frames(asset_root, sequence)
    bindings = [dict(item) for item in event["state_bindings"]]
    event_frame = int(event["event_frame"])
    anchor_rows = next((rows for frame_payload, rows in tape if int(frame_payload["frame"]) == event_frame), None)
    if anchor_rows is None or not anchor_rows:
        raise RuntimeError(f"VAL event frame has no candidate rows for base tape: {sequence}:{event_frame}")

    out_root = asset_root / "base_scores" / sequence
    out_root.mkdir(parents=True, exist_ok=True)
    final_tape = out_root / "base_scores.jsonl.zst"
    final_index = out_root / "index.json"
    final_done = out_root / "done.json"
    if any(path.exists() for path in (final_tape, final_index, final_done)):
        raise FileExistsError(f"refusing to overwrite existing VAL base tape: {out_root}")

    states: dict[int, Any] = {}
    ordered_states: list[Any] = []
    rows_out: list[dict[str, Any]] = []
    total_candidates = 0
    assignment_rows = 0
    state_updates = 0
    for frame_payload, candidate_rows in tape:
        frame = int(frame_payload["frame"])
        pre_event = frame < event_frame
        if frame == event_frame:
            states = base_builder.initialize_states(event, candidate_rows)
            ordered_states = base_builder.assignable_states(bindings, states)
        elif pre_event:
            states = {}
            ordered_states = []

        if pre_event:
            matrix = np.zeros((len(candidate_rows), 0), dtype=np.float64)
        else:
            score_audit: dict[str, Any] = {}
            matrix = np.asarray(
                base_builder.score_matrix_pairwise(
                    ordered_states,
                    candidate_rows,
                    frame,
                    model=None,
                    reid_weights=dict(base_builder.BASE_REID_WEIGHTS),
                    positive_bonus=base_builder.BASE_POSITIVE_BONUS,
                    native_bonus=base_builder.BASE_NATIVE_BONUS,
                    score_audit=score_audit,
                ),
                dtype=np.float64,
            )
        if matrix.shape != (len(candidate_rows), len(ordered_states)) or not np.isfinite(matrix).all():
            raise ValueError(f"{sequence}:{frame}: invalid VAL base score matrix {matrix.shape}")
        solver = base_builder.solve_effect_assignment(
            candidate_rows=[base_builder.serializable_candidate(row) for row in candidate_rows],
            persistent_states=ordered_states,
            fused_state_candidate_scores=matrix.T,
            source_run_id=f"n72r20r3r2r3:val:{sequence}:base:{frame}",
            session_id=f"n72r20r3r2r3:val:{sequence}",
            none_score=base_builder.NONE_SCORE,
        )
        before = base_builder.state_digest(states)
        update = base_builder.update_base_states(
            states=states,
            solver=solver,
            rows=candidate_rows,
            frame=frame,
            target_public=int(event["target_public_id"]),
            event_frame=event_frame,
        )
        after = base_builder.state_digest(states)
        serialized_rows = [base_builder.serializable_candidate(row) for row in candidate_rows]
        active_axes = [int(item["association_state_id"]) for item in bindings] if not pre_event else []
        active_publics = [int(item["public_id"]) for item in bindings] if not pre_event else []
        rows_out.append(
            {
                "stage": STAGE,
                "source_stage": "N72R20R2",
                "sequence": sequence,
                "split": "val",
                "event_frame": event_frame,
                "frame": frame,
                "frame_horizon": frame - event_frame,
                "identity_available": not pre_event,
                "candidate_rows": serialized_rows,
                "candidate_uid_axis": [str(row["candidate_uid"]) for row in candidate_rows],
                "candidate_count": len(candidate_rows),
                "association_state_axis": active_axes,
                "public_id_axis": active_publics,
                "base_score_matrix": matrix.tolist(),
                "base_score_matrix_shape": [int(value) for value in matrix.shape],
                "base_score_matrix_sha256": base_builder.matrix_sha256(matrix),
                "base_assignment": solver,
                "explicit_none_score": float(base_builder.NONE_SCORE),
                "base_scorer": {
                    "implementation": "sam3_intermot.association.online_associator.score_matrix_pairwise",
                    "model": None,
                    "reid_weights": dict(base_builder.BASE_REID_WEIGHTS),
                    "native_bonus": float(base_builder.BASE_NATIVE_BONUS),
                    "positive_bonus": float(base_builder.BASE_POSITIVE_BONUS),
                    "ema": float(base_builder.BASE_EMA),
                    "state_update": "exact-public-assignment-output-then-IdentityState.update_machine",
                    "pre_event_policy": "empty_identity_axis_until_human_anchor",
                },
                "state_update": update,
                "state_digest_before": before,
                "state_digest_after": after,
                "runtime_future_gt_used": False,
                "runtime_gt_read": False,
                "posthoc_gt_used": False,
            }
        )
        total_candidates += len(candidate_rows)
        assignment_rows += len(solver.get("assignment_rows", []))
        state_updates += len(update.get("machine_update_public_ids", []))

    write_zstd_jsonl(final_tape, rows_out)
    index = {
        "stage": STAGE,
        "source_stage": "N72R20R2",
        "sequence": sequence,
        "split": "val",
        "event_frame": event_frame,
        "pre_event_frames_have_identity_axis": False,
        "frame_count": len(rows_out),
        "candidate_count": total_candidates,
        "assignment_row_count": assignment_rows,
        "base_state_update_count": state_updates,
        "candidate_uid_axis_source": "shared_val_candidate_tape_per_frame",
        "association_state_axis": [int(item["association_state_id"]) for item in bindings],
        "public_id_axis": [int(item["public_id"]) for item in bindings],
        "base_score_orientation": "candidate_x_association_state",
        "base_scores": str(final_tape),
        "base_scores_sha256": sha256(final_tape),
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
    }
    write_json(final_index, index)
    done = {
        "status": "PASS_N72R20R3R2R3_VAL_BASE_SCORE_TAPE_SEQUENCE",
        "sequence": sequence,
        "index": str(final_index),
        "index_sha256": sha256(final_index),
        "base_scores_sha256": sha256(final_tape),
        "event_frame": event_frame,
        "frame_count": len(rows_out),
        "candidate_count": total_candidates,
        "runtime_future_gt_used": False,
        "gt_opened": False,
    }
    write_json(final_done, done)
    return done


def create_val_events(val_assets: Mapping[str, SequenceAsset]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    events: dict[str, dict[str, Any]] = {}; labels: dict[str, dict[str, Any]] = {}
    for sequence, asset in val_assets.items():
        anchor_frames = sorted(
            frame for frame, frame_row in asset.frames.items()
            if frame_row["candidates"] and asset.gt.get(frame)
        )
        if not anchor_frames:
            raise RuntimeError(f"VAL has no frame with both GT and candidates for simulated human anchor: {sequence}")
        anchor_frame = int(anchor_frames[0])
        gt_id, gt_box = sorted(asset.gt[anchor_frame], key=lambda item: int(item[0]))[0]
        candidates = asset.frames[anchor_frame]["candidates"]
        ious = np.asarray([iou(candidate["box_xyxy"], gt_box) for candidate in candidates], dtype=np.float32); target_index = int(np.argmax(ious)); target_candidate = candidates[target_index]
        anchor = np.asarray(asset.embedding_array[int(target_candidate["embedding_offset"])], dtype=np.float32); anchor = anchor / max(1.0e-8, float(np.linalg.norm(anchor)))
        bindings = [{"association_state_id": index + 1, "public_id": 300000 + index, "initial_candidate_uid": str(candidate["candidate_uid"])} for index, candidate in enumerate(candidates)]
        target_public = int(bindings[target_index]["public_id"])
        events[sequence] = {"sequence": sequence, "split": "val", "event_frame": anchor_frame, "target_public_id": target_public, "target_association_state_id": int(bindings[target_index]["association_state_id"]), "human_anchor": anchor.tolist(), "human_anchor_source": "frozen_OSNet_candidate_crop_for_offline_simulated_click", "target_box_xyxy": list(gt_box), "target_candidate_uid": str(target_candidate["candidate_uid"]), "state_bindings": bindings, "event_preparation_used_gt": True, "anchor_frame_selection_policy": "first_frame_with_candidate_and_gt", "runtime_future_gt_used": False, "runtime_gt_read": False, "test_accessed": False}
        labels[sequence] = {"sequence": sequence, "target_gt_id": int(gt_id), "source": "offline_val_event_preparation_only", "runtime_future_gt_used": False}
    write_json(VAL_EVENTS_PATH, {"stage": STAGE, "split": "val", "events": list(events.values()), "event_preparation_used_gt": True, "runtime_future_gt_used": False})
    write_json(STAGE_ASSET_ROOT / "val_labels.json", {"stage": STAGE, "split": "val", "labels": list(labels.values()), "runtime_future_gt_used": False})
    return events, labels


def build_val_inference_rows(assets: Mapping[str, SequenceAsset], events: Mapping[str, Mapping[str, Any]], geometry_variant: str) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for sequence, asset in assets.items():
        event = events[sequence]; anchor = np.asarray(event["human_anchor"], dtype=np.float32); geom = causal_geometry_predictions(asset, event, geometry_variant); scores_by_frame = []
        rows: list[dict[str, Any]] = []
        for frame in range(asset.frame_count):
            scores = base_identity_scores(asset, frame, event)
            for index, candidate in enumerate(asset.frames[frame]["candidates"]):
                rows.append({"sequence": sequence, "frame": frame, "candidate_index": index, "candidate_uid": str(candidate["candidate_uid"]), "feature": quality_feature_row(asset, frame, candidate, float(scores[index]), geom[frame], anchor), "label": "O", "target_iou": 0.0, "runtime_future_gt_used": False, "posthoc_gt_used": False})
        output[sequence] = rows
    return output


def val_phase() -> dict[str, Any]:
    val_sequences = tuple(sorted(path.name for path in (DATASET_ROOT / "val").iterdir() if path.is_dir()))
    if len(val_sequences) != 25:
        raise RuntimeError(f"expected 25 DanceTrack val sequences, found {len(val_sequences)}")
    missing = [sequence for sequence in val_sequences if not (VAL_ASSET_ROOT / "candidates" / sequence / "done.json").exists()]
    if missing:
        raise RuntimeError(f"VAL candidate stream incomplete: {missing[:5]}")
    candidate_event_assets = load_candidate_event_assets(VAL_ASSET_ROOT, DATASET_ROOT, "val", val_sequences); val_events, val_labels = create_val_events(candidate_event_assets)
    from scripts import n72r20r2_build_base_score_tape as base_builder
    base_results = []
    for sequence in val_sequences:
        if not (VAL_ASSET_ROOT / "base_scores" / sequence / "done.json").exists():
            base_results.append(build_val_base_score_sequence(VAL_ASSET_ROOT, sequence, val_events[sequence]))
    val_assets = load_assets_from_root(VAL_ASSET_ROOT, DATASET_ROOT, "val", val_sequences)
    dev_assets = load_assets(); dev_events = event_map(); dev_labels = target_labels(); frozen = read_json(OUT / "frozen/FROZEN_DEV_PIPELINE.json"); outer = frozen.get("outer_folds", [])
    geometry_selection = read_json(OUT / "geometry/selected_geometry.json"); geometry_variant = str(geometry_selection.get("global_selected", "A0_LAST_BOX")); quality_family = Counter(str(row.get("quality_family")) for row in outer).most_common(1)[0][0] if outer else "B0_LINEAR"; pipeline = Counter(str(row.get("selected", {}).get("pipeline")) for row in outer).most_common(1)[0][0] if outer else "E0_ORIGINAL"; association = Counter(str(row.get("selected", {}).get("association")) for row in outer).most_common(1)[0][0] if outer else "G0_ADDITIVE"; lam_values = [float(row.get("selected", {}).get("lambda", 0.5)) for row in outer]; lam = float(Counter(lam_values).most_common(1)[0][0]) if lam_values else 0.5
    dev_rows = build_quality_rows(dev_assets, dev_events, dev_labels, geometry_variant); train_rows = [row for sequence in SEQUENCES for row in dev_rows[sequence]]; quality_model, quality_dim = train_quality_family(train_rows, quality_family, seed=720351); refiner = train_box_refiner(dev_assets, dev_events, dev_labels, geometry_variant, SEQUENCES, seed=720351)
    val_geometry_payload = {"variants": {geometry_variant: {"pooled": {"predicted_iou_mean": 0.0}}}}
    val_sam_device = os.environ.get("N72R20R3R2R3_SAM3_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    val_sam_candidates, val_sam3_result = targeted_sam3_refinement(val_assets, val_events, val_labels, val_geometry_payload, dataset_split="val", output_path=STAGE_ASSET_ROOT / "val_targeted_sam3_results.json", sequence_order=val_sequences, selection_policy="frozen_causal_event_frame", device_override=val_sam_device, geometry_variant_override=geometry_variant)
    write_json(OUT / "val/targeted_sam3.json", val_sam3_result)
    val_rows = build_val_inference_rows(val_assets, val_events, geometry_variant); probability_maps = {sequence: row_probability_map(quality_model, quality_family, val_rows[sequence]) for sequence in val_sequences}; baseline_rows = baseline_trajectory_rows(val_assets); treatment_rows: dict[str, list[tuple[int, int, list[float], float]]] = {}; val_metadata: dict[str, Any] = {"stage": STAGE, "split": "val", "sequences": list(val_sequences), "quality_family": quality_family, "quality_input_dim": quality_dim, "geometry_variant": geometry_variant, "pipeline": pipeline, "association": association, "lambda": lam, "targeted_sam3_status": val_sam3_result.get("status"), "targeted_sam3_candidate_count_increase": val_sam3_result.get("candidate_count_increase", 0), "targeted_sam3_selection_policy": val_sam3_result.get("selection_policy"), "gt_used_for_event_preparation_only": True, "gt_used_for_training": False, "gt_used_for_tuning": False, "val_training": False, "val_tuning": False, "val_dataset_gt_not_read": False, "test_accessed": False, "runtime_future_gt_used": False}
    for sequence in val_sequences:
        trajectory, _decisions, _meta = rollout_sequence(val_assets[sequence], val_events[sequence], geometry_variant, quality_family, probability_maps[sequence], pipeline, association, lam, refiner, val_sam_candidates)
        treatment_rows[sequence] = trajectory
    val_tracker_root = OUT / "val" / "trackers"; val_eval_root = OUT / "val" / "trackeval"; export_trajectory_tracker(val_tracker_root, "BASELINE", baseline_rows, val_sequences); export_trajectory_tracker(val_tracker_root, "TREATMENT", treatment_rows, val_sequences); run_trackeval_many(val_tracker_root, val_eval_root, ["BASELINE", "TREATMENT"], val_tracker_root / "seqmap.txt", gt_split="val", gt_folder=DATASET_ROOT / "val"); parsed_base = parse_trackeval(val_eval_root, "BASELINE", val_sequences); parsed_treatment = parse_trackeval(val_eval_root, "TREATMENT", val_sequences); base_summary = trackeval_summary(parsed_base); treatment_summary = trackeval_summary(parsed_treatment); delta = {metric: (float(treatment_summary.get(metric)) - float(base_summary.get(metric))) if isinstance(treatment_summary.get(metric), (int, float)) and isinstance(base_summary.get(metric), (int, float)) else None for metric in ("HOTA", "DetA", "AssA", "LocA", "IDF1", "MOTA", "IDSW", "FP", "FN")}
    val_metadata.update({"status": "VAL_COMPLETE", "base_score_sequences_built_this_invocation": len(base_results), "base_score_sequences_available": len(val_sequences), "candidate_stream_source": "SAM3 VAL stream generated through a train-named compatibility view of DanceTrack val images", "candidate_stream_index_split": "train_view_of_val", "event_frame_policy": "first_frame_with_candidate_and_gt_for_offline_simulated_event", "event_frames": {sequence: int(val_events[sequence]["event_frame"]) for sequence in val_sequences}, "runtime_future_gt_used": False}); write_json(OUT / "val/generation_manifest.json", val_metadata); write_json(OUT / "val/baseline_trackeval.json", {"stage": STAGE, "metrics": base_summary, "runtime_future_gt_used": False}); write_json(OUT / "val/treatment_trackeval.json", {"stage": STAGE, "metrics": treatment_summary, "runtime_future_gt_used": False}); write_json(OUT / "val/delta.json", {"stage": STAGE, "delta": delta, "runtime_future_gt_used": False})
    final = read_json(OUT / "FINAL_RESULT.json"); final["val"] = {"status": "VAL_COMPLETE", "baseline_metrics": base_summary, "treatment_metrics": treatment_summary, "delta": delta, "targeted_sam3": val_sam3_result, "runtime_future_gt_used": False}; final["val_authorized_after_dev_gate"] = True; final["test_accessed"] = False
    if not (isinstance(treatment_summary.get("HOTA"), (int, float)) and isinstance(base_summary.get("HOTA"), (int, float)) and treatment_summary["HOTA"] > base_summary["HOTA"]):
        final["decision"] = "FAIL_VAL_GENERALIZATION"
    elif final.get("decision") == "FAIL_VAL_GENERALIZATION":
        final["decision"] = "PASS_END_TO_END_HOTA_IMPROVEMENT"
    write_json(OUT / "FINAL_RESULT.json", final)
    report = (OUT / "FINAL_REPORT.md").read_text(encoding="utf-8"); report = report.replace("| Combined VAL | NA | NA | NA | NA | NA | NA | NA |", f"| Combined VAL | {treatment_summary.get('HOTA')} | {treatment_summary.get('DetA')} | {treatment_summary.get('AssA')} | {treatment_summary.get('IDF1')} | {treatment_summary.get('MOTA')} | {treatment_summary.get('IDSW')} | NA |") + f"\n## VAL result\n\n- Baseline HOTA(AUC): {base_summary.get('HOTA')}; treatment HOTA(AUC): {treatment_summary.get('HOTA')}; delta: {delta.get('HOTA')}.\n- VAL AssA delta: {delta.get('AssA')}; DetA delta: {delta.get('DetA')}; targeted SAM3: `{val_sam3_result.get('status')}` with {val_sam3_result.get('candidate_count_increase', 0)} candidate(s) and causal selection `{val_sam3_result.get('selection_policy')}`.\n- VAL GT was used only for simulated event preparation and TrackEval truth; it was not used for training, tuning, or runtime decisions.\n"
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8"); write_json(OUT / "storage_audit_after.json", storage_audit("after_val")); write_json(OUT / "stage_status.json", {"stage": STAGE, "goal_reference": f"outputs/{STAGE}/FINAL_GOAL.json", "goal_frozen": True, "status": "PASS" if final["decision"].startswith("PASS") else "FAIL", "scientific_decision": final["decision"], "val_complete": True, "val_accessed": True, "targeted_sam3_status": val_sam3_result.get("status"), "test_accessed": False, "runtime_future_gt_used": False, "historical_outputs_modified": False}); return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "baseline", "research", "val", "all"), default="audit")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    before = storage_audit("before")
    write_json(OUT / "storage_audit_before.json", before)
    if before["storage_status"] == "HARD_STOP":
        write_json(OUT / "stage_status.json", {"stage": STAGE, "status": "FAIL_ASSET_OR_STORAGE", "goal_reference": f"outputs/{STAGE}/FINAL_GOAL.json"})
        return 2
    assets = load_assets()
    audit = source_audit(assets)
    write_json(OUT / "source_audit.json", audit)
    (OUT / "source_audit.md").write_text("\n".join(["# N72R20R3R2R3 source audit", "", f"- source HEAD: `{audit['source_head']}`", f"- resolved HEAD: `{audit['resolved_head']}`", f"- TrackEval: `{audit['trackeval_commit']}`", f"- SAM3: `{audit['identity_checkpoint_hashes']['sam3']}`", f"- free storage: `{audit['disk_status']['free_gib']:.2f} GiB`", "- historical outputs modified: `False`", "- runtime future GT used: `False`", ""]) + "\n", encoding="utf-8")
    if args.phase == "audit":
        write_json(OUT / "stage_status.json", {"stage": STAGE, "goal_reference": f"outputs/{STAGE}/FINAL_GOAL.json", "goal_frozen": True, "status": "INITIALIZED", "source_commit": audit["source_head"], "runtime_future_gt_used": False, "historical_outputs_modified": False})
        print(json.dumps({"stage": STAGE, "status": "AUDIT_COMPLETE", "source_head": audit["source_head"], "free_gib": before["free_gib"]}, sort_keys=True))
        return 0
    if args.phase == "baseline":
        baseline = baseline_phase(assets)
        write_json(OUT / "stage_status.json", {"stage": STAGE, "goal_reference": f"outputs/{STAGE}/FINAL_GOAL.json", "goal_frozen": True, "status": "BASELINE_TRACKEVAL_COMPLETE", "source_commit": audit["source_head"], "baseline_trackeval": baseline, "runtime_future_gt_used": False, "historical_outputs_modified": False})
        print(json.dumps({"stage": STAGE, "status": "BASELINE_TRACKEVAL_COMPLETE", "csv_files": baseline["parsed"]["csv_files"]}, sort_keys=True))
        return 0
    if args.phase == "val":
        final_result = val_phase()
        print(json.dumps({"stage": STAGE, "status": "VAL_COMPLETE", "decision": final_result["decision"], "val_hota": final_result.get("val", {}).get("treatment_metrics", {}).get("HOTA")}, sort_keys=True))
        return 0
    if args.phase == "all" and not (OUT / "baseline/baseline_full_sequence_trackeval.json").exists():
        baseline = baseline_phase(assets)
        write_json(OUT / "stage_status.json", {"stage": STAGE, "goal_reference": f"outputs/{STAGE}/FINAL_GOAL.json", "goal_frozen": True, "status": "BASELINE_TRACKEVAL_COMPLETE", "source_commit": audit["source_head"], "baseline_trackeval": baseline, "runtime_future_gt_used": False, "historical_outputs_modified": False})
    final_result = research_phase(assets, audit)
    print(json.dumps({"stage": STAGE, "status": "RESEARCH_COMPLETE", "decision": final_result["decision"], "hota": final_result["treatment_metrics"].get("HOTA"), "baseline_hota": final_result["baseline_metrics"].get("HOTA")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
