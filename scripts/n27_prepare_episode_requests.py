#!/usr/bin/env python3
"""Prepare N27 causal parent episodes and sharded crop requests.

This stage reads train/held-out labels only.  It does not extract features, read
DanceTrack val25, or count repeated prefixes/rounds as independent parents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(".")
OUT = ROOT / "outputs/n27"
DATA = OUT / "data"
SPLIT_MANIFEST = OUT / "dataset_split_manifest.json"
DT = Path("/path/to/dancetrack/train")
MOT17 = Path("/path/to/MOT17/train")
MOT20 = Path("/path/to/MOT20/images/train")
BDD = Path("/path/to/masa/data/bdd")
KITTI_IMAGES = Path("/path/to/KITTI_tracking/training/image_02")
KITTI_LABELS = Path("/path/to/CenterTrack/CenterTrack/src/tools/eval_kitti_track/data/tracking/label_02")
SEED = 27
NUM_SHARDS = 4
TARGET_TRAIN = 55_000
TARGET_HELDOUT = 12_000
TRAIN_QUOTAS = {"MOT17": 8_000, "MOT20": 18_000, "BDD100K": 23_000, "KITTI": 6_000}
HELDOUT_QUOTAS = {"MOT17": 2_500, "MOT20": 3_500, "BDD100K": 5_000, "KITTI": 1_000}
PER_ID_CAP = {"MOT17": 45, "MOT20": 30, "BDD100K": 24, "KITTI": 90}
GT_MISSING_PROBABILITY = 0.05
GT_CENTER_JITTER_FRACTION = 0.035
GT_LOG_SCALE_JITTER = 0.05
PUBLIC_DETECTOR_PROBABILITY = 0.5


@dataclass(frozen=True)
class Observation:
    frame: int
    track_id: str
    box: tuple[float, float, float, float]
    visibility: float
    image_path: str


@dataclass(frozen=True)
class Detection:
    frame: int
    box: tuple[float, float, float, float]
    score: float


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stable_hex(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def stable_unit(value: str, offset: int = 0) -> float:
    digest = hashlib.sha256(f"{SEED}:{value}:{offset}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return intersection / max(area_a + area_b - intersection, 1e-12)


def spatial_hardness(target: Iterable[float], other: Iterable[float]) -> float:
    tx1, ty1, tx2, ty2 = target
    ox1, oy1, ox2, oy2 = other
    tw, th = max(tx2 - tx1, 1.0), max(ty2 - ty1, 1.0)
    ow, oh = max(ox2 - ox1, 1.0), max(oy2 - oy1, 1.0)
    center = math.hypot((ox1 + ox2 - tx1 - tx2) / 2, (oy1 + oy2 - ty1 - ty2) / 2)
    normalized_center = center / math.hypot(tw, th)
    scale = abs(math.log((ow * oh) / (tw * th)))
    return normalized_center + 0.2 * scale - 0.75 * iou(target, other)


def jitter_box(box: tuple[float, float, float, float], token: str) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    width, height = max(x2 - x1, 2.0), max(y2 - y1, 2.0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dx = (2 * stable_unit(token, 1) - 1) * GT_CENTER_JITTER_FRACTION * width
    dy = (2 * stable_unit(token, 2) - 1) * GT_CENTER_JITTER_FRACTION * height
    sw = math.exp((2 * stable_unit(token, 3) - 1) * GT_LOG_SCALE_JITTER)
    sh = math.exp((2 * stable_unit(token, 4) - 1) * GT_LOG_SCALE_JITTER)
    width, height = width * sw, height * sh
    return (cx + dx - width / 2, cy + dy - height / 2, cx + dx + width / 2, cy + dy + height / 2)


def load_mot(dataset: str, video: str) -> tuple[dict[str, list[Observation]], dict[int, list[Observation]], dict[int, list[Detection]]]:
    if dataset == "MOT17":
        scene = video.split("-")[-1]
        directory = MOT17 / f"MOT17-{scene}-FRCNN"
    elif dataset == "MOT20":
        directory = MOT20 / video
    else:
        raise ValueError(dataset)
    tracks: dict[str, list[Observation]] = defaultdict(list)
    frames: dict[int, list[Observation]] = defaultdict(list)
    for line in (directory / "gt/gt.txt").read_text(encoding="utf-8").splitlines():
        fields = line.split(",")
        if len(fields) < 6:
            continue
        frame = int(float(fields[0])) - 1
        track_id = str(int(float(fields[1])))
        x, y, width, height = map(float, fields[2:6])
        confidence = float(fields[6]) if len(fields) > 6 else 1.0
        category = int(float(fields[7])) if len(fields) > 7 else 1
        visibility = float(fields[8]) if len(fields) > 8 else 1.0
        if confidence != 1.0 or category != 1 or width <= 1 or height <= 1:
            continue
        observation = Observation(
            frame,
            track_id,
            (x, y, x + width, y + height),
            visibility,
            str(directory / "img1" / f"{frame + 1:06d}.jpg"),
        )
        tracks[track_id].append(observation)
        frames[frame].append(observation)
    detections: dict[int, list[Detection]] = defaultdict(list)
    det_path = directory / "det/det.txt"
    if det_path.is_file():
        for line in det_path.read_text(encoding="utf-8").splitlines():
            fields = line.split(",")
            if len(fields) < 6:
                continue
            frame = int(float(fields[0])) - 1
            x, y, width, height = map(float, fields[2:6])
            score = float(fields[6]) if len(fields) > 6 else 1.0
            if width > 1 and height > 1:
                detections[frame].append(Detection(frame, (x, y, x + width, y + height), score))
    return tracks, frames, detections


def load_bdd(video: str, role: str) -> tuple[dict[str, list[Observation]], dict[int, list[Observation]], dict[int, list[Detection]]]:
    split = "train" if role == "external_train" else "val"
    annotation_path = BDD / "annotations/box_track_20" / split / f"{video}.json"
    image_root = BDD / "bdd100k/images/track" / split / video
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    tracks: dict[str, list[Observation]] = defaultdict(list)
    frames: dict[int, list[Observation]] = defaultdict(list)
    for frame_row in payload:
        frame = int(frame_row.get("frameIndex", 0))
        image_path = str(image_root / str(frame_row["name"]))
        for label in frame_row.get("labels", []):
            if label.get("category") != "pedestrian" or "box2d" not in label:
                continue
            box = label["box2d"]
            xyxy = tuple(float(box[key]) for key in ("x1", "y1", "x2", "y2"))
            if xyxy[2] <= xyxy[0] + 1 or xyxy[3] <= xyxy[1] + 1:
                continue
            attributes = label.get("attributes", {})
            visibility = 0.0 if bool(attributes.get("occluded", False)) else 1.0
            observation = Observation(frame, str(label["id"]), xyxy, visibility, image_path)
            tracks[observation.track_id].append(observation)
            frames[frame].append(observation)
    return tracks, frames, defaultdict(list)


def load_kitti(video: str) -> tuple[dict[str, list[Observation]], dict[int, list[Observation]], dict[int, list[Detection]]]:
    tracks: dict[str, list[Observation]] = defaultdict(list)
    frames: dict[int, list[Observation]] = defaultdict(list)
    for line in (KITTI_LABELS / f"{video}.txt").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 10:
            continue
        frame, track_id, category = int(fields[0]), int(fields[1]), fields[2]
        if track_id < 0 or category != "Pedestrian":
            continue
        x1, y1, x2, y2 = map(float, fields[6:10])
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            continue
        occlusion = min(3, max(0, int(float(fields[4]))))
        observation = Observation(
            frame,
            str(track_id),
            (x1, y1, x2, y2),
            1.0 - occlusion / 3.0,
            str(KITTI_IMAGES / video / f"{frame:06d}.png"),
        )
        tracks[observation.track_id].append(observation)
        frames[frame].append(observation)
    return tracks, frames, defaultdict(list)


def select_public_candidates(
    target: Observation,
    people: list[Observation],
    detections: list[Detection],
    event_key: str,
) -> list[dict[str, Any]]:
    if not detections:
        return []
    unused = set(range(len(detections)))
    selected: list[int] = []
    target_order = sorted(unused, key=lambda index: (-iou(target.box, detections[index].box), -detections[index].score, index))
    if target_order and iou(target.box, detections[target_order[0]].box) >= 0.1:
        selected.append(target_order[0])
        unused.remove(target_order[0])
    distractors = sorted((person for person in people if person.track_id != target.track_id), key=lambda person: spatial_hardness(target.box, person.box))
    for person in distractors:
        if len(selected) >= 5 or not unused:
            break
        order = sorted(unused, key=lambda index: (-iou(person.box, detections[index].box), -detections[index].score, index))
        if order and iou(person.box, detections[order[0]].box) >= 0.1:
            selected.append(order[0])
            unused.remove(order[0])
    fill = sorted(unused, key=lambda index: (spatial_hardness(target.box, detections[index].box), -detections[index].score, index))
    selected.extend(fill[: max(0, 5 - len(selected))])
    selected = selected[:5]
    if not selected:
        return []
    best_target = max(selected, key=lambda index: iou(target.box, detections[index].box))
    target_iou = iou(target.box, detections[best_target].box)
    output = []
    for index in selected:
        detection = detections[index]
        output.append({
            "box": list(detection.box),
            "correct": bool(index == best_target and target_iou >= 0.5),
            "origin_track_id": None,
            "detector_score": detection.score,
            "target_iou": iou(target.box, detection.box),
            "augmentation": "NONE",
        })
    output.sort(key=lambda row: stable_hex(f"{event_key}:public-order:{row['box']}"))
    return output


def select_gt_candidates(target: Observation, people: list[Observation], event_key: str) -> list[dict[str, Any]]:
    distractors = sorted((person for person in people if person.track_id != target.track_id), key=lambda person: spatial_hardness(target.box, person.box))
    simulate_missing = stable_unit(f"{event_key}:missing") < GT_MISSING_PROBABILITY
    chosen = distractors[:5] if simulate_missing else [target, *distractors[:4]]
    output = []
    for person in chosen[:5]:
        jitter_token = f"{target.image_path}:{person.track_id}:gt-jitter"
        output.append({
            "box": list(jitter_box(person.box, jitter_token)),
            "correct": bool(person.track_id == target.track_id),
            "origin_track_id": person.track_id,
            "detector_score": 1.0,
            "target_iou": iou(target.box, person.box),
            "augmentation": "TRAIN_ONLY_DETERMINISTIC_CENTER_SCALE_JITTER",
        })
    output.sort(key=lambda row: stable_hex(f"{event_key}:gt-order:{row['origin_track_id']}"))
    return output


def load_video(dataset: str, video: str, role: str):
    if dataset in {"MOT17", "MOT20"}:
        return load_mot(dataset, video)
    if dataset == "BDD100K":
        return load_bdd(video, role)
    if dataset == "KITTI":
        return load_kitti(video)
    raise ValueError(dataset)


def build_video_events(dataset: str, video: str, role: str, fold: int | None) -> list[dict[str, Any]]:
    tracks, frames, detections = load_video(dataset, video, role)
    output = []
    for track_id, observations in sorted(tracks.items()):
        observations = sorted(observations, key=lambda row: row.frame)
        if len(observations) < 2:
            continue
        root = observations[0]
        eligible = [
            observation
            for observation in observations[1:]
            if len(frames[observation.frame]) >= 2 or len(detections[observation.frame]) >= 2
        ]
        cap = PER_ID_CAP[dataset]
        selected = sorted(eligible, key=lambda row: stable_hex(f"cap:{dataset}:{video}:{track_id}:{row.frame}"))[:cap]
        for target in selected:
            event_key = f"{dataset}:{video}:{track_id}:{target.frame}"
            public_requested = dataset in {"MOT17", "MOT20"} and stable_unit(f"{event_key}:source") < PUBLIC_DETECTOR_PROBABILITY
            if public_requested and len(detections[target.frame]) >= 2:
                candidates = select_public_candidates(target, frames[target.frame], detections[target.frame], event_key)
                candidate_source = "PUBLIC_DETECTOR_BOX"
            else:
                candidates = select_gt_candidates(target, frames[target.frame], event_key)
                candidate_source = "GT_BOX"
            if not candidates:
                continue
            output.append({
                "event_key": event_key,
                "dataset": dataset,
                "video": video,
                "track_id": track_id,
                "decision_frame": target.frame,
                "role": role,
                "fold": fold,
                "root_frame": root.frame,
                "root_box": list(root.box),
                "root_image_path": root.image_path,
                "target_box": list(target.box),
                "target_image_path": target.image_path,
                "target_visibility": target.visibility,
                "candidate_source": candidate_source,
                "candidate_set_positive_present": any(row["correct"] for row in candidates),
                "candidates": candidates,
                "parent_weight": 1.0,
                "prefix_cluster_size": 1,
                "feedback_visible_to_current_prediction": False,
                "target_state_reliable": True,
            })
    return output


def quota_select(pool: dict[str, list[dict[str, Any]]], quotas: dict[str, int], target: int) -> list[dict[str, Any]]:
    selected = []
    remaining: dict[str, list[dict[str, Any]]] = {}
    for dataset, rows in pool.items():
        rows = sorted(rows, key=lambda row: stable_hex(f"quota:{row['event_key']}"))
        take = min(quotas.get(dataset, 0), len(rows))
        selected.extend(rows[:take])
        remaining[dataset] = rows[take:]
    while len(selected) < target:
        progressed = False
        for dataset in sorted(remaining, key=lambda name: (len([r for r in selected if r["dataset"] == name]), name)):
            if remaining[dataset]:
                selected.append(remaining[dataset].pop(0))
                progressed = True
                if len(selected) >= target:
                    break
        if not progressed:
            break
    return sorted(selected, key=lambda row: (row["role"], row["dataset"], row["video"], row["track_id"], row["decision_frame"]))


def crop_id(image_path: str, box: list[float]) -> str:
    rounded = ",".join(f"{value:.4f}" for value in box)
    return hashlib.sha256(f"{image_path}|{rounded}".encode()).hexdigest()[:32]


def add_crop(crops: dict[str, dict[str, Any]], image_path: str, box: list[float], usage: str) -> str:
    identifier = crop_id(image_path, box)
    if identifier not in crops:
        source = Path(image_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        stat = source.stat()
        crops[identifier] = {
            "crop_id": identifier,
            "image_path": image_path,
            "box": box,
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "usages": [usage],
        }
    elif usage not in crops[identifier]["usages"]:
        crops[identifier]["usages"].append(usage)
    return identifier


def attach_crop_ids(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    crops: dict[str, dict[str, Any]] = {}
    for event in events:
        event["root_crop_id"] = add_crop(crops, event["root_image_path"], event.pop("root_box"), "ROOT")
        event["feedback_positive_crop_id"] = add_crop(crops, event["target_image_path"], event.pop("target_box"), "HUMAN_POSITIVE_IF_ERROR")
        for candidate in event["candidates"]:
            candidate["crop_id"] = add_crop(crops, event["target_image_path"], candidate["box"], "CANDIDATE")
    return events, crops


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-train", type=int, default=TARGET_TRAIN)
    parser.add_argument("--target-heldout", type=int, default=TARGET_HELDOUT)
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    entries = [entry for entry in split["entries"] if entry["dataset"] != "DanceTrack"]
    pool: dict[str, dict[str, list[dict[str, Any]]]] = {
        "external_train": defaultdict(list),
        "external_heldout": defaultdict(list),
    }
    video_generation = []
    for index, entry in enumerate(entries, 1):
        rows = build_video_events(entry["dataset"], entry["video"], entry["role"], entry.get("fold"))
        pool[entry["role"]][entry["dataset"]].extend(rows)
        video_generation.append({
            "dataset": entry["dataset"],
            "video": entry["video"],
            "role": entry["role"],
            "eligible_after_per_identity_cap": len(rows),
        })
        if index % 25 == 0:
            print(f"PREPARED {index}/{len(entries)} videos", flush=True)
    train = quota_select(pool["external_train"], TRAIN_QUOTAS, args.target_train)
    heldout = quota_select(pool["external_heldout"], HELDOUT_QUOTAS, args.target_heldout)
    if len(train) < 50_000:
        raise RuntimeError(f"independent train parents below hard target: {len(train)}")
    events, crops = attach_crop_ids([*train, *heldout])
    event_path = DATA / "episode_requests.jsonl"
    write_jsonl(event_path, events)
    crop_rows = sorted(crops.values(), key=lambda row: (row["image_path"], row["crop_id"]))
    shard_rows = [[] for _ in range(NUM_SHARDS)]
    for row in crop_rows:
        shard = int(stable_hex(f"image-shard:{row['image_path']}")[:8], 16) % NUM_SHARDS
        row["shard"] = shard
        shard_rows[shard].append(row)
    shard_manifests = []
    for shard, rows in enumerate(shard_rows):
        path = DATA / f"crop_requests_shard{shard}.jsonl"
        write_jsonl(path, rows)
        shard_manifests.append({"shard": shard, "path": str(path), "requests": len(rows), "sha256": sha256(path)})
    provenance_rows = []
    for event in events:
        for rank, candidate in enumerate(event["candidates"]):
            provenance_rows.append({
                "event_key": event["event_key"],
                "dataset": event["dataset"],
                "video": event["video"],
                "decision_frame": event["decision_frame"],
                "candidate_rank_storage_only": rank,
                "candidate_source": event["candidate_source"],
                "crop_id": candidate["crop_id"],
                "correct_or_none_label": candidate["correct"],
                "human_explicit_negative": False,
                "ordinary_hard_negative": not candidate["correct"],
                "augmentation": candidate["augmentation"],
            })
    write_jsonl(OUT / "candidate_provenance.jsonl", provenance_rows)
    disk = shutil.disk_usage("/data1")
    train_counter = Counter(row["dataset"] for row in train)
    heldout_counter = Counter(row["dataset"] for row in heldout)
    source_counter = Counter(row["candidate_source"] for row in events)
    identities = {(row["dataset"], row["video"], row["track_id"]) for row in train}
    videos = {(row["dataset"], row["video"]) for row in train}
    keys = [row["event_key"] for row in events]
    manifest = {
        "phase": "N27",
        "status": "CROP_REQUESTS_COMPLETE_FEATURES_PENDING",
        "seed": SEED,
        "parent_key_fields": ["dataset", "video", "track_id", "decision_frame"],
        "independent_parent_definition": "one row per unique parent key; no horizon, prefix, augmentation, or round replication",
        "train_independent_parents": len(train),
        "external_heldout_parents": len(heldout),
        "unique_event_keys_all_roles": len(set(keys)),
        "duplicate_event_keys": len(keys) - len(set(keys)),
        "train_unique_identities": len(identities),
        "train_unique_videos": len(videos),
        "train_by_dataset": dict(train_counter),
        "heldout_by_dataset": dict(heldout_counter),
        "candidate_sources_before_sam3_p2": dict(source_counter),
        "sam3_real_p2_parent_source": "outputs/n26/dense_dataset/round0_train30.npz plus parent ledger; added after feature extraction and clustered across rounds",
        "episode_request_path": str(event_path),
        "episode_request_sha256": sha256(event_path),
        "unique_crop_requests": len(crops),
        "feature_dimension": 1280,
        "feature_dtype": "float16",
        "feature_shards": shard_manifests,
        "feedback_causality": "prediction uses pre-event memory; correction writes are available from the next parent only",
        "human_negative_policy": "only the model-selected wrong candidate becomes HUMAN_EXPLICIT_NEGATIVE after rollout",
        "ordinary_hard_negative_policy": "unselected distractors remain separate and never receive a human-negative label",
        "gt_box_missing_probability": GT_MISSING_PROBABILITY,
        "gt_center_jitter_fraction": GT_CENTER_JITTER_FRACTION,
        "gt_log_scale_jitter": GT_LOG_SCALE_JITTER,
        "public_detector_probability_mot17_mot20": PUBLIC_DETECTOR_PROBABILITY,
        "test_labels_used": False,
        "val25_read": False,
        "data1_free_bytes_after_requests": disk.free,
        "minimum_reserved_bytes": 40 * 1024**3,
        "reserve_satisfied": disk.free >= 40 * 1024**3,
        "video_generation": video_generation,
    }
    atomic_json(OUT / "large_episode_manifest.json", manifest)
    config = {
        "phase": "N27",
        "frozen_before_feature_values": True,
        "seed": SEED,
        "target_train": args.target_train,
        "target_heldout": args.target_heldout,
        "train_quotas": TRAIN_QUOTAS,
        "heldout_quotas": HELDOUT_QUOTAS,
        "per_identity_caps": PER_ID_CAP,
        "candidate_k": 5,
        "gt_missing_probability": GT_MISSING_PROBABILITY,
        "gt_center_jitter_fraction": GT_CENTER_JITTER_FRACTION,
        "gt_log_scale_jitter": GT_LOG_SCALE_JITTER,
        "public_detector_probability": PUBLIC_DETECTOR_PROBABILITY,
        "selection": "deterministic hash priority within dataset, complete-video split fixed before episodes",
        "test_labels_used": False,
        "val25_read": False,
    }
    atomic_json(OUT / "episode_build_config.json", config)
    print(json.dumps({
        "train": len(train),
        "heldout": len(heldout),
        "train_by_dataset": dict(train_counter),
        "heldout_by_dataset": dict(heldout_counter),
        "train_identities": len(identities),
        "train_videos": len(videos),
        "candidate_sources": dict(source_counter),
        "unique_crops": len(crops),
        "shard_requests": [len(rows) for rows in shard_rows],
        "free_gib": disk.free / 1024**3,
        "val25_read": False,
    }, indent=2, sort_keys=True), flush=True)
    print("N27_EPISODE_REQUESTS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
