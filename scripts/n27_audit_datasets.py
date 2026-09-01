#!/usr/bin/env python3
"""Audit N27 identity data without opening DanceTrack val25 or test labels."""

from __future__ import annotations

import configparser
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(".")
OUT = ROOT / "outputs/n27"
DT = Path("/path/to/dancetrack/train")
MOT17 = Path("/path/to/MOT17/train")
MOT20 = Path("/path/to/MOT20/images/train")
BDD = Path("/path/to/masa/data/bdd")
KITTI_IMAGES = Path("/path/to/KITTI_tracking/training/image_02")
KITTI_LABELS = Path("/path/to/CenterTrack/CenterTrack/src/tools/eval_kitti_track/data/tracking/label_02")
KITTI_LABELS_COPY = Path("/path/to/P2PMFT/CenterTrack/src/tools/eval_kitti_track/data/tracking/label_02")
TAO = Path("/path/to/TAO/TAO-download/TAO-Amodal")
SEED = 27


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(path: Path) -> str:
    """Return the digest used by TAO's official frame checksum manifests."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_fold(dataset: str, video: str) -> int:
    token = hashlib.sha256(f"{SEED}:{dataset}:{video}".encode()).digest()
    return int.from_bytes(token[:4], "big") % 5


def summarize_tracks(
    frames_by_track: dict[str, list[int]],
    people_per_frame: Counter[int],
    visibility: list[float],
) -> dict[str, Any]:
    lengths = [len(set(frames)) for frames in frames_by_track.values()]
    gaps: list[int] = []
    potential = 0
    for frames in frames_by_track.values():
        ordered = sorted(set(frames))
        gaps.extend(b - a for a, b in zip(ordered, ordered[1:]))
        if ordered:
            first = ordered[0]
            potential += sum(frame > first and people_per_frame[frame] >= 2 for frame in ordered)
    return {
        "independent_track_identities": len(frames_by_track),
        "person_annotations": int(sum(lengths)),
        "annotated_person_frames": len(people_per_frame),
        "potential_parent_keys_with_prior_and_distractor": int(potential),
        "mean_track_observations": float(sum(lengths) / len(lengths)) if lengths else math.nan,
        "median_track_observations": float(sorted(lengths)[len(lengths) // 2]) if lengths else math.nan,
        "adjacent_annotation_gap_fraction_gt1": float(sum(gap > 1 for gap in gaps) / len(gaps)) if gaps else math.nan,
        "mean_visibility": float(sum(visibility) / len(visibility)) if visibility else math.nan,
    }


def read_seqinfo(path: Path) -> dict[str, int]:
    parser = configparser.ConfigParser()
    parser.read(path)
    section = parser["Sequence"]
    return {
        "seq_length": int(section["seqLength"]),
        "width": int(section["imWidth"]),
        "height": int(section["imHeight"]),
        "frame_rate": int(section.get("frameRate", 0)),
    }


def audit_mot_csv_sequence(
    sequence: str,
    directory: Path,
    *,
    dataset: str,
    conf_required: bool,
    class_required: int | None,
) -> dict[str, Any]:
    info = read_seqinfo(directory / "seqinfo.ini")
    frames_by_track: dict[str, list[int]] = defaultdict(list)
    people_per_frame: Counter[int] = Counter()
    visibility: list[float] = []
    category_counts: Counter[str] = Counter()
    all_rows = 0
    for line in (directory / "gt/gt.txt").read_text(encoding="utf-8").splitlines():
        fields = line.split(",")
        if len(fields) < 6:
            continue
        all_rows += 1
        frame, track = int(float(fields[0])) - 1, int(float(fields[1]))
        confidence = float(fields[6]) if len(fields) > 6 else 1.0
        category = int(float(fields[7])) if len(fields) > 7 else 1
        category_counts[str(category)] += 1
        valid = (not conf_required or confidence == 1.0) and (class_required is None or category == class_required)
        if not valid or float(fields[4]) <= 0 or float(fields[5]) <= 0:
            continue
        frames_by_track[str(track)].append(frame)
        people_per_frame[frame] += 1
        visibility.append(float(fields[8]) if len(fields) > 8 else 1.0)
    images = sorted((directory / "img1").glob("*"))
    track = summarize_tracks(frames_by_track, people_per_frame, visibility)
    return {
        "dataset": dataset,
        "sequence": sequence,
        **info,
        "image_files": len(images),
        "image_sequence_complete": len(images) == info["seq_length"],
        "gt_rows_all_categories": all_rows,
        "category_row_counts": dict(category_counts),
        **track,
    }


def aggregate_sequence_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "videos_or_sequences": len(rows),
        "independent_track_identities": sum(row["independent_track_identities"] for row in rows),
        "available_image_frames": sum(row["image_files"] for row in rows),
        "person_annotations": sum(row["person_annotations"] for row in rows),
        "potential_parent_keys_with_prior_and_distractor": sum(row["potential_parent_keys_with_prior_and_distractor"] for row in rows),
        "complete_video_count": sum(bool(row["image_sequence_complete"]) for row in rows),
        "mean_visibility": float(sum(row["mean_visibility"] * row["person_annotations"] for row in rows if math.isfinite(row["mean_visibility"])) / max(1, sum(row["person_annotations"] for row in rows if math.isfinite(row["mean_visibility"])))),
    }


def audit_dancetrack() -> dict[str, Any]:
    split = json.loads((ROOT / "outputs/n15/n15_frozen.json").read_text(encoding="utf-8"))["split"]
    train30 = list(split["train30"])
    cal10 = list(split["calibration10"])
    train_rows = [audit_mot_csv_sequence(name, DT / name, dataset="DanceTrack", conf_required=True, class_required=1) for name in train30]
    cal_rows = [audit_mot_csv_sequence(name, DT / name, dataset="DanceTrack", conf_required=True, class_required=1) for name in cal10]
    return {
        "role": "train30 for model fitting/folds; historical cal10 for final N27 development gate only",
        "class_policy": "all valid DanceTrack GT rows are person; class 1/confidence 1",
        "license": "DanceTrack research benchmark terms; source redistribution not performed by N27",
        "train30": {**aggregate_sequence_rows(train_rows), "sequences": train_rows},
        "cal10": {**aggregate_sequence_rows(cal_rows), "sequences": cal_rows},
        "val25_read": False,
        "val25_listing_or_statistics_computed": False,
    }


def audit_mot17() -> tuple[dict[str, Any], dict[str, Any]]:
    scenes = sorted({path.name.split("-")[1] for path in MOT17.iterdir() if path.is_dir()})
    rows = []
    duplicate_rows = []
    for scene in scenes:
        variants = [MOT17 / f"MOT17-{scene}-{detector}" for detector in ("DPM", "FRCNN", "SDP")]
        selected = variants[1]
        rows.append(audit_mot_csv_sequence(f"MOT17-{scene}", selected, dataset="MOT17", conf_required=True, class_required=1))
        sample_names = []
        images = sorted((selected / "img1").glob("*"))
        if images:
            sample_names = [images[0].name, images[len(images) // 2].name, images[-1].name]
        sample_hashes = {variant.name: [sha256(variant / "img1" / name) for name in sample_names] for variant in variants}
        gt_hashes = {variant.name: sha256(variant / "gt/gt.txt") for variant in variants}
        duplicate_rows.append({
            "original_scene": f"MOT17-{scene}",
            "detector_variants": [variant.name for variant in variants],
            "sample_frame_names": sample_names,
            "sample_images_identical_across_variants": len({tuple(value) for value in sample_hashes.values()}) == 1,
            "gt_identical_across_variants": len(set(gt_hashes.values())) == 1,
            "selected_once_for_identity_training": selected.name,
            "variant_sample_hashes": sample_hashes,
            "variant_gt_hashes": gt_hashes,
        })
    return ({
        "role": "Tier A; official train labels only; one copy per original scene",
        "class_policy": "confidence=1 and class=1 Pedestrian only; static/person-on-vehicle classes excluded",
        "license": "MOTChallenge/MOT17 benchmark terms; non-commercial source terms may apply",
        "raw_detector_directories": len(scenes) * 3,
        "original_scenes": len(scenes),
        **aggregate_sequence_rows(rows),
        "sequences": rows,
    }, {"mot17_detector_variant_deduplication": duplicate_rows})


def audit_mot20() -> dict[str, Any]:
    directories = sorted(path for path in MOT20.iterdir() if path.is_dir())
    rows = [audit_mot_csv_sequence(path.name, path, dataset="MOT20", conf_required=True, class_required=1) for path in directories]
    return {
        "role": "Tier A; official train labels only",
        "class_policy": "confidence=1 and class=1 Pedestrian only",
        "license": "MOTChallenge/MOT20 benchmark terms; non-commercial source terms may apply",
        **aggregate_sequence_rows(rows),
        "sequences": rows,
    }


def audit_bdd_split(split: str) -> dict[str, Any]:
    annotations = BDD / "annotations/box_track_20" / split
    image_root = BDD / "bdd100k/images/track" / split
    annotation_names = {path.stem for path in annotations.glob("*.json")}
    image_names = {path.name for path in image_root.iterdir() if path.is_dir()}
    intersection = sorted(annotation_names & image_names)
    category_counts: Counter[str] = Counter()
    video_rows = []
    for name in intersection:
        frames = json.loads((annotations / f"{name}.json").read_text(encoding="utf-8"))
        actual_images = {path.name for path in (image_root / name).glob("*.jpg")}
        expected_images = {str(frame["name"]) for frame in frames}
        by_track: dict[str, list[int]] = defaultdict(list)
        people_per_frame: Counter[int] = Counter()
        occluded: list[float] = []
        for frame in frames:
            frame_index = int(frame.get("frameIndex", 0))
            for label in frame.get("labels", []):
                category = str(label.get("category", "UNKNOWN"))
                category_counts[category] += 1
                if category != "pedestrian" or "box2d" not in label:
                    continue
                box = label["box2d"]
                if float(box["x2"]) <= float(box["x1"]) or float(box["y2"]) <= float(box["y1"]):
                    continue
                by_track[str(label["id"])].append(frame_index)
                people_per_frame[frame_index] += 1
                occluded.append(0.0 if bool(label.get("attributes", {}).get("occluded", False)) else 1.0)
        stats = summarize_tracks(by_track, people_per_frame, occluded)
        video_rows.append({
            "video": name,
            "annotation_frames": len(frames),
            "image_files": len(actual_images),
            "missing_annotated_images": len(expected_images - actual_images),
            "extra_images": len(actual_images - expected_images),
            "image_sequence_complete": expected_images <= actual_images,
            **stats,
        })
    return {
        "annotation_videos": len(annotation_names),
        "image_videos": len(image_names),
        "matched_videos": len(intersection),
        "annotation_only_videos": len(annotation_names - image_names),
        "image_only_videos": len(image_names - annotation_names),
        "category_row_counts": dict(category_counts),
        **aggregate_sequence_rows(video_rows),
        "videos": video_rows,
    }


def audit_bdd() -> dict[str, Any]:
    return {
        "role": "Tier A uses only actual tracking train image/annotation intersection; official val is external held-out",
        "class_policy": "pedestrian only; rider and all vehicle categories excluded from the main model",
        "license": "BDD100K research/data license; no source images are copied by N27",
        "train": audit_bdd_split("train"),
        "validation": audit_bdd_split("val"),
    }


def audit_kitti() -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    label_copy_equal = []
    for image_dir in sorted(path for path in KITTI_IMAGES.iterdir() if path.is_dir()):
        sequence = image_dir.name
        label_path = KITTI_LABELS / f"{sequence}.txt"
        copy_path = KITTI_LABELS_COPY / f"{sequence}.txt"
        by_track: dict[str, list[int]] = defaultdict(list)
        people_per_frame: Counter[int] = Counter()
        visibility: list[float] = []
        categories: Counter[str] = Counter()
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 10:
                continue
            frame, track, category = int(fields[0]), int(fields[1]), fields[2]
            categories[category] += 1
            if category != "Pedestrian" or track < 0:
                continue
            x1, y1, x2, y2 = map(float, fields[6:10])
            if x2 <= x1 or y2 <= y1:
                continue
            by_track[str(track)].append(frame)
            people_per_frame[frame] += 1
            occlusion = min(3, max(0, int(float(fields[4]))))
            visibility.append(1.0 - occlusion / 3.0)
        images = sorted(image_dir.glob("*.png"))
        rows.append({
            "dataset": "KITTI_Tracking",
            "sequence": sequence,
            "seq_length": len(images),
            "image_files": len(images),
            "image_sequence_complete": bool(images) and max((int(path.stem) for path in images), default=-1) + 1 == len(images),
            "category_row_counts": dict(categories),
            **summarize_tracks(by_track, people_per_frame, visibility),
        })
        label_copy_equal.append({
            "sequence": sequence,
            "canonical_label_path": str(label_path),
            "second_local_copy_path": str(copy_path),
            "sha256": sha256(label_path),
            "second_copy_sha256": sha256(copy_path),
            "copies_identical": sha256(label_path) == sha256(copy_path),
        })
    return ({
        "role": "Tier B admitted for Pedestrian identity episodes after standard tracking label_02 was found locally",
        "class_policy": "Pedestrian only; Person_sitting, Van, Car, Cyclist and DontCare excluded",
        "license": "CC BY-NC-SA 3.0 according to the official KITTI benchmark page",
        "local_detection_label_2_rejected": True,
        "standard_tracking_label_02_path": str(KITTI_LABELS),
        **aggregate_sequence_rows(rows),
        "sequences": rows,
    }, {"kitti_tracking_label_copy_integrity": label_copy_equal})


def person_category_id(payload: dict[str, Any]) -> int:
    for category in payload["categories"]:
        if category.get("synset") == "person.n.01":
            return int(category["id"])
    raise RuntimeError("TAO person category not found")


def audit_tao_file(relative: str, checksums_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads((TAO / relative).read_text(encoding="utf-8"))
    category_id = person_category_id(payload)
    images = {int(image["id"]): image for image in payload["images"]}
    videos = {int(video["id"]): video for video in payload["videos"]}
    tracks: dict[tuple[int, int], list[int]] = defaultdict(list)
    people_per_frame: dict[int, Counter[int]] = defaultdict(Counter)
    visibility: list[float] = []
    person_annotations = []
    for annotation in payload["annotations"]:
        if int(annotation.get("category_id", -1)) != category_id:
            continue
        image = images[int(annotation["image_id"])]
        video_id = int(annotation["video_id"])
        frame = int(image["frame_index"])
        track_id = int(annotation["track_id"])
        tracks[(video_id, track_id)].append(frame)
        people_per_frame[video_id][frame] += 1
        visibility.append(float(annotation.get("visibility", 1.0)))
        person_annotations.append(annotation)
    person_image_ids = {int(annotation["image_id"]) for annotation in person_annotations}
    person_video_ids = {int(annotation["video_id"]) for annotation in person_annotations}
    frame_exists = 0
    manifest_listed = 0
    checksum_manifest = json.loads((TAO / f"annotations/checksums/{checksums_name}").read_text(encoding="utf-8"))
    checksum_values_nonempty = 0
    checksum_values_verified = 0
    checksum_mismatch_examples = []
    missing_examples = []
    for image_id in sorted(person_image_ids):
        image = images[image_id]
        frame_path = TAO / "frames" / image["file_name"]
        frame_exists += frame_path.is_file()
        video_key = str(image["video"])
        frame_name = Path(image["file_name"]).name
        listed = video_key in checksum_manifest and frame_name in checksum_manifest[video_key]
        manifest_listed += listed
        expected_checksum = checksum_manifest[video_key][frame_name] if listed else ""
        if expected_checksum:
            checksum_values_nonempty += 1
            if frame_path.is_file() and md5(frame_path) == expected_checksum:
                checksum_values_verified += 1
            elif len(checksum_mismatch_examples) < 10:
                checksum_mismatch_examples.append(str(frame_path))
        if not frame_path.is_file() and len(missing_examples) < 10:
            missing_examples.append(str(frame_path))
    source_counts = Counter(str(videos[video_id].get("metadata", {}).get("dataset", "UNKNOWN")) for video_id in person_video_ids)
    potential = 0
    lengths = []
    gaps = []
    for (video_id, _), frames in tracks.items():
        ordered = sorted(set(frames))
        lengths.append(len(ordered))
        gaps.extend(b - a for a, b in zip(ordered, ordered[1:]))
        if ordered:
            potential += sum(frame > ordered[0] and people_per_frame[video_id][frame] >= 2 for frame in ordered)
    non_exhaustive_person_videos = sum(category_id in set(map(int, videos[video_id].get("not_exhaustive_category_ids", []))) for video_id in person_video_ids)
    bdd_video_names = sorted(videos[video_id]["name"] for video_id in person_video_ids if str(videos[video_id].get("metadata", {}).get("dataset", "")).lower().startswith("bdd"))
    return ({
        "annotation_file": relative,
        "videos_all": len(videos),
        "images_all": len(images),
        "person_category_id": category_id,
        "person_videos": len(person_video_ids),
        "person_tracks": len(tracks),
        "person_annotations": len(person_annotations),
        "person_annotated_images": len(person_image_ids),
        "person_frame_files_present": frame_exists,
        "person_frame_files_missing": len(person_image_ids) - frame_exists,
        "checksum_inventory_entries_present": manifest_listed,
        "checksum_values_nonempty": checksum_values_nonempty,
        "checksum_values_verified": checksum_values_verified,
        "checksum_mismatch_examples": checksum_mismatch_examples,
        "cryptographic_checksum_verification": "MD5 verified against all non-empty official manifest values for person-category frames",
        "non_exhaustive_person_videos": non_exhaustive_person_videos,
        "source_dataset_counts": dict(source_counts),
        "potential_parent_keys_with_prior_and_distractor": potential,
        "mean_track_observations": float(sum(lengths) / len(lengths)) if lengths else math.nan,
        "adjacent_annotation_gap_fraction_gt1": float(sum(gap > 1 for gap in gaps) / len(gaps)) if gaps else math.nan,
        "mean_visibility": float(sum(visibility) / len(visibility)) if visibility else math.nan,
        "missing_person_frame_examples": missing_examples,
    }, {"tao_person_bdd_source_videos": bdd_video_names})


def audit_tao() -> tuple[dict[str, Any], dict[str, Any]]:
    train, train_overlap = audit_tao_file("amodal_annotations/train.json", "train_checksums.json")
    validation, validation_overlap = audit_tao_file("amodal_annotations/validation.json", "validation_checksums.json")
    return ({
        "role": "EXCLUDED from N27 main training: sparse/non-exhaustive labels, mixed underlying source licenses, and explicit BDD source overlap make reliable NONE and clean source governance unavailable",
        "class_policy": "only synset person.n.01/category 805 considered; reflection/dnt helper categories excluded",
        "license": "TAO toolkit is permissive, but underlying source videos retain source licenses; local annotation licenses field is Unknown",
        "annotation_rate": "approximately 1 FPS sparse federated annotation",
        "train": train,
        "validation": validation,
    }, {**train_overlap, "tao_validation_person_bdd_source_videos": validation_overlap["tao_person_bdd_source_videos"]})


def build_split_manifest(audit: dict[str, Any]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for split_name in ("train30", "cal10"):
        for row in audit["datasets"]["DanceTrack"][split_name]["sequences"]:
            entries.append({
                "dataset": "DanceTrack", "video": row["sequence"],
                "role": "train_fold" if split_name == "train30" else "historical_cal10_gate",
                "fold": deterministic_fold("DanceTrack", row["sequence"]) if split_name == "train30" else None,
                "identity_namespace": f"DanceTrack/{row['sequence']}",
            })
    mot17_rows = audit["datasets"]["MOT17"]["sequences"]
    mot17_held = min(mot17_rows, key=lambda row: hashlib.sha256(f"{SEED}:MOT17:{row['sequence']}".encode()).hexdigest())["sequence"]
    for row in mot17_rows:
        entries.append({"dataset": "MOT17", "video": row["sequence"], "role": "external_heldout" if row["sequence"] == mot17_held else "external_train", "fold": deterministic_fold("MOT17", row["sequence"]), "identity_namespace": f"MOT17/{row['sequence']}"})
    mot20_rows = audit["datasets"]["MOT20"]["sequences"]
    mot20_held = min(mot20_rows, key=lambda row: hashlib.sha256(f"{SEED}:MOT20:{row['sequence']}".encode()).hexdigest())["sequence"]
    for row in mot20_rows:
        entries.append({"dataset": "MOT20", "video": row["sequence"], "role": "external_heldout" if row["sequence"] == mot20_held else "external_train", "fold": deterministic_fold("MOT20", row["sequence"]), "identity_namespace": f"MOT20/{row['sequence']}"})
    for split_name, role in (("train", "external_train"), ("validation", "external_heldout")):
        for row in audit["datasets"]["BDD100K_Tracking"][split_name]["videos"]:
            entries.append({"dataset": "BDD100K", "video": row["video"], "role": role, "fold": deterministic_fold("BDD100K", row["video"]), "identity_namespace": f"BDD100K/{row['video']}"})
    kitti_rows = audit["datasets"]["KITTI_Tracking"]["sequences"]
    ordered = sorted(kitti_rows, key=lambda row: hashlib.sha256(f"{SEED}:KITTI:{row['sequence']}".encode()).hexdigest())
    kitti_held = {row["sequence"] for row in ordered[: max(1, math.ceil(len(ordered) * 0.2))]}
    for row in kitti_rows:
        entries.append({"dataset": "KITTI", "video": row["sequence"], "role": "external_heldout" if row["sequence"] in kitti_held else "external_train", "fold": deterministic_fold("KITTI", row["sequence"]), "identity_namespace": f"KITTI/{row['sequence']}"})
    return {
        "phase": "N27",
        "seed": SEED,
        "split_unit": "complete video/sequence",
        "same_identity_or_video_cross_split": False,
        "MOT17_detector_variants_count_as_one_scene": True,
        "BDD_official_validation_used_as_external_heldout": True,
        "TAO_status": "EXCLUDED",
        "test_labels_used": False,
        "val25_read": False,
        "entries": entries,
        "counts_by_role": dict(Counter(entry["role"] for entry in entries)),
        "counts_by_dataset": dict(Counter(entry["dataset"] for entry in entries)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage("/data1")
    dancetrack = audit_dancetrack()
    mot17, mot17_overlap = audit_mot17()
    mot20 = audit_mot20()
    bdd = audit_bdd()
    kitti, kitti_overlap = audit_kitti()
    tao, tao_overlap = audit_tao()
    audit = {
        "phase": "N27",
        "generated_before_episode_build": True,
        "data1_free_bytes": disk.free,
        "minimum_reserved_bytes": 40 * 1024**3,
        "reserve_satisfied": disk.free >= 40 * 1024**3,
        "test_labels_used": False,
        "val25_read": False,
        "datasets": {
            "DanceTrack": dancetrack,
            "MOT17": mot17,
            "MOT20": mot20,
            "BDD100K_Tracking": bdd,
            "KITTI_Tracking": kitti,
            "TAO_Amodal": tao,
        },
    }
    overlap = {
        "phase": "N27",
        "identity_namespace": "dataset/video/local_track_id; no cross-video identity equality assumed",
        "MOT17": mot17_overlap,
        "KITTI": kitti_overlap,
        "TAO_BDD": tao_overlap,
        "BDD_TAO_policy": "TAO is excluded; if used in a future phase, every metadata.dataset=BDD video must be removed before combining with BDD100K",
        "other_exact_source_overlap_detected": False,
        "test_labels_used": False,
        "val25_read": False,
    }
    split_manifest = build_split_manifest(audit)
    atomic_json(OUT / "dataset_audit.json", audit)
    atomic_json(OUT / "dataset_overlap_audit.json", overlap)
    atomic_json(OUT / "dataset_split_manifest.json", split_manifest)
    print(json.dumps({
        "free_gib": disk.free / 1024**3,
        "datasets": {
            "DanceTrack_train30": dancetrack["train30"]["videos_or_sequences"],
            "MOT17_original": mot17["videos_or_sequences"],
            "MOT20": mot20["videos_or_sequences"],
            "BDD_train_match": bdd["train"]["matched_videos"],
            "BDD_val_match": bdd["validation"]["matched_videos"],
            "KITTI": kitti["videos_or_sequences"],
            "TAO_train_person_tracks_excluded": tao["train"]["person_tracks"],
        },
        "split_roles": split_manifest["counts_by_role"],
        "val25_read": False,
    }, indent=2, sort_keys=True))
    print("N27_DATASET_AUDIT_COMPLETE")


if __name__ == "__main__":
    main()
