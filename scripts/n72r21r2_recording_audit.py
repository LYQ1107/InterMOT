"""Full image-byte duplicate audit; no GT parsing or identity selection.

Identical frames can reveal duplicate files/fragments. No matches cannot prove
different recordings, scenes or people: public recording/global IDs are absent.
"""
from collections import defaultdict
import json
from pathlib import Path
import time
from scripts.n72r21r2_common import OUT, ASSETS, TRAIN, preregistration, read_json, write_json, sha256, append_log


def run():
    protocol = preregistration()
    started = time.monotonic()
    groups = {s: role for role, names in protocol["split"].items() for s in names}
    hashes = defaultdict(list)
    counts = {}
    manifest_path = ASSETS / "input_integrity/all_TRAIN_image_byte_SHA.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x") as handle:
        for sequence in sorted(groups):
            images = sorted((TRAIN / sequence / "img1").glob("*.jpg"))
            counts[sequence] = len(images)
            for image in images:
                digest = sha256(image)
                item = {"sequence": sequence, "original_frame_1based": int(image.stem), "sha256": digest}
                hashes[digest].append(item)
                handle.write(json.dumps(item, sort_keys=True) + "\n")
            print({"image_byte_audit_complete": sequence, "frames": len(images)}, flush=True)
    cross_video = [items for items in hashes.values() if len({i["sequence"] for i in items}) > 1]
    crossed_split = [items for items in cross_video if len({groups[i["sequence"]] for i in items}) > 1]
    write_json("data/RECORDING_DUPLICATE_AUDIT_V1.json", {"stage": "N72R21R2", "all40_TRAIN_video_image_axes_SHA_audited": True,
               "sequence_frame_counts": counts, "actual_frame_files": sum(counts.values()), "image_manifest_path": str(manifest_path),
               "image_manifest_sha256": sha256(manifest_path), "exact_cross_video_shared_frame_hashes": len(cross_video),
               "cross_split_shared_frame_hashes": len(crossed_split), "cross_split_shared_frame_records": crossed_split,
               "source_sha256": sha256(Path(__file__)), "seconds": time.monotonic() - started, "CONFIRM_input_integrity_only_no_truth_parse": True,
               "recording_global_ID_metadata_available": False, "person_global_ID_metadata_available": False,
               "no_byte_duplicate_is_not_recording_scene_or_person_disjointness_proof": True,
               "statistical_independence_scope": "Sequence cluster, explicit unverified recording/person scope; no person-disjoint claim",
               "frozen_split_not_reshuffled_by_effects": True})
    append_log("M0_FULL_TRAIN_IMAGE_DUPLICATE_AUDIT_COMPLETE", frame_files=sum(counts.values()), cross_split_exact_duplicates=len(crossed_split))


if __name__ == "__main__":
    run()
