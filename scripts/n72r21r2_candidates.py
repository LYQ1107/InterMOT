"""Frozen fresh FIT/INNER extraction using unmodified historical SAM3 worker.

The inherited worker's payload stage names denote its schema, not an old
observation. R2 seals identify the newly computed image/weight/config lineage.
No policy, anchor or GT file is opened by generation or input-integrity audit.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

from scripts.n72r21r2_common import ROOT, OUT, R1, ASSETS, TRAIN, development_sequence, read_json, sha256, write_json, storage, append_log
from scripts.n72r20r2_candidate_stream import run as inherited_run
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames
from scripts.n72r21_baselines_geometry_repair import valid_geometry


def integrity(sequence):
    protocol = development_sequence(sequence)
    sequence_root = ASSETS / "candidates" / sequence
    index_path = sequence_root / "index.json"
    index = read_json(index_path)
    done_path = sequence_root / "done.json"
    done = read_json(done_path)
    assert done["status"] == "PASS_N72R20R2_CANDIDATE_STREAM_SEQUENCE"
    assert index["sam3_checkpoint_sha256"] == protocol["candidate_generation"]["sam3_checkpoint_sha256"]
    assert index["osnet_checkpoint_sha256"] == protocol["candidate_generation"]["osnet_checkpoint_sha256"]
    assert index["sam3_chunk_size"] == 160 and index["sam3_trim_past_non_cond_mem_for_eval"]
    assert not index["runtime_gt_read"] and not index["runtime_future_gt_used"]
    for field in ("metadata", "embeddings"):
        assert sha256(index[field]) == index[field + "_sha256"]
    expected = len(list((TRAIN / sequence / "img1").glob("*.jpg")))
    assert index["frame_count"] == expected
    assert Path(index["embeddings"]).stat().st_size == index["embedding_count"] * 512 * 2
    frames = load_candidate_frames(ASSETS, sequence)
    assert [int(p["frame"]) for p, _ in frames] == list(range(expected))
    counts, accepted, seen, offsets = [], [], set(), set()
    for payload, rows in frames:
        f = int(payload["frame"])
        assert payload["candidate_set_complete"] and payload["candidate_count"] == len(rows)
        assert not payload["runtime_gt_read"] and not payload["runtime_future_gt_used"]
        for row in rows:
            uid = row["candidate_uid"]
            assert uid.startswith(sequence + ":" + str(f) + ":") and uid not in seen
            seen.add(uid)
            assert row["embedding_offset"] not in offsets
            offsets.add(row["embedding_offset"])
            assert row["source_public_id"] is None and np.isfinite(row["feature"]).all()
            assert not row["runtime_future_gt_used"]
            assert isinstance(row["feature_sha256"], str) and len(row["feature_sha256"]) == 64
        counts.append(len(rows))
        accepted.append(sum(valid_geometry(r) for r in rows))
    assert offsets == set(range(index["embedding_count"]))
    strata = Counter("SPARSE" if n <= 4 else "MEDIUM" if n <= 8 else "CROWDED" for n in accepted)
    whole = next((k for k, count in strata.items() if count / expected >= .8), "MIXED")
    seal = {"stage": "N72R21R2", "sequence": sequence, "status": "PASS_FULL_ORIGINAL_AXIS_UID_FEATURE_INTEGRITY_NOT_SCIENTIFIC_PASS",
            "newly_generated_real_candidates": True, "inherited_payload_schema_stage": index["stage"],
            "index_path": str(index_path), "index_sha256": sha256(index_path), "done_sha256": sha256(done_path),
            "metadata_sha256": index["metadata_sha256"], "embeddings_sha256": index["embeddings_sha256"],
            "frames": expected, "candidate_count": len(seen), "candidate_count_per_frame": counts,
            "valid_geometry_count_per_frame": accepted, "density_frame_counts": dict(strata), "whole_video_density": whole,
            "density_frozen_before_any_policy_effects": True,
            "float32_feature_hash_before_float16_storage": "Inherited canonical hash field, whole stored tape SHA verified; f16 cannot reproduce original f32 hash",
            "sam3_checkpoint_sha256": index["sam3_checkpoint_sha256"], "osnet_checkpoint_sha256": index["osnet_checkpoint_sha256"],
            "full_axis_unique_current_UIDs_features_offsets": True, "runtime_GT_read": False,
            "source_sha256": sha256(Path(__file__)), "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"),
            "inherited_generator_sha256": sha256(ROOT / "scripts/n72r20r2_candidate_stream.py")}
    write_json("data/candidate_integrity/" + sequence + ".json", seal)
    append_log("M1_FRESH_SEQUENCE_INTEGRITY_SEALED", sequence=sequence, frames=expected, candidates=len(seen), whole_video_density=whole)
    return seal


def generate(sequence, device):
    protocol = development_sequence(sequence)
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("Real frozen SAM3 requires an available CUDA device")
    if os.environ.get("N72R20R2_DISABLE_TRIM", "").lower() in {"1", "true", "yes", "on"}:
        raise ValueError("External trim override would change the frozen candidate policy")
    if (OUT / "data/candidate_integrity" / (sequence + ".json")).exists():
        seal = read_json(OUT / "data/candidate_integrity" / (sequence + ".json"))
        assert seal["source_sha256"] == sha256(Path(__file__))
        index = read_json(seal["index_path"])
        assert sha256(seal["index_path"]) == seal["index_sha256"]
        for field in ("metadata", "embeddings"):
            assert sha256(index[field]) == seal[field + "_sha256"]
        print(json.dumps({"verified_already_complete": sequence}), flush=True)
        return
    if (ASSETS / "candidates" / sequence / "done.json").exists():
        print(json.dumps({"integrity": integrity(sequence)["status"], "sequence": sequence}), flush=True)
        return
    storage(8 << 30)
    # Inherited failure cleanup affects only these newly allocated temporary
    # outputs. Record their exact predictable paths before any creation.
    seqroot = ASSETS / "candidates" / sequence
    temps = [seqroot / ("." + name + "." + str(os.getpid()) + ".tmp") for name in ("metadata.jsonl.zst", "embeddings.f16")]
    write_json("data/extraction_preflight/" + sequence + ".json", {"stage": "N72R21R2", "sequence": sequence, "device": device,
               "pid": os.getpid(), "storage": storage(8 << 30), "cleanup_dry_run_exact_new_regenerable_temp_targets": [str(p) for p in temps],
               "temp_cleanup_only_if_inherited_worker_fails": True, "unique_final_outputs_never_deleted": True,
               "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"), "source_sha256": sha256(Path(__file__))})
    args = argparse.Namespace(asset_manifest=R1 / "outputs/N72R20R2/asset_manifest.json", dataset_root=TRAIN.parent,
                              checkpoint=None, osnet_checkpoint=None, output_root=ASSETS, split="train", sequence=sequence,
                              max_frames=0, chunk_size=protocol["candidate_generation"]["chunk_size"], device=device)
    started = time.monotonic()
    append_log("M1_FRESH_SEQUENCE_EXTRACTION_STARTED", sequence=sequence, device=device, pid=os.getpid())
    try:
        profile = inherited_run(args)
    except Exception as exc:
        write_json("data/extraction_failures/" + sequence + "__pid" + str(os.getpid()) + ".json", {"stage": "N72R21R2", "sequence": sequence,
                   "error_type": type(exc).__name__, "error": str(exc), "seconds": time.monotonic() - started,
                   "regenerable_temporary_files_removed_by_inherited_worker": [str(p) for p in temps if not p.exists()],
                   "final_outputs_preserved": True, "source_sha256": sha256(Path(__file__))})
        raise
    seal = integrity(sequence)
    print(json.dumps({"fresh_real_sequence_complete": sequence, "profile": profile, "integrity": seal["status"], "seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["generate", "integrity"])
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    generate(args.sequence, args.device) if args.action == "generate" else print(json.dumps(integrity(args.sequence)))
