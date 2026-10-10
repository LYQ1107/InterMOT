"""Resource-binding recovery; unchanged frozen candidate generator and schema.

Official SAM3 calls .cuda() without the adapter device argument. Isolating
exactly one physical GPU is therefore required for each generation process.
V1 failed attempts/preflights remain immutable and are never overwritten.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import time
import traceback
import torch

from scripts.n72r21r2_common import ROOT, OUT, R1, ASSETS, TRAIN, read_json, write_json, sha256, storage, append_log, development_sequence
from scripts.n72r21r2_candidates import integrity
from scripts.n72r20r2_candidate_stream import run as inherited_run


def run(sequence):
    protocol = development_sequence(sequence)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit() or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly one verified idle physical GPU before starting this process")
    torch.cuda.set_device(0)
    if os.environ.get("N72R20R2_DISABLE_TRIM", "").lower() in {"1", "true", "yes", "on"}:
        raise ValueError("Frozen trim policy override forbidden")
    seal_path = OUT / "data/candidate_integrity" / (sequence + ".json")
    if seal_path.exists():
        seal = read_json(seal_path)
        index = read_json(seal["index_path"])
        assert sha256(seal["index_path"]) == seal["index_sha256"]
        assert all(sha256(index[k]) == seal[k + "_sha256"] for k in ("metadata", "embeddings"))
        print(json.dumps({"already_complete_verified": sequence}), flush=True)
        return
    seqroot = ASSETS / "candidates" / sequence
    if (seqroot / "done.json").exists():
        integrity(sequence)
        return
    if seqroot.exists() and list(seqroot.iterdir()):
        raise FileExistsError("Partial output exists; audit before versioned recovery, do not overwrite")
    storage(8 << 30)
    log_path = ASSETS / "generation_logs_v2" / (sequence + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise FileExistsError("Preserve previous worker log; explicit new attempt namespace required")
    temp_paths = [seqroot / ("." + name + "." + str(os.getpid()) + ".tmp") for name in ("metadata.jsonl.zst", "embeddings.f16")]
    preflight_path = write_json("data/extraction_preflight_v2/" + sequence + ".json", {"stage": "N72R21R2", "sequence": sequence,
                               "visible_physical_GPU": visible, "logical_device": "cuda:0", "pid": os.getpid(),
                               "inherited_builder_current_cuda_device": 0, "storage": storage(8 << 30),
                               "cleanup_dry_run_exact_new_regenerable_temp_targets": [str(p) for p in temp_paths],
                               "no_unique_asset_deletion": True, "source_sha256": sha256(Path(__file__)),
                               "inherited_generator_sha256": sha256(ROOT / "scripts/n72r20r2_candidate_stream.py"),
                               "preregistration_sha256": sha256(OUT / "PREREGISTRATION.json"), "log_path": str(log_path),
                               "model_candidate_configuration_changed": False})
    args = argparse.Namespace(asset_manifest=R1 / "outputs/N72R20R2/asset_manifest.json", dataset_root=TRAIN.parent,
                              checkpoint=None, osnet_checkpoint=None, output_root=ASSETS, split="train", sequence=sequence,
                              max_frames=0, chunk_size=160, device="cuda:0")
    started = time.monotonic()
    append_log("M1_GPU_ISOLATED_EXTRACTION_STARTED", sequence=sequence, physical_GPU=visible, pid=os.getpid())
    error = None
    with log_path.open("x") as handle, redirect_stdout(handle), redirect_stderr(handle):
        try:
            profile = inherited_run(args)
        except Exception as exc:
            error = {"error_type": type(exc).__name__, "error": str(exc)}
            traceback.print_exc()
    receipt = {"stage": "N72R21R2", "sequence": sequence, "status": "FAILED_RETAINED" if error else "COMPLETE_ACTUAL_NEW_EXTRACTION",
               "preflight_sha256": sha256(preflight_path), "source_sha256": sha256(Path(__file__)),
               "log_path": str(log_path), "log_sha256": sha256(log_path), "seconds": time.monotonic() - started,
               "visible_physical_GPU": visible, "logical_device": "cuda:0", "model_candidate_configuration_changed": False,
               "error": error, "temporary_outputs_removed_after_predeclared_failure_cleanup": bool(error and all(not p.exists() for p in temp_paths))}
    if not error:
        seal = integrity(sequence)
        receipt.update(profile=profile, integrity_path=str(seal_path), integrity_sha256=sha256(seal_path))
    write_json("data/extraction_receipts_v2/" + sequence + ".json", receipt)
    append_log("M1_GPU_ISOLATED_EXTRACTION_FINISHED", sequence=sequence, status=receipt["status"], seconds=receipt["seconds"])
    print(json.dumps(receipt), flush=True)
    if error:
        raise RuntimeError("Retained worker failure: " + error["error_type"] + ": " + error["error"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    args = parser.parse_args()
    run(args.sequence)
