"""Versioned exact native debug-field recovery, never change frozen candidates."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from scripts.n72r21r2_common import ROOT, OUT, ASSETS, TRAIN, R1, read_json, write_json, sha256, storage, development_sequence, append_log
from scripts.n72r21r2_integrity_v2 import integrity
from scripts.n72r21r2_input_driver import alive_owned_worker
from scripts.n72r20r2_candidate_stream import run as inherited_run
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames
from sam3_intermot.backend.sam3_trim_schema_compat import pinned_sam3_trim_compatibility


def preserve_partial(sequence):
    path = OUT / "data/extraction_preflight_v2" / (sequence + ".json")
    if not path.exists():
        return []
    preflight = read_json(path)
    if alive_owned_worker(preflight["pid"], "scripts.n72r21r2_candidates"):
        raise RuntimeError("Cannot recover an owned live extraction")
    root = ASSETS / "candidates" / sequence
    records = []
    exact = [root / ("." + n + "." + str(preflight["pid"]) + ".tmp") for n in ("metadata.jsonl.zst", "embeddings.f16")]
    if root.exists() and any(p not in exact for p in root.iterdir()):
        raise RuntimeError("Only exact failed-worker temporary paths may be relocated")
    for source in exact:
        if source.exists():
            destination = ASSETS / "retained_failed_attempts_v4" / sequence / source.name
            records.append({"source": str(source), "destination": str(destination), "bytes": source.stat().st_size, "sha256": sha256(source)})
    write_json("data/recovery_v4_dry_run/" + sequence + ".json", {"stage": "N72R21R2", "prior_preflight_sha256": sha256(path),
               "old_worker_verified_not_alive": True, "moves_not_deletions": records, "no_unique_asset_deletion": True,
               "uncaught_termination_cause_not_assumed": True})
    for record in records:
        source, destination = Path(record["source"]), Path(record["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError("Do not overwrite retained evidence")
        source.rename(destination)
        assert sha256(destination) == record["sha256"]
    return records


def run(sequence, diagnostic_prefix=False):
    development_sequence(sequence)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit() or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Explicit one-idle-physical-GPU process isolation required")
    if os.environ.get("N72R20R2_DISABLE_TRIM", "").lower() in {"1", "true", "on", "yes"}:
        raise ValueError("Trim policy override forbidden")
    torch.cuda.set_device(0)
    if diagnostic_prefix:
        assert sequence in read_json(OUT / "PREREGISTRATION.json")["fresh_candidate_pilot"]
        root = ASSETS / "trim_compatibility_prefix_v4"
        assert (OUT / "data/candidate_integrity" / (sequence + ".json")).exists()
        retained = []
    else:
        proof = read_json(OUT / "data/TRIM_SCHEMA_COMPATIBILITY_PREFIX_V4.json")
        assert proof["raw_candidates_UIDs_boxes_scores_features_exactly_equal"]
        assert proof["compatibility_source_sha256"] == sha256(ROOT / "sam3_intermot/backend/sam3_trim_schema_compat.py")
        root = ASSETS
        if (root / "candidates" / sequence / "done.json").exists():
            return integrity(sequence)
        retained = preserve_partial(sequence)
    storage(8 << 30)
    namespace = "trim_prefix_v4" if diagnostic_prefix else "extraction_v4"
    log = ASSETS / "generation_logs_v4" / (namespace + "__" + sequence + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    preflight = write_json("data/" + namespace + "_preflight/" + sequence + ".json", {"stage": "N72R21R2", "sequence": sequence,
               "pid": os.getpid(), "visible_physical_GPU": visible, "logical_device": "cuda:0", "source_sha256": sha256(Path(__file__)),
               "compatibility_source_sha256": sha256(ROOT / "sam3_intermot/backend/sam3_trim_schema_compat.py"),
               "inherited_generator_sha256": sha256(ROOT / "scripts/n72r20r2_candidate_stream.py"), "retained_failed_temporary_files": retained,
               "frozen_candidate_configuration_changed": False, "diagnostic_prefix_only": diagnostic_prefix, "log_path": str(log)})
    args = argparse.Namespace(asset_manifest=R1 / "outputs/N72R20R2/asset_manifest.json", dataset_root=TRAIN.parent,
                              checkpoint=None, osnet_checkpoint=None, output_root=root, split="train", sequence=sequence,
                              max_frames=160 if diagnostic_prefix else 0, chunk_size=160, device="cuda:0")
    started = time.monotonic()
    error, counts, profile = None, None, None
    with log.open("x") as handle, redirect_stdout(handle), redirect_stderr(handle):
        try:
            with pinned_sam3_trim_compatibility() as counts:
                profile = inherited_run(args)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            traceback.print_exc()
    receipt = {"stage": "N72R21R2", "sequence": sequence, "status": "FAILED_RETAINED" if error else "COMPLETE_ACTUAL_V4_NATIVE_SCHEMA_COMPATIBILITY",
               "diagnostic_prefix_only": diagnostic_prefix, "preflight_sha256": sha256(preflight), "error": error,
               "native_compatibility_counts": counts, "log_path": str(log), "log_sha256": sha256(log), "profile": profile,
               "seconds": time.monotonic() - started, "source_sha256": sha256(Path(__file__)), "candidate_policy_changed": False}
    if not error and diagnostic_prefix:
        fresh = load_candidate_frames(root, sequence)
        existing = load_candidate_frames(ASSETS, sequence)[:160]
        assert len(fresh) == len(existing) == 160
        for (a, ar), (b, br) in zip(fresh, existing, strict=True):
            assert a == b
            assert len(ar) == len(br)
            for r, s in zip(ar, br, strict=True):
                assert set(r) == set(s)
                assert all(np.array_equal(r[k], s[k]) if isinstance(r[k], np.ndarray) else r[k] == s[k] for k in r)
        receipt.update(raw_candidates_UIDs_boxes_scores_features_exactly_equal=True,
                       existing_candidate_index_sha256=sha256(ASSETS / "candidates" / sequence / "index.json"),
                       compatibility_source_sha256=sha256(ROOT / "sam3_intermot/backend/sam3_trim_schema_compat.py"),
                       diagnostic_original_frames=160, not_whole_video_equivalence_claim=True)
        write_json("data/TRIM_SCHEMA_COMPATIBILITY_PREFIX_V4.json", receipt)
    elif not error:
        integrity(sequence)
    write_json("data/" + namespace + "_receipts/" + sequence + ".json", receipt)
    append_log("M1_V4_NATIVE_DEBUG_SCHEMA_WORKER", sequence=sequence, status=receipt["status"], diagnostic_prefix=diagnostic_prefix)
    print({"V4_candidate_worker": sequence, "status": receipt["status"], "native_schema_counts": counts, "diagnostic_prefix": diagnostic_prefix}, flush=True)
    if error:
        raise RuntimeError("Preserve failed V4 attempt: " + str(error))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--diagnostic-prefix", action="store_true")
    args = parser.parse_args()
    run(args.sequence, args.diagnostic_prefix)
