"""Serialize NumPy scalar audit counts without rerunning completed inference.

V1 partial JSON temporary files are preserved. This changes only the evidence
writer, not candidates, features, weights, filters or any scientific gate.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np

from scripts.n72r21r2_common import OUT, ROOT, read_json, write_json, sha256, append_log
import scripts.n72r21r2_candidates as inherited


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Unserializable evidence type: " + type(value).__name__)


def normalized_writer(relative, value):
    # Validate and normalize BEFORE allocating a temporary output.
    payload = json.dumps(value, default=scalar, sort_keys=True, indent=2, allow_nan=False) + "\n"
    normalized = json.loads(payload)
    path = OUT / relative
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError("R2 output isolation required")
    if path.exists():
        if read_json(path) != normalized:
            raise FileExistsError("Never overwrite completed integrity evidence")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".serialization_v2." + str(os.getpid()) + ".tmp")
    with temporary.open("x") as handle:
        handle.write(payload)
    temporary.replace(path)
    return path


def integrity(sequence):
    previous = inherited.write_json
    inherited.write_json = normalized_writer
    try:
        seal = inherited.integrity(sequence)
    finally:
        inherited.write_json = previous
    path = OUT / "data/candidate_integrity" / (sequence + ".json")
    partial = path.with_name(path.name + ".tmp")
    receipt = {"stage": "N72R21R2", "sequence": sequence, "status": "COMPLETE_NUMERIC_AUDIT_SERIALIZATION",
               "source_sha256": sha256(Path(__file__)), "inherited_integrity_source_sha256": sha256(ROOT / "scripts/n72r21r2_candidates.py"),
               "candidate_integrity_sha256": sha256(path), "inference_rerun": False,
               "scientific_policy_changed": False, "all_input_UID_feature_frame_checks_reexecuted": True,
               "old_partial_temporary_preserved": str(partial) if partial.exists() else None,
               "old_partial_sha256": sha256(partial) if partial.exists() else None}
    write_json("data/serialization_repair_v2/" + sequence + ".json", receipt)
    append_log("M1_CANDIDATE_INTEGRITY_SERIALIZED_V2", sequence=sequence, frames=seal["frames"], old_partial_preserved=partial.exists())
    return seal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    args = parser.parse_args()
    print(json.dumps({"integrity": integrity(args.sequence)["status"], "sequence": args.sequence}), flush=True)
