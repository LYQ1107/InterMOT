"""V2 physical-GPU isolation plus versioned NumPy-count evidence writer."""
import argparse
from pathlib import Path
from scripts.n72r21r2_common import write_json, sha256
from scripts.n72r21r2_integrity_v2 import integrity
import scripts.n72r21r2_candidates_v2 as worker


def run(sequence):
    worker.integrity = integrity
    write_json("data/extraction_wrapper_v3/" + sequence + ".json", {"stage": "N72R21R2", "sequence": sequence,
               "wrapper_source_sha256": sha256(Path(__file__)), "resource_worker_source_sha256": sha256(Path(worker.__file__)),
               "changes": "One-GPU process isolation and pure audit NumPy scalar serialization only",
               "scientific_candidate_configuration_changed": False})
    worker.run(sequence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    args = parser.parse_args()
    run(args.sequence)
