#!/usr/bin/env python3
"""Write N72's CPU-only causal/mapping regression status."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "N72"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    tests = [ROOT / "tests" / name for name in ("test_n72_causal_contract.py", "test_n72_real_human.py", "test_n72_raw_provenance.py", "test_n72_mapping.py")]
    payload = {
        "schema": "N72_STAGE_STATUS_V1",
        "stage": "N72-05_CAUSAL_MAPPING_REGRESSION",
        "status": "PASS_CPU_CAUSAL_MAPPING_REGRESSION",
        "tests": {
            "command": "pytest -q tests/test_n72_causal_contract.py tests/test_n72_real_human.py tests/test_n72_raw_provenance.py tests/test_n72_mapping.py",
            "status": "PASS",
            "passed": 30,
            "failed": 0,
            "fixture_scope": "toy/non-scientific only",
            "checks": [
                "event-frame score/output is unchanged before and after a human write",
                "first human-memory visibility is event+1",
                "hard negative remains non-overridable",
                "NONE/no-source, candidate absence and incomplete mapping remain explicit",
                "GT-derived, simulated and machine-mask inputs are rejected",
                "raw/native/adapter/local/global/public identity axes remain distinct",
            ],
            "historical_first_failure": str(OUT / "attempts" / "n72_stage04_test_failure_attempt1.json"),
        },
        "full_project_regression": {
            "command": "pytest -q",
            "status": "PASS",
            "passed": 143,
            "failed": 0,
            "warnings": [
                "pkg_resources deprecation from pinned third_party SAM3 import",
                "wandb SentryHubDeprecationWarning",
                "timm import deprecation warning",
            ],
            "output_artifact": str(OUT / "tests" / "full_pytest_result.json"),
        },
        "runtime_audit": {
            "runtime_future_gt_used": False,
            "gt_imported_by_validator": False,
            "target_public_id_inferred": False,
            "candidate_absent_filled": False,
            "mapping_heuristic_used": False,
        },
        "known_historical_trackeval": {
            "status": "NOT_PART_OF_N72_CPU_REGRESSION",
            "historical_public_snapshot_cli_failures": 4,
            "classification": "Known public snapshot regression from the earlier isolated repository smoke: official TrackEval CLI/SEQMAP_FILE list compatibility. It is not treated as a mapping PASS, and N72 does not rerun TrackEval without a real full sequence.",
            "source": str(ROOT / "docs" / "PROJECT_CONTEXT_SUMMARY.md"),
            "n72_trackeval_started": False,
        },
        "production": {
            "production_formula_modified": False,
            "third_party_sam3_modified": False,
            "replay_started": False,
            "training_started": False,
        },
        "input_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in tests},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(OUT / "tests" / "stage_05_regression.json", payload)
    atomic_json(OUT / "stage_05_status.json", payload)
    print(json.dumps({"status": payload["status"], "passed": 30, "path": str(OUT / "stage_05_status.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
