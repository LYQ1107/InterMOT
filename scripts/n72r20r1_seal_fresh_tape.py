#!/usr/bin/env python3
"""Seal the compact train-smoke candidate/base-score lineage manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = Path("/data3/liuyeqiang/InterMOT_N72R20R1_assets")
SEQUENCES = ("dancetrack0001", "dancetrack0002")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def code_record(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    return {"path": relative, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/N72R20R1/fresh_tape_manifest.json")
    args = parser.parse_args()
    asset_root = args.asset_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"fresh tape manifest already exists: {output}")
    candidate_tapes: dict[str, Any] = {}
    base_tapes: dict[str, Any] = {}
    for sequence in args.sequences:
        sequence = str(sequence)
        candidate_root = asset_root / "candidates" / sequence
        base_root = asset_root / "base_scores" / sequence
        candidate_done = json.loads((candidate_root / "done.json").read_text(encoding="utf-8"))
        base_done = json.loads((base_root / "done.json").read_text(encoding="utf-8"))
        if candidate_done.get("status") != "PASS_N72R20R1_CANDIDATE_STREAM_SMOKE_SEQUENCE":
            raise ValueError(f"candidate tape is not complete: {sequence}")
        if base_done.get("status") != "PASS_N72R20R1_BASE_SCORE_TAPE_SEQUENCE":
            raise ValueError(f"base tape is not complete: {sequence}")
        candidate_tapes[sequence] = {
            name: file_record(candidate_root / name)
            for name in ("metadata.jsonl.zst", "embeddings.f16", "index.json", "done.json")
        }
        base_tapes[sequence] = {
            name: file_record(base_root / name)
            for name in ("base_scores.jsonl.zst", "index.json", "done.json")
        }
    event_path = asset_root / "interaction_events.json"
    labels_path = asset_root / "posthoc_event_labels.json"
    storage_audit = ROOT / "outputs/N72R20R1/storage_audit_before.json"
    payload: dict[str, Any] = {
        "stage": "N72R20R1",
        "fresh_lineage": True,
        "fresh_tape_sealed": True,
        "seal_scope": "train_smoke_development",
        "formal_val_requires_separate_lineage": True,
        "historical_n72r15_reproduction": False,
        "historical_n72r15_used": False,
        "historical_n72r20_status": "BLOCKED_NOT_RECOVERED",
        "asset_root": str(asset_root),
        "dataset_provenance": {
            "root": "/data3/liuyeqiang/InterMOT_N72R16_assets/dataset",
            "split": "train",
            "sequences": [str(value) for value in args.sequences],
            "test_downloaded_or_used": False,
            "storage_audit": file_record(storage_audit),
        },
        "interaction_protocol": {
            "events": file_record(event_path),
            "posthoc_labels": file_record(labels_path),
            "interaction_source": "simulated_from_gt",
            "runtime_consumes_gt": False,
            "target_event_frame": 0,
        },
        "candidate_generator": {
            "implementation": "current Sam3Backend plus export_frame_candidates; one process/sequence",
            "code": [
                code_record("scripts/n72r20r1_candidate_stream_smoke.py"),
                code_record("sam3_intermot/backend/sam3_backend.py"),
                code_record("sam3_intermot/adaptation/sam3_loader.py"),
                code_record("sam3_intermot/association/decoder_candidate_bridge.py"),
            ],
            "sam3_checkpoint_sha256": "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6",
            "feature_encoder_sha256": "2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154",
        },
        "base_scorer": {
            "implementation": "sam3_intermot.association.online_associator.score_matrix_pairwise",
            "code": [
                code_record("scripts/n72r20r1_build_base_score_tape.py"),
                code_record("sam3_intermot/association/online_associator.py"),
                code_record("sam3_intermot/association/identity_state.py"),
            ],
            "reid_weights": {"sim": 1.5, "iou": 1.0, "native": 0.5, "gap": 0.1},
            "native_bonus": 3.0,
            "positive_bonus": 5.0,
            "ema": 0.9,
            "gt_used": False,
        },
        "candidate_tapes": candidate_tapes,
        "base_score_tapes": base_tapes,
        "frozen_components": {
            "osnet_checkpoint_sha256": "2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154",
            "n72r18_gru_checkpoint_sha256": "94ba9c44e177b254ee54984195e839f37c78a648c010ffc723dd3f1d553976a2",
            "assignment_solver_code": [
                code_record("sam3_intermot/association/effect_assignment.py"),
                code_record("sam3_intermot/association/public_assignment.py"),
            ],
            "public_authority_code": [
                code_record("sam3_intermot/identity/public_authority.py"),
                code_record("sam3_intermot/identity/persistent_runtime.py"),
            ],
            "learned_memory_code": [
                code_record("sam3_intermot/association/learned_identity_memory.py"),
                code_record("sam3_intermot/association/learned_identity_state_edge.py"),
            ],
        },
        "shared_by_e0_e1_e2": [
            "candidate tape",
            "512-D OSNet feature payload",
            "candidate UID axis",
            "association state axis",
            "public ID axis",
            "fresh base-score matrix",
            "explicit NONE score 0.0",
            "exact public assignment solver",
            "public authority axis",
        ],
        "runtime_future_gt_used": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_N72R20R1_FRESH_TAPE_SEALED", "manifest": str(output), "manifest_sha256": payload["manifest_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
