#!/usr/bin/env python3
"""Freeze the N72R7 provenance and candidate-stream contract.

This stage is intentionally read-only with respect to N36--N72R6.  It hashes
the already sealed artifacts and records the exact external repositories used
as design references; it does not run SAM or select events from outcomes.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/N72R7"
N72R6 = ROOT / "outputs/N72R6"
N72R5 = ROOT / "outputs/N72R5"
N72R5R1 = ROOT / "outputs/N72R5R1"
FALLBACK_CHECKPOINT = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/Interactive/"
    "SAM3_InterMOT/checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
)
PROMPT = Path(
    "/home/user/.codex/attachments/aef5ef75-dca6-471b-a104-5cfae425c70d/"
    "pasted-text.txt"
)

EXTERNAL_REFERENCES: list[dict[str, Any]] = [
    {
        "name": "DAM4SAM",
        "paper": "https://arxiv.org/abs/2411.17576",
        "repository": "https://github.com/jovanavidenovic/DAM4SAM",
        "commit": "9c954504b39ebca4c412f207be0787c26bfac85a",
        "inspected_files": [
            "dam4sam_tracker.py",
            "sam2/sam2_video_predictor.py",
        ],
        "venue_year": "CVPR 2025; IJCV 2026 extension",
        "license": "No repository LICENSE file found in inspected commit; idea-only, no code copied",
        "reusable_mechanism": "reliable recent memory admission and distractor-resolving memory",
        "code_reused": False,
    },
    {
        "name": "SAM2Long",
        "paper": "https://arxiv.org/abs/2410.16268",
        "repository": "https://github.com/Mark12Ding/SAM2Long",
        "commit": "d70b50a7936fec55af201244ecde3d4433aff943",
        "inspected_files": [
            "tools/vos_inference.py",
            "sam2/sam2_video_predictor.py",
        ],
        "venue_year": "ICCV 2025",
        "license": "CC BY-NC 4.0 for the majority of the repository (README/LICENSE.txt)",
        "reusable_mechanism": "diverse multi-path hypotheses and accumulated path scores",
        "code_reused": False,
    },
    {
        "name": "TrackTrack",
        "paper": "https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html",
        "repository": "https://github.com/kamkyu94/TrackTrack",
        "commit": "ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da",
        "inspected_files": [
            "3. Tracker/trackers/tracker.py",
            "3. Tracker/trackers/track.py",
            "3. Tracker/trackers/utils.py",
        ],
        "venue_year": "CVPR 2025",
        "license": "MIT",
        "reusable_mechanism": "track-perspective association and use of low-confidence track support",
        "code_reused": False,
    },
    {
        "name": "MOTIP",
        "paper": "https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.html",
        "repository": "https://github.com/MCG-NJU/MOTIP",
        "commit": "ffc0e905ac196a603027eca8d18fb0dff48c8bcc",
        "inspected_files": [
            "models/runtime_tracker.py",
            "models/motip/id_decoder.py",
            "models/motip/id_criterion.py",
            "models/motip/trajectory_modeling.py",
        ],
        "venue_year": "CVPR 2025",
        "license": "Apache-2.0",
        "reusable_mechanism": "trajectory context plus current detections for identity prediction",
        "code_reused": False,
    },
    {
        "name": "SeC",
        "paper": "https://arxiv.org/abs/2507.15852",
        "repository": "https://github.com/OpenIXCLab/SeC",
        "commit": "0a797af5028623831c016692169df5c621037170",
        "inspected_files": [
            "inference/modeling_sec.py",
            "inference/sam2_video_predictor.py",
            "training/sec/models/sec.py",
            "training/sec/models/sam2_train.py",
            "training/TRAIN.md",
        ],
        "venue_year": "ICLR 2026",
        "license": "Apache-2.0",
        "reusable_mechanism": "progressive concept construction and enhanced pixel-level association",
        "code_reused": False,
    },
]


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_checkpoint() -> Path:
    local = ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
    return local if local.is_file() else FALLBACK_CHECKPOINT


def main() -> int:
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    if status:
        raise RuntimeError(f"Stage 00 requires a clean worktree, found: {status}")

    event_manifest = N72R5 / "mechanism_rounds/round_06_event_policy/real_event_manifest.json"
    stage08 = N72R5R1 / "controller/round_05_branch_isolation_v0/full/stage08_runtime_manifest.json"
    target_manifest = N72R6 / "recovery_target_stream_manifest_attempt3.json"
    replay_root = N72R6 / "public_replay/human_anchor_fallback_attempt1"
    replay_batch = replay_root / "replay_batch_status.json"
    frozen_inputs = [
        ROOT / "docs/N72R6_FINAL_REPORT.md",
        N72R6 / "n72r6_final_gate.json",
        N72R6 / "protocol.json",
        target_manifest,
        replay_batch,
        event_manifest,
        stage08,
        ROOT / "research_log.md",
    ]
    missing = [str(path) for path in frozen_inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen input(s): {missing}")

    target_payload = read_json(target_manifest)
    if target_payload.get("status") != "PASS_N72R6_TARGET_SESSION_RECOVERY_32_OF_32_VALIDATED":
        raise RuntimeError(f"unexpected N72R6 target manifest status: {target_payload.get('status')}")
    selected = {str(item["event_id"]): item for item in target_payload.get("selected", [])}
    if len(selected) != 32:
        raise RuntimeError(f"expected 32 selected N72R6 events, found {len(selected)}")

    b0_streams: list[dict[str, Any]] = []
    for event_id in sorted(selected):
        event_dir = replay_root / event_id
        event_file = event_dir / "event_manifest.json"
        if not event_file.is_file():
            raise FileNotFoundError(f"missing replay event manifest: {event_file}")
        manifest = read_json(event_file)
        c0 = Path(str(manifest["c0"]["path"]))
        if not c0.is_absolute():
            c0 = ROOT / c0
        if not c0.is_file():
            raise FileNotFoundError(f"missing frozen B0 stream: {c0}")
        lines = c0.read_text(encoding="utf-8").splitlines()
        if len(lines) != 101:
            raise RuntimeError(f"B0 frame count mismatch for {event_id}: {len(lines)}")
        rows = [json.loads(line) for line in lines]
        frames = [int(row.get("frame", -1)) for row in rows]
        event_frame = int(manifest["event_frame"])
        if frames != list(range(event_frame, event_frame + 101)):
            raise RuntimeError(f"B0 frame axis mismatch for {event_id}")
        if any(row.get("branch") != "B0_NO_INTERVENTION" for row in rows):
            raise RuntimeError(f"B0 branch marker mismatch for {event_id}")
        if any(row.get("runtime_future_gt_used") is not False for row in rows):
            raise RuntimeError(f"B0 runtime GT flag mismatch for {event_id}")
        b0_streams.append(
            {
                "event_id": event_id,
                "sequence": str(manifest["sequence"]),
                "event_frame": event_frame,
                "path": str(c0),
                "sha256": sha256_file(c0),
                "frame_count": len(rows),
                "candidate_rows": int(sum(len(row.get("candidate_rows", [])) for row in rows)),
            }
        )

    stream_digest = hashlib.sha256(
        "\n".join(f"{item['event_id']}:{item['sha256']}" for item in b0_streams).encode("utf-8")
    ).hexdigest()
    checkpoint = resolve_checkpoint()
    historical_hashes = {str(path): sha256_file(path) for path in frozen_inputs}
    historical_hashes[str(checkpoint)] = sha256_file(checkpoint)
    historical_hashes[str(PROMPT)] = sha256_file(PROMPT)

    protocol = {
        "schema_version": "N72R7_FROZEN_PROTOCOL_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_question": "human_conditioned_closed_loop_reacquisition_and_raw_binding_switching",
        "source_branch": branch,
        "source_commit": commit,
        "code_baseline_commit": "9bab1e4344796e43c035a87f192b70de48d5dded",
        "repository": "https://github.com/LYQ1107/InterMOT",
        "historical_outputs_read_only": True,
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "runtime_future_gt_allowed": False,
        "max_gpu_count": 4,
        "bootstrap": {"seed": 7202, "repetitions": 2000, "cluster_unit": "independent_sequence"},
        "frozen_definitions": {
            "event_frame_memory_read": False,
            "first_memory_visible_frame": "event_frame+1",
            "horizon": 100,
            "candidate_sources": [
                "MAIN_B0_CANDIDATE",
                "TARGET_SESSION_CURRENT_RAW",
                "TARGET_SESSION_ALTERNATIVE_RAW",
                "FRESH_REQUERY_CANDIDATE",
            ],
            "initial_pool_order": ["MAIN_B0_FULL_POOL", "TARGET_SESSION_POOL"],
            "none_is_explicit": True,
            "public_id_authority": "persistent_public_identity_supplied_by_event; selector_cannot_invent_ids",
            "raw_sam_id_is_authority": False,
            "solver": "existing_exact_global_public_assignment_with_explicit_none",
            "metrics": [
                "H20",
                "H50",
                "H100",
                "identity_error",
                "missing",
                "wrong_reassociation",
                "true_correct_crossing",
                "true_incorrect_crossing",
                "protected_regression",
                "candidate_recall",
                "sequence_cluster_bootstrap",
            ],
        },
        "inputs": {
            "historical_sha256": historical_hashes,
            "n72r6_final_replay_root": str(replay_root),
            "n72r6_final_replay_batch": str(replay_batch),
            "n72r6_b0_stream_count": len(b0_streams),
            "n72r6_b0_stream_candidate_row_count": int(sum(item["candidate_rows"] for item in b0_streams)),
            "n72r6_b0_stream_digest": stream_digest,
            "b0_streams": b0_streams,
        },
        "external_references": EXTERNAL_REFERENCES,
        "prohibited": [
            "modify_third_party_sam3",
            "modify_N36_to_N72R6_evidence",
            "future_gt_runtime_read",
            "GT_directly_assigns_public_id",
            "raw_sam_id_as_public_identity",
            "discard_alternative_candidates_before_selection",
            "tune_0.85_as_main_strategy",
            "change_checkpoint_candidate_definition_solver_or_metrics",
            "train_on_val_or_test",
        ],
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    protocol["protocol_sha256"] = protocol_hash
    atomic_json(OUT / "protocol.json", protocol)
    atomic_json(
        OUT / "stage_00_status.json",
        {
            "schema_version": "N72R7_STAGE_STATUS_V1",
            "stage": "N72R7-00_PROVENANCE_AND_REFERENCE_FREEZE",
            "status": "PASS_FROZEN_INPUTS",
            "protocol_sha256": protocol_hash,
            "branch": branch,
            "source_commit": commit,
            "code_baseline_commit": protocol["code_baseline_commit"],
            "worktree_status": status,
            "n72r6_event_count": len(selected),
            "n72r6_b0_stream_count": len(b0_streams),
            "n72r6_b0_stream_candidate_row_count": int(sum(item["candidate_rows"] for item in b0_streams)),
            "external_reference_count": len(EXTERNAL_REFERENCES),
            "third_party_sam3_modified": False,
            "historical_outputs_modified": False,
            "runtime_future_gt_used": False,
            "interaction_source": "simulated_from_gt",
            "real_human_evidence": False,
            "next_stage": "N72R7-01_NATIVE_SCOPE_FIX_AND_CANDIDATE_SOURCE_FORENSICS",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps({
        "status": "PASS_FROZEN_INPUTS",
        "protocol_sha256": protocol_hash,
        "b0_stream_count": len(b0_streams),
        "b0_candidate_row_count": int(sum(item["candidate_rows"] for item in b0_streams)),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
