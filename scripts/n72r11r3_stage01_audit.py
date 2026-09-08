#!/usr/bin/env python3
"""Record the N72R11R3 shared-state and interface audit without SAM3."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.reacquisition.temporal_state_policy import TEMPORAL_FEATURE_SCHEMA


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/N72R11R3/stage_01_shared_state_audit.json")
    args = parser.parse_args()
    files = [
        ROOT / "sam3_intermot/reacquisition/temporal_state_policy.py",
        ROOT / "sam3_intermot/association/target_edge_interface.py",
        ROOT / "sam3_intermot/association/target_edge_bridge.py",
        ROOT / "scripts/n72r11_build_causal_corpus.py",
        ROOT / "scripts/n72r11_train_v3.py",
        ROOT / "scripts/n72r11_train_target_edge_bridge.py",
        ROOT / "scripts/n72r11_on_demand_replay.py",
    ]
    texts = {str(path.relative_to(ROOT)): path.read_text(encoding="utf-8") for path in files}
    checks = {
        "shared_temporal_state_module": "class TemporalIdentityState" in texts["sam3_intermot/reacquisition/temporal_state_policy.py"],
        "builder_uses_shared_temporal_features": "build_temporal_features(" in texts["scripts/n72r11_build_causal_corpus.py"],
        "builder_uses_shared_state_update": "update_temporal_state(" in texts["scripts/n72r11_build_causal_corpus.py"],
        "training_uses_shared_temporal_features": "build_temporal_features(" in texts["scripts/n72r11_train_v3.py"],
        "training_uses_shared_state_update": "update_temporal_state(" in texts["scripts/n72r11_train_v3.py"],
        "runtime_uses_shared_temporal_features": "build_temporal_features(" in texts["scripts/n72r11_on_demand_replay.py"],
        "runtime_uses_shared_state_update": "update_temporal_state(" in texts["scripts/n72r11_on_demand_replay.py"],
        "runtime_uses_correct_future_flag": "has_future_requery=" in texts["scripts/n72r11_on_demand_replay.py"],
        "runtime_uses_post_session_materializer": "post_session_feature_materializer" in texts["scripts/n72r11_on_demand_replay.py"],
        "bridge_requires_explicit_motion": "motion_iou: float" in texts["sam3_intermot/association/target_edge_bridge.py"],
        "bridge_has_scalar_builder": "build_target_edge_feature_from_scalars" in texts["sam3_intermot/association/target_edge_bridge.py"],
        "bridge_training_calls_shared_scalar_builder": "build_target_edge_feature_from_scalars" in texts["scripts/n72r11_train_target_edge_bridge.py"],
        "runtime_has_three_component_variants": all(name in texts["scripts/n72r11_on_demand_replay.py"] for name in ("E1_V3_LEGACY_INJECTION", "E2_V3_CORRECTED_BRIDGE", "E3_V3_CORRECTED_BRIDGE_LIVE")),
    }
    result = {
        "schema_version": "N72R11R3_STAGE_01_SHARED_STATE_AUDIT_V1",
        "status": "PASS_N72R11R3_SHARED_STATE_AUDIT" if all(checks.values()) else "FAIL_N72R11R3_SHARED_STATE_AUDIT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": "1f59671309d676c19903ccca9439011b7bbc8887",
        "files": {key: {"path": str(ROOT / key), "sha256": sha256(ROOT / key)} for key in texts},
        "checks": checks,
        "temporal_feature_schema": list(TEMPORAL_FEATURE_SCHEMA),
        "historical_evidence_read_only": [
            str(ROOT / "outputs/N72R11R2/resource_censoring_audit.json"),
            str(ROOT / "outputs/N72R11R2/v3_training_resource_censored/v3_readiness_audit.json"),
            str(ROOT / "outputs/N72R11R2/formal_replay_resource_censored_combined_manifest.json"),
        ],
        "scientific_effect_result": None,
        "runtime_future_gt_used": False,
        "production_authorized": False,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    atomic_json(output, result)
    print(json.dumps({**result, "output": str(output), "output_sha256": sha256(output)}, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
