#!/usr/bin/env python3
"""Run the N33 CCAM mechanism ablation on a complete candidate tape.

This is intentionally an offline smoke/replay driver.  It does not discover
identities, read GT, score a selected candidate against future labels, or
invent MOT metrics.  Without a real candidate-complete tape it writes an
explicit NOT_AVAILABLE artifact and performs no replay.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_INPUT = ROOT / "outputs" / "n33" / "candidate_complete_tape.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "n33" / "ccam_ablation.json"


VARIANTS = {
    "M0": {
        "description": "K1-only baseline; CCAM disabled",
        "use_appearance_memory": False,
    },
    "M1": {
        "description": "human EMA prototype only",
        "use_appearance_memory": True,
        "appearance_anchor_cap": 0,
        "appearance_negative_cap": 0,
    },
    "M2": {
        "description": "human EMA prototype plus multi-positive anchors",
        "use_appearance_memory": True,
        "appearance_anchor_cap": 8,
        "appearance_negative_cap": 0,
    },
    "M3": {
        "description": "multi-positive anchors plus competing negative bank",
        "use_appearance_memory": True,
        "appearance_anchor_cap": 8,
        "appearance_negative_cap": 16,
    },
    "M4": {
        "description": "M3 plus reliability/age gate",
        "use_appearance_memory": True,
        "appearance_anchor_cap": 8,
        "appearance_negative_cap": 16,
        "appearance_reliability_threshold": 0.5,
        "appearance_decay_frames": 60.0,
    },
}


def _atomic_json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _not_available(reason: str, validation: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "protocol": "N33_CCAM_M0_M4_CANDIDATE_COMPLETE_REPLAY",
        "status": "NOT_AVAILABLE",
        "candidate_complete_replay": "NOT_AVAILABLE",
        "identity_effect": "NOT_COMPUTABLE",
        "reason": reason,
        "validation": validation,
        "metrics": {
            "future_h20_iou": "NOT_COMPUTABLE",
            "future_h50_iou": "NOT_COMPUTABLE",
            "future_missing_rate": "NOT_COMPUTABLE",
            "id_switch_count": "NOT_COMPUTABLE",
            "re_correction_count": "NOT_COMPUTABLE",
            "protected_identity_regression": "NOT_COMPUTABLE",
            "sequence_cluster_bootstrap_ci": "NOT_COMPUTABLE",
            "idf1": "NOT_COMPUTABLE",
            "hota": "NOT_COMPUTABLE",
        },
        "variants": {
            name: {"status": "NOT_RUN", "description": spec["description"]}
            for name, spec in VARIANTS.items()
        },
    }


def run(input_path: Path, output_path: Path) -> Dict[str, Any]:
    if not input_path.exists():
        artifact = _not_available(f"candidate_complete_tape_missing:{input_path}")
        _atomic_json_write(output_path, artifact)
        return artifact
    try:
        with input_path.open("r", encoding="utf-8") as handle:
            tape = json.load(handle)
    except Exception as exc:
        artifact = _not_available(f"candidate_complete_tape_unreadable:{type(exc).__name__}:{exc}")
        _atomic_json_write(output_path, artifact)
        return artifact

    try:
        from sam3_intermot.association.ccam_replay import paired_replay, validate_candidate_tape
        from sam3_intermot.association.state_manager import StateManagerConfig
    except Exception as exc:
        artifact = _not_available(f"replay_import_unavailable:{type(exc).__name__}:{exc}")
        _atomic_json_write(output_path, artifact)
        return artifact

    validation = validate_candidate_tape(tape)
    if not validation["valid"] or not validation["candidate_complete"]:
        artifact = _not_available("candidate_tape_not_complete", validation)
        _atomic_json_write(output_path, artifact)
        return artifact

    base_config = StateManagerConfig(variant="reid")
    variants: Dict[str, Any] = {}
    overall_status = "PASS"
    for name, spec in VARIANTS.items():
        config = replace(
            base_config,
            **{key: value for key, value in spec.items() if key != "description"},
        )
        try:
            replay = paired_replay(tape, config=config)
            variants[name] = {
                "status": replay.get("status", "FAIL"),
                "description": spec["description"],
                "paired_replay": replay,
                "identity_effect": replay.get("identity_effect", "NOT_COMPUTABLE"),
            }
            if replay.get("status") != "PASS":
                overall_status = "FAIL"
        except Exception as exc:
            overall_status = "FAIL"
            variants[name] = {
                "status": "FAIL",
                "description": spec["description"],
                "error": f"{type(exc).__name__}: {exc}",
                "identity_effect": "NOT_COMPUTABLE",
            }
    artifact = {
        "protocol": "N33_CCAM_M0_M4_CANDIDATE_COMPLETE_REPLAY",
        "status": overall_status,
        "candidate_complete_replay": "READY" if overall_status == "PASS" else "FAIL",
        "identity_effect": "NOT_COMPUTABLE_NO_POSTHOC_IDENTITY_LABELS",
        "input": str(input_path),
        "validation": validation,
        "runtime_label_inputs": False,
        "metrics": {
            "future_h20_iou": "NOT_COMPUTABLE",
            "future_h50_iou": "NOT_COMPUTABLE",
            "future_missing_rate": "NOT_COMPUTABLE",
            "id_switch_count": "NOT_COMPUTABLE",
            "re_correction_count": "NOT_COMPUTABLE",
            "protected_identity_regression": "NOT_COMPUTABLE",
            "sequence_cluster_bootstrap_ci": "NOT_COMPUTABLE",
            "idf1": "NOT_COMPUTABLE",
            "hota": "NOT_COMPUTABLE",
        },
        "variants": variants,
    }
    _atomic_json_write(output_path, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = run(args.input, args.output)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "candidate_complete_replay": artifact["candidate_complete_replay"],
                "identity_effect": artifact["identity_effect"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
