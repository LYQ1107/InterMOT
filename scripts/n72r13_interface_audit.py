#!/usr/bin/env python3
"""Record the concrete N72R13 interfaces and frozen input provenance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _signature(value: Any) -> str:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return "<signature-unavailable>"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/N72R13/interface_audit.json")
    args = parser.parse_args()

    from sam3_intermot.association.effect_assignment import solve_effect_assignment
    from sam3_intermot.association.public_assignment import solve_exact_public_assignment
    from sam3_intermot.reacquisition import temporal_state_policy as state_policy
    from sam3_intermot.reacquisition.target_candidate_pool import (
        build_candidate_pool,
        build_candidate_pool_with_future_requery,
    )
    from sam3_intermot.reacquisition.temporal_scorer_adapter import (
        PublicCompetitionScorerAdapter,
        V3TemporalScorerAdapter,
    )
    from scripts import n72r11_on_demand_replay as replay

    source_files = {
        "replay": Path(inspect.getfile(replay)),
        "temporal_state_policy": Path(inspect.getfile(state_policy)),
        "target_candidate_pool": Path(inspect.getfile(build_candidate_pool)),
        "temporal_scorer_adapter": Path(inspect.getfile(V3TemporalScorerAdapter)),
        "effect_assignment": Path(inspect.getfile(solve_effect_assignment)),
        "public_assignment": Path(inspect.getfile(solve_exact_public_assignment)),
    }
    frozen_files = {
        "n72r12_final_report": ROOT / "docs/N72R12_FINAL_REPORT.md",
        "n72r12_final_gate": ROOT / "outputs/N72R12/n72r12_final_gate.json",
        "n72r12_controller": ROOT / "outputs/N72R12/CONTROLLER_STATUS.json",
        "n72r9_protocol": ROOT / "outputs/N72R9/protocol.json",
        "n72r11r4_corpus_manifest": ROOT / "outputs/N72R11R4/exact_onpolicy_v3/corpus_manifest.json",
        "pctis_checkpoint": ROOT / "outputs/N72R11R4/pctis_onpolicy_finetune/pctis_onpolicy_finetuned.pt",
    }
    missing = [str(path) for path in [*source_files.values(), *frozen_files.values()] if not path.is_file()]
    if missing:
        payload = {
            "schema_version": "N72R13_INTERFACE_AUDIT_V1",
            "status": "BLOCKED_MISSING_FROZEN_INTERFACE_INPUT",
            "missing": missing,
            "created_at_utc": now_utc(),
            "historical_outputs_modified": False,
        }
        atomic_json(args.output.resolve(), payload)
        atomic_json(
            args.output.resolve().parent / "stage_00_status.json",
            {
                "schema_version": "N72R13_STAGE_STATUS_V1",
                "stage": "N72R13-00-INTERFACE-AUDIT",
                "status": "BLOCKED_MISSING_FROZEN_INTERFACE_INPUT",
                "failure_artifact": str(args.output.resolve()),
                "created_at_utc": now_utc(),
                "historical_outputs_modified": False,
            },
        )
        print(json.dumps(payload, sort_keys=True))
        return 1

    payload = {
        "schema_version": "N72R13_INTERFACE_AUDIT_V1",
        "status": "PASS_N72R13_INTERFACE_AUDIT",
        "created_at_utc": now_utc(),
        "historical_outputs_modified": False,
        "runtime_contract": {
            "candidate_pool": {
                "build_candidate_pool": _signature(build_candidate_pool),
                "build_candidate_pool_with_future_requery": _signature(build_candidate_pool_with_future_requery),
                "source_authority": "candidate public_id is null before exact solver",
                "positive_geometry": "require_positive_geometry is opt-in and filters before solver",
            },
            "scoring": {
                "_score_pool": _signature(replay._score_pool),
                "edge_modes": [replay.EDGE_MODE_BASE, replay.EDGE_MODE_LEGACY_INJECTION, replay.EDGE_MODE_BRIDGE],
                "pctis_adapter": {
                    "class": "PublicCompetitionScorerAdapter",
                    "score": _signature(PublicCompetitionScorerAdapter.score),
                },
                "v3_adapter": {
                    "class": "V3TemporalScorerAdapter",
                    "score": _signature(V3TemporalScorerAdapter.score),
                },
            },
            "temporal_state": {
                "class": "TemporalIdentityState",
                "constructor": _signature(state_policy.TemporalIdentityState),
                "update_temporal_state": _signature(state_policy.update_temporal_state),
                "state_audit": _signature(state_policy.state_audit),
                "feature_schema": list(state_policy.TEMPORAL_FEATURE_SCHEMA),
                "trusted_age_index": int(state_policy.TEMPORAL_TRUSTED_AGE_INDEX),
            },
            "exact_solver": {
                "effect_wrapper": _signature(solve_effect_assignment),
                "public_solver": _signature(solve_exact_public_assignment),
                "solver": "scipy.optimize.linear_sum_assignment over candidate x public + explicit NONE",
            },
        },
        "n72r13_contract": {
            "primary_value_horizon": 20,
            "value_horizons": [5, 20, 50],
            "passive_future_assignment": "BASE matrix + exact solver; no PCTIS injection/live re-query",
            "branch_state": "deep independent TemporalIdentityState copies",
            "runtime_future_gt_used": False,
            "runtime_gt_read": False,
            "posthoc_gt_used": False,
        },
        "source_files": {
            key: {"path": str(path), "sha256": sha256_file(path)} for key, path in source_files.items()
        },
        "frozen_inputs": {
            key: {"path": str(path), "sha256": sha256_file(path)} for key, path in frozen_files.items()
        },
        "checkpoint_paths": {"pctis": str(frozen_files["pctis_checkpoint"])},
        "checkpoint_sha256": sha256_file(frozen_files["pctis_checkpoint"]),
        "interaction_source": "simulated_from_gt",
        "real_human_evidence": False,
        "not_real_human_evidence": True,
    }
    atomic_json(args.output.resolve(), payload)
    atomic_json(
        args.output.resolve().parent / "stage_00_status.json",
        {
            "schema_version": "N72R13_STAGE_STATUS_V1",
            "stage": "N72R13-00-INTERFACE-AUDIT",
            "status": "PASS_N72R13_INTERFACE_AUDIT",
            "interface_audit": str(args.output.resolve()),
            "created_at_utc": now_utc(),
            "historical_outputs_modified": False,
            "runtime_future_gt_used": False,
            "real_human_evidence": False,
        },
    )
    print(json.dumps({"status": payload["status"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
