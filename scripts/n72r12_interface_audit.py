#!/usr/bin/env python3
"""Record the frozen N72R12 interfaces before any replay is started.

This audit is intentionally read-only with respect to historical artifacts.  It
only imports the production modules and records the exact source/checkpoint
inputs that the later CSI replay is allowed to use.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association import counterfactual_safe_intervention as csi  # noqa: E402
from sam3_intermot.association.target_edge_interface import (  # noqa: E402
    apply_legacy_injection,
    select_candidate_from_logits,
)
from scripts import n72r11_on_demand_replay as replay  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
        Path(temporary).replace(path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> int:
    source_files = [
        ROOT / "sam3_intermot/association/target_edge_interface.py",
        ROOT / "sam3_intermot/association/effect_assignment.py",
        ROOT / "sam3_intermot/association/public_assignment.py",
        ROOT / "sam3_intermot/reacquisition/temporal_state_policy.py",
        ROOT / "sam3_intermot/reacquisition/target_candidate_pool.py",
        ROOT / "sam3_intermot/association/counterfactual_safe_intervention.py",
        ROOT / "scripts/n72r11_on_demand_replay.py",
    ]
    missing = [str(path) for path in source_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing audited source files: " + ", ".join(missing))
    signatures = {
        "select_candidate_from_logits": str(inspect.signature(select_candidate_from_logits)),
        "apply_legacy_injection": str(inspect.signature(apply_legacy_injection)),
        "replay._score_pool": str(inspect.signature(replay._score_pool)),
        "csi.build_counterfactual_audit": str(inspect.signature(csi.build_counterfactual_audit)),
        "csi.decide_counterfactual_safe_v1": str(inspect.signature(csi.decide_counterfactual_safe_v1)),
        "csi.commit_counterfactual_decision": str(inspect.signature(csi.commit_counterfactual_decision)),
    }
    source_text = (ROOT / "scripts/n72r11_on_demand_replay.py").read_text(encoding="utf-8")
    required_source_markers = {
        "proposal_solver_alias": '"proposal_solver": solver' in source_text,
        "proposal_target_uid_alias": '"proposal_target_uid": target_uid' in source_text,
        "proposal_fused_matrix_alias": '"proposal_fused_matrix": fused' in source_text,
        "e1c_variant": '"E1C_PCTIS_SAFE"' in source_text,
        "runtime_gt_flags": '"runtime_future_gt_used": False' in source_text,
    }
    frozen = {}
    for path in (
        ROOT / "outputs/N72R9/protocol.json",
        ROOT / "outputs/N72R11R5R1/formal_e1a_metrics.json",
        ROOT / "outputs/N72R11R5R1/formal_e1b_metrics.json",
        ROOT / "outputs/N72R11R5R1/formal_e1a_manifest.json",
        ROOT / "outputs/N72R11R5R1/formal_e1b_manifest.json",
    ):
        frozen[str(path.relative_to(ROOT))] = {"exists": path.is_file(), "sha256": _sha256(path) if path.is_file() else None}
    payload = {
        "schema_version": "N72R12_INTERFACE_AUDIT_V1",
        "status": "PASS_INTERFACE_AUDIT" if all(required_source_markers.values()) and not missing else "FAIL_INTERFACE_AUDIT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": _git("symbolic-ref", "--short", "HEAD"),
        "git_commit": _git("rev-parse", "HEAD"),
        "source_files": {str(path.relative_to(ROOT)): _sha256(path) for path in source_files},
        "frozen_inputs": frozen,
        "signatures": signatures,
        "source_markers": required_source_markers,
        "prior_attempts": [
            {
                "command": "python scripts/n72r12_interface_audit.py",
                "exit_code": 1,
                "status": "FAIL_ENVIRONMENT_IMPORT",
                "error_type": "ModuleNotFoundError",
                "error": "No module named 'torch'",
                "resolved_by": "/home/lwr/anaconda3/envs/intermot/bin/python",
            }
        ],
        "constraints": {
            "historical_outputs_modified": False,
            "third_party_modified": False,
            "runtime_future_gt_used": False,
            "solver_changed": False,
            "checkpoint_changed": False,
            "live_requery": "NOT_RUN_IN_N72R12",
        },
    }
    out = ROOT / "outputs/N72R12/interface_audit.json"
    _atomic_json(out, payload)
    _atomic_json(
        ROOT / "outputs/N72R12/stage_00_status.json",
        {
            "schema_version": "N72R12_STAGE_STATUS_V1",
            "stage": "Stage00",
            "status": payload["status"],
            "audit": str(out),
            "python": sys.executable,
            "prior_failed_attempt_preserved": True,
            "historical_outputs_modified": False,
            "third_party_modified": False,
        },
    )
    print(json.dumps({"status": payload["status"], "output": str(out)}, sort_keys=True))
    return 0 if payload["status"] == "PASS_INTERFACE_AUDIT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
