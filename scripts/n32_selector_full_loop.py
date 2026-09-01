#!/usr/bin/env python3
"""Guard the N32 deployment claim behind the explicit Learn Gate.

The existing N31/N30 real full-loop adapters are multi-identity modules.  N32
selector rollouts are single-target policy episodes, so this adapter refuses
to convert them into an identity/MOT full-loop claim unless the separately
audited multi-ID interface is present.  A failed Learn Gate is a normal,
recorded outcome and takes the bounded association route.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/n32"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def run(*, output: Path = OUT_DIR / "full_loop_results.json") -> dict[str, Any]:
    learn_path = OUT_DIR / "learn_gate.json"
    oracle_path = OUT_DIR / "policy_oracle_689.json"
    audit_path = OUT_DIR / "selector_feature_audit.json"
    learn = json.loads(learn_path.read_text(encoding="utf-8")) if learn_path.is_file() else {"status": "MISSING"}
    oracle = json.loads(oracle_path.read_text(encoding="utf-8")) if oracle_path.is_file() else {"status": "MISSING"}
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    identity_count = audit.get("identity_features_available_episode_count", (audit.get("feature_availability") or {}).get("identity_features_available_episode_count"))
    identity_coverage = audit.get("identity_feature_coverage")
    if oracle.get("status") != "PASS":
        result = {
            "protocol": "N32-H-FULL-LOOP",
            "status": "NOT_RUN_ORACLE_GATE_FAIL",
            "reason": "the frozen 689-episode policy Oracle Gate failed; selector deployment and identity/MOT full-loop evaluation are not authorized",
            "learn_gate_status": learn.get("status"),
            "oracle_status": oracle.get("status"),
            "primary_identity_metric": "NOT_RUN",
            "future_correction_burden": "NOT_RUN",
            "det_a_harm_check": "NOT_RUN",
            "trackeval": "NOT_RUN_TRAIN_FOLD_ONLY_SCOPE",
            "same_human_budget": True,
            "association_fallback_required": True,
            "identity_features_available_episode_count": identity_count,
            "identity_feature_coverage": identity_coverage,
            "identity_aware_learning_valid": audit.get("identity_aware_learning_valid"),
            "val25_read": False,
            "test_labels_used": False,
        }
    elif learn.get("status") != "PASS":
        result = {
            "protocol": "N32-H-FULL-LOOP",
            "status": "NOT_RUN_LEARN_GATE_FAIL",
            "reason": "strategy selector Learn Gate did not authorize deployment; association fallback is required",
            "learn_gate_status": learn.get("status"),
            "oracle_status": oracle.get("status"),
            "primary_identity_metric": "NOT_RUN",
            "future_correction_burden": "NOT_RUN",
            "det_a_harm_check": "NOT_RUN",
            "trackeval": "NOT_RUN_TRAIN_FOLD_ONLY_SCOPE",
            "same_human_budget": True,
            "identity_features_available_episode_count": identity_count,
            "identity_feature_coverage": identity_coverage,
            "identity_aware_learning_valid": audit.get("identity_aware_learning_valid"),
            "val25_read": False,
            "test_labels_used": False,
        }
    else:
        # A single-ID policy tape cannot establish identity preservation or a
        # future re-correction burden.  Keep this explicit rather than
        # reusing a prior N31 identity result as if it were selector output.
        result = {
            "protocol": "N32-H-FULL-LOOP",
            "status": "NOT_RUN_MULTI_ID_ADAPTER_REQUIRED",
            "reason": "Learn Gate passed but this checkout has no authorized adapter that applies the N32 policy selector to the N31 multi-ID full-loop stream; no MOT/TrackEval claim is made",
            "learn_gate_status": learn.get("status"),
            "oracle_status": oracle.get("status"),
            "primary_identity_metric": "NOT_RUN",
            "future_correction_burden": "NOT_RUN",
            "det_a_harm_check": "NOT_RUN",
            "trackeval": "NOT_RUN_TRAIN_FOLD_ONLY_SCOPE",
            "same_human_budget": True,
            "association_fallback_required": True,
            "identity_features_available_episode_count": identity_count,
            "identity_feature_coverage": identity_coverage,
            "identity_aware_learning_valid": audit.get("identity_aware_learning_valid"),
            "val25_read": False,
            "test_labels_used": False,
        }
    _write(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_DIR / "full_loop_results.json")
    args = parser.parse_args()
    result = run(output=args.output)
    print(json.dumps({"protocol": result["protocol"], "status": result["status"], "reason": result["reason"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
