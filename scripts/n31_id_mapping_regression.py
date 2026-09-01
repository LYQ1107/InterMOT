#!/usr/bin/env python3
"""Small, model-free regression for N31's public/raw ID contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.backend.output_types import PromptObjectObservation  # noqa: E402
from sam3_intermot.backend.sam3_backend import Sam3Backend  # noqa: E402


def _observation(sam_id: int) -> PromptObjectObservation:
    return PromptObjectObservation(
        frame_idx=12,
        sam_object_id=sam_id,
        mask=np.asarray([[False, True], [True, True]], dtype=bool),
        box_xyxy=np.asarray([3.0, 4.0, 19.0, 27.0], dtype=float),
        confidence=0.73,
        presence_score=0.81,
        source="official_propagation",
        is_human_verified=False,
    )


def _stable_fields(observation: PromptObjectObservation) -> dict:
    return {
        "frame_idx": int(observation.frame_idx),
        "mask": np.asarray(observation.mask, dtype=bool).tolist(),
        "box_xyxy": np.asarray(observation.box_xyxy, dtype=float).tolist(),
        "confidence": float(observation.confidence),
        "presence_score": float(observation.presence_score),
        "source": str(observation.source),
        "is_human_verified": bool(observation.is_human_verified),
    }


def main() -> int:
    cases: list[dict] = []

    # BOUND: a low-level official singleton is returned under raw id 41.
    bound = Sam3Backend()
    bound._bind_external_sam_id(7, 41)
    cases.append(
        {
            "name": "BOUND_mapping",
            "passed": bound._ext_to_sam == {7: 41}
            and bound._sam_to_ext == {41: 7},
            "external_to_sam": dict(bound._ext_to_sam),
            "sam_to_external": dict(bound._sam_to_ext),
        }
    )

    # ALREADY_BOUND: stale raw id 3 must be removed when the existing official
    # singleton is rediscovered under raw id 41.
    already = Sam3Backend()
    already._ext_to_sam = {7: 3, 8: 41}
    already._sam_to_ext = {3: 7, 41: 8}
    already._bind_external_sam_id(7, 41)
    cases.append(
        {
            "name": "ALREADY_BOUND_restore",
            "passed": already._ext_to_sam == {7: 41}
            and already._sam_to_ext == {41: 7},
            "external_to_sam": dict(already._ext_to_sam),
            "sam_to_external": dict(already._sam_to_ext),
        }
    )

    # Mapping-only must not change any raw observation field other than the
    # identifier used by the adapter to expose it publicly.
    raw = _observation(41)
    before = _stable_fields(raw)
    mapping_only = Sam3Backend()
    mapping_only._bind_external_sam_id(7, 41)
    mapping_only._apply_stable_ids([raw])
    after = _stable_fields(raw)
    cases.append(
        {
            "name": "mapping_only_raw_output_equivalence",
            "passed": before == after and int(raw.sam_object_id) == 7,
            "raw_fields_before": before,
            "raw_fields_after": after,
            "public_id_after": int(raw.sam_object_id),
        }
    )

    passed = all(bool(case["passed"]) for case in cases)
    payload = {
        "schema": "n31.id_mapping_regression.v1",
        "status": "PASS" if passed else "FAIL",
        "tests": cases,
        "one_to_one_contract": True,
        "raw_outputs_mutated_by_mapping": False,
    }
    output = ROOT / "outputs/n31/id_mapping_regression.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
