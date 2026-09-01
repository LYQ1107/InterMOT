#!/usr/bin/env python3
"""N29-D gate and offline episodic decoder-adaptation entry point.

N29-D is deliberately a declared fallback for a failed/unavailable faithful
online decoder update.  It must consume an explicit episode manifest whose
support/query tensors are tied to the official propagation decoder.  The
script refuses relation-feature caches, pseudo episodes without legal spatial
targets, and anything mentioning the blind validation/test split.  With no
such manifest in the current checkout it records ``NOT_RUN`` rather than
training on synthetic or silently substituted data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "outputs" / "n29" / "decoder_episode_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "n29" / "n29d_result.json"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _contains_blind(value: Any) -> bool:
    text = json.dumps(value, sort_keys=True, default=str).lower()
    return any(token in text for token in ("val25", "validation", "/val/", "test/"))


def _episode_records(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = data.get("episodes", data.get("entries", []))
    if not isinstance(records, list):
        raise ValueError("episode manifest must contain a list under episodes or entries")
    return [record for record in records if isinstance(record, Mapping)]


def _audit_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "NOT_RUN",
            "reason": "no_legal_decoder_support_query_episode_manifest",
            "manifest": str(path),
            "ready_episode_count": 0,
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "NOT_RUN",
            "reason": f"invalid_episode_manifest: {type(exc).__name__}: {exc}",
            "manifest": str(path),
            "ready_episode_count": 0,
        }
    if not isinstance(data, Mapping):
        return {
            "status": "NOT_RUN",
            "reason": "episode_manifest_root_is_not_an_object",
            "manifest": str(path),
            "ready_episode_count": 0,
        }
    if bool(data.get("val25_read", False)) or _contains_blind(data):
        return {
            "status": "NOT_RUN",
            "reason": "blind_boundary_violation_manifest_rejected",
            "manifest": str(path),
            "ready_episode_count": 0,
        }
    try:
        records = _episode_records(data)
    except ValueError as exc:
        return {
            "status": "NOT_RUN",
            "reason": str(exc),
            "manifest": str(path),
            "ready_episode_count": 0,
        }
    required = (
        "video_id",
        "public_id",
        "support_input_path",
        "query_input_path",
        "target_mask_path",
    )
    ready: list[Mapping[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        role = str(record.get("role", ""))
        missing = [key for key in required if not record.get(key)]
        if role and role not in {"train_fold", "external_train"}:
            missing.append("role=train_fold_or_external_train")
        if _contains_blind(record):
            missing.append("blind_boundary")
        if missing:
            rejected.append({"index": index, "missing_or_invalid": missing})
        else:
            ready.append(record)
    if not ready:
        return {
            "status": "NOT_RUN",
            "reason": "manifest_has_no_legal_support_query_target_records",
            "manifest": str(path),
            "record_count": len(records),
            "ready_episode_count": 0,
            "rejected": rejected[:32],
        }
    return {
        "status": "NOT_RUN",
        "reason": "offline_executor_not_bound_to_official_decoder_tensor_manifest",
        "manifest": str(path),
        "record_count": len(records),
        "ready_episode_count": len(ready),
        "rejected_count": len(rejected),
        "ready_video_ids": sorted({str(record["video_id"]) for record in ready}),
        "note": "Schema validation is complete; no weights were updated and no synthetic episode was substituted.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-episodes", type=int, default=4)
    args = parser.parse_args()
    if args.max_episodes <= 0:
        raise SystemExit("--max-episodes must be positive")
    result: dict[str, Any] = {
        "protocol": "N29-D",
        "status": "NOT_RUN",
        "variant": "offline_episodic_decoder_adaptation",
        "val25_read": False,
        "max_gpu_count": 4,
        "max_episodes": args.max_episodes,
        "architecture": "pretrained official SAM3 propagation decoder with the exact N29 A+B adapter boundary",
        "legal_provenance_required": [
            "HUMAN_CONFIRMED_MASK",
            "POINT_REFINED_CONFIRMED_MASK",
            "BOX_PROMPTED_CONFIRMED_MASK",
            "BOX_DERIVED_PSEUDO_MASK",
        ],
    }
    result.update(_audit_manifest(args.episode_manifest))
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
