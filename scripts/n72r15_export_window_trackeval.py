#!/usr/bin/env python3
"""Export the sealed N72R15 windows for the pinned official TrackEval.

The window writer is shared with the already-frozen N72R14 exporter through a
short-lived compatibility manifest.  No N72R14 file is edited; the translated
manifest exists only while the exporter reads it and the resulting manifest is
rewritten with the N72R15 provenance and variant names.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import traceback
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import n72r14_export_window_trackeval as legacy_export


HORIZONS = (20, 50, 100)
VARIANTS = (
    "E0_BASELINE_B0",
    "E1I_HUMAN_RELATIVE_STATE",
    "E1J_TRUSTED_GLOBAL_RELATIVE_STATE",
)
PINNED_TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"
DEFAULT_MANIFEST = ROOT / "outputs/N72R15/formal_attempt_04/formal_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/N72R15/trackeval_attempt_01"
DEFAULT_DATA = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    legacy_export.atomic_json(path, payload)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _compat_manifest(formal: dict[str, Any]) -> dict[str, Any]:
    """Translate only status labels required by the frozen generic writer."""

    view = dict(formal)
    view["status"] = "PASS_N72R14_FORMAL_REPLAY"
    view["events"] = []
    for event in formal.get("events", []):
        translated = dict(event)
        translated["status"] = "PASS_N72R14_FORMAL_EVENT"
        translated["variants"] = []
        for variant in event.get("variants", []):
            item = dict(variant)
            item["status"] = "PASS_N72R14_VARIANT"
            translated["variants"].append(item)
        view["events"].append(translated)
    return view


def export(*, output_root: Path, formal_manifest: Path, data_root: Path, trackeval_root: Path) -> dict[str, Any]:
    formal = read_json(formal_manifest)
    if formal.get("status") != "PASS_N72R15_FORMAL_REPLAY" or int(formal.get("event_count", -1)) != 32:
        raise legacy_export.wt.WindowTrackEvalError("N72R15 formal manifest is not complete")
    legacy_export.VARIANTS = VARIANTS
    legacy_export.PINNED_TRACKEVAL_COMMIT = PINNED_TRACKEVAL_COMMIT
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="n72r15-formal-view-") as temporary:
        compatibility = Path(temporary) / "formal_manifest.json"
        atomic_json(compatibility, _compat_manifest(formal))
        manifest = legacy_export.export(
            output_root=output_root,
            formal_manifest=compatibility,
            data_root=data_root,
            trackeval_root=trackeval_root,
        )
    manifest.update(
        {
            "schema_version": "N72R15_WINDOW_TRACKEVAL_EXPORT_V1",
            "status": "PASS_EXPORT_N72R15",
            "source_formal_manifest": str(formal_manifest),
            "source_formal_manifest_sha256": sha256_file(formal_manifest),
            "logical_variants": list(VARIANTS),
            "trackeval_commit": PINNED_TRACKEVAL_COMMIT,
            "runtime_future_gt_used": False,
            "gt_used_only_for_posthoc_evaluation": True,
            "interaction_source": "simulated_from_gt",
            "not_real_human_evidence": True,
            "historical_outputs_modified": False,
            "created_at_utc": now_utc(),
        }
    )
    atomic_json(output_root / "export_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--formal-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--trackeval-root", type=Path, default=ROOT / "third_party/MOTIP/TrackEval")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    try:
        manifest = export(
            output_root=output_root,
            formal_manifest=args.formal_manifest.resolve(),
            data_root=args.data_root.resolve(),
            trackeval_root=args.trackeval_root.resolve(),
        )
        status = {
            "schema_version": "N72R15_STAGE_STATUS_V1",
            "stage": "N72R15-09-TRACKEVAL-EXPORT",
            "status": manifest["status"],
            "manifest": str(output_root / "export_manifest.json"),
            "record_count": int(manifest["record_count"]),
            "expected_record_count": 32 * len(HORIZONS) * len(VARIANTS),
            "trackeval_commit": manifest["trackeval_commit"],
            "runtime_future_gt_used": False,
            "historical_outputs_modified": False,
            "created_at_utc": now_utc(),
        }
        atomic_json(output_root.parent / "stage_09_status.json", status)
        print(json.dumps(status, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "N72R15_FAILURE_V1",
            "status": "FAIL_N72R15_TRACKEVAL_EXPORT",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "created_at_utc": now_utc(),
            "runtime_future_gt_used": False,
            "historical_outputs_modified": False,
        }
        atomic_json(output_root / "export_failure.json", failure)
        atomic_json(output_root.parent / "stage_09_status.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
