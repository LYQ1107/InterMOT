#!/usr/bin/env python3
"""N31 target-scope regression on one real two-identity train case."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SAM3_ROOT = ROOT / "third_party" / "sam3"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

from sam3_intermot.adaptation.correction_state_candidates import (  # noqa: E402
    BOX_RECTANGLE_MASKLET,
    protected_state_signatures,
    rectangle_mask,
    tracker_ids,
    write_target_mask,
)
from scripts.n29_lit_online_replay import (  # noqa: E402
    _image_files,
    _make_backend,
    _read_gt,
    _session,
)
from scripts.n30_multi_identity_write_ablation import (  # noqa: E402
    _bind_official_multi_clean,
    _official_action_history,
)


CHECKPOINT = ROOT / "checkpoints/sam3.1_mirror/sam3.1_multiplex.pt"
DEFAULT_CASES = ROOT / "outputs/n30/multi_identity_case_manifest.json"
TRAIN_ROOT = Path("/path/to/dancetrack/train")


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    raise TypeError(type(value).__name__)


def run(*, case_manifest: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads(case_manifest.read_text(encoding="utf-8"))
    case = manifest["cases"][0]
    sequence = Path(case["sequence_path"])
    gt = _read_gt(sequence)
    images = _image_files(sequence)
    start = int(case["start_frame"])
    correction = int(case["interaction_frame"])
    query_end = min(int(case["query_end"]), len(images) - 1)
    public_ids = [int(value) for value in case["public_ids"]]
    identities = [int(value) for value in case["identity_ids"]]
    boxes = [np.asarray(gt[start][identity], dtype=float) for identity in identities]
    target_public = int(case["correction_public_id"])
    backend = _make_backend(checkpoint)
    diagnostic_state = None
    diagnostic_history = None
    try:
        _session(backend, sequence)
        binding = _bind_official_multi_clean(backend, start, public_ids, boxes)
        _official_action_history(backend, public_ids, start)
        diagnostic_state = backend._predictor._all_inference_states[backend._session_id]["state"]
        diagnostic_history = list(diagnostic_state.get("action_history", []))
        prefix = backend.propagate(start, correction, start_frame_index=start)
        state = backend._predictor._all_inference_states[backend._session_id]["state"]
        before_ids = sorted(tracker_ids(backend))
        before_protected = protected_state_signatures(backend, exclude_public_id=target_public)
        mask = rectangle_mask(
            gt[correction][int(case["correction_dataset_identity"])],
            int(state["orig_height"]),
            int(state["orig_width"]),
            device=state["device"],
        )
        writer = write_target_mask(
            backend,
            frame_idx=correction,
            public_id=target_public,
            mask=mask,
            provenance=BOX_RECTANGLE_MASKLET,
        )
        raw_target = int(backend._ext_to_sam.get(target_public, target_public))
        add_history = getattr(backend._predictor.model, "add_action_history", None)
        if add_history is not None:
            add_history(state, action_type="refine", frame_idx=correction, obj_ids=[raw_target])
        after_ids = sorted(tracker_ids(backend))
        after_protected = protected_state_signatures(backend, exclude_public_id=target_public)
        future = backend.propagate(correction + 1, query_end, start_frame_index=correction + 1)
        protected_equal = before_protected == after_protected
        namespace_equal = before_ids == after_ids and set(before_protected) == set(after_protected)
        result = {
            "protocol": "N31-PROTECTED-IDENTITY-SCOPE",
            "status": "PASS" if writer.get("status") == "WRITTEN" and namespace_equal and protected_equal else "FAIL",
            "case_id": str(case["case_id"]),
            "sequence": str(case["sequence"]),
            "split": str(case["split"]),
            "target_public_id": target_public,
            "protected_public_ids": [value for value in public_ids if value != target_public],
            "binding": binding,
            "prefix_frame_count": len(prefix),
            "future_frame_count": len(future),
            "object_ids_before": before_ids,
            "object_ids_after": after_ids,
            "writer": writer,
            "protected_state_before": before_protected,
            "protected_state_after": after_protected,
            "unaffected_ids_preserved": bool(protected_equal),
            "protected_state_signature_equal": bool(protected_equal),
            "target_namespace_unchanged": bool(namespace_equal),
            "note": "The official reconditioning path is invoked for one raw target ID; protected comparison is the tracker-state/control namespace, while official non-overlap may alter same-frame raster suppression when masks overlap.",
            "val25_read": False,
            "test_labels_used": False,
            "future_gt_used_for_selection": False,
        }
    except Exception as exc:
        result = {
            "protocol": "N31-PROTECTED-IDENTITY-SCOPE",
            "status": "FAIL",
            "failure": f"{type(exc).__name__}: {exc}",
            "failure_traceback": traceback.format_exc(limit=24),
            "action_history_before_prefix": diagnostic_history,
            "state_keys_before_prefix": sorted(str(key) for key in diagnostic_state.keys()) if isinstance(diagnostic_state, dict) else None,
            "case_id": str(case.get("case_id")),
            "val25_read": False,
            "test_labels_used": False,
            "future_gt_used_for_selection": False,
        }
    finally:
        backend.close()
    _write(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/n31/protected_identity_scope.json")
    args = parser.parse_args()
    result = run(case_manifest=args.case_manifest, checkpoint=args.checkpoint, output=args.output)
    print(json.dumps({key: result.get(key) for key in ("protocol", "status", "case_id", "unaffected_ids_preserved")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
