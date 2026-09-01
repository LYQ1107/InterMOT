#!/usr/bin/env python3
"""N29-C causal full-loop gate and candidate-matrix runner.

The existing ``full_loop_v0`` core is the only permitted lifecycle engine.
This entry point accepts a *delivered* candidate tape only when it contains
the complete per-frame target/candidate matrix inputs.  It merges original
and decoder candidates before one Hungarian solve and never assigns a public
ID from a decoder slot.  A short N29-B pilot is not promoted to a full-video
tape, so the default invocation records ``NOT_RUN`` instead of inventing a
tracking stream or TrackEval result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_B = ROOT / "outputs" / "n29" / "n29b_result.json"
DEFAULT_TAPE = ROOT / "outputs" / "n29" / "candidate_tape.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs" / "n29" / "n29c_result.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.association.decoder_candidate_bridge import (  # noqa: E402
    DecoderCandidate,
    build_decoder_assignment,
    merge_decoder_candidates,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)}")


def _contains_blind(value: Any) -> bool:
    text = json.dumps(value, sort_keys=True, default=str).lower()
    return any(token in text for token in ("val25", "validation", "/val/", "test/"))


def _candidate(payload: Mapping[str, Any], frame_idx: int) -> DecoderCandidate:
    required = ("mask_logits", "mask", "box_xyxy", "presence", "iou_pred")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"candidate at frame {frame_idx} missing {missing}")
    mask_logits = np.asarray(payload["mask_logits"], dtype=np.float32)
    mask = np.asarray(payload["mask"], dtype=bool)
    if mask_logits.ndim != 2 or mask.ndim != 2:
        raise ValueError("candidate masks must be two-dimensional")
    return DecoderCandidate(
        frame_idx=int(frame_idx),
        mask_logits=mask_logits,
        mask=mask,
        box_xyxy=tuple(float(value) for value in payload["box_xyxy"]),
        presence=float(payload["presence"]),
        iou_pred=float(payload["iou_pred"]),
        decoder_token=(
            None
            if payload.get("decoder_token") is None
            else np.asarray(payload["decoder_token"], dtype=np.float32)
        ),
        clip_feature=(
            None
            if payload.get("clip_feature") is None
            else np.asarray(payload["clip_feature"], dtype=np.float32)
        ),
        source=str(payload.get("source", "sam3_lora_singleton")),
        source_public_id=payload.get("source_public_id"),
        adapter_version=int(payload.get("adapter_version", 0)),
        valid=bool(payload.get("valid", True)),
        reject_reason=payload.get("reject_reason"),
    )


def _load_records(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        root = json.loads(path.read_text(encoding="utf-8"))
        records = root.get("frames", []) if isinstance(root, Mapping) else root
    if not isinstance(records, list) or not all(isinstance(row, Mapping) for row in records):
        raise ValueError("candidate tape must be a list of frame objects")
    if _contains_blind(records):
        raise ValueError("blind boundary violation in candidate tape")
    return list(records)


def _run_candidate_bridge(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for row in records:
        frame_idx = int(row["frame_idx"])
        identity_ids = list(row["identity_ids"])
        anchor_scores = np.asarray(row["anchor_scores"], dtype=np.float64)
        original = [_candidate(item, frame_idx) for item in row.get("original_candidates", [])]
        decoder = [_candidate(item, frame_idx) for item in row.get("decoder_candidates", [])]
        candidates = merge_decoder_candidates(original, decoder)
        if not candidates and anchor_scores.shape[1] != 0:
            raise ValueError(f"frame {frame_idx}: candidates are empty but scores are not")
        bridge = build_decoder_assignment(
            anchor_scores,
            candidates,
            identity_ids,
            delta_scores=np.asarray(row.get("delta_scores", np.zeros_like(anchor_scores)), dtype=np.float64),
            none_scores=np.asarray(row.get("none_scores", np.zeros(len(identity_ids))), dtype=np.float64),
            candidate_mask=(
                None
                if row.get("candidate_mask") is None
                else np.asarray(row["candidate_mask"], dtype=bool)
            ),
        )
        results.append(
            {
                "frame_idx": frame_idx,
                "identity_count": len(identity_ids),
                "candidate_count": len(candidates),
                "matrix_shape": list(bridge.matrix.shape),
                "assignment": bridge.assignment.assignment.tolist(),
                "sources": [candidate.source for candidate in candidates],
                "source_public_ids": [candidate.source_public_id for candidate in candidates],
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-result", type=Path, default=DEFAULT_B)
    parser.add_argument("--candidate-tape", type=Path, default=DEFAULT_TAPE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "protocol": "N29-C",
        "status": "NOT_RUN",
        "val25_read": False,
        "full_loop_engine": "sam3_intermot.evaluation.full_loop_v0",
        "candidate_bridge": "merge_then_one_global_hungarian_with_NONE",
        "trackeval_authorized": False,
    }
    if not args.b_result.is_file():
        result["reason"] = "n29b_result_missing"
    else:
        b_result = json.loads(args.b_result.read_text(encoding="utf-8"))
        result["b_status"] = b_result.get("status")
        result["b_variant"] = b_result.get("variant")
        sequence_results = b_result.get("sequence_results", [])
        b_gate = bool(
            b_result.get("status") == "PASS"
            and sequence_results
            and all(row.get("status") == "PASS" for row in sequence_results)
            and all(row.get("update", {}).get("status") == "COMMIT" for row in sequence_results)
        )
        result["b_mechanism_gate"] = b_gate
        if not b_gate:
            result["reason"] = "n29b_mechanism_gate_not_passed"
        elif not args.candidate_tape.is_file():
            result["reason"] = "no_delivered_full_video_candidate_tape"
            result["detail"] = (
                "N29-B is a bounded six-frame official decoder binding pilot; it is not a full-loop stream. "
                "No candidate tape, public-ID trace, or recovery callbacks were fabricated."
            )
        else:
            try:
                rows = _run_candidate_bridge(_load_records(args.candidate_tape))
                result["candidate_stream_status"] = "PASS"
                result["candidate_frames"] = rows
                result["reason"] = "full_loop_v0_executor_requires_delivered_p0_recovery_and_reactivation_callbacks"
            except Exception as exc:
                result["reason"] = f"candidate_tape_rejected: {type(exc).__name__}: {exc}"
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
