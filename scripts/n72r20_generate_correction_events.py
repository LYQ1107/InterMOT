#!/usr/bin/env python3
"""Generate simulated human correction events from a small train smoke.

The SAM3 candidate worker is GT-free.  This script runs only after that
runtime output exists and reads GT offline to discover identity errors.  It
does not change SAM3, OSNet, the frozen N72R18 GRU, or identity memory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam3_intermot.interaction.correction_event import HumanCorrectionEvent
from scripts.n72r20_identity_bridge_eval import (
    _gru_state,
    _top_index,
    attach_features,
    box_iou_xyxy,
    load_anchor_encoder,
    load_feature_matrix,
    load_frozen_gru,
    load_json,
    normalize,
    read_zstd_jsonl,
    validate_candidate_index,
    xyxy_from_tlwh,
)


IOU_THRESHOLD = 0.50
METHOD = "B2_FROZEN_N72R18_GRU"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_protocol_path(protocol: dict[str, Any], split: str) -> Path:
    source = protocol.get("protocol_sources", {}).get(split, {})
    path = source.get("path") if isinstance(source, dict) else None
    if not path:
        raise ValueError(f"missing frozen {split} protocol source")
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"missing frozen protocol: {result}")
    return result


def load_sequence(dataset_root: Path, split: str, sequence: str):
    from sam3_intermot.identity_probe.dataset import discover_sequences

    matches = [item for item in discover_sequences(dataset_root, split) if item.name == sequence]
    if len(matches) != 1:
        raise ValueError(f"expected one {split} sequence named {sequence}, found {len(matches)}")
    return matches[0]


def load_candidates(candidate_root: Path, sequence: str) -> dict[int, list[dict[str, Any]]]:
    candidate_dir = candidate_root / "candidates" / sequence
    metadata_path = candidate_dir / "metadata.jsonl.zst"
    embeddings_path = candidate_dir / "embeddings.f16"
    validate_candidate_index(candidate_dir, metadata_path, embeddings_path)
    rows = read_zstd_jsonl(metadata_path)
    feature_matrix = load_feature_matrix(candidate_dir)
    attach_features(rows, feature_matrix)
    return rows


def _anchor_box(anchor: dict[str, Any]) -> tuple[float, float, float, float]:
    return xyxy_from_tlwh(anchor["tlwh"])


def _target_and_competitors(sequence: Any, frame: int, identity_id: int):
    boxes = tuple(sequence.boxes(frame))
    target = next((box for box in boxes if int(box.track_id) == identity_id), None)
    target_box = None if target is None else xyxy_from_tlwh(target.tlwh)
    competitors = tuple(
        xyxy_from_tlwh(box.tlwh)
        for box in boxes
        if int(box.track_id) != identity_id and int(box.track_id) >= 0
    )
    return target, target_box, competitors


def _candidate_context(rows: list[dict[str, Any]], scores: np.ndarray, selected_index: int | None) -> tuple[dict[str, Any], ...]:
    order = sorted(range(len(rows)), key=lambda index: (-float(scores[index]), index))
    ranks = {index: rank + 1 for rank, index in enumerate(order)}
    context: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        context.append(
            {
                "candidate_id": row.get("candidate_uid"),
                "box_xyxy": [float(value) for value in row["box_xyxy"]],
                "score": float(scores[index]),
                "rank": int(ranks[index]),
                "selected": bool(selected_index == index),
                "confidence": float(row.get("confidence", 0.0)),
                "source": str(row.get("source", "automatic_propagation")),
                "raw_native_id": row.get("raw_native_id"),
                "adapter_external_id": row.get("adapter_external_id"),
            }
        )
    return tuple(context)


def _positive_candidate(
    rows: list[dict[str, Any]],
    target_box: tuple[float, float, float, float] | None,
) -> tuple[int | None, float | None]:
    if target_box is None:
        return None, None
    values = [(index, box_iou_xyxy(row["box_xyxy"], target_box)) for index, row in enumerate(rows)]
    eligible = [(index, iou) for index, iou in values if iou >= IOU_THRESHOLD]
    if not eligible:
        return None, None
    return max(eligible, key=lambda item: (item[1], -item[0]))


def _hard_negative(
    rows: list[dict[str, Any]],
    selected_index: int | None,
    competitors: tuple[tuple[float, float, float, float], ...],
) -> bool:
    if selected_index is None or not competitors:
        return False
    selected_box = rows[selected_index]["box_xyxy"]
    return any(box_iou_xyxy(selected_box, competitor) >= IOU_THRESHOLD for competitor in competitors)


def _error_type(
    *,
    target_visible: bool,
    target_candidate_available: bool,
    selected_index: int | None,
    positive_index: int | None,
    previous_target_candidate_available: bool | None,
) -> str | None:
    if not target_visible:
        return None
    if not target_candidate_available:
        return "missed_target"
    if selected_index is None or selected_index != positive_index:
        if previous_target_candidate_available is False:
            return "wrong_recovery"
        return "identity_switch"
    return None


def _stats(events: list[HumanCorrectionEvent], opportunities: dict[str, int], sequences: list[str]) -> dict[str, Any]:
    by_sequence: dict[str, list[HumanCorrectionEvent]] = defaultdict(list)
    for event in events:
        by_sequence[event.sequence_id].append(event)
    type_counts = Counter(event.error_type for event in events)
    intervals = [event.error_interval_frames for event in events if event.error_interval_frames is not None]
    identity_events = [event for event in events if event.error_type in {"identity_switch", "wrong_recovery"}]
    hard_negative_events = [event for event in events if event.hard_negative]

    def interval_summary(values: list[int]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "median": None, "p10": None, "p90": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": len(values),
            "median": float(np.median(array)),
            "p10": float(np.percentile(array, 10)),
            "p90": float(np.percentile(array, 90)),
        }

    sequence_stats = {}
    for sequence in sequences:
        rows = by_sequence.get(sequence, [])
        counts = Counter(event.error_type for event in rows)
        sequence_intervals = [event.error_interval_frames for event in rows if event.error_interval_frames is not None]
        sequence_stats[sequence] = {
            "correction_opportunities": int(opportunities.get(sequence, 0)),
            "total_error_count": len(rows),
            "identity_switch_count": int(counts["identity_switch"]),
            "missed_identity_count": int(counts["missed_target"]),
            "wrong_recovery_count": int(counts["wrong_recovery"]),
            "hard_negative_count": sum(bool(event.hard_negative) for event in rows),
            "error_rate_per_opportunity": len(rows) / opportunities[sequence] if opportunities.get(sequence) else None,
            "error_interval_frames": interval_summary(sequence_intervals),
        }

    return {
        "total_error_count": len(events),
        "identity_switch_count": int(type_counts["identity_switch"]),
        "missed_identity_count": int(type_counts["missed_target"]),
        "wrong_recovery_count": int(type_counts["wrong_recovery"]),
        "correction_opportunity_count": int(sum(opportunities.values())),
        "correction_opportunity_by_sequence": {key: int(value) for key, value in sorted(opportunities.items())},
        "error_rate_per_opportunity": len(events) / sum(opportunities.values()) if sum(opportunities.values()) else None,
        "hard_negative_count": len(hard_negative_events),
        "hard_negative_proportion": len(hard_negative_events) / len(events) if events else None,
        "identity_error_hard_negative_proportion": (
            sum(bool(event.hard_negative) for event in identity_events) / len(identity_events)
            if identity_events
            else None
        ),
        "error_interval_frames": interval_summary(intervals),
        "by_sequence": sequence_stats,
    }


def generate_sequence_events(
    *,
    sequence_name: str,
    records_dir: Path,
    candidates: dict[int, list[dict[str, Any]]],
    sequence: Any,
    anchors: list[dict[str, Any]],
    encoder: Any,
    gru: Any,
    timestamp: str,
) -> tuple[list[HumanCorrectionEvent], int]:
    record_path = records_dir / f"{sequence_name}.immediate.jsonl"
    if not record_path.is_file():
        raise FileNotFoundError(f"missing train B2 immediate record: {record_path}")
    records = []
    with record_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("method") == METHOD:
                    records.append(row)
    by_anchor: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_anchor[(int(row["anchor_frame"]), int(row["track_id"]))].append(row)
    anchor_by_key = {
        (int(item["anchor_frame"]), int(item["anchor"]["track_id"])): item
        for item in anchors
    }
    if set(by_anchor) != set(anchor_by_key):
        raise ValueError(
            f"record/protocol anchor mismatch for {sequence_name}: "
            f"records={sorted(by_anchor)}, protocol={sorted(anchor_by_key)}"
        )

    events: list[HumanCorrectionEvent] = []
    opportunities = 0
    corrected_crop_cache: dict[tuple[int, int], np.ndarray] = {}
    for anchor_key in sorted(by_anchor):
        anchor_record = anchor_by_key[anchor_key]
        anchor = anchor_record["anchor"]
        state = normalize(
            encoder.encode(sequence.image_path(int(anchor_record["anchor_frame"])), [_anchor_box(anchor)])[0]
        )
        previous_target_candidate_available: bool | None = None
        previous_error_frame: int | None = None
        for row in sorted(by_anchor[anchor_key], key=lambda value: int(value["frame"])):
            frame = int(row["frame"])
            target, target_box, competitors = _target_and_competitors(sequence, frame, int(anchor["track_id"]))
            if target is None:
                previous_target_candidate_available = None
                continue
            opportunities += 1
            candidate_rows = candidates.get(frame - 1, [])
            feature_matrix = np.asarray([item["_feature"] for item in candidate_rows], dtype=np.float32)
            scores = (
                (torch.from_numpy(feature_matrix) @ torch.from_numpy(state)).numpy().astype(np.float32, copy=False)
                if len(candidate_rows)
                else np.empty((0,), dtype=np.float32)
            )
            selected_index = row.get("selected_candidate_index")
            selected_index = None if selected_index is None else int(selected_index)
            if selected_index is not None and selected_index >= len(candidate_rows):
                raise ValueError(f"selected candidate index is out of range at {sequence_name}:{frame}")
            positive_index, _positive_iou = _positive_candidate(candidate_rows, target_box)
            target_candidate_available = positive_index is not None
            error_type = _error_type(
                target_visible=True,
                target_candidate_available=target_candidate_available,
                selected_index=selected_index,
                positive_index=positive_index,
                previous_target_candidate_available=previous_target_candidate_available,
            )
            if error_type is not None:
                if positive_index is not None:
                    corrected_embedding = normalize(candidate_rows[positive_index]["_feature"])
                    corrected_candidate_id = str(candidate_rows[positive_index]["candidate_uid"])
                else:
                    cache_key = (frame, int(anchor["track_id"]))
                    if cache_key not in corrected_crop_cache:
                        corrected_crop_cache[cache_key] = normalize(
                            encoder.encode(sequence.image_path(frame), [target_box])[0]
                        )
                    corrected_embedding = corrected_crop_cache[cache_key]
                    corrected_candidate_id = None
                predicted_candidate_id = (
                    None if selected_index is None else str(candidate_rows[selected_index]["candidate_uid"])
                )
                event = HumanCorrectionEvent(
                    frame_id=frame,
                    sequence_id=sequence_name,
                    identity_id=int(anchor["track_id"]),
                    predicted_candidate_id=predicted_candidate_id,
                    corrected_candidate_id=corrected_candidate_id,
                    predicted_identity_state=tuple(float(value) for value in state),
                    corrected_embedding=tuple(float(value) for value in corrected_embedding),
                    candidate_context=_candidate_context(candidate_rows, scores, selected_index),
                    timestamp=timestamp,
                    error_type=error_type,
                    error_interval_frames=(
                        None if previous_error_frame is None else frame - previous_error_frame
                    ),
                    hard_negative=_hard_negative(candidate_rows, selected_index, competitors),
                )
                events.append(event)
                previous_error_frame = frame
            previous_target_candidate_available = target_candidate_available
            update_index = row.get("update_candidate_index")
            if bool(row.get("update_applied")) and update_index is not None:
                update_index = int(update_index)
                if update_index < 0 or update_index >= len(candidate_rows):
                    raise ValueError(f"update candidate index is out of range at {sequence_name}:{frame}")
                state = _gru_state(gru, state, candidate_rows[update_index]["_feature"])
    return events, opportunities


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(Path(args.asset_manifest).expanduser().resolve())
    protocol = load_json(Path(args.protocol).expanduser().resolve())
    if manifest.get("stage") != "N72R20" or protocol.get("stage") != "N72R20":
        raise ValueError("asset manifest and protocol must both be N72R20")
    if args.split != "train":
        raise ValueError("N72R20 correction-event construction is train-only")
    allowed = [str(item) for item in protocol["train_dev_sequences"]]
    sequences = (
        [item.strip() for item in args.sequences.split(",") if item.strip()]
        if args.sequences
        else allowed
    )
    if not sequences or not set(sequences).issubset(set(allowed)):
        raise ValueError(f"correction events require the frozen small train sequence set: {allowed}")

    dataset_root = Path(args.dataset_root or manifest["DANCETRACK_ROOT"]).expanduser().resolve()
    candidate_root = Path(args.candidate_root).expanduser().resolve()
    records_dir = Path(args.records_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = load_json(source_protocol_path(protocol, "train"))
    source_anchors = source.get("anchors", [])
    osnet_path = Path(
        args.osnet_checkpoint or manifest["frozen_identity_assets"]["osnet_x1_0_market1501"]["path"]
    ).expanduser().resolve()
    gru_path = Path(
        args.gru_checkpoint or manifest["frozen_identity_assets"]["n72r18_gru"]["path"]
    ).expanduser().resolve()
    if not osnet_path.is_file() or not gru_path.is_file():
        raise FileNotFoundError("frozen OSNet and N72R18 GRU checkpoints are required")

    encoder = load_anchor_encoder(osnet_path, args.device)
    gru = load_frozen_gru(gru_path, args.device)
    generated_at = datetime.now(timezone.utc).isoformat()
    all_events: list[HumanCorrectionEvent] = []
    opportunities: dict[str, int] = {}
    sequence_files: dict[str, str] = {}
    for sequence_name in sequences:
        sequence = load_sequence(dataset_root, args.split, sequence_name)
        candidates = load_candidates(candidate_root, sequence_name)
        anchors = [
            item
            for item in source_anchors
            if str(item.get("anchor", {}).get("sequence")) == sequence_name
        ]
        if not anchors:
            raise ValueError(f"no frozen train anchors for {sequence_name}")
        events, opportunity_count = generate_sequence_events(
            sequence_name=sequence_name,
            records_dir=records_dir,
            candidates=candidates,
            sequence=sequence,
            anchors=anchors,
            encoder=encoder,
            gru=gru,
            timestamp=generated_at,
        )
        output_path = output_dir / f"{sequence_name}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        sequence_files[sequence_name] = str(output_path)
        opportunities[sequence_name] = opportunity_count
        all_events.extend(events)

    summary = {
        "stage": "N72R20",
        "final_goal_ref": "outputs/N72R20/FINAL_GOAL.json",
        "goal": "Human Correction Driven Persistent Identity Adaptation",
        "central_question": "When a human corrects a tracking error, can the system learn from this correction and improve future identity tracking?",
        "status": "PASS_N72R20_TRAIN_CORRECTION_EVENTS",
        "split": "train",
        "sequences": sequences,
        "sequence_count": len(sequences),
        "method": METHOD,
        "candidate_stream": "real SAM3 candidate caches generated without GT",
        "error_discovery": "offline GT comparison only",
        "interaction_source": "simulated_from_gt",
        "runtime_future_gt_used": False,
        "runtime_gt_read": False,
        "training_performed": False,
        "memory_updated": False,
        "events": len(all_events),
        "event_files": sequence_files,
        "statistics": _stats(all_events, opportunities, sequences),
        "candidate_identity_matcher": "preserved frozen connector",
        "frozen_n72r18_gru": str(gru_path),
        "frozen_osnet": str(osnet_path),
        "generated_at_utc": generated_at,
    }
    summary_path = output_dir / "correction_event_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "events": len(all_events), "summary": str(summary_path)}, ensure_ascii=False))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--sequences")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/N72R20/correction_events")
    parser.add_argument("--asset-manifest", type=Path, default=ROOT / "outputs/N72R20/asset_manifest.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "outputs/N72R20/protocol.json")
    parser.add_argument("--osnet-checkpoint", type=Path)
    parser.add_argument("--gru-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"N72R20 correction-event generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
