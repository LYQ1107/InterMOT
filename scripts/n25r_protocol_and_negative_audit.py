#!/usr/bin/env python
"""Reconstruct N25 train/cal protocol provenance and canonicalize N21 ledgers.

This is an audit/repair-input builder.  It never overwrites N20/N25 artifacts and
does not use GT for a feature or decision score.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(".")
N20 = ROOT / "outputs/n20"
N25 = ROOT / "outputs/n25/dataset"
OUT = ROOT / "outputs/n25r"
PROTOCOL_OUT = OUT / "protocol_audit"
NEG_OUT = OUT / "human_negative_ledger"
DATA_ROOT = Path("/path/to/dancetrack")

SPLITS = {
    "train30": {
        "attempts": N20 / "dataset_attempts_train30.csv",
        "cache": N20 / "full_shadow_cache_train30",
        "episodes": N25 / "episodes_train30.jsonl",
        "nshards": 2,
    },
    "cal10": {
        "attempts": N20 / "dataset_attempts_cal10.csv",
        "cache": N20 / "full_shadow_cache_cal10",
        "episodes": N25 / "episodes_cal10.jsonl",
        "nshards": 4,
    },
}

LEDGERS = {
    "cal10_n21": ("cal10", ROOT / "outputs/n21/human_supervision_ledger_cal10.jsonl"),
    "cal10_phase2": ("cal10", ROOT / "outputs/n21/phase2_human_supervision_ledger_cal10.jsonl"),
    "train30_direct": ("train30", ROOT / "outputs/n21/correction_events_train30.jsonl"),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        if keys:
            writer.writeheader()
            writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_key(row: dict[str, Any], frame_name: str = "frame") -> tuple[str, int, int]:
    return str(row["sequence"]), int(row[frame_name]), int(row["gid"])


def iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def density_at(seq: str, frame: int, cache: dict[str, tuple[np.ndarray, np.ndarray]]) -> int | None:
    if seq not in cache:
        path = ROOT / "outputs/n18/route_c/gfn_cache" / f"{seq}.npz"
        if not path.is_file():
            return None
        with np.load(path) as z:
            cache[seq] = (z["frames"].astype(np.int64), z["offsets"].astype(np.int64))
    frames, offsets = cache[seq]
    pos = int(np.searchsorted(frames, frame))
    if pos >= len(frames) or int(frames[pos]) != frame:
        return None
    lo = int(offsets[pos - 1]) if pos else 0
    return int(offsets[pos]) - lo


def score_metrics(groups: dict[tuple[str, int, int], list[dict[str, Any]]], horizon: int) -> dict[str, Any]:
    present = 0
    top1 = 0
    top3 = 0
    pair_wins = 0.0
    pair_total = 0
    margins: list[float] = []
    missing_score_groups = 0
    for members in groups.values():
        positives_all = [row for row in members if bool(row["positive"])]
        if not positives_all:
            continue
        valid = [row for row in members if row["scores"]["B2_GFN_R0"].get(str(horizon)) is not None]
        positives = [row for row in valid if bool(row["positive"])]
        if not positives:
            missing_score_groups += 1
            continue
        present += 1
        ordered = sorted(
            valid,
            key=lambda row: (-float(row["scores"]["B2_GFN_R0"][str(horizon)]), int(row["candidate_rank"])),
        )
        first_positive = next(i + 1 for i, row in enumerate(ordered) if bool(row["positive"]))
        top1 += int(first_positive == 1)
        top3 += int(first_positive <= 3)
        ps = [float(row["scores"]["B2_GFN_R0"][str(horizon)]) for row in positives]
        ns = [float(row["scores"]["B2_GFN_R0"][str(horizon)]) for row in valid if not bool(row["positive"])]
        if ns:
            margins.append(max(ps) - max(ns))
        for p in ps:
            for n in ns:
                pair_total += 1
                pair_wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return {
        "evaluated_present_groups": present,
        "missing_positive_score_groups": missing_score_groups,
        "top1": top1 / present if present else None,
        "top3": top3 / present if present else None,
        "pair_auc": pair_wins / pair_total if pair_total else None,
        "hardest_negative_margin": float(np.mean(margins)) if margins else None,
    }


def audit_split(split: str, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[tuple[str, int, int], list[dict[str, Any]]]]:
    attempts_all = list(csv.DictReader(cfg["attempts"].open(newline="", encoding="utf-8")))
    attempts = [row for row in attempts_all if row.get("target_present") == "1"]
    attempts.sort(key=lambda row: (row["sequence"], int(row["frame"]), int(row["gid"])))
    attempt_keys = [event_key(row) for row in attempts]
    attempt_index = {key: i for i, key in enumerate(attempt_keys)}

    cache_rows: list[dict[str, Any]] = []
    cache_file_rows: dict[str, list[dict[str, Any]]] = {}
    cache_paths = sorted(Path(cfg["cache"]).glob("*.jsonl"))
    for path in cache_paths:
        rows = load_jsonl(path)
        cache_file_rows[path.name] = rows
        cache_rows.extend(rows)
    cache_groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in cache_rows:
        cache_groups[event_key(row)].append(row)

    episodes = load_jsonl(cfg["episodes"])
    episode_groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        episode_groups[(str(row["sequence"]), int(row["decision_frame"]), int(row["gid"]))].append(row)

    missing_keys = sorted(set(attempt_keys) - set(cache_groups))
    extra_keys = sorted(set(cache_groups) - set(attempt_keys))
    membership_rows: list[dict[str, Any]] = []
    cache_key_to_file: dict[tuple[str, int, int], str] = {}
    for name, rows in cache_file_rows.items():
        for row in rows:
            cache_key_to_file.setdefault(event_key(row), name)
    for index, key in enumerate(attempt_keys):
        expected_shard = index % int(cfg["nshards"])
        actual_file = cache_key_to_file.get(key)
        actual_shard = None
        if actual_file and "_s" in actual_file:
            actual_shard = int(actual_file.rsplit("_s", 1)[1].split(".", 1)[0])
        membership_rows.append(
            {
                "split": split,
                "sequence": key[0],
                "decision_frame": key[1],
                "gid": key[2],
                "sorted_attempt_index": index,
                "expected_shard": expected_shard,
                "actual_shard": actual_shard,
                "cache_present": int(key in cache_groups),
                "shard_consistent": int(actual_shard == expected_shard) if actual_shard is not None else 0,
            }
        )

    rank_keys = [(event_key(row), int(row["candidate_rank"])) for row in cache_rows]
    exact_duplicate_rows = len(rank_keys) - len(set(rank_keys))
    duplicate_start_boxes = 0
    near_duplicate_pairs = 0
    inconsistent_label_groups = 0
    per_group_rows: list[dict[str, Any]] = []
    density_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, rows in sorted(cache_groups.items()):
        ranks = [int(row["candidate_rank"]) for row in rows]
        boxes = [tuple(round(float(x), 4) for x in row["start_box"]) for row in rows]
        duplicate_start_boxes += len(boxes) - len(set(boxes))
        near_duplicate_pairs += sum(iou(rows[i]["start_box"], rows[j]["start_box"]) >= 0.95 for i in range(len(rows)) for j in range(i + 1, len(rows)))
        positives = [int(row["candidate_rank"]) for row in rows if int(row.get("is_correct", 0))]
        inconsistent_label_groups += int(len(positives) > 1)
        ep = episode_groups.get(key, [])
        query_frames = {int(row["correction_frame"]) for row in ep}
        shadow_frames = [frame for row in rows for frame in row.get("frames", [])]
        valid_shadow = sum(frame.get("box") is not None for frame in shadow_frames)
        expected_shadow = len(shadow_frames)
        observed_per_candidate = [sum(frame.get("box") is not None for frame in row.get("frames", [])) for row in rows]
        decision_jpeg = DATA_ROOT / "train" / key[0] / "img1" / f"{key[1] + 1:08d}.jpg"
        start_frame_consistent = all(not row.get("frames") or int(row["frames"][0]["frame"]) == key[1] for row in rows)
        per_group_rows.append(
            {
                "split": split,
                "sequence": key[0],
                "decision_frame": key[1],
                "gid": key[2],
                "candidate_count": len(rows),
                "candidate_ranks": ";".join(map(str, sorted(ranks))),
                "positive_ranks": ";".join(map(str, sorted(positives))),
                "candidate_set_positive_present": int(bool(positives)),
                "three_state_label": "VISIBLE_AND_CANDIDATE_PRESENT" if positives else "CANDIDATE_SET_POSITIVE_ABSENT",
                "query_frame": next(iter(query_frames)) if len(query_frames) == 1 else None,
                "query_age": key[1] - next(iter(query_frames)) if len(query_frames) == 1 else None,
                "query_frame_consistent": int(len(query_frames) == 1),
                "decision_first_shadow_frame_consistent": int(start_frame_consistent),
                "decision_jpeg_f_plus_1_exists": int(decision_jpeg.is_file()),
                "gfn_frame_density": density_at(key[0], key[1], density_cache),
                "shadow_observed_mean": float(np.mean(observed_per_candidate)) if observed_per_candidate else 0.0,
                "shadow_observed_min": min(observed_per_candidate) if observed_per_candidate else 0,
                "shadow_lost_fraction": 1.0 - valid_shadow / expected_shadow if expected_shadow else None,
            }
        )

    per_sequence: list[dict[str, Any]] = []
    by_seq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_group_rows:
        by_seq[str(row["sequence"])].append(row)
    episode_by_seq: dict[str, dict[tuple[str, int, int], list[dict[str, Any]]]] = defaultdict(dict)
    for key, rows in episode_groups.items():
        episode_by_seq[key[0]][key] = rows
    for seq, rows in sorted(by_seq.items()):
        seq_groups = episode_by_seq.get(seq, {})
        base = {
            "split": split,
            "sequence": seq,
            "groups": len(rows),
            "candidate_availability": float(np.mean([row["candidate_set_positive_present"] for row in rows])),
            "candidate_set_positive_absent_fraction": 1.0 - float(np.mean([row["candidate_set_positive_present"] for row in rows])),
            "mean_candidates": float(np.mean([row["candidate_count"] for row in rows])),
            "mean_query_age": float(np.mean([row["query_age"] for row in rows if row["query_age"] is not None])),
            "mean_density": float(np.mean([row["gfn_frame_density"] for row in rows if row["gfn_frame_density"] is not None])),
            "mean_shadow_lost_fraction": float(np.mean([row["shadow_lost_fraction"] for row in rows if row["shadow_lost_fraction"] is not None])),
            "mean_observed_frames": float(np.mean([row["shadow_observed_mean"] for row in rows])),
        }
        for horizon in (1, 5, 10):
            for name, value in score_metrics(seq_groups, horizon).items():
                base[f"B2_H{horizon}_{name}"] = value
        per_sequence.append(base)

    shard_summary = []
    for shard in range(int(cfg["nshards"])):
        expected = {key for i, key in enumerate(attempt_keys) if i % int(cfg["nshards"]) == shard}
        present = {key for key, name in cache_key_to_file.items() if name.rsplit("_s", 1)[-1].startswith(str(shard))}
        shard_summary.append(
            {
                "shard": shard,
                "expected_attempts": len(expected),
                "present_attempts": len(expected & present),
                "missing_attempts": len(expected - present),
            }
        )

    positive_rank_counts = Counter(
        rank
        for rows in cache_groups.values()
        for rank in [next((int(row["candidate_rank"]) for row in rows if int(row.get("is_correct", 0))), None)]
        if rank is not None
    )
    summary = {
        "split": split,
        "upstream_attempt_rows": len(attempts_all),
        "upstream_target_present_rows": len(attempts),
        "upstream_filter": "target_present == 1",
        "honest_absent_label": "CANDIDATE_SET_POSITIVE_ABSENT",
        "cache_rows": len(cache_rows),
        "cache_groups": len(cache_groups),
        "episode_rows": len(episodes),
        "episode_groups": len(episode_groups),
        "candidate_availability": sum(any(int(row.get("is_correct", 0)) for row in rows) for rows in cache_groups.values()) / max(1, len(cache_groups)),
        "candidate_set_positive_absent_groups": sum(not any(int(row.get("is_correct", 0)) for row in rows) for rows in cache_groups.values()),
        "candidate_count_distribution": dict(sorted(Counter(len(rows) for rows in cache_groups.values()).items())),
        "positive_rank_distribution": dict(sorted(positive_rank_counts.items())),
        "missing_attempts": len(missing_keys),
        "extra_attempts": len(extra_keys),
        "exact_duplicate_candidate_keys": exact_duplicate_rows,
        "exact_duplicate_start_boxes": duplicate_start_boxes,
        "near_duplicate_start_box_pairs_iou_ge_0_95": near_duplicate_pairs,
        "groups_with_multiple_positive_labels": inconsistent_label_groups,
        "frame_alignment": {
            "decision_first_shadow_frame_consistent_groups": sum(row["decision_first_shadow_frame_consistent"] for row in per_group_rows),
            "decision_jpeg_f_plus_1_exists_groups": sum(row["decision_jpeg_f_plus_1_exists"] for row in per_group_rows),
            "query_frame_consistent_groups": sum(row["query_frame_consistent"] for row in per_group_rows),
            "groups": len(per_group_rows),
        },
        "shards": shard_summary,
        "B2": {f"H{h}": score_metrics(episode_groups, h) for h in (1, 5, 10)},
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                "sha256": sha256(path),
            }
            for path in [cfg["attempts"], cfg["episodes"], *cache_paths]
        ],
    }
    missing_rows = [
        {
            "split": split,
            "sequence": key[0],
            "decision_frame": key[1],
            "gid": key[2],
            "sorted_attempt_index": attempt_index[key],
            "expected_shard": attempt_index[key] % int(cfg["nshards"]),
        }
        for key in missing_keys
    ]
    write_csv(PROTOCOL_OUT / f"group_audit_{split}.csv", per_group_rows)
    write_csv(PROTOCOL_OUT / f"membership_{split}.csv", membership_rows)
    write_csv(PROTOCOL_OUT / f"missing_attempts_{split}.csv", missing_rows)
    return summary, per_sequence, episode_groups


def parse_ledger_attempt(row: dict[str, Any]) -> tuple[str, int, int] | None:
    extra = row.get("extra") or {}
    value = extra.get("attempt") or extra.get("att")
    if not value:
        return None
    fields = str(value).split(":")
    if len(fields) != 3:
        return None
    return fields[0], int(fields[1]), int(fields[2])


def audit_ledgers(episode_groups_by_split: dict[str, dict[tuple[str, int, int], list[dict[str, Any]]]]) -> dict[str, Any]:
    canonical_occurrences: list[dict[str, Any]] = []
    source_summary: dict[str, Any] = {}
    for source_name, (split, path) in LEDGERS.items():
        rows = load_jsonl(path)
        row_matched = 0
        explicit_occurrences = 0
        matched_explicit = 0
        rank_present = 0
        positive_occurrences = 0
        positive_consistent = 0
        recovered_public = 0
        unique_groups: set[tuple[str, int, int]] = set()
        unique_negative_keys: set[tuple[str, int, int, int]] = set()
        for row_index, row in enumerate(rows):
            key = parse_ledger_attempt(row)
            group = episode_groups_by_split[split].get(key) if key is not None else None
            if group is not None:
                row_matched += 1
                unique_groups.add(key)
            positive = row.get("positive") or {}
            positive_rank = int(positive["candidate_rank"]) if positive.get("candidate_rank") is not None else None
            if positive_rank is not None:
                positive_occurrences += 1
                if group is not None and any(int(candidate["candidate_rank"]) == positive_rank and bool(candidate["positive"]) for candidate in group):
                    positive_consistent += 1
            public_id = int(row.get("public_id", -1))
            if public_id == -1 and group is not None:
                pids = {candidate.get("public_identity_id") for candidate in group if candidate.get("public_identity_id") is not None}
                if len(pids) == 1:
                    public_id = int(next(iter(pids)))
                    recovered_public += 1
            for negative in row.get("explicit_negatives") or []:
                explicit_occurrences += 1
                negative_rank = int(negative["candidate_rank"])
                rank_exists = bool(group is not None and any(int(candidate["candidate_rank"]) == negative_rank for candidate in group))
                negative_is_posthoc_wrong = bool(group is not None and any(int(candidate["candidate_rank"]) == negative_rank and not bool(candidate["positive"]) for candidate in group))
                if group is not None:
                    matched_explicit += 1
                rank_present += int(rank_exists)
                if key is not None:
                    unique_negative_keys.add((*key, negative_rank))
                canonical_occurrences.append(
                    {
                        "split": split,
                        "sequence": key[0] if key else str(row.get("sequence")),
                        "decision_frame": key[1] if key else int(row.get("frame", -1)),
                        "gid": key[2] if key else None,
                        "public_identity_id": public_id if public_id != -1 else None,
                        "candidate_rank": negative_rank,
                        "positive_candidate_rank": positive_rank,
                        "correction_type": row.get("correction_type"),
                        "memory_role": "HUMAN_EXPLICIT_NEGATIVE",
                        "source_ledger": source_name,
                        "source_row_index": row_index,
                        "source": row.get("source"),
                        "provenance": row.get("provenance"),
                        "gt_used": bool(row.get("gt_used", False)),
                        "event_matched": group is not None,
                        "candidate_rank_exists": rank_exists,
                        "negative_consistent_with_n25_label": negative_is_posthoc_wrong,
                        "positive_consistent_with_n25_label": positive_rank is None or bool(group is not None and any(int(candidate["candidate_rank"]) == positive_rank and bool(candidate["positive"]) for candidate in group)),
                    }
                )
        source_summary[source_name] = {
            "split": split,
            "path": str(path.relative_to(ROOT)),
            "rows": len(rows),
            "correction_types": dict(Counter(str(row.get("correction_type")) for row in rows)),
            "matched_rows": row_matched,
            "unique_matched_event_groups": len(unique_groups),
            "explicit_negative_occurrences": explicit_occurrences,
            "matched_explicit_negative_occurrences": matched_explicit,
            "unique_matched_negative_keys": len(unique_negative_keys),
            "rank_existing_occurrences": rank_present,
            "positive_occurrences": positive_occurrences,
            "positive_label_consistent_occurrences": positive_consistent,
            "public_id_minus_one_rows": sum(int(row.get("public_id", -1)) == -1 for row in rows),
            "public_id_safely_recovered_rows": recovered_public,
            "sha256": sha256(path),
        }

    by_key: dict[tuple[str, str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in canonical_occurrences:
        if row["gid"] is not None:
            by_key[(row["split"], row["sequence"], int(row["decision_frame"]), int(row["gid"]), int(row["candidate_rank"]))].append(row)

    canonical: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for key, rows in sorted(by_key.items()):
        positive_ranks = sorted({int(row["positive_candidate_rank"]) for row in rows if row["positive_candidate_rank"] is not None})
        correction_types = sorted({str(row["correction_type"]) for row in rows})
        public_ids = sorted({int(row["public_identity_id"]) for row in rows if row["public_identity_id"] is not None})
        conflict_reasons = []
        if len(positive_ranks) > 1:
            conflict_reasons.append("POSITIVE_RANK_DISAGREEMENT")
        if len(public_ids) > 1:
            conflict_reasons.append("PUBLIC_ID_DISAGREEMENT")
        if not all(bool(row["candidate_rank_exists"]) for row in rows):
            conflict_reasons.append("NEGATIVE_RANK_MISSING")
        if not all(bool(row["negative_consistent_with_n25_label"]) for row in rows):
            conflict_reasons.append("NEGATIVE_LABEL_DISAGREEMENT")
        if not all(bool(row["positive_consistent_with_n25_label"]) for row in rows):
            conflict_reasons.append("POSITIVE_LABEL_DISAGREEMENT")
        item = {
            "split": key[0],
            "sequence": key[1],
            "decision_frame": key[2],
            "gid": key[3],
            "candidate_rank": key[4],
            "public_identity_id": public_ids[0] if len(public_ids) == 1 else None,
            "positive_candidate_rank": positive_ranks[0] if len(positive_ranks) == 1 else None,
            "correction_types": correction_types,
            "memory_role": "HUMAN_EXPLICIT_NEGATIVE",
            "occurrences": len(rows),
            "source_ledgers": sorted({row["source_ledger"] for row in rows}),
            "admissible": not conflict_reasons,
            "conflict_reasons": conflict_reasons,
        }
        if len(rows) > 1:
            duplicates.append(item)
        if conflict_reasons:
            conflicts.append(item)
        else:
            canonical.append(item)

    for path, rows in [
        (NEG_OUT / "canonical_human_explicit_negatives.jsonl", canonical),
        (NEG_OUT / "duplicate_negative_keys.jsonl", duplicates),
        (NEG_OUT / "conflicting_negative_keys.jsonl", conflicts),
        (NEG_OUT / "all_negative_occurrences.jsonl", canonical_occurrences),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "sources": source_summary,
        "occurrences": len(canonical_occurrences),
        "canonical_key_count_before_conflict_filter": len(by_key),
        "admissible_unique_human_negative_keys": len(canonical),
        "duplicate_key_count": len(duplicates),
        "conflicting_key_count": len(conflicts),
        "maximum_occurrences_for_one_key": max((len(rows) for rows in by_key.values()), default=0),
        "admissible_by_split": dict(Counter(row["split"] for row in canonical)),
        "policy": "Only deduplicated, event/rank-matched, identity-specific HUMAN_EXPLICIT_NEGATIVE records without label conflicts are admissible for B10.",
    }
    dump_json(NEG_OUT / "negative_ledger_summary.json", summary)
    return summary


def main() -> None:
    PROTOCOL_OUT.mkdir(parents=True, exist_ok=True)
    NEG_OUT.mkdir(parents=True, exist_ok=True)
    split_summaries: dict[str, Any] = {}
    all_per_sequence: list[dict[str, Any]] = []
    episode_groups: dict[str, dict[tuple[str, int, int], list[dict[str, Any]]]] = {}
    for split, cfg in SPLITS.items():
        summary, per_sequence, groups = audit_split(split, cfg)
        split_summaries[split] = summary
        all_per_sequence.extend(per_sequence)
        episode_groups[split] = groups
    protocol_summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator_script": {
            "path": "scripts/run_n20_all_candidate_shadow.py",
            "sha256_current": sha256(ROOT / "scripts/run_n20_all_candidate_shadow.py"),
        },
        "dataset_builder": {
            "path": "scripts/build_n25_dataset_and_gate.py",
            "sha256_current": sha256(ROOT / "scripts/build_n25_dataset_and_gate.py"),
        },
        "splits": split_summaries,
        "conclusion": {
            "protocol_error": "INCOMPLETE_TRAIN_SHARD_0",
            "detail": "train30 sorted shard 0 contains 335/500 attempts; 165 expected target-present attempts are absent after the surviving worker exited with code 143. cal10 contains all 1200 expected attempts.",
            "label_limit": "All upstream attempts were filtered to target_present==1, so unmatched top-5 groups are CANDIDATE_SET_POSITIVE_ABSENT, not proven scene-level NONE.",
            "repair_policy": "Create a new N25R cache, preserve old N20/N25 artifacts, resume only the missing train shard, and recompute the baseline before representation claims.",
        },
    }
    dump_json(PROTOCOL_OUT / "protocol_summary.json", protocol_summary)
    write_csv(PROTOCOL_OUT / "per_sequence.csv", all_per_sequence)
    negative_summary = audit_ledgers(episode_groups)
    dump_json(OUT / "negative_ledger_summary.json", negative_summary)
    print(json.dumps({"protocol": protocol_summary["conclusion"], "negative": negative_summary}, indent=2))
    print("N25R_PROTOCOL_AND_NEGATIVE_AUDIT_DONE")


if __name__ == "__main__":
    main()
