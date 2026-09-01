#!/usr/bin/env python3
"""Simulate causal train30 human corrections on the frozen B2/H5 stream.

The model decision is finalized before any post-hoc train label is inspected.
Only the candidate explicitly denied by the simulated human becomes
``HUMAN_EXPLICIT_NEGATIVE``.  Unselected wrong candidates remain merely
``UNSELECTED_CANDIDATE`` and are never admitted to B10 negative memory.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(".")
SOURCE = ROOT / "outputs/n25r/repaired_dataset/episodes_train30.jsonl"
OUT = ROOT / "outputs/n25r/human_negative_ledger"
HISTORY = "5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def event_key(row: dict[str, Any]) -> str:
    return f"{row['sequence']}:{int(row['decision_frame'])}:{int(row['gid'])}"


def candidate_key(row: dict[str, Any]) -> str:
    return f"{event_key(row)}:r{int(row['candidate_rank'])}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows(SOURCE)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[event_key(row)].append(row)
    ordered_groups = sorted(
        groups.values(),
        key=lambda members: (
            str(members[0]["sequence"]),
            int(members[0]["decision_frame"]),
            int(members[0]["gid"]),
        ),
    )
    corrections = []
    negatives = []
    positives = []
    provenance = []
    sequence_event_index: Counter[str] = Counter()
    correction_type_counts: Counter[str] = Counter()
    for members in ordered_groups:
        members = sorted(members, key=lambda row: int(row["candidate_rank"]))
        first = members[0]
        sequence = str(first["sequence"])
        sequence_event_index[sequence] += 1
        valid = [row for row in members if row["scores"]["B2_GFN_R0"].get(HISTORY) is not None]
        if not valid:
            raise RuntimeError(f"B2/H5 has no valid candidate for {event_key(first)}")
        # This is the frozen model output.  Labels have not been read here.
        selected = min(
            valid,
            key=lambda row: (
                -float(row["scores"]["B2_GFN_R0"][HISTORY]),
                int(row["candidate_rank"]),
            ),
        )
        selected_correct = bool(selected["positive"])
        positive_candidates = [row for row in members if bool(row["positive"])]
        correction_type = None
        admitted_positive = None
        if not selected_correct:
            correction_type = "ID_WRONG" if positive_candidates else "CANDIDATE_SET_MISS"
            correction_type_counts[correction_type] += 1
            if positive_candidates:
                # If multiple shadow candidates satisfy the post-hoc match,
                # the simulated human admits one deterministic representative.
                admitted_positive = min(
                    positive_candidates,
                    key=lambda row: (
                        -float(row["scores"]["B2_GFN_R0"].get(HISTORY) or -1e9),
                        int(row["candidate_rank"]),
                    ),
                )
            base = {
                "sequence": sequence,
                "sequence_event_index": int(sequence_event_index[sequence]),
                "event_key": event_key(first),
                "decision_frame": int(first["decision_frame"]),
                "correction_frame": int(first["correction_frame"]),
                "gid": int(first["gid"]),
                "public_identity_id": int(first["public_identity_id"]),
                "decision_model": "frozen_B2_GFN_R0_H5_repaired_train30",
                "decision_completed_before_correction": True,
                "correction_type": correction_type,
                "selected_candidate_rank": int(selected["candidate_rank"]),
                "selected_candidate_key": candidate_key(selected),
                "selected_score": float(selected["scores"]["B2_GFN_R0"][HISTORY]),
                "positive_candidate_rank": None if admitted_positive is None else int(admitted_positive["candidate_rank"]),
                "positive_candidate_key": None if admitted_positive is None else candidate_key(admitted_positive),
            }
            corrections.append(base)
            negatives.append(
                {
                    **base,
                    "candidate_rank": int(selected["candidate_rank"]),
                    "candidate_key": candidate_key(selected),
                    "memory_role": "HUMAN_EXPLICIT_NEGATIVE",
                    "pre_correction_role": "MODEL_INDUCED_HARD_NEGATIVE",
                    "admission_time": "after_current_event_output_and_simulated_human_denial",
                    "eligible_for_same_event_scoring": False,
                    "eligible_for_future_same_identity_scoring": True,
                    "identity_scope": f"{sequence}:{int(first['public_identity_id'])}",
                }
            )
            if admitted_positive is not None:
                positives.append(
                    {
                        **base,
                        "candidate_rank": int(admitted_positive["candidate_rank"]),
                        "candidate_key": candidate_key(admitted_positive),
                        "memory_role": "HUMAN_EXPLICIT_POSITIVE",
                        "admission_time": "after_current_event_output_and_simulated_human_correction",
                        "eligible_for_same_event_scoring": False,
                        "eligible_for_future_same_identity_scoring": True,
                        "identity_scope": f"{sequence}:{int(first['public_identity_id'])}",
                    }
                )
        for row in members:
            if candidate_key(row) == candidate_key(selected) and not selected_correct:
                role = "HUMAN_EXPLICIT_NEGATIVE"
            elif admitted_positive is not None and candidate_key(row) == candidate_key(admitted_positive):
                role = "HUMAN_EXPLICIT_POSITIVE"
            elif candidate_key(row) == candidate_key(selected):
                role = "MODEL_SELECTED_CORRECT"
            elif bool(row["positive"]):
                role = "POSTHOC_POSITIVE_NOT_CORRECTED"
            else:
                role = "UNSELECTED_CANDIDATE"
            provenance.append(
                {
                    "sequence": sequence,
                    "event_key": event_key(row),
                    "candidate_key": candidate_key(row),
                    "decision_frame": int(row["decision_frame"]),
                    "gid": int(row["gid"]),
                    "public_identity_id": int(row["public_identity_id"]),
                    "candidate_rank": int(row["candidate_rank"]),
                    "selected_by_frozen_B2": candidate_key(row) == candidate_key(selected),
                    "posthoc_candidate_positive": bool(row["positive"]),
                    "provenance_role_after_event": role,
                    "eligible_for_B10_negative_memory": role == "HUMAN_EXPLICIT_NEGATIVE",
                }
            )

    paths = {
        "corrections": OUT / "simulated_train_b2_h5_corrections.jsonl",
        "explicit_negatives": OUT / "simulated_train_human_explicit_negatives.jsonl",
        "explicit_positives": OUT / "simulated_train_human_explicit_positives.jsonl",
        "candidate_provenance": OUT / "train_candidate_provenance.jsonl",
    }
    write_jsonl(paths["corrections"], corrections)
    write_jsonl(paths["explicit_negatives"], negatives)
    write_jsonl(paths["explicit_positives"], positives)
    write_jsonl(paths["candidate_provenance"], provenance)
    summary = {
        "status": "COMPLETE",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "decision_model": "frozen_B2_GFN_R0_H5_repaired_train30",
        "groups": len(ordered_groups),
        "candidate_rows": len(rows),
        "model_correct_groups": len(ordered_groups) - len(corrections),
        "simulated_human_corrections": len(corrections),
        "correction_types": dict(sorted(correction_type_counts.items())),
        "human_explicit_negative_writes": len(negatives),
        "human_explicit_positive_writes": len(positives),
        "unselected_candidates_never_promoted_to_negative": True,
        "candidate_set_miss_does_not_fabricate_positive": True,
        "current_correction_never_scores_current_event": True,
        "target_specific_memory_key": "sequence + public_identity_id",
        "outputs": {
            name: {"path": str(path.relative_to(ROOT)), "rows": sum(1 for _ in path.open(encoding="utf-8")), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }
    summary_path = OUT / "simulated_train_negative_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
