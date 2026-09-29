"""N72R17 wrappers around the unchanged N72R16 hard-negative metric logic."""

from __future__ import annotations

from pathlib import Path

from sam3_intermot.identity_probe.metrics import (
    _cosine_scores,
    _spatial_index,
    sample_records as _sample_records,
    summarize_records as _summarize_records,
    write_records,
)


def sample_records(document: dict[str, object], store, spatial_mode: str = "nearest_center") -> list[dict[str, object]]:
    """Use the exact N72R16 positive/hard-negative definition."""

    return _sample_records(document, store, spatial_mode=spatial_mode)


def summarize_records(records: list[dict[str, object]], bootstrap_reps: int = 2000, seed: int = 7216) -> dict[str, object]:
    summary = _summarize_records(records, bootstrap_reps=bootstrap_reps, seed=seed)
    summary["metric_version"] = "N72R17-hard-negative-metrics-v1-inherited-N72R16"
    summary["stage"] = "N72R17"
    return summary


__all__ = ["_cosine_scores", "_spatial_index", "sample_records", "summarize_records", "write_records"]
