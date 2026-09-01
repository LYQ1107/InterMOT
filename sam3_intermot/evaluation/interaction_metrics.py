"""Interaction metrics from transaction logs."""

from typing import Dict, List


def summarize_transactions(transaction_log: List[dict]) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    for entry in transaction_log:
        action = entry["action_type"]
        row = summary.setdefault(
            action, {"total": 0, "accepted": 0, "rejected": 0, "rolled_back": 0}
        )
        row["total"] += 1
        if entry.get("rolled_back"):
            row["rolled_back"] += 1
        if entry.get("accepted"):
            row["accepted"] += 1
        else:
            row["rejected"] += 1
    return summary
