"""R4: the complete exact public assignment, never a target-column merge."""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np

from .public_assignment import solve_exact_public_assignment, validate_exact_public_assignment


def solve_global(rows: Sequence[Mapping[str, Any]], scores: np.ndarray, states: Sequence[Any], *, frame: int, none_score: float = 0.0) -> dict[str, Any]:
    result = solve_exact_public_assignment(
        [{"candidate_uid": str(r["candidate_uid"]), "candidate_index": i} for i, r in enumerate(rows)],
        scores, [int(s.pid) for s in states], [int(s.public_id) for s in states],
        none_score=none_score, source_run_id=f"N72R20R4:{frame}", runtime_future_gt_used=False,
    )
    validate_global(result, rows)
    return result


def validate_global(result: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    errors = validate_exact_public_assignment(result)
    uids = [str(r["candidate_uid"]) for r in result["assignment_rows"]]
    expected = [str(r["candidate_uid"]) for r in rows]
    if uids != expected or len(set(uids)) != len(uids):
        errors.append("candidate_axis_or_uniqueness")
    if errors:
        raise RuntimeError(f"global assignment invariant failure: {errors}")


def public_map(result: Mapping[str, Any]) -> dict[int, str | None]:
    return {int(r["public_id"]): r["candidate_uid"] for r in result["public_assignments"]}
