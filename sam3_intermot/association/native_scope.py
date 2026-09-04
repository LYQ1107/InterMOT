"""Session-scoped native identity compatibility helpers.

Native SAM/adapter IDs are only meaningful inside the session (or stream
scope) that produced them.  Public identity is owned by the outer runtime.
The helpers in this module make that boundary explicit while retaining a
legacy-compatible fallback for old observations that predate scope metadata.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


def _value(obj: Any, *names: str) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return None
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def native_id_of(obj: Any) -> Optional[int]:
    """Return the native/adapter ID without treating it as a public ID."""

    value = _value(obj, "native_tid", "last_native_tid", "sam_object_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def native_scope_of(obj: Any) -> Optional[str]:
    """Read the canonical scope field used for native-ID compatibility."""

    value = _value(
        obj,
        "native_scope",
        "native_tid_scope",
        "last_native_scope",
        "target_session_scope",
        "session_id",
    )
    if value is None or value == "":
        return None
    return str(value)


def scopes_compatible(left: Any, right: Any) -> bool:
    """Return whether two native IDs may be compared.

    A missing scope means the observation/state is legacy metadata.  It is
    intentionally accepted for backward compatibility, but once both sides
    carry a scope it must match exactly.
    """

    left_scope = native_scope_of(left)
    right_scope = native_scope_of(right)
    return left_scope is None or right_scope is None or left_scope == right_scope


def native_same(state: Any, observation: Any) -> bool:
    """Check native ID equality with an explicit session-scope boundary."""

    state_id = native_id_of(state)
    observation_id = native_id_of(observation)
    if state_id is None or observation_id is None or state_id != observation_id:
        return False
    return scopes_compatible(state, observation)


__all__ = [
    "native_id_of",
    "native_scope_of",
    "native_same",
    "scopes_compatible",
]
