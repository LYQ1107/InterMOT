"""Window-boundary snapshot contract for the persistent identity runtime."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from sam3_intermot.identity.persistent_runtime import SequencePersistentIdentityRuntime


@dataclass(frozen=True)
class PersistentRuntimeSnapshot:
    sequence: str
    snapshot_frame: int
    next_window_start: int
    active_session_id: str | None
    payload: dict[str, Any]

    @classmethod
    def capture(
        cls,
        runtime: SequencePersistentIdentityRuntime,
        *,
        snapshot_frame: int,
        next_window_start: int,
    ) -> "PersistentRuntimeSnapshot":
        expected = int(next_window_start) - 1
        if int(snapshot_frame) != expected:
            raise ValueError(
                "persistent snapshot must be captured at window_B.frame_start - 1: "
                f"got {snapshot_frame}, expected {expected}"
            )
        return cls(
            sequence=runtime.sequence,
            snapshot_frame=int(snapshot_frame),
            next_window_start=int(next_window_start),
            active_session_id=runtime.active_session_id,
            payload=deepcopy(runtime.snapshot()),
        )

    def restore_into(self, runtime: SequencePersistentIdentityRuntime) -> None:
        if runtime.sequence != self.sequence:
            raise ValueError("snapshot sequence does not match runtime sequence")
        runtime.restore(deepcopy(self.payload))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "N72R3_PERSISTENT_RUNTIME_SNAPSHOT_V1",
            "sequence": self.sequence,
            "snapshot_frame": self.snapshot_frame,
            "next_window_start": self.next_window_start,
            "active_session_id": self.active_session_id,
            "payload_metadata": {
                "identity_count": len(self.payload.get("identities", [])),
                "public_ids": [
                    int(item["public_id"]) for item in self.payload.get("identities", [])
                ],
                "track_manager_present": "track_manager" in self.payload,
                "lineages_present": "lineages" in self.payload,
                "future_state_from_after_window_start": False,
            },
        }


__all__ = ["PersistentRuntimeSnapshot"]
