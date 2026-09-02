"""Runtime-facing persistent identity authority for N72R3.

The former N72R2 ``ActiveTrackAuthority`` created a second TrackManager and
registered every candidate before association.  N72R3 keeps the name as a
small compatibility adapter, but it can only operate on an already-created
``SequencePersistentIdentityRuntime``.  Candidate-first registration is
rejected explicitly.
"""

from __future__ import annotations

from typing import Any, Optional

from sam3_intermot.identity.persistent_runtime import (
    PersistentIdentityRecord,
    SequencePersistentIdentityRuntime,
)


class ActiveTrackAuthority:
    """Compatibility adapter over one sequence-persistent runtime.

    No TrackManager or lineage registry is constructed here.  The outer
    runtime owns them for the complete sequence lifetime.
    """

    def __init__(self, runtime: SequencePersistentIdentityRuntime) -> None:
        if not isinstance(runtime, SequencePersistentIdentityRuntime):
            raise TypeError("ActiveTrackAuthority requires a persistent runtime")
        self.runtime = runtime
        self.manager = runtime.manager
        self.lineages = runtime.lineages

    def register(self, frame_idx: int, observation: Any):
        """Reject the retired candidate-first authority path."""

        raise RuntimeError(
            "candidate-first authority is retired; associate with an existing "
            "persistent identity or make an outer birth decision"
        )

    def bind_existing(
        self,
        identity: int | PersistentIdentityRecord,
        candidate_uid: str,
        observation: Any,
        frame_idx: int,
        *,
        session_id: Optional[str] = None,
        adapter_external_id: Optional[int] = None,
        raw_sam_id: Optional[int] = None,
    ) -> PersistentIdentityRecord:
        return self.runtime.bind_candidate(
            identity,
            candidate_uid,
            observation,
            frame_idx,
            session_id=session_id,
            adapter_external_id=adapter_external_id,
            raw_sam_id=raw_sam_id,
        )

    def audit(self) -> dict[str, Any]:
        audit = self.runtime.audit()
        audit.update(
            {
                "schema_version": "N72R3_ACTIVE_TRACK_AUTHORITY_ADAPTER_V1",
                "candidate_first_authority": False,
                "auxiliary_track_manager_count": 0,
                "public_authority_source": "SequencePersistentIdentityRuntime.public_id",
            }
        )
        return audit


__all__ = [
    "ActiveTrackAuthority",
    "PersistentIdentityRecord",
    "SequencePersistentIdentityRuntime",
]
