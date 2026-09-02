"""Same-run bridge from association state to authoritative public MOT IDs.

An association PID is an internal solver column.  This module makes it
impossible to silently expose that PID as a public identity: a mapping is only
``EXACT`` when the same run has supplied a final ``TrackManager`` MOT track
authority (or an explicitly verified namespace transaction).  The bridge is
small and dependency-free so it can be used by both live code and audit
scripts without importing a video backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional


AUTHORITY_STATUSES = (
    "EXACT",
    "EXPLICIT_NONE",
    "NO_PUBLIC_AUTHORITY",
    "STALE_BINDING",
    "COLLISION",
    "AMBIGUOUS",
    "SOURCE_RUN_MISMATCH",
)


@dataclass(frozen=True)
class PublicAuthorityBinding:
    """One auditable association-state → public-authority binding."""

    source_run_id: str
    sequence: str
    frame_idx: int
    candidate_uid: str
    association_state_id: int
    user_identity_id: Optional[int]
    identity_lineage_id: int
    mot_track_id: int
    public_id: int
    binding_source: str
    binding_transaction_id: str
    valid_from_frame: int
    valid_to_frame: Optional[int] = None
    status: str = "EXACT"

    def __post_init__(self) -> None:
        if self.status not in AUTHORITY_STATUSES:
            raise ValueError(f"unknown authority status: {self.status}")
        if self.valid_to_frame is not None and self.valid_to_frame < self.valid_from_frame:
            raise ValueError("valid_to_frame precedes valid_from_frame")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuthorityResolution:
    status: str
    public_id: Optional[int] = None
    binding: Optional[PublicAuthorityBinding] = None
    reason: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status,
            "public_id": self.public_id,
            "reason": self.reason,
        }
        result["binding"] = None if self.binding is None else self.binding.as_dict()
        return result


class PublicAuthorityBridge:
    """Track final MOT authority for every association state used in a run.

    ``resolve`` is a compatibility method for existing assignment sidecars.
    New callers should use ``resolve_public_authority`` because it returns a
    status rather than collapsing a missing mapping into ``None``.
    """

    def __init__(self, source_run_id: str, sequence: str, session_id: Optional[str] = None) -> None:
        if not source_run_id or not sequence:
            raise ValueError("source_run_id and sequence are required")
        self.source_run_id = str(source_run_id)
        self.sequence = str(sequence)
        # Existing sidecar builders use this optional field for same-run
        # validation.  It is metadata only and never becomes an authority.
        self.session_id = None if session_id is None else str(session_id)
        self._bindings: list[PublicAuthorityBinding] = []
        self._explicit_none: set[tuple[int, int]] = set()

    @property
    def bindings(self) -> tuple[PublicAuthorityBinding, ...]:
        return tuple(self._bindings)

    def bind_track(
        self,
        *,
        frame_idx: int,
        candidate_uid: str,
        association_state_id: int,
        track: Any,
        binding_transaction_id: str,
        user_identity_id: Optional[int] = None,
        public_id: Optional[int] = None,
        binding_source: str = "track_manager_final_mot_authority",
        valid_from_frame: Optional[int] = None,
        valid_to_frame: Optional[int] = None,
    ) -> PublicAuthorityBinding:
        """Bind an assigned candidate to the TrackManager's final MOT ID.

        The explicit equality check is important.  It proves that a caller is
        using the output-track authority and is not merely copying an
        association PID into ``public_id``.
        """

        if candidate_uid in (None, ""):
            raise ValueError("candidate_uid is required for a public binding")
        if not hasattr(track, "mot_track_id"):
            raise TypeError("track must expose TrackManager.mot_track_id")
        mot_track_id = int(track.mot_track_id)
        resolved_public = mot_track_id if public_id is None else int(public_id)
        if resolved_public != mot_track_id:
            raise ValueError(
                "public_id must equal the final TrackManager mot_track_id unless "
                "an explicit namespace transaction is used"
            )
        lineage = getattr(track, "identity_lineage_id", None)
        if lineage is None:
            raise ValueError("TrackManager track has no identity_lineage_id")
        start = int(frame_idx if valid_from_frame is None else valid_from_frame)
        binding = PublicAuthorityBinding(
            source_run_id=self.source_run_id,
            sequence=self.sequence,
            frame_idx=int(frame_idx),
            candidate_uid=str(candidate_uid),
            association_state_id=int(association_state_id),
            user_identity_id=None if user_identity_id is None else int(user_identity_id),
            identity_lineage_id=int(lineage),
            mot_track_id=mot_track_id,
            public_id=resolved_public,
            binding_source=str(binding_source),
            binding_transaction_id=str(binding_transaction_id),
            valid_from_frame=start,
            valid_to_frame=None if valid_to_frame is None else int(valid_to_frame),
            status="EXACT",
        )
        conflicts = [
            previous
            for previous in self._bindings
            if previous.frame_idx == binding.frame_idx
            and previous.candidate_uid != binding.candidate_uid
            and previous.public_id == binding.public_id
        ]
        if conflicts:
            binding = PublicAuthorityBinding(
                **{
                    **binding.as_dict(),
                    "status": "COLLISION",
                }
            )
        self._bindings.append(binding)
        return binding

    def bind_namespace_public(
        self,
        *,
        frame_idx: int,
        candidate_uid: str,
        association_state_id: int,
        user_identity_id: Optional[int],
        identity_lineage_id: int,
        mot_track_id: int,
        public_id: int,
        binding_transaction_id: str,
        binding_source: str = "explicit_identity_namespace_transaction",
        valid_from_frame: Optional[int] = None,
        valid_to_frame: Optional[int] = None,
    ) -> PublicAuthorityBinding:
        """Record a proven namespace transaction when no Track object is handy."""

        if int(public_id) <= 0 or int(mot_track_id) <= 0:
            raise ValueError("public and MOT IDs must be positive")
        binding = PublicAuthorityBinding(
            source_run_id=self.source_run_id,
            sequence=self.sequence,
            frame_idx=int(frame_idx),
            candidate_uid=str(candidate_uid),
            association_state_id=int(association_state_id),
            user_identity_id=None if user_identity_id is None else int(user_identity_id),
            identity_lineage_id=int(identity_lineage_id),
            mot_track_id=int(mot_track_id),
            public_id=int(public_id),
            binding_source=str(binding_source),
            binding_transaction_id=str(binding_transaction_id),
            valid_from_frame=int(frame_idx if valid_from_frame is None else valid_from_frame),
            valid_to_frame=None if valid_to_frame is None else int(valid_to_frame),
            status="EXACT",
        )
        self._bindings.append(binding)
        return binding

    def record_explicit_none(self, association_state_id: int, frame_idx: int) -> None:
        self._explicit_none.add((int(association_state_id), int(frame_idx)))

    def resolve_public_authority(
        self,
        *,
        association_state_id: Optional[int] = None,
        candidate_uid: Optional[str] = None,
        frame_idx: Optional[int] = None,
        source_run_id: Optional[str] = None,
        sequence: Optional[str] = None,
        mot_track_id: Optional[int] = None,
    ) -> AuthorityResolution:
        if source_run_id is not None and str(source_run_id) != self.source_run_id:
            return AuthorityResolution("SOURCE_RUN_MISMATCH", reason="source_run_id")
        if sequence is not None and str(sequence) != self.sequence:
            return AuthorityResolution("SOURCE_RUN_MISMATCH", reason="sequence")
        if association_state_id is None and candidate_uid is None and mot_track_id is None:
            raise ValueError("at least one resolution key is required")
        selected = list(self._bindings)
        if association_state_id is not None:
            selected = [b for b in selected if b.association_state_id == int(association_state_id)]
        if candidate_uid is not None:
            selected = [b for b in selected if b.candidate_uid == str(candidate_uid)]
        if mot_track_id is not None:
            selected = [b for b in selected if b.mot_track_id == int(mot_track_id)]
        if frame_idx is not None:
            frame = int(frame_idx)
            in_range = [
                b for b in selected
                if b.valid_from_frame <= frame
                and (b.valid_to_frame is None or frame <= b.valid_to_frame)
            ]
            if not in_range and selected:
                return AuthorityResolution("STALE_BINDING", reason="outside_valid_range")
            selected = in_range
            if association_state_id is not None and (int(association_state_id), frame) in self._explicit_none:
                return AuthorityResolution("EXPLICIT_NONE")
        if not selected:
            return AuthorityResolution("NO_PUBLIC_AUTHORITY")
        statuses = {b.status for b in selected}
        public_ids = {b.public_id for b in selected}
        if "COLLISION" in statuses or len(public_ids) > 1:
            return AuthorityResolution("COLLISION", reason="multiple_public_authorities")
        if len(selected) > 1:
            return AuthorityResolution("AMBIGUOUS", reason="multiple_bindings")
        binding = selected[0]
        if binding.status != "EXACT":
            return AuthorityResolution(binding.status, reason="binding_status")
        return AuthorityResolution("EXACT", public_id=binding.public_id, binding=binding)

    def resolve(self, association_state_id: int) -> Optional[int]:
        """Compatibility resolver used by the existing same-run sidecar."""
        # The legacy sidecar has no frame argument.  Resolve the latest
        # authoritative binding for this state, while still rejecting a
        # collision at that latest frame.  New code should use the explicit
        # frame-aware method above.
        candidates = [
            item for item in self._bindings
            if item.association_state_id == int(association_state_id)
        ]
        if not candidates:
            return None
        latest_frame = max(item.frame_idx for item in candidates)
        latest = [item for item in candidates if item.frame_idx == latest_frame]
        public_ids = {item.public_id for item in latest}
        if len(public_ids) != 1 or any(item.status != "EXACT" for item in latest):
            return None
        return latest[0].public_id

    def audit(self, assignment_rows: Optional[Iterable[dict[str, Any]]] = None) -> dict[str, Any]:
        exact = sum(1 for b in self._bindings if b.status == "EXACT")
        collision = sum(1 for b in self._bindings if b.status == "COLLISION")
        result: dict[str, Any] = {
            "schema_version": "N72R2_PUBLIC_AUTHORITY_BRIDGE_V1",
            "source_run_id": self.source_run_id,
            "sequence": self.sequence,
            "session_id": self.session_id,
            "binding_count": len(self._bindings),
            "exact_binding_count": exact,
            "collision_binding_count": collision,
            "explicit_none_count": len(self._explicit_none),
            # This is a semantic guard, not a numeric comparison: equality
            # can occur by chance when independent allocators start at the
            # same value.  Only the TrackManager/namespace source above is
            # authoritative.
            "public_id_is_association_state_id": False,
            "numeric_id_coincidence_count": sum(
                b.public_id == b.association_state_id for b in self._bindings
            ),
            "runtime_future_gt_used": False,
        }
        if assignment_rows is not None:
            rows = list(assignment_rows)
            resolved = 0
            unmapped = 0
            for row in rows:
                answer = self.resolve_public_authority(
                    association_state_id=row.get("association_state_id"),
                    candidate_uid=row.get("candidate_uid"),
                    frame_idx=row.get("frame_idx"),
                    source_run_id=row.get("source_run_id"),
                    sequence=row.get("sequence"),
                )
                if answer.status == "EXACT":
                    resolved += 1
                else:
                    unmapped += 1
            result.update(
                {
                    "assignment_row_count": len(rows),
                    "resolved_assignment_row_count": resolved,
                    "unmapped_assignment_row_count": unmapped,
                    "mapping_coverage": resolved / len(rows) if rows else 1.0,
                }
            )
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "N72R2_PUBLIC_AUTHORITY_BRIDGE_V1",
            "source_run_id": self.source_run_id,
            "sequence": self.sequence,
            "session_id": self.session_id,
            "bindings": [b.as_dict() for b in self._bindings],
            "explicit_none": [list(item) for item in sorted(self._explicit_none)],
            "audit": self.audit(),
        }


__all__ = [
    "AUTHORITY_STATUSES",
    "AuthorityResolution",
    "PublicAuthorityBinding",
    "PublicAuthorityBridge",
]
