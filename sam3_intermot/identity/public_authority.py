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
    "NO_CANDIDATE_ASSIGNED",
    "NO_PUBLIC_AUTHORITY",
    "NO_AUTHORITY",
    "TERMINATED",
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
class IdentityStateAuthorityBinding:
    """Immutable identity-state → public authority binding.

    Unlike the legacy candidate-scoped binding above, this record contains no
    candidate UID.  A candidate is a per-frame observation that may change or
    disappear; the persistent identity binding is created once and survives
    those changes.
    """

    source_run_id: str
    sequence: str
    association_state_id: int
    identity_lineage_id: int
    mot_track_id: int
    public_id: int
    created_frame: int
    binding_transaction_id: str
    valid_to_frame: Optional[int] = None
    status: str = "EXACT"

    def __post_init__(self) -> None:
        if self.status not in {"EXACT", "TERMINATED"}:
            raise ValueError(f"invalid persistent authority status: {self.status}")
        if int(self.public_id) != int(self.mot_track_id):
            raise ValueError("persistent public_id must equal mot_track_id")
        if self.valid_to_frame is not None and int(self.valid_to_frame) < int(self.created_frame):
            raise ValueError("persistent authority valid_to_frame precedes created_frame")

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
        self._identity_bindings: dict[int, IdentityStateAuthorityBinding] = {}
        self._identity_none: set[tuple[int, int]] = set()

    @property
    def bindings(self) -> tuple[PublicAuthorityBinding, ...]:
        return tuple(self._bindings)

    @property
    def identity_bindings(self) -> tuple[IdentityStateAuthorityBinding, ...]:
        return tuple(self._identity_bindings.values())

    def bind_identity_state(
        self,
        *,
        association_state_id: int,
        public_id: int,
        mot_track_id: int,
        lineage_id: int,
        created_frame: int,
        transaction_id: str,
        valid_to_frame: Optional[int] = None,
    ) -> IdentityStateAuthorityBinding:
        """Create the one immutable authority binding for a persistent state.

        Repeating the exact same binding is idempotent.  A different public ID
        for an existing state is rejected before any new record is appended.
        This is the only exact state→public path used by N72R3.
        """

        state_id = int(association_state_id)
        public = int(public_id)
        mot = int(mot_track_id)
        lineage = int(lineage_id)
        existing = self._identity_bindings.get(state_id)
        if existing is not None:
            if (
                existing.public_id != public
                or existing.mot_track_id != mot
                or existing.identity_lineage_id != lineage
            ):
                raise ValueError(
                    f"association state {state_id} already has immutable public authority "
                    f"{existing.public_id}"
                )
            return existing
        occupied = [
            item
            for item in self._identity_bindings.values()
            if item.public_id == public and item.association_state_id != state_id
        ]
        if occupied:
            raise ValueError(f"public_id {public} is already bound to another identity state")
        binding = IdentityStateAuthorityBinding(
            source_run_id=self.source_run_id,
            sequence=self.sequence,
            association_state_id=state_id,
            identity_lineage_id=lineage,
            mot_track_id=mot,
            public_id=public,
            created_frame=int(created_frame),
            binding_transaction_id=str(transaction_id),
            valid_to_frame=None if valid_to_frame is None else int(valid_to_frame),
        )
        self._identity_bindings[state_id] = binding
        return binding

    def record_identity_no_candidate(self, association_state_id: int, frame_idx: int) -> None:
        """Record a legal boundary decision without altering identity authority."""

        state_id = int(association_state_id)
        if state_id not in self._identity_bindings:
            raise ValueError(f"unknown persistent identity state: {state_id}")
        self._identity_none.add((state_id, int(frame_idx)))

    def resolve_identity_state(
        self, association_state_id: int, frame_idx: Optional[int] = None
    ) -> AuthorityResolution:
        state_id = int(association_state_id)
        binding = self._identity_bindings.get(state_id)
        if binding is None:
            return AuthorityResolution("NO_AUTHORITY", reason="identity_state_not_bound")
        if frame_idx is not None:
            frame = int(frame_idx)
            if (state_id, frame) in self._identity_none:
                return AuthorityResolution("EXPLICIT_NONE", reason="persistent_identity_has_no_candidate")
            if frame < binding.created_frame or (
                binding.valid_to_frame is not None and frame > binding.valid_to_frame
            ):
                return AuthorityResolution("STALE_BINDING", reason="outside_identity_valid_range")
        if binding.status == "TERMINATED":
            return AuthorityResolution("TERMINATED", public_id=binding.public_id, binding=binding)
        return AuthorityResolution("EXACT", public_id=binding.public_id, reason="persistent_identity", binding=binding)

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
        if association_state_id is not None and int(association_state_id) in self._identity_bindings:
            if candidate_uid is None and mot_track_id is None:
                return self.resolve_identity_state(int(association_state_id), frame_idx)
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
        identity = self._identity_bindings.get(int(association_state_id))
        if identity is not None:
            return int(identity.public_id)
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
            "schema_version": "N72R3_PUBLIC_AUTHORITY_BRIDGE_V2",
            "source_run_id": self.source_run_id,
            "sequence": self.sequence,
            "session_id": self.session_id,
            "binding_count": len(self._bindings),
            "exact_binding_count": exact,
            "collision_binding_count": collision,
            "explicit_none_count": len(self._explicit_none),
            "persistent_identity_binding_count": len(self._identity_bindings),
            "persistent_identity_no_candidate_count": len(self._identity_none),
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
            "schema_version": "N72R3_PUBLIC_AUTHORITY_BRIDGE_V2",
            "source_run_id": self.source_run_id,
            "sequence": self.sequence,
            "session_id": self.session_id,
            "bindings": [b.as_dict() for b in self._bindings],
            "explicit_none": [list(item) for item in sorted(self._explicit_none)],
            "identity_bindings": [b.as_dict() for b in self._identity_bindings.values()],
            "identity_no_candidate": [list(item) for item in sorted(self._identity_none)],
            "audit": self.audit(),
        }

    def snapshot(self) -> dict[str, Any]:
        """Capture both legacy evidence and N72R3 identity authority."""

        return {
            "schema_version": "N72R3_PUBLIC_AUTHORITY_BRIDGE_SNAPSHOT_V1",
            "source_run_id": self.source_run_id,
            "sequence": self.sequence,
            "session_id": self.session_id,
            "bindings": [item.as_dict() for item in self._bindings],
            "explicit_none": [list(item) for item in sorted(self._explicit_none)],
            "identity_bindings": [item.as_dict() for item in self._identity_bindings.values()],
            "identity_no_candidate": [list(item) for item in sorted(self._identity_none)],
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        if str(snapshot.get("source_run_id")) != self.source_run_id:
            raise ValueError("authority snapshot source_run_id mismatch")
        if str(snapshot.get("sequence")) != self.sequence:
            raise ValueError("authority snapshot sequence mismatch")
        self.session_id = snapshot.get("session_id")
        self._bindings = [PublicAuthorityBinding(**item) for item in snapshot.get("bindings", [])]
        self._explicit_none = {
            (int(item[0]), int(item[1])) for item in snapshot.get("explicit_none", [])
        }
        self._identity_bindings = {
            int(item["association_state_id"]): IdentityStateAuthorityBinding(**item)
            for item in snapshot.get("identity_bindings", [])
        }
        self._identity_none = {
            (int(item[0]), int(item[1])) for item in snapshot.get("identity_no_candidate", [])
        }


__all__ = [
    "AUTHORITY_STATUSES",
    "AuthorityResolution",
    "IdentityStateAuthorityBinding",
    "PublicAuthorityBinding",
    "PublicAuthorityBridge",
]
