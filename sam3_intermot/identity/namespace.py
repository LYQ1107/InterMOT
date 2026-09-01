"""N6 identity namespace: four disjoint identity layers.

Layers (never mixed):

1. dataset_gt_id          -- simulator/evaluator only
2. user_identity_id       -- continuous human observer identity (allocated)
3. identity_lineage_id    -- internal permanent identity
4. public_mot_id          -- the only id written to MOT files (allocator)

Plus SAM-local ``(segment_id, sam_object_id)`` and P0-backbone ``auto_track_id``,
which are bound to lineages but never used as public MOT ids.
"""

import hashlib
import json
from typing import Dict, Optional, Tuple


class PublicTrackIDAllocator:
    """Independent, monotonic public MOT id allocator."""

    def __init__(self, start: int = 1000) -> None:
        self._next = start
        self.allocations_total = 0
        self.allocations_by_action: Dict[str, int] = {}

    def allocate(self, action: str = "ADD_NEW_IDENTITY") -> int:
        pid = self._next
        self._next += 1
        self.allocations_total += 1
        self.allocations_by_action[action] = (
            self.allocations_by_action.get(action, 0) + 1
        )
        return pid

    @property
    def next_id(self) -> int:
        return self._next

    def snapshot(self) -> dict:
        return {
            "next": self._next,
            "total": self.allocations_total,
            "by_action": dict(self.allocations_by_action),
        }

    def restore(self, snap: dict) -> None:
        self._next = snap["next"]
        self.allocations_total = snap["total"]
        self.allocations_by_action = dict(snap["by_action"])


class IdentityNamespace:
    """Bijective user/lineage/public identity maps with transactions."""

    def __init__(
        self,
        allocator: Optional[PublicTrackIDAllocator] = None,
        user_start: int = 1,
        lineage_start: int = 1,
    ) -> None:
        self.allocator = allocator or PublicTrackIDAllocator()
        self._next_user = user_start
        self._next_lineage = lineage_start
        self.user_to_lineage: Dict[int, int] = {}
        self.lineage_to_user: Dict[int, int] = {}
        self.lineage_to_public: Dict[int, int] = {}
        self.public_to_lineage: Dict[int, int] = {}
        self.auto_to_user: Dict[int, int] = {}       # P0 backbone track id -> user
        self.user_to_auto: Dict[int, Optional[int]] = {}
        self.sam_to_lineage: Dict[Tuple[int, int], int] = {}  # (segment_id, sam_id)
        self.user_first_seen: Dict[int, int] = {}

    # -- creation ----------------------------------------------------------
    def create_user(self, frame_idx: int) -> Tuple[int, int, int]:
        """ADD_NEW_IDENTITY: allocate user, lineage and public id."""
        uid = self._next_user
        self._next_user += 1
        lid = self._next_lineage
        self._next_lineage += 1
        pid = self.allocator.allocate("ADD_NEW_IDENTITY")
        self.user_to_lineage[uid] = lid
        self.lineage_to_user[lid] = uid
        self.lineage_to_public[lid] = pid
        self.public_to_lineage[pid] = lid
        self.user_first_seen[uid] = frame_idx
        self.user_to_auto[uid] = None
        return uid, lid, pid

    def first_confirm(self, auto_tid: int, frame_idx: int) -> Tuple[int, int, int]:
        """User first identifies an existing automatic track."""
        existing = self.auto_to_user.get(auto_tid)
        if existing is not None:
            return (
                existing,
                self.user_to_lineage[existing],
                self.lineage_to_public[self.user_to_lineage[existing]],
            )
        uid, lid, pid = self.create_user(frame_idx)
        self.bind_auto(auto_tid, uid)
        return uid, lid, pid

    # -- binding -----------------------------------------------------------
    def bind_auto(self, auto_tid: int, uid: int) -> None:
        old = self.auto_to_user.get(auto_tid)
        self.auto_to_user[auto_tid] = uid
        self.user_to_auto[uid] = auto_tid
        if old is not None and old != uid and self.user_to_auto.get(old) == auto_tid:
            self.user_to_auto[old] = None

    def recover(self, uid: int) -> Tuple[int, int, int]:
        """RECOVER: no new public id; returns stable triple."""
        lid = self.user_to_lineage.get(uid)
        if lid is None:
            raise ValueError(f"user identity {uid} has no lineage")
        pid = self.lineage_to_public.get(lid)
        if pid is None:
            raise ValueError(f"lineage {lid} has no public id")
        self.user_to_auto[uid] = None
        return uid, lid, pid

    def reassign(self, auto_tid: int, dest_uid: int) -> None:
        self.bind_auto(auto_tid, dest_uid)

    def swap(self, auto_a: int, auto_b: int) -> None:
        ua = self.auto_to_user.get(auto_a)
        ub = self.auto_to_user.get(auto_b)
        self.auto_to_user[auto_a] = ub
        self.auto_to_user[auto_b] = ua
        if ua is not None:
            self.user_to_auto[ua] = auto_b
        if ub is not None:
            self.user_to_auto[ub] = auto_a

    def bind_sam(self, segment_id: int, sam_object_id: int, uid: int) -> None:
        lid = self.user_to_lineage.get(uid)
        if lid is None:
            raise ValueError(f"user identity {uid} has no lineage")
        self.sam_to_lineage[(segment_id, sam_object_id)] = lid

    # -- lookups -----------------------------------------------------------
    def public_id_for(self, uid: int) -> Optional[int]:
        lid = self.user_to_lineage.get(uid)
        return self.lineage_to_public.get(lid) if lid is not None else None

    def lineage_for(self, uid: int) -> Optional[int]:
        return self.user_to_lineage.get(uid)

    def user_for_auto(self, auto_tid: int) -> Optional[int]:
        return self.auto_to_user.get(auto_tid)

    def user_for_public(self, pid: int) -> Optional[int]:
        lid = self.public_to_lineage.get(pid)
        return self.lineage_to_user.get(lid) if lid is not None else None

    # -- invariants --------------------------------------------------------
    def violations(self) -> list:
        out = []
        for uid, lid in self.user_to_lineage.items():
            if self.lineage_to_user.get(lid) != uid:
                out.append(f"user {uid} <-> lineage {lid} not bijective")
        for lid, pid in self.lineage_to_public.items():
            if self.public_to_lineage.get(pid) != lid:
                out.append(f"lineage {lid} <-> public {pid} not bijective")
        for uid in self.user_to_lineage:
            pid = self.public_id_for(uid)
            if pid is None:
                out.append(f"user {uid} has no public id")
        seen_pid = {}
        for uid in self.user_to_lineage:
            pid = self.public_id_for(uid)
            if pid in seen_pid:
                out.append(f"public id {pid} mapped to users {seen_pid[pid]}, {uid}")
            seen_pid[pid] = uid
        return out

    # -- transaction support ----------------------------------------------
    def snapshot(self) -> dict:
        return {
            "allocator": self.allocator.snapshot(),
            "next_user": self._next_user,
            "next_lineage": self._next_lineage,
            "user_to_lineage": dict(self.user_to_lineage),
            "lineage_to_user": dict(self.lineage_to_user),
            "lineage_to_public": dict(self.lineage_to_public),
            "public_to_lineage": dict(self.public_to_lineage),
            "auto_to_user": dict(self.auto_to_user),
            "user_to_auto": dict(self.user_to_auto),
            "sam_to_lineage": {
                f"{s}:{o}": l for (s, o), l in self.sam_to_lineage.items()
            },
            "user_first_seen": dict(self.user_first_seen),
        }

    def restore(self, snap: dict) -> None:
        self.allocator.restore(snap["allocator"])
        self._next_user = snap["next_user"]
        self._next_lineage = snap["next_lineage"]
        self.user_to_lineage = dict(snap["user_to_lineage"])
        self.lineage_to_user = dict(snap["lineage_to_user"])
        self.lineage_to_public = dict(snap["lineage_to_public"])
        self.public_to_lineage = dict(snap["public_to_lineage"])
        self.auto_to_user = dict(snap["auto_to_user"])
        self.user_to_auto = dict(snap["user_to_auto"])
        self.sam_to_lineage = {
            tuple(int(x) for x in k.split(":")): v
            for k, v in snap["sam_to_lineage"].items()
        }
        self.user_first_seen = dict(snap["user_first_seen"])

    def mutable_state_hash(self) -> str:
        payload = json.dumps(
            {
                "allocator": self.allocator.snapshot(),
                "next_user": self._next_user,
                "next_lineage": self._next_lineage,
                "user_to_lineage": {str(k): v for k, v in sorted(self.user_to_lineage.items())},
                "lineage_to_user": {str(k): v for k, v in sorted(self.lineage_to_user.items())},
                "lineage_to_public": {str(k): v for k, v in sorted(self.lineage_to_public.items())},
                "public_to_lineage": {str(k): v for k, v in sorted(self.public_to_lineage.items())},
                "auto_to_user": {str(k): v for k, v in sorted(self.auto_to_user.items())},
                "user_to_auto": {str(k): v for k, v in sorted(self.user_to_auto.items())},
                "sam_to_lineage": {
                    f"{s}:{o}": l
                    for (s, o), l in sorted(self.sam_to_lineage.items())
                },
                "user_first_seen": {str(k): v for k, v in sorted(self.user_first_seen.items())},
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
