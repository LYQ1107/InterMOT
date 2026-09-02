"""Allocator-backed atomic transaction for ADD_NEW_IDENTITY."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sam3_intermot.identity.namespace import IdentityNamespace


@dataclass(frozen=True)
class AllocationPreview:
    user_identity_id: int
    identity_lineage_id: int
    public_mot_id: int
    public_id_source: str = "system_allocator"


class AddIdentityTransaction:
    """Snapshot allocator/namespace/manager before a server-side Add.

    The user supplies only a spatial input.  ``public_mot_id`` is returned by
    ``IdentityNamespace.create_user`` and cannot be injected into ``prepare``.
    A backend cleanup callback is required when a backend object was created
    before a later operation failed.
    """

    def __init__(self, namespace: IdentityNamespace, manager: Any = None, backend: Any = None) -> None:
        self.namespace = namespace
        self.manager = manager
        self.backend = backend
        self._namespace_snapshot = None
        self._manager_snapshot = None
        self._preview: AllocationPreview | None = None
        self._committed = False
        self._backend_cleanup: Callable[[AllocationPreview], None] | None = None

    @property
    def preview(self) -> AllocationPreview | None:
        return self._preview

    def prepare(self, frame_idx: int) -> AllocationPreview:
        if self._preview is not None:
            raise RuntimeError("ADD transaction is already prepared")
        self._namespace_snapshot = self.namespace.snapshot()
        self._manager_snapshot = self.manager.snapshot() if self.manager is not None else None
        uid, lineage, public = self.namespace.create_user(int(frame_idx))
        self._preview = AllocationPreview(uid, lineage, public)
        return self._preview

    def commit(self) -> AllocationPreview:
        if self._preview is None:
            raise RuntimeError("ADD transaction was not prepared")
        self._committed = True
        return self._preview

    def rollback(self) -> None:
        if self._committed:
            raise RuntimeError("cannot roll back a committed ADD transaction")
        if self._preview is not None and self._backend_cleanup is not None:
            self._backend_cleanup(self._preview)
        if self._namespace_snapshot is not None:
            self.namespace.restore(self._namespace_snapshot)
        if self.manager is not None and self._manager_snapshot is not None:
            self.manager.restore(self._manager_snapshot)
        self._preview = None

    def execute(
        self,
        frame_idx: int,
        apply_fn: Callable[[AllocationPreview], Any],
        *,
        backend_cleanup: Callable[[AllocationPreview], None] | None = None,
    ) -> tuple[bool, Any, str | None]:
        self._backend_cleanup = backend_cleanup
        preview = self.prepare(frame_idx)
        try:
            result = apply_fn(preview)
        except Exception as exc:
            self.rollback()
            return False, None, f"{type(exc).__name__}: {exc}"
        self.commit()
        return True, result, None

    def __enter__(self) -> "AddIdentityTransaction":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None and not self._committed:
            self.rollback()
        elif exc_type is None and self._preview is not None and not self._committed:
            self.commit()
        return False


__all__ = ["AddIdentityTransaction", "AllocationPreview"]
