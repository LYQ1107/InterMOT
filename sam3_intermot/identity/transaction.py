"""Small transaction helper providing snapshot/rollback over transactionables."""

from typing import Any, List, Tuple


class Transaction:
    """Snapshot a set of transactionable objects and roll back on demand.

    ``transactionable`` objects must implement ``snapshot()`` and
    ``restore(snapshot)`` (e.g. TrackManager and IdentityLineageRegistry).
    Backend rollback is performed by interaction handlers with explicit
    remove/re-prompt calls, since a model session cannot be deep-copied.
    """

    def __init__(self, *transactionables: Any) -> None:
        self._snapshots: List[Tuple[Any, Any]] = [
            (obj, obj.snapshot()) for obj in transactionables
        ]

    def rollback(self) -> None:
        for obj, snapshot in reversed(self._snapshots):
            obj.restore(snapshot)
        self._snapshots = []

    def commit(self) -> None:
        self._snapshots = []

    def __enter__(self) -> "Transaction":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.rollback()
            return False
        self.commit()
        return False
