"""Accessors that enforce exact inheritance of the N72R16 probe protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

from sam3_intermot.identity_probe.protocol import (
    Anchor,
    FutureObservation,
    PROTOCOL_VERSION as N72R16_PROTOCOL_VERSION,
    anchor_from_dict,
    iter_anchors as _iter_anchors,
    read_protocol as _read_protocol,
)

PROTOCOL_VERSION = N72R16_PROTOCOL_VERSION
FINAL_GOAL = "Human Identity Representation Probe"
CENTRAL_QUESTION = "Can one human-confirmed identity observation reliably recognize the same person against hard competing identities over future frames?"


def read_frozen_protocol(path: str | Path) -> dict[str, object]:
    """Read a copied N72R16 protocol and reject any semantic drift."""

    document = _read_protocol(path)
    if document.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("N72R17 requires the exact N72R16 protocol version")
    if document.get("stage") != "N72R16":
        raise ValueError("N72R17 protocol must retain the N72R16 stage marker")
    if document.get("goal") != FINAL_GOAL:
        raise ValueError("protocol Goal does not match the frozen Final Goal")
    if document.get("central_question") != CENTRAL_QUESTION:
        raise ValueError("protocol central question does not match the frozen Final Goal")
    return document


def iter_anchors(document: dict[str, object]) -> Iterator[Anchor]:
    """Yield anchors through the original N72R16 deserializer."""

    yield from _iter_anchors(document)


def protocol_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "Anchor",
    "FutureObservation",
    "FINAL_GOAL",
    "CENTRAL_QUESTION",
    "PROTOCOL_VERSION",
    "anchor_from_dict",
    "iter_anchors",
    "protocol_sha256",
    "read_frozen_protocol",
]
