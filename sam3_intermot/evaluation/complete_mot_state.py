"""Explicit all-attribute MOT snapshots, additional to legacy subset hashes.

Never change a tracker. Unknown/opaque values are rejected instead of silently
excluded or stringified. Arrays retain dtype, shape and exact data bytes.
Identity actor/controller state is separate from the joint MOT tracker state.
"""
import hashlib
import json
import math
from collections import deque
from dataclasses import is_dataclass, fields
import numpy as np
from sam3_intermot.association.identity_state import IdentityState
from sam3_intermot.association.state_manager import StateManager
from sam3_intermot.association.appearance_memory import AppearanceMemory


def canonical(value):
    if isinstance(value, np.generic):
        return {"numpy_scalar_dtype": value.dtype.str, "value": canonical(value.item())}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value): raise ValueError("Nonfinite tracker state is not valid AA evidence")
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject: raise ValueError("Opaque object arrays cannot be silently hashed")
        if value.dtype.kind in "fc" and not np.isfinite(value).all(): raise ValueError("Nonfinite tensor state")
        return {"ndarray_dtype": value.dtype.str, "shape": list(value.shape),
                "exact_C_order_bytes_hex": np.ascontiguousarray(value).tobytes().hex()}
    if isinstance(value, dict):
        pairs = [[canonical(k), canonical(v)] for k, v in value.items()]
        pairs.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True, separators=(",", ":")))
        return {"dict_items_preserving_key_types": pairs}
    if isinstance(value, deque):
        return {"deque": [canonical(v) for v in value], "maxlen": value.maxlen}
    if isinstance(value, (list, tuple)):
        return {type(value).__name__: [canonical(v) for v in value]}
    if isinstance(value, (set, frozenset)):
        items = [canonical(v) for v in value]
        items.sort(key=lambda v: json.dumps(v, sort_keys=True, separators=(",", ":")))
        return {type(value).__name__: items}
    if isinstance(value, (IdentityState, StateManager, AppearanceMemory)):
        return {"class": type(value).__module__ + "." + type(value).__name__,
                "all_instance_attributes": canonical(vars(value))}
    if is_dataclass(value) and not isinstance(value, type):
        registered = {"sam3_intermot.association.identity_authority.AuthorityConfig",
                      "sam3_intermot.association.opportunity_tracker.InterventionPolicy",
                      "sam3_intermot.association.opportunity_memory.MemoryPolicy",
                      "sam3_intermot.association.state_manager.StateManagerConfig",
                      "sam3_intermot.association.appearance_memory.AppearanceAnchor",
                      "sam3_intermot.association.appearance_memory.AppearanceRecord"}
        name = type(value).__module__ + "." + type(value).__name__
        if name in registered:
            attrs = vars(value)
            assert {f.name for f in fields(value)} <= set(attrs)
            return {"dataclass": name, "all_fields_and_extra_attributes": canonical(attrs)}
    raise TypeError("Unsupported tracker snapshot value: " + type(value).__module__ + "." + type(value).__name__)


def complete_mot_snapshot(bridge):
    tracker = bridge.tracker
    assert len(tracker.states) == len(tracker.manager.states)
    alias_contract = {}
    for public, state in tracker.states.items():
        assert tracker.manager.states.get(state.pid) is state
        assert tracker.manager.external_public_ids[state.pid] == public == state.public_id
        alias_contract[str(public)] = {"manager_state_id": state.pid, "shared_state_instance": True}
    return {"tracker_class": type(tracker).__module__ + "." + type(tracker).__name__,
            "all_tracker_attribute_names": sorted(vars(tracker)),
            "manager_public_state_alias_contract": alias_contract,
            "all_tracker_attributes": canonical(vars(tracker))}


def complete_mot_fingerprint(bridge):
    encoded = json.dumps(complete_mot_snapshot(bridge), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def assert_complete_mot_AA(left, right):
    a, b = complete_mot_snapshot(left), complete_mot_snapshot(right)
    encoded_a = json.dumps(a, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    encoded_b = json.dumps(b, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if encoded_a != encoded_b:
        raise AssertionError("Complete MOT attributes differ; legacy semantic/subset fingerprints cannot substitute")
    return hashlib.sha256(encoded_a).hexdigest()
