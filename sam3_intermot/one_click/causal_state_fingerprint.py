"""Tensor-inclusive runtime fingerprints, not just legacy semantic digests."""
import hashlib
import json
import numpy as np


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, set):
        return sorted(serial(v) for v in value)
    return value


def tracker_state(bridge):
    tracker = bridge.tracker
    return {"frame": tracker.frame, "target_public": tracker.target_public,
            "next_public": tracker.next_public, "next_state": tracker.next_state,
            "last_trusted_frame": tracker.last_trusted_frame,
            "states": {str(p): serial(vars(s)) for p, s in sorted(tracker.states.items())}}


def fingerprint(value):
    return hashlib.sha256(json.dumps(serial(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def full_tracker_fingerprint(bridge):
    return fingerprint(tracker_state(bridge))
