"""GT-free reconstruction of the frozen P0 observed-native recurrence.

Derived audit measurement, not a field directly recorded in compact traces,
not the inactive memory-write/native_streak counter, never a model input.
"""


def observed_native_before_after(trace, native_axes, event_frame):
    before, after = {}, {}
    streak, key, last_observation = 0, None, -1
    if type(event_frame) is not int or event_frame < 0:
        raise ValueError("Exact original human event frame")
    for index, row in enumerate(trace):
        frame = row["frame"]
        if type(frame) is not int or frame != index or frame not in native_axes:
            raise ValueError("Full original contiguous frame and candidate axes required")
        if row.get("runtime_GT_read", row.get("runtime_gt_read", False)) or row.get("runtime_future_GT_used", row.get("runtime_future_gt_used", False)):
            raise ValueError("Only sealed GT-free runtime")
        if row.get("identity_memory_write", row.get("joint_identity_memory_write", False)):
            raise ValueError("This recurrence audit is restricted to original no-write P0")
        before[frame] = streak
        if frame > event_frame:
            uid = row["target_uid"]
            if uid is None:
                streak, key = 0, None
            else:
                if uid not in native_axes[frame]:
                    raise ValueError("Committed UID outside full original current candidate axis")
                native = native_axes[frame][uid]
                if not isinstance(native, tuple) or len(native) != 2 or type(native[1]) is not int:
                    raise ValueError("Exact native_scope/int native_tid composite key")
                streak = streak + 1 if native == key and last_observation == frame - 1 else 1
                key, last_observation = native, frame
        after[frame] = streak
    return before, after


def controlled_prefix_probe_values(baseline, parent, native_axes, event_frame, probes):
    start = parent["source_frame"]
    prefix = parent["own_actual_source_prefix"]
    if not prefix or prefix[0]["frame"] != start or any(r["frame"] != start+i for i,r in enumerate(prefix)):
        raise ValueError("Exact own source prefix required")
    trace = baseline[:start] + prefix
    before, _ = observed_native_before_after(trace, native_axes, event_frame)
    values = {}
    for probe in probes:
        frame = probe["frame"]
        if frame not in before:
            raise ValueError("No nearest-frame or extrapolated source probe")
        recorded = probe["causal_state_distribution_observables"]["observed_native_streak"]
        if type(recorded) is not int or recorded != before[frame]:
            raise ValueError("Derived recurrence disagrees with actually recorded source counter")
        values[frame] = before[frame]
    return values
