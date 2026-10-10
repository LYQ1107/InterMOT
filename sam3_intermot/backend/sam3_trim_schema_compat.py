"""Process-local compatibility for a discarded SAM3 interactive debug field.

The pinned demo compactor omits multistep_point_inputs, while the native
trim helper unconditionally indexes it. First execute the UNCHANGED helper.
Only that exact KeyError allows a retry with a read-only missing-key view.
No model tensor, trim flag, detection threshold or source file is changed.
"""
from contextlib import contextmanager
from functools import wraps


class MissingPointDebugView(dict):
    def __missing__(self, key):
        if key == "multistep_point_inputs":
            return [None]
        raise KeyError(key)


def compatible_trim(original, counts):
    @wraps(original)
    def trim(self, frame_idx, output_dict, current_out, memory_encoder_was_used):
        try:
            return original(self, frame_idx, output_dict, current_out, memory_encoder_was_used)
        except KeyError as error:
            if error.args != ("multistep_point_inputs",):
                raise
        non_cond = output_dict["non_cond_frame_outputs"]
        views = []
        for frame, value in list(non_cond.items()):
            if isinstance(value, dict) and "multistep_point_inputs" not in value:
                view = MissingPointDebugView(value)
                non_cond[frame] = view
                views.append((frame, value, view))
        if not views:
            raise RuntimeError("Missing debug key is not in a compact past output; compatibility does not apply")
        counts["native_missing_point_debug_retries"] += 1
        try:
            return original(self, frame_idx, output_dict, current_out, memory_encoder_was_used)
        finally:
            # Keep only the outputs actually replaced by the official helper.
            for frame, value, view in views:
                if non_cond.get(frame) is view:
                    non_cond[frame] = value
    return trim


@contextmanager
def pinned_sam3_trim_compatibility():
    from sam3.model.video_tracking_multiplex import VideoTrackingMultiplex
    original = VideoTrackingMultiplex._trim_output_and_memory
    counts = {"native_missing_point_debug_retries": 0}
    VideoTrackingMultiplex._trim_output_and_memory = compatible_trim(original, counts)
    try:
        yield counts
    finally:
        VideoTrackingMultiplex._trim_output_and_memory = original
