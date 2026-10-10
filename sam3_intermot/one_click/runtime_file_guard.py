"""Executable GT/label file boundary for frozen research replay workers."""
from contextlib import contextmanager
import builtins
import io
import os


def forbidden_runtime_file(file):
    if isinstance(file, int):
        return False
    try:
        path = os.fsdecode(os.fspath(file)).replace("\\", "/")
    except TypeError:
        return False
    return "/gt/" in path or "/initialization_truth/" in path or "/events/corpus/" in path or "/events/corpus_v2/" in path or "/offline_supervision/" in path


@contextmanager
def runtime_file_guard():
    previous_builtin, previous_io = builtins.open, io.open

    def guarded(original):
        def open_file(file, *args, **kwargs):
            if forbidden_runtime_file(file):
                raise ValueError("GT/offline-label file access forbidden in runtime")
            return original(file, *args, **kwargs)
        return open_file

    builtins.open, io.open = guarded(previous_builtin), guarded(previous_io)
    try:
        yield
    finally:
        builtins.open, io.open = previous_builtin, previous_io
