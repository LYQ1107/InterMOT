#!/usr/bin/env python3
"""Stable local entry point for the pinned TrackEval CLI.

The pinned CLI uses ``nargs='+'`` for scalar optional fields.  This wrapper
unwraps only those scalar fields and leaves genuine list-valued arguments
unchanged.  The third-party TrackEval checkout is not modified.
"""

from __future__ import annotations

import argparse
import runpy
import sys

import numpy as np


for name, value in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, name):
        setattr(np, name, value)

_parse_args = argparse.ArgumentParser.parse_args


def _parse_args_compat(self, *args, **kwargs):
    namespace = _parse_args(self, *args, **kwargs)
    for field in ("SEQMAP_FILE", "OUTPUT_FOLDER"):
        value = getattr(namespace, field, None)
        if isinstance(value, list) and len(value) == 1:
            setattr(namespace, field, value[0])
    return namespace


argparse.ArgumentParser.parse_args = _parse_args_compat

if len(sys.argv) < 2:
    raise SystemExit("usage: n72r20r3r2r3_trackeval_entry.py RUN_MOT_CHALLENGE.py [args]")

official_entry = sys.argv[1]
sys.argv = [official_entry, *sys.argv[2:]]
runpy.run_path(official_entry, run_name="__main__")
