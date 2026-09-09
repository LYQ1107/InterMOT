#!/usr/bin/env python
"""Run the pinned TrackEval CLI with NumPy compatibility aliases only.

The checked-out TrackEval commit predates NumPy 2's removal of np.float/np.int.
This bootstrap does not alter the submodule; it only restores aliases in the
child interpreter before executing the official entry point.
"""

from __future__ import annotations

import runpy
import sys
import argparse

import numpy as np


for name, value in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, name):
        setattr(np, name, value)


# The pinned CLI declares scalar options whose defaults are None with nargs='+'.
# Its own conversion leaves those one-element lists in the config.  Unwrap only
# the two scalar paths used here; list-valued options such as SEQ_INFO, METRICS,
# and TRACKERS_TO_EVAL remain untouched.
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
    raise SystemExit("usage: n72r11r5_trackeval_entry.py RUN_MOT_CHALLENGE.py [args]")
official_entry = sys.argv[1]
sys.argv = [official_entry, *sys.argv[2:]]
runpy.run_path(official_entry, run_name="__main__")
