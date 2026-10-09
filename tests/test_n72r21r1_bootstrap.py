from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.n72r21r1_common import output_path, HISTORY
from scripts.n72r21r1_reproduce_v2 import diagnostic_step


def test_history_escape_is_rejected():
    with pytest.raises(ValueError):
        output_path('../../../InterMOT/outputs/N72R21/stage_status.json')
    assert not output_path('tests/new.json').resolve().is_relative_to(HISTORY)


def test_precommit_scalars_do_not_follow_mutable_state():
    class Tracker:
        def __init__(self):
            self.states = {1: SimpleNamespace(last_native_tid=7, last_seen_frame=0, pid=1)}
            self.event = {'human_anchor': np.array([1., 0.])}

        def clone(self):
            return deepcopy(self)

        def step(self, rows, frame):
            self.states[1].last_native_tid = 8
            self.states[1].last_seen_frame = frame
            return {'target_public_id': 1, 'target_uid': 'real:1', 'states_before_commit_axis': [1], 'base_matrix': np.array([[1.]]), 'outputs': [{'public_id': 1, 'candidate_uid': 'real:1'}], 'solver': {'assignment_rows': [{'score': 1.}]}}

    class Bridge:
        def __init__(self):
            self.tracker = Tracker()
            self.identity = None

        def step(self, frame, rows):
            result = self.tracker.step(rows, frame)
            result.update(identity_decision=None, selected_action=None)
            return result

    bridge = Bridge()
    rows = [{'candidate_uid': 'real:1', 'native_tid': 8, 'conf': .8, 'box_xyxy': [0., 0., 2., 2.], 'feature': np.array([1., 0.])}]
    result = diagnostic_step(bridge, 1, rows)['r21r1_precommit_diagnostic']
    assert result['precommit_native_tid'] == 7
    assert result['precommit_track_gap'] == 1
    assert bridge.tracker.states[1].last_native_tid == 8
    assert not result['GT_used_for_diagnostic']
