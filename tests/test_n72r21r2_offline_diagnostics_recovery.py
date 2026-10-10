from scripts.n72r21r2_offline_diagnostics_driver_v2 import OPERATIONS, next_task, live_scoped_pid


def test_dependency_and_failure_preserving_resumption():
    seq = ["dancetrack0074", "dancetrack0020"]
    ready = lambda dep, s: dep == "events/label_audit"
    task = next_task(seq, set(), set(), ready)
    assert task == (OPERATIONS[0][0], seq[0], OPERATIONS[0][2], OPERATIONS[0][3])
    key = OPERATIONS[0][0] + "/" + seq[0]
    for done, failed in [({key}, set()), (set(), {key})]:
        assert next_task(seq, done, failed, ready)[1] == seq[1]
    assert next_task(seq, set(), set(), lambda dep, s: False) is None


def test_nonexistent_pid_is_not_live_owner():
    assert not live_scoped_pid(-1)
