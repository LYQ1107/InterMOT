from scripts.n72r21r2_continuation_checkpoint_v6 import active_procs


def test_multi_single_and_waiting_driver_active_shapes_are_preserved():
    assert active_procs(None) == []
    assert active_procs({"pid": None}) == [{"pid": None, "is_live_exact_R2_process": False}]
    result = active_procs([{"pid": 999999999}, {"pid": None}])
    assert len(result) == 2 and all(not r["is_live_exact_R2_process"] for r in result)
