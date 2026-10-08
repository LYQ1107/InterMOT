import pytest
from scripts.n72r20r4_reporting import timing_summary


def test_latency_is_a_cached_feature_measurement_not_live_backbone_FPS():
    b=[{"sequence":"a","frames":100,"seconds":2}]
    t=[{"sequence":"a","frames":100,"seconds":3}]
    r=timing_summary(t,b)
    assert r["cached_feature_fps"]==pytest.approx(100/3)
    assert r["incremental_milliseconds_per_frame_vs_causal_baseline"]==10
    assert r["live_SAM3_OSNet_end_to_end_FPS"].startswith("NOT_RUN")
    with pytest.raises(ValueError,match="axis"):
        timing_summary([{**t[0],"sequence":"b"}],b)
