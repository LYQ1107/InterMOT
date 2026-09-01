from sam3_intermot.evaluation.trackeval_runner import find_trackeval_root


def test_find_trackeval_returns_path_or_none(tmp_path):
    # no TrackEval present -> None (or existing path on this server, if found)
    result = find_trackeval_root(".")
    assert result is None or result.endswith("TrackEval")
