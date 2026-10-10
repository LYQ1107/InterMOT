from pathlib import Path
import pytest
from sam3_intermot.one_click.runtime_file_guard import runtime_file_guard, forbidden_runtime_file


@pytest.mark.parametrize("path", ["/dataset/train/x/gt/gt.txt", "/stage/initialization_truth/x.json", "/stage/events/corpus_v2/x.json"])
def test_actual_GT_and_future_label_file_reads_are_rejected(path):
    assert forbidden_runtime_file(path)
    with runtime_file_guard(), pytest.raises(ValueError):
        Path(path).read_text()


def test_current_candidate_and_weight_files_remain_readable(tmp_path):
    path = tmp_path / "current_candidate.json"
    path.write_text("{}")
    with runtime_file_guard():
        assert path.read_text() == "{}"
    assert not forbidden_runtime_file("/weights/best.pt")
