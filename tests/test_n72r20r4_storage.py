import hashlib
import json
import os

import pytest

from scripts.n72r20r4_storage import share_identical_exports


def manifest(path, source, digest):
    path.write_text(json.dumps({"trajectory_path": str(source), "trajectory_sha256": digest}))


def test_only_identical_sealed_outputs_share_storage(tmp_path):
    a = tmp_path / "a.txt"; b = tmp_path / "b.txt"; c = tmp_path / "c.txt"
    a.write_text("same result\n"); b.write_text(a.read_text()); c.write_text("other result\n")
    paths = []
    for i, source in enumerate((a, b, c)):
        path = tmp_path / f"manifest{i}.json"
        manifest(path, source, hashlib.sha256(source.read_bytes()).hexdigest())
        paths.append(path)
    result = share_identical_exports(tmp_path, paths)
    assert result["new_shared_paths"] == 1
    assert os.path.samefile(a, b) and not os.path.samefile(a, c)
    assert all(p.exists() for p in (a, b, c))
    assert share_identical_exports(tmp_path, paths)["new_shared_paths"] == 0


def test_mismatched_or_external_outputs_are_not_modified(tmp_path):
    stage = tmp_path / "stage"; stage.mkdir()
    a = stage / "a.txt"; a.write_text("result\n")
    path = stage / "manifest.json"
    manifest(path, a, "not the content SHA")
    with pytest.raises(ValueError, match="SHA mismatch"):
        share_identical_exports(stage, [path])
    assert a.read_text() == "result\n"
    external = tmp_path / "historical.txt"; external.write_text("historical result\n")
    manifest(path, external, hashlib.sha256(external.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="outside"):
        share_identical_exports(stage, [path])
    assert external.read_text() == "historical result\n"
