from pathlib import Path
import subprocess

import pytest

from scripts import n72r21r2_common as common
from scripts import n72r21r2_storage_noise_archive_v2 as noise


@pytest.mark.parametrize("relative", ["/tmp/x.jsonl", "../escape.jsonl", "a.pt", "a.json", "a.jsonl.zst"])
def test_noise_archive_is_explicit_relative_jsonl_only(relative, tmp_path):
    with pytest.raises(ValueError): noise.checked_path(tmp_path, relative)
    with pytest.raises(ValueError): noise.checked_path(tmp_path, relative, archive=True)


def test_complete_byte_preserving_archive_fsynced_and_sealed_before_unlink(tmp_path, monkeypatch):
    stage = tmp_path / "legacy"; stage.mkdir()
    target = stage / "noise_results"; target.mkdir()
    (stage / "stage_status.json").write_text('{"status":"FINAL_RESULT_WRITTEN"}')
    (stage / "FINAL_RESULT.json").write_text('{"old_result":"keep"}')
    (stage / "protocol.json").write_text('{"frozen":"keep"}')
    src = target / "old_case.jsonl"; src.write_bytes(b'{"unchanged":true}\n' * 500)
    original_sha = common.sha256(src)
    assets, out = tmp_path / "assets", tmp_path / "out"; assets.mkdir(); out.mkdir()
    for name, value in (("STAGE", stage), ("TARGET", target), ("ASSETS", assets), ("ARCHIVE", assets / "cold"), ("OUT", out)):
        monkeypatch.setattr(noise, name, value)
    monkeypatch.setattr(common, "OUT", out)
    monkeypatch.setattr(noise, "append_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(noise, "update_status", lambda **kwargs: None)
    actual = subprocess.run
    def sealed(command, **kwargs):
        if command[0] == "unlink":
            row = common.read_json(out / noise.PREFIX / "prepared/old_case.jsonl.json")
            assert row["lossless_round_trip_and_archive_fsync_before_unlink"]
            assert noise.restored_digest(row["archive"])[0] == original_sha
        return actual(command, **kwargs)
    monkeypatch.setattr(noise.subprocess, "run", sealed)
    noise.run()
    assert not src.exists()
    archive = noise.ARCHIVE / "old_case.jsonl.zst"
    assert noise.restored_digest(archive)[0] == original_sha
    result = common.read_json(out / noise.PREFIX / "COMPLETE.json")
    assert result["archived_files"] == 1 and not result["scientific_goal_complete"]
    assert (stage / "FINAL_RESULT.json").read_text() == '{"old_result":"keep"}'


def test_noise_source_symlinks_and_other_users_rejected(tmp_path, monkeypatch):
    src = tmp_path / "real.jsonl"; src.write_bytes(b"old")
    (tmp_path / "alias.jsonl").symlink_to(src)
    with pytest.raises(ValueError): noise.checked_path(tmp_path, "alias.jsonl")
    monkeypatch.setattr(noise.os, "getuid", lambda: src.stat().st_uid + 1)
    with pytest.raises(ValueError): noise.checked_path(tmp_path, "real.jsonl")
