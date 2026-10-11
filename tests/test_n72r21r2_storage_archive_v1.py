import os
from pathlib import Path
import subprocess

import pytest

from scripts import n72r21r2_common as common
from scripts import n72r21r2_storage_archive_v1 as archive
from scripts import n72r21r2_storage_archive_readback_v1 as readback


@pytest.mark.parametrize("relative", ["/tmp/absolute.json", "../escape.json", "dev/../../escape.json", "dev/a.pt", "dev/a.json.zst"])
def test_relative_path_guard_never_accepts_broad_or_non_json_targets(relative):
    with pytest.raises(ValueError): archive.checked_source(relative)
    with pytest.raises(ValueError): readback.checked_archive_path(relative)


def test_missing_symlink_parent_and_other_owner_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "TARGET", tmp_path)
    with pytest.raises(ValueError): archive.checked_source("missing.json")
    src = tmp_path / "ok.json"; src.write_bytes(b"original")
    assert archive.checked_source("ok.json") == src
    (tmp_path / "link.json").symlink_to(src)
    with pytest.raises(ValueError): archive.checked_source("link.json")
    directory = tmp_path / "real"; directory.mkdir(); (directory / "a.json").write_bytes(b"x")
    (tmp_path / "alias").symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError): archive.checked_source("alias/a.json")
    monkeypatch.setattr(archive.os, "getuid", lambda: src.stat().st_uid + 1)
    with pytest.raises(ValueError): archive.checked_source("ok.json")


def test_actual_zstd_round_trip_and_corruption_detection(tmp_path):
    src = tmp_path / "bytes.json"; src.write_bytes(b'[{"old":true}]\n' * 1000)
    dst = tmp_path / "bytes.json.zst"
    subprocess.run(["zstd", "-q", "-T1", "--check", str(src), "-o", str(dst)], check=True)
    assert archive.restored_digest(dst) == (common.sha256(src), src.stat().st_size)
    dst.write_bytes(b"not a valid zstd frame")
    with pytest.raises(ValueError): archive.restored_digest(dst)


def fixture_archive(tmp_path, monkeypatch):
    history, assets, out = tmp_path / "legacy", tmp_path / "assets", tmp_path / "out"
    target = history / "outputs/N72R19R1/metrics/eval_cache"
    target.mkdir(parents=True); assets.mkdir(); out.mkdir()
    stage = history / "outputs/N72R19R1"
    (stage / "stage_status.json").write_text('{"status":"COMPLETE_FAIL_CLEAN_PRESERVATION","stop_rule_applied":true}')
    for name in ("FINAL_RESULT.json", "protocol.json", "corruption_manifest.json"):
        (stage / name).write_text('{"preserve":true}')
    (history / "scripts").mkdir(); (history / "scripts/n72r19r1_evaluate.py").write_text("# preserve source\n")
    (target / "dev").mkdir(); (target / "val").mkdir()
    for relative in ("dev/a.json", "val/b.json"):
        (target / relative).write_bytes(b'{"old_scientific_bytes":"keep"}\n' * 500)
    monkeypatch.setattr(common, "OUT", out)
    for module in (archive, readback):
        monkeypatch.setattr(module, "OUT", out)
        monkeypatch.setattr(module, "append_log", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "update_status", lambda **kwargs: None)
    monkeypatch.setattr(archive, "HISTORY", history)
    monkeypatch.setattr(archive, "ASSETS", assets)
    monkeypatch.setattr(archive, "TARGET", target)
    monkeypatch.setattr(archive, "ARCHIVE", assets / "cold")
    return out, target


def test_all_temp_files_sealed_before_unlink_then_independently_recoverable(tmp_path, monkeypatch):
    out, target = fixture_archive(tmp_path, monkeypatch)
    original_run = subprocess.run
    removed = []
    def checked_command(command, **kwargs):
        if command[0] == "unlink":
            src = Path(command[1]); relative = str(src.relative_to(target))
            receipt = common.read_json(out / "audit/storage_legacy_cache_archive_v1/prepared" / relative)
            assert receipt["original_uncompressed_copy_removed"] is False
            assert common.sha256(src) == receipt["original_sha256"]
            assert archive.restored_digest(receipt["archive"]) == (receipt["original_sha256"], receipt["restored_bytes"])
            removed.append(relative)
        return original_run(command, **kwargs)
    monkeypatch.setattr(archive.subprocess, "run", checked_command)
    archive.run()
    assert sorted(removed) == ["dev/a.json", "val/b.json"]
    assert not list(target.rglob("*.json"))
    readback.run()
    result = common.read_json(out / readback.RESULT)
    assert result["archived_files"] == 2
    assert result["all_original_byte_streams_independently_decoded_and_SHA_verified"]
    assert result["logical_bytes_reclaimed"] > 0
    assert not result["scientific_goal_complete"]
    with pytest.raises(FileExistsError): readback.run()


def test_unsealed_archive_is_not_overwritten_and_source_remains(tmp_path, monkeypatch):
    out, target = fixture_archive(tmp_path, monkeypatch)
    stale = archive.ARCHIVE / "dev/a.json.zst"; stale.parent.mkdir(parents=True); stale.write_bytes(b"old failed attempt")
    with pytest.raises(FileExistsError): archive.run()
    assert stale.read_bytes() == b"old failed attempt"
    assert (target / "dev/a.json").exists()
    assert not (out / "audit/storage_legacy_cache_archive_v1/COMPLETE.json").exists()


def test_changed_protected_result_prevents_mutation(tmp_path, monkeypatch):
    out, target = fixture_archive(tmp_path, monkeypatch)
    stale = archive.ARCHIVE / "dev/a.json.zst"; stale.parent.mkdir(parents=True); stale.write_bytes(b"old")
    with pytest.raises(FileExistsError): archive.run()
    protected = archive.HISTORY / "outputs/N72R19R1/FINAL_RESULT.json"
    protected.write_text('{"changed":true}')
    with pytest.raises(AssertionError): archive.run()
    assert len(list(target.rglob("*.json"))) == 2


def test_source_signature_attests_uid_inode_size_and_mtime(tmp_path):
    path = tmp_path / "x.json"; path.write_bytes(b"x")
    signature = archive.file_signature(path)
    assert signature["uid"] == os.getuid() and signature["bytes"] == 1
    assert signature["inode"] == path.stat().st_ino
    assert signature["mtime_ns"] == path.stat().st_mtime_ns
