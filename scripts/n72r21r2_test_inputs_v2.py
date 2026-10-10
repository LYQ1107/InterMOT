"""Versioned local fixture-link compatibility; no data copies or old writes.

The pinned zstd CLI refuses symlinks. Replace only our two newly created local
fixture symlinks with same-filesystem hardlinks, preserving all source bytes.
"""
import os
from pathlib import Path
from scripts.n72r21r2_common import ROOT, OUT, R1, read_json, write_json, sha256


def run():
    previous = read_json(OUT / "tests/LOCAL_HISTORICAL_FIXTURE_REUSE.json")
    records = []
    for record in previous["records"]:
        if not record["relative_path"].endswith(".jsonl.zst"):
            continue
        source = Path(record["source_path"])
        target = ROOT / record["relative_path"]
        assert source.is_file() and sha256(source) == record["sha256"]
        assert target.is_symlink() and target.resolve() == source.resolve()
        assert source.stat().st_dev == target.parent.stat().st_dev
        old_destination = os.readlink(target)
        temporary = target.with_name(target.name + ".hardlink_v2")
        assert not temporary.exists()
        os.link(source, temporary)
        temporary.replace(target)
        assert not target.is_symlink() and os.path.samefile(target, source)
        assert sha256(target) == record["sha256"]
        records.append({**record, "old_local_symlink_destination": old_destination, "symlink_replaced_by_verified_same_filesystem_hardlink": True,
                        "source_bytes_unchanged": True, "copied_bytes": 0})
    write_json("tests/LOCAL_FIXTURE_HARDLINK_RECOVERY_V2.json", {"stage": "N72R21R2", "records": records,
               "reason": "zstd refuses symlink input unless forced; historical source/tests/CLI remain unmodified",
               "only_local_new_fixture_symlinks_replaced": True, "old_symlinks_can_be_recreated_from_record": True})


if __name__ == "__main__":
    run()
