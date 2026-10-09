"""Versioned local fixture repair: zstd refuses symlinks without --force.

Keep its CLI and historical reader unchanged. Hardlinks reuse the same inode
on this mount; neither tape bytes nor historical output are copied/rewritten.
"""
from scripts.n72r21r1_common import ROOT, OUT, HISTORY, write_json, sha256


def run():
    records = []
    for name in ('oracle_clean_features.jsonl.zst', 'frame_runtime_features.jsonl.zst'):
        relative = 'outputs/N72R20R3/presence/' + name
        source, target = HISTORY / relative, ROOT / relative
        backup = OUT / 'tests/initial_fixture_symlinks' / name
        assert source.is_file() and source.stat().st_dev == target.parent.stat().st_dev
        assert target.is_symlink() and target.resolve() == source.resolve()
        if backup.exists() or backup.is_symlink(): raise FileExistsError('preserve old fixture-link evidence')
        backup.parent.mkdir(parents=True, exist_ok=True)
        target.rename(backup); target.hardlink_to(source)
        assert target.stat().st_ino == source.stat().st_ino and sha256(target) == sha256(source)
        records.append({'source': str(source), 'target': str(target), 'source_SHA': sha256(source),
            'same_inode_no_duplicate_tape_bytes': True, 'initial_symlink_preserved': str(backup)})
    write_json('tests/ZSTD_HARDLINK_FIXTURE_REPAIR.json', {'status': 'LOCAL_ONLY_DEPENDENCY_REUSE_REPAIR', 'records': records,
        'third_party_or_test_reader_modified': False, 'old_outputs_content_modified': False,
        'test_attempt_14fail_and7fail_preserved': True, 'files_deleted': 0})


if __name__ == '__main__': run()
