"""Minimal local-only regression assets; no copies/uploads or old writes."""
from scripts.n72r21r1_common import ROOT, HISTORY, write_json, sha256


def run():
    paths = ['outputs/N72R18/FINAL_GOAL.json', 'outputs/N72R19/FINAL_GOAL.json',
        'outputs/N72R18/checkpoints/identity_memory_gru.pt',
        'outputs/N72R20R3/presence/oracle_clean_features.jsonl.zst',
        'outputs/N72R20R3/presence/frame_runtime_features.jsonl.zst']
    records = []
    for relative in paths:
        source, target = HISTORY / relative, ROOT / relative
        assert source.is_file()
        if target.is_symlink(): assert target.resolve() == source.resolve()
        elif target.exists(): raise FileExistsError('never overwrite local fixture or new project output')
        else:
            target.parent.mkdir(parents=True, exist_ok=True); target.symlink_to(source)
        records.append({'relative_path': relative, 'source_path': str(source), 'sha256': sha256(source),
            'source_bytes': source.stat().st_size, 'reused_by_local_symlink': True})
    write_json('tests/LOCAL_HISTORICAL_FIXTURE_REUSE.json', {'status': 'LOCAL_TEST_INPUTS_LINKED_NOT_PUBLISHED',
        'records': records, 'files_copied': 0, 'historical_results_modified': False,
        'initial_regression_failures_preserved': 'outputs/N72R21R1/tests/FULL_REGRESSION_INITIAL.xml',
        'historical_branch_and_TrackEval_CLI_failures_not_fixed_or_hidden': True})


if __name__ == '__main__': run()
