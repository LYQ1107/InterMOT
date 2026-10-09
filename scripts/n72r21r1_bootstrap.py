"""Audit real local R21 evidence and freeze fresh TRAIN groups, no fitting."""
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.n72r21r1_common import ROOT, OUT, HISTORY, CANDIDATES, TRAIN, historical, read_json, write_json, sha256, storage, update_status, utcnow
from sam3_intermot.one_click.datasets import dancetrack_info


def audit_history():
    result_path = historical('mot_pilot/RESULT.json')
    result = read_json(result_path)
    verification = read_json(historical('mot_pilot/RESULT_VERIFICATION.json'))
    assert verification['RESULT_sha256'] == sha256(result_path)
    assert result['TrackEval_returncode'] == 0 and result['next_stage_authorized'] is False
    verified = []
    models = {}
    snapshot = {}
    for key, digest in result['runtime_seal_sha256'].items():
        seal_path = historical('mot_pilot/runtime_seals/' + key + '.json')
        assert sha256(seal_path) == digest
        seal = read_json(seal_path)
        for path, h in seal['code_sha256'].items():
            assert sha256(HISTORY / path) == h and sha256(ROOT / path) == h
            snapshot[str(HISTORY / path)] = h
        for artifact in seal['artifacts']:
            assert sha256(artifact['path']) == artifact['sha256']
            snapshot[artifact['path']] = artifact['sha256']
        model = seal['source_model']
        if model:
            assert sha256(model['path']) == model['sha256']
            assert sha256(model['fit_record_path']) == model['fit_record_sha256']
            models[model['path']] = model
        verified.append(key)
        snapshot[str(seal_path)] = digest
    assert len(verified) == 32 and len(models) == 6
    table = read_json(historical('evaluation/TABLES_A_TO_E.json'))
    assert set(table['tables']) == {f'Table_{s}' for s in 'ABCDE'}
    for record in table['source_evidence']:
        assert sha256(HISTORY / record['file']) == record['sha256']
    catalog_path = historical('checkpoints/SHA256_MANIFEST.json')
    catalog = read_json(catalog_path)
    weights = catalog['fitted_checkpoints']
    assert len(weights) == 248 and len(catalog['fits']) == 124
    for record in weights:
        assert sha256(record['path']) == record['sha256']
        snapshot[record['path']] = record['sha256']
    exposed = set()
    for record in catalog['fits']:
        exposed.update(record['fit_sequences'])
        exposed.add(record['inner_sequence'])
        exposed.add(record['outer_sequence'])
    for path in [result_path, historical('FINAL_GOAL.json'), historical('RESEARCH_DIRECTION_OVERRIDE.json'), historical('FINAL_RESULT.json'), historical('mot_pilot/RESULT_VERIFICATION.json'), catalog_path, historical('evaluation/TABLES_A_TO_E.json')]:
        snapshot[str(path)] = sha256(path)
    write_json('source_audit/HISTORY_SHA_MANIFEST.json', {'files': snapshot, 'historical_root': str(HISTORY), 'all_present_SHA_verified': True})
    write_json('source_audit/EXISTING_CHECKPOINTS.json', {'stage': 'N72R21R1', 'actual_old_checkpoint_files_SHA_reverified': len(weights), 'actual_old_fits': 124, 'eligible_old_fits': 120, 'archived_original_fits': 4, 'original_catalog': str(catalog_path), 'catalog_sha256': sha256(catalog_path), 'pilot_models': models, 'new_strict_loader_run': False, 'prior_strict_load_catalog_binds_unchanged_weights': True, 'no_weights_copied': True})
    write_json('source_audit/BASELINE_SOURCE_VERIFICATION.json', {'old_scene_cases_verified': verified, 'old_pilot_models_verified': 6, 'result_sha256': sha256(result_path), 'TrackEval_commit': result['TrackEval_commit'], 'TrackEval_returncode': 0, 'original_next_stage_authorized': False, 'new_reproduction_completed': False, 'old_result_not_new_experiment': True, 'historically_exposed_sequences_actual_R21_fit_records': sorted(exposed), 'hypothesis_not_yet_counterfactually_proven': 'Old bridge executes feasible identity proposals without independent benefit/risk authority.'})
    return exposed


def audit_train(exposed):
    prereg = read_json(OUT / 'PREREGISTRATION.json')
    rows = []
    for directory in sorted(TRAIN.glob('dancetrack*')):
        if not directory.is_dir():
            continue
        info = dancetrack_info(directory)
        images = sorted(p.name for p in (directory / 'img1').glob('*.jpg'))
        if images != [f'{i:08d}.jpg' for i in range(1, info['frames'] + 1)]:
            raise ValueError('Incomplete original frame axis: ' + directory.name)
        gt = directory / 'gt/gt.txt'
        if not gt.is_file() or not gt.stat().st_size:
            raise ValueError('Missing TRAIN GT: ' + directory.name)
        index_path = CANDIDATES / 'candidates' / directory.name / 'index.json'
        index = read_json(index_path) if index_path.is_file() else None
        if index and (index.get('runtime_gt_read') or index.get('runtime_future_gt_used')):
            raise ValueError('Unlawful candidate lineage')
        row = {'sequence': directory.name, 'root': str(directory), **info, 'duration_seconds': info['frames'] / info['fps'], 'GT_sha256': sha256(gt), 'all_images_present': True, 'images': len(images), 'image_bytes': sum((directory / 'img1' / n).stat().st_size for n in images), 'historically_exposed_R21': directory.name in exposed, 'candidate_index': str(index_path) if index else None, 'candidate_index_sha256': sha256(index_path) if index else None, 'existing_candidate_frames': index['frame_count'] if index else None, 'existing_candidate_embedding_count': index.get('embedding_count') if index else None, 'additional_candidate_preparation_required': index is None}
        rows.append(row)
    if len(rows) != 40:
        raise ValueError('Exactly40 actual TRAIN sequences required before split freeze')
    historical_dev = set(prereg['historical_development_sequences'])
    assert exposed <= historical_dev
    eligible = sorted([r['sequence'] for r in rows if r['sequence'] not in historical_dev], key=lambda s: hashlib.sha256(('N72R21R1_SPLIT_V1:' + s).encode()).hexdigest())
    assert len(eligible) == 32
    split = {'confirmation': eligible[:8], 'inner': eligible[8:16], 'fit': eligible[16:], 'historical_development': sorted(historical_dev)}
    assert not set(split['confirmation']) & set(split['fit'] + split['inner'])
    write_json('data/TRAIN_ASSET_AUDIT.json', {'stage': 'N72R21R1', 'root': str(TRAIN), 'actual_sequences': 40, 'actual_frames': sum(r['frames'] for r in rows), 'sequences': rows, 'new_downloads_or_copies': False, 'candidate_quality_or_new_model_effects_not_used_for_split': True})
    write_json('data/NEW_SEQUENCE_SPLIT.json', {'stage': 'N72R21R1', 'status': 'FROZEN_BEFORE_NEW_R1_MODEL_EFFECTS', 'preregistration_sha256': sha256(OUT / 'PREREGISTRATION.json'), 'split': split, 'rule': prereg['new_split_rule'], 'global_historical_virginity_claim': False, 'exposure_scope': 'Actual R21 fit/inner/outer records and explicitly historical8. No universal across-project, backbone or person virginity claim.', 'confirmation_policy_runs_completed': 0, 'new_fit_or_runtime_seen_before_freeze': False})
    write_json('data/DENSITY_GROUPS.json', {'definition': 'Current valid real candidate count; same GT-free geometry filter as frozen pilot', 'sparse': [0, 4], 'medium': [5, 8], 'crowded': [9, None], 'all_strata_reported': True, 'boundaries_selected_by_new_model_effect': False})
    return rows, split


def run():
    storage(512 << 20)
    branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
    assert branch == 'codex/n72r21r1-safe-joint-mot-intervention'
    exposed = audit_history()
    rows, split = audit_train(exposed)
    update_status(source_history_audit_complete=True, original_scene_cases_SHA_verified=32, original_checkpoint_files_SHA_verified=248, actual_TRAIN_sequences=40, independent_split_frozen=True, confirmation_sequences=split['confirmation'], status='ACTIVE_PHASE_A_BASELINE_REPRODUCTION_AND_FIRST_DIVERGENCE', resource_snapshot=storage())
    print(json.dumps({'old_rollout_seals_verified': 32, 'old_checkpoints_SHA_verified': 248, 'actual_TRAIN_sequences': len(rows), 'split_sizes': {k: len(v) for k, v in split.items()}, 'existing_candidate_train_sequences': sum(r['candidate_index'] is not None for r in rows), 'no_new_fitting': True, 'new_reproduction_not_yet_complete': True}), flush=True)


if __name__ == '__main__':
    run()
