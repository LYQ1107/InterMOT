"""TRAIN-only sole-click preparation and common frozen identity-model lineage.

Historical diagnostic scenes only. Initial RGB crop, never GT-created future
candidate. Exact old anchors may be reused; missing clicked candidates fail.
"""
from pathlib import Path
import json
import numpy as np
import torch
from scripts.n72r21r1_common import ROOT, OUT, ASSETS, TRAIN, historical, read_json, write_json, sha256, storage
from scripts.n72r21r1_reproduce_v2 import checked_frames
from sam3_intermot.one_click.datasets import dancetrack_annotations, dancetrack_info
from sam3_intermot.evaluation.one_click_protocol import iou
from sam3_intermot.identity_probe.dataset import GTBox
from sam3_intermot.identity_probe.crop import load_person_crops
from sam3_intermot.identity_probe.encoders import OSNetEncoder
from sam3_intermot.one_click.acib_memory import ACIBMemoryNetwork


def run():
    torch.set_num_threads(1); storage(16 << 20)
    protocol_path = OUT / 'protocol/JOINT_STATE_CORPUS.json'; protocol = read_json(protocol_path)
    fit_path = Path(protocol['frozen_identity_fit_record']); fit = read_json(fit_path)
    cfg = fit['schema']['configuration']
    assert fit['completed'] and cfg['fit_sequences'] == protocol['identity_model_FIT_and_new_controller_diagnostic_FIT']
    assert cfg['inner_sequence'] == protocol['historical_INNER'] and cfg['outer_sequence'] == protocol['historical_outer_diagnostic_not_virgin']
    assert sha256(fit['best_checkpoint_path']) == fit['best_checkpoint_sha256']
    for path, digest in cfg['code_sha256'].items(): assert sha256(ROOT / path) == digest
    saved = torch.load(fit['best_checkpoint_path'], map_location='cpu', weights_only=True)
    assert saved['schema'] == fit['schema']
    network = ACIBMemoryNetwork(); network.load_state_dict(saved['model'], strict=True)
    model_source = {'path': fit['best_checkpoint_path'], 'sha256': fit['best_checkpoint_sha256'], 'fit_record_path': str(fit_path),
        'fit_record_sha256': sha256(fit_path), 'fit_sequences': cfg['fit_sequences'], 'inner_sequence': cfg['inner_sequence'], 'outer_sequence': cfg['outer_sequence'],
        'strict_loader_actual': True, 'frozen_before_new_controller_effects': True}
    old_input = read_json(historical('development/RUNTIME_INPUTS.json'))
    assert sha256(old_input['anchor_path']) == old_input['anchor_sha256']
    old_labels = {r['episode_uid']: r['target_gt_identity'] for r in read_json(historical('development/INITIALIZATION_TRUTH.json'))['labels']}
    old_anchors = np.load(old_input['anchor_path'], mmap_mode='r')
    checkpoint = ROOT.parent / 'InterMOT_N72R16_assets/checkpoints/osnet_x1_0_market1501.pth'
    assert sha256(checkpoint) == '2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154'
    encoder = None; events = []; truth = []; vectors = []
    sequences = sorted(cfg['fit_sequences'] + [cfg['inner_sequence'], cfg['outer_sequence']])
    for sequence in sequences:
        info = dancetrack_info(TRAIN / sequence); gt = dancetrack_annotations(TRAIN / sequence)
        candidates, index_sha = checked_frames(sequence); initial = {}
        for frame, rows in sorted(gt.items()):
            for row in sorted(rows, key=lambda r: r['identity']):
                x, y, right, bottom = row['box']
                if row['identity'] not in initial and right - x >= 16 and bottom - y >= 48:
                    initial[row['identity']] = (frame, row)
        selected = sorted(initial.items(), key=lambda x: (x[1][0], x[0]))[:3]
        for identity, (frame, row) in selected:
            old = next((e for e in old_input['inputs'] if e['sequence'] == sequence and e['frame'] == frame and old_labels[e['episode_uid']] == identity and e['box_xyxy'] == list(row['box'])), None)
            if old is not None:
                vector = np.array(old_anchors[old['anchor_index']], np.float32); origin = 'EXACT_OLD_SOLE_RGB_CROP_REUSED'
            else:
                if encoder is None: encoder = OSNetEncoder(checkpoint, 'cpu')
                x, y, right, bottom = row['box']; box = GTBox(sequence, frame + 1, identity, x, y, right - x, bottom - y, row['confidence'], 1, row['visibility'])
                crops = load_person_crops(TRAIN / sequence / 'img1' / f'{frame+1:08d}.jpg', [box])
                vector = encoder.encode(torch.stack(crops)).cpu().numpy()[0]; origin = 'NEW_SOLE_EVENT_FRAME_RGB_CROP_SAME_ENCODER'
            real_rows = candidates[frame][1]
            best = min(real_rows, key=lambda r: (-iou(row['box'], r['box_xyxy']), str(r['candidate_uid']))) if real_rows else None
            clicked_uid = str(best['candidate_uid']) if best is not None and iou(row['box'], best['box_xyxy']) >= .5 else None
            key = sequence + '__R1click' + str(len(events)).zfill(3)
            events.append({'sequence': sequence, 'recording_id': sequence, 'frame': frame, 'box_xyxy': list(row['box']), 'episode_uid': key,
                **info, 'anchor_index': len(vectors), 'clicked_candidate_uid': clicked_uid, 'initialization_failure': clicked_uid is None,
                'candidate_index_sha256': index_sha, 'initialization_source': origin, 'sole_click': True, 'runtime_future_GT_used': False,
                'model_source': model_source, 'diagnostic_role': 'FIT' if sequence in cfg['fit_sequences'] else 'INNER' if sequence == cfg['inner_sequence'] else 'HISTORICAL_OUTER_NOT_VIRGIN'})
            truth.append({'episode_uid': key, 'target_gt_identity': identity, 'offline_only': True}); vectors.append(vector)
        print(json.dumps({'joint_corpus_sole_clicks_prepared': sequence, 'actual_identities': len(selected)}), flush=True)
    path = ASSETS / 'corpus/sole_anchors.npy'; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle: np.save(handle, np.stack(vectors).astype(np.float32))
    write_json('corpus/RUNTIME_INPUTS.json', {'stage': 'N72R21R1', 'inputs': events, 'anchor_path': str(path), 'anchor_sha256': sha256(path),
        'encoder_sha256': sha256(checkpoint), 'protocol_sha256': sha256(protocol_path), 'preparation_source_sha256': sha256(Path(__file__)),
        'runtime_GT_identity_fields': False, 'frozen_identity_model': model_source, 'future_GT_not_crossing_runtime_boundary': True})
    write_json('corpus/INITIALIZATION_TRUTH.json', {'labels': truth, 'GT_future_used_as_runtime_input': False})


if __name__ == '__main__': run()
