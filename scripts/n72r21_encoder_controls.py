"""Frozen actual person-ReID/general controls on the identical SAM3 UID axis.

No future GT is read by preparation or extraction. Cached embeddings are
stateless current-crop inference, not an oracle identity memory or a new fit.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.io import ImageReadMode, read_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize
from scripts.n72r21_common import ROOT, ASSETS, OUT, read_json, write_json, sha256, storage
from scripts.n72r21_sot import checked_device
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames
from scripts.n72r21_baselines_geometry_repair import valid_geometry

NAMES = ('SECOND_REID_MARKET_RESNET50', 'GENERAL_IMAGENET_RESNET50_MATCHED', 'GENERAL_IMAGENET_RESNET50_NATIVE')
GENERAL_SHA = '0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a'


def roi_tensor(image, box):
    """Exact historical rounding/clipping, including disclosed empty RGB fallback."""
    _, height, width = image.shape
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    empty = x2 <= x1 or y2 <= y1
    return (torch.zeros((3, 8, 8), dtype=torch.uint8) if empty else image[:, y1:y2, x1:x2]), empty


def matched_tensor(crop):
    value = resize(crop.float()/255., [256, 128], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return normalize(value, [.485, .456, .406], [.229, .224, .225])


def models(device):
    record = read_json(OUT/'baselines/SECOND_REID_CHECKPOINT_MANIFEST.json')
    if sha256(record['destination']) != record['sha256']: raise ValueError('second ReID weight SHA')
    for p, expected in record['source_files'].items():
        if sha256(p) != expected: raise ValueError('official source SHA')
    source = ASSETS/'official_sources/Torchreid/source/resnet.py'
    spec = importlib.util.spec_from_file_location('n72r21_official_torchreid_resnet', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # The official legacy checkpoint stores evaluation statistics as NumPy
    # scalars. Keep the restricted weights-only loader, allowing only these
    # known numeric primitives locally, never arbitrary pickle globals.
    numeric_globals = [np.core.multiarray.scalar, np.dtype, type(np.dtype('float64')), type(np.dtype('float32'))]
    with torch.serialization.safe_globals(numeric_globals):
        checkpoint = torch.load(record['destination'], map_location='cpu', weights_only=True, encoding='latin1')
    state = checkpoint.get('state_dict', checkpoint)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    if state['classifier.weight'].shape != (751, 2048): raise ValueError('official Market classifier schema')
    person = module.resnet50(num_classes=751, pretrained=False)
    person.load_state_dict(state, strict=True)
    general_path = ROOT.parent/'.cache/torch/hub/checkpoints/resnet50-0676ba61.pth'
    if sha256(general_path) != GENERAL_SHA: raise ValueError('existing general weight SHA')
    general = resnet50(weights=None)
    general.load_state_dict(torch.load(general_path, map_location='cpu', weights_only=True), strict=True)
    general.fc = torch.nn.Identity()
    for model in (person, general):
        model.eval().to(device)
        for parameter in model.parameters(): parameter.requires_grad_(False)
    return {NAMES[0]: person, NAMES[1]: general, NAMES[2]: general}, {
        'second_ReID': {'checkpoint_sha256': record['sha256'], 'checkpoint_bytes': record['bytes'],
                        'source_code_sha256': sha256(source), 'feature_parameters': sum(p.numel() for n,p in person.named_parameters() if not n.startswith('classifier.')),
                        'total_parameters_with_unused_classifier': sum(p.numel() for p in person.parameters()),
                        'source_pretraining': 'ImageNet initialization then Market1501 identity softmax (official model zoo)',
                        'strict_full_model_load': True, 'weights_only_deserialization': True,
                        'legacy_numeric_allowlist': ['numpy.core.multiarray.scalar','numpy.dtype','Float64DType','Float32DType'],
                        'legacy_Python2_string_decoding': 'latin1',
                        'arbitrary_pickle_loader_enabled': False},
        'general': {'checkpoint_path': str(general_path), 'checkpoint_sha256': GENERAL_SHA,
                    'official_url': ResNet50_Weights.IMAGENET1K_V1.url, 'existing_cache_reused_not_copied': True,
                    'feature_parameters': sum(p.numel() for p in general.parameters()), 'source_pretraining': 'ImageNet1K V1',
                    'strict_full_model_load_before_classifier_removal': True,
                    'source_code_license': 'Torchvision BSD-3-Clause', 'weight_republication': False}}


def freeze():
    torch.set_num_threads(1)
    all_models, lineage = models(torch.device('cpu'))
    inputs = read_json(OUT/'development/RUNTIME_INPUTS.json')
    event = inputs['inputs'][0]
    path = ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/event['sequence']/'img1'/f"{event['frame']+1:08d}.jpg"
    crop, empty = roi_tensor(read_image(str(path), mode=ImageReadMode.RGB), event['box_xyxy'])
    actual = {}
    with torch.inference_mode():
        for name, model in all_models.items():
            tensor = ResNet50_Weights.IMAGENET1K_V1.transforms()(crop) if name == NAMES[2] else matched_tensor(crop)
            vector = F.normalize(model(tensor[None]).float(), dim=1)
            if vector.shape != (1, 2048) or not torch.isfinite(vector).all(): raise ValueError('real loader smoke')
            actual[name] = {'actual_real_image_output_shape': list(vector.shape), 'finite': True,
                            'vector_sha256': hashlib.sha256(vector.numpy().tobytes()).hexdigest()}
    write_json('baselines/ENCODER_CONTROLS_LOADER_SMOKE.json', {
        'status': 'PASS_STRICT_LOAD_AND_REAL_SINGLE_CLICK_CROP_FORWARD', 'lineage': lineage,
        'actual_outputs': actual, 'real_image_path': str(path), 'real_image_sha256': sha256(path),
        'one_click_input_sha256': sha256(OUT/'development/RUNTIME_INPUTS.json'), 'crop_empty': empty,
        'new_training': False, 'scientific_metric': False})
    protocol = {
        'status': 'FROZEN_BEFORE_ENCODER_CONTROL_FEATURE_EXTRACTION_OR_COMPARISON',
        'final_goal_file': 'outputs/N72R21/FINAL_GOAL.json', 'sequences': read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')['sequences'],
        'encoders': list(NAMES), 'lineage': lineage, 'candidate_axis': 'Identical valid positive-area finite boxes/UIDs as versioned B0-B5 geometry repair',
        'matched_preprocessing': 'Same original RGB ROI; historical round+clip; tensor bilinear antialias resize 256x128; ImageNet mean/std; no augmentation/context',
        'native_general_preprocessing': 'Same original RGB ROI then official Torchvision IMAGENET1K_V1 resize 256, center crop224 and normalization',
        'reid_preprocessing_caveat': 'Resolution/normalization match official 256x128; tensor antialiased resize harmonized with existing OSNet, not silently asserted PIL-identical',
        'empty_clipped_ROI_policy': 'Preserve UID with disclosed historical black8x8 fallback; never select/drop it from future GT',
        'embedding': 'FP32 model forward, L2 normalized, float16 storage then renormalize for scoring',
        'exactly_one_actual_clicked_crop_anchor_per_target': True, 'no_oracle_memory': True,
        'training_or_threshold_fit': False, 'future_GT_during_extraction': False,
        'competitive_negative_truth': 'Evaluation only; deterministic one-to-one IoU>=0.5 verified other identity; unresolved not verified negatives',
        'missing_positive_candidate_not_counted_as_discrimination_failure': True,
        'all_visible_localization_and_NONE_are_separate_endpoints': True,
        'no_deployment_threshold_reselected_from_outer_results': True, 'not_cross_recording_final_science': True,
        'feature_cache_limit_bytes': 2 << 30, 'expected_total_new_feature_bytes': 49189*2048*3*2,
        'one_GPU_job_only': True, 'GPU_allocation_cap_GiB': 12, 'max_seconds_per_sequence': 1800,
        'code_sha256': {str(ROOT/'scripts/n72r21_encoder_controls.py'): sha256(ROOT/'scripts/n72r21_encoder_controls.py')}}
    destination = OUT/'protocol/FROZEN_ENCODER_CONTROLS.json'
    if destination.exists() and read_json(destination) != protocol: raise ValueError('frozen protocol differs; no overwrite')
    write_json('protocol/FROZEN_ENCODER_CONTROLS.json', protocol)
    print(json.dumps({'encoder_loader': 'PASS', 'protocol_frozen': True, 'parameters': lineage}), flush=True)


def extract(sequences, gpu):
    torch.set_num_threads(1)
    protocol = read_json(OUT/'protocol/FROZEN_ENCODER_CONTROLS.json')
    if not set(sequences) <= set(protocol['sequences']): raise ValueError('frozen TRAIN-only scope')
    for p, expected in protocol['code_sha256'].items():
        if sha256(p) != expected: raise ValueError('frozen extraction code changed')
    storage(protocol['expected_total_new_feature_bytes'])
    device = checked_device(gpu); all_models, lineage = models(device)
    if lineage != protocol['lineage']: raise ValueError('model lineage changed')
    inputs = read_json(OUT/'development/RUNTIME_INPUTS.json')['inputs']
    transform = ResNet50_Weights.IMAGENET1K_V1.transforms()
    for sequence in sequences:
        done = OUT/'baselines/encoder_controls/seals'/f'{sequence}.json'
        if done.exists():
            seal = read_json(done)
            if seal['protocol_sha256'] != sha256(OUT/'protocol/FROZEN_ENCODER_CONTROLS.json'): raise ValueError('protocol changed')
            for r in seal['artifacts']:
                if sha256(r['path']) != r['sha256']: raise ValueError('cached feature changed')
            print({'verified_reused_encoder_sequence': sequence}, flush=True); continue
        started = time.monotonic(); source = ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/sequence
        frames = [(p,[r for r in rows if valid_geometry(r)]) for p,rows in load_candidate_frames(ROOT.parent/'InterMOT_N72R20R2_assets', sequence)]
        events = [e for e in inputs if e['sequence'] == sequence]
        count = sum(len(rows) for p,rows in frames)
        directory = ASSETS/'encoder_controls'/sequence; directory.mkdir(parents=True, exist_ok=True)
        arrays = {}; anchor_arrays = {}; paths = []
        for name in NAMES:
            p = directory/f'{name}.npy'; a = directory/f'{name}.anchors.npy'
            if p.exists() or a.exists(): raise FileExistsError('preserve unsealed partial feature files')
            arrays[name] = np.lib.format.open_memmap(p, mode='w+', dtype=np.float16, shape=(count,2048))
            anchor_arrays[name] = np.lib.format.open_memmap(a, mode='w+', dtype=np.float32, shape=(len(events),2048))
            paths.extend([p,a])
        axis = []; offset = 0; empty_ROIs = []; image_digest = hashlib.sha256()
        for payload, rows in frames:
            frame = int(payload['frame']); anchors = [(i,e) for i,e in enumerate(events) if e['frame'] == frame]
            path = source/'img1'/f'{frame+1:08d}.jpg'
            image_digest.update(f'{frame}:'.encode()); image_digest.update(bytes.fromhex(sha256(path)))
            image = read_image(str(path), mode=ImageReadMode.RGB)
            crops = []
            for r in rows:
                crop, empty = roi_tensor(image,r['box_xyxy']); crops.append(crop)
                if empty: empty_ROIs.append({'frame': frame, 'candidate_uid': str(r['candidate_uid'])})
            for i,e in anchors:
                crop, empty = roi_tensor(image,e['box_xyxy'])
                if empty: raise ValueError('invalid actual clicked ROI')
                crops.append(crop)
            for start in range(0,len(crops),16):
                current = crops[start:start+16]
                matched = torch.stack([matched_tensor(c) for c in current]).to(device)
                native = torch.stack([transform(c) for c in current]).to(device)
                with torch.inference_mode():
                    for name,model in all_models.items():
                        vectors = F.normalize(model(native if name == NAMES[2] else matched).float(),dim=1).cpu().numpy()
                        if not np.isfinite(vectors).all(): raise ValueError('nonfinite actual features')
                        for j,v in enumerate(vectors):
                            index = start+j
                            if index < len(rows): arrays[name][offset+index] = v
                            else: anchor_arrays[name][anchors[index-len(rows)][0]] = v
            axis.append({'frame': frame, 'offset': offset, 'count': len(rows), 'UIDs': [str(r['candidate_uid']) for r in rows]})
            offset += len(rows)
            if frame % 200 == 0: print(json.dumps({'encoder_sequence': sequence, 'frame': frame, 'features_per_encoder': offset, 'seconds': round(time.monotonic()-started,1)}),flush=True)
            if time.monotonic()-started > protocol['max_seconds_per_sequence']: raise RuntimeError('resource wall limit; preserve partial')
        for array in list(arrays.values())+list(anchor_arrays.values()): array.flush()
        del arrays, anchor_arrays
        seal = {'sequence': sequence, 'protocol_sha256': sha256(OUT/'protocol/FROZEN_ENCODER_CONTROLS.json'),
                'one_click_inputs_sha256': sha256(OUT/'development/RUNTIME_INPUTS.json'), 'candidate_count': count,
                'candidate_axis': axis, 'episodes_in_anchor_order': [e['episode_uid'] for e in events],
                'empty_clipped_ROIs': empty_ROIs, 'original_image_stream_sha256': image_digest.hexdigest(),
                'source_candidate_index_sha256': sha256(ROOT.parent/'InterMOT_N72R20R2_assets/candidates'/sequence/'index.json'),
                'artifacts': [{'path':str(p),'sha256':sha256(p),'bytes':p.stat().st_size} for p in paths],
                'future_GT_read': False, 'stateless_current_frame_features': True, 'runtime_device': str(device),
                'model_forward_FP32': True, 'seconds': time.monotonic()-started,
                'peak_GPU_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0}
        write_json(f'baselines/encoder_controls/seals/{sequence}.json',seal)
        print(json.dumps({'sealed_encoder_sequence':sequence,'candidates':count,'empty_clipped_ROIs':len(empty_ROIs)}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze','extract'])
    parser.add_argument('--gpu',type=int); parser.add_argument('--sequences',nargs='+')
    args = parser.parse_args()
    if args.action == 'freeze': freeze()
    else: extract(args.sequences or read_json(OUT/'protocol/FROZEN_ENCODER_CONTROLS.json')['sequences'],args.gpu)
