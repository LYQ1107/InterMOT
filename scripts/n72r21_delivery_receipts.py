"""Read-only history/model/video checks and new, scoped delivery receipts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import cv2
from scripts.n72r21_common import ROOT, OUT, ASSETS, read_json, write_json, sha256, storage, utcnow


def history():
    source = OUT/'historical_audit/HISTORY_BEFORE.json'; record = read_json(source)
    changed = []; missing = []
    for relative, digest in record['sha256'].items():
        path = ROOT/relative
        if not path.is_file(): missing.append(relative)
        elif sha256(path) != digest: changed.append(relative)
    models = read_json(OUT/'historical_audit/REUSABLE_ASSETS.json')['models']
    model_checks = []
    for r in models:
        path = Path(r['path']); exists = path.is_file()
        model_checks.append({'path':str(path),'exists':exists,
                             'sha256_matches_original':exists and sha256(path)==r['sha256'],
                             'bytes_match_original':exists and path.stat().st_size==r['bytes']})
    result = {'utc':utcnow(),'original_history_manifest_sha256':sha256(source),
              'metadata_and_historical_code_checked':len(record['sha256']),
              'changed':changed,'missing':missing,'historical_model_checks':model_checks,
              'historical_model_count':len(models),
              'all_history_and_models_unchanged':not changed and not missing and all(r['sha256_matches_original'] and r['bytes_match_original'] for r in model_checks),
              'original_before_manifest_or_historical_files_written':False}
    write_json('historical_audit/HISTORY_DELIVERY_CHECK.json',result)
    if not result['all_history_and_models_unchanged']: raise ValueError('history immutability check failed')
    return result


def videos():
    manifests = ['visualizations/SMOKE_VIDEO_MANIFEST.json','visualizations/T2_FAILURE_VIDEO_MANIFEST.json','visualizations/MOT_FAILURE_VIDEO_MANIFEST.json']
    results = []
    for relative in manifests:
        record = read_json(OUT/relative)
        for r in record['clips']:
            path = Path(r['path'])
            if not path.resolve().is_relative_to((ASSETS/'visualizations').resolve()): raise ValueError('video outside own current assets')
            if path.stat().st_size != r['bytes'] or sha256(path)!=r['sha256']: raise ValueError('licensed local example changed')
            capture = cv2.VideoCapture(str(path)); fps = capture.get(cv2.CAP_PROP_FPS); decoded = 0
            width = None; height = None
            if not capture.isOpened(): raise ValueError('actual video cannot be opened')
            while True:
                ok, frame = capture.read()
                if not ok: break
                if width is None: height,width=frame.shape[:2]
                decoded += 1
            capture.release()
            expected = r.get('encoded_frames',r.get('frames_encoded'))
            original_fps = r.get('FPS',r.get('actual_fps'))
            if decoded != expected or abs(fps-original_fps)>1e-6: raise ValueError('actual encoded frame axis/FPS differs')
            preview = r.get('preview')
            if preview and sha256(preview['path']) != preview['sha256']: raise ValueError('inspection image changed')
            results.append({'manifest':relative,'manifest_sha256':sha256(OUT/relative),
                            'path':str(path),'sha256':r['sha256'],'bytes':r['bytes'],
                            'all_actual_frames_decoded':decoded,'FPS':fps,'actual_duration_seconds':decoded/fps,
                            'width':width,'height':height,'real_frames_not_generated_or_stitched':not r['synthetic_or_stitched_frames'],
                            'public_pixels_published':False})
    observations = {
        'MOT_FIRST_TARGET_REGRESSION':'Inspected preview: same real frame, sole-anchor inset, CLICK_C0 correct versus FULL wrong public target, labels readable.',
        'MOT_RAW_ANCHOR_FIRST_NONE_CONTROL':'Inspected preview: RAW_ANCHOR outputs true frozen NONE while CLICK_C0 claims; not a fabricated FULL NONE.',
        'MOT_FIRST_COLLATERAL_ID_REGRESSION':'Inspected preview: public target swap also affects orange collateral identity; not an isolated one-target trajectory.',
        'MOT_FIRST_STRICT_IDENTITY_RECOVERY':'Inspected preview: both C0 and FULL recover; no claim of improvement or physically verified occlusion.',
    }
    result = {'utc':utcnow(),'clips':results,'actual_video_count':len(results),
              'MOT_preview_observations_from_main_agent_visual_inspection':observations,
              'inspection_is_not_scientific_performance_or_license_expansion':True,
              'license_and_source': [f'outputs/N72R21/{p}' for p in manifests],
              'cross_recording_video_available':False,'new_SOT_inference_or_training':False,
              'raw_video_or_faces_uploaded_to_Git':False}
    write_json('visualizations/DELIVERY_VIDEO_RECEIPT.json',result)
    return result


def storage_receipt(final):
    result = storage()
    def cmd(args):
        r = subprocess.run(args,capture_output=True,text=True,timeout=45)
        return {'command':args,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    result.update(owner_uid=os.getuid(),own_assets_realpath=str(ASSETS.resolve()),
                  own_assets_disk_usage=cmd(['du','-sb','--',str(ASSETS)]),
                  inodes=cmd(['df','-i',str(ROOT)]),mount=cmd(['findmnt','-T',str(ROOT)]),
                  GPU_visibility_not_allocation=cmd(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader']),
                  deleted_material_this_delivery=[],new_dataset_or_environment_copy=False,
                  current_SOT_work='DEFERRED_BY_USER',new_heavy_download=False,
                  final_snapshot=final,not_a_claim_that_all_mount_occupancy_is_this_project=True)
    write_json('storage/STORAGE_AFTER.json' if final else 'storage/DELIVERY_STORAGE_SNAPSHOT.json',result)
    return result


def run(final=False):
    if final:
        result = read_json(OUT/'FINAL_RESULT.json')
        if result['complete_frozen_VAL_scene_cases'] != 375: raise ValueError('final storage requires completed frozen evidence')
    h = history(); v = videos(); s = storage_receipt(final)
    print(json.dumps({'history_checked':h['metadata_and_historical_code_checked'],'old_models_checked':h['historical_model_count'],
                      'all_unchanged':h['all_history_and_models_unchanged'],'actual_readable_videos':v['actual_video_count'],
                      'free_GiB':s['free_gib'],'final_snapshot':final}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args();run(args.final)
