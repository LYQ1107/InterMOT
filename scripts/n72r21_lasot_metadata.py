"""Small official LaSOT metadata only; no video/archive acquisition.

The existing local TAO TRAIN windows are not full original LaSOT clips. The
official toolkit box files can establish pixel-frame alignment, but neither
box files nor sparse TAO labels prove target absence or source-video FPS.
"""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from scripts.n72r21_common import ROOT, OUT, ASSETS, read_json, write_json, sha256, storage, utcnow
from scripts.n72r21_metadata import fetch


def acquire(path):
    destination=ASSETS/'official_sources/LaSOT'/path
    raw=f'https://raw.githubusercontent.com/HengLan/LaSOT_Evaluation_Toolkit/master/{path}'
    record=fetch(raw,destination,max_bytes=1<<20)
    if record['status']!='METADATA_NETWORK_OR_ACCESS_BLOCKED':return record
    # Direct-first HTTP failures are preserved. This existing authenticated
    # GitHub transport retrieves public metadata, not restricted pixels.
    for attempt in range(3):
        try:
            r=subprocess.run(['gh','api',f'repos/HengLan/LaSOT_Evaluation_Toolkit/contents/{path}?ref=master'],
                             capture_output=True,text=True,timeout=18)
        except subprocess.TimeoutExpired:
            record['attempts'].append({'transport':'EXISTING_GH_API','attempt':attempt+1,'status':'TIMEOUT_18_SECONDS'})
            continue
        if r.returncode:
            record['attempts'].append({'transport':'EXISTING_GH_API','attempt':attempt+1,'status':'FAILED','error':r.stderr.strip()[:200]})
            continue
        try:
            content=json.loads(r.stdout)
            if content.get('encoding')!='base64' or content['size']>1<<20:raise ValueError('not bounded metadata')
            body=base64.b64decode(content['content'],validate=False)
            actual=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
            if len(body)!=content['size'] or actual!=content['sha']:raise ValueError('official Git blob verification')
            storage(len(body));destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as handle:handle.write(body)
        except (ValueError,KeyError,json.JSONDecodeError) as e:
            record['attempts'].append({'transport':'EXISTING_GH_API','attempt':attempt+1,'status':'INVALID_METADATA','error':str(e)[:200]})
            continue
        record.update({'status':'DOWNLOADED_VERIFIED_OFFICIAL_GIT_BLOB_METADATA','bytes':len(body),
                       'sha256':sha256(destination),'official_git_blob_sha1':actual,'utc':utcnow()})
        return record
    return record


def run():
    result={'utc':utcnow(),'source':'https://github.com/HengLan/LaSOT_Evaluation_Toolkit',
            'media_downloaded':False,'archive_downloaded':False,'code_license_is_not_pixel_license':True,
            'metadata':[],'local_train_windows':[],'original_full_clip_claim':False}
    for name in ('README.md','LICENSE','annos/person-2.txt','annos/person-3.txt','annos/person-7.txt'):
        record=acquire(name);result['metadata'].append(record)
        write_json('datasets/LASOT_OFFICIAL_METADATA.json',result)
        print(json.dumps({'source_path':name,'status':record['status']}),flush=True)
    tao_path=Path('/data3/liuyeqiang/TAO-Amodal/annotations/train.json')
    tao=json.loads(tao_path.read_text())
    for name in ('person-2','person-3','person-7'):
        directory=Path('/data3/liuyeqiang/TAO-Amodal/frames/train/LaSOT')/name
        filenames=sorted(directory.glob('*.jpg'));numbers=[int(p.stem) for p in filenames]
        ann=ASSETS/'official_sources/LaSOT/annos'/f'{name}.txt'
        video=next(v for v in tao['videos'] if v['name']==f'train/LaSOT/{name}')
        images=[i for i in tao['images'] if i['video_id']==video['id']]
        row={'name':name,'root':str(directory),'original_split':video['metadata']['original_dataset_split'],
             'frames':len(numbers),'first_original_frame_1based':min(numbers),'last_original_frame_1based':max(numbers),
             'contiguous_pixel_window':numbers==list(range(min(numbers),max(numbers)+1)),
             'tao_sparse_annotation_frames':len(images),'tao_metadata_source_sha256':sha256(tao_path),
             'original_FPS_verified':False,'absence_visibility_labels_available':False,'complete_person_labels':False,
             'do_not_treat_missing_TAO_annotation_as_absence':True,'pixels_copied':False}
        if ann.exists():
            boxes=np.loadtxt(ann,delimiter=',')
            if boxes.ndim!=2 or boxes.shape[1]!=4 or not np.isfinite(boxes).all():raise ValueError('invalid official box file')
            row.update({'official_original_box_frames':len(boxes),'all_window_frames_have_official_boxes':max(numbers)<=len(boxes),
                        'official_annotation_sha256':sha256(ann),'all_boxes_positive_size':bool((boxes[:,2:]>0).all())})
            # Verify the TAO adapter retains original image numbering; do not
            # silently resample 1fps annotations as 30fps continuous truth.
            row['TAO_original_frame_number_alignment']=all(int(Path(i['file_name']).stem)==i['frame_index']+1 for i in images)
        else:row['official_original_boxes_available']=False
        result['local_train_windows'].append(row)
    write_json('datasets/LASOT_OFFICIAL_METADATA.json',result)
    print(json.dumps({'train_windows':len(result['local_train_windows']),'media_downloaded':False}),flush=True)


if __name__=='__main__':run()
