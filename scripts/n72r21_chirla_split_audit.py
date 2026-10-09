"""All official public CSV membership; no image/video input or split tuning."""
import csv
from datetime import datetime
import json
from pathlib import PurePosixPath
import re
from scripts.n72r21_common import OUT,ASSETS,read_json,write_json,sha256


def run():
    audit=read_json(OUT/'datasets/CHIRLA_METADATA_AUDIT.json')
    if audit['missing_official_CSVs']:raise ValueError('complete official metadata prerequisite')
    source=ASSETS/'official_sources/CHIRLA/benchmark/metadata';tables={}
    for path in sorted(source.glob('*.csv')):
        with path.open() as stream:tables[path.stem]=list(csv.DictReader(stream))
    manifests={};matrix={}
    keys={name:{r['image_path'] for r in rows} for name,rows in tables.items()}
    def frame_key(row):
        match=re.fullmatch(r'frame_([0-9]+)',PurePosixPath(row['image_path']).stem)
        if not match:raise ValueError('unsupported authoritative crop frame naming; no guessed join')
        return row['sequence'],row['camera'],int(row['id']),int(match.group(1))
    semantic={name:{frame_key(r) for r in rows} for name,rows in tables.items()}
    recordings={name:{(r['sequence'],r['camera']) for r in rows} for name,rows in tables.items()}
    for name,rows in tables.items():
        records=sorted({(r['sequence'],r['camera']) for r in rows});parsed=[]
        for sequence,camera in records:
            pieces=camera.split('_',2)
            timestamp=datetime.strptime(pieces[2],'%Y-%m-%d-%H:%M:%S')
            parsed.append({'sequence':sequence,'recording_id':camera,'camera_number':int(pieces[1]),
                           'timestamp_from_authoritative_metadata':timestamp.isoformat(),
                           'timestamp_not_measured_video_PTS':True,'actual_video_FPS_verified':False})
        manifests[name]={'path':str(source/f'{name}.csv'),'SHA256':sha256(source/f'{name}.csv'),'rows':len(rows),
                         'unique_crop_paths':len(keys[name]),'IDs':sorted({int(r['id']) for r in rows}),
                         'subsets':sorted({r.get('subset','TRACKING_NO_SUBSET') for r in rows}),
                         'records':parsed,'media_or_features_accessed':False}
        matrix[name]={other:len(keys[name]&values) for other,values in keys.items() if other!=name}
    protocols={}
    for scenario in ['long_term','multi_camera','multi_camera_long_term','reappearance']:
        train=f'reid_{scenario}_train';val=f'reid_{scenario}_val';gallery=f'reid_{scenario}_gallery';query=f'reid_{scenario}_query'
        train_ids={int(r['id']) for r in tables[train] if int(r['id'])>=0}
        query_ids={int(r['id']) for r in tables[query] if int(r['id'])>=0}
        protocols[scenario]={'official_train_subset':'train_0','official_inner_val_subset':'test_0',
                             'final_gallery':'all train subsets except train_0; anchor/features only, NEVER parameter fit',
                             'final_query':'all test subsets except test_0; NEVER development/selection',
                             'train_final_gallery_crop_overlap':len(keys[train]&keys[gallery]),
                             'train_final_query_crop_overlap':len(keys[train]&keys[query]),
                             'final_query_seen_TRAIN_IDs':sorted(train_ids&query_ids),
                             'final_query_unseen_TRAIN_IDs':sorted(query_ids-train_ids),
                             'train_recordings':manifests[train]['records'],
                             'can_train_cross_recording_positive_pairs_in_official_ReID_train':len(manifests[train]['records'])>1,
                             'continuous_target_tracking_from_crop_retrieval_claim_allowed':False}
    tracking_gallery={}
    for name in tables:
        if not name.startswith('tracking_') or not name.endswith('_train'):continue
        tracking_gallery[name]={gallery:{'path_overlap':len(keys[name]&keys[gallery]),
                                        'semantic_recording_ID_frame_overlap':len(semantic[name]&semantic[gallery]),
                                        'shared_original_recordings':sorted([list(r) for r in recordings[name]&recordings[gallery]])}
                               for gallery in tables if gallery.startswith('reid_') and gallery.endswith('_gallery')}
    write_json('protocol/CHIRLA_OFFICIAL_MEMBERSHIP.json',{
        'status':'COMPLETE_OFFICIAL_PUBLIC_METADATA_ONLY_MEDIA_PROTOCOL_NOT_EXECUTED',
        'final_goal_file':'outputs/N72R21/FINAL_GOAL.json','source_metadata_audit_sha256':sha256(OUT/'datasets/CHIRLA_METADATA_AUDIT.json'),
        'official_memberships':manifests,'crop_path_intersection_matrix':matrix,'ReID_protocols':protocols,
        'semantic_recording_ID_frame_intersection_matrix':{name:{other:len(values&semantic[other]) for other in semantic if other!=name} for name,values in semantic.items()},
        'original_recording_intersection_matrix':{name:{other:len(values&recordings[other]) for other in recordings if other!=name} for name,values in recordings.items()},
        'tracking_TRAIN_vs_final_ReID_gallery_crop_overlap':tracking_gallery,
        'strict_rule':'Keep all scenario-specific ReID/Tracking parameter-fit lineages separate; train_0 is scenario-specific, not one universal recording. Path prefixes differ, so zero path overlap alone is not proof of independent pixels/recordings. Do not mix another benchmark TRAIN with final-gallery/query source records.',
        'no_raw_video_or_image_read':True,'no_model_parameter_or_threshold_selection':True,
        'cross_recording_episode_freeze_requires_lawful_actual_video_annotation_FPS_alignment':True,
        'scientific_success':False,'next_stage_authorized':False})
    print(json.dumps({'official_CSVs':len(tables),'metadata_recordings':len({r['camera'] for rows in tables.values() for r in rows}),
                      'tracking_gallery_overlap':tracking_gallery,'media_accessed':False}),flush=True)


if __name__=='__main__':run()
