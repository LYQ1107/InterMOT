"""Audit downloaded official metadata, never infer media availability from CSVs."""
from collections import defaultdict
import csv
import hashlib
import json
from scripts.n72r21_common import ASSETS,OUT,read_json,write_json,sha256,utcnow


def run():
    source=ASSETS/'official_sources/CHIRLA'
    tree=read_json(source/'TREE.json');entries={e['path']:e for e in tree['tree'] if e['type']=='blob'}
    verified=[];summaries={};missing=[]
    for name,entry in entries.items():
        path=source/name
        if not path.exists():
            if name.startswith('benchmark/metadata/') and name.endswith('.csv'):missing.append(name)
            continue
        data=path.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if blob!=entry['sha']:raise ValueError('official blob mismatch')
        verified.append({'path':str(path),'github_blob_sha1':blob,'sha256':sha256(path),'bytes':len(data)})
        if name.startswith('benchmark/metadata/') and name.endswith('.csv'):
            with path.open() as handle:rows=list(csv.DictReader(handle))
            per_person=defaultdict(set)
            for row in rows:per_person[int(row['id'])].add((row['sequence'],row['camera']))
            role='FINAL_QUERY_NO_DEVELOPMENT' if name.endswith('_query.csv') else 'FINAL_GALLERY_FEATURE_EXTRACTION_ONLY' if name.endswith('_gallery.csv') else 'OFFICIAL_DEVELOPMENT_VAL' if name.endswith('_val.csv') else 'OFFICIAL_TRAIN'
            summaries[name]={'rows':len(rows),'role':role,'positive_global_ids':sorted(i for i in per_person if i>=0),
                'negative_distractor_ids':sorted(i for i in per_person if i<0),'recordings':sorted({r['camera'] for r in rows}),
                'sequences':sorted({r['sequence'] for r in rows}),'subsets':sorted({r.get('subset','NO_TRACKING_SUBSET') for r in rows}),
                'person_recording_metadata':{str(i):[list(x) for x in sorted(recordings)] for i,recordings in per_person.items()},
                'cross_recording_positive_ids':sorted(i for i,v in per_person.items() if i>=0 and len(v)>1),
                'media_downloaded':False,'images_loaded':False,'metadata_only':True}
    result={'utc':utcnow(),'github_tree_sha':tree['sha'],'verified_official_files':verified,'CSV_audit':summaries,
            'missing_official_CSVs':missing,'complete_protocol_can_be_frozen':not missing,
            'raw_video_pilot_available':False,'raw_annotation_pilot_available':False,
            'official_global_ID_claim_source':'CHIRLA authoritative documentation, not local-ID coincidence',
            'cross_session_runtime_executed':False,'metadata_does_not_prove_video_or_crop_download_permission':True,
            'warning':'train_0 is scenario-specific, not universally seq_000. Inspect each complete CSV and canonical recording/ID/frame memberships. Do not mix scenario-specific TRAIN with another protocol final gallery/query; different path prefixes alone do not prove independent pixels.',
            'hf_page_access_gate_source':'https://huggingface.co/datasets/bdager/CHIRLA',
            'hf_gate_verified_by_official_page':'Requires login/accepting contact-sharing conditions; server API currently network-blocked, no gated media requested',
            'actual_fps_not_inferred_from_filename':True}
    write_json('datasets/CHIRLA_METADATA_AUDIT.json',result)
    local=read_json(OUT/'datasets/DATASET_DISCOVERY.json')
    latest=read_json(OUT/'datasets/LASOT_OFFICIAL_METADATA.json') if (OUT/'datasets/LASOT_OFFICIAL_METADATA.json').exists() else {}
    boxes_available=bool(latest.get('local_train_windows')) and all(r.get('all_window_frames_have_official_boxes') for r in latest['local_train_windows'])
    write_json('datasets/LASOT_PERSON_AUDIT.json',{'status':'EXISTING_TAO_CONTIGUOUS_TEMPORALLY_TRIMMED_WINDOWS_NOT_FULL_ORIGINAL_LASOT',
               'existing_subsets':local['lasot_person_existing'],'complete_original_sequences':0,'original_GT_boxes_verified_for_three_TRAIN_windows':boxes_available,
               'official_box_annotation_manifest':'outputs/N72R21/datasets/LASOT_OFFICIAL_METADATA.json' if latest else None,
               'original_FPS_verified':False,
               'true_absence_labels_verified':False,'GT_gaps_as_absence_allowed':False,'cross_video_global_identity_claim_allowed':False,
               'download_priority':'BAIDU_DIRECT_FIRST_ALREADY_ATTEMPTED_NO_TRANSFER',
               'baidu_url':'https://pan.baidu.com/s/1xFANiqkBHytE7stMOLUpLQ','share_link_alive_or_extraction_code_verified':False,
               'existing_tool':'BaiduPCS-Go v4.0.2 in existing GMT project; read-only inspection; password/cookie login not invoked',
               'direct_https_connect':'5s connect timeout; zero transferred bytes','direct_http_connect':'5s connect timeout; zero transferred bytes',
               'throughput':'UNDEFINED_NO_SUCCESSFUL_DOWNLOAD; not guessed from metadata requests',
               'downloaded_test_or_whole_archive':False,'duplicate_data_copies':False})
    print(json.dumps({'official_files_blob_verified':len(verified),'CSV_files':len(summaries),'missing_CSVs':len(missing),'media_downloaded':False}))


if __name__=='__main__':run()
