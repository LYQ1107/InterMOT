"""Frozen validation: sole real-image clicks, no fitting or threshold selection."""
import argparse
from collections import defaultdict
import json
import numpy as np
import torch
from scripts.n72r21_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from sam3_intermot.one_click.datasets import dancetrack_info,dancetrack_annotations
from sam3_intermot.identity_probe.dataset import GTBox
from sam3_intermot.identity_probe.crop import load_person_crops
from sam3_intermot.identity_probe.encoders import OSNetEncoder

PROTOCOL=OUT/'protocol/DANCETRACK_FROZEN_VALIDATION.json'


def model_manifest():
    protocol=read_json(PROTOCOL);records=[]
    for seed in protocol['seeds']:
        p=OUT/'training/T2_COUPLED_V1'/f"{protocol['model_source_outer_configuration']}__seed{seed}.json"
        fitted=read_json(p);configuration=fitted['schema']['configuration']
        if not fitted['completed'] or not fitted['future_safe_head_trained_on_verified_paired_supervision']:raise ValueError('actual T2 fitted models required before VAL')
        if set(configuration['fit_sequences']+[configuration['inner_sequence']])&set(protocol['sequences']):raise ValueError('VAL entered fitting')
        if configuration['outer_sequence']!=protocol['model_source_outer_configuration'] or sha256(fitted['best_checkpoint_path'])!=fitted['best_checkpoint_sha256']:raise ValueError('frozen model source')
        for code,h in configuration['code_sha256'].items():
            if sha256(ROOT/code)!=h:raise ValueError('fitting source changed')
        records.append({'seed':seed,'path':fitted['best_checkpoint_path'],'sha256':fitted['best_checkpoint_sha256'],
            'actual_fit_record_path':str(p),'actual_fit_record_sha256':sha256(p),
            'fit_sequences':configuration['fit_sequences'],'inner_sequence':configuration['inner_sequence']})
    record={'protocol_sha256':sha256(PROTOCOL),'models':records,'no_VAL_based_model_selection':True,'next_stage_authorized':False}
    destination=OUT/'validation/FROZEN_MODEL_MANIFEST.json'
    if destination.exists():
        if read_json(destination)!=record:raise ValueError('frozen VAL model manifest changed')
    else:write_json('validation/FROZEN_MODEL_MANIFEST.json',record)
    return record


def prepare(sequences):
    torch.set_num_threads(1);protocol=read_json(PROTOCOL)
    if not set(sequences)<=set(protocol['sequences']):raise ValueError('frozen VAL scope')
    models=model_manifest() # Must seal models before opening any VAL GT/pixel.
    checkpoint=ROOT.parent/'InterMOT_N72R16_assets/checkpoints/osnet_x1_0_market1501.pth'
    if sha256(checkpoint)!='2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154':raise ValueError('OSNet source')
    encoder=OSNetEncoder(checkpoint,'cpu');audit=read_json(OUT/'datasets/DANCETRACK_VAL_CANDIDATE_METADATA_AUDIT.json')
    registered={r['sequence']:r for r in audit['records']}
    for sequence in sequences:
        done=OUT/'validation/initialization'/f'{sequence}.json'
        if done.exists():
            old=read_json(done)
            if old['preparation_code_sha256']!=sha256(ROOT/'scripts/n72r21_prepare_validation.py') or old['protocol_sha256']!=sha256(PROTOCOL) or sha256(old['anchor_path'])!=old['anchor_sha256']:raise ValueError('sealed initializations changed')
            continue
        storage(2<<20);source=ROOT.parent/'InterMOT_N72R16_assets/dataset/val'/sequence
        info=dancetrack_info(source);gt=dancetrack_annotations(source);initial={}
        if info['frames']!=registered[sequence]['frames']:raise ValueError('partial historical VAL candidate axis')
        for frame,rows in sorted(gt.items()):
            for row in sorted(rows,key=lambda r:r['identity']):
                x,y,right,bottom=row['box'];w,h=right-x,bottom-y
                if row['identity'] not in initial and w>=16 and h>=32 and w*h>=1024 and row['visibility']>=.5:initial[row['identity']]=(frame,row)
        grouped=defaultdict(list)
        for identity,(frame,row) in sorted(initial.items()):grouped[frame].append((identity,row))
        inputs=[];labels=[];vectors=[]
        for frame,rows in sorted(grouped.items()):
            boxes=[GTBox(sequence,frame+1,identity,r['box'][0],r['box'][1],r['box'][2]-r['box'][0],r['box'][3]-r['box'][1],r['confidence'],1,r['visibility']) for identity,r in rows]
            path=source/'img1'/f'{frame+1:08d}.jpg';crops=load_person_crops(path,boxes)
            embeddings=encoder.encode(torch.stack(crops)).cpu().numpy()
            for (identity,row),vector in zip(rows,embeddings,strict=True):
                uid=f'{sequence}__click{len(inputs):04d}';vectors.append(vector)
                inputs.append({'sequence':sequence,'recording_id':sequence,'episode_uid':uid,'anchor_index':len(vectors)-1,
                    'frame':frame,'box_xyxy':list(row['box']),**info,'initialization_label':'SIMULATED_ONE_CLICK_FROM_GT',
                    'initial_image_sha256':sha256(path),'extra_clicks':0})
                labels.append({'episode_uid':uid,'target_gt_identity':identity})
        if not vectors:raise ValueError('no eligible clicks; retain as initialization failure, no silent scene exclusion')
        anchor=ASSETS/'validation/anchors'/f'{sequence}.npy';anchor.parent.mkdir(parents=True,exist_ok=True)
        if anchor.exists():raise FileExistsError('preserve unsealed partial anchor array')
        np.save(anchor,np.stack(vectors).astype(np.float32))
        write_json(f'validation/initialization_truth/{sequence}.json',{'labels':labels,'GT_sha256':sha256(source/'gt/gt.txt'),
            'builder_parsed_GT_only_to_simulate_first_eligible_single_click':True,'runtime_never_opens_this_file':True})
        write_json(f'validation/initialization/{sequence}.json',{'inputs':inputs,'anchor_path':str(anchor),'anchor_sha256':sha256(anchor),
            'protocol_sha256':sha256(PROTOCOL),'frozen_model_manifest_sha256':sha256(OUT/'validation/FROZEN_MODEL_MANIFEST.json'),
            'preparation_code_sha256':sha256(ROOT/'scripts/n72r21_prepare_validation.py'),
            'candidate_metadata_audit_sha256':sha256(OUT/'datasets/DANCETRACK_VAL_CANDIDATE_METADATA_AUDIT.json'),
            'runtime_GT_identity_fields':False,'model_or_threshold_fitting':False})
        print(json.dumps({'VAL_one_click_initialization_sealed':sequence,'all_eligible_targets':len(inputs)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sequences',nargs='+')
    args=parser.parse_args();prepare(args.sequences or read_json(PROTOCOL)['sequences'])
