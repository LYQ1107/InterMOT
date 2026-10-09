"""Separate GT-free process: real sealed candidates, no annotation imports."""
from __future__ import annotations
import argparse
import json
import subprocess
import numpy as np
from sam3_intermot.one_click import Candidate,OneClickRecognizer,RuntimeConfig
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames
from scripts.n72r21_common import ROOT,ASSETS,OUT,read_json,write_json,sha256,storage


def run():
    manifest=read_json(OUT/'smoke/RUNTIME_INPUTS.json')
    if sha256(manifest['anchor_path'])!=manifest['anchor_sha256']:raise ValueError('anchor seal')
    anchors=np.load(manifest['anchor_path'],mmap_mode='r')
    root=ROOT.parent/'InterMOT_N72R20R2_assets'
    artifacts=[]
    for sequence in sorted({e['sequence'] for e in manifest['inputs']}):
        index=read_json(root/'candidates'/sequence/'index.json')
        if index.get('runtime_future_gt_used') or index.get('runtime_gt_read'):raise ValueError('candidate lineage GT unsafe')
        for key in ('metadata','embeddings'):
            if sha256(index[key])!=index[key+'_sha256']:raise ValueError('candidate tape SHA mismatch')
        frames=load_candidate_frames(root,sequence)
        for event in [e for e in manifest['inputs'] if e['sequence']==sequence]:
            recognizer=OneClickRecognizer(RuntimeConfig())
            recognizer.initialize(anchors[event['anchor_index']],event['box_xyxy'],recording_id=event['recording_id'],frame=event['frame'])
            storage(2<<20)
            output=ASSETS/'smoke/runtime'/f"{event['episode_uid']}.jsonl.zst";output.parent.mkdir(parents=True,exist_ok=True)
            if output.exists():raise FileExistsError('sealed smoke exists; do not overwrite runtime')
            count=0
            with output.open('wb') as handle:
                process=subprocess.Popen(['zstd','-q','-c','-T1','-3'],stdin=subprocess.PIPE,stdout=handle)
                try:
                    for payload,rows in frames:
                        frame=int(payload['frame'])
                        if frame<=event['frame']:continue
                        candidates=[Candidate(str(r['candidate_uid']),r['feature'],tuple(r['box_xyxy']),float(np.clip(r.get('conf',0.),0.,1.)),str(r['native_tid'])) for r in rows
                                    if np.isfinite(r['box_xyxy']).all() and r['box_xyxy'][2]>r['box_xyxy'][0] and r['box_xyxy'][3]>r['box_xyxy'][1]]
                        decision=recognizer.step(frame,candidates,fps=event['fps'])
                        process.stdin.write((json.dumps(decision,sort_keys=True,allow_nan=False)+'\n').encode());count+=1
                    process.stdin.close()
                    if process.wait()!=0:raise RuntimeError('runtime compression failed')
                finally:
                    if process.poll() is None:process.terminate();process.wait()
            artifacts.append({'episode_uid':event['episode_uid'],'path':str(output),'sha256':sha256(output),'frames':count,'runtime_gt_used':False,'runtime_future_gt_used':False,
                              'candidate_metadata_sha256':index['metadata_sha256'],'candidate_embeddings_sha256':index['embeddings_sha256']})
        print(json.dumps({'sequence':sequence,'GT_free_runtime_sealed_episodes':len([r for r in artifacts if r['episode_uid'].startswith(sequence)])}),flush=True)
    write_json('smoke/RUNTIME_SEAL.json',{'artifacts':artifacts,'ground_truth_files_opened_by_runtime_process':False,'new_SAM3_inference':False,'candidate_axis_changed':False,
               'code_sha256':{p:sha256(ROOT/p) for p in ['scripts/n72r21_runtime_smoke.py','sam3_intermot/one_click/runtime.py']}})


if __name__=='__main__':run()
