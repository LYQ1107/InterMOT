"""Actual full-MOT failure/recovery video comparisons, local licensed pixels only."""
import hashlib
import json
from pathlib import Path
import subprocess
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from scripts.n72r21_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from scripts.n72r21_mot_pilot import checked_frames,DATASET,CODE
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl
from sam3_intermot.one_click.datasets import dancetrack_annotations,strict_candidate_matching


def run():
    result=read_json(OUT/'mot_pilot/RESULT.json');chosen={};registered={
        'MOT_FIRST_STRICT_IDENTITY_RECOVERY':'ACIB_FULL_SEED72101',
        'MOT_FIRST_TARGET_REGRESSION':'ACIB_FULL_SEED72101',
        'MOT_FIRST_COLLATERAL_ID_REGRESSION':'ACIB_FULL_SEED72101',
        'MOT_RAW_ANCHOR_FIRST_NONE_CONTROL':'RAW_ANCHOR'}
    # Fixed protocol scene order / seed / first eligible frame, never best seed,
    # optional second click, fabricated recovery or modified runtime decision.
    for sequence in read_json(OUT/'protocol/MOT_TRAIN_PILOT.json')['sequences']:
        init=read_json(OUT/'mot_pilot/initialization'/f'{sequence}.json');event=init['event']
        identity=next(r['target_gt_identity'] for r in read_json(OUT/'development/INITIALIZATION_TRUTH.json')['labels'] if r['episode_uid']==event['episode_uid'])
        traces={};seals={}
        for case in ['CLICK_C0','ACIB_FULL_SEED72101','RAW_ANCHOR']:
            sealpath=OUT/'mot_pilot/runtime_seals'/case/f'{sequence}.json';seal=read_json(sealpath)
            if sha256(sealpath)!=result['runtime_seal_sha256'][f'{case}/{sequence}']:raise ValueError('completed MOT runtime seal changed')
            for file,h in seal['code_sha256'].items():
                if sha256(ROOT/file)!=h:raise ValueError('sealed runtime source changed')
            for a in seal['artifacts']:
                if sha256(a['path'])!=a['sha256']:raise ValueError('actual MOT predictions changed')
            a=next(a for a in seal['artifacts'] if a['kind']=='trace');traces[case]=read_zstd_jsonl(Path(a['path']));seals[case]=seal
        gt=dancetrack_annotations(DATASET/sequence);frames,_=checked_frames(sequence)
        matched={int(p['frame']):strict_candidate_matching(rows,gt.get(int(p['frame']),[])) for p,rows in frames}
        target_public=traces['CLICK_C0'][event['frame']]['target_public_id'];origin={}
        for row in traces['CLICK_C0']:
            for o in row['outputs']:
                label=matched[row['frame']].get(o['candidate_uid'])
                if o['public_id'] not in origin and label is not None:origin[o['public_id']]=label
        full=traces['ACIB_FULL_SEED72101'];base=traces['CLICK_C0'];raw=traces['RAW_ANCHOR']
        def strict(row):return matched[row['frame']].get(row['target_uid'])==identity
        def save(tag,focus,extra=None):
            if tag not in chosen:chosen[tag]=(sequence,event,focus,traces,gt,seals,extra or {})
        first=event['frame']+1
        previous_visible=True
        for f in range(first,len(full)):
            visible=any(a['identity']==identity for a in gt.get(f,[]))
            if visible and not previous_visible:
                stop=next((k for k in range(f,len(full)) if not any(a['identity']==identity for a in gt.get(k,[]))),len(full))
                recovered=next((k for k in range(f,stop) if strict(full[k])),None)
                if recovered is not None:save('MOT_FIRST_STRICT_IDENTITY_RECOVERY',recovered,{'return_frame':f,'verified_recovery_delay_frames':recovered-f,'visible_GT_gap_not_physical_exit_or_occlusion':True})
            if strict(base[f]) and not strict(full[f]):save('MOT_FIRST_TARGET_REGRESSION',f)
            if raw[f]['target_uid'] is None:save('MOT_RAW_ANCHOR_FIRST_NONE_CONTROL',f)
            b={o['public_id']:o for o in base[f]['outputs']};c={o['public_id']:o for o in full[f]['outputs']}
            for public in sorted(origin):
                if public==target_public:continue
                if matched[f].get(b.get(public,{}).get('candidate_uid'))==origin[public] and matched[f].get(c.get(public,{}).get('candidate_uid'))!=origin[public]:
                    save('MOT_FIRST_COLLATERAL_ID_REGRESSION',f,{'affected_runtime_public_id':public});break
            previous_visible=visible
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14);ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();clips=[]
    for tag,(sequence,event,focus,traces,gt,seals,extra) in chosen.items():
        case=registered[tag];path=ASSETS/'visualizations/MOT'/f'{tag}.mp4';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError('preserve original actual comparison video')
        start=max(event['frame'],focus-int(event['fps']));stop=min(event['frames']-1,focus+int(2*event['fps']))
        anchorpath=DATASET/sequence/'img1'/f"{event['frame']+1:08d}.jpg"
        with Image.open(anchorpath) as image:thumbnail=image.convert('RGB').crop(tuple(event['box_xyxy'])).resize((70,112))
        storage(25<<20);digest=hashlib.sha256();encoded=0;preview=path.with_name(path.stem+'_INSPECTION.png')
        proc=subprocess.Popen([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-n','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
            '-s','1280x510','-r',str(event['fps']),'-i','-','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','25','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE)
        try:
            for f in range(start,stop+1):
                imagepath=DATASET/sequence/'img1'/f'{f+1:08d}.jpg';digest.update(bytes.fromhex(sha256(imagepath)))
                with Image.open(imagepath) as image:original=image.convert('RGB');scaled=original.resize((640,360));sx=640/original.width;sy=360/original.height
                canvas=Image.new('RGB',(1280,510),(12,16,20));draw=ImageDraw.Draw(canvas)
                identity=next(r['target_gt_identity'] for r in read_json(OUT/'development/INITIALIZATION_TRUTH.json')['labels'] if r['episode_uid']==event['episode_uid'])
                target=[a['box'] for a in gt.get(f,[]) if a['identity']==identity]
                for offset,method in [(0,'CLICK_C0'),(640,case)]:
                    canvas.paste(scaled,(offset,0));row=traces[method][f]
                    def box(b,color,width=2):
                        if b is not None:draw.rectangle((offset+b[0]*sx,b[1]*sy,offset+b[2]*sx,b[3]*sy),outline=color,width=width)
                    for o in row['outputs']:
                        is_target=o['public_id']==row['target_public_id'];is_affected=o['public_id']==extra.get('affected_runtime_public_id')
                        box(o['box_xyxy'],'lime' if is_target else 'orange' if is_affected else 'dodgerblue',3 if is_target or is_affected else 1)
                    if target:box(target[0],'yellow')
                    if f==event['frame']:box(event['box_xyxy'],'cyan',3)
                    canvas.paste(thumbnail,(offset+560,8));draw.rectangle((offset+559,7,offset+631,121),outline='cyan',width=2)
                    draw.text((offset+7,366),method,font=font,fill='white')
                    status='NONE' if row['target_uid'] is None else 'TARGET CLAIM'
                    draw.text((offset+7,386),f"{status} | joint WRITE={row['joint_identity_memory_write']} | public target={row['target_public_id']}",font=font,fill='white')
                draw.text((8,410),f'{tag} | original frame={f+1} | {f/event["fps"]:.2f}s | same real input image on both sides',font=font,fill='white')
                draw.text((8,430),f'SIMULATED ONE CLICK FROM GT, frame={event["frame"]+1}; inset is that sole anchor, not a new click; extra clicks=0',font=font,fill='cyan')
                draw.text((8,450),'Green: runtime clicked ID; blue: all other runtime IDs; orange: collateral ID; yellow: GT EVALUATOR ONLY',font=font,fill='white')
                draw.text((8,470),'DanceTrack / Sun et al. CVPR 2022 | local non-commercial research | not cross-recording or real-human-study evidence',font=font,fill='white')
                if f==focus:draw.rectangle((0,0,1279,359),outline='orange',width=3);canvas.save(preview)
                proc.stdin.write(canvas.tobytes());encoded+=1
            proc.stdin.close()
            if proc.wait()!=0:raise RuntimeError('real MOT comparison encoding failed')
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
        clips.append({'tag':tag,'case':case,'sequence':sequence,'episode_uid':event['episode_uid'],'path':str(path),'sha256':sha256(path),'bytes':path.stat().st_size,
            'start':start,'stop':stop,'focus_original_frame_0based':focus,'FPS':event['fps'],'encoded_frames':encoded,
            'actual_pixel_stream_sha256':digest.hexdigest(),'sole_click_image_sha256':sha256(anchorpath),'extra':extra,
            'runtime_seal_sha256':{m:sha256(OUT/'mot_pilot/runtime_seals'/m/f'{sequence}.json') for m in ('CLICK_C0',case)},
            'preview':{'path':str(preview),'sha256':sha256(preview)},'GT_used_only_for_posthoc_selection_and_overlay':True,
            'selection':'First fixed-scene/fixed-seed/fixed-frame eligible event; not best result','synthetic_or_stitched_frames':False,'public_pixel_upload_authorized':False})
    write_json('visualizations/MOT_FAILURE_VIDEO_MANIFEST.json',{'clips':clips,'source_sha256':sha256(Path(__file__)),
        'registered_tags':registered,'unavailable_tags':[t for t in registered if t not in chosen],
        'license_source':'https://github.com/DanceTrack/DanceTrack#agreement','usage':'LOCAL_ONLY_NON_COMMERCIAL_RESEARCH',
        'cross_recording_examples_available':False,'no_new_inference_fitting_or_SOT_work':True,'next_stage_authorized':False})
    print(json.dumps({'actual_full_MOT_failure_and_recovery_videos':len(clips),'bytes':sum(c['bytes'] for c in clips),'unavailable_tags':[t for t in registered if t not in chosen]}))


if __name__=='__main__':run()
