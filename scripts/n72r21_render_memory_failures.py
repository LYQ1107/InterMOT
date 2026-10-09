"""Local-only actual T2 tracking/write failures; posthoc GT is labelled."""
import hashlib
import json
from pathlib import Path
import subprocess
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from scripts.n72r21_common import ROOT,OUT,ASSETS,read_json,write_json,sha256,storage
from scripts.n72r20r1_build_base_score_tape import load_candidate_frames,read_zstd_jsonl
from scripts.n72r21_baselines_geometry_repair import valid_geometry
from sam3_intermot.one_click.datasets import dancetrack_annotations,dancetrack_truth,strict_candidate_matching


def run():
    sequence='dancetrack0001';cases=['T2_FULL_K8_SEED72101','T2_WITHOUT_SAFE_WRITE_P1_SEED72101']
    inputs={e['episode_uid']:e for e in read_json(OUT/'development/RUNTIME_INPUTS.json')['inputs'] if e['sequence']==sequence}
    identities={e['episode_uid']:e['target_gt_identity'] for e in read_json(OUT/'development/INITIALIZATION_TRUTH.json')['labels']}
    seals={case:read_json(OUT/'experiments/T2/runtime_seals'/case/f'{sequence}.json') for case in cases}
    for seal in seals.values():
        for artifact in seal['artifacts']:
            if sha256(artifact['path'])!=artifact['sha256']:raise ValueError('unsealed actual runtime')
    source=ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/sequence
    gt=dancetrack_annotations(source);frames=[(p,[r for r in rows if valid_geometry(r)]) for p,rows in load_candidate_frames(ROOT.parent/'InterMOT_N72R20R2_assets',sequence)]
    matched={int(p['frame']):strict_candidate_matching(rows,gt.get(int(p['frame']),[])) for p,rows in frames};chosen={}
    for case in cases:
        for artifact in sorted(seals[case]['artifacts'],key=lambda a:a['episode_uid']):
            event=inputs[artifact['episode_uid']];identity=identities[artifact['episode_uid']]
            trace=read_zstd_jsonl(Path(artifact['path']));runtime={r['frame']:r for r in trace}
            truth={int(p['frame']):dancetrack_truth(int(p['frame']),rows,gt.get(int(p['frame']),[]),identity) for p,rows in frames if int(p['frame'])>event['frame']}
            previous_visible=True
            for r in trace:
                t=truth[r['frame']];label=matched[r['frame']].get(r['selected_candidate_uid']);tags=[]
                if case==cases[0]:
                    if label==identity:tags.append('FULL_FIRST_CORRECT_IDENTITY')
                    if r['selected_candidate_uid'] is None:tags.append('FULL_FIRST_NONE')
                    if not t.target_visible:tags.append('FULL_FIRST_VISIBLE_GT_GAP')
                    if t.target_visible and not previous_visible:tags.append('FULL_FIRST_REAPPEARANCE')
                    if r['selected_candidate_uid'] is not None and label is not None and label!=identity:tags.append('FULL_FIRST_VERIFIED_WRONG_IDENTITY')
                elif r['memory_write']:
                    if label==identity:tags.append('UNSAFE_P1_FIRST_VERIFIED_CORRECT_WRITE')
                    elif label is not None:tags.append('UNSAFE_P1_FIRST_VERIFIED_WRONG_WRITE')
                for tag in tags:
                    if tag not in chosen:chosen[tag]=(case,artifact,event,r['frame'],runtime,truth)
                previous_visible=t.target_visible
    if not chosen:raise ValueError('no actual events, never fabricate demonstrations')
    clips=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13);ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    for tag,(case,artifact,event,focus,runtime,truth) in chosen.items():
        path=ASSETS/'visualizations/T2'/f'{tag}.mp4';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError('preserve prior local evidence video')
        start=max(event['frame'],focus-int(event['fps']));stop=min(event['frames']-1,focus+int(2*event['fps']))
        initial_path=source/'img1'/f"{event['frame']+1:08d}.jpg"
        with Image.open(initial_path) as image:
            thumbnail=image.convert('RGB').crop(tuple(event['box_xyxy'])).resize((72,116))
        storage(20<<20);digest=hashlib.sha256();encoded=0
        proc=subprocess.Popen([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-n','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
            '-s','640x464','-r',str(event['fps']),'-i','-','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','25','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE)
        try:
            for f in range(start,stop+1):
                image_path=source/'img1'/f'{f+1:08d}.jpg';digest.update(bytes.fromhex(sha256(image_path)))
                with Image.open(image_path) as image:
                    original=image.convert('RGB');sx=640/original.width;sy=360/original.height;scaled=original.resize((640,360))
                canvas=Image.new('RGB',(640,464),(12,16,20));canvas.paste(scaled,(0,0));draw=ImageDraw.Draw(canvas)
                def box(value,color):
                    if value is not None:draw.rectangle((value[0]*sx,value[1]*sy,value[2]*sx,value[3]*sy),outline=color,width=2)
                r=runtime.get(f);t=truth.get(f)
                if t is not None:box(t.target_box_xyxy,'yellow')
                if r is not None:box(r['predicted_box_xyxy'],'lime')
                if f==event['frame']:box(event['box_xyxy'],'cyan')
                canvas.paste(thumbnail,(556,8));draw.rectangle((555,7,629,125),outline='cyan',width=2)
                draw.text((467,129),f"sole click frame={event['frame']+1}",font=font,fill='cyan')
                policy='UNSAFE P1 DIAGNOSTIC (not deployment)' if case==cases[1] else 'FULL LEARNED SAFE (no writes in this scene)'
                status='SINGLE CLICK' if r is None else 'NONE' if r['selected_candidate_uid'] is None else 'TARGET CLAIM'
                write=bool(r and r['memory_write']);bank=r.get('machine_bank_size',0) if r else 0
                draw.text((8,364),f'{tag} | original frame={f+1} | {f/event["fps"]:.2f}s',font=font,fill='white')
                draw.text((8,382),policy,font=font,fill='white')
                draw.text((8,400),f'{status} | WRITE={write} | committed bank={bank} | extra clicks=0',font=font,fill='orange' if write else 'white')
                draw.text((8,418),'Green: runtime; yellow: GT EVALUATOR ONLY; cyan: sole real click',font=font,fill='white')
                draw.text((8,436),'DanceTrack / Sun et al. CVPR 2022 | local non-commercial research',font=font,fill='white')
                if f==focus:draw.rectangle((0,0,639,359),outline='orange',width=3)
                proc.stdin.write(canvas.tobytes());encoded+=1
            proc.stdin.close()
            if proc.wait()!=0:raise RuntimeError('video encoding failure')
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
        clips.append({'tag':tag,'case':case,'episode_uid':event['episode_uid'],'path':str(path),'sha256':sha256(path),'bytes':path.stat().st_size,
            'runtime_sha256':artifact['sha256'],'focus_original_frame_0based':focus,'start':start,'stop':stop,'FPS':event['fps'],'encoded_frames':encoded,
            'sole_anchor_image_sha256':sha256(initial_path),'actual_pixel_stream_sha256':digest.hexdigest(),
            'selection':'FIRST_IN_FIXED_CASE_SEQUENCE_EPISODE_FRAME_ORDER_NOT_BEST_RESULT','GT_overlay_posthoc_only':True,
            'visible_GT_gap_not_verified_physical_exit_or_occlusion':True,'synthetic_or_stitched_frames':False,'public_pixel_upload_authorized':False})
    write_json('visualizations/T2_FAILURE_VIDEO_MANIFEST.json',{'clips':clips,'real_events_not_synthetic':True,
        'license_source':'https://github.com/DanceTrack/DanceTrack#agreement','usage':'LOCAL_ONLY_NON_COMMERCIAL_RESEARCH; no public raw pixel redistribution',
        'attribution':'DanceTrack / Sun et al. CVPR 2022','cross_recording_examples_available':False,
        'safe_model_zero_writes_not_hidden':True,'unsafe_write_control_clearly_labelled':True,'next_stage_authorized':False})
    print(json.dumps({'actual_T2_evidence_videos':len(clips),'bytes':sum(c['bytes'] for c in clips),'public_upload':False}))


if __name__=='__main__':run()
