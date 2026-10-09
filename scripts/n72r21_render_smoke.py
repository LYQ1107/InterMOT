"""Render local-only, attributed research excerpts from real sealed smoke.

No synthetic imagery, interpolated detections, identity-name labels or stitched
session timelines. Yellow truth boxes are evaluator-only, never runtime input.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont
from scripts.n72r21_common import ROOT, OUT, ASSETS, read_json, write_json, sha256, storage
from scripts.n72r20r1_build_base_score_tape import read_zstd_jsonl, load_candidate_frames
from sam3_intermot.one_click.datasets import dancetrack_annotations, dancetrack_truth
from sam3_intermot.evaluation.one_click_protocol import iou


def run():
    seal=read_json(OUT/'smoke/RUNTIME_SEAL.json')
    # Runtime code at this historical smoke is sealed in published commit
    # 879f228. Any later source edits do not retroactively rerun that smoke.
    inputs={e['episode_uid']:e for e in read_json(OUT/'smoke/RUNTIME_INPUTS.json')['inputs']}
    labels={e['episode_uid']:e for e in read_json(OUT/'smoke/INITIALIZATION_TRUTH.json')['labels']}
    chosen={};frames_cache={};gt_cache={}
    for record in seal['artifacts']:
        if sha256(record['path'])!=record['sha256']:raise ValueError('unsealed runtime')
        event=inputs[record['episode_uid']];sequence=event['sequence']
        source=ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/sequence
        if sequence not in frames_cache:
            frames_cache[sequence]=load_candidate_frames(ROOT.parent/'InterMOT_N72R20R2_assets',sequence)
            gt_cache[sequence]=dancetrack_annotations(source)
        runtime=read_zstd_jsonl(Path(record['path']));by_frame={r['frame']:r for r in runtime}
        truth={f:dancetrack_truth(f,rows,gt_cache[sequence].get(f,[]),labels[record['episode_uid']]['target_gt_identity'])
               for p,rows in frames_cache[sequence] if (f:=int(p['frame']))>event['frame']}
        last_visible=True
        for r in runtime:
            t=truth[r['frame']];correct=t.target_visible and iou(r['predicted_box_xyxy'],t.target_box_xyxy)>=.5
            takeover=bool(r['selected_candidate_uid'] is not None and not correct and any(iou(r['predicted_box_xyxy'],b)>=.5 for b in t.other_person_boxes_xyxy))
            tags=[]
            if correct:tags.append('FIRST_CORRECT_SELECTION')
            if not t.target_visible:tags.append('FIRST_VISIBLE_GT_GAP')
            if t.target_visible and not last_visible:tags.append('FIRST_REAPPEARANCE')
            if takeover:tags.append('FIRST_VERIFIED_WRONG_PERSON')
            if r['selected_candidate_uid'] is None:tags.append('FIRST_NONE_DECISION')
            for tag in tags:
                if tag not in chosen:chosen[tag]=(record,event,r['frame'],by_frame,truth)
            last_visible=t.target_visible
        if len(chosen)==5:break
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    clips=[];ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    for tag,(record,event,focus,runtime,truth) in chosen.items():
        start=max(event['frame'],focus-int(event['fps']));stop=min(event['frames']-1,focus+int(2*event['fps']))
        path=ASSETS/'visualizations/smoke'/f'{tag}.mp4';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError('do not overwrite completed evidence video')
        storage(20<<20)
        process=subprocess.Popen([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-n',
            '-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','640x448','-r',str(event['fps']),'-i','-',
            '-an','-c:v','libx264','-threads','1','-preset','fast','-crf','25','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
        source=ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/event['sequence'];pixel_digest=hashlib.sha256()
        try:
            for f in range(start,stop+1):
                image_path=source/'img1'/f'{f+1:08d}.jpg'
                pixel_digest.update(bytes.fromhex(sha256(image_path)))
                with Image.open(image_path) as image:
                    original=image.convert('RGB');scaled=original.resize((640,360));sx=640/original.width;sy=360/original.height
                canvas=Image.new('RGB',(640,448),(12,16,20));canvas.paste(scaled,(0,0));draw=ImageDraw.Draw(canvas)
                def box(value,color):
                    if value is not None:draw.rectangle((value[0]*sx,value[1]*sy,value[2]*sx,value[3]*sy),outline=color,width=2)
                r=runtime.get(f);t=truth.get(f)
                if f==event['frame']:box(event['box_xyxy'],'cyan')
                if t is not None:box(t.target_box_xyxy,'yellow')
                if r is not None:box(r['predicted_box_xyxy'],'lime')
                selection='CLICK (exactly one initialization)' if r is None else 'NONE' if r['selected_candidate_uid'] is None else 'TARGET OUTPUT'
                draw.text((8,364),f"{tag} | {event['sequence']} frame={f+1} t={f/event['fps']:.2f}s",font=font,fill='white')
                draw.text((8,384),f'{selection} | frozen anchor P0 | writes=0 (not a memory-safety PASS)',font=font,fill='white')
                draw.text((8,404),'Cyan: single click; green: runtime; yellow: GT EVALUATOR ONLY',font=font,fill='white')
                draw.text((8,424),'DanceTrack / Sun et al. CVPR 2022 | local non-commercial research only',font=font,fill='white')
                if f==focus:draw.rectangle((0,0,639,359),outline='orange',width=3)
                process.stdin.write(canvas.tobytes())
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError('video encoder failed')
        finally:
            if process.poll() is None:process.terminate();process.wait()
        clips.append({'case':tag,'path':str(path),'sha256':sha256(path),'bytes':path.stat().st_size,
            'episode_uid':event['episode_uid'],'runtime_sha256':record['sha256'],'focus_original_frame_0based':focus,
            'first_original_frame_0based':start,'last_original_frame_0based':stop,'actual_fps':event['fps'],
            'frames_encoded':stop-start+1,'pixel_source_sha256_stream':pixel_digest.hexdigest(),
            'selection_rule':'FIRST_IN_FROZEN_SEQUENCE_EPISODE_FRAME_ORDER_NOT_BEST_RESULT',
            'raw_video_or_images_committed_to_Git':False,'public_redistribution_authorized':False,
            'absence_semantics':'visible GT gap only, no inferred physical exit','synthetic_or_stitched_frames':False})
    write_json('visualizations/SMOKE_VIDEO_MANIFEST.json',{'clips':clips,'license_source':'https://github.com/DanceTrack/DanceTrack#agreement',
        'usage':'LOCAL_ONLY_NON_COMMERCIAL_RESEARCH; not a blanket redistribution license',
        'attribution':'DanceTrack, Sun et al., CVPR 2022','tool':'imageio-ffmpeg==0.6.0 bundled ffmpeg; CPU thread1',
        'GT_overlay_posthoc_only':True,'cross_session_examples_available':False,'memory_update_example_available':False})
    print(json.dumps({'real_local_evidence_videos':len(clips),'bytes':sum(r['bytes'] for r in clips),'public_upload':False}))


if __name__=='__main__':run()
