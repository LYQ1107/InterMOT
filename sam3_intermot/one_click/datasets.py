"""Explicit offline truth adapters; never infer cross-video IDs or sparse absence."""
from __future__ import annotations
from pathlib import Path
import configparser
import numpy as np
from scipy.optimize import linear_sum_assignment
from sam3_intermot.evaluation.one_click_protocol import FrameTruth, iou


def dancetrack_info(sequence_root):
    config=configparser.ConfigParser();config.read(Path(sequence_root)/'seqinfo.ini')
    values=config['Sequence']
    return {'fps':float(values['frameRate']),'frames':int(values['seqLength']),
            'width':int(values['imWidth']),'height':int(values['imHeight'])}


def dancetrack_annotations(sequence_root):
    result={}
    for line in (Path(sequence_root)/'gt/gt.txt').read_text().splitlines():
        if not line.strip():continue
        values=[float(x) for x in line.split(',')]
        if len(values)<6:raise ValueError('invalid GT row')
        frame,identity=int(values[0])-1,int(values[1])
        x,y,w,h=values[2:6]
        confidence=values[6] if len(values)>6 else 1.
        category=int(values[7]) if len(values)>7 else 1
        visibility=values[8] if len(values)>8 else 1.
        if identity<0 or confidence<=0 or category!=1 or w<=0 or h<=0:continue
        result.setdefault(frame,[]).append({'identity':identity,'box':(x,y,x+w,y+h),
                                           'visibility':visibility,'confidence':confidence})
    return result


def strict_candidate_matching(candidates, annotations):
    annotations=sorted(annotations,key=lambda r:r['identity'])
    matrix=np.zeros((len(candidates),len(annotations)+len(candidates)))
    for i,candidate in enumerate(candidates):
        for j,annotation in enumerate(annotations):
            overlap=iou(candidate['box_xyxy'],annotation['box'])
            matrix[i,j]=overlap if overlap>=.5 else -1.
    matches={str(c['candidate_uid']):None for c in candidates}
    ii,jj=linear_sum_assignment(-matrix)
    for i,j in zip(ii,jj):
        if j<len(annotations) and matrix[i,j]>=.5:matches[str(candidates[i]['candidate_uid'])]=annotations[j]['identity']
    return matches


def dancetrack_truth(frame, candidates, annotations, target_identity):
    matches=strict_candidate_matching(candidates,annotations)
    target=[a for a in annotations if a['identity']==target_identity]
    if len(target)>1:raise ValueError('duplicate target truth in frame')
    return FrameTruth(int(frame),bool(target),tuple(target[0]['box']) if target else None,
                      frozenset(uid for uid,identity in matches.items() if identity==target_identity),
                      tuple(tuple(a['box']) for a in annotations if a['identity']!=target_identity))


def validate_cross_session_pair(anchor, query, *, authoritative_global_ids=False):
    if not authoritative_global_ids:raise ValueError('authoritative global ID mapping required')
    if anchor['recording_id']==query['recording_id']:raise ValueError('not independent recordings')
    if anchor['global_identity']!=query['global_identity']:raise ValueError('not the same global person')
    if not anchor.get('timestamp') or not query.get('timestamp'):raise ValueError('actual session timestamps required')
    return True
