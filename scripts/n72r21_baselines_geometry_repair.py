"""Versioned GT-free input validity repair for the final TRAIN sequence.

Seven completed baselines retain their original source/results/seals. The raw
0072 SAM3 tape has 13 zero-area boxes; these cannot be candidate observations
under either reference recognizer or unchanged actual tracker contracts. Only
invalid geometry is removed, identically for all nine cases, without GT or
score/threshold selection. Raw candidate tapes are never modified.
"""
import argparse
from pathlib import Path
import numpy as np
from scripts import n72r21_baselines as original
from scripts.n72r21_common import OUT,ASSETS,read_json,write_json,sha256

SEQUENCE='dancetrack0072'


def valid_geometry(row):
    b=np.asarray(row['box_xyxy'],float)
    return b.shape==(4,) and bool(np.isfinite(b).all()) and b[2]>b[0] and b[3]>b[1]


def run(command):
    pure_loader=original.load_candidate_frames;invalid=[]
    def load(root,sequence):
        if sequence!=SEQUENCE:raise ValueError('this repair is registered only for the newly failed sequence')
        frames=pure_loader(root,sequence);filtered=[]
        for payload,rows in frames:
            dropped=[{'frame':int(payload['frame']),'candidate_uid':str(r['candidate_uid']),
                      'box_xyxy':list(r['box_xyxy']),'reason':'NONFINITE_OR_NONPOSITIVE_GEOMETRY'} for r in rows if not valid_geometry(r)]
            invalid.extend(dropped);filtered.append((payload,[r for r in rows if valid_geometry(r)]))
        return filtered
    original.load_candidate_frames=load
    original.SEALED_CODE=(*original.SEALED_CODE,'scripts/n72r21_baselines_geometry_repair.py')
    normal_writer=original.write_json
    def writer(relative,value):
        if str(relative).startswith('baselines/runtime_seals/'):
            value={**value,'candidate_axis_changed':True,'input_validity_repair':'GT_FREE_REMOVE_ONLY_ZERO_AREA_NONFINITE_BOXES',
                   'invalid_geometry_excluded':list({r['candidate_uid']:r for r in invalid}.values()),
                   'raw_source_tape_modified':False,'all_baselines_same_retained_candidate_UID_axis':True,
                   'protocol_repair_manifest_sha256':sha256(OUT/'protocol/CANDIDATE_VALIDITY_REPAIR.json')}
        return normal_writer(relative,value)
    original.write_json=writer
    if command=='replay':
        # Preserve the sole empty failed output, without deleting evidence or
        # touching a completed seal. This is a recoverable stage-only move.
        partial=ASSETS/'baselines/runtime/B0_RAW_ANCHOR/dancetrack0072__click0047.jsonl.zst'
        destination=ASSETS/'diagnostics/failed_initial_0072_geometry/dancetrack0072__click0047.jsonl.zst'
        if partial.exists() and not (OUT/'baselines/runtime_seals/B0_RAW_ANCHOR/dancetrack0072.json').exists():
            if partial.stat().st_size!=0:raise ValueError('unexpected non-empty partial requires separate evidence-preserving recovery')
            if destination.exists():raise FileExistsError('do not overwrite prior failed-run evidence')
            destination.parent.mkdir(parents=True,exist_ok=True);digest=sha256(partial);partial.rename(destination)
            write_json('baselines/INPUT_REPAIR_RECOVERY.json',{'status':'EMPTY_FAILED_STAGE_OUTPUT_MOVED_RECOVERABLY',
                'source':str(partial),'destination':str(destination),'sha256':digest,'bytes':0,'material_bytes_deleted':0,
                'historical_source_or_completed_results_changed':False})
        original.replay([SEQUENCE],list(original.CASES))
    else:original.evaluate([SEQUENCE],list(original.CASES))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['replay','evaluate']);a=p.parse_args();run(a.command)
