"""Explicit metadata-only repair of two failed shadow registrations.

V1 accidentally placed an explanatory suffix in shadow_head. Its exact intended
checkpoint already exists among registered cases. Keep V1 source/protocol and
failed logs immutable; repair only its metadata reader for these shadow runs.
The original runtime/solver/features are otherwise executed byte-for-byte.
"""
import argparse
from scripts.n72r21r1_common import OUT,ROOT,ASSETS,read_json,write_json,sha256
from scripts import n72r21r1_learned_pilot as original


def corrected_protocol(protocol):
    name=protocol['shadow_head'].split(' ',1)[0]
    if name!='LOGISTIC__MIXED__H100_GLOBAL_RISK__seed72111': raise ValueError('not the frozen intended shadow checkpoint')
    if sum(c['case']==name for c in protocol['cases'])!=1: raise ValueError('ambiguous or absent shadow checkpoint')
    return dict(protocol,shadow_head=name)


def run(sequence):
    protocol=read_json(original.PROTOCOL);corrected=corrected_protocol(protocol)
    failed_log=ASSETS/'learned_pilot/logs'/('LEARNED_SHADOW__'+sequence+'.log')
    assert 'StopIteration' in failed_log.read_text()
    receipt=write_json('learned_pilot/shadow_metadata_repair/'+sequence+'.json',{
        'stage':'N72R21R1','status':'FROZEN_METADATA_REPAIR_BEFORE_REPLAY',
        'original_protocol_path':str(original.PROTOCOL),'original_protocol_SHA':sha256(original.PROTOCOL),
        'original_shadow_field':protocol['shadow_head'],'corrected_shadow_field':corrected['shadow_head'],
        'checkpoint_selection_changed':False,'solver_or_runtime_code_changed':False,
        'failed_original_log_path':str(failed_log),'failed_original_log_SHA':sha256(failed_log),
        'repair_source_SHA':sha256(ROOT/'scripts/n72r21r1_learned_shadow_repair.py'),
        'goal_file':'outputs/N72R21R1/FINAL_GOAL.json','only_cases':['LEARNED_SHADOW'],
        'reader_override':'Only metadata object from this exact protocol path has its shadow_head explanatory suffix removed.'})
    unmodified_reader=original.read_json
    def reader(path):
        return corrected if path==original.PROTOCOL else unmodified_reader(path)
    original.read_json=reader
    try:original.run(sequence,'LEARNED_SHADOW')
    finally:original.read_json=unmodified_reader
    seal=OUT/'learned_pilot/runtime_seals/LEARNED_SHADOW'/(sequence+'.json')
    write_json('learned_pilot/shadow_metadata_repair/'+sequence+'_COMPLETE.json',{
        'status':'COMPLETE_ACTUAL_SHADOW_WITH_EXPLICIT_METADATA_REPAIR',
        'repair_receipt_SHA':sha256(receipt),'actual_runtime_seal_path':str(seal),'actual_runtime_seal_SHA':sha256(seal),
        'original_protocol_and_source_preserved':True,'checkpoint_selection_changed':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sequence',required=True);args=parser.parse_args();run(args.sequence)
