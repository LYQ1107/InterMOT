"""Output-only recovery after lost host session; never modify sealed source."""
import argparse
from pathlib import Path
from scripts import n72r21r1_teacher_diagnostic as diagnostic
from scripts.n72r21r1_common import ROOT,OUT,ASSETS,write_json,sha256


def main(sequence):
    if sequence!='dancetrack0002':raise ValueError('only the verified interrupted sequence is in recovery scope')
    partial=ASSETS/'diagnostics/teacher_state_v1/dancetrack0002__R1click003.jsonl.zst'
    if not partial.is_file():raise ValueError('original interrupted attempt must be retained')
    protocol=diagnostic.PROTOCOL
    record={'reason':'HOST_SESSION_HANDLE_LOST_AND_NO_CORRESPONDING_LIVE_PROCESS; no completed episode seal',
        'sequence':sequence,'retained_partial':str(partial),'retained_partial_SHA':sha256(partial),
        'original_protocol_SHA':sha256(protocol),'unchanged_original_source_SHA':sha256(Path(diagnostic.__file__)),
        'recovery_wrapper_SHA':sha256(Path(__file__)),
        'output_only_recovery_asset_root':str(ASSETS/'diagnostic_recovery_attempt_v2'),
        'no_original_code_or_partial_outputs_overwritten':True}
    write_json('diagnostics/teacher_state_v1/RECOVERY_ATTEMPT_V2.json',record)
    diagnostic.ASSETS=Path(record['output_only_recovery_asset_root'])
    diagnostic.run(sequence)
    write_json('diagnostics/teacher_state_v1/RECOVERY_ATTEMPT_V2_COMPLETE.json',{'sequence':sequence,
        'attempt_receipt_SHA':sha256(OUT/'diagnostics/teacher_state_v1/RECOVERY_ATTEMPT_V2.json'),
        'original_partial_still_same_SHA':sha256(partial)==record['retained_partial_SHA'],'status':'COMPLETE'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',required=True);main(p.parse_args().sequence)
