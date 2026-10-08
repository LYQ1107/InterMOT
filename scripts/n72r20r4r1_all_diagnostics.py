"""Await actual sealed outer results, then finish fixed posthoc analytics."""
import time
from scripts.n72r20r4r1_common import *
from scripts.n72r20r4r1_trajectory_diagnostics import sequence_audit
from scripts.n72r20r4r1_proposal_diagnostics import run_fold as proposal
from scripts.n72r20r4r1_memory_audit import run_fold as memory
from scripts.n72r20r4r1_combine import run as combine
from scripts.n72r20r4r1_analytics import run as analytics

if __name__=='__main__':
    torch.set_num_threads(1)
    for s in SEQUENCES:
        path=OUT/'authority/outer'/f'{s}.json'
        while not path.exists():time.sleep(5)
        while True:
            try:
                if read_json(path)['status']=='COMPLETE':break
            except (ValueError,KeyError):pass
            time.sleep(1)
        if not (OUT/'target_tracking/ownership_audits'/f'{s}.json').exists():sequence_audit(s)
        proposal(s);memory(s)
        print(json.dumps({'completed_posthoc':s,'no_selection_feedback':True}),flush=True)
    combine();analytics();print('ALL_STRICT_DEV_AND_R0_R8_ANALYTICS_COMPLETE',flush=True)
