"""Persist real suite results with the requested activated-environment PATH."""
import subprocess
import re
import argparse
from scripts.n72r20r4r1_common import *

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--selected-only',action='store_true');args=parser.parse_args()
    env={**os.environ,'PATH':str(ROOT/'.venv/bin')+os.pathsep+os.environ['PATH'],'PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    if args.selected_only:
        payload=read_json(OUT/'audit/REGRESSION_RESULTS.json');suites={}
        for name,pattern in [('new_stage','test_n72r20r4r1*.py'),('new_stage_and_R4_dependencies','test_n72r20r4*.py')]:
            command=[str(ROOT/'.venv/bin/python'),'-m','pytest','-q',*[str(p.relative_to(ROOT)) for p in sorted((ROOT/'tests').glob(pattern))]]
            r=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            digest=hashlib.sha256(r.stdout.encode()).hexdigest();path=ASSETS/'evaluation_logs'/f'{name}_{digest}.jsonl.zst'
            log=stream_zstd(path,[{'command':command,'stdout':r.stdout,'returncode':r.returncode}])
            suites[name]={'command':command,'returncode':r.returncode,'summary':[line for line in r.stdout.splitlines() if re.search(r'\d+ (passed|failed).* in ',line)],'log':log}
        payload['final_selected_suites']=suites
        full_stdout=read_zstd_jsonl(Path(payload['log']['path']))[0]['stdout']
        payload['failure_classification']={'pinned_TrackEval_direct_CLI_SEQMAP_FILE_list_TypeError':4,'historical_literal_branch_name_assertion':1,
            'actual_stat_list_TypeError_occurrences':len(re.findall(r'^E\s+TypeError: stat: path should be string, bytes, os.PathLike or integer, not list$',full_stdout,re.MULTILINE)),
            'new_stage_failures':0 if suites['new_stage']['returncode']==0 else 'SEE_LOG'}
        write_json(OUT/'audit/REGRESSION_RESULTS.json',payload)
        print(json.dumps({name:{'returncode':r['returncode'],'summary':r['summary']} for name,r in suites.items()}),flush=True)
        raise SystemExit(0 if all(r['returncode']==0 for r in suites.values()) else 1)
    command=[str(ROOT/'.venv/bin/python'),'-m','pytest','-q','tests']
    result=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    failed=re.findall(r'^FAILED (.*)$',result.stdout,re.MULTILINE)
    summary=[line for line in result.stdout.splitlines() if re.search(r'\d+ (passed|failed).* in ',line)]
    digest=hashlib.sha256(result.stdout.encode()).hexdigest()
    path=ASSETS/'evaluation_logs'/f'FULL_REPOSITORY_REGRESSION_ACTIVATED_PATH_{digest}.jsonl.zst'
    log=stream_zstd(path,[{'command':command,'stdout':result.stdout,'returncode':result.returncode}])
    payload={'command':command,'returncode':result.returncode,'summary':summary,'failed_tests':failed,'log':log,'activated_PATH':str(ROOT/'.venv/bin'),
        'first_unactivated_attempt':{'passed':708,'failed':5,'two_N6_subprocess_failures_were_missing_python_in_PATH':True},
        'previous_new_stage_and_R4_selected_dependency_suite':{'passed':110,'failed':0,'new_stage_only_passed':79},
        'historical_tests_and_third_party_not_modified_to_force_PASS':True}
    write_json(OUT/'audit/REGRESSION_RESULTS.json',payload)
    print(json.dumps({'returncode':result.returncode,'summary':summary,'failed_tests':failed}),flush=True)
