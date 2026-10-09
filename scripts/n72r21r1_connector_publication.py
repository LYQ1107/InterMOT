"""Code-only same-tree API publication, preserving every offline local commit."""
import argparse
import json
from pathlib import Path
import subprocess
from scripts.n72r21r1_common import ROOT,ASSETS,OUT,read_json,write_json,sha256,utcnow
from scripts.n72r21_connector_publication import remote_commit_bytes
from scripts.n72r20r4_publish_git import commit_payload

BRANCH='codex/n72r21r1-safe-joint-mot-intervention'
DIRECTORY=ASSETS/'git_connector_publication'


def git(*args,input=None):
    return subprocess.run(['git',*args],cwd=ROOT,input=input,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=30).stdout


def entries(ref):
    # Do not reuse historical publisher's fixed ROOT for this worktree.
    result={}
    for line in git('ls-tree','-r',ref).decode().splitlines():
        metadata,path=line.split('\t',1);mode,kind,h=metadata.split(' ')
        result[path]={'path':path,'mode':mode,'type':kind,'sha':h}
    return result


def save(name,value):
    path=DIRECTORY/name;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if read_json(path)==value:return path
        raise FileExistsError('preserve previous publication attempts; explicit versioned recovery required')
    with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return path


def prepare(parent):
    assert git('branch','--show-current').decode().strip()==BRANCH
    assert not git('-c','core.fsmonitor=false','status','--porcelain').strip()
    git('merge-base','--is-ancestor',parent,'HEAD')
    head=git('rev-parse','HEAD').decode().strip();old,new=entries(parent),entries('HEAD')
    if old.keys()-new.keys():raise ValueError('no deletions in this scoped publication')
    changes=[new[p] for p in sorted(new) if new[p]!=old.get(p)]
    for r in changes:
        p=r['path'];allowed=p.startswith(('scripts/n72r21r1_','tests/test_n72r21r1_','docs/N72R21R1','sam3_intermot/one_click/','sam3_intermot/evaluation/','outputs/N72R21R1/protocol/')) or p in ('outputs/N72R21R1/FINAL_GOAL.json','outputs/N72R21R1/PREREGISTRATION.json')
        if not allowed or r['type']!='blob' or Path(p).suffix not in ('.py','.md','.json'):raise ValueError('outside current-stage code/documentation scope: '+p)
        raw=git('cat-file','blob',r['sha']);raw.decode('utf-8')
        if len(raw)>128*1024:raise ValueError('large artifact not permitted: '+p)
        r['bytes']=len(raw)
    payload=commit_payload(git('cat-file','commit',head))
    result={'repository':'LYQ1107/InterMOT','branch':BRANCH,'remote_parent':parent,
        'base_tree':git('rev-parse',parent+'^{tree}').decode().strip(),'original_local_head':head,
        'target_tree':payload['tree'],'message':payload['message'],'changes':changes,
        'no_binary_weights_media_data_or_bulk_results':True,'force':False,'original_offline_history_will_be_preserved':True}
    path=save('PREPARED.json',result)
    print(json.dumps({'prepared_path':str(path),'files':len(changes),'bytes':sum(r['bytes'] for r in changes),'tree':payload['tree'],'head':head}))


def emit_chunk(chunk):
    prepared=read_json(DIRECTORY/'PREPARED.json');selected=prepared['changes'][chunk*6:(chunk+1)*6]
    print(json.dumps([{'path':r['path'],'mode':r['mode'],'type':'blob','content':git('cat-file','blob',r['sha']).decode('utf-8')} for r in selected]))


def import_commit(path):
    record=read_json(path);prepared=read_json(DIRECTORY/'PREPARED.json')
    assert record['tree']['sha']==prepared['target_tree']
    assert [p['sha'] for p in record['parents']]==[prepared['remote_parent']]
    raw=remote_commit_bytes(record);h=git('hash-object','-t','commit','-w','--stdin',input=raw).decode().strip()
    assert h==record['sha'];save('IMPORTED.json',{'sha':h,'tree':record['tree']['sha'],'parent':prepared['remote_parent']})
    print(json.dumps({'exact_canonical_commit_imported':h,'tree_verified':True,'remote_ref_changed':False}))


def adopt(remote):
    prepared=read_json(DIRECTORY/'PREPARED.json');imported=read_json(DIRECTORY/'IMPORTED.json')
    assert imported['sha']==remote and git('rev-parse','HEAD').decode().strip()==prepared['original_local_head']
    assert not git('-c','core.fsmonitor=false','status','--porcelain').strip()
    assert git('rev-parse',remote+'^{tree}').decode().strip()==prepared['target_tree']
    backup='refs/heads/codex/n72r21r1-offline-'+prepared['original_local_head'][:12]
    git('update-ref',backup,prepared['original_local_head'],'0'*40)
    git('update-ref','refs/heads/'+BRANCH,remote,prepared['original_local_head'])
    git('update-ref','refs/remotes/origin/'+BRANCH,remote)
    assert not git('-c','core.fsmonitor=false','status','--porcelain').strip()
    record={'utc':utcnow(),'repository':'LYQ1107/InterMOT','branch':BRANCH,'local_HEAD':remote,'fresh_remote_HEAD':remote,
        'tree':prepared['target_tree'],'clean_worktree_fsmonitor_disabled':True,'force_push':False,
        'preserved_original_local_HEAD':prepared['original_local_head'],'preserved_backup_ref':backup,
        'same_tree_no_working_file_rewrite':True,'no_assets_or_credentials_published':True,'scientific_goal_complete':False,'next_stage_authorized':False}
    save('receipt__'+remote+'.json',record);write_json('git_delivery/PUBLISH_VERIFICATION__'+remote+'.json',record)
    print(json.dumps(record))


if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare');mode.add_argument('--emit-chunk',type=int);mode.add_argument('--import-commit');mode.add_argument('--adopt')
    a=parser.parse_args()
    if a.prepare:prepare(a.prepare)
    elif a.emit_chunk is not None:emit_chunk(a.emit_chunk)
    elif a.import_commit:import_commit(a.import_commit)
    else:adopt(a.adopt)
