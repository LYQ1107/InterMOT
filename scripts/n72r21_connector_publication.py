"""Verified same-tree GitHub-connector delivery without losing offline commits.

The connector chooses commit authorship/time, unlike the exact-object CLI
publisher. Import its actual hash-verified commit and preserve every unpublished
local commit under a new backup ref before adopting the identical tree. Never
force a remote ref, alter the working tree, hide a hash mismatch or lose history.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from scripts.n72r21_common import ROOT, ASSETS, OUT, sha256, utcnow, read_json
from scripts.n72r20r4_publish_git import tree_entries, commit_payload

BRANCH='codex/n72r21-oneclick-longterm-identity'
DIRECTORY=ASSETS/'git_connector_publication'


def git(*args,input=None):
    return subprocess.run(['git',*args],cwd=ROOT,input=input,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=30).stdout


def digest(raw):
    return hashlib.sha1(b'commit '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def remote_commit_bytes(record):
    if record.get('verification',{}).get('signature'):
        raise ValueError('signed remote commit requires an exact raw-object import; do not guess headers')
    def person(r):
        if any(c in r['name']+r['email'] for c in '\r\n\0<>'):
            raise ValueError('unsupported Git person metadata')
        d=datetime.fromisoformat(r['date'].replace('Z','+00:00'))
        if d.utcoffset() is None:raise ValueError('timezone required')
        minutes=int(d.utcoffset().total_seconds()/60)
        offset=f'{"+" if minutes>=0 else "-"}{abs(minutes)//60:02d}{abs(minutes)%60:02d}'
        return f"{r['name']} <{r['email']}> {int(d.timestamp())} {offset}"
    headers=[f"tree {record['tree']['sha']}"]+[f"parent {r['sha']}" for r in record['parents']]
    headers.extend(['author '+person(record['author']),'committer '+person(record['committer'])])
    raw=('\n'.join(headers)+'\n\n'+record['message']).encode('utf-8')
    # GitHub's JSON view may omit the final message newline. The hash, not a
    # formatting assumption, decides which exact byte string is accepted.
    for candidate in (raw,raw+b'\n'):
        if digest(candidate)==record['sha']:return candidate
    raise ValueError('remote metadata does not reconstruct the actual commit SHA')


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def prepare(remote_parent):
    if git('branch','--show-current').decode().strip()!=BRANCH:raise ValueError('exact task branch required')
    if git('-c','core.fsmonitor=false','status','--porcelain').strip():raise ValueError('clean committed worktree required')
    git('merge-base','--is-ancestor',remote_parent,'HEAD')
    head=git('rev-parse','HEAD').decode().strip();old=tree_entries(remote_parent);new=tree_entries(head)
    if old.keys()-new.keys():raise ValueError('connector publication does not delete any repository files')
    changes=[new[p] for p in sorted(new) if new[p]!=old.get(p)]
    if any(r['type']!='blob' for r in changes):raise ValueError('no submodule changes in this delivery')
    allowed=lambda p: p.startswith(('outputs/N72R21/','docs/N72R21','scripts/n72r21_','sam3_intermot/evaluation/','tests/test_n72r21_')) or p=='research_log.md'
    if any(not allowed(r['path']) for r in changes):raise ValueError('publication exceeds current-stage file scope')
    known={r['sha'] for r in old.values() if r['type']=='blob'};objects=[]
    for h in sorted({r['sha'] for r in changes}-known):
        raw=git('cat-file','blob',h);content=raw.decode('utf-8')
        paths=[r['path'] for r in changes if r['sha']==h]
        if any(Path(p).suffix not in {'.py','.json','.md','.xml'} for p in paths):raise ValueError('no binary/raw media/weights publication')
        objects.append({'sha':h,'bytes':len(raw),'UTF16_code_units':len(content.encode('utf-16-le'))//2,'paths':paths})
    payload=commit_payload(git('cat-file','commit',head))
    manifest={'utc':utcnow(),'repository':'LYQ1107/InterMOT','branch':BRANCH,'remote_parent':remote_parent,
              'original_local_head':head,'target_tree':payload['tree'],'message':payload['message'],
              'changes':changes,'objects':objects,'force':False,'no_source_tree_or_old_history_rewrite':True,
              'original_unpublished_commits_will_be_kept_under_a_new_local_backup_ref':True}
    save(DIRECTORY/'PREPARED.json',manifest)
    print(json.dumps({'manifest':str(DIRECTORY/'PREPARED.json'),'objects':len(objects),'changed_files':len(changes),
                      'bytes':sum(r['bytes'] for r in objects),'target_tree':payload['tree'],'original_local_head':head}))


def import_commit(path):
    record=read_json(path);prepared=read_json(DIRECTORY/'PREPARED.json')
    if record['tree']['sha']!=prepared['target_tree'] or [r['sha'] for r in record['parents']]!=[prepared['remote_parent']]:
        raise ValueError('canonical commit tree or parent differs from prepared delivery')
    raw=remote_commit_bytes(record);h=git('hash-object','-t','commit','-w','--stdin',input=raw).decode().strip()
    if h!=record['sha']:raise ValueError('actual imported commit SHA differs')
    save(DIRECTORY/'IMPORTED.json',{'sha':h,'tree':record['tree']['sha'],'parent':prepared['remote_parent'],
                                  'metadata_bytes_hash_verified':True,'remote_ref_updated_by_this_command':False})
    print(json.dumps({'canonical_commit_imported_exactly':h,'tree_verified':True,'remote_ref_changed':False}))


def adopt(remote_sha):
    prepared=read_json(DIRECTORY/'PREPARED.json');imported=read_json(DIRECTORY/'IMPORTED.json')
    if imported['sha']!=remote_sha:raise ValueError('read-back remote SHA differs from imported canonical object')
    if git('rev-parse','HEAD').decode().strip()!=prepared['original_local_head'] or git('-c','core.fsmonitor=false','status','--porcelain').strip():
        raise ValueError('local work changed concurrently; never overwrite it')
    if git('rev-parse',remote_sha+'^{tree}').decode().strip()!=prepared['target_tree']:
        raise ValueError('same-tree adoption required')
    archive='refs/heads/codex/n72r21-offline-delivery-'+prepared['original_local_head'][:12]
    # Ref creation is guarded; this preserves original offline ancestry, including
    # the earlier preparation commit, rather than amending or resetting it away.
    git('update-ref',archive,prepared['original_local_head'],'0'*40)
    git('update-ref','refs/heads/'+BRANCH,remote_sha,prepared['original_local_head'])
    git('update-ref','refs/remotes/origin/'+BRANCH,remote_sha)
    if git('-c','core.fsmonitor=false','status','--porcelain').strip():raise ValueError('unexpected dirty worktree after same-tree adoption')
    receipt={'utc':utcnow(),'repository':'LYQ1107/InterMOT','branch':BRANCH,'local_head':remote_sha,'remote_head':remote_sha,
             'tree':prepared['target_tree'],'fsmonitor_disabled_clean_worktree':True,'force_push':False,
             'original_unpublished_local_head':prepared['original_local_head'],'preserved_local_backup_ref':archive,
             'same_source_tree_verified':True,'remote_commit_metadata_imported_not_falsely_claimed_same_as_original_offline_commit':True,
             'old_published_ancestry_preserved':True,'credentials_emitted_or_saved':False,'next_stage_authorized':False}
    save(DIRECTORY/f'receipt__{remote_sha}.json',receipt)
    save(OUT/'git_delivery/FINAL_PUBLISH_VERIFICATION.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare');mode.add_argument('--import-commit');mode.add_argument('--adopt')
    a=parser.parse_args()
    if a.prepare:prepare(a.prepare)
    elif a.import_commit:import_commit(a.import_commit)
    else:adopt(a.adopt)
