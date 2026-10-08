#!/usr/bin/env python3
"""SHA-identical fast-forward Git publication through existing gh auth.

Fallback for a host that can reach api.github.com but not Git transport.
No credential is read, printed or saved by this program. gh owns existing
authentication. Every blob/tree/commit must retain its local object SHA;
metadata normalization or a competing remote update is a hard failure.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import re

from scripts.n72r20r4_common import *

REPOSITORY="LYQ1107/InterMOT"
BRANCH="codex/n72r20r4-causal-identity-trajectory-transfer"
API_ROOT=f"repos/{REPOSITORY}/git"


def git(*args: str, data: bytes | None=None) -> bytes:
    return subprocess.run(["git",*args],cwd=ROOT,input=data,stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,check=True).stdout


def api(method: str, path: str, body: dict | None=None) -> dict:
    if not path.startswith(API_ROOT+"/"):raise ValueError("publisher repository scope exceeded")
    command=["gh","api","--method",method,path]
    if body is not None:command.extend(["--input","-"])
    environment=dict(os.environ,GH_DEBUG="",GH_PROMPT_DISABLED="1")
    for attempt in range(1,13):
        try:
            result=subprocess.run(command,input=None if body is None else json.dumps(body).encode(),
                                  stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=environment,timeout=30)
        except subprocess.TimeoutExpired:
            if attempt<12:continue
            raise RuntimeError(f"GitHub API {method} timeout after twelve attempts; no branch success claimed") from None
        if result.returncode==0:return json.loads(result.stdout)
        # Report only classified errors, never credential/debug/header text.
        error=result.stderr.decode(errors="replace")
        status=re.search(r"HTTP ([0-9]{3})",error)
        network=any(s in error.lower() for s in ("eof","timeout","reset by peer","dial tcp","tls","connection"))
        if network and attempt<12:continue
        raise RuntimeError(f"GitHub API {method} failed: HTTP={None if status is None else status[1]}, network_error={network}, exit={result.returncode}; no branch success claimed")


def tree_entries(commit: str) -> dict[str,dict]:
    records={}
    for line in git("ls-tree","-r","-z",commit).split(b"\0"):
        if not line:continue
        meta,path=line.split(b"\t",1);mode,kind,sha=meta.decode().split()
        records[path.decode()]={"path":path.decode(),"mode":mode,"type":kind,"sha":sha}
    return records


def git_person(value: str) -> dict:
    match=re.fullmatch(r"(.*) <([^<>]*)> ([0-9]+) ([+-][0-9]{4})",value)
    if match is None:raise ValueError("unsupported Git person metadata")
    name,email,seconds,offset=match.groups()
    minutes=int(offset[1:3])*60+int(offset[3:])
    if offset[0]=="-":minutes=-minutes
    date=datetime.fromtimestamp(int(seconds),timezone(timedelta(minutes=minutes))).isoformat()
    return {"name":name,"email":email,"date":date}


def commit_payload(raw: bytes) -> dict:
    headers,message=raw.decode("utf-8").split("\n\n",1)
    parsed={};parents=[]
    for line in headers.splitlines():
        key,value=line.split(" ",1)
        if key=="parent":parents.append(value)
        elif key in {"tree","author","committer"}:parsed[key]=value
        else:raise ValueError("unsupported signed/extended commit header; use ordinary Git transport")
    return {"tree":parsed["tree"],"parents":parents,"author":git_person(parsed["author"]),
            "committer":git_person(parsed["committer"]),"message":message}


def inline_tree_entries(changes: list[dict], uploaded: set[str]) -> list[dict]:
    """Batch unuploaded UTF-8 blobs via documented tree.content; binary stays SHA.

    Source bytes always come from sealed Git objects, not the working tree.
    The returned tree SHA still has to equal the complete local tree SHA.
    """
    entries=[]
    for record in changes:
        entry=dict(record)
        if entry["type"]=="blob" and entry["sha"] is not None and entry["sha"] not in uploaded:
            raw=git("cat-file","blob",entry["sha"])
            try:content=raw.decode("utf-8")
            except UnicodeDecodeError:pass
            else:
                if content.encode("utf-8")!=raw:raise ValueError("UTF-8 blob roundtrip changed")
                del entry["sha"]
                entry["content"]=content
        entries.append(entry)
    return entries


def run(head: str, *, inline_text: bool=False) -> dict:
    if git("branch","--show-current").decode().strip()!=BRANCH:
        raise ValueError("publisher requires the task's exact branch")
    head=git("rev-parse","--verify",head+"^{commit}").decode().strip()
    remote=api("GET",f"{API_ROOT}/ref/heads/{BRANCH}")["object"]["sha"]
    if remote==head:
        git("update-ref",f"refs/remotes/origin/{BRANCH}",head)
        receipt={"status":"PASS_ALREADY_EQUAL_VERIFIED","branch":BRANCH,"local_head":head,"remote_head":remote,
                 "force_push":False,"credentials_emitted_or_saved":False}
        write_json(ASSETS/"git_api_publication"/f"receipt__{head}.json",receipt)
        return receipt
    git("merge-base","--is-ancestor",remote,head)
    commits=git("rev-list","--reverse",f"{remote}..{head}").decode().splitlines()
    cache_root=ASSETS/"git_api_publication";cache_root.mkdir(parents=True,exist_ok=True)
    uploaded=[]
    for commit in commits:
        payload=commit_payload(git("cat-file","commit",commit))
        if len(payload["parents"])!=1:raise ValueError("fallback only handles this task's linear commits")
        parent=payload["parents"][0]
        old=tree_entries(parent);new=tree_entries(commit)
        changes=[new[p] for p in sorted(new) if new[p]!=old.get(p)]
        changes.extend({"path":p,"mode":old[p]["mode"],"type":old[p]["type"],"sha":None} for p in sorted(old.keys()-new.keys()))
        if inline_text:
            cached={p.stem.removeprefix("blob__") for p in cache_root.glob("blob__*.json")}
            changes=inline_tree_entries(changes,cached)
        blobs={r["sha"] for r in changes if r["type"]=="blob" and r.get("sha") is not None}
        def upload_blob(sha: str) -> str:
            done=cache_root/f"blob__{sha}.json"
            if done.exists():return sha
            content=git("cat-file","blob",sha)
            record=api("POST",f"{API_ROOT}/blobs",{"encoding":"base64","content":base64.b64encode(content).decode()})
            if record["sha"]!=sha:raise RuntimeError("uploaded blob SHA changed")
            write_json(done,{"sha":sha,"bytes":len(content),"verified_GitHub_response":True})
            return sha
        with ThreadPoolExecutor(max_workers=2) as pool:
            for sha in pool.map(upload_blob,sorted(blobs)):
                print(json.dumps({"phase":"blob_verified","commit":commit,"sha":sha}),flush=True)
        tree_body={"base_tree":git("rev-parse",parent+"^{tree}").decode().strip(),"tree":changes}
        print(json.dumps({"phase":"tree_upload","commit":commit,"entries":len(changes),"inline_text_entries":sum("content" in r for r in changes),"request_bytes":len(json.dumps(tree_body).encode())}),flush=True)
        tree=api("POST",f"{API_ROOT}/trees",tree_body)
        if tree["sha"]!=payload["tree"]:raise RuntimeError("uploaded tree SHA changed")
        created=api("POST",f"{API_ROOT}/commits",payload)
        write_json(cache_root/f"commit__{commit}.json",{"local_commit":commit,"GitHub_commit":created["sha"],"tree":tree["sha"],"objects_equal":created["sha"]==commit})
        if created["sha"]!=commit:
            raise RuntimeError("GitHub normalized commit metadata; remote ref remains untouched")
        uploaded.append(commit)
        print(json.dumps({"phase":"commit_SHA_verified","sha":commit}),flush=True)
    latest=api("GET",f"{API_ROOT}/ref/heads/{BRANCH}")["object"]["sha"]
    if latest!=remote:raise RuntimeError("remote changed concurrently; no ref update attempted")
    api("PATCH",f"{API_ROOT}/refs/heads/{BRANCH}",{"sha":head,"force":False})
    observed=api("GET",f"{API_ROOT}/ref/heads/{BRANCH}")["object"]["sha"]
    if observed!=head:raise RuntimeError("post-publication remote HEAD differs")
    # Normal remote-tracking update; no branch reset, checkout or tree edit.
    git("update-ref",f"refs/remotes/origin/{BRANCH}",head)
    receipt={"status":"PASS_SHA_IDENTICAL_FAST_FORWARD_PUBLICATION","branch":BRANCH,"local_head":head,
             "remote_head":observed,"previous_remote_head":remote,"uploaded_commits":uploaded,
             "all_blob_tree_commit_SHAs_preserved":True,"force_push":False,"credentials_emitted_or_saved":False}
    write_json(cache_root/f"receipt__{head}.json",receipt)
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--head",default="HEAD")
    parser.add_argument("--inline-text",action="store_true",help="batch remaining UTF-8 blobs via GitHub tree.content")
    args=parser.parse_args()
    print(json.dumps(run(args.head,inline_text=args.inline_text),sort_keys=True))
