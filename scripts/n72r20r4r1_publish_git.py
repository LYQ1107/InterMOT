"""Stage-scoped SHA-identical publication using existing authenticated gh."""
from __future__ import annotations
import argparse
from scripts import n72r20r4_publish_git as publisher
from scripts.n72r20r4r1_common import BRANCH,ASSETS,SOURCE_SHA


def run(head="HEAD"):
    publisher.BRANCH=BRANCH;publisher.ASSETS=ASSETS
    refs=publisher.api("GET",publisher.API_ROOT+"/matching-refs/heads/"+BRANCH)
    exact=[r for r in refs if r["ref"]=="refs/heads/"+BRANCH]
    if len(exact)>1:raise RuntimeError("ambiguous branch ref")
    if not exact:
        # Publish only the audited source first. Never reset an existing ref.
        publisher.git("merge-base","--is-ancestor",SOURCE_SHA,head)
        created=publisher.api("POST",publisher.API_ROOT+"/refs",{"ref":"refs/heads/"+BRANCH,"sha":SOURCE_SHA})
        if created["object"]["sha"]!=SOURCE_SHA:raise RuntimeError("new remote branch did not start at audited source")
    return publisher.run(head,inline_text=True,tree_chunk_bytes=196608)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--head",default="HEAD");args=parser.parse_args()
    print(publisher.json.dumps(run(args.head),sort_keys=True))
