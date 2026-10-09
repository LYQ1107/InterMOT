"""N72R21-scoped non-force SHA-identical publication via existing gh auth."""
from scripts import n72r20r4_publish_git as publisher
from scripts.n72r21_common import ASSETS

BRANCH='codex/n72r21-oneclick-longterm-identity'
SOURCE_SHA='becfdf9c2b317e7c8b5427a894bc1f64b572809f'


def run(head='HEAD'):
    publisher.BRANCH=BRANCH;publisher.ASSETS=ASSETS
    refs=publisher.api('GET',publisher.API_ROOT+'/matching-refs/heads/'+BRANCH)
    exact=[r for r in refs if r['ref']=='refs/heads/'+BRANCH]
    if len(exact)>1:raise RuntimeError('ambiguous branch')
    if not exact:
        publisher.git('merge-base','--is-ancestor',SOURCE_SHA,head)
        created=publisher.api('POST',publisher.API_ROOT+'/refs',{'ref':'refs/heads/'+BRANCH,'sha':SOURCE_SHA})
        if created['object']['sha']!=SOURCE_SHA:raise RuntimeError('incorrect remote branch base')
    return publisher.run(head,inline_text=True,tree_chunk_bytes=196608)


if __name__=='__main__':print(publisher.json.dumps(run(),sort_keys=True))
