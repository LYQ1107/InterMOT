"""Verify public GitHub-connector metadata bytes before placing stage assets.

Only public UTF-8 CSV/code/document blobs are accepted, never media, weights,
credentials, a arbitrary repository path or an existing changed file. This
alternative transport preserves original CRLF and verifies Git's blob hash.
"""
import argparse
import base64
import hashlib
from pathlib import Path
from scripts.n72r21_common import ASSETS,storage,sha256,write_json,read_json,OUT


def ingest(relative, blob_sha, payload):
    rel=Path(relative)
    if not rel.parts or rel.parts[0] not in ('CHIRLA','OSTrack'):raise ValueError('official source allowlist')
    if rel.suffix.lower() not in ('.csv','.py','.md','.yaml','.txt','.json') and rel.name!='LICENSE':raise ValueError('text metadata only')
    tree=read_json(ASSETS/'official_sources'/rel.parts[0]/'TREE.json')
    expected={e['path']:e['sha'] for e in tree['tree'] if e['type']=='blob'}
    if expected.get(str(Path(*rel.parts[1:])))!=blob_sha:raise ValueError('blob not in frozen official Git tree')
    target=ASSETS/'official_sources'/rel
    if not target.resolve().is_relative_to((ASSETS/'official_sources').resolve()):raise ValueError('metadata path escape')
    body=base64.b64decode(payload,validate=True)
    if len(body)>1<<20:raise ValueError('bounded public metadata required')
    body.decode('utf-8')
    actual=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
    if actual!=blob_sha:raise ValueError('official Git blob SHA mismatch')
    if target.exists():
        if target.read_bytes()!=body:raise ValueError('existing changed source: no overwrite')
    else:
        storage(len(body));target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as handle:handle.write(body)
    row={'path':str(target),'official_git_blob_sha1':actual,'bytes':len(body),'sha256':sha256(target),
         'transport':'OFFICIAL_PUBLIC_GITHUB_CONNECTOR_AFTER_LOCAL_TRANSPORT_FAILURE','media':False}
    manifest=OUT/'datasets/CONNECTOR_METADATA_INGEST.json'
    result=read_json(manifest) if manifest.exists() else {'records':[],'media_downloaded':False}
    result['records']=[r for r in result['records'] if r['path']!=str(target)]+[row]
    write_json('datasets/CONNECTOR_METADATA_INGEST.json',result)
    print({'path':relative,'blob_verified':True,'bytes':len(body)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--relative',required=True);p.add_argument('--blob-sha',required=True);p.add_argument('--base64',required=True)
    args=p.parse_args();ingest(args.relative,args.blob_sha,args.base64)
