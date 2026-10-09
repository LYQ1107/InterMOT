"""Resume only missing frozen official CHIRLA CSV blobs, not dataset pixels.

Initial direct-no-proxy and existing-proxy failures remain in the original
download manifest. This bounded fallback uses the existing GitHub API client
without reading or printing its authentication configuration.
"""
import base64
import hashlib
import json
import subprocess
from scripts.n72r21_common import ASSETS,OUT,read_json,write_json,sha256,storage,utcnow


def run():
    root=ASSETS/'official_sources/CHIRLA';tree=read_json(root/'TREE.json')
    previous=OUT/'datasets/CHIRLA_METADATA_RESUME.json'
    records=read_json(previous)['records'] if previous.exists() else []
    for entry in tree['tree']:
        name=entry['path'];path=root/name
        if entry['type']!='blob' or not name.startswith('benchmark/metadata/') or not name.endswith('.csv') or path.exists():continue
        if entry.get('size',0)>8<<20:raise ValueError('not bounded metadata')
        row={'path':str(path),'official_git_blob_sha1':entry['sha'],'metadata_only':True,'attempts':[]}
        for attempt in range(3):
            try:
                response=subprocess.run(['gh','api',f"repos/bdager/CHIRLA/git/blobs/{entry['sha']}"],capture_output=True,text=True,timeout=12)
            except subprocess.TimeoutExpired:
                row['attempts'].append({'attempt':attempt+1,'transport':'EXISTING_GH_API_AFTER_RECORDED_DIRECT_FAILURE','status':'TIMEOUT_12S'});continue
            if response.returncode:
                row['attempts'].append({'attempt':attempt+1,'transport':'EXISTING_GH_API_AFTER_RECORDED_DIRECT_FAILURE','status':'FAILED','error':response.stderr.strip()[:160]});continue
            content=json.loads(response.stdout)
            if content['encoding']!='base64':raise ValueError('unexpected official blob encoding')
            data=base64.b64decode(content['content'])
            blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if blob!=entry['sha'] or len(data)!=entry['size']:raise ValueError('official frozen blob verification')
            storage(len(data));path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as handle:handle.write(data)
            row.update({'status':'DOWNLOADED_OFFICIAL_FROZEN_BLOB','sha256':sha256(path),'bytes':len(data)});break
        row.setdefault('status','NETWORK_BLOCKED_NO_FAKE_DATA');records.append(row)
        write_json('datasets/CHIRLA_METADATA_RESUME.json',{'utc':utcnow(),'frozen_git_tree_sha1':tree['sha'],'records':records,
            'no_media_downloads':True,'no_final_query_pixels_accessed':True,'original_attempts_manifest':'outputs/N72R21/datasets/DOWNLOAD_MANIFEST.json'})
        print(json.dumps({'name':name,'status':row['status']}),flush=True)


if __name__=='__main__':run()
