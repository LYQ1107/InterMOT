"""Extract only verified official OSTrack inference source/configuration.

The small official code archive is not a dataset snapshot. Original paths,
bytes and Git blob hashes are preserved. Unrelated assets/train datasets and
all links are excluded; no historical repository or package is patched.
"""
import hashlib
from pathlib import PurePosixPath
import tarfile
from scripts.n72r21_common import ASSETS,read_json,write_json,sha256,storage


def run():
    base=ASSETS/'official_sources/OSTrack';tree=read_json(base/'TREE.json')
    entries={e['path']:e for e in tree['tree'] if e['type']=='blob'}
    archive=base/'SOURCE_MAIN.tar.gz';dest=base/'source';records=[]
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar.getmembers():
            if not member.isfile():continue
            p=PurePosixPath(member.name)
            if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe archive path')
            rel=PurePosixPath(*p.parts[1:]);name=str(rel)
            selected=name in ('LICENSE','README.md','lib/__init__.py','lib/models/__init__.py','lib/config/__init__.py','lib/utils/__init__.py','lib/train/__init__.py','lib/train/data/__init__.py')
            selected=selected or name.startswith(('lib/models/','lib/utils/','lib/config/ostrack/')) and name.endswith('.py')
            selected=selected or name in ('lib/train/data/processing_utils.py','lib/test/utils/hann.py',
                                         'lib/test/tracker/ostrack.py','lib/test/tracker/data_utils.py')
            selected=selected or name.startswith('experiments/ostrack/') and name.endswith('.yaml')
            if not selected:continue
            if name not in entries or member.size>1<<20:raise ValueError('unexpected source entry')
            body=tar.extractfile(member).read()
            blob=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
            if blob!=entries[name]['sha']:raise ValueError('official source archive differs from frozen Git tree')
            path=dest/name
            if not path.resolve().is_relative_to(dest.resolve()):raise ValueError('source destination escape')
            if path.exists():
                if path.read_bytes()!=body:raise ValueError('existing changed source: do not overwrite')
            else:
                storage(len(body));path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as handle:handle.write(body)
            records.append({'path':str(path),'relative':name,'bytes':len(body),'sha256':sha256(path),'official_git_blob_sha1':blob})
    write_json('baselines/SOT_SOURCE_MANIFEST.json',{'source':'https://github.com/botaoye/OSTrack',
        'git_tree_sha1_not_commit':tree['sha'],'archive_sha256':sha256(archive),'archive_bytes':archive.stat().st_size,
        'files':records,'official_source_bytes_changed':False,'dataset_or_unrelated_asset_extracted':False,
        'code_license':'MIT; separate off-repository checkpoint terms still inspected separately',
        'actual_sot_inference_executed':False})
    print({'verified_official_inference_source_files':len(records),'bytes':sum(r['bytes'] for r in records)})


if __name__=='__main__':run()
