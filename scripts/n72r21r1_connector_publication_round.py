"""Versioned code-only deliveries and hash-verified GitHub timezone recovery.

GitHub JSON normalizes dates to UTC, while raw Git person headers may retain
another offset. Only an exact SHA match permits import; metadata guesses never
authorize a different object, branch force, or working-file rewrite.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import itertools
import json
import re
from pathlib import Path
from scripts import n72r21r1_connector_publication as publication
from scripts.n72r21_connector_publication import remote_commit_bytes


def verified_dates(record):
    if record.get('verification',{}).get('signature'):raise ValueError('signed commit requires exact raw-object import')
    try:
        remote_commit_bytes(record);return deepcopy(record)
    except ValueError:pass
    instants={k:datetime.fromisoformat(record[k]['date'].replace('Z','+00:00')) for k in ('author','committer')}
    if any(t.utcoffset() is None for t in instants.values()):raise ValueError('GitHub commit dates require timezone')
    offsets=[0,480]+[i for i in range(-840,841,15) if i not in (0,480)]
    # Shared offsets first; each trial changes presentation, not the UTC instant.
    trials=[(i,i) for i in offsets]+[(a,c) for a,c in itertools.product(offsets,repeat=2) if a!=c]
    for author,committer in trials:
        candidate=deepcopy(record)
        for k,offset in zip(('author','committer'),(author,committer),strict=True):
            candidate[k]['date']=instants[k].astimezone(timezone(timedelta(minutes=offset))).isoformat()
        try:remote_commit_bytes(candidate);return candidate
        except ValueError:continue
    raise ValueError('no raw commit bytes match remote SHA; preserve attempt, do not adopt')


def configure(namespace):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',namespace):raise ValueError('safe explicit delivery namespace required')
    publication.DIRECTORY=publication.ASSETS/'git_connector_publication'/namespace


def import_verified(path):
    original=publication.read_json(path);verified=verified_dates(original)
    publication.save('ORIGINAL_REMOTE_COMMIT.json',original)
    verified_path=publication.save('CANONICAL_REMOTE_COMMIT_WITH_VERIFIED_OFFSETS.json',verified)
    publication.import_commit(verified_path)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--namespace',required=True);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare');mode.add_argument('--emit-chunk',type=int);mode.add_argument('--import-commit');mode.add_argument('--adopt')
    args=parser.parse_args();configure(args.namespace)
    if args.prepare:publication.prepare(args.prepare)
    elif args.emit_chunk is not None:publication.emit_chunk(args.emit_chunk)
    elif args.import_commit:import_verified(Path(args.import_commit))
    else:publication.adopt(args.adopt)
