"""R21R1-scoped writers; historical artifacts are read-only inputs."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/N72R21R1'
HISTORY = Path(os.environ.get('N72R21R1_HISTORY_ROOT', str(ROOT.parent / 'InterMOT'))).resolve()
ASSETS = Path(os.environ.get('N72R21R1_ASSET_ROOT', str(ROOT.parent / 'InterMOT_N72R21R1_assets'))).resolve()
OLD_ASSETS = ROOT.parent / 'InterMOT_N72R21_assets'
CANDIDATES = ROOT.parent / 'InterMOT_N72R20R2_assets'
TRAIN = ROOT.parent / 'InterMOT_N72R16_assets/dataset/train'


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def storage(reserve_bytes=0):
    free = shutil.disk_usage(ROOT).free
    if free - reserve_bytes < 60 * (1 << 30):
        raise RuntimeError('R21R1 60GiB personal-mount reserve would be violated')
    return {'utc': utcnow(), 'free_bytes': free, 'free_GiB': free / (1 << 30), 'floor_GiB': 60}


def output_path(relative):
    path = OUT / relative
    if not path.resolve().is_relative_to(OUT.resolve()) or path.resolve().is_relative_to(HISTORY):
        raise ValueError('R21R1 output cannot escape or write into historical repository')
    return path


def write_json(relative, value, *, mutable=False):
    path = output_path(relative)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if path.exists() and not mutable:
        if read_json(path) == value:
            return path
        raise FileExistsError('Preserve completed/partial R21R1 evidence; version changes explicitly')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    if temporary.exists():
        raise FileExistsError('Preserve partial R21R1 temporary output')
    temporary.write_text(payload)
    temporary.replace(path)
    return path


def update_status(**fields):
    status = read_json(OUT / 'stage_status.json')
    status.update(fields, last_progress_utc=utcnow())
    write_json('stage_status.json', status, mutable=True)


def historical(path):
    return HISTORY / 'outputs/N72R21' / path
