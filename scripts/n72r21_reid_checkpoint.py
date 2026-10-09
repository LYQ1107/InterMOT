"""Bounded anonymous download of the official Torchreid Market ResNet50.

Only the public file linked from the official model zoo is accepted. This is
not a dataset download. Authentication/CAPTCHA is a stop, not a bypass target.
Failed responses are retained under unique attempt names, never overwritten.
"""
from __future__ import annotations
import json
import subprocess
import time
from urllib.parse import urlencode, urlsplit
from scripts.n72r21_common import ASSETS, read_json, sha256, storage, utcnow, write_json, OUT
from scripts.n72r21_sot_checkpoint import DownloadForm

FILE_ID = '1dUUZ4rHDWohmsQXCRe2C_HbYkzz94iBV'
LIMIT = 128 << 20


def run():
    source = ASSETS / 'official_sources/Torchreid'
    view = source / 'RESNET_REID_VIEW.html'
    if FILE_ID not in view.read_text() or 'resnet50_market_xent.pth.tar' not in view.read_text():
        raise ValueError('official public viewer/file-name prerequisite')
    if 'MIT License' not in (source / 'LICENSE').read_text():
        raise ValueError('source license prerequisite')
    destination = ASSETS / 'checkpoints/resnet50_market_xent.pth.tar'
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'baselines/SECOND_REID_CHECKPOINT_MANIFEST.json'
    manifest = {
        'stage': 'N72R21', 'official_model_zoo': 'https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html',
        'official_model': 'resnet50 / same-domain Market1501 / softmax / input 256x128',
        'public_file_id': FILE_ID, 'destination': str(destination), 'maximum_bytes': LIMIT,
        'listed_size_not_exposed_by_public_viewer': True,
        'download_limit_and_60GiB_reserve_checked_before_transfer': True,
        'source_code_license': 'MIT', 'separate_off_repository_weight_redistribution_terms_not_explicit': True,
        'weights_republished': False, 'dataset_downloaded': False,
        'upstream_published_SHA256_available': False, 'vendor_checksum_verified': False,
        'source_files': {str(p): sha256(p) for p in [view, source/'LICENSE', source/'source/resnet.py']},
        'no_login_captcha_or_gated_access_bypass': True, 'attempts': []}
    if destination.exists():
        previous = read_json(manifest_path)
        if sha256(destination) != previous['sha256']:
            raise ValueError('existing checkpoint changed')
        print({'status': 'REUSED_VERIFIED_SECOND_REID_WEIGHT'}); return
    if manifest_path.exists():
        manifest['attempts'] = read_json(manifest_path).get('attempts', [])
    url = f'https://drive.usercontent.google.com/download?id={FILE_ID}&export=download'
    for index in range(7):
        direct = index == 0
        storage(LIMIT)
        number = len(manifest['attempts']) + 1
        partial = destination.with_name(destination.name + f'.attempt{number}.partial')
        if partial.exists():
            raise FileExistsError('preserve previous failed response')
        args = ['curl', '-4', '--location', '--fail', '--silent', '--show-error', '--connect-timeout', '8',
                '--max-time', '300', '--max-filesize', str(LIMIT), '--output', str(partial),
                '--write-out', '%{http_code}|%{size_download}|%{time_total}']
        if direct: args += ['--noproxy', '*']
        started = time.monotonic()
        proc = subprocess.Popen(args+[url], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        last_report = started
        while proc.poll() is None:
            time.sleep(1)
            if time.monotonic()-last_report >= 15:
                size = partial.stat().st_size if partial.exists() else 0
                print(json.dumps({'download': 'SECOND_REID', 'attempt': number, 'bytes': size,
                                  'seconds': round(time.monotonic()-started, 1), 'direct_no_proxy': direct}), flush=True)
                last_report = time.monotonic()
        stdout, stderr = proc.communicate()
        size = partial.stat().st_size if partial.exists() else 0
        record = {'attempt': number, 'direct_no_proxy': direct, 'HTTP': stdout.split('|')[0],
                  'exit_code': proc.returncode, 'bytes': size, 'seconds': time.monotonic()-started,
                  'error': stderr.strip()[:250], 'URL_origin_path': urlsplit(url).netloc+urlsplit(url).path,
                  'query_parameters_not_logged': True}
        manifest['attempts'].append(record)
        if proc.returncode == 0 and size > 50 << 20:
            with partial.open('rb') as stream: header = stream.read(16)
            if not (header.startswith(b'PK') or header.startswith(b'\x80')):
                raise ValueError('not a Torch checkpoint; never deserialize an HTML response')
            partial.replace(destination)
            manifest.update(status='OFFICIAL_PUBLIC_WEIGHT_DOWNLOADED_LOADER_PENDING',
                            sha256=sha256(destination), bytes=size, utc=utcnow())
            write_json('baselines/SECOND_REID_CHECKPOINT_MANIFEST.json', manifest)
            print(json.dumps({'status': manifest['status'], 'bytes': size, 'sha256': manifest['sha256']}), flush=True)
            return
        if proc.returncode == 0 and 0 < size <= 1 << 20:
            form = DownloadForm(); form.feed(partial.read_text(errors='replace'))
            action = urlsplit(form.action or '')
            if (action.scheme == 'https' and action.hostname == 'drive.usercontent.google.com'
                    and form.values.get('id') == FILE_ID and form.values.get('export') == 'download'):
                url = form.action+'?'+urlencode(form.values)
                record['ordinary_public_virus_scan_confirmation'] = True
            else:
                manifest['status'] = 'PUBLIC_FILE_NOT_DOWNLOADABLE_WITHOUT_UNRESOLVED_GATE'
                write_json('baselines/SECOND_REID_CHECKPOINT_MANIFEST.json', manifest); return
        if record['HTTP'] in ('401', '403'):
            manifest['status'] = 'ACCESS_DENIED_STOPPED_NO_AUTH_BYPASS'
            write_json('baselines/SECOND_REID_CHECKPOINT_MANIFEST.json', manifest); return
        write_json('baselines/SECOND_REID_CHECKPOINT_MANIFEST.json', manifest)
    manifest['status'] = 'NETWORK_BLOCKED_AFTER_BOUNDED_PUBLIC_ATTEMPTS'
    write_json('baselines/SECOND_REID_CHECKPOINT_MANIFEST.json', manifest)


if __name__ == '__main__': run()
