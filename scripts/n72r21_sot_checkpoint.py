"""Single official public OSTrack checkpoint, bounded and source-recorded.

No folder snapshot, dataset, login, CAPTCHA, cookies or account credentials.
A public large-file virus-scan confirmation is followed only when its ordinary
download form still names exactly the preregistered public file and host.
"""
from __future__ import annotations
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlsplit, urlencode
from scripts.n72r21_common import ASSETS,OUT,read_json,write_json,sha256,storage,utcnow

FILE_ID='1dySfPrSk-knAEo0lTXqZ1kMk3BCvVofV'
LIMIT=512<<20


class DownloadForm(HTMLParser):
    def __init__(self):super().__init__();self.action=None;self.values={};self.active=False
    def handle_starttag(self,tag,attrs):
        values=dict(attrs)
        if tag=='form' and values.get('id')=='download-form':self.action=values.get('action');self.active=True
        if self.active and tag=='input' and values.get('name') in ('id','export','confirm','uuid'):
            self.values[values['name']]=values.get('value','')
    def handle_endtag(self,tag):
        if tag=='form':self.active=False


def run():
    source=ASSETS/'official_sources/OSTrack'
    listed=(source/'GOOGLE_DRIVE_VITB256_CE.html').read_text()
    if FILE_ID not in listed or 'OSTrack_ep0300.pth.tar' not in listed or '353 MB' not in listed:raise ValueError('official file list/size prerequisite')
    if not (source/'source/LICENSE').exists():raise ValueError('official source license prerequisite')
    destination=ASSETS/'checkpoints/OSTrack_vitb256_ce_ep300.pth.tar'
    destination.parent.mkdir(parents=True,exist_ok=True)
    manifest={'stage':'N72R21','source':'Official botaoye/OSTrack README -> Google Drive models -> vitb_256_mae_ce_32x4_ep300',
        'official_repository':'https://github.com/botaoye/OSTrack','public_file_id':FILE_ID,
        'listed_size':'353 MB','maximum_bytes':LIMIT,'destination':str(destination),'attempts':[],
        'upstream_code_license':'MIT','checkpoint_is_off_repository_public_research_release':True,
        'checkpoint_separate_redistribution_license_not_explicitly_stated':True,'public_checkpoint_republication':False,
        'no_training_or_data_download':True,'no_login_captcha_or_gated_access_bypass':True,
        'upstream_config_training_datasets':['LaSOT','GOT10K_vottrain','COCO17','TrackingNet'],
        'LaSOT_TRAIN_pilot_is_not_pretraining_unseen_data':True,'upstream_published_SHA256_available':False}
    if destination.exists():
        old=read_json(OUT/'baselines/SOT_CHECKPOINT_MANIFEST.json')
        if old.get('sha256')!=sha256(destination):raise ValueError('existing checkpoint changed')
        print({'status':'REUSED_VERIFIED_STAGE_CHECKPOINT','sha256':old['sha256']});return
    url=f'https://drive.usercontent.google.com/download?id={FILE_ID}&export=download'
    for attempt in range(7):
        direct=attempt==0;storage(LIMIT)
        partial=destination.with_suffix(destination.suffix+'.partial')
        # A failed binary partial is not overwritten or silently discarded.
        partial_exists=partial.exists() and partial.stat().st_size>1<<20
        if partial_exists:raise RuntimeError('preserved interrupted binary; resume requires checking range/ETag, no overwrite')
        output=partial if not partial.exists() else partial.with_name(partial.name+f'.attempt{attempt+1}')
        args=['curl','-4','--location','--fail','--silent','--show-error','--connect-timeout','8','--max-time','600',
              '--max-filesize',str(LIMIT),'--output',str(output),'--write-out','%{http_code}|%{size_download}|%{time_total}']
        if direct:args+=['--noproxy','*']
        started=time.monotonic();process=subprocess.Popen(args+[url],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        last_report=started
        while process.poll() is None:
            time.sleep(2)
            now=time.monotonic()
            if now-last_report>=15:
                size=output.stat().st_size if output.exists() else 0
                print(json.dumps({'attempt':attempt+1,'direct_no_proxy':direct,'downloaded_bytes':size,
                                  'elapsed_seconds':round(now-started,1),'mean_bytes_per_second':round(size/max(now-started,1))}),flush=True)
                last_report=now
        stdout,stderr=process.communicate()
        size=output.stat().st_size if output.exists() else 0
        record={'attempt':attempt+1,'direct_no_proxy':direct,'HTTP':stdout.split('|')[0],'exit_code':process.returncode,
                'bytes':size,'seconds':time.monotonic()-started,'error':stderr.strip()[:250],
                'URL_origin_path':urlsplit(url).netloc+urlsplit(url).path,'query_parameters_not_logged':True}
        manifest['attempts'].append(record)
        if process.returncode==0 and size>50<<20:
            with output.open('rb') as handle:header=handle.read(16)
            if not (header.startswith(b'PK') or header.startswith(b'\x80')):raise ValueError('not a Torch checkpoint, do not load response')
            output.replace(destination);manifest.update({'status':'OFFICIAL_PUBLIC_CHECKPOINT_DOWNLOADED_NOT_LOADER_SMOKE_YET',
                'sha256':sha256(destination),'bytes':size,'utc':utcnow(),'vendor_checksum_verified':False})
            write_json('baselines/SOT_CHECKPOINT_MANIFEST.json',manifest)
            print(json.dumps({'status':manifest['status'],'bytes':size,'sha256':manifest['sha256']}),flush=True);return
        if process.returncode==0 and 0<size<=1<<20:
            body=output.read_text(errors='replace');form=DownloadForm();form.feed(body)
            parsed=urlsplit(form.action or '')
            if parsed.scheme=='https' and parsed.hostname=='drive.usercontent.google.com' and form.values.get('id')==FILE_ID and form.values.get('export')=='download':
                url=form.action+'?'+urlencode(form.values);record['ordinary_public_virus_scan_download_form_followed']=True
            else:
                manifest['status']='NO_VALID_PUBLIC_DOWNLOAD_FORM_STOPPED_WITHOUT_AUTH_BYPASS'
                write_json('baselines/SOT_CHECKPOINT_MANIFEST.json',manifest);print({'status':manifest['status']});return
        write_json('baselines/SOT_CHECKPOINT_MANIFEST.json',manifest)
    manifest['status']='NETWORK_BLOCKED_AFTER_BOUNDED_OFFICIAL_ATTEMPTS'
    write_json('baselines/SOT_CHECKPOINT_MANIFEST.json',manifest);print({'status':manifest['status']})


if __name__=='__main__':run()
