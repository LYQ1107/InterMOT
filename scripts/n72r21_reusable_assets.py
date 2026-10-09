"""SHA-check sealed model lineage, without running any old stage writer."""
from pathlib import Path
import json
from scripts.n72r21_common import ROOT, OUT, write_json, read_json, sha256, utcnow


def records(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str) and len(value['sha256'])==64:
            yield value
        for child in value.values(): yield from records(child)
    elif isinstance(value,list):
        for child in value: yield from records(child)


def run():
    source=read_json(ROOT/'outputs/N72R20R4R1/checkpoints/CHECKPOINT_MANIFEST.json')
    checked=[];seen=set()
    for record in records(source):
        path=Path(record['path'])
        if not path.is_absolute(): path=ROOT/path
        if str(path) in seen: continue
        seen.add(str(path))
        if not path.exists():raise FileNotFoundError(path)
        if sha256(path)!=record['sha256']:raise ValueError(f'Historical checkpoint changed: {path}')
        checked.append({'path':str(path),'sha256':record['sha256'],'bytes':path.stat().st_size,
                        'actual_fit':record.get('actual_training_sequences',record.get('fit')),
                        'role':'HISTORICAL_COMPARATOR_NOT_N72R21_SUCCESS','copied':False})
    encoder=ROOT.parent/'InterMOT_N72R16_assets/checkpoints/osnet_x1_0_market1501.pth'
    if sha256(encoder)!=source['encoder_sha256']:raise ValueError('Encoder SHA changed')
    result={'utc':utcnow(),'historical_checkpoint_count_verified':len(checked),'models':checked,
            'OSNet':{'path':str(encoder),'sha256':source['encoder_sha256'],'bytes':encoder.stat().st_size,'frozen':True},
            'dancetrack':{'root':str(ROOT.parent/'InterMOT_N72R16_assets/dataset'),'train_sequences':40,'val_sequences':25,'copied':False},
            'train_candidate_tape':str(ROOT.parent/'InterMOT_N72R20R2_assets'),
            'historical_val_candidate_tape':{'root':str(ROOT.parent/'InterMOT_N72R20R3R3_assets/val_candidates'),'media_read_for_development':False},
            'unavailable_required_assets':['Verified lawful CHIRLA pilot media/annotations and complete official split metadata','Original complete LaSOT/person sequences and original visibility labels','Licensed public SOT checkpoint and independent encoder control'],
            'no_dataset_or_weight_binaries_in_git':True}
    write_json('historical_audit/REUSABLE_ASSETS.json',result)
    print(json.dumps({'verified_models':len(checked),'encoder_sha256':source['encoder_sha256']}))


if __name__=='__main__':run()
