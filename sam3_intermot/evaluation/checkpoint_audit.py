"""Restricted local checkpoint integrity/strict loading, not scientific success."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import torch


def verify_checkpoint(path,expected_sha256,expected_schema,model):
    path=Path(path);digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):digest.update(block)
    if digest.hexdigest()!=expected_sha256:
        raise ValueError('checkpoint SHA mismatch before deserialization')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(payload,dict) or 'model' not in payload or 'schema' not in payload:
        raise ValueError('checkpoint is missing model/schema')
    # JSON serialization canonicalizes string/int metadata dictionary keys;
    # it must not change or overwrite the original checkpoint/fit record.
    if json.loads(json.dumps(payload['schema'],allow_nan=False))!=expected_schema:
        raise ValueError('checkpoint schema differs from its fit record')
    for name,value in payload['model'].items():
        if not isinstance(value,torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f'nonfinite/non-tensor model state: {name}')
    model.load_state_dict(payload['model'],strict=True)
    model.eval()
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':expected_sha256,
        'strict_state_dict_loaded':True,'all_model_tensors_finite':True,
        'schema_matches_actual_fit_record':True,'weights_only_deserialization':True,
        'epoch':payload.get('epoch'),'seed':payload.get('seed'),
        'model_parameters':sum(p.numel() for p in model.parameters()),
        'optimizer_and_rng_state_saved':all(k in payload for k in ('optimizer','rng')),
        'new_training':False,'scientific_success_from_loader':False}
