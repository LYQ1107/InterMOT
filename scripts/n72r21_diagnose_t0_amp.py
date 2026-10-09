"""Reproduce the failed TRAIN minibatch; no weights/checkpoints are selected."""
import hashlib
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.n72r21_train_t0 import T0Frames,collate,split,forward,loss
from sam3_intermot.one_click.acib import ACIBNetwork
from scripts.n72r21_sot import checked_device
from scripts.n72r21_common import OUT,ROOT,read_json,write_json,sha256


def run():
    torch.set_num_threads(1);seed=72102;outer='dancetrack0002'
    sequences=read_json(OUT/'protocol/F1_WITHIN_VIDEO_DEVELOPMENT.json')['sequences']
    fits,inner=split(outer,sequences);data=T0Frames(fits,stride=5)
    device=checked_device(0);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    generator=torch.Generator().manual_seed(seed);model=ACIBNetwork().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scaler=torch.amp.GradScaler('cuda')
    for index,source in enumerate(DataLoader(data,batch_size=128,shuffle=True,generator=generator,collate_fn=collate,num_workers=0)):
        batch={k:v.to(device) for k,v in source.items()};optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):out=forward(model,batch);parts=loss(out,batch)
        scaler.scale(parts['total']).backward();scaler.unscale_(optimizer)
        failed=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        if not failed:
            torch.nn.utils.clip_grad_norm_(model.parameters(),25.,error_if_nonfinite=True);scaler.step(optimizer);scaler.update();continue
        scale=float(scaler.get_scale());comparisons={}
        for name,enabled,trial_scale in [('FP32_UNSCALED',False,1.),('FP16_SCALE1024',True,1024.)]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=enabled):trial=loss(forward(model,batch),batch)
            (trial['total']*trial_scale).backward()
            invalid=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            comparisons[name]={'loss':float(trial['total'].detach()),'nonfinite_gradient_parameters':invalid,
                               'all_gradients_finite':not invalid,'scale':trial_scale}
        report={'status':'ACTUAL_FAILED_MINIBATCH_REPRODUCED_AND_CONTROLLED_PRECISION_CHECK',
                'seed':seed,'outer':outer,'inner':inner,'fit_sequences':fits,'failed_batch_index':index,
                'initial_loss':float(parts['total'].detach()),'initial_grad_scaler_scale':scale,
                'initial_nonfinite_gradient_parameters':failed,'same_weights_same_minibatch_comparisons':comparisons,
                'batch_tensor_SHA256':{k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in source.items()},
                'source_code_sha256':{p:sha256(ROOT/p) for p in ['scripts/n72r21_train_t0.py','sam3_intermot/one_click/acib.py','scripts/n72r21_diagnose_t0_amp.py']},
                'failed_optimizer_step_applied':False,'outer_labels_or_pixels_read':False,
                'checkpoint_or_threshold_selection':False,'diagnostic_model_weights_saved':False}
        write_json('training/INITIAL_AMP_NONFINITE_DIAGNOSTIC.json',report)
        print(json.dumps({k:report[k] for k in ['failed_batch_index','initial_grad_scaler_scale','same_weights_same_minibatch_comparisons']}),flush=True)
        return
    write_json('training/INITIAL_AMP_NONFINITE_DIAGNOSTIC.json',{'status':'NOT_REPRODUCED_ON_REPLAY_NO_INVENTED_CAUSE','seed':seed,'outer':outer})


if __name__=='__main__':run()
