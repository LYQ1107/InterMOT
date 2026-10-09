"""A/A check against the actual SHA-verified upstream track/map equations.

Only two pure inference methods are compiled from official trusted source;
debug/visdom/environment setup is irrelevant and not imported. Device-neutral
normalization is checked separately against the exact upstream arithmetic.
"""
import ast
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from scripts.n72r21_sot import FrozenSOT
from scripts.n72r21_common import ROOT,ASSETS,OUT,read_json,write_json,sha256


def run():
    torch.set_num_threads(1)
    source=ASSETS/'official_sources/OSTrack/source/lib/test/tracker/ostrack.py'
    tree=ast.parse(source.read_text());klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OSTrack')
    methods=[n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name in ('track','map_box_back')]
    isolated=ast.ClassDef(name='OriginalTrackEquations',bases=[],keywords=[],body=methods,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[isolated],type_ignores=[]))
    wrapper=FrozenSOT(torch.device('cpu'))
    environment={'torch':torch,'sample_target':wrapper.sample_target,'clip_box':wrapper.clip_box}
    exec(compile(module,str(source),'exec'),environment)
    original=environment['OriginalTrackEquations']()
    event=read_json(OUT/'development/RUNTIME_INPUTS.json')['inputs'][0]
    root=ROOT.parent/'InterMOT_N72R16_assets/dataset/train'/event['sequence']/'img1'
    initial=cv2.cvtColor(cv2.imread(str(root/f"{event['frame']+1:08d}.jpg")),cv2.COLOR_BGR2RGB)
    wrapper.initialize(initial,event['box_xyxy'])
    def process(image,mask):
        actual=torch.tensor(image).float().permute(2,0,1).unsqueeze(0)
        mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
        expected=((actual/255)-mean)/std
        torch.testing.assert_close(wrapper.preprocess(image),expected,rtol=0,atol=0)
        return SimpleNamespace(tensors=expected)
    original.state=list(wrapper.state);original.frame_id=0;original.network=wrapper.model;original.output_window=wrapper.window
    original.z_dict1=SimpleNamespace(tensors=wrapper.template);original.box_mask_z=wrapper.mask
    original.preprocessor=SimpleNamespace(process=process);original.params=SimpleNamespace(search_factor=4.,search_size=256)
    original.debug=False;original.save_all_boxes=False
    records=[]
    for f in range(event['frame']+1,event['frame']+6):
        image=cv2.cvtColor(cv2.imread(str(root/f'{f+1:08d}.jpg')),cv2.COLOR_BGR2RGB)
        expected=original.track(image)['target_bbox'];box,_=wrapper.step(image)
        actual=[box[0],box[1],box[2]-box[0],box[3]-box[1]]
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-5)
        records.append({'frame':f,'maximum_box_coordinate_difference':float(np.max(np.abs(np.asarray(actual)-expected)))})
    write_json('baselines/SOT_UPSTREAM_EQUIVALENCE.json',{'status':'PASS_AA_ACTUAL_UPSTREAM_TRACK_METHOD',
        'official_original_method_source_sha256':sha256(source),'records':records,'real_TRAIN_images':True,
        'device':'CPU','preprocessing_exact_equality':True,'maximum_box_tolerance':1e-5,
        'no_source_or_weight_patch':True,'not_a_generalization_metric':True})
    print({'status':'PASS_AA_ACTUAL_UPSTREAM_TRACK_METHOD','real_frames':len(records)})


if __name__=='__main__':run()
