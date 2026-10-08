"""Fit-only strict-fold action features, natural-label gates and small models."""
from __future__ import annotations
import argparse
from collections import defaultdict
import random
from copy import deepcopy
import numpy as np
import torch
from scripts.n72r20r4r1_common import *
from scripts.n72r20r4r1_supervision import supervision_gate
from sam3_intermot.association.opportunity_tracker import FEATURE_NAMES
from sam3_intermot.association.opportunity_models import ActionValueModel,action_loss


def decode_sequence(sequence):
    summary=read_json(OUT/"counterfactual/per_sequence"/f"{sequence}.json")
    path=Path(summary["artifact"]["path"])
    if sha256(path)!=summary["artifact"]["sha256"]:raise RuntimeError("training corpus SHA mismatch")
    records=read_zstd_jsonl(path);events_by_key={}
    actions=defaultdict(list);seen=set()
    for r in records:
        key=(r["identity_key"][1],r["frame"])
        if r["record_type"]=="event":events_by_key[key]=r
        elif r["record_type"]=="action" and r.get("feasible") and not r.get("is_reference"):
            unique=(*key,r["action"]["family"],r["action"]["candidate_uid"])
            if unique in seen:raise RuntimeError("duplicated natural training action")
            seen.add(unique);actions[key].append(r)
    return events_by_key,actions,summary


def causal_features(sequence,adapter,*,k=5,heldout=None,role="fit"):
    if role=="fit":
        if heldout is None:raise ValueError("fit reader requires explicit outer fold")
        assert_fit_sequence(sequence,heldout)
    elif role=="inner":
        if heldout is None or sequence!=fold_split(heldout)[1]:raise ValueError("inner axis mismatch")
    else:raise ValueError("outer labels cannot enter feature or policy selection")
    er,actions,summary=decode_sequence(sequence);frames=load_frames(sequence)
    x=np.stack([r["feature"] for _,rows in frames for r in rows]);offsets=np.cumsum([0]+[len(rows) for _,rows in frames]);encoded=adapter.encode_candidates(x)
    examples=[];opportunities=oracle_positive=runtime_positive=0;anchor_cache={}
    for key,event in sorted(er.items()):
        identity,f=key;rows=frames[f][1];uids=[str(r["candidate_uid"]) for r in rows]
        if identity not in anchor_cache:
            af=event["anchor_frame"];auid=event["anchor_candidate_uid"]
            anchor=np.asarray(next(r["feature"] for r in frames[af][1] if str(r["candidate_uid"])==auid),dtype=np.float32)
            anchor_cache[identity]=anchor/np.linalg.norm(anchor)
        anchor=anchor_cache[identity]
        identity_scores=adapter.scores(anchor,np.asarray([r["feature"] for r in rows]),[v[offsets[f]:offsets[f+1]] for v in encoded]) if rows else np.zeros(0)
        raw=np.asarray([float(np.dot(anchor,r["feature"])) for r in rows])
        matrix=np.asarray(event["base_scores"]).reshape(len(rows),len(event["public_axis"]))
        j=event["public_axis"].index(event["target_public"])
        def top(values):return sorted(range(len(rows)),key=lambda i:(-float(values[i]),uids[i]))[:k]
        proposed={uids[i] for i in set(top(identity_scores)+top(raw)+top(matrix[:,j]))}
        oracle=[r for r in actions[key] if r["H5"]["beneficial"]]
        oracle_positive+=len(oracle);opportunities+=int(bool(oracle))
        for row in actions[key]:
            uid=row["action"]["candidate_uid"]
            if uid is not None and uid not in proposed:continue
            feature=np.asarray(row["features"],dtype=np.float32).copy()
            if uid is not None:
                i=uids.index(uid);feature[0]=identity_scores[i]
                feature[1]=identity_scores[i]-max([0.]+[float(v) for z,v in enumerate(identity_scores) if z!=i])
                feature[18]=sum(identity_scores>identity_scores[i])/max(1,len(rows))
            runtime_positive+=int(row["H5"]["beneficial"])
            examples.append({**row,"features":feature.tolist(),"group_key":[sequence,identity,f]})
    return examples,{"sequence":sequence,"role":role,"K":k,"actions":len(examples),"naturally_beneficial":sum(r["H5"]["beneficial"] for r in examples),
        "oracle_beneficial_actions":oracle_positive,"runtime_beneficial_actions":runtime_positive,"runtime_beneficial_proposal_recall":runtime_positive/oracle_positive if oracle_positive else None,
        "oracle_opportunity_events":opportunities,"source_sha256":summary["artifact"]["sha256"],"adapter_manifest":adapter.manifest,"GT_used_only_for_labels_after_proposal":True}


def tensors(examples):
    group_keys={};groups=[];labels=[];components=[]
    for r in examples:
        key=tuple(r["group_key"]);groups.append(group_keys.setdefault(key,len(group_keys)))
        h=r["H5"];labels.append([float(h["beneficial"]),float(h["harmful"]),h["value"]])
        components.append([h["current_target_gain"]+h["future_target_gain"],h["current_other_damage"]+h["future_other_damage"],float(h["potential_wrong_write"]),float(h["wrong_override"])])
    return (torch.tensor(np.asarray([r["features"] for r in examples]),dtype=torch.float32),torch.tensor(labels,dtype=torch.float32),torch.tensor(components,dtype=torch.float32),torch.tensor(groups,dtype=torch.int64))


def fit_model(family,loss,examples,seed,epochs):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);torch.set_num_threads(1)
    x,y,c,g=tensors(examples);model=ActionValueModel(family,feature_dim=x.shape[1])
    model.mean.copy_(x.mean(0));model.scale.copy_(x.std(0,unbiased=False).clamp_min(.05))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
    weights=[((len(y)-y[:,i].sum())/y[:,i].sum().clamp_min(1)).clamp(.1,100.) for i in range(2)]
    group_indices=defaultdict(list)
    for index,group in enumerate(g.tolist()):group_indices[group].append(index)
    unique=sorted(group_indices)
    losses=[]
    for epoch in range(epochs):
        # Keep contiguous action groups together for pairwise supervision.
        order=torch.randperm(len(unique)).tolist();batches=[];batch=[]
        for position in order:
            indices=group_indices[unique[position]]
            if batch and len(batch)+len(indices)>512:batches.append(batch);batch=[]
            batch.extend(indices)
        if batch:batches.append(batch)
        total=0.
        for batch in batches:
            indices=torch.tensor(batch);optimizer.zero_grad(set_to_none=True)
            value=action_loss(model,x[indices],y[indices],c[indices],g[indices],loss,weights)
            if not torch.isfinite(value):raise RuntimeError("nonfinite fit loss")
            value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();total+=float(value.detach())
        losses.append(total/max(1,len(batches)))
    model.eval()
    with torch.inference_mode():
        output=model(x);pred=output["value"].numpy();actual=y[:,2].numpy()
        approved=(torch.sigmoid(output["benefit_logit"])>=.7)&(torch.sigmoid(output["harm_logit"])<=.3)&(output["value"]>0)
        diagnostics={"value_MAE":float(np.mean(abs(pred-np.clip(actual,-3,3)))),"predicted_approved":int(approved.sum()),"approved_actual_value_mean":float(y[approved,2].mean()) if approved.any() else None,
            "approved_harmful":int(y[approved,1].sum()),"fit_only_diagnostic_not_generalization":True,"epoch_losses":losses}
    return model,diagnostics


def save_model(model,heldout,family,loss,seed,epochs,diagnostics,provenance):
    path=ASSETS/"models"/f"{heldout}__{family}__{loss}__seed{seed}__epoch{epochs}.pt"
    if path.exists():raise FileExistsError("refusing to overwrite completed learned checkpoint")
    check_storage(reserve_mib=.03);path.parent.mkdir(parents=True,exist_ok=True)
    fit,inner=fold_split(heldout)
    checkpoint={"stage":STAGE,"family":family,"loss_family":loss,"feature_names":FEATURE_NAMES,"feature_dim":len(FEATURE_NAMES),"parameters":model.parameter_count,"state_dict":model.state_dict(),
        "actual_training_sequences":fit,"forbidden_sequences":[inner,heldout],"seed":seed,"epochs":epochs,"training_provenance":provenance,"natural_positive_gate_passed":True,"frozen_GRU_sha256":MEMORY_SHA}
    torch.save(checkpoint,path)
    record={"path":path,"sha256":sha256(path),"family":family,"loss":loss,"seed":seed,"epochs":epochs,"parameters":model.parameter_count,"fit_sequences":fit,"inner":inner,"outer":heldout,"diagnostics":diagnostics,"training_provenance":provenance}
    write_json(path.with_suffix(".json"),record);return plain(record)


def run_fold(heldout):
    torch.set_num_threads(1);fit,inner=fold_split(heldout);adapter=strict_ensemble(heldout)
    examples=[];provenance=[]
    for sequence in fit:
        rows,record=causal_features(sequence,adapter,heldout=heldout);examples.extend(rows);provenance.append(record)
    gate=supervision_gate(examples)
    write_json(OUT/"authority/label_gates"/f"{heldout}.json",{"gate":gate,"fit":fit,"inner":inner,"outer":heldout,"sources":provenance})
    if not gate["pass"]:return {"status":"NOT_RUN","reason":gate["decision"],"gate":gate}
    provenance_path=OUT/"authority/fit_corpus"/f"{heldout}.json"
    write_json(provenance_path,{"sources":provenance,"fit_sequences":fit,"inner":inner,"outer":heldout,"gate":gate})
    compact_provenance={"path":str(provenance_path),"sha256":sha256(provenance_path),
        "source_sha256":{r["sequence"]:r["source_sha256"] for r in provenance},
        "strict_adapter_sha256":[r["sha256"] for r in adapter.manifest],"gate":gate}
    results={};specs=[("C2","L0"),("C3","L0"),("C4","L0"),("C5","L1")]+[("C6",f"L{i}") for i in range(4)]
    for family,loss in specs:
        candidates=[]
        for seed in SEEDS:
            for epochs in (10,30):
                path=ASSETS/"models"/f"{heldout}__{family}__{loss}__seed{seed}__epoch{epochs}.json"
                if path.exists():
                    record=read_json(path)
                    if sha256(Path(record["path"]))!=record["sha256"]:raise RuntimeError("existing learned checkpoint SHA mismatch")
                else:
                    model,diagnostics=fit_model(family,loss,examples,seed,epochs)
                    record=save_model(model,heldout,family,loss,seed,epochs,diagnostics,compact_provenance)
                candidates.append(record)
                print(json.dumps({"trained_fold":heldout,"family":family,"loss":loss,"seed":seed,"epochs":epochs,"parameters":record["parameters"]}),flush=True)
        # Epoch is chosen only with fit-side diagnostic error. Real policy
        # choice still requires inner full-sequence TrackEval before outer.
        best_epoch=min((10,30),key=lambda e:np.mean([r["diagnostics"]["value_MAE"] for r in candidates if r["epochs"]==e]))
        results[family+"_"+loss]={"candidates":candidates,"fit_selected_epoch":best_epoch,"selected":[r for r in candidates if r["epochs"]==best_epoch]}
        write_json(OUT/"authority/training"/f"{heldout}.json",{"status":"COMPLETE_FIT_ONLY" if len(results)==len(specs) else "PARTIAL_RESUMABLE_FIT_ONLY","gate":gate,"models":results,"fit":fit,"inner":inner,"outer":heldout,"outer_policy_frozen":False,"outer_treatment_evaluated":False})
    return results


class ValueEnsemble:
    def __init__(self,records,heldout):
        self.models=[];self.manifest=records
        fit,inner=fold_split(heldout)
        for r in records:
            if sha256(Path(r["path"]))!=r["sha256"]:raise ValueError("learned authority SHA mismatch")
            c=torch.load(r["path"],map_location="cpu",weights_only=False)
            if c["actual_training_sequences"]!=fit or set(c["actual_training_sequences"])&{heldout,inner}:raise ValueError("controller fold leakage")
            if tuple(c["feature_names"])!=FEATURE_NAMES:raise ValueError("controller feature allowlist changed")
            model=ActionValueModel(c["family"]);model.load_state_dict(c["state_dict"],strict=True);model.eval()
            for p in model.parameters():p.requires_grad_(False)
            self.models.append(model)
    def predict(self,features):
        predictions=[m.predict(features) for m in self.models]
        return {k:float(np.mean([r[k] for r in predictions])) for k in predictions[0]}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--heldouts",nargs="+",default=list(SEQUENCES));args=parser.parse_args()
    if not set(args.heldouts)<=set(SEQUENCES):raise ValueError("invalid outer fold")
    for s in args.heldouts:run_fold(s)
