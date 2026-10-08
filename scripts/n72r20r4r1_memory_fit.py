"""Fit-only safe-write and native reliability labels from actual causal states."""
from __future__ import annotations
import argparse
import hashlib
from scripts.n72r20r4r1_common import *
from scripts.n72r20r4r1_fit import fit_model
from scripts.n72r20r4r1_supervision import prepare_anchors
from scripts.n72r20r3_common import gt_by_frame
from sam3_intermot.association.identity_authority import AuthorityConfig
from sam3_intermot.association.opportunity_tracker import OpportunityTracker,InterventionPolicy
from sam3_intermot.association.opportunity_memory import MemoryPolicy
from sam3_intermot.association.opportunity_models import ActionValueModel

RELIABILITY_FEATURES=("identity_score","identity_margin","anchor_score","anchor_state_agreement","motion","quality","native_streak","trusted_gap","native_same","candidate_count")


def example(sequence,identity,frame,uid,features,correct):
    # Correctness is attached only after the corresponding runtime decision.
    return {"sequence":sequence,"identity_key":[sequence,identity],"frame":frame,"candidate_uid":uid,"features":features,"group_key":[sequence,identity,frame],
        "H5":{"beneficial":bool(correct),"harmful":not correct,"value":1. if correct else -1.,"current_target_gain":int(correct),"future_target_gain":0.,"current_other_damage":0.,"future_other_damage":0.,"potential_wrong_write":not correct,"wrong_override":False}}


def run_fold(heldout):
    done=OUT/"memory/fit"/f"{heldout}.json"
    if done.exists():
        result=read_json(done)
        for records in result["models"].values():
            for r in records:
                if sha256(Path(r["path"]))!=r["sha256"]:raise RuntimeError("reliability SHA mismatch")
        return result
    torch.set_num_threads(1);fit,inner=fold_split(heldout);adapter=strict_ensemble(heldout)
    writes=[];native_examples=[];sources={};trace_digest=hashlib.sha256()
    for sequence in fit:
        assert_fit_sequence(sequence,heldout);frames=load_frames(sequence)
        gt=gt_by_frame(DATASET/"train"/sequence/"gt/gt.txt");anchors,matches=prepare_anchors(sequence,frames,gt)
        flat=np.stack([r["feature"] for _,rows in frames for r in rows]);enc=adapter.encode_candidates(flat);offset=np.cumsum([0]+[len(rows) for _,rows in frames])
        summary={"eligible_identities":len(anchors),"write_observations":0,"native_active_observations":0,"correct_write_observations":0}
        for identity,prepared in sorted(anchors.items()):
            event=prepared["event"]
            tracker=OpportunityTracker(config=AuthorityConfig(memory="P1"),event=event,adapter=adapter,bank=make_bank(),memory_policy=MemoryPolicy("P1"),audit_hashes=False)
            for payload,rows in frames:
                f=int(payload["frame"]);features=[]
                if f>event["event_frame"] and tracker.target_public is not None and rows:
                    state=tracker.states[tracker.target_public];query=tracker.bank.records[tracker.target_public].current_state
                    raw=np.asarray([float(np.dot(query,r["feature"])) for r in rows])
                    for i,r in enumerate(rows):
                        if state.last_native_tid==int(r["native_tid"]) and state.last_native_scope==r.get("native_scope"):
                            features.append((str(r["candidate_uid"]),tracker.native_features(state,r,rows,i,raw,query,f)))
                d=tracker.step(rows,f,encoded_candidates=[v[offset[f]:offset[f+1]] for v in enc])
                trace_digest.update(json.dumps({"sequence":sequence,"identity":identity,"frame":f,"assignments":d["assignments"],"memory":d["memory"]},sort_keys=True,separators=(",",":"),allow_nan=False).encode())
                # Offline supervision cannot flow back into tracker state.
                for uid,feature in features:
                    native_examples.append(example(sequence,identity,f,uid,feature,matches[f].get(uid)==identity));summary["native_active_observations"]+=1
                if d["memory"].get("eligible"):
                    correct=matches[f].get(d["memory"]["candidate_uid"])==identity
                    writes.append(example(sequence,identity,f,d["memory"]["candidate_uid"],d["write_features"],correct))
                    summary["write_observations"]+=1;summary["correct_write_observations"]+=int(correct)
            print(json.dumps({"memory_fit":heldout,"sequence":sequence,"identity":identity}),flush=True)
        sources[sequence]=summary
    models={}
    for purpose,rows in (("WRITE_SAFETY",writes),("NATIVE_RELIABILITY",native_examples)):
        positive=sum(r["H5"]["beneficial"] for r in rows);negative=len(rows)-positive
        if not positive or not negative:
            models[purpose]=[];continue
        models[purpose]=[]
        for seed in SEEDS:
            path=ASSETS/"models"/f"{heldout}__{purpose}__seed{seed}.pt"
            if path.exists():
                record=read_json(path.with_suffix(".json"))
                if sha256(path)!=record["sha256"]:raise RuntimeError("resumed reliability SHA mismatch")
                old=torch.load(path,map_location="cpu",weights_only=False)
                if old["trace_digest"]!=trace_digest.hexdigest() or old["actual_training_sequences"]!=fit or old["purpose"]!=purpose:raise RuntimeError("resumed reliability training lineage differs")
                models[purpose].append(record);continue
            model,diagnostics=fit_model("C3","L0",rows,seed,30)
            check_storage(reserve_mib=.02)
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():raise FileExistsError("completed reliability model already exists")
            torch.save({"stage":STAGE,"purpose":purpose,"family":"C3","feature_names":RELIABILITY_FEATURES,"parameters":model.parameter_count,"state_dict":model.state_dict(),"actual_training_sequences":fit,"inner":inner,"outer":heldout,"seed":seed,"trace_digest":trace_digest.hexdigest(),"strict_adapter_manifest":adapter.manifest},path)
            record={"path":str(path),"sha256":sha256(path),"seed":seed,"purpose":purpose,"parameters":model.parameter_count,"positive":positive,"negative":negative,"diagnostics":diagnostics}
            write_json(path.with_suffix(".json"),record);models[purpose].append(record)
    result={"status":"COMPLETE_FIT_ONLY","fit":fit,"inner":inner,"outer":heldout,"models":models,"sources":sources,
        "trace_sha256":trace_digest.hexdigest(),"native_supervision":"actual activated target native edges before commit","write_supervision":"actual P1 assigned observations and evolving frozen GRU state","runtime_GT_used":False,"positive_history_model":"NOT_RUN_NO_NATURAL_ACTIVATION"}
    write_json(done,result);return result


class ReliabilityEnsemble:
    def __init__(self,records,heldout,purpose):
        if not records:raise ValueError("no identifiable reliability model")
        self.models=[];self.manifest=records;fit,inner=fold_split(heldout)
        for r in records:
            if sha256(Path(r["path"]))!=r["sha256"]:raise ValueError("reliability checkpoint SHA mismatch")
            c=torch.load(r["path"],map_location="cpu",weights_only=False)
            if c["purpose"]!=purpose or c["actual_training_sequences"]!=fit or tuple(c["feature_names"])!=RELIABILITY_FEATURES:raise ValueError("reliability lineage/schema mismatch")
            model=ActionValueModel("C3",10);model.load_state_dict(c["state_dict"],strict=True);model.eval()
            for p in model.parameters():p.requires_grad_(False)
            self.models.append(model)
    def predict(self,features):
        predictions=[m.predict(features) for m in self.models]
        return {k:float(np.mean([r[k] for r in predictions])) for k in predictions[0]}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--heldouts",nargs="+",default=list(SEQUENCES));args=parser.parse_args()
    if not set(args.heldouts)<=set(SEQUENCES):raise ValueError("unknown fold")
    for s in args.heldouts:run_fold(s)
