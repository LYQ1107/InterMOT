"""Inner-only shortlist, full-sequence selection and frozen outer policies."""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
from itertools import product
from collections import defaultdict
from scripts.n72r20r4r1_common import *
from scripts.n72r20r4r1_fit import causal_features,ValueEnsemble
from scripts.n72r20r4r1_memory_fit import ReliabilityEnsemble
from scripts.n72r20r4r1_evaluate import run_runtime,posthoc,baseline_trace,project,EvaluationBatch
from sam3_intermot.association.opportunity_tracker import InterventionPolicy
from sam3_intermot.association.opportunity_memory import MemoryPolicy


def spec(name,policy=None,memory=None,controller=None,native=False,safety=False):
    return {"name":name,"policy":asdict(policy or InterventionPolicy()),"memory":asdict(memory or MemoryPolicy()),"controller_key":controller,"native_predictor":native,"memory_predictor":safety}


def threshold_shortlist(examples,controller):
    groups=defaultdict(list)
    for r in examples:groups[tuple(r["group_key"])].append((controller.predict(r["features"]),r["H5"]["value"]))
    rankings=[]
    for p,h,v in product((.5,.7,.9),(.1,.3),(0.,.05)):
        values=[]
        for entries in groups.values():
            approved=[(pred,truth) for pred,truth in entries if pred["beneficial"]>=p and pred["harmful"]<=h and pred["value"]>v]
            values.append(max(approved,key=lambda item:item[0]["value"])[1] if approved else 0.)
        rankings.append({"thresholds":[p,h,v],"inner_counterfactual_mean_value":float(np.mean(values)) if values else 0.})
    rankings.sort(key=lambda r:(-r["inner_counterfactual_mean_value"],-r["thresholds"][0],r["thresholds"][1],-r["thresholds"][2]))
    selected=[r["thresholds"] for r in rankings[:2]]
    if [.9,.1,.05] not in selected:selected.append([.9,.1,.05])
    return selected,rankings


def controller_objects(heldout,training):
    return {key:ValueEnsemble(r["selected"],heldout) for key,r in training["models"].items()}


def reliability_objects(heldout):
    trained=read_json(OUT/"memory/fit"/f"{heldout}.json")
    write=ReliabilityEnsemble(trained["models"]["WRITE_SAFETY"],heldout,"WRITE_SAFETY") if trained["models"]["WRITE_SAFETY"] else None
    native=ReliabilityEnsemble(trained["models"]["NATIVE_RELIABILITY"],heldout,"NATIVE_RELIABILITY") if trained["models"]["NATIVE_RELIABILITY"] else None
    return write,native


def metric_key(name,metrics,baseline,stats):
    m=metrics[name];guard=m["DetA"]>=baseline["DetA"]-.005
    memory=stats[name]["memory"]
    return (guard,m["HOTA"],m["AssA"],-m["IDSW"],memory["safety_pass"],-(memory["wrong_write_rate"] or 0.),
        memory["correct_write_retention"] or 0.,-stats[name]["funnel"].get("authority_approved_frames",0))


def run_fold(heldout):
    frozen_path=OUT/"authority/frozen_outer"/f"{heldout}.json"
    if frozen_path.exists():return read_json(frozen_path)
    torch.set_num_threads(1);fit,inner=fold_split(heldout);adapter=strict_ensemble(heldout)
    training=read_json(OUT/"authority/training"/f"{heldout}.json")
    if len(training["models"])!=8 or not training["gate"]["pass"]:raise ValueError("strict fit training incomplete or unidentifiable")
    controllers=controller_objects(heldout,training);write_predictor,native_predictor=reliability_objects(heldout)
    proposal_audits={};k_examples={}
    for k in (1,3,5):
        rows,record=causal_features(inner,adapter,heldout=heldout,role="inner",k=k);k_examples[k]=rows;proposal_audits[str(k)]=record
    k=max((1,3,5),key=lambda z:(proposal_audits[str(z)]["runtime_beneficial_proposal_recall"] or 0.,-z))
    default=InterventionPolicy(top_k=k);cases=[spec("BASELINE",default)]
    fixed=[];controller_cases=defaultdict(list);memory_cases=defaultdict(list);native_cases=[];shortlists={}
    for strength in (.5,1.,2.,4.,8.):
        name=f"C1_s{str(strength).replace('.','p')}";fixed.append(name);cases.append(spec(name,replace(default,family="C1",strength=strength)))
    for key,controller in controllers.items():
        shortlist,ranking=threshold_shortlist(k_examples[k],controller);shortlists[key]=ranking
        family=key.split("_")[0]
        for index,(p,h,v) in enumerate(shortlist):
            name=f"{key}_a{index}";controller_cases[key].append(name)
            cases.append(spec(name,replace(default,family=family,beneficial_min=p,harmful_max=h,value_min=v),controller=key))
    for p in ("P0","P1","P2"):
        name="MEMORY_"+p;memory_cases[p].append(name);cases.append(spec(name,default,MemoryPolicy(p)))
    for p in ("P3","P4"):
        for confirm,anchor in product((2,3),(.3,.5,.7)):
            name=f"MEMORY_{p}_K{confirm}_A{int(anchor*10)}";memory_cases[p].append(name)
            cases.append(spec(name,default,MemoryPolicy(p,confirmation_k=confirm,anchor_min=anchor)))
    if write_predictor is not None:
        for probability,anchor in product((.5,.7,.9,.95,.99),(.3,.5,.7)):
            name=f"MEMORY_P5_P{int(probability*100)}_A{int(anchor*10)}";memory_cases["P5"].append(name)
            cases.append(spec(name,default,MemoryPolicy("P5",anchor_min=anchor,probability_min=probability),safety=True))
    for anchor in (.3,.5,.7):
        name=f"MEMORY_P6_A{int(anchor*10)}";memory_cases["P6"].append(name);cases.append(spec(name,default,MemoryPolicy("P6",anchor_min=anchor)))
    for discount in (0.,.25,.5,1.):
        name=f"NATIVE_GLOBAL_{int(discount*100)}";native_cases.append(name)
        cases.append(spec(name,replace(default,native_discount=discount,discount_scope="global")))
    if native_predictor is not None:cases.append(spec("NATIVE_IDENTITY",default,native=True))
    frames=load_frames(inner);event=events()[inner];base,_=baseline_trace(inner);encoded=project(adapter,frames)
    calibration=read_json(R4ASSETS/"dev/manifests/B1_CAUSAL_BASELINE"/f"{heldout}.json")["configuration"]
    # Calibration is bound to the requesting outer fold's six fit sequences;
    # never import the inner sequence's historical own-fold calibration.
    write_json(OUT/"authority/inner_plan"/f"{heldout}.json",{"fit":fit,"inner":inner,"outer":heldout,"proposal_K":k,"proposal_audits":proposal_audits,"threshold_rankings":shortlists,"cases":cases,"calibration":calibration,"frozen_before_inner_runtime":True})
    stats={};batch=EvaluationBatch("inner/"+heldout,[inner])
    try:
        for case in cases:
            name=case["name"]
            trace,profile=run_runtime(inner,frames,event,adapter,InterventionPolicy(**case["policy"]),MemoryPolicy(**case["memory"]),
                controller=controllers.get(case["controller_key"]),memory_predictor=write_predictor if case["memory_predictor"] else None,
                native_predictor=native_predictor if case["native_predictor"] else None,encoded=encoded,calibration_config=calibration)
            audit=batch.add(name,inner,trace,profile);stats[name]=posthoc(inner,frames,event,trace,base)
            print(json.dumps({"inner_fold":heldout,"inner_sequence":inner,"case":name,"seconds":profile["seconds"],"writes":stats[name]["memory"]["accepted_writes"]}),flush=True)
            write_json(OUT/"authority/inner_progress"/f"{heldout}.json",{"completed":list(stats),"outer_not_accessed":True})
        evaluated=batch.evaluate()
    finally:batch.close()
    metrics=evaluated["metrics"];baseline=metrics["BASELINE"];by_name={c["name"]:c for c in cases}
    choose=lambda names:max(names,key=lambda n:metric_key(n,metrics,baseline,stats))
    selected={"C0":by_name["BASELINE"],"C1":by_name[choose(fixed)],"NATIVE_GLOBAL":by_name[choose(native_cases)]}
    for key,names in controller_cases.items():selected[key]=by_name[choose(names)]
    for p,names in memory_cases.items():selected[p]=by_name[choose(names)]
    if "NATIVE_IDENTITY" in by_name:selected["NATIVE_IDENTITY"]=by_name["NATIVE_IDENTITY"]
    safe_memories=[n for names in memory_cases.values() for n in names if stats[n]["memory"]["safety_pass"]]
    best_memory=by_name[choose(safe_memories)] if safe_memories else by_name["MEMORY_P0"]
    controller_names=[selected[key]["name"] for key in ("C0","C1",*controllers.keys())]
    best_controller=by_name[choose(controller_names)]
    result={"fit":fit,"inner":inner,"outer":heldout,"proposal_K":k,"selected":selected,"best_controller":best_controller,"best_safe_memory":best_memory,
        "active_memory_safety_passed":bool(safe_memories),"positive_history_reliability":"NOT_RUN_NO_NATURAL_ACTIVATION","controller_models":{key:r["selected"] for key,r in training["models"].items()},
        "reliability_models":read_json(OUT/"memory/fit"/f"{heldout}.json")["models"],"inner_metrics":metrics,"inner_statistics":stats,"calibration":calibration,
        "source_fit_training_manifest_sha256":sha256(OUT/"authority/training"/f"{heldout}.json"),"shortlist_protocol_sha256":sha256(OUT/"INNER_SHORTLIST_PROTOCOL.json"),
        "frozen_before_outer":True,"outer_labels_used_to_select":False,"VAL_used_to_select":False,"goal_file":"outputs/N72R20R4R1/FINAL_GOAL.json"}
    write_json(frozen_path,result);return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--heldouts",nargs="+",default=list(SEQUENCES));args=parser.parse_args()
    if not set(args.heldouts)<=set(SEQUENCES):raise ValueError("unknown fold")
    for s in args.heldouts:run_fold(s)
