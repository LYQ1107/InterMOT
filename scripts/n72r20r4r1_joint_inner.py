"""Preregistered component-combination checks before any outer rollout."""
from copy import deepcopy
import argparse
from scripts.n72r20r4r1_common import *
from scripts.n72r20r4r1_inner import controller_objects,reliability_objects,metric_key
from scripts.n72r20r4r1_evaluate import run_runtime,posthoc,baseline_trace,project,EvaluationBatch
from sam3_intermot.association.opportunity_tracker import InterventionPolicy
from sam3_intermot.association.opportunity_memory import MemoryPolicy


def run_fold(heldout):
    done=OUT/"authority/frozen_joint"/f"{heldout}.json"
    if done.exists():return read_json(done)
    source=OUT/"authority/frozen_outer"/f"{heldout}.json";frozen=read_json(source)
    inner=frozen["inner"];adapter=strict_ensemble(heldout);frames=load_frames(inner);event=events()[inner];encoded=project(adapter,frames);base,_=baseline_trace(inner)
    training=read_json(OUT/"authority/training"/f"{heldout}.json");controllers=controller_objects(heldout,training)
    memory_predictor,native_predictor=reliability_objects(heldout);cases=[]
    def combination(name,controller,native=False,memory=None):
        c=deepcopy(controller);c["name"]=name;c["native_predictor"]=native and native_predictor is not None
        if memory is not None:
            c["memory"]=deepcopy(memory["memory"]);c["memory_predictor"]=memory["memory_predictor"]
        return c
    best=frozen["best_controller"];c6=frozen["selected"]["C6_L0"];safe=frozen["best_safe_memory"]
    cases.append(combination("BEST_WITH_SAFE_MEMORY",best,memory=safe))
    cases.append(combination("C6_WITH_SAFE_MEMORY",c6,memory=safe))
    cases.append(combination("C6_WITH_P1",c6,memory=frozen["selected"]["P1"]))
    if native_predictor is not None:
        cases.append(combination("BEST_NATIVE_SAFE_MEMORY",best,native=True,memory=safe))
        cases.append(combination("C6_NATIVE",c6,native=True))
        cases.append(combination("C6_NATIVE_SAFE_MEMORY",c6,native=True,memory=safe))
        cases.append(combination("FIXED_NATIVE",frozen["selected"]["C1"],native=True))
    # Include a KEEP selection control in the same fresh evaluator invocation.
    control=deepcopy(frozen["selected"]["C0"]);control["name"]="JOINT_BASELINE";cases.insert(0,control)
    batch=EvaluationBatch("joint_inner/"+heldout,[inner]);stats={}
    try:
        for case in cases:
            trace,profile=run_runtime(inner,frames,event,adapter,InterventionPolicy(**case["policy"]),MemoryPolicy(**case["memory"]),
                controller=controllers.get(case["controller_key"]),memory_predictor=memory_predictor if case["memory_predictor"] else None,
                native_predictor=native_predictor if case["native_predictor"] else None,encoded=encoded,calibration_config=frozen["calibration"])
            batch.add(case["name"],inner,trace,profile);stats[case["name"]]=posthoc(inner,frames,event,trace,base)
            print(json.dumps({"joint_inner_fold":heldout,"case":case["name"],"seconds":profile["seconds"]}),flush=True)
        evaluated=batch.evaluate()
    finally:batch.close()
    metrics=evaluated["metrics"];best_case=max(cases,key=lambda c:metric_key(c["name"],metrics,metrics["JOINT_BASELINE"],stats))
    # The ultimate selected policy also competes with the family-level winner.
    old_name=frozen["best_controller"]["name"]
    old_key=metric_key(old_name,frozen["inner_metrics"],frozen["inner_metrics"]["BASELINE"],frozen["inner_statistics"])
    new_key=metric_key(best_case["name"],metrics,metrics["JOINT_BASELINE"],stats)
    final=best_case if new_key>old_key else frozen["best_controller"]
    result={"fit":frozen["fit"],"inner":inner,"outer":heldout,"cases":cases,"metrics":metrics,"statistics":stats,
        "final_selected":final,"source_frozen_family_file":str(source),"source_frozen_family_sha256":sha256(source),
        "frozen_before_outer":True,"outer_labels_or_metrics_used":False,"joint_selection_uses_actual_inner_full_sequence_TrackEval":True}
    write_json(done,result);return result


if __name__=="__main__":
    torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument("--heldouts",nargs="+",default=list(SEQUENCES));args=p.parse_args()
    if not set(args.heldouts)<=set(SEQUENCES):raise ValueError("unknown fold")
    for s in args.heldouts:run_fold(s)
