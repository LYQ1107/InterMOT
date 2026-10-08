#!/usr/bin/env python3
"""Re-evaluate sealed trajectories in a new namespace, never tune a policy."""
from scripts.n72r20r4_common import *
from scripts.n72r20r4_trackeval import evaluate


def run() -> dict:
    original_dev=read_json(OUT/"dev_trackeval/ABLATIONS.json")
    original_val=read_json(OUT/"val/RESULT.json")
    results={}
    for group,split,names,sequences,original in (
        ("dev","train",list(original_dev),list(SEQUENCES),original_dev),
        ("val","val",["BASELINE_CAUSAL","TREATMENT_CAUSAL"],sorted(original_val["manifests"]),
         {"BASELINE_CAUSAL":original_val["baseline"],"TREATMENT_CAUSAL":original_val["treatment"]})):
        check_storage(reserve_gib=0.02)
        namespace=group+"_trackeval_reproduction"
        for name in names:
            directory=ASSETS/namespace/"trackers"/name/"data"
            directory.mkdir(parents=True,exist_ok=True)
            for s in sequences:
                source=ASSETS/group/"trackers"/name/"data"/f"{s}.txt"
                dest=directory/f"{s}.txt"
                if dest.exists():
                    if sha256(dest)!=sha256(source):raise RuntimeError("reproduction source changed")
                else:dest.hardlink_to(source)
        evaluation=evaluate(namespace,names,sequences,split=split)
        if evaluation["metrics"]!=original:
            write_json(OUT/"audit/TRACK_EVAL_REPRODUCTION_FAILURE.json",evaluation)
            raise RuntimeError("official TrackEval metrics are not exactly reproducible")
        results[group]={"all_combined_and_per_sequence_metrics_exact":True,
                        "variants":len(names),"sequences":len(sequences),
                        "frozen_trajectory_only":True,"policy_reselection":False,
                        "evaluation_provenance":evaluation["provenance"]}
    result={"status":"PASS_EXACT_TRACK_EVAL_REPRODUCTION","results":results,
            "original_evaluation_logs_preserved":True,"not_a_second_tuned_experiment":True}
    write_json(OUT/"audit/TRACK_EVAL_REPRODUCTION.json",result)
    return result


if __name__=="__main__":
    print(json.dumps(run(),sort_keys=True))
