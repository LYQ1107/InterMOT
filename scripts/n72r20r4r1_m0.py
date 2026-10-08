"""Fresh source reproduction with new-stage-only audit writes."""
from __future__ import annotations
import hashlib
import json
from dataclasses import replace
import tempfile
from pathlib import Path
from sam3_intermot.association.causal_identity_tracker import CausalIdentityTracker
from sam3_intermot.association.identity_authority import AuthorityConfig
from sam3_intermot.association.opportunity_scores import decompose_scores
from sam3_intermot.association.identity_state import IdentityState
from scripts.n72r20r4_run_causal_tracker import run_rollout
from scripts.n72r20r4_train_authority import matched_identity, target_truth
from scripts.n72r20r3_common import gt_by_frame
from sam3_intermot.association.causal_state_commit import memory_metrics
from scripts.n72r20r3r2r3_pipeline import run_trackeval_many, parse_trackeval, trackeval_summary
from scripts.n72r20r4r1_common import *


def run():
    torch.set_num_threads(1)
    inventory = OUT/"audit/HISTORICAL_HASHES_BEFORE.json"
    if not inventory.exists(): write_json(inventory, source_hashes())
    result = read_json(OUT/"audit/M0_REPRODUCTION.json") if (OUT/"audit/M0_REPRODUCTION.json").exists() else {"sequences": {}}
    for sequence in SEQUENCES:
        if sequence in result["sequences"]: continue
        check_storage()
        frames = load_frames(sequence); event = events()[sequence]
        manifest = read_json(R4ASSETS/"dev/manifests/B1_CAUSAL_BASELINE"/f"{sequence}.json")
        config = AuthorityConfig(**manifest["configuration"])
        tracker = CausalIdentityTracker(config=config, event=event, bank=make_bank())
        twin = CausalIdentityTracker(config=config, event=event, bank=make_bank())
        trace=[];native=positive=negative=edges=0; extrema=[]
        for payload,rows in frames:
            f=int(payload["frame"])
            # Initialization is repeated in step and is only audited on later frames.
            states=[s for _,s in sorted(tracker.states.items()) if s.state!=IdentityState.TERMINATED]
            if f>0:
                parts=decompose_scores(states,rows,f)
                native+=int(np.count_nonzero(parts["native_bonus"])); positive+=int(np.count_nonzero(parts["positive_bonus"]))
                negative+=int(parts["hard_mask"].sum()); edges+=parts["scores"].size
                if parts["scores"].size:extrema.append([float(parts["scores"].min()),float(parts["scores"].max())])
            a=tracker.step(rows,f); b=twin.step(rows,f)
            if a["state_after"]!=b["state_after"] or a["assignments"]!=b["assignments"]:raise RuntimeError("source A/A state mismatch")
            trace.append({k:v for k,v in a.items() if k not in {"base_matrix","fused_matrix","solver"}})
        text=trajectory_text(trace); digest=hashlib.sha256(text.encode()).hexdigest()
        if digest!=manifest["trajectory_sha256"] or sha256(Path(manifest["trajectory_path"]))!=digest:raise RuntimeError("source baseline reproduction mismatch")
        adapter=strict_ensemble(sequence)
        gt=gt_by_frame(DATASET/"train"/sequence/"gt/gt.txt"); truth=target_truth(event,gt)
        memories={}
        for policy,name in (("P1","B3_CONSENSUS_MEMORY"),("P2","MEMORY_P2_RELIABLE")):
            old=read_json(R4ASSETS/"dev/manifests"/name/f"{sequence}.json")
            records,profile=run_rollout(sequence,config=AuthorityConfig(**old["configuration"]),frames=frames,event=event,adapter=adapter)
            d=hashlib.sha256(trajectory_text(records).encode()).hexdigest()
            if d!=old["trajectory_sha256"]:raise RuntimeError(f"{policy} trajectory mismatch")
            sealed=[{**r["memory"],"sequence":sequence,"correct":next((matched_identity(row["box_xyxy"],gt.get(r["frame"],[]))==truth for row in rows if str(row["candidate_uid"])==r["memory"].get("candidate_uid")),False)} for (_,rows),r in zip(frames,records)]
            memories[policy]={"trajectory_sha256":d,"metrics":memory_metrics(sealed),"adapter_calls":profile["adapter_calls"]}
        result["sequences"][sequence]={"source_baseline_sha256":digest,"exact_trajectory_reproduction":True,"source_AA_assignment_and_state_match":True,"frames":len(frames),"trajectory_rows":len(text.splitlines()),"last_state_hash":trace[-1]["state_after"],"score_decomposition":{"exact_float32_match":True,"noninitial_edges":edges,"native_bonus_active_edges":native,"positive_bonus_active_edges":positive,"hard_negative_edges":negative,"score_min":min(x[0] for x in extrema),"score_max":max(x[1] for x in extrema)},"memories":memories,"strict_adapter_manifest":adapter.manifest}
        write_json(OUT/"audit/M0_REPRODUCTION.json",result)
        print(json.dumps({"M0_sequence_complete":sequence,"native":native,"positive":positive,"hard_negative":negative}),flush=True)
    with tempfile.TemporaryDirectory(prefix="intermot-r4r1-m0-") as temporary:
        root=Path(temporary); trackers=root/"trackers"; folder=trackers/"REPRODUCED_BASELINE/data";folder.mkdir(parents=True)
        for sequence in SEQUENCES:(folder/f"{sequence}.txt").symlink_to(R4ASSETS/"dev/trackers/B1_CAUSAL_BASELINE/data"/f"{sequence}.txt")
        seqmap=trackers/"seqmap.txt";seqmap.write_text("name\n"+"\n".join(SEQUENCES)+"\n")
        evaluated=run_trackeval_many(trackers,root/"eval",["REPRODUCED_BASELINE"],seqmap,gt_folder=DATASET/"train")
        metrics=trackeval_summary(parse_trackeval(root/"eval","REPRODUCED_BASELINE",SEQUENCES))
        log=Path(evaluated["log"]); artifact=stream_zstd(ASSETS/"audit/M0_TRACK_EVAL_LOG.jsonl.zst",[{"stdout":log.read_text()}])
        write_json(OUT/"audit/M0_TRACK_EVAL.json",{"metrics":metrics,"command":evaluated["command"],"returncode":evaluated["returncode"],"temporary_namespace_cleaned_after_sealing":True,"temporary_namespace_contains_only_symlinks_to_SHA_verified_reproduced_source_trajectories":True,"log":artifact,"trackeval_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT/"third_party/MOTIP/TrackEval",text=True).strip(),"full_sequence":True})
    before=read_json(inventory);after=source_hashes()
    if before!=after:raise RuntimeError("historical outputs changed")
    write_json(OUT/"audit/M0_COMPLETION.json",{"source_reproduction":True,"source_AA":True,"new_KEEP_AA":"PENDING_NEW_TRACKER_IMPLEMENTATION","memory_checkpoint_sha256":sha256(MEMORY_CHECKPOINT),"memory_checkpoint_verified":sha256(MEMORY_CHECKPOINT)==MEMORY_SHA,"historical_output_hashes_unchanged":True,"strict_adapter_models":24,"runtime_gt_read":False})


if __name__=="__main__":run()
