"""Bounded full-trajectory evaluation with reconstructible ownership deltas."""
from __future__ import annotations
from dataclasses import asdict
from collections import Counter
import tempfile
import time
from scripts.n72r20r4r1_common import *
from scripts.n72r20r4_train_authority import target_truth
from scripts.n72r20r4r1_supervision import match_frame
from scripts.n72r20r3_common import gt_by_frame
from scripts.n72r20r3r2r3_pipeline import run_trackeval_many,parse_trackeval,trackeval_summary
from sam3_intermot.association.opportunity_tracker import OpportunityTracker,InterventionPolicy
from sam3_intermot.association.opportunity_memory import MemoryPolicy
from sam3_intermot.association.identity_authority import AuthorityConfig
from sam3_intermot.association.causal_state_commit import memory_metrics


def baseline_trace(sequence):
    manifest=read_json(R4ASSETS/"dev/manifests/B1_CAUSAL_BASELINE"/f"{sequence}.json")
    path=Path(manifest["trace_path"])
    if sha256(path)!=manifest["trace_sha256"]:raise RuntimeError("source baseline trace SHA mismatch")
    return read_zstd_jsonl(path),manifest


def project(adapter,frames):
    if adapter is None:return None
    features=np.stack([r["feature"] for _,rows in frames for r in rows]);offset=np.cumsum([0]+[len(rows) for _,rows in frames])
    enc=adapter.encode_candidates(features)
    return [[x[offset[f]:offset[f+1]] for x in enc] for f in range(len(frames))]


def run_runtime(sequence,frames,event,adapter,policy,memory,*,controller=None,memory_predictor=None,native_predictor=None,encoded=None,calibration_config=None):
    cfg=AuthorityConfig(**(calibration_config or read_json(R4ASSETS/"dev/manifests/B1_CAUSAL_BASELINE"/f"{sequence}.json")["configuration"]))
    cfg=AuthorityConfig(**{**cfg.to_dict(),"memory":memory.family if memory.family in ("P0","P1","P2") else "P0"})
    tracker=OpportunityTracker(config=cfg,event=event,adapter=adapter,bank=make_bank(),controller=controller,
        intervention_policy=policy,memory_policy=memory,memory_predictor=memory_predictor,native_predictor=native_predictor)
    trace=[];started=time.perf_counter();proposal_digest=hashlib.sha256();state_digest=hashlib.sha256()
    for payload,rows in frames:
        f=int(payload["frame"]);d=tracker.step(rows,f,encoded_candidates=None if encoded is None else encoded[f],collect_proposals=True)
        proposals=[{k:v for k,v in p.items() if k!="forced"} for p in d["proposals"]]
        proposal_digest.update(json.dumps(proposals,sort_keys=True,separators=(",",":"),allow_nan=False).encode())
        state_digest.update(d["state_after"].encode())
        trace.append({k:v for k,v in d.items() if k not in {"base_matrix","solver","proposals","states_before_commit_axis"}})
        trace[-1]["proposals"]=proposals
        trace[-1]["beneficial_min"]=policy.beneficial_min
        trace[-1]["harmful_max"]=policy.harmful_max
        trace[-1]["value_min"]=policy.value_min
    return trace,{"seconds":time.perf_counter()-started,"frames":len(frames),"adapter_calls":tracker.adapter_calls,
        "proposal_stream_sha256":proposal_digest.hexdigest(),"committed_state_stream_sha256":state_digest.hexdigest(),
        "policy":asdict(policy),"memory_policy":asdict(memory),"runtime_GT_used":False,"score_before_write":True,
        "adapter_manifest":[] if adapter is None else adapter.manifest,
        "controller_manifest":[] if controller is None else controller.manifest,
        "memory_predictor_manifest":[] if memory_predictor is None else memory_predictor.manifest,
        "native_predictor_manifest":[] if native_predictor is None else native_predictor.manifest}


def posthoc(sequence,frames,event,trace,baseline):
    # Called only after all runtime frames have completed. No labels are fed
    # back to the already-sealed tracker or its immutable learned models.
    gt=gt_by_frame(DATASET/"train"/sequence/"gt/gt.txt");truth=target_truth(event,gt)
    writes=[];margins=[];rank1=competitive=visible=covered=0;drift=[];funnel=Counter();takeovers=[]
    continuity_changes=0;previous_identity=None;recoveries=0;last_none=False
    origins={};target_correct=[];baseline_correct=[];other_damage=[]
    for f,((_,rows),d,b) in enumerate(zip(frames,trace,baseline)):
        matched=match_frame(rows,gt.get(f,[]));by_uid={str(r["candidate_uid"]):r for r in rows}
        for output in b["outputs"]:
            p=output["public_id"]
            if p not in origins:origins[p]=matched.get(output["candidate_uid"])
        if f<=int(event["event_frame"]):continue
        target=d["target_public_id"];origins[target]=truth
        uid=d["target_uid"];base_uid=b["target_uid"]
        correct=matched.get(uid)==truth;bc=matched.get(base_uid)==truth
        target_correct.append((f,correct));baseline_correct.append((f,bc))
        identity=matched.get(uid)
        if previous_identity is not None and identity is not None:continuity_changes+=int(previous_identity!=identity)
        previous_identity=identity
        recoveries+=int(last_none and correct);last_none=uid is None
        visible+=int(any(p==truth for p,_ in gt.get(f,[])))
        positives=[i for i,r in enumerate(rows) if matched[str(r["candidate_uid"])]==truth];negatives=[i for i in range(len(rows)) if i not in positives]
        covered+=int(bool(positives));scores=np.asarray(d["identity_scores"])
        if positives and negatives:
            competitive+=1;gap=float(max(scores[positives])-max(scores[negatives]));margins.append(gap);rank1+=int(gap>0)
        write={**d["memory"],"sequence":sequence,"correct":matched.get(d["memory"].get("candidate_uid"))==truth};writes.append(write)
        if "state_divergence_from_anchor" in write:drift.append(write["state_divergence_from_anchor"])
        bmaps={r["public_id"]:r["candidate_uid"] for r in b["outputs"]};tmaps={r["public_id"]:r["candidate_uid"] for r in d["outputs"]}
        damage=sum(origin is not None and matched.get(bmaps.get(p))==origin and matched.get(tmaps.get(p))!=origin for p,origin in origins.items() if p!=target)
        other_damage.append(damage)
        proposals=d["proposals"];funnel["eligible_frames"]+=1;funnel["proposed_actions"]+=len(proposals);funnel["feasible_actions"]+=sum(p["feasible"] for p in proposals)
        funnel["predicted_beneficial_actions"]+=sum(p.get("prediction",{}).get("beneficial",0)>=d.get("beneficial_min",.7) for p in proposals)
        if d["selected_action"] is not None:
            funnel["authority_approved_frames"]+=1
            changed=d["global_assignment_changed"];funnel["assignment_changed_frames"]+=int(changed)
            funnel["correct_target_recovered_frames"]+=int(correct and not bc);funnel["other_tracks_preserved_frames"]+=int(damage==0)
            takeovers.append({"frame":f,"action":d["selected_action"],"target_correct":correct,"baseline_target_correct":bc,"non_target_damage":damage,
                "proposed":len(proposals),"memory_accepted":d["memory"].get("accepted",False),"identity_scores":d["identity_scores"]})
    persistent={}
    delta={f:int(c)-int(bc) for (f,c),(_,bc) in zip(target_correct,baseline_correct)}
    for h in (5,10,30):
        eligible=[r["frame"] for r in takeovers if r["target_correct"] and not r["baseline_target_correct"] and r["frame"]+h<len(frames)]
        successes=sum(all(delta.get(f+k,0)>0 for k in range(1,h+1)) for f in eligible)
        persistent[f"H{h}"]={"eligible_corrections":len(eligible),"all_future_frames_persist":successes,"rate":successes/len(eligible) if eligible else None}
    memory_summary=memory_metrics(writes)
    return {"memory":memory_summary,"identity":{"competitive":competitive,"hard_negative_wins":rank1,"hard_negative_win_rate":rank1/competitive if competitive else None,"median_margin":float(np.median(margins)) if margins else None,"visible":visible,"covered":covered,"coverage":covered/visible if visible else None,"mean_state_drift":float(np.mean(drift)) if drift else None},
        "target":{"strict_identity_correct_frames":sum(v for _,v in target_correct),"baseline_strict_correct_frames":sum(v for _,v in baseline_correct),"continuity_identity_changes":continuity_changes,"recoveries_from_NONE":recoveries,"non_target_correct_frame_damage":sum(other_damage),"persistent":persistent},
        "funnel":dict(funnel),"executed_interventions":takeovers,"GT_opened_only_after_runtime_sealed":True}


class EvaluationBatch:
    def __init__(self,group,sequences):
        self.group=group;self.sequences=list(sequences);self.temporary=tempfile.TemporaryDirectory(prefix="intermot-r4r1-eval-")
        self.root=Path(self.temporary.name);self.names=[];self.manifests={}
    def add(self,name,sequence,trace,profile):
        if sequence not in self.sequences:raise ValueError("evaluator sequence axis")
        if name not in self.names:self.names.append(name)
        b,manifest=baseline_trace(sequence);text=trajectory_text(trace);digest=hashlib.sha256(text.encode()).hexdigest()
        path=self.root/"trackers"/name/"data"/f"{sequence}.txt";path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
        changes=[]
        for d,source in zip(trace,b):
            source_axis={r["candidate_uid"]:r["public_id"] for r in source["outputs"]};axis={r["candidate_uid"]:r["public_id"] for r in d["outputs"]}
            if set(axis)!=set(source_axis):raise RuntimeError("full candidate output coverage changed")
            delta={u:p for u,p in axis.items() if p!=source_axis[u]}
            if delta:changes.append({"frame":d["frame"],"changed_ownership":delta})
        artifact_path=ASSETS/"trajectories"/f"{digest}.jsonl.zst"
        artifact=stream_zstd(artifact_path,changes) if not artifact_path.exists() else {"path":str(artifact_path),"sha256":sha256(artifact_path),"rows":len(changes)}
        audit={**profile,"sequence":sequence,"name":name,"trajectory_sha256":digest,"trajectory_rows":len(text.splitlines()),"delta":artifact,"base_trajectory_sha256":manifest["trajectory_sha256"],"base_trace_sha256":manifest["trace_sha256"],"base_trace_path":manifest["trace_path"],"complete_non_target_outputs":True,"delta_reconstruction_requires_only_existing_source_outputs_and_UID_public_changes":True}
        self.manifests.setdefault(name,{})[sequence]=audit
        return audit
    def evaluate(self):
        seqmap=self.root/"seqmap.txt";seqmap.write_text("name\n"+"\n".join(self.sequences)+"\n")
        run=run_trackeval_many(self.root/"trackers",self.root/"eval",self.names,seqmap,gt_folder=DATASET/"train")
        metrics={n:trackeval_summary(parse_trackeval(self.root/"eval",n,self.sequences)) for n in self.names}
        for n,m in metrics.items():
            if any(m[k] is None for k in ("HOTA","AssA","DetA","LocA","IDF1","MOTA","IDSW","FP","FN")):raise RuntimeError("missing official metric")
        log_path=ASSETS/"evaluation_logs"/(self.group.replace("/","__")+".jsonl.zst")
        log=stream_zstd(log_path,[{"stdout":Path(run["log"]).read_text()}])
        result={"metrics":metrics,"manifests":self.manifests,"command":run["command"],"returncode":run["returncode"],"log":log,"full_sequence":True,"same_evaluator_config_for_all_variants":True,"temporary_raw_MOT_copies_removed_after_sealing":True}
        write_json(OUT/"evaluations"/(self.group+".json"),result);return result
    def close(self):self.temporary.cleanup()


def reconstruct_mot(manifest):
    base=read_zstd_jsonl(Path(manifest["base_trace_path"]))
    if sha256(Path(manifest["base_trace_path"]))!=manifest["base_trace_sha256"]:raise ValueError("reconstruction source SHA mismatch")
    delta=manifest["delta"]
    if sha256(Path(delta["path"]))!=delta["sha256"]:raise ValueError("ownership delta SHA mismatch")
    changes={r["frame"]:r["changed_ownership"] for r in read_zstd_jsonl(Path(delta["path"]))}
    for d in base:
        for row in d["outputs"]:row["public_id"]=changes.get(d["frame"],{}).get(row["candidate_uid"],row["public_id"])
    text=trajectory_text(base)
    if hashlib.sha256(text.encode()).hexdigest()!=manifest["trajectory_sha256"]:raise ValueError("reconstructed trajectory SHA differs")
    return text
