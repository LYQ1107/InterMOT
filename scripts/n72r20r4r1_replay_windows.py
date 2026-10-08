"""Add official windows for a corpus completed before window export existed."""
from scripts.n72r20r4r1_mine import compact,future,baseline_origins
from scripts.n72r20r4r1_window_audit import WindowAudit
from scripts.n72r20r4r1_supervision import prepare_anchors,measure_branch
from scripts.n72r20r4r1_common import *
from scripts.n72r20r3_common import gt_by_frame
from sam3_intermot.association.opportunity_tracker import OpportunityTracker,InterventionPolicy
from sam3_intermot.association.identity_authority import AuthorityConfig
from sam3_intermot.association.opportunity_solver import AssociationAction


def run(sequence):
    torch.set_num_threads(1);path=OUT/"counterfactual/window_trackeval"/f"{sequence}.json"
    if path.exists():return read_json(path)
    frames=load_frames(sequence);gt=gt_by_frame(DATASET/"train"/sequence/"gt/gt.txt");anchors,matches=prepare_anchors(sequence,frames,gt)
    records=read_zstd_jsonl(ASSETS/"counterfactual"/f"{sequence}.jsonl.zst")
    selected=[r for r in records if r["record_type"]=="long_horizon"]
    windows=WindowAudit(sequence,gt)
    try:
        for identity,prepared in sorted(anchors.items()):
            local=[r for r in selected if r["identity_key"][1]==identity]
            by_frame={f:[r for r in local if r["frame"]==f] for f in sorted({r["frame"] for r in local})}
            tracker=OpportunityTracker(config=AuthorityConfig(source="raw"),event=prepared["event"],bank=make_bank(),intervention_policy=InterventionPolicy(source="raw"),audit_hashes=False)
            origins={}
            for payload,rows in frames:
                f=int(payload["frame"])
                if by_frame and f>max(by_frame):break
                before=tracker.clone() if f in by_frame else None
                d=tracker.step(rows,f);origins=baseline_origins(origins,[d],matches)
                if tracker.target_public is not None:origins[tracker.target_public]=identity
                for index,r in enumerate(by_frame.get(f,[])):
                    b=before.clone();t=before.clone();action=AssociationAction(**r["action"])
                    bt=[compact(b.step(rows,f))]+future(b,frames,f+1,30)
                    tt=[compact(t.step(rows,f,forced_action=action))]+future(t,frames,f+1,30)
                    measured=measure_branch(bt,tt,matches,baseline_origins(origins,bt,matches),action.public_id,identity)
                    if measured!=r["H30"]:raise RuntimeError("window replay diverged from sealed H30")
                    windows.add(identity,f,r["stratum"],index,bt,tt)
        result=windows.evaluate()
    finally:windows.close()
    summary_path=OUT/"counterfactual/per_sequence"/f"{sequence}.json";s=read_json(summary_path);s["window_trackeval"]=result["status"];write_json(summary_path,s)
    return result


if __name__=="__main__":run("dancetrack0001")
