"""TRAIN-only stratified counterfactual windows, never summed as full HOTA."""
from __future__ import annotations
import tempfile
from pathlib import Path
from scripts.n72r20r4r1_common import *
from scripts.n72r20r3r2r3_pipeline import run_trackeval_many,parse_trackeval,trackeval_summary


class WindowAudit:
    def __init__(self,sequence,gt):
        self.sequence=sequence;self.gt=gt;self.temporary=tempfile.TemporaryDirectory(prefix="intermot-r4r1-window-")
        self.root=Path(self.temporary.name);self.windows={}

    def add(self,identity,frame,stratum,index,baseline,treatment):
        name=f"w{identity}_{frame}_{stratum}_{index}"
        folder=self.root/"gt"/name/"gt";folder.mkdir(parents=True)
        lines=[]
        for f in range(frame,frame+31):
            for p,(x,y,x2,y2) in self.gt.get(f,[]):
                lines.append(f"{f-frame+1},{p},{x},{y},{x2-x},{y2-y},1,1,1\n")
        (folder/"gt.txt").write_text("".join(lines))
        (folder.parent/"seqinfo.ini").write_text("[Sequence]\nname="+name+"\nseqLength=31\n")
        for variant,trace in (("WINDOW_BASE",baseline),("WINDOW_FORCED",treatment)):
            path=self.root/"trackers"/variant/"data"/f"{name}.txt";path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(trajectory_text([{**d,"frame":d["frame"]-frame} for d in trace]))
        self.windows[name]={"source_sequence":self.sequence,"identity":identity,"frame":frame,"stratum":stratum,
            "all_GT_identities_retained":True,"all_public_outputs_retained":True,"independent_branch_state":True}

    def evaluate(self):
        if not self.windows:return {"status":"NOT_RUN_NO_LONG_EVENTS"}
        names=list(self.windows);seqmap=self.root/"seqmap.txt";seqmap.write_text("name\n"+"\n".join(names)+"\n")
        run=run_trackeval_many(self.root/"trackers",self.root/"eval",["WINDOW_BASE","WINDOW_FORCED"],seqmap,gt_folder=self.root/"gt")
        metrics={name:trackeval_summary(parse_trackeval(self.root/"eval",name,names)) for name in ("WINDOW_BASE","WINDOW_FORCED")}
        artifact=stream_zstd(ASSETS/"counterfactual/window_logs"/f"{self.sequence}.jsonl.zst",[{"stdout":Path(run["log"]).read_text()}])
        result={"status":"COMPLETE","windows":self.windows,"metrics":metrics,"log":artifact,"command":run["command"],"returncode":run["returncode"],
            "TRAIN_ONLY_POSTHOC_DIAGNOSTIC":True,"overlapping_local_gains_not_summed_as_full_sequence_HOTA":True,"window_combined_metric_not_full_sequence_result":True,
            "window_GT_uses_all_identities_not_clicked_target_only":True,"temporary_inputs_removed_after_sealing":True}
        write_json(OUT/"counterfactual/window_trackeval"/f"{self.sequence}.json",result)
        return result

    def close(self):self.temporary.cleanup()
