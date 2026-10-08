"""New KEEP runtime versus immutable source state at every observed frame."""
from __future__ import annotations
from sam3_intermot.association.opportunity_tracker import OpportunityTracker
from sam3_intermot.association.causal_identity_tracker import CausalIdentityTracker
from sam3_intermot.association.identity_authority import AuthorityConfig
from scripts.n72r20r4r1_common import *


def run():
    torch.set_num_threads(1);result={}
    for sequence in SEQUENCES:
        event=events()[sequence];frames=load_frames(sequence)
        cfg=AuthorityConfig(**read_json(R4ASSETS/"dev/manifests/B1_CAUSAL_BASELINE"/f"{sequence}.json")["configuration"])
        original=CausalIdentityTracker(config=cfg,event=event,bank=make_bank())
        current=OpportunityTracker(config=cfg,event=event,bank=make_bank())
        digest=hashlib.sha256()
        for payload,rows in frames:
            f=int(payload["frame"]);a=original.step(rows,f);b=current.step(rows,f)
            for key in ("assignments","outputs","births","deaths","state_before","state_after","memory"):
                if a[key]!=b[key]:raise RuntimeError(f"KEEP mismatch {sequence}:{f}:{key}")
            if not np.array_equal(a["base_matrix"],b["base_matrix"]):raise RuntimeError("KEEP score mismatch")
            digest.update(trajectory_text([b]).encode())
        sealed=read_json(R4ASSETS/"dev/manifests/B1_CAUSAL_BASELINE"/f"{sequence}.json")["trajectory_sha256"]
        if digest.hexdigest()!=sealed:raise RuntimeError("KEEP trajectory SHA mismatch")
        result[sequence]={"frames":len(frames),"exact_scores_solver_state_lifecycle_memory_outputs":True,"sha256":sealed}
        write_json(OUT/"audit/NEW_KEEP_AA.json",result)
        print(sequence,"new KEEP identical",flush=True)
    completion=read_json(OUT/"audit/M0_COMPLETION.json");completion["new_KEEP_AA"]="PASS_ALL_EIGHT_SEQUENCES";write_json(OUT/"audit/M0_COMPLETION.json",completion)


if __name__=="__main__":run()
