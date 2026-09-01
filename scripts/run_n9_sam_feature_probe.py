"""Probe pinned SAM3 for per-object native features (obj_ptr / maskmem)."""

import json
from pathlib import Path

import numpy as np
import yaml

from sam3_intermot.backend.sam3_backend import Sam3Backend


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    video = Path("/path/to/dancetrack/val/dancetrack0004/img1")
    backend = Sam3Backend(
        checkpoint_path=cfg["backend"]["checkpoint_path"],
        max_num_objects=cfg["backend"]["max_num_objects"],
        multiplex_count=cfg["backend"]["multiplex_count"],
        use_fa3=False,
        use_rope_real=True,
        compile=False,
        warm_up=False,
        async_loading_frames=False,
    )
    try:
        backend.start_video(str(video))
        dets = backend.detect_concept(0, "person")
        print("det0", len(dets), flush=True)
        out = backend.propagate(0, 60, start_frame_index=0, keep_masks=False)
        print("prop frames", len(out), flush=True)
        state = backend._predictor._all_inference_states[backend._session_id]["state"]
        od = state.get("output_dict", {})
        print("output_dict keys", list(od.keys()), flush=True)
        report = {"frames": {}, "samples": []}
        for key in od:
            for f, out in list(od[key].items()):
                if f > 60:
                    continue
                info = {}
                for k in ("obj_ptr", "maskmem_features", "obj_id_to_idx", "out_obj_ids"):
                    v = out.get(k)
                    if v is not None:
                        try:
                            info[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
                        except Exception as e:
                            info[k] = {"type": str(type(v)), "err": str(e)}
                report["frames"][str(f)] = info
        # per-frame obj_ptr raw values for the first few frames
        for key in od:
            for f in sorted(od[key]):
                if f > 12:
                    break
                out = od[key][f]
                ptr = out.get("obj_ptr")
                if ptr is not None:
                    report["samples"].append(
                        {
                            "frame": int(f),
                            "key": key,
                            "obj_ptr_norm": float(np.linalg.norm(ptr.detach().cpu().float().numpy())),
                            "obj_ptr_shape": list(ptr.shape),
                        }
                    )
        out_path = ROOT / "outputs/n9/sam_feature_probe.json"
        out_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, default=str)[:6000], flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
