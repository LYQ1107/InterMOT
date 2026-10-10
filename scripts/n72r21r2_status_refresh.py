"""Refresh verified progress counts without claiming scientific completion."""
from scripts.n72r21r2_common import OUT, ASSETS, read_json, update_status, storage
from scripts.n72r21r2_input_driver import alive_owned_worker


def run():
    candidates = sorted(p.stem for p in (OUT / "data/candidate_integrity").glob("*.json"))
    baselines = sorted(p.stem for p in (OUT / "mot/baseline_results").glob("*.json"))
    active = []
    for p in (OUT / "data").glob("extraction_preflight_v2/*.json"):
        value = read_json(p)
        if alive_owned_worker(value["pid"], "scripts.n72r21r2_candidates"):
            active.append({"sequence": value["sequence"], "pid": value["pid"], "physical_GPU": value["visible_physical_GPU"]})
    update_status(new_fresh_sequence_candidate_extraction_completed=len(candidates), verified_fresh_candidate_sequences=candidates,
                  fresh_baselines_completed=len(baselines), verified_fresh_baseline_sequences=baselines,
                  fresh_current_workers=active, resource_snapshot=storage(), historical_baseline_actual_replays=8,
                  historical_baseline_all_nine_metrics_AA=True, application_goal_active=True,
                  scientific_success=None, next_stage_authorized=False)
    print({"fresh_candidates": len(candidates), "full_fresh_baselines": len(baselines), "active_workers": active, "goal": "ACTIVE"})


if __name__ == "__main__":
    run()
