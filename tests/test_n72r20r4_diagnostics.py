from scripts.n72r20r4_diagnostics import merged_windows, owner_reallocations, preserve_gt_window


def test_local_metrics_windows_cover_all_interventions_and_preserve_GT(tmp_path):
    events=[{"frame":70},{"frame":20},{"frame":10}]
    assert merged_windows(events,100)==[(9,50),(69,99)]
    source=tmp_path/"gt.txt"
    source.write_text("1,4,1,2,10,20,0,8,0.25\n2,7,4,5,10,20,1,1,0.75\n3,7,4,5,10,20,1,1,0.8\n")
    assert preserve_gt_window(source,1,1)=="1,7,4,5,10,20,1,1,0.75\n"


def test_offline_owner_reallocations_distinguish_changes_from_collisions():
    row={"baseline_all_public_assignments":{"1":"a","2":"b","3":None},
         "treatment_all_public_assignments":{"1":"b","2":"a","3":None}}
    assert owner_reallocations(row)==[
        {"candidate_uid":"a","baseline_public_id":"1","treatment_public_id":"2"},
        {"candidate_uid":"b","baseline_public_id":"2","treatment_public_id":"1"}]
