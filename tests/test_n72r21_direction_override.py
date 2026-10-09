from scripts import n72r21_progress as progress
from scripts.n72r21_common import OUT,read_json


def test_latest_MOT_direction_is_preserved_by_progress_refresh():
    status={'status':'ACTIVE_T2_FROZEN_VALIDATION_AND_LAWFUL_DOMAIN_DIAGNOSTICS'}
    progress.apply_direction_override(status)
    assert status['status']=='ACTIVE_MOT_REFOCUS_WITH_FROZEN_IDENTITY_DIAGNOSTICS'
    assert status['research_primary_track']=='INTERACTIVE_MULTI_OBJECT_TRACKING'
    assert status['SOT_work_status']=='DEFERRED_BY_USER'
    assert not status['new_SOT_work_authorized']
    assert not status['independent_one_click_replays_are_full_MOT']


def test_direction_change_does_not_rewrite_history_or_claim_MOT_success():
    direction=read_json(OUT/'RESEARCH_DIRECTION_OVERRIDE.json')
    assert not direction['original_frozen_spec_rewritten']
    assert not direction['new_association_redesign_or_training_started_by_this_record']
    assert direction['scientific_success'] is None
    assert not direction['next_stage_authorized']
