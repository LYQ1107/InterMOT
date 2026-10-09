from scripts.n72r21r1_connector_publication import entries,BRANCH


def test_base_tree_contains_only_expected_git_object_kinds():
    base=entries('051e593f073216e34b3744b17eff11c5f56b3956')
    assert base['README.md']['type']=='blob'
    assert base['third_party/MOTIP/TrackEval']['type']=='commit'
    assert base['third_party/MOTIP/TrackEval']['sha']=='12c8791b303e0a0b50f753af204249e622d0281a'
    assert BRANCH=='codex/n72r21r1-safe-joint-mot-intervention'
