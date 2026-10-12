import pytest
from scripts import n72r21r2_diagnostic_storage_resume_v3 as recovery


def test_only_original_two_diagnostic_kinds_registered():
    assert set(recovery.KINDS) == {"EXACT_CF", "PURE_MARGIN"}
    assert recovery.fresh_marker("EXACT_CF") != recovery.fresh_marker("PURE_MARGIN")


@pytest.mark.parametrize("argv,expected", [
    (["python", "other_module"], False),
    (["python", "scripts.n72r21r2_diagnostic_storage_resume_v3"], True),
    (["python", "scripts.n72r21r2_diagnostic_storage_resume_v3", "--kind"], True),
    (["python", "scripts.n72r21r2_diagnostic_storage_resume_v3", "--kind", "EXACT_CF"], True),
    (["python", "scripts.n72r21r2_diagnostic_storage_resume_v3", "--kind", "PURE_MARGIN"], False),
])
def test_duplicate_same_kind_rejected_other_kind_may_run_independently(argv, expected):
    assert recovery.same_kind_worker(argv, "EXACT_CF") is expected


@pytest.mark.parametrize("kind,prefix,artifact,folder", [
    ("EXACT_CF", "availability/exact_cf_v1", "exact_cf_verifier_v1/predictions", "predictions"),
    ("PURE_MARGIN", "availability/pure_margin_v1", "open_set_pure_margin_v1/exact_CF", "CF_predictions"),
])
def test_partial_artifact_or_one_sided_receipt_never_retry_authority(tmp_path, monkeypatch, kind, prefix, artifact, folder):
    monkeypatch.setattr(recovery, "OUT", tmp_path / "out")
    monkeypatch.setattr(recovery, "ASSETS", tmp_path / "assets")
    sequence = "dancetrack0075"
    assert recovery.no_partial_score(kind, sequence)
    partial = recovery.ASSETS / artifact / (sequence + ".jsonl.zst.partial")
    partial.parent.mkdir(parents=True)
    partial.touch()
    assert not recovery.no_partial_score(kind, sequence)
    partial.unlink()
    receipt = recovery.OUT / prefix / folder / (sequence + ".json")
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}")
    assert not recovery.no_partial_score(kind, sequence)
