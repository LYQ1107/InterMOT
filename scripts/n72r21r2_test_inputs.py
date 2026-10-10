"""SHA-checked read-only reuse of five unuploaded historical unit fixtures."""
from pathlib import Path
from scripts.n72r21r2_common import ROOT, OUT, R1, sha256, read_json, write_json


def run():
    previous = read_json(R1 / "outputs/N72R21R1/tests/LOCAL_HISTORICAL_FIXTURE_REUSE.json")
    records = []
    for record in previous["records"]:
        source = Path(record["source_path"])
        assert sha256(source) == record["sha256"]
        target = ROOT / record["relative_path"]
        if target.exists():
            assert sha256(target) == record["sha256"]
        else:
            assert not target.is_symlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)
        records.append({"source_path": str(source), "relative_path": record["relative_path"], "sha256": record["sha256"], "reused_by_local_symlink": True})
    write_json("tests/LOCAL_HISTORICAL_FIXTURE_REUSE.json", {"stage": "N72R21R2", "status": "LOCAL_TEST_INPUTS_LINKED_NOT_PUBLISHED",
               "records": records, "files_copied": 0, "historical_sources_modified": False,
               "TrackEval_local_shared_clone": "Pinned source checkout reuses local Git objects; no network/submodule version change",
               "five_historical_failure_tests_not_hidden_or_patched": True})


if __name__ == "__main__":
    run()
