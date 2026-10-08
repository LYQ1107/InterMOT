import pytest
from scripts.n72r20r4_publish_git import commit_payload, git_person


def test_git_commit_transfer_preserves_person_timezone_and_message():
    person="Research Bot <bot@example.org> 0 +0800"
    assert git_person(person)=={"name":"Research Bot","email":"bot@example.org","date":"1970-01-01T08:00:00+08:00"}
    raw=("tree "+"a"*40+"\nparent "+"b"*40+"\nauthor "+person+"\ncommitter "+person+"\n\nexact message\n").encode()
    payload=commit_payload(raw)
    assert payload["message"]=="exact message\n"
    assert payload["parents"]==["b"*40]
    assert payload["tree"]=="a"*40


def test_git_transfer_rejects_unsupported_metadata_instead_of_rewriting_it():
    with pytest.raises(ValueError):git_person("invalid identity")
    with pytest.raises(ValueError,match="unsupported"):
        commit_payload(b"tree abc\ngpgsig signature\n\nmessage")
