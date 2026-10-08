import pytest
from scripts.n72r20r4_publish_git import commit_payload, git_person
from scripts import n72r20r4_publish_git as publisher


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


def test_inline_tree_batches_exact_utf8_and_retains_binary_deleted_and_cached(monkeypatch):
    contents={"text":"中文\r\nexact trailing newline\n".encode(),"binary":b"\xff\x00"}
    def read_blob(*args):
        assert args[:2]==("cat-file","blob")
        return contents[args[2]]
    monkeypatch.setattr(publisher,"git",read_blob)
    records=[{"path":name,"mode":"100644","type":"blob","sha":sha}
             for name,sha in [("a","text"),("b","binary"),("c","cached"),("d",None)]]
    result=publisher.inline_tree_entries(records,{"cached"})
    assert result[0]["content"].encode()==contents["text"] and "sha" not in result[0]
    assert result[1:]==records[1:]
    assert records[0]["sha"]=="text" and "content" not in records[0]


def test_tree_chunks_preserve_all_entries_and_never_split_one_file():
    entries=[{"path":"large","content":"x"*1000},{"path":"small1","content":"a"},{"path":"small2","sha":None}]
    chunks=publisher.tree_chunks(entries,128)
    assert [entry for chunk in chunks for entry in chunk]==entries
    assert chunks[0]==[entries[0]]
    assert chunks[1]==entries[1:]
    with pytest.raises(ValueError):publisher.tree_chunks(entries,0)
