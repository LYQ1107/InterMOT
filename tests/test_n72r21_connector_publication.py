import pytest
from scripts.n72r21_connector_publication import remote_commit_bytes,digest


def record():
    r={'tree':{'sha':'a'*40},'parents':[{'sha':'b'*40}],
       'author':{'name':'Test','email':'test@example.org','date':'2026-10-09T00:00:00Z'},
       'committer':{'name':'Test','email':'test@example.org','date':'2026-10-09T00:00:00Z'},
       'message':'actual delivery\n','verification':{'signature':None}}
    raw=('tree '+'a'*40+'\nparent '+'b'*40+'\nauthor Test <test@example.org> 1791504000 +0000\ncommitter Test <test@example.org> 1791504000 +0000\n\nactual delivery\n').encode()
    r['sha']=digest(raw)
    return r,raw


def test_actual_canonical_bytes_are_SHA_checked_not_metadata_guessed():
    r,raw=record();assert remote_commit_bytes(r)==raw
    r['message']=r['message'].rstrip('\n');assert remote_commit_bytes(r)==raw


def test_changed_tree_or_author_cannot_pass_actual_remote_SHA():
    r,_=record();r['tree']['sha']='c'*40
    with pytest.raises(ValueError,match='actual commit SHA'):remote_commit_bytes(r)


def test_signed_remote_headers_must_not_be_silently_discarded():
    r,_=record();r['verification']['signature']='a signature'
    with pytest.raises(ValueError,match='signed remote'):remote_commit_bytes(r)
