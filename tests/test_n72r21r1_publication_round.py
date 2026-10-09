from copy import deepcopy
import pytest
from scripts.n72r21_connector_publication import digest,remote_commit_bytes
from scripts.n72r21r1_connector_publication_round import verified_dates,configure


def test_normalized_UTC_metadata_recovers_only_exact_raw_Git_SHA():
    timestamp=1791568800
    raw=('tree '+'a'*40+'\nparent '+'b'*40+'\nauthor Unit <unit@example.invalid> '+str(timestamp)+' +0800\n'
         'committer Unit <unit@example.invalid> '+str(timestamp)+' +0800\n\nmessage\n').encode()
    from datetime import datetime,timezone
    normalized=datetime.fromtimestamp(timestamp,timezone.utc).isoformat()
    r={'sha':digest(raw),'tree':{'sha':'a'*40},'parents':[{'sha':'b'*40}],
        'author':{'name':'Unit','email':'unit@example.invalid','date':normalized},
        'committer':{'name':'Unit','email':'unit@example.invalid','date':normalized},'message':'message'}
    original=deepcopy(r);recovered=verified_dates(r)
    assert r==original and recovered['author']['date'].endswith('+08:00')
    assert remote_commit_bytes(recovered)==raw


def test_signed_remote_commit_and_unsafe_namespace_cannot_be_guessed():
    with pytest.raises(ValueError,match='signed'):verified_dates({'verification':{'signature':'signature'}})
    with pytest.raises(ValueError,match='namespace'):configure('../old-attempt')
