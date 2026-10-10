import json
import numpy as np
import pytest
from scripts.n72r21r2_integrity_v2 import scalar


def test_numpy_audit_counts_roundtrip_native_JSON():
    values = {"counts": [np.int64(3), np.int64(0)], "valid": np.bool_(True)}
    assert json.loads(json.dumps(values, default=scalar)) == {"counts": [3, 0], "valid": True}


def test_other_objects_not_silently_stringified():
    with pytest.raises(TypeError):
        json.dumps({"bad": object()}, default=scalar)
