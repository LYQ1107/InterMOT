import pytest

from sam3_intermot.backend.base import NotSupportedError
from sam3_intermot.backend.sam3_backend import (
    CheckpointUnavailableError,
    Sam3Backend,
)


def test_sam3_backend_blocks_without_checkpoint():
    backend = Sam3Backend(checkpoint_path=None)
    with pytest.raises(CheckpointUnavailableError):
        backend.start_video("dummy")


def test_sam3_backend_mask_prompt_not_supported():
    backend = Sam3Backend(checkpoint_path=None)
    with pytest.raises(NotSupportedError):
        backend.add_mask(0, 1, None)


def test_sam3_backend_reset_object_not_supported():
    backend = Sam3Backend(checkpoint_path=None)
    with pytest.raises(NotSupportedError):
        backend.reset_object(1)
