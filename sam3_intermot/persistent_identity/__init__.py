"""N14 Human-Writable Persistent Identity State (HWPIS, working name).

The state is only called "internal" if it provably participates in the
detector computation graph (see query causal smoke).
"""

from .human_write_encoder import HumanWriteEncoder, roi_pool_feature
from .head_adapter import SlotHeadAdapter
from .identity_state import PersistentIdentityState
from .injection import (
    build_tgt_with_queries,
    build_ref_boxes_with_queries,
    install_query_patch,
    run_decoder_with_tgt,
)

__all__ = [
    "HumanWriteEncoder",
    "PersistentIdentityState",
    "SlotHeadAdapter",
    "build_ref_boxes_with_queries",
    "build_tgt_with_queries",
    "install_query_patch",
    "roi_pool_feature",
    "run_decoder_with_tgt",
]
