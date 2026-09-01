"""N21 HumanSupervisionLedger.

Records every human-confirmed correction event as supervised identity
evidence. The ledger is the *only* source of online training labels.

Causality contract:
  - each record is appended at the frame where the human correction happens;
  - no record may contain information from future frames or future GT;
  - positive observations are human-confirmed ("this observation is public
    identity P");
  - explicit negative observations are only stored when the correction
    semantics directly certify them ("the wrongly committed hypothesis is
    NOT P");
  - miss corrections store a positive only;
  - box-only corrections are stored with identity_positive=None and are
    never used as identity negatives.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class CorrectionRecord:
    timestamp: str          # ISO string of the correction time
    sequence: str
    frame: int
    public_id: int          # canonical public identity (Human Root namespace)
    correction_type: str    # MISS | ID_WRONG | SWAP | BOX_ONLY | NEW_TARGET
    positive: Optional[dict] = None
    # positive observation: {"candidate_rank": int, "evidence": {...}}
    explicit_negatives: list = field(default_factory=list)
    # list of dicts {"candidate_rank": int, "evidence": {...}} certified by
    # the correction itself (ID_WRONG / SWAP only)
    confidence: str = "HUMAN"
    source: str = "correction_simulator"
    provenance: str = "causal"          # never "future_gt"
    gt_used: bool = False               # offline labelling flag, never True
                                        # inside the live runner
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class HumanSupervisionLedger:
    """Append-only JSONL ledger of human-confirmed supervision."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, rec: CorrectionRecord) -> None:
        if rec.gt_used and rec.source != "offline_labeling":
            raise ValueError("gt_used=True is only allowed for offline labeling")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
        return out

    def summary(self) -> dict[str, Any]:
        recs = self.records()
        by_type: dict[str, int] = {}
        n_pos = n_neg = 0
        for r in recs:
            by_type[r["correction_type"]] = by_type.get(r["correction_type"], 0) + 1
            n_pos += int(r.get("positive") is not None)
            n_neg += len(r.get("explicit_negatives", []))
        return {
            "n_records": len(recs),
            "by_type": by_type,
            "n_positive_observations": n_pos,
            "n_explicit_negatives": n_neg,
            "any_gt_leak": any(r.get("gt_used") and
                               r.get("source") != "offline_labeling"
                               for r in recs),
        }
