"""Leakage guard helpers used by simulators and evaluators."""

from dataclasses import dataclass


@dataclass
class LeakageReport:
    future_gt_reads: int = 0
    future_detections_used: int = 0
    future_identity_used: int = 0
    details: list = None

    def __post_init__(self):
        if self.details is None:
            self.details = []

    @property
    def clean(self) -> bool:
        return (
            self.future_gt_reads == 0
            and self.future_detections_used == 0
            and self.future_identity_used == 0
        )
