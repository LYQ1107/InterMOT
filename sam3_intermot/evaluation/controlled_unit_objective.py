"""Offline paired benefit targets, never a runtime identity/GT feature."""
from copy import deepcopy
from sam3_intermot.one_click.event_authority_learning import objective_target

CONDITIONS = ("LEGACY_H100_BENEFIT", "CONTROLLED_UNIT_CORRECTION")


def paired_targets(current, future, correction, *, correct_background=False):
    old = objective_target({"current_t": current, "future": future}, "L1_EVENT_CORRECTION")
    if old is None:
        return None
    if correct_background:
        if correction is not None or current["KEEP_outcome"] != "TARGET":
            raise ValueError("Background must be current-correct C0, not an unproved error")
        corrected = 0
    else:
        if correction is None:
            return None  # UNKNOWN and original duplicate arms are not negatives.
        if current["KEEP_outcome"] not in ("VERIFIED_OTHER", "NONE") or current["outcome"] == "UNKNOWN":
            raise ValueError("Known-source wrong KEEP and a verified current outcome required")
        corrected = correction["current_error_unit_corrected"]
        if type(corrected) is not int or corrected not in (0, 1) or corrected != int(current["outcome"] == "TARGET"):
            raise ValueError("Actual current unit-correction label inconsistent")
        if corrected and (current["N01"] != 1 or current["N10"]):
            raise ValueError("Actual current N01 correction required")
    new = deepcopy(old)
    new.update(benefit=float(corrected), scope="CONTROLLED_UNIT_CURRENT_CORRECTION_SHARED_H100_RISK_VALUE_NOT_NATURAL_G1")
    assert new["risk"] == old["risk"] and new["value"] == old["value"]
    return dict(zip(CONDITIONS, (old, new)))
