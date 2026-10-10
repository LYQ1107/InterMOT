"""Explicit endpoint boundary: current action t is separate from future t+k."""


def future_window(rows, event_frame, horizon):
    if horizon not in (1, 5, 20, 50, 100):
        raise ValueError("Unregistered future horizon")
    start, end = int(event_frame) + 1, int(event_frame) + int(horizon)
    selected = [row for row in rows if start <= int(row["frame"]) <= end]
    if [int(r["frame"]) for r in selected] != list(range(start, start + len(selected))):
        raise ValueError("Future window must preserve the contiguous original frame axis")
    return selected, len(selected) == horizon
