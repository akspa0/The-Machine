from __future__ import annotations

"""Utility to split a list of calls into N balanced sub-shows.

A *call* is a dict with keys:
    id          – unique identifier (call_id)
    start       – absolute start time (seconds) within the show
    end         – absolute end time (seconds)
    duration    – float seconds (redundant but pre-computed for speed)
    tags        – set[str] of category strings (lower-case)

`segment_calls()` returns a list (len ≤ num_segments) where each item is a
list of calls (preserving original order) destined for one sub-show.

Algorithm (simple but effective):
    1.  Sort by timeline order.
    2.  Maintain `num_segments` bins.  For each incoming call choose the bin
        with the *lowest* cumulative duration so far.  If several bins tie,
        prefer the one whose tag set has the **fewest overlaps** with this
        call's tags – this nudges the algorithm toward theme diversity.

This runs in O(C * N) time where C = number of calls.
"""

from collections import Counter
from typing import List, Dict, Any, Set


def _tag_overlap(a: Set[str], b: Set[str]) -> int:
    """Return |a ∩ b|."""
    return len(a & b)


def segment_calls(
    calls: List[Dict[str, Any]],
    num_segments: int = 4,
) -> List[List[Dict[str, Any]]]:
    """Segment *calls* into *num_segments* balanced buckets.

    The function never creates empty leading bins.  If there are fewer calls
    than *num_segments*, some bins at the end may be empty and will be
    omitted from the return value.
    """

    if not calls:
        return []

    # Ensure timeline order.
    calls_sorted = sorted(calls, key=lambda c: c["start"])

    # Initialise bins.
    bins: List[Dict[str, Any]] = [
        {"calls": [], "duration": 0.0, "tags": Counter()} for _ in range(num_segments)
    ]

    for call in calls_sorted:
        c_dur = call["duration"]
        c_tags: Set[str] = set(call.get("tags", []))

        # Compute a simple score for each bin: (duration, overlap)
        best_idx = None
        best_score = None
        for idx, b in enumerate(bins):
            dur_score = b["duration"]  # we minimise this
            overlap = _tag_overlap(c_tags, set(b["tags"].keys()))
            score = (dur_score, overlap)  # tuple comparison prioritises dur
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx

        # Assign to selected bin.
        tgt = bins[best_idx]
        tgt["calls"].append(call)
        tgt["duration"] += c_dur
        for tag in c_tags:
            tgt["tags"][tag] += 1

    # Drop trailing empty bins.
    result = [b["calls"] for b in bins if b["calls"]]
    return result 