"""Report aggregation with one deliberate, real hot spot.

Seeded for Milestone 14's Optimizer fixture (spec 19.3's "real code,
known-good fix" applied to performance rather than correctness):
`count_unique_visitors` does a linear scan of a list for every element,
which is O(n^2). The known-good fix is a set. Everything here is
correct -- it is merely slow, which is the point: the Optimizer must
make it faster WITHOUT changing what it returns.

Do not "helpfully" optimize this by hand.
"""


def count_unique_visitors(visitor_ids: list[str]) -> int:
    """Count distinct visitor ids.

    Correct, but quadratic: `in` over a list is a full scan each time.
    """
    seen: list[str] = []
    for visitor in visitor_ids:
        if visitor not in seen:
            seen.append(visitor)
    return len(seen)


def summarize(visitor_ids: list[str]) -> dict:
    return {
        "total_events": len(visitor_ids),
        "unique_visitors": count_unique_visitors(visitor_ids),
    }
