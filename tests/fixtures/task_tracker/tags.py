"""Tag normalization and merging."""


def normalize_tags(tags: list[str]) -> list[str]:
    """Lowercase, strip, dedupe (order-preserving), drop empties."""
    seen = set()
    result = []
    for tag in tags:
        cleaned = tag.strip().lower()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def merge_tag_sets(a: list[str], b: list[str]) -> list[str]:
    """Union of two tag lists, normalized, order-preserving (a's order
    first, then b's new tags)."""
    return normalize_tags([*a, *b])


def tags_match_any(task_tags: list[str], wanted: list[str]) -> bool:
    """True if task_tags shares at least one tag with wanted (after
    normalizing both sides)."""
    return bool(set(normalize_tags(task_tags)) & set(normalize_tags(wanted)))
