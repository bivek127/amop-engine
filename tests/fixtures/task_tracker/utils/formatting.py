"""Small text-formatting helpers shared by reports.py and notifications.py."""


def truncate(text: str, length: int) -> str:
    """Shorten `text` to `length` chars, appending '...' if truncated."""
    if len(text) <= length:
        return text
    return text[: max(0, length - 3)] + "..."


def pluralize(word: str, count: int) -> str:
    """Naive English pluralization -- good enough for report copy."""
    if count == 1:
        return word
    if word.endswith("y"):
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "ch", "sh")):
        return word + "es"
    return word + "s"


def humanize_list(items: list[str]) -> str:
    """['a'] -> 'a'; ['a','b'] -> 'a and b'; ['a','b','c'] -> 'a, b, and c'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"
