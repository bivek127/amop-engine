"""A tiny URL helper for the outdated-dependency fixture repo.

Deliberately does NOT import `requests` at module level, even though the
manifest pins it. The sandbox container runs with no network
(network_mode="none") and has only pytest installed, so a fixture whose
tests need the pinned package installed could never pass there -- and
this fixture's job is to exercise the manifest bump and the full-suite
run, not to prove pip works offline.
"""


def normalize(url: str) -> str:
    """Strip whitespace and a single trailing slash from a URL."""
    cleaned = url.strip()
    if cleaned.endswith("/") and len(cleaned) > 1:
        cleaned = cleaned[:-1]
    return cleaned


def join_path(base: str, path: str) -> str:
    """Join a base URL and a path with exactly one separating slash."""
    return f"{normalize(base)}/{path.strip().lstrip('/')}"
