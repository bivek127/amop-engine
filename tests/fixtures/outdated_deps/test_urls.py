"""These all pass, on purpose.

Unlike tests/fixtures/buggy_calculator (a seeded bug the chain must
find), this fixture's suite is green from the start: Section 6.7's
dependency flow is "bump the manifest, then confirm the suite still
passes", so a green baseline is what makes a post-bump failure mean
something.
"""

from app.urls import join_path, normalize


def test_normalize_strips_whitespace():
    assert normalize("  https://example.com  ") == "https://example.com"


def test_normalize_strips_one_trailing_slash():
    assert normalize("https://example.com/") == "https://example.com"


def test_normalize_keeps_a_bare_slash():
    assert normalize("/") == "/"


def test_join_path_uses_exactly_one_slash():
    assert join_path("https://example.com/", "/v1/users") == "https://example.com/v1/users"


def test_join_path_handles_a_clean_base():
    assert join_path("https://example.com", "v1") == "https://example.com/v1"
