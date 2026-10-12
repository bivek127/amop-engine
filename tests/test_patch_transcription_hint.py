"""Pure-function tests for the "right region, mistranscribed" rejection
detail: _check_content_fidelity must still make the same accept/reject
decisions it always did, but a rejection now also names the nearest real
block and how the diff's transcription differed from it. No sandbox or
Docker needed."""

from amop.sandbox.tools import (
    _check_content_fidelity,
    _describe_transcription_error,
    _locate_nearest_block,
)

FILE = (
    "def add(a, b):\n"
    "    return a + b\n"
    "\n"
    "\n"
    "def divide(a, b):\n"
    "    if b == 0:\n"
    "        raise ZeroDivisionError('b is zero')\n"
    "    return a / b\n"
)


def _diff(old_lines: list[str], new_lines: list[str], start: int = 5) -> str:
    body = "".join(f" {line}\n" for line in old_lines[:1])
    body += "".join(f"-{line}\n" for line in old_lines[1:])
    body += "".join(f"+{line}\n" for line in new_lines)
    return f"@@ -{start},{len(old_lines)} +{start},{len(old_lines)} @@\n{body}"


def test_faithful_diff_still_accepted_in_both_modes():
    diff = _diff(
        ["def divide(a, b):", "    if b == 0:"], ["    if b == 0.0:"], start=5
    )
    assert _check_content_fidelity(diff, FILE, position_trustworthy=False) is None
    assert _check_content_fidelity(diff, FILE, position_trustworthy=True) is None


def test_indentation_drift_is_named_and_real_block_shown():
    diff = _diff(
        ["def divide(a, b):", "  if b == 0:", "      raise ZeroDivisionError('b is zero')"],
        ["  if b <= 0:"],
    )
    message = _check_content_fidelity(diff, FILE, position_trustworthy=False)
    assert message is not None
    assert "indentation/whitespace differs only" in message
    assert "Closest real block starts at line 5" in message
    # the real text is quoted verbatim so a retry can copy it
    assert "6:     if b == 0:" in message


def test_dropped_blank_line_is_named():
    diff = _diff(
        ["def add(a, b):", "    return a + b", "", "def divide(a, b):"],
        ["def add(a, b):"],
        start=1,
    )
    message = _check_content_fidelity(diff, FILE, position_trustworthy=False)
    assert message is not None
    assert "dropped real file line" in message


def test_garbled_token_reports_text_difference():
    diff = _diff(
        ["def divide(a, b):", "    if b == 0:", "        raise ZeroDivisonError('b is zero')"],
        ["        pass"],
    )
    message = _check_content_fidelity(diff, FILE, position_trustworthy=False)
    assert message is not None
    assert "text differs" in message
    assert "ZeroDivisonError" in message


def test_strict_mode_appends_the_same_detail():
    diff = _diff(
        ["def divide(a, b):", "  if b == 0:"], ["  if b <= 0:"], start=5
    )
    message = _check_content_fidelity(diff, FILE, position_trustworthy=True)
    assert message is not None
    assert "just shown this exact region" in message  # original wording intact
    assert "Closest real block starts at line 5" in message


def test_nothing_resembling_the_file_gets_no_detail_but_still_rejects():
    diff = _diff(["totally_unrelated()", "nothing_here()"], ["x = 1"])
    message = _check_content_fidelity(diff, FILE, position_trustworthy=False)
    assert message is not None
    assert "ANYWHERE in the file" in message
    assert "Closest real block" not in message
    assert _locate_nearest_block(["totally_unrelated()"], FILE.splitlines()) is None
    assert _describe_transcription_error(["totally_unrelated()"], FILE.splitlines()) is None


def test_trailing_real_lines_past_the_block_are_not_reported_as_defects():
    real = FILE.splitlines()
    seq = ["def divide(a, b):", "  if b == 0:"]  # indentation wrong, 2 lines
    detail = _describe_transcription_error(seq, real)
    assert detail is not None
    assert "dropped real file line" not in detail
