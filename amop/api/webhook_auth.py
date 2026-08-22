"""GitHub webhook signature verification — the actual security-critical
piece of Milestone 21 (spec 8.5/3.3, GitHub's own documented HMAC-SHA256
scheme). Treated with the same weight as any other auth gate in this
project (`api/auth.py`'s `require_operator_token`, the Safety Engine's
`evaluate()`): pure, synchronous, unit-testable without a live server,
and checked BEFORE anything downstream ever treats the payload as data.

Verification must run over the EXACT bytes GitHub signed. This is why
the webhook route (`routes/webhooks.py`) reads `await request.body()`
directly rather than accepting a Pydantic body model — FastAPI would
have to deserialize-then-reserialize to hand back raw bytes, and any
difference in whitespace or key order between what GitHub sent and what
gets re-serialized would invalidate a signature that was never actually
wrong.
"""

import hashlib
import hmac

SIGNATURE_HEADER = "X-Hub-Signature-256"
DELIVERY_HEADER = "X-GitHub-Delivery"
EVENT_HEADER = "X-GitHub-Event"


def compute_signature(raw_body: bytes, secret: str) -> str:
    """GitHub's own documented format: `sha256=<hex hmac>`."""
    digest = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_signature(raw_body: bytes, signature_header: str | None, secret: str) -> bool:
    """True only if `signature_header` is a well-formed `sha256=...`
    value that matches `raw_body` under `secret`.

    `hmac.compare_digest`, not `==`: a naive equality check short-
    circuits on the first mismatched byte, leaking how many leading
    characters were correct through response timing -- the same
    reasoning `api/auth.py`'s token check already applies to the one
    shared API secret, applied here to the webhook secret.

    A missing header, an empty secret, or a header that doesn't even
    start with the expected prefix all fail closed (False) rather than
    raising -- the caller's job is to reject on False, not to
    distinguish "malformed" from "wrong" cases that carry no different
    consequence.
    """
    if not signature_header or not secret:
        return False
    if not signature_header.startswith("sha256="):
        return False
    expected = compute_signature(raw_body, secret)
    return hmac.compare_digest(expected, signature_header)
