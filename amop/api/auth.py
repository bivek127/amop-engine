"""Section 15.3 / Design Decision D-10: authentication for the internal
API.

Simplified per CLAUDE.md's own scope decision for this milestone: one
shared-secret token (`AMOP_API_TOKEN`), required on every MUTATING
endpoint (`Depends(require_operator_token)` declared per-route, never
globally on the app -- GET routes stay open, matching both the spec
table's "any" auth column and CLAUDE.md's literal "Single shared-secret
token auth on all mutating endpoints"). D-10's fuller per-surface
mapping (a separate Telegram allowlist, a separate web password) is not
built here as three auth mechanisms -- Telegram's `allowed_user_ids`
allowlist is a *separate*, still-required gate (Stage 2), but it answers
"who may issue bot commands", not "is this HTTP request authenticated";
the bot itself is just another holder of this one token.
"""

import os
import secrets

from fastapi import Depends, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from amop.api.errors import api_error

# auto_error=False: a missing Authorization header should fall through
# to OUR error envelope below, not FastAPI's own default 403 response
# shape, which doesn't match Section 15.2's spec'd envelope.
_bearer_scheme = HTTPBearer(auto_error=False)


def _configured_token() -> str:
    token = os.environ.get("AMOP_API_TOKEN")
    if not token:
        # Fails loud, not silently-open: an unset token must never be
        # read as "no auth required" for a mutating endpoint. Raised
        # inside the request path (not at import time) so a route that
        # doesn't need auth can still be hit while this is being set up.
        raise RuntimeError(
            "AMOP_API_TOKEN is not set -- refusing to authenticate a "
            "mutating request with no configured secret. Set it in "
            ".env, e.g. AMOP_API_TOKEN=<a long random string>."
        )
    return token


async def require_operator_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    """A FastAPI dependency -- add via
    `dependencies=[Depends(require_operator_token)]` on the specific
    route (or router) that mutates state. Raises the standard error
    envelope (401) on a missing or wrong token.

    Constant-time comparison (`secrets.compare_digest`), not `==`: a
    naive equality check short-circuits on the first mismatched byte,
    which leaks how many leading characters of the guess were correct
    through response timing -- a real, if narrow, side channel when a
    single shared secret gates every write this whole system can make.
    """
    expected = _configured_token()
    if credentials is None or not secrets.compare_digest(
        credentials.credentials, expected
    ):
        raise api_error(
            status.HTTP_401_UNAUTHORIZED,
            "UNAUTHORIZED",
            "missing or invalid API token",
        )
