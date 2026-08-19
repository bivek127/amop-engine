"""Session-cookie login for the web dashboard (Section 15.3 / D-10),
gated by the same single shared secret every other interface already
holds (`AMOP_API_TOKEN`) -- no separate web password, applying
CLAUDE.md's "a single shared-secret API token is sufficient for v1"
scope decision consistently across all three interfaces, not just the
API and Telegram.

A signed, time-limited cookie (itsdangerous), never the raw token
handed back to the browser: the submitted token is checked once, on
POST /web/login, via the same constant-time comparison Stage 1's API
auth uses (`amop.api.auth`); the cookie itself only carries a signed
"this browser proved it once" assertion, so a leaked cookie doesn't
also leak the shared secret.
"""

import secrets

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from starlette.requests import Request

from amop.api.auth import _configured_token

SESSION_COOKIE_NAME = "amop_session"
# Deliberately short-lived for a v1 single-shared-secret setup -- there's
# no per-user revocation, so a day-long cookie caps how long a stolen
# one stays useful without adding real session management this
# milestone doesn't call for.
SESSION_MAX_AGE_SECONDS = 24 * 60 * 60
_SESSION_SALT = "amop-web-session"


class WebAuthRequired(Exception):
    """Raised by `require_web_session` when the browser has no valid
    session cookie. Caught by app.py's own handler and turned into a
    redirect to the login page -- a distinct exception type (not
    `api_error`'s 401 envelope) because a browser without a session and
    an API caller without a bearer token need genuinely different
    responses: a redirect for one, JSON for the other.
    """


def _serializer() -> URLSafeTimedSerializer:
    # Built fresh per call, not cached at import time, matching
    # `_configured_token()`'s own "read the env var inside the request
    # path" discipline -- a token rotated in .env takes effect on the
    # next request rather than requiring a process restart to notice.
    return URLSafeTimedSerializer(_configured_token(), salt=_SESSION_SALT)


def token_is_valid(submitted_token: str) -> bool:
    return secrets.compare_digest(submitted_token, _configured_token())


def make_session_cookie_value() -> str:
    return _serializer().dumps({"authenticated": True})


def require_web_session(request: Request) -> None:
    cookie = request.cookies.get(SESSION_COOKIE_NAME)
    if not cookie:
        raise WebAuthRequired()
    try:
        _serializer().loads(cookie, max_age=SESSION_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired):
        raise WebAuthRequired()
