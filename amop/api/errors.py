"""Section 15.2's standard error envelope:
{"error_code": "string", "message": "string", "detail": {}}

Applied uniformly regardless of which layer actually raised -- an
explicit `api_error()` call from a route, the state machine's own
`IllegalTransitionError`, FastAPI's own request validation, or an
unhandled exception. A caller should never have to guess which shape a
given error takes; app.py registers one exception handler per case,
all producing this exact envelope.
"""

from fastapi import HTTPException


def api_error(
    status_code: int, error_code: str, message: str, detail: dict | None = None
) -> HTTPException:
    """Raise this from a route body: `raise api_error(404, "X_NOT_FOUND", "...")`.

    The dict shape passed as `detail` here is what app.py's HTTPException
    handler recognizes and serializes as the response body directly --
    see that handler's docstring for why a plain `HTTPException(detail=...)`
    isn't enough on its own (FastAPI's default handler nests it one level
    too deep: `{"detail": {...}}` instead of `{...}`).
    """
    return HTTPException(
        status_code=status_code,
        detail={"error_code": error_code, "message": message, "detail": detail or {}},
    )
