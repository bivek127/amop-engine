"""FastAPI app -- Section 3.1's Command Layer.

Every interface calls through here rather than getting a privileged
path into the orchestrator/database directly. The CLI is the one
partial exception (Milestone 15's own scope decision: it stays on
direct orchestrator calls rather than a full rewrite to HTTP, with a
consistency check in tests/test_milestone15.py standing in for "goes
through the API" -- see the plan's own reasoning); Telegram (Stage 2)
and the Web dashboard (Stage 3, mounted as a second router on this same
app) both call these routes, no exceptions.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from amop.api.deps import configure_session_factory
from amop.api.routes import memory, pull_requests, reports, repositories, tasks
from amop.database.session import init_db, make_engine, make_session_factory
from amop.orchestrator.state_machine import IllegalTransitionError

logger = logging.getLogger("amop.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = make_engine()
    await init_db(engine)
    configure_session_factory(make_session_factory(engine))
    yield
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(title="AMOP API", version="1", lifespan=lifespan)

    app.include_router(tasks.router)
    app.include_router(repositories.router)
    app.include_router(pull_requests.router)
    app.include_router(memory.router)
    app.include_router(reports.router)

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        # api_error() (amop/api/errors.py) already builds `exc.detail`
        # as the exact envelope -- passed through as-is here. Anything
        # that raised a plain HTTPException (FastAPI's own built-ins, or
        # a route that didn't go through api_error) gets wrapped into
        # the same shape instead of leaking FastAPI's default
        # {"detail": "..."} form, which Section 15.2's spec doesn't use.
        if isinstance(exc.detail, dict) and "error_code" in exc.detail:
            body = exc.detail
        else:
            body = {"error_code": "HTTP_ERROR", "message": str(exc.detail), "detail": {}}
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(IllegalTransitionError)
    async def _illegal_transition_handler(
        request: Request, exc: IllegalTransitionError
    ) -> JSONResponse:
        # Belt-and-braces: routes/tasks.py already catches this and
        # raises the envelope directly, but a future route that forgets
        # to still gets the right shape rather than a raw 500.
        return JSONResponse(
            status_code=409,
            content={"error_code": "ILLEGAL_TRANSITION", "message": str(exc), "detail": {}},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error_code": "VALIDATION_ERROR",
                "message": "request validation failed",
                "detail": {"errors": exc.errors()},
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Logged here explicitly: registering a custom Exception handler
        # replaces Starlette's own default logging, and a 500 with no
        # server-side trace at all would be worse than the bug itself.
        logger.exception("unhandled error in %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "INTERNAL_ERROR",
                "message": "an unexpected error occurred",
                "detail": {},
            },
        )

    return app


app = create_app()
