"""Section 12.7 / ADR-10's emergency shutdown, exposed over the API --
what both the Telegram bot's /pause and /resume and (if wired) the web
dashboard's own control actually call, per Section 3.1's Command Layer
principle: no interface gets a privileged path around this that the
others don't also have.

GET is unauthenticated (read-only, matches every other GET route in
this API); POST requires the shared operator token, same as every
other mutating endpoint.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from amop.api.auth import require_operator_token
from amop.api.deps import get_session
from amop.api.schemas import PauseRequest, PauseStateOut
from amop.orchestrator import pause as pause_mod

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/pause", response_model=PauseStateOut)
async def pause_state_endpoint(
    session: AsyncSession = Depends(get_session),
) -> PauseStateOut:
    row = await pause_mod.get_state(session)
    return PauseStateOut.model_validate(row)


@router.post(
    "/pause",
    response_model=PauseStateOut,
    dependencies=[Depends(require_operator_token)],
)
async def pause_endpoint(
    body: PauseRequest = PauseRequest(),
    session: AsyncSession = Depends(get_session),
) -> PauseStateOut:
    row = await pause_mod.set_paused(session, paused=True, by=body.by or "api")
    return PauseStateOut.model_validate(row)


@router.post(
    "/resume",
    response_model=PauseStateOut,
    dependencies=[Depends(require_operator_token)],
)
async def resume_endpoint(
    body: PauseRequest = PauseRequest(),
    session: AsyncSession = Depends(get_session),
) -> PauseStateOut:
    row = await pause_mod.set_paused(session, paused=False, by=body.by or "api")
    return PauseStateOut.model_validate(row)
