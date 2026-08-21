"""Request/response Pydantic models for the API layer.

Deliberately separate from `amop.agents.handoffs` -- those are agent
handoff *contracts* (validated model output, Section 4.7); these are
HTTP wire shapes. They overlap in places (a MemoryItem row looks a lot
like the memory content dict), but conflating "what shape does the API
return" with "what shape must a model's answer validate against" would
make a future change to one silently risk the other.
"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class TaskCreate(BaseModel):
    """POST /tasks body, per Section 15.2's `{repo_id, task_type,
    description}` -- `repo_id` accepts a registered Repository's id;
    `repo` is also accepted directly as a raw path, since `repo_id` is
    optional registry metadata here (see database/models.py's
    Repository docstring), not the hard identity every other table
    already keys on."""

    task_type: str
    description: str | None = None
    repo_id: uuid.UUID | None = None
    repo: str | None = None


class TaskOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    task_type: str
    state: str
    severity: str | None = None
    task_context: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime | None = None
    resolved_at: datetime | None = None
    version: int


class TaskTransitionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    from_state: str | None = None
    to_state: str
    trigger: str | None = None
    actor: str | None = None
    timestamp: datetime


class DiffOut(BaseModel):
    task_id: uuid.UUID
    diff: str
    available: bool


class RepositoryCreate(BaseModel):
    repo_path: str
    display_name: str | None = None
    # Milestone 20 / spec 14.2. All optional so Milestone 15's existing
    # two-field POST body stays valid unchanged.
    url: str | None = None
    default_branch: str | None = None
    permission_overrides: dict[str, Any] | None = None


class RepositoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    repo_path: str
    display_name: str | None = None
    url: str | None = None
    default_branch: str = "main"
    detected_stack: dict[str, Any] | None = None
    index_status: str = "unindexed"
    permission_overrides: dict[str, Any] | None = None
    created_at: datetime


class RepositoryPermissionsUpdate(BaseModel):
    """PATCH /repositories/{id}/permissions -- spec 15.2's own endpoint.
    Shape per Section 12.1's worked example:
        {"default": "observer", "agents": {"dependency_updater": "autonomous"}}
    """

    permission_overrides: dict[str, Any] | None = None


class PullRequestOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    task_id: uuid.UUID | None = None
    repo_path: str
    url: str
    number: int | None = None
    status: str
    created_at: datetime


class MemoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    repo_path: str
    task_id: uuid.UUID | None = None
    memory_type: str
    content: dict[str, Any]
    disputed: bool
    created_at: datetime


class MemoryDisputeUpdate(BaseModel):
    """PATCH /memory/{id} body -- Section 10.4's dispute toggle."""

    disputed: bool = True
