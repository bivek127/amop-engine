"""Agent handoff schemas — spec Section 4.7's `handoff_payload`, the
*only* channel between agents ("agents don't call each other"). The
orchestrator validates an agent's output against its role's schema
BEFORE transitioning state; a validation failure is an agent failure
(Section 13), never a silent pass-through.

Field names follow the spec's schemas (6.2, 6.3.10, 6.4, 6.5), trimmed
to the subset Milestone 4 declares. Deferred spec fields are noted per
model rather than silently dropped.

Ground truth vs. model judgment
-------------------------------
Fields that are mechanically observable are computed by the orchestrator
from the real system -- never parsed out of a model's self-report. A
model can say "the tests pass"; only pytest gets to decide that they do.
This is the project's core rule ("software decides what's allowed")
applied to handoffs, and it's marked field-by-field below.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Section 6.2: "confidence < 0.6 routes to NEEDS_HUMAN_INPUT rather than
# letting a low-confidence guess reach the Coder -- this is the single
# highest-leverage guardrail against the Coder agent confidently fixing
# the wrong thing." Lives here as one named constant so the threshold is
# impossible to drift between the router and its tests.
CONFIDENCE_THRESHOLD = 0.6

_MAX_EXCERPT_CHARS = 200


class Evidence(BaseModel):
    """One citation backing the Investigator's causal claim (6.2)."""

    type: Literal["log", "commit", "code"]
    ref: str  # "path:line or sha or query"
    excerpt: str

    @field_validator("excerpt")
    @classmethod
    def _truncate_excerpt(cls, value: str) -> str:
        # Spec caps excerpts at <=200 chars. Truncate rather than reject:
        # an over-long excerpt is a verbose model, not a failed handoff,
        # and failing the whole report over it would lose real evidence.
        if len(value) <= _MAX_EXCERPT_CHARS:
            return value
        return value[: _MAX_EXCERPT_CHARS - 1] + "…"


class RootCauseReport(BaseModel):
    """Investigator's handoff (6.2). `confidence` is a genuine model
    judgment -- it cannot be computed -- which is exactly why the
    threshold that acts on it is enforced in code (see
    orchestrator/chain.py's route_after_investigation)."""

    task_id: str
    root_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
    affected_files: list[str] = Field(default_factory=list)
    suggested_fix_plan: str
    # Deferred from 6.2's full schema: severity_assessment,
    # requires_reproduction (Tester's regression-authoring behavior that
    # consumes it is out of scope this milestone).


class CodeChangeReport(BaseModel):
    """Coder's handoff (6.3.10).

    Built by the ORCHESTRATOR from git, not emitted by the Coder model.
    Section 6.3.9 is explicit that the Reviewer "is not allowed to trust
    Coder's self-report alone", so files_changed/commit_sha/branch are
    read back out of the repo itself. This also means agents/coder.py
    keeps its Milestone 0/2 free-form final answer unchanged.
    """

    task_id: str
    status: Literal["success", "failed"]
    branch: str
    commit_sha: str | None = None
    files_changed: list[str] = Field(default_factory=list)
    diff_summary: str = ""
    iterations_used: int = 0
    failure_diagnostic: str | None = None
    # Milestone 5 bugfix: True when this coding attempt made zero
    # successful mutating tool calls (no write_file/patch_file). Set by
    # the orchestrator from the tool-call log, never the model. Exists
    # because files_changed/status alone can't distinguish "this attempt
    # produced a fix" from "an earlier attempt's commit is still sitting
    # on the branch and this attempt did nothing" -- changed_files() diffs
    # against the baseline, not against the start of this attempt, so it
    # stays non-empty either way. A caller must check no_op before
    # trusting status=="success" as "this attempt did something".
    no_op: bool = False
    # Deferred from 6.3.10: scope_justification (needs the edit-plan
    # declaration step, 6.3.9, not built), test_result (carried
    # separately as TestReport this milestone).


class TestReport(BaseModel):
    """Tester's handoff (6.4).

    `all_passed` is mechanical -- the orchestrator sets it from the
    parsed pytest result, never from the model's prose. The model's
    contribution is `details` (interpretation), which is advisory.
    """

    task_id: str
    all_passed: bool
    details: str = ""
    # Deferred from 6.4: new_tests_added / regression_confirmed (Tester
    # authoring regression tests is out of scope this milestone).


class Finding(BaseModel):
    """One Reviewer objection (6.5)."""

    severity: Literal["low", "medium", "high", "critical"]
    description: str = Field(
        description=(
            "Must quote the exact offending code verbatim from the diff, "
            "plus what is concretely wrong with it. A restated doubt with "
            "no quoted code (e.g. 'does not address the root cause') is "
            "not a valid finding."
        )
    )
    file: str | None = None
    line: int | None = None


class ReviewVerdict(BaseModel):
    """Reviewer's handoff (6.5). `approved` is a model judgment (judging
    the diff is the agent's actual job), but the orchestrator applies
    mechanical checks on top of it -- see chain.enforce_review_checks().

    `addresses_reported_symptom` exists because of an observed failure,
    not a hypothetical one. Given a bug report about a CSV export, the
    Investigator found an unrelated arithmetic bug, fixed that, watched
    the suite go green, and the task reached RESOLVED -- reporting
    success for work it never did. Every gate agreed, because every gate
    was checking internal consistency (does the diff match the *stated*
    root cause?) and nothing compared the root cause back to what was
    actually reported.

    Section 6.4 anticipated this: the Tester's inputs include the
    original RootCauseReport "to check the fix actually addresses
    reported behavior, not just 'tests pass'".

    It is a required field with no default on purpose. A default would
    let a model that ignores the question sail through on silence, and
    silence is exactly how the original failure looked.
    """

    task_id: str
    approved: bool
    addresses_reported_symptom: bool
    findings: list[Finding] = Field(default_factory=list)
    rejection_reason: str | None = Field(
        default=None,
        description=(
            "Required when approved=false. Must quote the exact offending "
            "line(s) from the diff verbatim, with file/line, and state "
            "concretely what is wrong with that code. A generic restatement "
            "of doubt with no quoted code is not a valid rejection reason."
        ),
    )
    counterexample: str | None = Field(
        default=None,
        description=(
            "A concrete INPUT -> OUTPUT pair showing the diff produces "
            "wrong behavior, e.g. 'urgency=9, impact=8, effort=1 -> "
            "compute_priority_score returns 31.0, but a task with "
            "urgency=2, impact=1, effort=10 (score 8.0) should NOT "
            "outrank it, and with this diff it does'. Optional when you "
            "approve. When you reject a diff where the test suite already "
            "passes and the change is confined to the affected files, "
            "this is what makes the rejection actionable -- without one, "
            "chain.py's mechanical review checks override the rejection "
            "to an approval, because a passing, in-scope diff with no "
            "identified counterexample gives a human nothing concrete to "
            "act on that the tests haven't already settled."
        ),
    )
