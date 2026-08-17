from enum import Enum


class TaskState(str, Enum):
    """The bug_fix task lifecycle, spec Section 4.1. Other task types
    (Section 4.4) use subsets of this — not built yet."""

    CREATED = "CREATED"
    TRIAGING = "TRIAGING"
    INVESTIGATING = "INVESTIGATING"
    PLANNING_FIX = "PLANNING_FIX"
    CODING = "CODING"
    TESTING = "TESTING"
    REVIEWING = "REVIEWING"
    PR_CREATION = "PR_CREATION"
    WAITING_FOR_APPROVAL = "WAITING_FOR_APPROVAL"
    MERGED = "MERGED"
    RESOLVED = "RESOLVED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    NEEDS_HUMAN_INPUT = "NEEDS_HUMAN_INPUT"
    MERGED_INTO_EXISTING = "MERGED_INTO_EXISTING"


# Section 4.1: "Terminal states: RESOLVED, FAILED, MERGED_INTO_EXISTING,
# CANCELLED, NEEDS_HUMAN_INPUT (soft-terminal — resumable on human input)."
TERMINAL_STATES = frozenset(
    {
        TaskState.RESOLVED,
        TaskState.FAILED,
        TaskState.MERGED_INTO_EXISTING,
        TaskState.CANCELLED,
        TaskState.NEEDS_HUMAN_INPUT,
    }
)

# Section 4.2's transition table, verbatim: (from_state, to_state) -> trigger.
_SPEC_TABLE_TRANSITIONS: dict[tuple[TaskState, TaskState], str] = {
    (TaskState.CREATED, TaskState.TRIAGING): "orchestrator picks up task",
    (TaskState.TRIAGING, TaskState.MERGED_INTO_EXISTING): (
        "dedup match against open incident (similarity >= 0.92)"
    ),
    (TaskState.TRIAGING, TaskState.INVESTIGATING): (
        "novel, severity >= configured floor"
    ),
    (TaskState.TRIAGING, TaskState.CANCELLED): (
        "severity below floor and no explicit human request"
    ),
    (TaskState.INVESTIGATING, TaskState.PLANNING_FIX): (
        "Investigator emits RootCauseReport with confidence >= 0.6"
    ),
    (TaskState.INVESTIGATING, TaskState.NEEDS_HUMAN_INPUT): (
        "confidence < 0.6 OR tool-call budget exhausted"
    ),
    (TaskState.PLANNING_FIX, TaskState.CODING): "Coder agent invoked with fix plan",
    (TaskState.CODING, TaskState.TESTING): "Coder emits diff + declares done",
    (TaskState.TESTING, TaskState.CODING): (
        "test failure AND retry_count < max_fix_iterations (default 4)"
    ),
    (TaskState.TESTING, TaskState.FAILED): "test failure AND retries exhausted",
    (TaskState.TESTING, TaskState.REVIEWING): "tests pass",
    (TaskState.REVIEWING, TaskState.CODING): (
        "Reviewer rejects with actionable feedback AND review_cycles < 2"
    ),
    (TaskState.REVIEWING, TaskState.FAILED): (
        "Reviewer rejects and cycles exhausted"
    ),
    (TaskState.REVIEWING, TaskState.PR_CREATION): "Reviewer approves",
    (TaskState.PR_CREATION, TaskState.WAITING_FOR_APPROVAL): "mode is suggestor",
    (TaskState.PR_CREATION, TaskState.MERGED): (
        "mode >= operator and auto_merge:true and branch is whitelisted"
    ),
    (TaskState.WAITING_FOR_APPROVAL, TaskState.MERGED): (
        "human approves via Telegram/CLI/Web"
    ),
    (TaskState.WAITING_FOR_APPROVAL, TaskState.CANCELLED): "human rejects",
    (TaskState.MERGED, TaskState.RESOLVED): (
        "post-merge checks pass (CI green, no immediate revert)"
    ),
}

# Section 4.6: "human-issued CANCELLED transition is legal from any
# non-terminal state." Section 4.2's table only spells out two of those
# (TRIAGING, WAITING_FOR_APPROVAL) explicitly; fill in the rest so the
# lookup table matches the spec as a whole, not just one table read in
# isolation.
_NON_TERMINAL_STATES = [s for s in TaskState if s not in TERMINAL_STATES]
_GENERIC_CANCEL_TRIGGER = (
    "human cancels (Section 4.6: legal from any non-terminal state)"
)
_CANCELLATION_TRANSITIONS: dict[tuple[TaskState, TaskState], str] = {
    (state, TaskState.CANCELLED): _GENERIC_CANCEL_TRIGGER
    for state in _NON_TERMINAL_STATES
    if (state, TaskState.CANCELLED) not in _SPEC_TABLE_TRANSITIONS
}

# Milestone 6 (CLAUDE.md, spec Section 29.1): two system-initiated abort
# edges that postdate Section 4.2's literal table, kept in their own dict
# rather than folded into _SPEC_TABLE_TRANSITIONS so that dict's "verbatim"
# claim about the spec's original table stays true.
_MILESTONE_6_TRANSITIONS: dict[tuple[TaskState, TaskState], str] = {
    (TaskState.CODING, TaskState.NEEDS_HUMAN_INPUT): (
        "diff exceeds coder.max_loc_per_task cap -- needs_decomposition "
        "(Section 29.1)"
    ),
    (TaskState.PR_CREATION, TaskState.NEEDS_HUMAN_INPUT): (
        "create_pull_request blocked or failed (secret detected, diff "
        "too large, or GitHub API/push error) -- Section 12.5.1"
    ),
}

TRANSITIONS: dict[tuple[TaskState, TaskState], str] = {
    **_SPEC_TABLE_TRANSITIONS,
    **_CANCELLATION_TRANSITIONS,
    **_MILESTONE_6_TRANSITIONS,
}


class IllegalTransitionError(Exception):
    """Raised when a (from_state, to_state) pair is not in TRANSITIONS."""

    def __init__(self, from_state: TaskState, to_state: TaskState) -> None:
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"Illegal transition: {from_state.value} -> {to_state.value} "
            "is not a valid state transition for the bug_fix task machine."
        )


def validate_transition(from_state: TaskState, to_state: TaskState) -> str:
    """Return the spec trigger text for a legal (from_state, to_state)
    pair, or raise IllegalTransitionError."""
    key = (from_state, to_state)
    if key not in TRANSITIONS:
        raise IllegalTransitionError(from_state, to_state)
    return TRANSITIONS[key]
