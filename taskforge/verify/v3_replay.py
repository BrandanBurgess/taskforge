"""V3 -- execution replay.

Take the certificate plan V2 produced and run it through the executor step by step,
asserting every precondition holds and that the goal predicate is true at the end.

This stage looks redundant -- V2's successor function is *built from* the same executor,
so in principle they cannot disagree. That is exactly why V3 is here. The classic silent
bug in a setup like this is an oracle that searches a slightly different world than the
one the agent acts in: a heuristic that quietly assumes an action is free, a canonical
state that drops a field, a successor generator that forgets a precondition. Every one
of those shows up as a V2/V3 disagreement and as nothing else. Any disagreement is a
P0 bug, not a rejected task, and the pipeline records it separately from ordinary
rejections so it can never be mistaken for one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from taskforge.dsl import Action, CanonState, TaskSpec, get_world
from taskforge.verify.v1_schema import StageResult


@dataclass
class ReplayTrace:
    states: list[CanonState] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    goal_reached: bool = False
    failed_at: int | None = None
    failure: str | None = None

    @property
    def length(self) -> int:
        return len(self.actions)


def replay(spec: TaskSpec, plan: list[Action]) -> ReplayTrace:
    """Execute ``plan`` from the initial state. Never raises; failures are reported."""
    world = get_world(spec.world)
    state = world.initial_state(spec)
    trace = ReplayTrace(states=[state])

    for i, action in enumerate(plan):
        nxt = world.apply(spec, state, action)  # type: ignore[attr-defined]
        if nxt is None:
            trace.failed_at = i
            trace.failure = f"illegal_action: step {i} {action} rejected by executor"
            return trace
        state = nxt
        trace.states.append(state)
        trace.actions.append(action)
        if world.is_dead(spec, state):
            trace.failed_at = i
            trace.failure = f"entered_dead_state: step {i} {action} made the goal unreachable"
            return trace

    trace.goal_reached = world.is_goal(spec, state)
    if not trace.goal_reached:
        trace.failure = "goal_not_reached: plan ran to completion without satisfying the goal"
    return trace


def check_v3(spec: TaskSpec, plan: list[Action], expected_cost: int | None) -> StageResult:
    trace = replay(spec, plan)
    reasons: list[str] = []
    if trace.failure:
        reasons.append(f"replay_failed: {trace.failure}")
    if trace.goal_reached and expected_cost is not None and trace.length != expected_cost:
        reasons.append(
            f"cost_mismatch: oracle reported cost {expected_cost} but replay took {trace.length}"
        )
    if len(plan) > spec.constraints.step_budget:
        reasons.append(
            f"over_budget: plan is {len(plan)} steps, budget is {spec.constraints.step_budget}"
        )
    return StageResult(
        ok=not reasons,
        stage="V3",
        reasons=reasons,
        detail={"replay_length": trace.length, "goal_reached": trace.goal_reached},
    )
