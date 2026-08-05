"""V2 -- the solvability oracle.

A* over the canonical symbolic state, with an admissible domain heuristic and a hard
node budget. Two guarantees matter and both are load-bearing:

**Optimality.** Costs are uniform (every action costs 1) and the heuristic never
overestimates, so the first goal state popped from the open list is optimal. The
certificate plan is therefore *the* shortest plan, which is what lets "plan length" be a
meaningful difficulty signal and a meaningful denominator for agent efficiency.

**Conservative rejection.** If the node budget is exhausted the task is REJECTED, never
accepted. The oracle answers "solvable" or "not proven solvable" -- it never answers
"probably fine". A slow verifier is worse than a picky one, so when budgets bite we
shrink the world rather than raise the ceiling.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from taskforge.dsl import Action, CanonState, TaskSpec, get_world

DEFAULT_NODE_BUDGET = 200_000
DEFAULT_VSTAR_BUDGET = 250_000


class OracleStatus(StrEnum):
    SOLVED = "solved"
    UNSOLVABLE = "unsolvable"  # search space exhausted with no goal found: a real proof
    BUDGET_EXHAUSTED = "budget_exhausted"  # unknown -> rejected
    OVER_STEP_BUDGET = "over_step_budget"  # solvable, but not within the spec's budget


@dataclass
class OracleResult:
    status: OracleStatus
    plan: list[Action] = field(default_factory=list)
    cost: int | None = None
    nodes_expanded: int = 0
    nodes_generated: int = 0
    frontier_peak: int = 0
    branching: float = 0.0
    states_seen: int = 0

    @property
    def solved(self) -> bool:
        return self.status is OracleStatus.SOLVED

    def to_json(self) -> dict:
        return {
            "status": self.status.value,
            "plan": [[a, arg] for a, arg in self.plan],
            "cost": self.cost,
            "nodes_expanded": self.nodes_expanded,
            "nodes_generated": self.nodes_generated,
            "frontier_peak": self.frontier_peak,
            "branching": round(self.branching, 3),
        }


def solve(
    spec: TaskSpec,
    node_budget: int = DEFAULT_NODE_BUDGET,
    step_budget: int | None = None,
) -> OracleResult:
    """A* to the goal predicate. Returns the optimal plan or a rejection reason."""
    world = get_world(spec.world)
    limit = step_budget if step_budget is not None else spec.constraints.step_budget

    start = world.initial_state(spec)
    if world.is_goal(spec, start):
        return OracleResult(OracleStatus.SOLVED, [], 0, 0, 0, 0, 0.0, 1)

    h0 = world.heuristic(spec, start)
    open_heap: list[tuple[int, int, CanonState]] = [(h0, 0, start)]
    came: dict[CanonState, tuple[CanonState, Action] | None] = {start: None}
    g_of: dict[CanonState, int] = {start: 0}

    expanded = 0
    generated = 0
    peak = 1

    # NOTE: this is A* **with reopening** -- there is deliberately no closed set, and
    # `g_of` alone decides whether a state is worth revisiting.
    #
    # The domain heuristic is admissible but *not consistent*: it is the max of two
    # independent lower bounds (a Held-Karp tour and a per-order round-trip argument),
    # and a max of bounds can drop by more than one across a single edge. Textbook A*
    # with a closed set and no reopening is only optimal for *consistent* heuristics; run
    # it on an inconsistent one and it quietly returns plans a few steps too long. That
    # is precisely what happened here -- the oracle reported 42 on a task whose true
    # optimum is 40, with no test failing and no admissibility violation anywhere.
    # Reopening restores optimality while keeping the stronger heuristic.
    while open_heap:
        f, g, s = heapq.heappop(open_heap)
        if g > g_of.get(s, 1 << 30):
            continue  # stale queue entry, superseded by a cheaper path
        expanded += 1

        if world.is_goal(spec, s):
            plan = _reconstruct(came, s)
            return OracleResult(
                OracleStatus.SOLVED,
                plan,
                g,
                expanded,
                generated,
                peak,
                generated / max(1, expanded),
                len(g_of),
            )

        if expanded > node_budget:
            return OracleResult(
                OracleStatus.BUDGET_EXHAUSTED,
                [],
                None,
                expanded,
                generated,
                peak,
                generated / max(1, expanded),
                len(g_of),
            )

        if g >= limit:
            continue  # any extension exceeds the spec's step budget

        for action, ns, cost in world.successors(spec, s):
            generated += 1
            ng = g + cost
            if ng > limit:
                continue
            if ng >= g_of.get(ns, 1 << 30):
                continue
            hn = world.heuristic(spec, ns)
            if hn >= (1 << 20):
                continue  # provably unreachable under the relaxation
            g_of[ns] = ng
            came[ns] = (s, action)
            heapq.heappush(open_heap, (ng + hn, ng, ns))
            peak = max(peak, len(open_heap))

    # Frontier drained inside the step budget without reaching the goal. This is a real
    # proof of unsolvability *under that budget*, not an "unknown".
    status = OracleStatus.OVER_STEP_BUDGET if limit < spec.constraints.step_budget else (
        OracleStatus.UNSOLVABLE
    )
    return OracleResult(
        status, [], None, expanded, generated, peak, generated / max(1, expanded), len(g_of)
    )


def _reconstruct(
    came: dict[CanonState, tuple[CanonState, Action] | None], goal: CanonState
) -> list[Action]:
    plan: list[Action] = []
    cur: CanonState | None = goal
    while cur is not None:
        edge = came.get(cur)
        if edge is None:
            break
        prev, action = edge
        plan.append(action)
        cur = prev
    plan.reverse()
    return plan


# --------------------------------------------------------------------------------------
# Exact cost-to-go table
# --------------------------------------------------------------------------------------


@dataclass
class CostToGo:
    """Exact V*(s) over the reachable state space, or ``None`` when the space was too
    large to enumerate inside the budget."""

    table: dict[CanonState, int]
    exact: bool
    states: int
    truncated: bool = False

    def value(self, state: CanonState) -> int | None:
        return self.table.get(state)


def cost_to_go(
    spec: TaskSpec, budget: int = DEFAULT_VSTAR_BUDGET, step_budget: int | None = None
) -> CostToGo:
    """Compute exact V* by forward-enumerating the reachable state space, then running a
    backward Bellman pass over that subgraph.

    Uniform action costs mean the backward pass is a plain BFS from the goal set over
    reversed edges -- no priority queue needed.

    If enumeration exceeds ``budget`` we return ``exact=False`` and callers fall back to
    the admissible heuristic as their potential. That fallback is still *sound* for
    reward shaping: potential-based shaping is policy-invariant for **any** potential
    function (Ng, Harada & Russell 1999); using -V* only makes it maximally informative.
    """
    world = get_world(spec.world)
    limit = step_budget if step_budget is not None else spec.constraints.step_budget
    start = world.initial_state(spec)

    preds: dict[CanonState, list[CanonState]] = {}
    goals: list[CanonState] = []
    seen = {start}
    queue = deque([(start, 0)])
    truncated = False

    while queue:
        s, depth = queue.popleft()
        if world.is_goal(spec, s):
            goals.append(s)
            continue
        if depth >= limit:
            continue
        for _a, ns, _c in world.successors(spec, s):
            preds.setdefault(ns, []).append(s)
            if ns not in seen:
                if len(seen) >= budget:
                    truncated = True
                    queue.clear()
                    break
                seen.add(ns)
                queue.append((ns, depth + 1))

    if truncated:
        return CostToGo({}, exact=False, states=len(seen), truncated=True)

    table: dict[CanonState, int] = {}
    bfs: deque[CanonState] = deque()
    for gs in goals:
        table[gs] = 0
        bfs.append(gs)
    while bfs:
        s = bfs.popleft()
        v = table[s] + 1
        for p in preds.get(s, ()):
            if p not in table:
                table[p] = v
                bfs.append(p)
    return CostToGo(table, exact=True, states=len(seen))


def positional_cost_to_go(spec: TaskSpec, ctg: CostToGo, template: CanonState) -> dict[int, int]:
    """Project V* onto grid positions, holding the non-positional part of ``template``
    fixed. This is what the cost-to-go heatmap renders: for every cell the robot could
    stand in, the exact number of actions still required to finish the job."""
    out: dict[int, int] = {}
    for state, v in ctg.table.items():
        if state[1:] == template[1:]:
            pos = state[0]
            if pos not in out or v < out[pos]:
                out[pos] = v
    return out


def plan_signature(plan: list[Action]) -> str:
    return " ".join(f"{n}({'' if a is None else a})" for n, a in plan)


def summarize(result: OracleResult) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "cost": result.cost,
        "expanded": result.nodes_expanded,
        "branching": round(result.branching, 2),
    }
