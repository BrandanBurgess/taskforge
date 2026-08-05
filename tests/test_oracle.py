"""Oracle correctness: optimality, soundness, determinism.

The single most valuable test here is ``test_astar_matches_bfs_optimum``. An
inadmissible heuristic does not crash and does not fail any type check -- it just
quietly returns plans that are a few steps too long, which corrupts every downstream
number (difficulty labels, agent efficiency ratios, the shaping potential). Uninformed
BFS is slow but incontestably optimal, so cross-checking against it on small tasks is
the only thing that actually pins optimality down.
"""

from __future__ import annotations

from collections import deque

import pytest

from taskforge.dsl import TaskSpec, get_world
from taskforge.verify import replay, solve, verify
from taskforge.verify.v2_oracle import OracleStatus, cost_to_go
from taskforge.worlds.warehouse import context_for, generate, heuristic, initial_state


def bfs_optimal(spec: TaskSpec, limit: int = 60) -> int | None:
    """Uninformed breadth-first search. No heuristic, therefore no way for a heuristic
    bug to hide in it."""
    world = get_world(spec.world)
    start = world.initial_state(spec)
    if world.is_goal(spec, start):
        return 0
    seen = {start}
    q = deque([(start, 0)])
    while q:
        s, g = q.popleft()
        if g >= limit:
            continue
        for _a, ns, _c in world.successors(spec, s):
            if ns in seen:
                continue
            if world.is_goal(spec, ns):
                return g + 1
            seen.add(ns)
            q.append((ns, g + 1))
    return None


@pytest.mark.parametrize("difficulty", [1, 2])
@pytest.mark.parametrize("seed", range(6))
def test_astar_matches_bfs_optimum(difficulty: int, seed: int) -> None:
    spec = generate(seed, difficulty)
    astar = solve(spec)
    brute = bfs_optimal(spec, limit=45)
    if astar.status is OracleStatus.SOLVED:
        assert brute is not None, "A* found a plan that BFS could not reach"
        assert astar.cost == brute, (
            f"A* returned cost {astar.cost} but the true optimum is {brute}; "
            "the heuristic is inadmissible"
        )
    else:
        assert brute is None, f"A* rejected a task BFS solved in {brute} steps"


@pytest.mark.parametrize("difficulty", [1, 2, 3])
@pytest.mark.parametrize("seed", range(6))
def test_astar_matches_exact_vstar(difficulty: int, seed: int) -> None:
    """A*'s reported cost must equal the exact optimum from the backward Bellman pass.

    This catches a failure mode that admissibility checks alone cannot see. A heuristic
    can be perfectly admissible at every single state and *still* cost A* its optimality
    if it is not consistent, because a closed-set A* without reopening will settle a
    state at a suboptimal g and never revisit it. The regression this test locks down:
    the oracle returned 42 on a task whose true optimum is 40, while
    ``test_heuristic_never_overestimates`` reported zero violations.

    It also reaches much deeper than the BFS cross-check can -- V* comes from a full
    enumeration, so plans of length 40+ are checkable in milliseconds.
    """
    spec = generate(seed, difficulty)
    ctg = cost_to_go(spec, budget=400_000)
    if not ctg.exact:
        pytest.skip("state space too large to enumerate exactly")
    ctx = context_for(spec)
    true_opt = ctg.value(initial_state(ctx))
    astar = solve(spec)
    if true_opt is None:
        assert astar.status is not OracleStatus.SOLVED
    else:
        assert astar.status is OracleStatus.SOLVED
        assert astar.cost == true_opt, (
            f"A* returned {astar.cost}, exact V* says {true_opt}"
        )


@pytest.mark.parametrize("difficulty", [1, 2, 3])
@pytest.mark.parametrize("seed", range(4))
def test_heuristic_never_overestimates(difficulty: int, seed: int) -> None:
    """Admissibility, checked directly: h(s) <= true cost-to-go for every reachable
    state with a known exact value."""
    spec = generate(seed, difficulty)
    ctx = context_for(spec)
    ctg = cost_to_go(spec, budget=40_000)
    if not ctg.exact:
        pytest.skip("state space too large to enumerate exactly")
    for state, true_cost in ctg.table.items():
        h = heuristic(ctx, state)
        assert h <= true_cost, f"h={h} > V*={true_cost} at {state}: heuristic is inadmissible"


@pytest.mark.parametrize("difficulty", [1, 2, 3])
def test_heuristic_zero_at_goal(difficulty: int) -> None:
    spec = generate(0, difficulty)
    ctx = context_for(spec)
    ctg = cost_to_go(spec, budget=40_000)
    if not ctg.exact:
        pytest.skip("state space too large")
    goals = [s for s, v in ctg.table.items() if v == 0]
    assert goals
    for g in goals:
        assert heuristic(ctx, g) == 0


@pytest.mark.parametrize("difficulty", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", range(3))
def test_accepted_plans_replay(difficulty: int, seed: int) -> None:
    """V2/V3 agreement. Any failure here is a P0, not a rejected task."""
    spec = generate(seed, difficulty)
    outcome = verify(spec)
    if not outcome.accepted:
        pytest.skip(f"rejected at {outcome.reject_stage}: {outcome.reject_reason}")
    trace = replay(spec, outcome.certificate.as_actions())
    assert trace.goal_reached, trace.failure
    assert trace.length == outcome.certificate.cost


@pytest.mark.parametrize("seed", range(10))
def test_soundness_fuzz_no_rejected_task_is_solvable(seed: int) -> None:
    """The other half of soundness: a task the oracle *rejected* must not be solvable by
    brute force within a small budget. Conservative rejection is allowed to throw away
    good tasks (budget exhaustion), so this only asserts on outright UNSOLVABLE
    verdicts, which are the ones that claim to be proofs."""
    spec = generate(1000 + seed, 2)
    result = solve(spec)
    if result.status is OracleStatus.UNSOLVABLE:
        assert bfs_optimal(spec, limit=40) is None, (
            "oracle declared a task unsolvable that BFS solved"
        )


def test_determinism_same_seed_same_everything() -> None:
    a = generate(7, 3)
    b = generate(7, 3)
    assert a.canonical_json() == b.canonical_json()
    assert a.content_hash() == b.content_hash()
    pa, pb = solve(a), solve(b)
    assert pa.status == pb.status
    assert pa.plan == pb.plan
    assert pa.nodes_expanded == pb.nodes_expanded


def test_initial_state_is_not_goal() -> None:
    spec = generate(0, 2)
    ctx = context_for(spec)
    world = get_world("warehouse")
    assert not world.is_goal(spec, initial_state(ctx))
