"""The three-stage verification pipeline and the accepted-task record it produces.

Accept only if V1 *and* V2 *and* V3 all pass. Everything else is a rejection, tagged
with the stage that rejected it and a machine-readable reason code -- that tagging is
what turns the funnel figure and the LLM repair loop into real instruments rather than
decoration.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from taskforge.dsl import Action, TaskSpec, get_world
from taskforge.verify.v1_schema import StageResult, check_v1
from taskforge.verify.v2_oracle import (
    DEFAULT_NODE_BUDGET,
    OracleResult,
    OracleStatus,
    solve,
)
from taskforge.verify.v3_replay import check_v3


@dataclass
class Certificate:
    """The proof object. Everything downstream -- shaping, difficulty, grading, replay --
    is derived from this, which is the whole argument of the project: the search that
    certifies the task is worth more than the task."""

    plan: list[list[Any]]
    cost: int
    nodes_expanded: int
    branching: float
    states_seen: int

    def as_actions(self) -> list[Action]:
        return [(n, a) for n, a in self.plan]


@dataclass
class Difficulty:
    score: float
    bucket: int
    plan_length: int
    scatter: float
    branching: float


@dataclass
class VerificationOutcome:
    accepted: bool
    task_id: str
    stages: dict[str, StageResult] = field(default_factory=dict)
    certificate: Certificate | None = None
    difficulty: Difficulty | None = None
    reject_stage: str | None = None
    reject_reason: str | None = None
    oracle_status: str | None = None
    disagreement: bool = False

    def reason_code(self) -> str:
        if self.accepted:
            return "accepted"
        return (self.reject_reason or "unknown").split(":", 1)[0]


# --------------------------------------------------------------------------------------
# Difficulty labelling
# --------------------------------------------------------------------------------------

# Bucket edges on the composite score. Chosen once from the quantiles of the generated
# corpus (see results/difficulty_calibration.json) and then frozen, so the label is a
# property of the task rather than of whatever batch it happened to be generated in.
BUCKET_EDGES = (18.0, 40.0, 72.0, 120.0)


def sku_scatter(spec: TaskSpec) -> float:
    """Mean packing-station distance to the SKUs an order actually needs, normalised by
    grid size. High scatter means the robot must range across the floor."""
    from taskforge.worlds.warehouse.sim import context_for

    ctx = context_for(spec)
    needed = [
        s
        for s in range(ctx.n_skus)
        if any(ctx.need_of(o, s) > 0 for o in range(ctx.n_orders))
    ]
    if not needed:
        return 0.0
    dists = []
    for s in needed:
        cells = ctx.access.get(s)
        if not cells:
            continue
        dists.append(min(int(ctx.dist[ctx.pack_idx][c]) for c in cells))
    if not dists:
        return 0.0
    return sum(dists) / len(dists) / (ctx.width + ctx.height)


def difficulty_of(spec: TaskSpec, oracle: OracleResult) -> Difficulty:
    """difficulty = plan length x SKU scatter x branching factor.

    Each factor captures something the others miss: plan length is raw work, scatter is
    how far that work is spread, branching is how many wrong turns are available at each
    step. The composite is calibrated against measured agent success rate, not asserted.
    """
    L = oracle.cost or 0
    scatter = sku_scatter(spec)
    b = max(1.0, oracle.branching)
    score = L * (0.5 + scatter) * math.log2(1.0 + b)
    return Difficulty(
        score=round(score, 3),
        bucket=bucket_for(score),
        plan_length=L,
        scatter=round(scatter, 4),
        branching=round(b, 3),
    )


def bucket_for(score: float) -> int:
    for i, edge in enumerate(BUCKET_EDGES):
        if score < edge:
            return i + 1
    return len(BUCKET_EDGES) + 1


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------


def verify(
    spec: TaskSpec,
    node_budget: int = DEFAULT_NODE_BUDGET,
) -> VerificationOutcome:
    out = VerificationOutcome(accepted=False, task_id=spec.task_id)

    v1 = check_v1(spec)
    out.stages["V1"] = v1
    if not v1.ok:
        out.reject_stage = "V1"
        out.reject_reason = v1.reasons[0]
        return out

    oracle = solve(spec, node_budget=node_budget)
    out.oracle_status = oracle.status.value
    v2_ok = oracle.status is OracleStatus.SOLVED
    out.stages["V2"] = StageResult(
        ok=v2_ok,
        stage="V2",
        reasons=[] if v2_ok else [f"{oracle.status.value}: oracle did not certify a plan"],
        detail={
            "cost": oracle.cost,
            "expanded": oracle.nodes_expanded,
            "branching": round(oracle.branching, 3),
        },
    )
    if not v2_ok:
        out.reject_stage = "V2"
        out.reject_reason = f"{oracle.status.value}: oracle did not certify a plan"
        return out

    v3 = check_v3(spec, oracle.plan, oracle.cost)
    out.stages["V3"] = v3
    if not v3.ok:
        # V2 said solvable and V3 could not reproduce it: the two disagree about the
        # world. That is a bug in this repo, not a property of the task.
        out.reject_stage = "V3"
        out.reject_reason = v3.reasons[0]
        out.disagreement = True
        return out

    out.accepted = True
    out.certificate = Certificate(
        plan=[[n, a] for n, a in oracle.plan],
        cost=oracle.cost or 0,
        nodes_expanded=oracle.nodes_expanded,
        branching=round(oracle.branching, 3),
        states_seen=oracle.states_seen,
    )
    out.difficulty = difficulty_of(spec, oracle)
    return out


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


@dataclass
class VerifiedTask:
    spec: TaskSpec
    certificate: Certificate
    difficulty: Difficulty

    def to_json(self) -> dict:
        return {
            "spec": json.loads(self.spec.canonical_json()),
            "certificate": asdict(self.certificate),
            "difficulty": asdict(self.difficulty),
            "content_hash": self.spec.content_hash(),
        }

    @staticmethod
    def from_json(data: dict) -> VerifiedTask:
        return VerifiedTask(
            spec=TaskSpec.model_validate(data["spec"]),
            certificate=Certificate(**data["certificate"]),
            difficulty=Difficulty(**data["difficulty"]),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n")

    @staticmethod
    def load(path: Path) -> VerifiedTask:
        return VerifiedTask.from_json(json.loads(Path(path).read_text()))


def load_specs(directory: str | Path, buckets: tuple[int, ...] | None = None) -> list[VerifiedTask]:
    """Load committed pre-verified tasks, optionally filtered by difficulty bucket."""
    tasks = [VerifiedTask.load(p) for p in sorted(Path(directory).glob("*.json"))]
    if buckets is not None:
        tasks = [t for t in tasks if t.difficulty.bucket in buckets]
    return tasks


def accepted_task(spec: TaskSpec, outcome: VerificationOutcome) -> VerifiedTask:
    assert outcome.accepted and outcome.certificate and outcome.difficulty
    return VerifiedTask(spec=spec, certificate=outcome.certificate, difficulty=outcome.difficulty)


def revalidate(task: VerifiedTask) -> bool:
    """Re-run V3 on a loaded task. Committed specs are only trustworthy if the plan they
    ship still replays against the current executor -- this is what CI checks."""
    world = get_world(task.spec.world)
    state = world.initial_state(task.spec)
    for action in task.certificate.as_actions():
        nxt = world.apply(task.spec, state, action)  # type: ignore[attr-defined]
        if nxt is None:
            return False
        state = nxt
    return world.is_goal(task.spec, state)
