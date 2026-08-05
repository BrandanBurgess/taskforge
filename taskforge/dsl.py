"""Domain-agnostic task DSL.

The verifier core (``taskforge.verify``) never imports a world package. It talks to
worlds only through :class:`WorldPack`, which is a tiny protocol: give me an initial
state, give me successors, tell me whether a state is a goal, and give me an
admissible heuristic. Everything warehouse-specific lives behind that boundary in
``taskforge.worlds.warehouse``.

A :class:`TaskSpec` is therefore an envelope: generic fields the verifier understands
(seed, tools, goal, constraints) plus an opaque ``payload`` that the named world pack
validates into its own typed model.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------------------
# Tool signatures (declarative, world-agnostic)
# --------------------------------------------------------------------------------------

ParamType = Literal["direction", "sku", "order_id", "zone_id", "none"]


class ParamSpec(BaseModel):
    """One argument of a tool."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: ParamType


class Predicate(BaseModel):
    """A named, declarative condition.

    Predicates are documentation-and-checkable rather than an interpreted logic: V1
    asserts that every predicate name a tool references is one the world pack declares
    it knows how to evaluate. The executor implements the semantics. Keeping them
    declared in the spec is what lets V1 catch a generator inventing a tool with a
    precondition nothing implements.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    args: list[str] = Field(default_factory=list)


class ToolSpec(BaseModel):
    """Signature + declared pre/post conditions for one action the agent may take."""

    model_config = ConfigDict(extra="forbid")

    name: str
    params: list[ParamSpec] = Field(default_factory=list)
    preconditions: list[Predicate] = Field(default_factory=list)
    effects: list[Predicate] = Field(default_factory=list)
    cost: int = 1

    @field_validator("cost")
    @classmethod
    def _positive_cost(cls, v: int) -> int:
        if v < 1:
            raise ValueError("tool cost must be >= 1 (zero-cost actions break A* optimality)")
        return v


class GoalSpec(BaseModel):
    """The goal predicate, named so the world pack can dispatch on it."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    args: dict[str, Any] = Field(default_factory=dict)


class Constraints(BaseModel):
    """Global limits. ``step_budget`` bounds any valid plan; the oracle refuses to
    return a plan longer than this even if one exists."""

    model_config = ConfigDict(extra="forbid")

    step_budget: int = Field(ge=1, le=4096)
    capacity: int = Field(ge=1, le=8)
    battery: int | None = Field(default=None, ge=1, le=512)
    irreversible: bool = True


class TaskSpec(BaseModel):
    """A complete, self-describing task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    world: str
    seed: int
    tools: list[ToolSpec]
    goal: GoalSpec
    constraints: Constraints
    payload: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def canonical_json(self) -> str:
        """Stable serialization. Two specs with the same content produce byte-identical
        output, which is what the determinism test asserts on."""
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# The world-pack boundary
# --------------------------------------------------------------------------------------

# A canonical state is any hashable tuple. The verifier treats it as opaque.
CanonState = tuple[Any, ...]
# An action is (tool_name, arg) where arg is None or a small int/str.
Action = tuple[str, Any]


@runtime_checkable
class WorldPack(Protocol):
    """Everything the domain-agnostic verifier needs from a domain.

    Implementations must be *pure*: ``successors`` may not mutate the state it is given,
    and repeated calls with the same state must yield the same successors in the same
    order. The oracle's determinism guarantee rests on that.
    """

    name: str

    def validate_payload(self, spec: TaskSpec) -> Any:
        """Parse ``spec.payload`` into a typed model, raising on malformed input."""

    def known_predicates(self) -> set[str]:
        """Predicate names this world can evaluate. V1 checks tool specs against it."""

    def initial_state(self, spec: TaskSpec) -> CanonState: ...

    def successors(self, spec: TaskSpec, state: CanonState) -> list[tuple[Action, CanonState, int]]:
        """Legal (action, next_state, cost) triples. Must be empty for dead states."""

    def is_goal(self, spec: TaskSpec, state: CanonState) -> bool: ...

    def is_dead(self, spec: TaskSpec, state: CanonState) -> bool:
        """True if the goal is provably unreachable from here (e.g. a ruined box)."""

    def heuristic(self, spec: TaskSpec, state: CanonState) -> int:
        """Admissible (never-overestimating) cost-to-go. Must return 0 at goal states."""


_REGISTRY: dict[str, WorldPack] = {}


def register_world(pack: WorldPack) -> WorldPack:
    _REGISTRY[pack.name] = pack
    return pack


def get_world(name: str) -> WorldPack:
    if name not in _REGISTRY:
        # Import for side effects only on demand, so the core stays domain-agnostic.
        if name == "warehouse":
            import taskforge.worlds.warehouse  # noqa: F401
    if name not in _REGISTRY:
        raise KeyError(f"unknown world pack {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registered_worlds() -> list[str]:
    return sorted(_REGISTRY)
