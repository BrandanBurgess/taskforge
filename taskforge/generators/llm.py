"""LLM task generator: Claude emits DSL JSON via forced tool use, never free-form code.

Two properties make this safe to point at a verifier:

1. **The model emits data, not code.** Forced tool use with a JSON schema means the
   worst a bad generation can do is fail validation. There is no path from model output
   to executed code.
2. **The repair loop is fed structured failures.** When a candidate is rejected, the
   exact stage and reason code go back to the model verbatim -- ``V1 missing_shelf: SKU 2
   has no shelf cell`` is actionable in a way that "invalid task" is not.

Every attempt is logged with the stage that rejected it, so accept rate, repair-attempt
histogram and the failure taxonomy all fall out of the run.

This module shares :class:`GenerationReport` with the procedural generator's funnel, so
both arms are measured the same way. It is unit-tested against a mocked client; see
``tests/test_llm_generator.py``.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

from taskforge.dsl import Constraints, TaskSpec
from taskforge.verify import VerificationOutcome, verify
from taskforge.worlds.warehouse.spec import (
    MAX_ORDERS,
    MAX_SKUS,
    WarehousePayload,
    warehouse_goal,
    warehouse_tools,
)

DEFAULT_MODEL = "claude-opus-4-5"

PAYLOAD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "width": {"type": "integer", "minimum": 6, "maximum": 14},
        "height": {"type": "integer", "minimum": 6, "maximum": 14},
        "tiles": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "One string per row, each exactly `width` characters. "
                "'#'=wall or rack, '.'=aisle, 'P'=packing station (exactly one), "
                "'C'=charge dock, '^v<>'=one-way conveyor, "
                "digit=shelf holding that SKU index."
            ),
        },
        "zones": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Same shape as tiles. '0' = open, '1'..'9' = locked zone id.",
        },
        "skus": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_SKUS},
        "stock": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        "start": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
            "description": "[x, y] of the robot's starting cell. Must be an open aisle.",
        },
        "orders": {
            "type": "array",
            "maxItems": MAX_ORDERS,
            "items": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "integer", "minimum": 0},
                    "need": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "description": "One entry per SKU: units required.",
                    },
                    "destination": {"type": "string"},
                },
                "required": ["order_id", "need"],
            },
        },
        "zone_keys": {
            "type": "object",
            "additionalProperties": {"type": "integer"},
            "description": "Map zone id (as a string) to the SKU index of its keycard.",
        },
        "battery_max": {"type": ["integer", "null"]},
        "capacity": {"type": "integer", "minimum": 1, "maximum": 6},
        "step_budget": {"type": "integer", "minimum": 10, "maximum": 600},
        "rationale": {"type": "string", "description": "One sentence on the intended challenge."},
    },
    "required": [
        "width", "height", "tiles", "zones", "skus", "stock", "start", "orders",
        "capacity", "step_budget",
    ],
}

SYSTEM = """You design grid-warehouse fulfilment tasks for a robot planner.

The robot can move N/S/E/W, pick(sku) from a shelf it is standing next to, place(sku)
back, pack(order_id), scan(order_id) to dispatch, charge at a dock, and unlock(zone).

Rules you must design around:
- `pack(order_id)` deposits the robot's ENTIRE held multiset into that order. Any item
  the order does not still need RUINS the box permanently. Good tasks make this a real
  hazard without making it unavoidable.
- Exactly one packing station 'P'.
- Each SKU index that appears in `tiles` must have exactly one shelf cell, and that cell
  must have at least one walkable neighbour.
- `stock` must cover the total units all orders require.
- Every locked zone needs a keycard SKU, placed OUTSIDE any locked zone, and no order may
  require a keycard.
- Conveyors are one-way: standing on one, the robot may only leave along its arrow. An
  arrow pointing into a wall is a trap and will be rejected.

Emit a task that is challenging but solvable. Prefer interesting routing over large grids.
"""

REPAIR = """That task was REJECTED by the verifier.

Stage: {stage}
Reason: {reason}

Fix exactly that problem and emit a corrected task. Keep everything else the same."""


class LLMClient(Protocol):
    """The slice of the Anthropic client this module uses. Narrow on purpose -- it is
    what makes the mocked unit test meaningful rather than a test of a stub."""

    def generate(self, messages: list[dict], system: str, schema: dict) -> dict: ...


class AnthropicClient:
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 4096):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, messages: list[dict], system: str, schema: dict) -> dict:
        tool = {
            "name": "emit_task",
            "description": "Emit one warehouse task as structured data.",
            "input_schema": schema,
        }
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": "emit_task"},
            messages=messages,
        )
        for block in resp.content:
            if block.type == "tool_use":
                return dict(block.input)
        raise RuntimeError("model returned no tool_use block")


@dataclass
class Attempt:
    index: int
    repair_round: int
    accepted: bool
    stage: str | None
    reason: str | None
    task_id: str | None = None


@dataclass
class GenerationReport:
    attempts: list[Attempt] = field(default_factory=list)
    accepted_specs: list[TaskSpec] = field(default_factory=list)

    @property
    def accept_rate(self) -> float:
        n = len({a.index for a in self.attempts})
        return len(self.accepted_specs) / max(1, n)

    def to_json(self) -> dict:
        rounds = Counter(a.repair_round for a in self.attempts if a.accepted)
        taxonomy: Counter = Counter()
        for a in self.attempts:
            if not a.accepted and a.reason:
                taxonomy[f"{a.stage}:{a.reason.split(':', 1)[0]}"] += 1
        return {
            "tasks_requested": len({a.index for a in self.attempts}),
            "accepted": len(self.accepted_specs),
            "accept_rate": round(self.accept_rate, 4),
            "total_model_calls": len(self.attempts),
            "repair_round_histogram": {str(k): v for k, v in sorted(rounds.items())},
            "failure_taxonomy": dict(taxonomy.most_common()),
            "attempts": [vars(a) for a in self.attempts],
        }


def payload_to_spec(raw: dict, task_id: str, seed: int) -> TaskSpec:
    """Convert the model's tool input into a TaskSpec. Raises on malformed data, which
    the caller records as a V1 rejection."""
    body = {k: v for k, v in raw.items() if k not in ("capacity", "step_budget", "rationale")}
    body.setdefault("zone_keys", {})
    body.setdefault("battery_max", None)
    if isinstance(body.get("start"), list):
        body["start"] = tuple(body["start"])
    payload = WarehousePayload.model_validate(body)
    return TaskSpec(
        task_id=task_id,
        world="warehouse",
        seed=seed,
        tools=warehouse_tools(),
        goal=warehouse_goal(),
        constraints=Constraints(
            step_budget=int(raw.get("step_budget", 200)),
            capacity=int(raw.get("capacity", 3)),
            battery=payload.battery_max,
            irreversible=True,
        ),
        payload=payload.model_dump(mode="json"),
        metadata={
            "generator": "llm",
            "rationale": raw.get("rationale", ""),
        },
    )


def generate_verified(
    client: LLMClient,
    n_tasks: int = 5,
    max_repairs: int = 3,
    brief: str = "Design a warehouse task of moderate difficulty.",
    node_budget: int = 120_000,
) -> GenerationReport:
    """Generate ``n_tasks``, repairing each rejection up to ``max_repairs`` times."""
    report = GenerationReport()

    for i in range(n_tasks):
        messages: list[dict] = [{"role": "user", "content": f"{brief} (task {i + 1})"}]
        for attempt_round in range(max_repairs + 1):
            try:
                raw = client.generate(messages, SYSTEM, PAYLOAD_SCHEMA)
            except Exception as e:
                report.attempts.append(
                    Attempt(i, attempt_round, False, "client", f"client_error: {e}")
                )
                break

            task_id = f"wh-llm-{i:03d}r{attempt_round}"
            try:
                spec = payload_to_spec(raw, task_id, seed=10_000 + i)
            except Exception as e:
                reason = f"payload_schema: {e}"
                report.attempts.append(Attempt(i, attempt_round, False, "V1", reason))
                messages += [
                    {"role": "assistant", "content": json.dumps(raw)[:4000]},
                    {"role": "user", "content": REPAIR.format(stage="V1", reason=reason)},
                ]
                continue

            outcome: VerificationOutcome = verify(spec, node_budget=node_budget)
            if outcome.accepted:
                report.attempts.append(
                    Attempt(i, attempt_round, True, None, None, task_id)
                )
                report.accepted_specs.append(spec)
                break

            report.attempts.append(
                Attempt(i, attempt_round, False, outcome.reject_stage, outcome.reject_reason)
            )
            messages += [
                {"role": "assistant", "content": json.dumps(raw)[:4000]},
                {
                    "role": "user",
                    "content": REPAIR.format(
                        stage=outcome.reject_stage, reason=outcome.reject_reason
                    ),
                },
            ]
    return report


def api_key_present() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def main() -> None:
    import argparse
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--max-repairs", type=int, default=3)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=str(root / "results" / "llm_generation.json"))
    args = ap.parse_args()

    if not api_key_present():
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. The LLM generation path is implemented and "
            "unit-tested against a mocked client; it cannot be benchmarked without a key."
        )

    report = generate_verified(
        AnthropicClient(args.model), n_tasks=args.tasks, max_repairs=args.max_repairs
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_json()
    payload["model"] = args.model
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"accepted {payload['accepted']}/{payload['tasks_requested']} "
          f"({payload['accept_rate'] * 100:.0f}%) in {payload['total_model_calls']} calls")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
