"""LLM generator, tested against a mocked client.

No API key is required or used. What these tests pin down is the part that is actually
ours: that model output is treated as *data* and never as code, that a malformed payload
becomes a clean V1 rejection instead of an exception, and that the repair loop feeds the
verifier's structured failure back to the model and stops when it succeeds.
"""

from __future__ import annotations

import json

from taskforge.generators.llm import (
    PAYLOAD_SCHEMA,
    GenerationReport,
    generate_verified,
    payload_to_spec,
)
from taskforge.verify import verify
from taskforge.worlds.warehouse import generate
from taskforge.worlds.warehouse.spec import WarehousePayload


def spec_to_tool_input(spec) -> dict:
    """Turn a known-good procedural spec into what the model would have emitted."""
    p = WarehousePayload.model_validate(spec.payload)
    return {
        "width": p.width,
        "height": p.height,
        "tiles": list(p.tiles),
        "zones": list(p.zones),
        "skus": list(p.skus),
        "stock": list(p.stock),
        "start": list(p.start),
        "orders": [o.model_dump() for o in p.orders],
        "zone_keys": dict(p.zone_keys),
        "battery_max": p.battery_max,
        "capacity": spec.constraints.capacity,
        "step_budget": spec.constraints.step_budget,
        "rationale": "mocked",
    }


class ScriptedClient:
    """Returns a queued sequence of tool inputs and records what it was told."""

    def __init__(self, outputs: list[dict]):
        self.outputs = list(outputs)
        self.calls: list[list[dict]] = []

    def generate(self, messages, system, schema):
        self.calls.append([dict(m) for m in messages])
        return self.outputs.pop(0) if self.outputs else self.outputs


def test_schema_is_valid_json_schema_with_required_fields() -> None:
    assert PAYLOAD_SCHEMA["type"] == "object"
    for field in ("tiles", "zones", "skus", "stock", "start", "orders"):
        assert field in PAYLOAD_SCHEMA["properties"]
        assert field in PAYLOAD_SCHEMA["required"]
    json.dumps(PAYLOAD_SCHEMA)  # must be serialisable for the tool definition


def test_good_payload_round_trips_to_an_accepted_spec() -> None:
    good = spec_to_tool_input(generate(0, 2))
    spec = payload_to_spec(good, "wh-llm-test", seed=1)
    assert spec.world == "warehouse"
    assert spec.metadata["generator"] == "llm"
    assert verify(spec).accepted


def test_first_attempt_accepted_uses_exactly_one_call() -> None:
    client = ScriptedClient([spec_to_tool_input(generate(1, 2))])
    report = generate_verified(client, n_tasks=1, max_repairs=3)
    assert len(report.accepted_specs) == 1
    assert len(client.calls) == 1
    out = report.to_json()
    assert out["accept_rate"] == 1.0
    assert out["repair_round_histogram"] == {"0": 1}


def test_malformed_payload_is_a_clean_v1_rejection_then_repaired() -> None:
    """A model that emits a grid with the wrong row width must produce a rejection the
    loop can act on -- not a traceback."""
    broken = spec_to_tool_input(generate(2, 2))
    broken["tiles"] = broken["tiles"][:-1]  # one row short
    good = spec_to_tool_input(generate(2, 2))

    client = ScriptedClient([broken, good])
    report = generate_verified(client, n_tasks=1, max_repairs=2)

    assert len(report.accepted_specs) == 1
    assert len(client.calls) == 2
    first, second = report.attempts
    assert not first.accepted and first.stage == "V1"
    assert second.accepted
    # the repair prompt must actually carry the failure back to the model
    repair_msg = client.calls[1][-1]["content"]
    assert "REJECTED" in repair_msg
    assert "V1" in repair_msg
    assert report.to_json()["repair_round_histogram"] == {"1": 1}


def test_unsolvable_payload_is_rejected_by_the_oracle_not_the_schema() -> None:
    """A structurally valid but unsolvable task must fall through V1 and be caught by V2 --
    that separation is the whole reason there is more than one stage."""
    bad = spec_to_tool_input(generate(3, 2))
    # wall the robot into a corner: valid grid, no route to anything
    w = bad["width"]
    tiles = [list(r) for r in bad["tiles"]]
    bad["start"] = [1, 1]
    for x, y in ((2, 1), (1, 2)):
        if 0 < x < w - 1:
            tiles[y][x] = "#"
    bad["tiles"] = ["".join(r) for r in tiles]

    spec = payload_to_spec(bad, "wh-llm-walled", seed=3)
    outcome = verify(spec, node_budget=20_000)
    assert not outcome.accepted
    assert outcome.reject_stage == "V2"


def test_repairs_give_up_after_the_limit() -> None:
    broken = spec_to_tool_input(generate(4, 2))
    broken["tiles"] = broken["tiles"][:-1]
    client = ScriptedClient([dict(broken) for _ in range(5)])
    report = generate_verified(client, n_tasks=1, max_repairs=2)
    assert report.accepted_specs == []
    assert len(client.calls) == 3  # initial attempt + 2 repairs
    assert report.to_json()["accept_rate"] == 0.0


def test_client_errors_are_recorded_not_raised() -> None:
    class Exploding:
        def generate(self, messages, system, schema):
            raise RuntimeError("connection reset")

    report = generate_verified(Exploding(), n_tasks=2, max_repairs=1)
    assert report.accepted_specs == []
    assert all(a.stage == "client" for a in report.attempts)
    assert len(report.attempts) == 2


def test_report_taxonomy_groups_by_stage_and_reason_code() -> None:
    report = GenerationReport()
    from taskforge.generators.llm import Attempt

    report.attempts = [
        Attempt(0, 0, False, "V1", "missing_shelf: SKU 2 has no shelf cell"),
        Attempt(0, 1, False, "V2", "unsolvable: oracle did not certify a plan"),
        Attempt(1, 0, False, "V1", "missing_shelf: SKU 0 has no shelf cell"),
    ]
    tax = report.to_json()["failure_taxonomy"]
    assert tax["V1:missing_shelf"] == 2
    assert tax["V2:unsolvable"] == 1
