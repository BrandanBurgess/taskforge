"""V1 -- schema, typing, and well-formedness.

Cheap structural checks that run before any search. V1 exists to stop the oracle from
burning its node budget on specs that were never coherent in the first place, and to
give a generator (especially an LLM one) a precise, machine-readable reason to repair.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from taskforge.dsl import TaskSpec, get_world


@dataclass
class StageResult:
    ok: bool
    stage: str
    reasons: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def fail_code(self) -> str | None:
        return self.reasons[0].split(":", 1)[0] if self.reasons else None


def check_v1(spec: TaskSpec) -> StageResult:
    reasons: list[str] = []
    detail: dict = {}

    try:
        world = get_world(spec.world)
    except KeyError as e:
        return StageResult(False, "V1", [f"unknown_world: {e}"])

    try:
        payload = world.validate_payload(spec)
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(p) for p in first["loc"])
        return StageResult(False, "V1", [f"payload_schema: {loc}: {first['msg']}"])
    except ValueError as e:
        return StageResult(False, "V1", [f"payload_schema: {e}"])

    # --- tool declarations reference only predicates the world can evaluate ------------
    known = world.known_predicates()
    tool_names = set()
    for tool in spec.tools:
        if tool.name in tool_names:
            reasons.append(f"duplicate_tool: {tool.name}")
        tool_names.add(tool.name)
        for pred in list(tool.preconditions) + list(tool.effects):
            if pred.name not in known:
                reasons.append(f"unknown_predicate: {tool.name}.{pred.name}")
    if not tool_names:
        reasons.append("no_tools: spec declares no actions")

    # --- goal must be one this world knows ---------------------------------------------
    if spec.goal.kind != "all_orders_dispatched":
        reasons.append(f"unknown_goal: {spec.goal.kind}")

    # --- warehouse well-formedness -------------------------------------------------------
    try:
        pack = payload.pack_cell()
    except ValueError as e:
        return StageResult(False, "V1", [f"packing_station: {e}"], detail)

    sx, sy = payload.start
    if not payload.is_passable(sx, sy):
        reasons.append(f"start_blocked: start {payload.start} is not a passable cell")
    if payload.zone(sx, sy) != 0:
        reasons.append("start_locked: robot starts inside a locked zone")

    shelves = payload.shelf_pos()
    for s in range(len(payload.skus)):
        if s not in shelves:
            reasons.append(f"missing_shelf: SKU {s} ({payload.skus[s]}) has no shelf cell")
            continue
        shx, shy = shelves[s]
        neighbours = [
            (shx + dx, shy + dy) for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0))
        ]
        if not any(payload.is_passable(nx, ny) for nx, ny in neighbours):
            reasons.append(f"unreachable_shelf: SKU {s} shelf at {(shx, shy)} has no free side")

    # every conveyor must point somewhere passable, or it is an instant trap
    from taskforge.worlds.warehouse.spec import CONVEYORS

    for y in range(payload.height):
        for x in range(payload.width):
            c = payload.tile(x, y)
            if c in CONVEYORS:
                dx, dy = CONVEYORS[c]
                if not payload.is_passable(x + dx, y + dy):
                    reasons.append(f"conveyor_trap: conveyor at {(x, y)} points into a wall")

    # stock must at least cover what the orders ask for
    required = payload.total_required()
    for s, r in enumerate(required):
        if r > payload.stock[s]:
            reasons.append(
                f"insufficient_stock: SKU {s} needs {r} units, shelf holds {payload.stock[s]}"
            )
    if sum(required) == 0:
        reasons.append("empty_goal: no order requires any SKU")

    # keycards must exist for every locked zone, and must never be orderable
    zones_present = set(payload.zone_cells())
    for z in zones_present:
        if str(z) not in payload.zone_keys:
            reasons.append(f"zone_without_key: zone {z} has no keycard SKU")
    for zid, key_sku in payload.zone_keys.items():
        if not 0 <= key_sku < len(payload.skus):
            reasons.append(f"bad_keycard: zone {zid} references SKU {key_sku}")
            continue
        if required[key_sku] > 0:
            reasons.append(f"orderable_keycard: keycard SKU {key_sku} is required by an order")
        kx, ky = shelves.get(key_sku, (-1, -1))
        if kx >= 0 and payload.zone(kx, ky) != 0:
            reasons.append(f"key_behind_own_lock: keycard for zone {zid} sits inside a locked zone")

    if payload.zone(*pack) != 0:
        reasons.append("pack_locked: packing station is inside a locked zone")

    # NOTE: there is deliberately no capacity check here. `pack` may be called more than
    # once per order, so an order needing more units than the robot can carry is still
    # solvable via multiple trips. Rejecting on capacity would be unsound in the
    # direction that matters least (throwing away good tasks) but would also make the
    # funnel lie about *why* things fail. V2 decides.

    detail["n_shelves"] = len(shelves)
    detail["required_units"] = sum(required)
    return StageResult(not reasons, "V1", reasons, detail)
