"""Procedural warehouse generator: seeded, offline, parameterized by difficulty.

Layouts follow the obvious real-world shape -- racking rows separated by aisles, with
clear vertical cross-corridors -- which keeps them connected by construction and makes
the rendered grids read as warehouses rather than as random noise.

The generator is deliberately *optimistic*: it does not try to prove the tasks it emits
are solvable. That is the verifier's job, and letting the generator propose freely is
what makes the verification funnel a meaningful measurement instead of a formality.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from taskforge.dsl import Constraints, TaskSpec
from taskforge.worlds.warehouse.spec import (
    AISLE,
    DOCK,
    PACK,
    WALL,
    Order,
    WarehousePayload,
    warehouse_goal,
    warehouse_tools,
)

SKU_NAMES = [
    "widget-A",
    "gasket-B",
    "coupler-C",
    "bearing-D",
    "sensor-E",
    "relay-F",
]


@dataclass
class GenParams:
    """Every knob the brief calls out, in one place."""

    difficulty: int
    width: int
    height: int
    n_item_skus: int
    n_orders: int
    capacity: int
    units_per_order: tuple[int, int]
    n_conveyors: int
    n_zones: int
    battery: bool
    decoy_stock: int
    scatter: float = 1.0  # 0 = SKUs clustered near packing, 1 = pushed to the far side
    step_budget: int = 220
    extras: dict = field(default_factory=dict)


def params_for_difficulty(d: int) -> GenParams:
    """The default curriculum. Sizes are chosen so the oracle's full cost-to-go table
    stays computable at the low end and A* stays inside budget at the high end."""
    # A note on why difficulty 5 has no battery. Battery is by far the most expensive
    # knob for exact search: it multiplies the canonical state space by the charge range
    # and it cannot be bucketed without making the oracle unsound. Measured on this
    # machine, a 13x11 / 3-order world with battery certifies 4 of 6 seeds in 15s, while
    # the same world without it certifies 6 of 6 in 1.5s. Rather than raise the node
    # budget -- which would trade a picky verifier for a slow one -- battery lives at
    # difficulty 4, where the world is small enough to absorb it, and difficulty 5 buys
    # its hardness from more orders, tighter routing and two gated zones instead.
    table = {
        1: GenParams(1, 7, 7, 2, 1, 3, (2, 2), 0, 0, False, 1, 0.25, 120),
        2: GenParams(2, 9, 9, 3, 1, 3, (2, 3), 1, 0, False, 1, 0.5, 160),
        3: GenParams(3, 11, 9, 3, 2, 2, (1, 2), 2, 0, False, 1, 0.7, 200),
        4: GenParams(4, 13, 11, 4, 2, 2, (1, 2), 3, 1, True, 1, 0.85, 260),
        5: GenParams(5, 13, 11, 4, 3, 3, (1, 2), 4, 2, False, 1, 1.0, 320),
    }
    if d not in table:
        raise ValueError(f"difficulty must be 1..5, got {d}")
    return table[d]


# --------------------------------------------------------------------------------------


def _blank_grid(w: int, h: int) -> list[list[str]]:
    g = [[AISLE for _ in range(w)] for _ in range(h)]
    for x in range(w):
        g[0][x] = WALL
        g[h - 1][x] = WALL
    for y in range(h):
        g[y][0] = WALL
        g[y][w - 1] = WALL
    return g


def _rack_cells(g: list[list[str]], rng: random.Random) -> list[tuple[int, int]]:
    """Lay racking on alternate interior rows, leaving cross-corridors at both ends and
    a random gap mid-row so the aisles feel like a floor plan."""
    h, w = len(g), len(g[0])
    cells: list[tuple[int, int]] = []
    for y in range(2, h - 2, 2):
        gap = rng.randrange(2, w - 2) if w > 6 else -1
        for x in range(2, w - 2):
            if x == gap:
                continue
            g[y][x] = WALL
            cells.append((x, y))
    return cells


def _passable(g: list[list[str]], x: int, y: int) -> bool:
    return 0 <= y < len(g) and 0 <= x < len(g[0]) and g[y][x] not in (WALL,) and not g[y][x].isdigit()


def _has_free_neighbour(g: list[list[str]], x: int, y: int) -> bool:
    return any(_passable(g, x + dx, y + dy) for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)))


def generate(seed: int, difficulty: int, params: GenParams | None = None) -> TaskSpec:
    """Produce one candidate spec. May well be unsolvable -- that is expected."""
    p = params or params_for_difficulty(difficulty)
    rng = random.Random(seed)
    w, h = p.width, p.height
    g = _blank_grid(w, h)
    racks = _rack_cells(g, rng)

    # --- packing station: bottom-centre aisle, the natural "outbound" spot ------------
    pack_y = h - 2
    pack_x = w // 2
    if g[pack_y][pack_x] == WALL:
        pack_x = max(1, pack_x - 1)
    g[pack_y][pack_x] = PACK
    pack = (pack_x, pack_y)

    # --- charge docks -----------------------------------------------------------------
    docks: list[tuple[int, int]] = []
    if p.battery:
        for cand in ((1, 1), (w - 2, h - 2), (1, h - 2)):
            cx, cy = cand
            if g[cy][cx] == AISLE and cand != pack:
                g[cy][cx] = DOCK
                docks.append(cand)
            if len(docks) >= 2:
                break

    # --- locked zones: a rectangular block in the top-left, gated by a keycard ---------
    zones = [[0] * w for _ in range(h)]
    zone_keys: dict[str, int] = {}
    n_item = p.n_item_skus
    n_keys = p.n_zones
    total_skus = n_item + n_keys
    if total_skus > 6:
        n_keys = 6 - n_item
        p.n_zones = n_keys
    zone_regions: list[list[tuple[int, int]]] = []
    for z in range(1, p.n_zones + 1):
        zw = max(2, w // 4)
        zh = max(2, h // 4)
        x0 = 1 if z == 1 else w - 1 - zw
        y0 = 1
        region = []
        for y in range(y0, min(y0 + zh, h - 1)):
            for x in range(x0, min(x0 + zw, w - 1)):
                if (x, y) == pack or (x, y) in docks:
                    continue
                zones[y][x] = z
                region.append((x, y))
        if region:
            zone_regions.append(region)
        else:
            p.n_zones = z - 1
            break

    # --- SKU shelf placement ----------------------------------------------------------
    # Scatter pushes item SKUs toward racks far from the packing station.
    def rack_score(c: tuple[int, int]) -> float:
        d = abs(c[0] - pack[0]) + abs(c[1] - pack[1])
        return d * p.scatter + rng.random() * 2.0

    open_racks = [c for c in racks if zones[c[1]][c[0]] == 0 and _has_free_neighbour(g, *c)]
    zoned_racks = [c for c in racks if zones[c[1]][c[0]] != 0 and _has_free_neighbour(g, *c)]
    open_racks.sort(key=rack_score, reverse=True)
    rng.shuffle(zoned_racks)

    shelf_for: dict[int, tuple[int, int]] = {}
    sku_zone: dict[int, int] = {}
    # Put one item SKU behind each locked zone when zones exist; that is what makes the
    # keycard a genuine ordering constraint rather than decoration.
    item_ids = list(range(n_item))
    gated_items = item_ids[: min(len(zoned_racks), p.n_zones)] if p.n_zones else []
    for i, s in enumerate(gated_items):
        c = zoned_racks[i]
        shelf_for[s] = c
        sku_zone[s] = zones[c[1]][c[0]]
    for s in item_ids:
        if s in shelf_for:
            continue
        if not open_racks:
            return _degenerate(seed, difficulty, p)
        shelf_for[s] = open_racks.pop(0)
    # keycard SKUs go on open racks, reachable without any unlock
    for k in range(n_keys):
        s = n_item + k
        if not open_racks:
            return _degenerate(seed, difficulty, p)
        shelf_for[s] = open_racks.pop(len(open_racks) // 2)
        zone_keys[str(k + 1)] = s

    for s, (sx, sy) in shelf_for.items():
        g[sy][sx] = str(s)

    # --- one-way conveyors along a vertical cross-corridor ----------------------------
    corridor_x = w - 2
    conveyor_cells: list[tuple[int, int]] = []
    ys = [y for y in range(2, h - 2) if g[y][corridor_x] == AISLE and zones[y][corridor_x] == 0]
    rng.shuffle(ys)
    arrow = "v" if rng.random() < 0.5 else "^"
    for y in ys[: p.n_conveyors]:
        g[y][corridor_x] = arrow
        conveyor_cells.append((corridor_x, y))

    # A conveyor whose arrow points into a wall is an instant dead end; V1 rejects
    # those, so avoid manufacturing them here.
    for cx, cy in list(conveyor_cells):
        dy = 1 if arrow == "v" else -1
        if not _passable(g, cx, cy + dy):
            g[cy][cx] = AISLE
            conveyor_cells.remove((cx, cy))

    # --- start position ----------------------------------------------------------------
    starts = [
        (x, y)
        for y in range(1, h - 1)
        for x in range(1, w - 1)
        if g[y][x] == AISLE and zones[y][x] == 0
    ]
    if not starts:
        return _degenerate(seed, difficulty, p)
    starts.sort(key=lambda c: abs(c[0] - pack[0]) + abs(c[1] - pack[1]))
    start = starts[min(len(starts) - 1, rng.randrange(0, max(1, len(starts) // 2) + 1))]

    # --- orders -------------------------------------------------------------------------
    n_sku_total = n_item + n_keys
    orders: list[Order] = []
    for o in range(p.n_orders):
        units = rng.randint(*p.units_per_order)
        need = [0] * n_sku_total
        pool = item_ids[:]
        rng.shuffle(pool)
        for u in range(units):
            need[pool[u % len(pool)]] += 1
        # cap any single SKU at the capacity limit so one pack trip is always feasible
        need = [min(n, p.capacity) for n in need]
        if sum(need) == 0:
            need[pool[0]] = 1
        orders.append(Order(order_id=o, need=need, destination=f"dock-{chr(65 + o)}"))

    required = [sum(o.need[s] for o in orders) for s in range(n_sku_total)]
    stock = list(required)
    # Decoy stock is what makes an irreversible mistake *reachable*: surplus units the
    # agent can pick up and wrongly pack. Without it, low difficulties have no way to ruin
    # a box, and the irreversibility study would have nothing to measure.
    for _ in range(p.decoy_stock):
        s = rng.randrange(n_item)
        stock[s] += 1
    # Every item SKU that has a shelf gets at least one unit. A shelf stocking zero units
    # is dead scenery -- it clutters the grid and can never be interacted with. Stocking
    # it instead turns each unordered SKU into a live hazard: the robot *can* pick it up,
    # and packing it ruins the box.
    for s in range(n_item):
        if stock[s] == 0:
            stock[s] = 1
    for k in range(n_keys):
        stock[n_item + k] = 1

    payload = WarehousePayload(
        width=w,
        height=h,
        tiles=["".join(row) for row in g],
        zones=["".join(str(z) for z in row) for row in zones],
        skus=SKU_NAMES[:n_item] + [f"keycard-{k + 1}" for k in range(n_keys)],
        stock=stock,
        start=start,
        orders=orders,
        zone_keys=zone_keys,
        battery_max=_battery_budget(p, w, h) if p.battery else None,
    )

    return TaskSpec(
        task_id=f"wh-d{difficulty}-s{seed:05d}",
        world="warehouse",
        seed=seed,
        tools=warehouse_tools(),
        goal=warehouse_goal(),
        constraints=Constraints(
            step_budget=p.step_budget,
            capacity=p.capacity,
            battery=payload.battery_max,
            irreversible=True,
        ),
        payload=payload.model_dump(mode="json"),
        metadata={
            "difficulty_target": difficulty,
            "generator": "procedural",
            "n_item_skus": n_item,
            "n_keycards": n_keys,
            "n_conveyors": len(conveyor_cells),
            "n_zones": p.n_zones,
            "scatter": p.scatter,
        },
    )


def _battery_budget(p: GenParams, w: int, h: int) -> int:
    """Generous enough that charging is a routing consideration, tight enough that
    ignoring it can strand the robot."""
    return int((w + h) * 2.2)


def _degenerate(seed: int, difficulty: int, p: GenParams) -> TaskSpec:
    """Emit a deliberately malformed-but-typed spec when placement fails.

    Returning something the verifier can reject (rather than raising) keeps failed
    placements visible in the funnel instead of silently disappearing from the
    denominator.
    """
    w, h = p.width, p.height
    g = _blank_grid(w, h)
    g[h - 2][w // 2] = PACK
    payload = WarehousePayload(
        width=w,
        height=h,
        tiles=["".join(r) for r in g],
        zones=["0" * w for _ in range(h)],
        skus=SKU_NAMES[:1],
        stock=[0],
        start=(1, 1),
        orders=[Order(order_id=0, need=[1])],
        zone_keys={},
        battery_max=None,
    )
    return TaskSpec(
        task_id=f"wh-d{difficulty}-s{seed:05d}",
        world="warehouse",
        seed=seed,
        tools=warehouse_tools(),
        goal=warehouse_goal(),
        constraints=Constraints(step_budget=p.step_budget, capacity=p.capacity),
        payload=payload.model_dump(mode="json"),
        metadata={"difficulty_target": difficulty, "generator": "procedural", "degenerate": True},
    )
