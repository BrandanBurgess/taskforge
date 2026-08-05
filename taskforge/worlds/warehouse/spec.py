"""Typed payload for the warehouse world pack, plus its declared tool signatures."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from taskforge.dsl import GoalSpec, ParamSpec, Predicate, ToolSpec

# Tile alphabet. Digits '0'-'9' are shelf cells; the digit is the SKU index stored there.
WALL = "#"
AISLE = "."
PACK = "P"
DOCK = "C"
CONVEYORS = {"^": (0, -1), "v": (0, 1), "<": (-1, 0), ">": (1, 0)}
SHELF_CHARS = set("0123456789")
PASSABLE_CHARS = set(AISLE + PACK + DOCK) | set(CONVEYORS)
ALL_CHARS = PASSABLE_CHARS | SHELF_CHARS | {WALL}

DIRECTIONS: dict[str, tuple[int, int]] = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}

# Hard caps. These keep the Gym action/observation spaces a fixed size across all
# difficulties, and they bound the oracle's state space.
MAX_SKUS = 6
MAX_ORDERS = 3
MAX_ZONES = 2


class Order(BaseModel):
    """A multiset of SKUs to be assembled into one box and dispatched."""

    model_config = ConfigDict(extra="forbid")

    order_id: int = Field(ge=0, lt=MAX_ORDERS)
    need: list[int]  # need[sku_index] = units required
    destination: str = "dock-A"

    @property
    def total_units(self) -> int:
        return sum(self.need)


class WarehousePayload(BaseModel):
    """The concrete world. Invariants that V1 depends on are enforced here."""

    model_config = ConfigDict(extra="forbid")

    width: int = Field(ge=4, le=16)
    height: int = Field(ge=4, le=16)
    tiles: list[str]
    zones: list[str]  # same shape as tiles; '0' = open, '1'..'9' = locked zone id
    skus: list[str] = Field(min_length=1, max_length=MAX_SKUS)
    stock: list[int]  # stock[sku] = units available on that SKU's shelf
    start: tuple[int, int]
    orders: list[Order] = Field(min_length=1, max_length=MAX_ORDERS)
    zone_keys: dict[str, int] = Field(default_factory=dict)  # zone id (str) -> keycard SKU index
    battery_max: int | None = None

    # ---- derived, cached on first access -------------------------------------------
    @model_validator(mode="after")
    def _check_shape(self) -> WarehousePayload:
        if len(self.tiles) != self.height:
            raise ValueError(f"tiles has {len(self.tiles)} rows, expected {self.height}")
        if len(self.zones) != self.height:
            raise ValueError(f"zones has {len(self.zones)} rows, expected {self.height}")
        for y, row in enumerate(self.tiles):
            if len(row) != self.width:
                raise ValueError(f"tiles row {y} has width {len(row)}, expected {self.width}")
            bad = set(row) - ALL_CHARS
            if bad:
                raise ValueError(f"tiles row {y} has unknown chars {sorted(bad)}")
        for y, row in enumerate(self.zones):
            if len(row) != self.width:
                raise ValueError(f"zones row {y} has width {len(row)}, expected {self.width}")
            if not all(c.isdigit() for c in row):
                raise ValueError(f"zones row {y} must be digits")
        if len(self.stock) != len(self.skus):
            raise ValueError("stock and skus must be the same length")
        for o in self.orders:
            if len(o.need) != len(self.skus):
                raise ValueError(f"order {o.order_id}.need must have one entry per SKU")
        ids = [o.order_id for o in self.orders]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate order_id")
        return self

    # ---- accessors -----------------------------------------------------------------
    def tile(self, x: int, y: int) -> str:
        return self.tiles[y][x]

    def zone(self, x: int, y: int) -> int:
        return int(self.zones[y][x])

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_passable(self, x: int, y: int) -> bool:
        """Passable ignoring locks. Shelves and walls are never enterable."""
        return self.in_bounds(x, y) and self.tile(x, y) in PASSABLE_CHARS

    def shelf_pos(self) -> dict[int, tuple[int, int]]:
        """SKU index -> its single shelf cell.

        Exactly one shelf cell per SKU is a deliberate modelling choice: it makes shelf
        stock a *derived* function of (held, filled, keys spent) instead of part of the
        state hash, which is a large state-space win for the oracle.
        """
        out: dict[int, tuple[int, int]] = {}
        for y in range(self.height):
            for x in range(self.width):
                c = self.tile(x, y)
                if c in SHELF_CHARS:
                    out[int(c)] = (x, y)
        return out

    def cells_of(self, char: str) -> list[tuple[int, int]]:
        return [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.tile(x, y) == char
        ]

    def pack_cell(self) -> tuple[int, int]:
        cells = self.cells_of(PACK)
        if len(cells) != 1:
            raise ValueError(f"expected exactly one packing station, found {len(cells)}")
        return cells[0]

    def dock_cells(self) -> list[tuple[int, int]]:
        return self.cells_of(DOCK)

    def zone_cells(self) -> dict[int, list[tuple[int, int]]]:
        out: dict[int, list[tuple[int, int]]] = {}
        for y in range(self.height):
            for x in range(self.width):
                z = self.zone(x, y)
                if z:
                    out.setdefault(z, []).append((x, y))
        return out

    def key_skus(self) -> set[int]:
        return set(self.zone_keys.values())

    def total_required(self) -> list[int]:
        return [sum(o.need[s] for o in self.orders) for s in range(len(self.skus))]


# --------------------------------------------------------------------------------------
# Declared tool signatures. These are what a generator emits and what V1 type-checks.
# --------------------------------------------------------------------------------------

WAREHOUSE_PREDICATES: set[str] = {
    "target_passable",
    "conveyor_direction_respected",
    "zone_unlocked",
    "has_battery",
    "adjacent_to_shelf",
    "shelf_has_stock",
    "under_capacity",
    "holding_sku",
    "at_pack_station",
    "at_charge_dock",
    "order_open",
    "order_complete",
    "holds_keycard",
    "adjacent_to_zone",
    "position_changed",
    "held_gains_sku",
    "held_loses_sku",
    "order_fill_increases",
    "order_ruined_on_overfill",
    "battery_restored",
    "zone_becomes_unlocked",
    "order_dispatched",
    "held_emptied",
}


def warehouse_tools() -> list[ToolSpec]:
    """The seven actions, with their preconditions and effects declared."""
    return [
        ToolSpec(
            name="move",
            params=[ParamSpec(name="direction", type="direction")],
            preconditions=[
                Predicate(name="target_passable", args=["direction"]),
                Predicate(name="conveyor_direction_respected", args=["direction"]),
                Predicate(name="zone_unlocked", args=["direction"]),
                Predicate(name="has_battery"),
            ],
            effects=[Predicate(name="position_changed", args=["direction"])],
        ),
        ToolSpec(
            name="pick",
            params=[ParamSpec(name="sku", type="sku")],
            preconditions=[
                Predicate(name="adjacent_to_shelf", args=["sku"]),
                Predicate(name="shelf_has_stock", args=["sku"]),
                Predicate(name="under_capacity"),
                Predicate(name="has_battery"),
            ],
            effects=[Predicate(name="held_gains_sku", args=["sku"])],
        ),
        ToolSpec(
            name="place",
            params=[ParamSpec(name="sku", type="sku")],
            preconditions=[
                Predicate(name="adjacent_to_shelf", args=["sku"]),
                Predicate(name="holding_sku", args=["sku"]),
                Predicate(name="has_battery"),
            ],
            effects=[Predicate(name="held_loses_sku", args=["sku"])],
        ),
        ToolSpec(
            name="pack",
            params=[ParamSpec(name="order_id", type="order_id")],
            preconditions=[
                Predicate(name="at_pack_station"),
                Predicate(name="order_open", args=["order_id"]),
                Predicate(name="has_battery"),
            ],
            effects=[
                Predicate(name="order_fill_increases", args=["order_id"]),
                Predicate(name="order_ruined_on_overfill", args=["order_id"]),
                Predicate(name="held_emptied"),
            ],
        ),
        ToolSpec(
            name="scan",
            params=[ParamSpec(name="order_id", type="order_id")],
            preconditions=[
                Predicate(name="at_pack_station"),
                Predicate(name="order_complete", args=["order_id"]),
                Predicate(name="has_battery"),
            ],
            effects=[Predicate(name="order_dispatched", args=["order_id"])],
        ),
        ToolSpec(
            name="charge",
            params=[],
            preconditions=[Predicate(name="at_charge_dock")],
            effects=[Predicate(name="battery_restored")],
        ),
        ToolSpec(
            name="unlock",
            params=[ParamSpec(name="zone_id", type="zone_id")],
            preconditions=[
                Predicate(name="holds_keycard", args=["zone_id"]),
                Predicate(name="adjacent_to_zone", args=["zone_id"]),
                Predicate(name="has_battery"),
            ],
            effects=[
                Predicate(name="zone_becomes_unlocked", args=["zone_id"]),
                Predicate(name="held_loses_sku", args=["keycard"]),
            ],
        ),
    ]


def warehouse_goal() -> GoalSpec:
    return GoalSpec(kind="all_orders_dispatched", args={})


DifficultyLevel = Literal[1, 2, 3, 4, 5]
