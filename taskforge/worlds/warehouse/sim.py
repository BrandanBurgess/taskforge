"""Warehouse executor: canonical symbolic state, action semantics, admissible heuristic.

The state is a plain hashable tuple::

    (pos_idx, held, filled, dispatched, unlocked, ruined, battery)

``held`` is a per-SKU count tuple; ``filled`` is a flattened (order x sku) count tuple;
``dispatched`` / ``unlocked`` / ``ruined`` are bitmasks; ``battery`` is exact (``-1``
when the battery constraint is disabled).

Two things deserve comment because they are what keep the oracle exact *and* tractable:

1. **Shelf stock is derived, not stored.** With exactly one shelf cell per SKU,
   ``stock[s] = available[s] - held[s] - packed[s] - keys_spent[s]``. Nothing about the
   live state is lost by omitting it, and the state space shrinks by a large factor.

2. **Battery is hashed exactly.** Bucketing it would be cheaper but *unsound* -- two
   states with different charge are genuinely different states, and collapsing them can
   make the oracle certify a plan the executor cannot run. Soundness wins; if a task is
   too big we reject it rather than approximate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from taskforge.dsl import TaskSpec, register_world
from taskforge.worlds.warehouse.spec import (
    CONVEYORS,
    DIRECTIONS,
    WAREHOUSE_PREDICATES,
    WarehousePayload,
)

UNREACHABLE = 1 << 20

State = tuple[int, tuple[int, ...], tuple[int, ...], int, int, int, int]
Action = tuple[str, Any]


# --------------------------------------------------------------------------------------
# Compiled, spec-derived context (built once, reused by oracle / env / renderer)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WarehouseContext:
    payload: WarehousePayload
    n_skus: int
    n_orders: int
    capacity: int
    battery_max: int  # -1 when disabled
    start_idx: int
    pack_idx: int
    dock_idx: frozenset[int]
    shelf_idx: dict[int, int]  # sku -> shelf cell index
    access: dict[int, tuple[int, ...]]  # sku -> passable cells orthogonally adjacent
    zone_of: tuple[int, ...]  # per cell index
    zone_ids: tuple[int, ...]
    zone_key: dict[int, int]  # zone id -> keycard sku
    zone_adjacent: dict[int, frozenset[int]]  # zone -> cells on or next to that zone
    need: tuple[int, ...]  # flattened (order x sku)
    stock: tuple[int, ...]
    passable: tuple[bool, ...]
    conveyor: tuple[int, ...]  # per cell: -1 or direction index into DIR_ORDER
    dist: Any  # np.ndarray [ncells, ncells] relaxed shortest-path distances
    all_dispatched: int
    tsp_nodes: tuple[tuple[int, int], ...] = ()  # (sku, access cell) pairs
    tsp: dict[int, tuple[int, ...]] = field(default_factory=dict)  # sku-mask -> per-node cost
    sku_pack_dist: tuple[int, ...] = ()  # per SKU: cheapest access-cell -> packing station

    @property
    def width(self) -> int:
        return self.payload.width

    @property
    def height(self) -> int:
        return self.payload.height

    def xy(self, idx: int) -> tuple[int, int]:
        return idx % self.width, idx // self.width

    def idx(self, x: int, y: int) -> int:
        return y * self.width + x

    def need_of(self, order: int, sku: int) -> int:
        return self.need[order * self.n_skus + sku]


DIR_ORDER = ("N", "S", "E", "W")


def _build_context(spec: TaskSpec) -> WarehouseContext:
    p = WarehousePayload.model_validate(spec.payload)
    w, h = p.width, p.height
    ncells = w * h
    passable = tuple(p.is_passable(i % w, i // w) for i in range(ncells))

    conveyor = []
    for i in range(ncells):
        c = p.tile(i % w, i // w)
        conveyor.append(DIR_ORDER.index(_dir_name(CONVEYORS[c])) if c in CONVEYORS else -1)

    shelf_pos = p.shelf_pos()
    shelf_idx = {s: (pos[1] * w + pos[0]) for s, pos in shelf_pos.items()}
    access: dict[int, tuple[int, ...]] = {}
    for s, (sx, sy) in shelf_pos.items():
        cells = []
        for dx, dy in DIRECTIONS.values():
            nx, ny = sx + dx, sy + dy
            if p.is_passable(nx, ny):
                cells.append(ny * w + nx)
        access[s] = tuple(sorted(cells))

    zone_of = tuple(p.zone(i % w, i // w) for i in range(ncells))
    zone_cells = p.zone_cells()
    zone_ids = tuple(sorted(zone_cells))
    zone_adjacent: dict[int, frozenset[int]] = {}
    for z, cells in zone_cells.items():
        near: set[int] = set()
        for cx, cy in cells:
            for dx, dy in DIRECTIONS.values():
                nx, ny = cx + dx, cy + dy
                if p.is_passable(nx, ny):
                    near.add(ny * w + nx)
            if p.is_passable(cx, cy):
                near.add(cy * w + cx)
        zone_adjacent[z] = frozenset(near)

    px, py = p.pack_cell()
    pack_idx = py * w + px
    need = tuple(
        p.orders[o].need[s] for o in range(len(p.orders)) for s in range(len(p.skus))
    )
    dist = _relaxed_all_pairs(p)
    tsp_nodes, tsp = _held_karp(len(p.skus), access, dist, pack_idx)

    return WarehouseContext(
        payload=p,
        n_skus=len(p.skus),
        n_orders=len(p.orders),
        capacity=spec.constraints.capacity,
        battery_max=p.battery_max if p.battery_max is not None else -1,
        start_idx=p.start[1] * w + p.start[0],
        pack_idx=pack_idx,
        dock_idx=frozenset(dy * w + dx for dx, dy in p.dock_cells()),
        shelf_idx=shelf_idx,
        access=access,
        zone_of=zone_of,
        zone_ids=zone_ids,
        zone_key={int(k): v for k, v in p.zone_keys.items()},
        zone_adjacent=zone_adjacent,
        need=need,
        stock=tuple(p.stock),
        passable=passable,
        conveyor=tuple(conveyor),
        dist=dist,
        all_dispatched=(1 << len(p.orders)) - 1,
        tsp_nodes=tsp_nodes,
        tsp=tsp,
        sku_pack_dist=tuple(
            min((int(dist[c][pack_idx]) for c in access.get(s, ())), default=UNREACHABLE)
            for s in range(len(p.skus))
        ),
    )


def _dir_name(delta: tuple[int, int]) -> str:
    for name, d in DIRECTIONS.items():
        if d == delta:
            return name
    raise ValueError(delta)


def _relaxed_all_pairs(p: WarehousePayload) -> np.ndarray:
    """All-pairs shortest path on the *relaxed* grid: conveyor one-way restrictions and
    zone locks are both ignored. Relaxing can only shorten paths, so distances taken
    from this table are guaranteed lower bounds -- exactly what an admissible heuristic
    needs."""
    w, h = p.width, p.height
    n = w * h
    dist = np.full((n, n), UNREACHABLE, dtype=np.int32)
    nbrs: list[list[int]] = [[] for _ in range(n)]
    for y in range(h):
        for x in range(w):
            if not p.is_passable(x, y):
                continue
            i = y * w + x
            for dx, dy in DIRECTIONS.values():
                nx, ny = x + dx, y + dy
                if p.is_passable(nx, ny):
                    nbrs[i].append(ny * w + nx)
    for src in range(n):
        if not p.is_passable(src % w, src // w):
            continue
        d = dist[src]
        d[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            du = d[u] + 1
            for v in nbrs[u]:
                if d[v] > du:
                    d[v] = du
                    q.append(v)
    return dist


# Keyed by id(spec); the spec itself is stored alongside the context so it stays alive
# and its id cannot be recycled onto a different object. This is on the hottest path in
# the codebase -- it is called once per successor expansion and once per heuristic
# evaluation -- so it must not hash or serialize the spec.
def _held_karp(
    n_skus: int,
    access: dict[int, tuple[int, ...]],
    dist: np.ndarray,
    pack_idx: int,
) -> tuple[tuple[tuple[int, int], ...], dict[int, tuple[int, ...]]]:
    """Exact shortest walk that visits one access cell of every SKU in a set and ends at
    the packing station.

    Why this is the right bound: the robot must stand next to SKU *s*'s shelf to pick it,
    and the very last action of any successful plan is a ``scan`` at the packing station,
    so every plan's trajectory is a walk from the current cell through one access cell
    per outstanding SKU, terminating at the pack cell. The length of the *shortest* such
    walk is therefore a lower bound on the moves remaining -- and with <= 6 SKUs, Held-Karp
    computes it exactly rather than approximating it.

    Precomputed once per spec over all 2^n SKU subsets, so the heuristic itself stays a
    cheap table lookup on the hot path.

    Returns ``(nodes, table)`` where ``nodes[j] = (sku, cell)`` and
    ``table[mask][j]`` is the cost of a walk starting at ``nodes[j]``'s cell, covering
    every SKU in ``mask``, and finishing at the packing station.
    """
    nodes: list[tuple[int, int]] = []
    for s in range(n_skus):
        for cell in access.get(s, ()):
            nodes.append((s, cell))
    n = len(nodes)
    table: dict[int, tuple[int, ...]] = {}
    if n == 0:
        return (), table

    big = UNREACHABLE
    # base case: a single outstanding SKU -- stand at its access cell, walk to packing
    for j, (s, cell) in enumerate(nodes):
        mask = 1 << s
        row = list(table.get(mask, (big,) * n))
        d = int(dist[cell][pack_idx])
        if d < row[j]:
            row[j] = d
        table[mask] = tuple(row)

    for mask in range(1, 1 << n_skus):
        if mask.bit_count() < 2:
            continue
        row = [big] * n
        for j, (s, cell) in enumerate(nodes):
            if not (mask >> s & 1):
                continue
            rest = mask & ~(1 << s)
            prev = table.get(rest)
            if prev is None:
                continue
            best = big
            for k, (s2, cell2) in enumerate(nodes):
                if not (rest >> s2 & 1):
                    continue
                cand = int(dist[cell][cell2]) + prev[k]
                if cand < best:
                    best = cand
            row[j] = min(best, big)
        table[mask] = tuple(row)
    table[0] = (0,) * n
    return tuple(nodes), table


_CTX_CACHE: dict[int, tuple[TaskSpec, WarehouseContext]] = {}
_CTX_CACHE_MAX = 64


def context_for(spec: TaskSpec) -> WarehouseContext:
    entry = _CTX_CACHE.get(id(spec))
    if entry is not None and entry[0] is spec:
        return entry[1]
    ctx = _build_context(spec)
    if len(_CTX_CACHE) >= _CTX_CACHE_MAX:
        _CTX_CACHE.pop(next(iter(_CTX_CACHE)))
    _CTX_CACHE[id(spec)] = (spec, ctx)
    return ctx


# --------------------------------------------------------------------------------------
# State helpers
# --------------------------------------------------------------------------------------


def initial_state(ctx: WarehouseContext) -> State:
    return (
        ctx.start_idx,
        (0,) * ctx.n_skus,
        (0,) * (ctx.n_orders * ctx.n_skus),
        0,
        0,
        0,
        ctx.battery_max,
    )


def keys_spent(ctx: WarehouseContext, unlocked: int) -> tuple[int, ...]:
    spent = [0] * ctx.n_skus
    for bit, z in enumerate(ctx.zone_ids):
        if unlocked >> bit & 1:
            spent[ctx.zone_key[z]] += 1
    return tuple(spent)


def stock_of(ctx: WarehouseContext, state: State) -> tuple[int, ...]:
    """Derived shelf stock. See module docstring for why this is not part of the hash."""
    _, held, filled, _, unlocked, _, _ = state
    spent = keys_spent(ctx, unlocked)
    out = []
    for s in range(ctx.n_skus):
        packed = sum(filled[o * ctx.n_skus + s] for o in range(ctx.n_orders))
        out.append(ctx.stock[s] - held[s] - packed - spent[s])
    return tuple(out)


def order_complete(ctx: WarehouseContext, filled: tuple[int, ...], order: int) -> bool:
    base = order * ctx.n_skus
    return all(filled[base + s] == ctx.need[base + s] for s in range(ctx.n_skus))


def is_goal(ctx: WarehouseContext, state: State) -> bool:
    _, _, _, dispatched, _, ruined, _ = state
    return ruined == 0 and dispatched == ctx.all_dispatched


def is_dead(ctx: WarehouseContext, state: State) -> bool:
    """A ruined box can never be dispatched, and the goal requires every order
    dispatched -- so a nonzero ruined mask is a proof of unsolvability from here.
    That is what makes irreversibility cheap to reason about: dead states get no
    successors at all."""
    return state[5] != 0


# --------------------------------------------------------------------------------------
# Action semantics
# --------------------------------------------------------------------------------------


def legal_actions(ctx: WarehouseContext, state: State) -> list[Action]:
    return [a for a, _, _ in successors(ctx, state)]


def apply_action(ctx: WarehouseContext, state: State, action: Action) -> State | None:
    """Apply one action. Returns ``None`` if the action is illegal in this state.

    This is *the* executor. The oracle's ``successors`` is built from it, so V2 and V3
    cannot drift apart by construction -- V3 still re-checks, because "cannot drift by
    construction" is a claim that deserves a test rather than a comment.
    """
    pos, held, filled, dispatched, unlocked, ruined, battery = state
    if ruined:
        return None
    name, arg = action
    bat_on = ctx.battery_max >= 0

    if name != "charge" and bat_on and battery <= 0:
        return None
    nb = (battery - 1) if (bat_on and name != "charge") else battery

    if name == "move":
        if arg not in DIRECTIONS:
            return None
        dx, dy = DIRECTIONS[arg]
        x, y = ctx.xy(pos)
        nx, ny = x + dx, y + dy
        if not (0 <= nx < ctx.width and 0 <= ny < ctx.height):
            return None
        nidx = ny * ctx.width + nx
        if not ctx.passable[nidx]:
            return None
        # one-way conveyor: standing on one, you may only leave along its arrow
        cv = ctx.conveyor[pos]
        if cv >= 0 and DIR_ORDER[cv] != arg:
            return None
        z = ctx.zone_of[nidx]
        if z and not _zone_unlocked(ctx, unlocked, z):
            return None
        return (nidx, held, filled, dispatched, unlocked, ruined, nb)

    if name == "pick":
        s = arg
        if not (0 <= s < ctx.n_skus):
            return None
        if pos not in ctx.access.get(s, ()):
            return None
        if sum(held) >= ctx.capacity:
            return None
        if stock_of(ctx, state)[s] <= 0:
            return None
        nh = list(held)
        nh[s] += 1
        return (pos, tuple(nh), filled, dispatched, unlocked, ruined, nb)

    if name == "place":
        s = arg
        if not (0 <= s < ctx.n_skus):
            return None
        if pos not in ctx.access.get(s, ()):
            return None
        if held[s] <= 0:
            return None
        nh = list(held)
        nh[s] -= 1
        return (pos, tuple(nh), filled, dispatched, unlocked, ruined, nb)

    if name == "pack":
        o = arg
        if not (0 <= o < ctx.n_orders):
            return None
        if pos != ctx.pack_idx:
            return None
        if dispatched >> o & 1:
            return None
        if sum(held) == 0:
            return None  # packing an empty box is a pure no-op; disallow
        base = o * ctx.n_skus
        nf = list(filled)
        new_ruined = ruined
        for s in range(ctx.n_skus):
            if held[s] == 0:
                continue
            total = nf[base + s] + held[s]
            if total > ctx.need[base + s]:
                new_ruined |= 1 << o  # wrong or surplus SKU in the box: irreversible
                total = ctx.need[base + s]
            nf[base + s] = total
        return (pos, (0,) * ctx.n_skus, tuple(nf), dispatched, unlocked, new_ruined, nb)

    if name == "scan":
        o = arg
        if not (0 <= o < ctx.n_orders):
            return None
        if pos != ctx.pack_idx:
            return None
        if dispatched >> o & 1:
            return None
        if not order_complete(ctx, filled, o):
            return None
        return (pos, held, filled, dispatched | (1 << o), unlocked, ruined, nb)

    if name == "charge":
        if not bat_on or pos not in ctx.dock_idx:
            return None
        if battery == ctx.battery_max:
            return None  # no-op
        return (pos, held, filled, dispatched, unlocked, ruined, ctx.battery_max)

    if name == "unlock":
        z = arg
        if z not in ctx.zone_key:
            return None
        bit = ctx.zone_ids.index(z)
        if unlocked >> bit & 1:
            return None
        key = ctx.zone_key[z]
        if held[key] <= 0:
            return None
        if pos not in ctx.zone_adjacent[z]:
            return None
        nh = list(held)
        nh[key] -= 1
        return (pos, tuple(nh), filled, dispatched, unlocked | (1 << bit), ruined, nb)

    return None


def _zone_unlocked(ctx: WarehouseContext, unlocked: int, zone: int) -> bool:
    if zone not in ctx.zone_key:
        return True
    return bool(unlocked >> ctx.zone_ids.index(zone) & 1)


_ACTION_CACHE: dict[int, tuple[WarehouseContext, tuple[Action, ...]]] = {}


def enumerate_actions(ctx: WarehouseContext) -> tuple[Action, ...]:
    """Every syntactically well-formed action for this spec, in a fixed order.

    The order is fixed because the oracle's determinism guarantee depends on successors
    being generated identically on every run.
    """
    hit = _ACTION_CACHE.get(id(ctx))
    if hit is not None and hit[0] is ctx:
        return hit[1]
    acts: list[Action] = [("move", d) for d in DIR_ORDER]
    acts += [("pick", s) for s in range(ctx.n_skus)]
    acts += [("place", s) for s in range(ctx.n_skus)]
    acts += [("pack", o) for o in range(ctx.n_orders)]
    acts += [("scan", o) for o in range(ctx.n_orders)]
    acts += [("charge", None)]
    acts += [("unlock", z) for z in ctx.zone_ids]
    out = tuple(acts)
    if len(_ACTION_CACHE) > 64:
        _ACTION_CACHE.pop(next(iter(_ACTION_CACHE)))
    _ACTION_CACHE[id(ctx)] = (ctx, out)
    return out


def successors(ctx: WarehouseContext, state: State) -> list[tuple[Action, State, int]]:
    if is_dead(ctx, state) or is_goal(ctx, state):
        return []
    out = []
    for a in enumerate_actions(ctx):
        ns = apply_action(ctx, state, a)
        if ns is not None and ns != state:
            out.append((a, ns, 1))
    return out


# --------------------------------------------------------------------------------------
# Admissible heuristic
# --------------------------------------------------------------------------------------


def heuristic(ctx: WarehouseContext, state: State) -> int:
    """Lower bound on remaining actions.

    Four action-count terms (picks, packs, scans, unlocks) that are pairwise disjoint,
    plus a movement lower bound computed on the relaxed grid. Because the terms count
    different action types they add without over-counting, and each is individually a
    valid bound -- so the sum is admissible.
    """
    pos, held, filled, dispatched, unlocked, ruined, _ = state
    if ruined:
        return UNREACHABLE
    if dispatched == ctx.all_dispatched:
        return 0

    S, n_orders = ctx.n_skus, ctx.n_orders
    remaining = [0] * S
    order_remaining = [0] * n_orders
    per_order = [[0] * S for _ in range(n_orders)]
    for o in range(n_orders):
        base = o * S
        for s in range(S):
            r = ctx.need[base + s] - filled[base + s]
            if r > 0:
                remaining[s] += r
                order_remaining[o] += r
                per_order[o][s] = r

    # Units that still have to be picked off a shelf (held stock covers the rest).
    to_pick = [max(0, remaining[s] - held[s]) for s in range(S)]
    picks = sum(to_pick)
    # Each order with outstanding need requires at least one more pack action.
    packs = sum(1 for o in range(n_orders) if order_remaining[o] > 0)
    # Each undispatched order requires exactly one scan.
    scans = sum(1 for o in range(n_orders) if not (dispatched >> o & 1))

    # Each still-locked zone that solely gates a still-needed SKU costs >= 1 unlock.
    unlocks = 0
    for bit, z in enumerate(ctx.zone_ids):
        if unlocked >> bit & 1:
            continue
        gated = any(
            to_pick[s] > 0
            and ctx.access.get(s)
            and all(ctx.zone_of[c] == z for c in ctx.access[s])
            for s in range(S)
        )
        if gated:
            unlocks += 1

    d = ctx.dist
    if picks > 0:
        # Shortest walk from here through one access cell of every outstanding SKU,
        # ending at the packing station. Exact, via the precomputed Held-Karp table.
        mask = 0
        for s in range(S):
            if to_pick[s] > 0:
                mask |= 1 << s
        row = ctx.tsp.get(mask)
        if row is None:
            return UNREACHABLE
        best = UNREACHABLE
        for j, (sku, cell) in enumerate(ctx.tsp_nodes):
            if not (mask >> sku & 1):
                continue
            cand = int(d[pos][cell]) + row[j]
            if cand < best:
                best = cand
        move_lb = best

        # Second, *independent* movement bound. `pack` deposits the entire held multiset
        # into a single order, so items destined for different orders can never ride
        # together: each order with outstanding need forces its own trip out from the
        # packing station and back. Those trips are disjoint in time, so their lengths
        # add -- one leg from the robot's current cell through a shelf to packing, then
        # a 2*(cheapest shelf->pack) round trip for every *other* outstanding order.
        # We drop the largest leg rather than guessing which order is served first.
        #
        # This is combined with the tour bound by `max`, never by addition: both bound
        # the same single walk, so summing them would double-count shared moves and make
        # the heuristic inadmissible -- which silently costs optimality, and is exactly
        # what tests/test_oracle.py::test_astar_matches_bfs_optimum exists to catch.
        legs = []
        for o in range(n_orders):
            if order_remaining[o] <= 0:
                continue
            cheapest = UNREACHABLE
            for s in range(S):
                if per_order[o][s] > held[s]:
                    cheapest = min(cheapest, ctx.sku_pack_dist[s])
            if cheapest < UNREACHABLE:
                legs.append(cheapest)
        if len(legs) > 1:
            first_leg = UNREACHABLE
            for s in range(S):
                if to_pick[s] <= 0:
                    continue
                for a in ctx.access.get(s, ()):
                    first_leg = min(first_leg, int(d[pos][a]) + ctx.sku_pack_dist[s])
            legs.sort()
            if first_leg < UNREACHABLE:
                move_lb = max(move_lb, first_leg + 2 * sum(legs[:-1]))
    else:
        move_lb = int(d[pos][ctx.pack_idx])
    if move_lb >= UNREACHABLE:
        return UNREACHABLE

    return picks + packs + scans + unlocks + move_lb


# --------------------------------------------------------------------------------------
# WorldPack implementation
# --------------------------------------------------------------------------------------


class WarehouseWorld:
    name = "warehouse"

    def validate_payload(self, spec: TaskSpec) -> WarehousePayload:
        return WarehousePayload.model_validate(spec.payload)

    def known_predicates(self) -> set[str]:
        return set(WAREHOUSE_PREDICATES)

    def initial_state(self, spec: TaskSpec) -> State:
        return initial_state(context_for(spec))

    def successors(self, spec: TaskSpec, state: State) -> list[tuple[Action, State, int]]:
        return successors(context_for(spec), state)

    def is_goal(self, spec: TaskSpec, state: State) -> bool:
        return is_goal(context_for(spec), state)

    def is_dead(self, spec: TaskSpec, state: State) -> bool:
        return is_dead(context_for(spec), state)

    def heuristic(self, spec: TaskSpec, state: State) -> int:
        return heuristic(context_for(spec), state)

    def apply(self, spec: TaskSpec, state: State, action: Action) -> State | None:
        return apply_action(context_for(spec), state, action)


WAREHOUSE = register_world(WarehouseWorld())
