"""DSL round-tripping and per-action precondition/effect correctness."""

from __future__ import annotations

import json

import pytest

from taskforge.dsl import TaskSpec, get_world, registered_worlds
from taskforge.verify import check_v1
from taskforge.worlds.warehouse import (
    apply_action,
    context_for,
    enumerate_actions,
    generate,
    initial_state,
    is_dead,
    is_goal,
    stock_of,
)
from taskforge.worlds.warehouse.spec import WarehousePayload


def test_world_registry() -> None:
    assert "warehouse" in registered_worlds()
    assert get_world("warehouse").name == "warehouse"


@pytest.mark.parametrize("difficulty", [1, 2, 3, 4, 5])
def test_spec_round_trip_is_byte_identical(difficulty: int) -> None:
    spec = generate(3, difficulty)
    again = TaskSpec.model_validate(json.loads(spec.canonical_json()))
    assert again.canonical_json() == spec.canonical_json()
    assert again.content_hash() == spec.content_hash()


@pytest.mark.parametrize("difficulty", [1, 2, 3, 4, 5])
def test_generated_specs_pass_v1(difficulty: int) -> None:
    for seed in range(5):
        res = check_v1(generate(seed, difficulty))
        assert res.ok, res.reasons


def test_payload_rejects_wrong_shape() -> None:
    spec = generate(0, 1)
    bad = dict(spec.payload)
    bad["tiles"] = bad["tiles"][:-1]
    with pytest.raises(ValueError):
        WarehousePayload.model_validate(bad)


# --------------------------------------------------------------------------------------
# Per-action semantics
# --------------------------------------------------------------------------------------


def test_move_blocked_by_wall_and_shelf() -> None:
    spec = generate(0, 2)
    ctx = context_for(spec)
    for idx in range(ctx.width * ctx.height):
        if ctx.passable[idx]:
            continue
        # no passable neighbour may step into an impassable cell
        for a in (("move", "N"), ("move", "S"), ("move", "E"), ("move", "W")):
            st = (idx, (0,) * ctx.n_skus, (0,) * (ctx.n_orders * ctx.n_skus), 0, 0, 0,
                  ctx.battery_max)
            _ = apply_action(ctx, st, a)  # from an impassable cell we simply don't care
    s = initial_state(ctx)
    for a in (("move", "N"), ("move", "S"), ("move", "E"), ("move", "W")):
        ns = apply_action(ctx, s, a)
        if ns is not None:
            assert ctx.passable[ns[0]]


def test_conveyor_is_one_way() -> None:
    from taskforge.worlds.warehouse.sim import DIR_ORDER

    for seed in range(20):
        spec = generate(seed, 4)
        ctx = context_for(spec)
        conv = [i for i in range(ctx.width * ctx.height) if ctx.conveyor[i] >= 0]
        if not conv:
            continue
        idx = conv[0]
        st = (idx, (0,) * ctx.n_skus, (0,) * (ctx.n_orders * ctx.n_skus), 0, 0, 0, ctx.battery_max)
        allowed = DIR_ORDER[ctx.conveyor[idx]]
        for d in DIR_ORDER:
            ns = apply_action(ctx, st, ("move", d))
            if d != allowed:
                assert ns is None, f"left conveyor {idx} against its arrow via {d}"
        return
    pytest.skip("no conveyor found in sampled specs")


def test_pick_requires_adjacency_stock_and_capacity() -> None:
    spec = generate(0, 2)
    ctx = context_for(spec)
    s0 = initial_state(ctx)
    # not adjacent to any shelf at the start -> no pick is legal
    for s in range(ctx.n_skus):
        if s0[0] not in ctx.access.get(s, ()):
            assert apply_action(ctx, s0, ("pick", s)) is None

    sku = 0
    cell = ctx.access[sku][0]
    st = (cell, (0,) * ctx.n_skus, (0,) * (ctx.n_orders * ctx.n_skus), 0, 0, 0, ctx.battery_max)
    ns = apply_action(ctx, st, ("pick", sku))
    assert ns is not None and ns[1][sku] == 1
    assert stock_of(ctx, ns)[sku] == ctx.stock[sku] - 1

    full = (cell, tuple(ctx.capacity if i == sku else 0 for i in range(ctx.n_skus)),
            st[2], 0, 0, 0, ctx.battery_max)
    assert apply_action(ctx, full, ("pick", sku)) is None, "picked past the capacity limit"


def test_place_is_the_inverse_of_pick() -> None:
    spec = generate(1, 2)
    ctx = context_for(spec)
    cell = ctx.access[0][0]
    st = (cell, (0,) * ctx.n_skus, (0,) * (ctx.n_orders * ctx.n_skus), 0, 0, 0, ctx.battery_max)
    picked = apply_action(ctx, st, ("pick", 0))
    assert picked is not None
    back = apply_action(ctx, picked, ("place", 0))
    assert back is not None
    assert back[1] == st[1]
    assert stock_of(ctx, back) == stock_of(ctx, st)


def test_pack_with_a_wrong_sku_ruins_the_order_irreversibly() -> None:
    """The defining mechanic: a surplus or unwanted SKU destroys the box, and no
    sequence of actions can recover."""
    spec = generate(0, 2)
    ctx = context_for(spec)
    # find a SKU that order 0 does not need
    surplus = [s for s in range(ctx.n_skus) if ctx.need_of(0, s) == 0]
    if not surplus:
        pytest.skip("every SKU is required by order 0 in this spec")
    s = surplus[0]
    held = tuple(1 if i == s else 0 for i in range(ctx.n_skus))
    st = (ctx.pack_idx, held, (0,) * (ctx.n_orders * ctx.n_skus), 0, 0, 0, ctx.battery_max)
    ns = apply_action(ctx, st, ("pack", 0))
    assert ns is not None
    assert ns[5] != 0, "packing an unwanted SKU did not ruin the order"
    assert is_dead(ctx, ns)
    assert not is_goal(ctx, ns)
    from taskforge.worlds.warehouse import successors

    assert successors(ctx, ns) == [], "a ruined state must have no successors"
    for a in enumerate_actions(ctx):
        assert apply_action(ctx, ns, a) is None, "a ruined state must accept no action"


def test_scan_requires_a_complete_order() -> None:
    spec = generate(0, 1)
    ctx = context_for(spec)
    st = (ctx.pack_idx, (0,) * ctx.n_skus, (0,) * (ctx.n_orders * ctx.n_skus), 0, 0, 0,
          ctx.battery_max)
    assert apply_action(ctx, st, ("scan", 0)) is None
    full = tuple(ctx.need[i] for i in range(ctx.n_orders * ctx.n_skus))
    st2 = (ctx.pack_idx, (0,) * ctx.n_skus, full, 0, 0, 0, ctx.battery_max)
    ns = apply_action(ctx, st2, ("scan", 0))
    assert ns is not None and ns[3] & 1


def test_unlock_consumes_the_keycard_and_opens_the_zone() -> None:
    for seed in range(20):
        spec = generate(seed, 4)
        ctx = context_for(spec)
        if not ctx.zone_ids:
            continue
        z = ctx.zone_ids[0]
        key = ctx.zone_key[z]
        cell = next(iter(sorted(ctx.zone_adjacent[z])))
        held = tuple(1 if i == key else 0 for i in range(ctx.n_skus))
        st = (cell, held, (0,) * (ctx.n_orders * ctx.n_skus), 0, 0, 0, ctx.battery_max)
        ns = apply_action(ctx, st, ("unlock", z))
        assert ns is not None
        assert ns[4] & 1
        assert ns[1][key] == 0, "unlock did not consume the keycard"
        # without the keycard it must fail
        st2 = (cell, (0,) * ctx.n_skus, st[2], 0, 0, 0, ctx.battery_max)
        assert apply_action(ctx, st2, ("unlock", z)) is None
        return
    pytest.skip("no locked zone found")


def test_battery_blocks_actions_at_zero_and_charge_restores() -> None:
    for seed in range(20):
        spec = generate(seed, 4)
        ctx = context_for(spec)
        if ctx.battery_max < 0 or not ctx.dock_idx:
            continue
        dock = next(iter(sorted(ctx.dock_idx)))
        empty = (dock, (0,) * ctx.n_skus, (0,) * (ctx.n_orders * ctx.n_skus), 0, 0, 0, 0)
        assert apply_action(ctx, empty, ("move", "N")) is None
        charged = apply_action(ctx, empty, ("charge", None))
        assert charged is not None and charged[6] == ctx.battery_max
        return
    pytest.skip("no battery task with a dock found")


def test_successors_are_pure_and_deterministic() -> None:
    from taskforge.worlds.warehouse import successors

    spec = generate(2, 3)
    ctx = context_for(spec)
    s = initial_state(ctx)
    a, b = successors(ctx, s), successors(ctx, s)
    assert a == b
    assert s == initial_state(ctx), "successors mutated the state it was given"


def test_no_action_is_a_self_loop() -> None:
    """Zero-progress transitions would inflate the branching factor and break the
    uniform-cost assumption A* optimality rests on."""
    from taskforge.worlds.warehouse import successors

    spec = generate(0, 3)
    ctx = context_for(spec)
    for _a, ns, cost in successors(ctx, initial_state(ctx)):
        assert ns != initial_state(ctx)
        assert cost == 1
