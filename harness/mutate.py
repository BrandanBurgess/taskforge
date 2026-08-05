"""MAP-Elites over verified tasks: compounding edits, each mutant re-verified.

The archive is a 2D grid over (difficulty bucket x entity multiset signature). Each cell
keeps the single best-performing elite found so far, and new mutants are drawn from
existing elites -- so edits compound and lineage depth grows.

Every mutant goes back through the full V1/V2/V3 pipeline. A mutation that breaks
solvability is simply not archived, which is the useful property: the archive can only
ever contain tasks that are still provably solvable, no matter how many generations of
edits produced them.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from taskforge.dsl import Constraints, TaskSpec
from taskforge.verify import VerifiedTask, verify
from taskforge.verify.pipeline import accepted_task
from taskforge.worlds.warehouse.spec import MAX_ORDERS, Order, WarehousePayload

MUTATIONS = (
    "add_unit",
    "drop_unit",
    "add_order",
    "tighten_capacity",
    "loosen_capacity",
    "move_sku",
    "add_conveyor",
    "flip_conveyor",
    "shift_start",
    "tighten_budget",
    "add_decoy_stock",
)


@dataclass
class Elite:
    task: VerifiedTask
    lineage: list[str] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return len(self.lineage)


def entity_signature(spec: TaskSpec) -> str:
    """The 'tool multiset' axis of the archive: which world features a task actually
    uses. Two tasks with the same signature pose structurally similar problems."""
    p = WarehousePayload.model_validate(spec.payload)
    bits = []
    bits.append(f"s{len(p.skus)}")
    bits.append(f"o{len(p.orders)}")
    if any(c in "^v<>" for row in p.tiles for c in row):
        bits.append("conv")
    if p.zone_keys:
        bits.append(f"z{len(p.zone_keys)}")
    if p.battery_max:
        bits.append("bat")
    return "+".join(bits)


def _rebuild(spec: TaskSpec, payload: WarehousePayload, tag: str, **cons) -> TaskSpec:
    c = spec.constraints.model_dump()
    c.update(cons)
    c["battery"] = payload.battery_max
    meta = dict(spec.metadata)
    meta["mutated_from"] = spec.task_id
    return TaskSpec(
        task_id=f"{spec.task_id}~{tag}",
        world=spec.world,
        seed=spec.seed,
        tools=spec.tools,
        goal=spec.goal,
        constraints=Constraints(**c),
        payload=payload.model_dump(mode="json"),
        metadata=meta,
    )


def mutate(spec: TaskSpec, rng: random.Random) -> tuple[TaskSpec, str] | None:
    """Apply one random edit. Returns ``None`` when the edit does not apply here."""
    p = WarehousePayload.model_validate(spec.payload)
    op = rng.choice(MUTATIONS)
    n_item = len(p.skus) - len(p.zone_keys)
    item_ids = [s for s in range(n_item)]
    if not item_ids:
        return None
    orders = [o.model_copy(deep=True) for o in p.orders]
    stock = list(p.stock)
    tiles = [list(r) for r in p.tiles]
    cons: dict = {}

    if op == "add_unit":
        o = rng.choice(orders)
        s = rng.choice(item_ids)
        o.need[s] += 1
        stock[s] += 1
    elif op == "drop_unit":
        cands = [(o, s) for o in orders for s in item_ids if o.need[s] > 0]
        if not cands:
            return None
        o, s = rng.choice(cands)
        if sum(o.need) <= 1:
            return None
        o.need[s] -= 1
    elif op == "add_order":
        if len(orders) >= MAX_ORDERS:
            return None
        need = [0] * len(p.skus)
        s = rng.choice(item_ids)
        need[s] = 1
        stock[s] += 1
        orders.append(
            Order(order_id=len(orders), need=need, destination=f"dock-{chr(65 + len(orders))}")
        )
    elif op == "tighten_capacity":
        if spec.constraints.capacity <= 1:
            return None
        cons["capacity"] = spec.constraints.capacity - 1
    elif op == "loosen_capacity":
        if spec.constraints.capacity >= 6:
            return None
        cons["capacity"] = spec.constraints.capacity + 1
    elif op == "add_decoy_stock":
        s = rng.choice(item_ids)
        stock[s] += 1
    elif op == "tighten_budget":
        cons["step_budget"] = max(8, int(spec.constraints.step_budget * 0.8))
    elif op == "move_sku":
        s = rng.choice(item_ids)
        here = p.shelf_pos().get(s)
        spots = [
            (x, y)
            for y in range(p.height)
            for x in range(p.width)
            if tiles[y][x] == "#"
            and 0 < x < p.width - 1
            and 0 < y < p.height - 1
            and any(
                p.is_passable(x + dx, y + dy) for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))
            )
        ]
        if not spots or here is None:
            return None
        nx, ny = rng.choice(spots)
        tiles[here[1]][here[0]] = "#"
        tiles[ny][nx] = str(s)
    elif op in ("add_conveyor", "flip_conveyor"):
        if op == "flip_conveyor":
            cells = [
                (x, y)
                for y in range(p.height)
                for x in range(p.width)
                if tiles[y][x] in "^v<>"
            ]
            if not cells:
                return None
            x, y = rng.choice(cells)
            tiles[y][x] = {"^": "v", "v": "^", "<": ">", ">": "<"}[tiles[y][x]]
        else:
            cells = [
                (x, y)
                for y in range(1, p.height - 1)
                for x in range(1, p.width - 1)
                if tiles[y][x] == "." and (x, y) != tuple(p.start)
            ]
            if not cells:
                return None
            x, y = rng.choice(cells)
            tiles[y][x] = rng.choice("^v<>")
    elif op == "shift_start":
        spots = [
            (x, y)
            for y in range(1, p.height - 1)
            for x in range(1, p.width - 1)
            if tiles[y][x] == "." and p.zone(x, y) == 0
        ]
        if not spots:
            return None
        p = p.model_copy(update={"start": rng.choice(spots)})

    try:
        new_payload = p.model_copy(
            update={
                "tiles": ["".join(r) for r in tiles],
                "orders": orders,
                "stock": stock,
            }
        )
        WarehousePayload.model_validate(new_payload.model_dump(mode="json"))
    except Exception:
        return None
    return _rebuild(spec, new_payload, op, **cons), op


@dataclass
class Archive:
    cells: dict[tuple[int, str], Elite] = field(default_factory=dict)
    attempts: int = 0
    accepted: int = 0
    rejected_by: dict[str, int] = field(default_factory=dict)
    ops_tried: dict[str, int] = field(default_factory=dict)
    ops_accepted: dict[str, int] = field(default_factory=dict)

    def key(self, task: VerifiedTask) -> tuple[int, str]:
        return (task.difficulty.bucket, entity_signature(task.spec))

    def add(self, task: VerifiedTask, lineage: list[str]) -> bool:
        k = self.key(task)
        cur = self.cells.get(k)
        # Elitism: keep the longest certificate in each cell -- within a niche, a task
        # that takes more optimal steps is the more demanding instance of that niche.
        if cur is None or task.certificate.cost > cur.task.certificate.cost:
            self.cells[k] = Elite(task, lineage)
            return True
        return False

    @property
    def coverage(self) -> int:
        return len(self.cells)

    @property
    def max_depth(self) -> int:
        return max((e.depth for e in self.cells.values()), default=0)

    def to_json(self) -> dict:
        buckets = sorted({k[0] for k in self.cells})
        sigs = sorted({k[1] for k in self.cells})
        return {
            "attempts": self.attempts,
            "accepted": self.accepted,
            "accept_rate": round(self.accepted / max(1, self.attempts), 4),
            "coverage_cells": self.coverage,
            "grid": {"difficulty_buckets": buckets, "entity_signatures": sigs},
            "max_lineage_depth": self.max_depth,
            "mean_lineage_depth": round(
                sum(e.depth for e in self.cells.values()) / max(1, len(self.cells)), 3
            ),
            "rejected_by_stage": self.rejected_by,
            "mutation_ops": {
                op: {
                    "tried": self.ops_tried.get(op, 0),
                    "accepted": self.ops_accepted.get(op, 0),
                    "accept_rate": round(
                        self.ops_accepted.get(op, 0) / max(1, self.ops_tried.get(op, 0)), 3
                    ),
                }
                for op in sorted(self.ops_tried)
            },
            "cells": [
                {
                    "difficulty": k[0],
                    "signature": k[1],
                    "task_id": e.task.spec.task_id,
                    "optimal_cost": e.task.certificate.cost,
                    "lineage": e.lineage,
                    "depth": e.depth,
                }
                for k, e in sorted(self.cells.items())
            ],
        }


def run_map_elites(
    seeds: list[VerifiedTask],
    iterations: int = 400,
    rng_seed: int = 0,
    node_budget: int = 120_000,
    verbose: bool = False,
) -> Archive:
    rng = random.Random(rng_seed)
    arch = Archive()
    for t in seeds:
        arch.add(t, [])

    for i in range(iterations):
        if not arch.cells:
            break
        parent = rng.choice(list(arch.cells.values()))
        m = mutate(parent.task.spec, rng)
        if m is None:
            continue
        spec, op = m
        arch.attempts += 1
        arch.ops_tried[op] = arch.ops_tried.get(op, 0) + 1
        outcome = verify(spec, node_budget=node_budget)
        if not outcome.accepted:
            stage = outcome.reject_stage or "?"
            arch.rejected_by[stage] = arch.rejected_by.get(stage, 0) + 1
            continue
        arch.accepted += 1
        arch.ops_accepted[op] = arch.ops_accepted.get(op, 0) + 1
        arch.add(accepted_task(spec, outcome), [*parent.lineage, op])
        if verbose and i % 50 == 0:
            print(f"  iter {i}: coverage {arch.coverage}, depth {arch.max_depth}")
    return arch


def main() -> None:
    import argparse

    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default=str(root / "specs"))
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--node-budget", type=int, default=120_000)
    ap.add_argument("--out", default=str(root / "results" / "map_elites.json"))
    args = ap.parse_args()

    from taskforge.verify import load_specs

    seeds = load_specs(args.specs)
    if not seeds:
        raise SystemExit("no specs; run scripts/build_specs.py first")
    print(f"seeding MAP-Elites with {len(seeds)} verified tasks")
    arch = run_map_elites(
        seeds, iterations=args.iterations, node_budget=args.node_budget, verbose=True
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(arch.to_json(), indent=2) + "\n")
    print(
        f"coverage {arch.coverage} cells | accept {arch.accepted}/{arch.attempts} "
        f"| max lineage depth {arch.max_depth}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
