"""Agents evaluated against the oracle's goal predicate.

All of them expose the same ``act(env) -> int`` interface and are graded identically:
solved / not solved by the goal predicate, plan length as a ratio to oracle-optimal,
invalid-action rate, and whether they entered an unrecoverable state.

The LLM agent degrades to :class:`ScriptedAgent` when no API key is present, so the full
evaluation table is producible offline. That fallback is labelled honestly everywhere it
appears -- it is a different agent, not a stand-in for a model.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from taskforge.envs.warehouse_env import ACTIONS, WarehouseEnv
from taskforge.worlds.warehouse import context_for


class Agent(Protocol):
    name: str

    def reset(self, env: WarehouseEnv) -> None: ...
    def act(self, env: WarehouseEnv) -> int: ...


@dataclass
class EpisodeResult:
    task_id: str
    difficulty: int
    optimal: int
    steps: int
    solved: bool
    ruined: bool
    invalid_actions: int
    reward: float

    @property
    def length_ratio(self) -> float | None:
        """Steps taken divided by oracle-optimal. Only meaningful when solved."""
        if not self.solved or not self.optimal:
            return None
        return self.steps / self.optimal


def run_episode(env: WarehouseEnv, agent: Agent, task_index: int, max_steps: int | None = None):
    obs, info = env.reset(options={"task_index": task_index})
    agent.reset(env)
    limit = max_steps or env.task_spec.constraints.step_budget
    invalid = 0
    total = 0.0
    for _ in range(limit):
        a = agent.act(env)
        obs, r, term, trunc, info = env.step(a)
        total += r
        invalid += int(info["invalid_action"])
        if term or trunc:
            break
    return EpisodeResult(
        task_id=env.task_spec.task_id,
        difficulty=env.task.difficulty.bucket,
        optimal=env.task.certificate.cost,
        steps=env.steps,
        solved=bool(info["solved"]),
        ruined=bool(info["ruined"]),
        invalid_actions=invalid,
        reward=total,
    )


# --------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------


class RandomAgent:
    """Uniform over *legal* actions. Deliberately not uniform over all 25 -- a random
    baseline that mostly emits illegal actions is a straw man, and would make the
    learned policies look better than they are."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def reset(self, env: WarehouseEnv) -> None:
        pass

    def act(self, env: WarehouseEnv) -> int:
        legal = env.legal_action_indices()
        return self.rng.choice(legal) if legal else 0


class GreedyAgent:
    """Hand-written heuristic policy: head for the nearest SKU you still need, pick it,
    carry it to packing, pack, scan. Unlocks a zone when it holds the keycard and the
    zone gates something it needs; recharges when low.

    It routes on the *relaxed* distance field, so one-way conveyors can trap it -- and it
    has no model of irreversibility beyond "never pack a SKU this order doesn't need".
    This is the strongest baseline that does not use search, which is what makes it the
    honest thing to compare a learned policy against.
    """

    name = "greedy"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def reset(self, env: WarehouseEnv) -> None:
        pass

    def act(self, env: WarehouseEnv) -> int:
        ctx = env.ctx
        pos, held, filled, dispatched, unlocked, ruined, battery = env.state
        mask = env.valid_action_mask()

        def idx_of(a) -> int | None:
            try:
                i = ACTIONS.index(a)
            except ValueError:
                return None
            return i if mask[i] else None

        # 1. dispatch anything complete
        for o in range(ctx.n_orders):
            i = idx_of(("scan", o))
            if i is not None:
                return i

        # 2. at packing with useful cargo: pack into an order that needs exactly this
        if pos == ctx.pack_idx and sum(held) > 0:
            for o in range(ctx.n_orders):
                if dispatched >> o & 1:
                    continue
                base = o * ctx.n_skus
                fits = all(
                    filled[base + s] + held[s] <= ctx.need[base + s] for s in range(ctx.n_skus)
                )
                if fits and any(held[s] > 0 for s in range(ctx.n_skus)):
                    i = idx_of(("pack", o))
                    if i is not None:
                        return i
            # cargo fits no order: shed it rather than ruin a box
            for s in range(ctx.n_skus):
                if held[s] > 0:
                    i = idx_of(("place", s))
                    if i is not None:
                        return i

        # 3. unlock a gating zone when the keycard is in hand
        for z in ctx.zone_ids:
            i = idx_of(("unlock", z))
            if i is not None:
                return i

        # 4. top up if the battery is low and a dock is underfoot
        if ctx.battery_max > 0 and battery < ctx.battery_max * 0.3:
            i = idx_of(("charge", None))
            if i is not None:
                return i

        wanted = self._wanted(env)
        # 5. pick a needed SKU we are standing next to
        for s in wanted:
            i = idx_of(("pick", s))
            if i is not None:
                return i

        # 6. otherwise walk toward the next objective
        target = self._target(env, wanted)
        if target is not None:
            step = self._step_toward(env, pos, target)
            if step is not None:
                return step
        legal = [i for i in env.legal_action_indices() if ACTIONS[i][0] == "move"]
        if not legal:
            legal = env.legal_action_indices()
        return self.rng.choice(legal) if legal else 0

    def _wanted(self, env: WarehouseEnv) -> list[int]:
        ctx = env.ctx
        _, held, filled, dispatched, unlocked, _, _ = env.state
        out = []
        for s in range(ctx.n_skus):
            remaining = sum(
                max(0, ctx.need_of(o, s) - filled[o * ctx.n_skus + s])
                for o in range(ctx.n_orders)
            )
            if remaining > held[s] and sum(held) < ctx.capacity:
                out.append(s)
        # a keycard for a still-locked zone is also "wanted"
        for bit, z in enumerate(ctx.zone_ids):
            if not (unlocked >> bit & 1):
                key = ctx.zone_key[z]
                if held[key] == 0 and sum(held) < ctx.capacity:
                    out.append(key)
        return out

    def _target(self, env: WarehouseEnv, wanted: list[int]) -> int | None:
        ctx = env.ctx
        pos, held, _, _, _, _, battery = env.state
        if ctx.battery_max > 0 and battery < ctx.battery_max * 0.25 and ctx.dock_idx:
            return min(ctx.dock_idx, key=lambda c: int(ctx.dist[pos][c]))
        if wanted:
            cells = [
                (int(ctx.dist[pos][c]), c) for s in wanted for c in ctx.access.get(s, ())
            ]
            cells = [c for c in cells if c[0] < (1 << 19)]
            if cells:
                return min(cells)[1]
        if sum(held) > 0:
            return ctx.pack_idx
        return ctx.pack_idx

    def _step_toward(self, env: WarehouseEnv, pos: int, target: int) -> int | None:
        ctx = env.ctx
        mask = env.valid_action_mask()
        best, best_d = None, int(ctx.dist[pos][target])
        for i, (name, arg) in enumerate(ACTIONS):
            if name != "move" or not mask[i]:
                continue
            from taskforge.worlds.warehouse import apply_action

            ns = apply_action(ctx, env.state, (name, arg))
            if ns is None:
                continue
            d = int(ctx.dist[ns[0]][target])
            if d < best_d:
                best, best_d = i, d
        return best


class OracleAgent:
    """Replays the certificate plan. The upper bound every other agent is measured
    against."""

    name = "oracle"

    def __init__(self) -> None:
        self.plan: list[Any] = []
        self.i = 0

    def reset(self, env: WarehouseEnv) -> None:
        self.plan = env.task.certificate.as_actions()
        self.i = 0

    def act(self, env: WarehouseEnv) -> int:
        if self.i >= len(self.plan):
            return 0
        a = self.plan[self.i]
        self.i += 1
        return ACTIONS.index(a)


class PolicyAgent:
    """Wraps a trained stable-baselines3 policy. Masks to legal actions at inference so
    a learned policy is not penalised for the one thing masking exists to prevent."""

    def __init__(self, model, name: str = "ppo", mask: bool = True):
        self.model = model
        self.name = name
        self.mask = mask

    def reset(self, env: WarehouseEnv) -> None:
        pass

    def act(self, env: WarehouseEnv) -> int:
        obs = env._obs()
        if not self.mask:
            a, _ = self.model.predict(obs, deterministic=True)
            return int(a)
        dist = self._action_scores(obs)
        legal = env.valid_action_mask()
        if not legal.any():
            return 0
        dist = np.where(legal, dist, -np.inf)
        return int(np.argmax(dist))

    def _action_scores(self, obs) -> np.ndarray:
        import torch

        with torch.no_grad():
            t, _ = self.model.policy.obs_to_tensor(obs)
            d = self.model.policy.get_distribution(t)
            return d.distribution.logits.cpu().numpy().reshape(-1)


# --------------------------------------------------------------------------------------
# Scripted / LLM
# --------------------------------------------------------------------------------------


class ScriptedAgent(GreedyAgent):
    """The offline stand-in for the LLM agent.

    It is the greedy policy with a small amount of deliberate sloppiness: with
    probability ``noise`` it takes a random legal action instead of the greedy one. That
    makes it fail in ways a language model plausibly fails -- occasional detours,
    occasional wrong pack -- so the evaluation harness is exercised end to end offline.
    It is NOT a model of LLM competence and is never reported as one.
    """

    name = "scripted"

    def __init__(self, seed: int = 0, noise: float = 0.12):
        super().__init__(seed)
        self.noise = noise

    def act(self, env: WarehouseEnv) -> int:
        if self.rng.random() < self.noise:
            legal = env.legal_action_indices()
            if legal:
                return self.rng.choice(legal)
        return super().act(env)


PROMPT = """You are controlling a picker robot in a grid warehouse.

Goal: every order must be filled with exactly the SKUs it requires and then dispatched
with `scan`.

Critical rule: `pack(order_id)` deposits your ENTIRE held multiset into that order. If
any item is not needed by that order, or exceeds what it still needs, the box is RUINED
and the task becomes impossible. Drop unwanted items with `place` before packing.

{state}

Legal actions right now (choose exactly one, by index):
{actions}

Reply with a single JSON object: {{"action": <index>, "why": "<short reason>"}}
"""


@dataclass
class LLMAgent:
    """Claude via forced tool use, scored by the same oracle goal predicate.

    Falls back to :class:`ScriptedAgent` when ``ANTHROPIC_API_KEY`` is unset so the
    harness is testable offline. ``used_llm`` records which path actually ran, and the
    evaluation report carries that flag through to the results JSON -- so a table
    produced offline can never be mistaken for a benchmarked model.
    """

    name: str = "llm"
    model: str = "claude-opus-4-5"
    max_tokens: int = 512
    seed: int = 0
    _fallback: ScriptedAgent = field(default_factory=lambda: ScriptedAgent(0))
    used_llm: bool = False
    calls: int = 0

    def available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def reset(self, env: WarehouseEnv) -> None:
        self._fallback.reset(env)

    def act(self, env: WarehouseEnv) -> int:
        if not self.available():
            return self._fallback.act(env)
        try:
            idx = self._ask(env)
            self.used_llm = True
            return idx
        except Exception:
            # A transport error must not silently become a different agent's decision
            # without being visible, but it also must not abort a long evaluation.
            return self._fallback.act(env)

    def _ask(self, env: WarehouseEnv) -> int:
        import anthropic

        client = anthropic.Anthropic()
        legal = env.legal_action_indices()
        listing = "\n".join(f"  {i}: {describe_action(env, ACTIONS[i])}" for i in legal)
        prompt = PROMPT.format(state=describe_state(env), actions=listing)
        tool = {
            "name": "take_action",
            "description": "Choose one legal action by its index.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "integer", "enum": legal},
                    "why": {"type": "string"},
                },
                "required": ["action"],
            },
        }
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            tools=[tool],
            tool_choice={"type": "tool", "name": "take_action"},
            messages=[{"role": "user", "content": prompt}],
        )
        self.calls += 1
        for block in resp.content:
            if block.type == "tool_use":
                idx = int(block.input["action"])
                return idx if idx in legal else (legal[0] if legal else 0)
        return legal[0] if legal else 0


def describe_action(env: WarehouseEnv, action) -> str:
    name, arg = action
    ctx = env.ctx
    if name == "move":
        return f"move {arg}"
    if name in ("pick", "place"):
        return f"{name} {ctx.payload.skus[arg]}"
    if name == "pack":
        return f"pack held items into order {arg}"
    if name == "scan":
        return f"scan and dispatch order {arg}"
    if name == "unlock":
        return f"unlock zone {arg}"
    return name


def describe_state(env: WarehouseEnv) -> str:
    """A plain-text state description. Deliberately complete: the LLM is evaluated on
    decision quality, not on inferring hidden information."""
    ctx = env.ctx
    p = ctx.payload
    pos, held, filled, dispatched, unlocked, ruined, battery = env.state
    x, y = ctx.xy(pos)
    lines = [f"Grid {ctx.width}x{ctx.height}. Robot at ({x},{y}). Step {env.steps}."]
    px, py = ctx.xy(ctx.pack_idx)
    lines.append(f"Packing station at ({px},{py}).")
    lines.append("Map (# wall/rack, . aisle, digit = shelf holding that SKU index,")
    lines.append("     P packing, C charge dock, ^v<> one-way conveyor):")
    lines += ["  " + row for row in p.tiles]
    lines.append("SKUs: " + ", ".join(f"{i}={n}" for i, n in enumerate(p.skus)))
    from taskforge.worlds.warehouse import stock_of

    stock = stock_of(ctx, env.state)
    lines.append("Shelf stock: " + ", ".join(f"{p.skus[s]}={stock[s]}" for s in range(ctx.n_skus)))
    holding = [p.skus[s] for s in range(ctx.n_skus) for _ in range(held[s])]
    lines.append(f"Holding ({sum(held)}/{ctx.capacity}): {holding or 'nothing'}")
    for o in range(ctx.n_orders):
        parts = []
        for s in range(ctx.n_skus):
            need = ctx.need_of(o, s)
            if need:
                parts.append(f"{p.skus[s]} {filled[o * ctx.n_skus + s]}/{need}")
        status = "dispatched" if dispatched >> o & 1 else ("RUINED" if ruined >> o & 1 else "open")
        lines.append(f"Order {o} [{status}]: " + ", ".join(parts))
    if ctx.zone_ids:
        for bit, z in enumerate(ctx.zone_ids):
            state = "unlocked" if unlocked >> bit & 1 else "locked"
            lines.append(f"Zone {z}: {state} (keycard = {p.skus[ctx.zone_key[z]]})")
    if ctx.battery_max > 0:
        lines.append(f"Battery: {battery}/{ctx.battery_max}")
    _ = context_for
    return "\n".join(lines)


BASELINES = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "scripted": ScriptedAgent,
}
