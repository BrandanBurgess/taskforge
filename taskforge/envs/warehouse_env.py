"""Gymnasium environment over verified warehouse tasks.

Two observation modes -- a structured state vector and rendered pixels -- and a typed,
fixed-size discrete action space with validity masking.

**On the shaping.** The shaped variant adds ``F(s, s') = gamma * Phi(s') - Phi(s)`` with
``Phi(s) = -V*(s)`` taken from the oracle's exact cost-to-go table. Potential-based
shaping of this form leaves the optimal policy unchanged (Ng, Harada & Russell, 1999) --
that is *why* it is safe to use, and it is worth being precise that policy invariance
holds for **any** potential function. Using ``-V*`` is not what makes it sound; it is
what makes it maximally informative. When the state space is too large to enumerate we
fall back to ``Phi(s) = -h(s)`` with the admissible heuristic, which is weaker but
equally sound.

And the honest corollary, stated here because it is easy to hide: a shaped task where
Phi is the exact optimal cost-to-go is **easy by construction**. Following the shaping
gradient *is* the optimal policy. It demonstrates that the plumbing works. It does not
demonstrate that the task is hard. The sparse variant is the real headline.

**Why the shaping discount defaults to 1.0.** With a shaping discount below 1, a step
that makes no progress still pays ``F = gamma*Phi(s) - Phi(s) = (1 - gamma) * V(s)``,
which is strictly positive and *grows with distance from the goal*. Ng et al. guarantee
the optimal policy is unchanged, and it is -- but a finite-horizon learner does not have
to find the optimal policy to be happy, and this one did not. Measured here at
gamma = 0.99: PPO converged to running out the full 120-step budget every episode
(``ep_len_mean = 120``, ``ep_rew_mean ~ 10``) rather than finishing in 8, because
loitering in high-V states paid about +0.065/step and reaching the goal ended the
income. At gamma = 1.0 the shaping telescopes exactly -- +1 per step of real progress, 0
for standing still -- and the exploit disappears. The policy-invariance argument still
holds: for an episodic task with an absorbing terminal state and Phi(terminal) = 0, the
shaping sums to Phi(end) - Phi(start), a constant independent of the route taken.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from taskforge.dsl import TaskSpec
from taskforge.verify.pipeline import VerifiedTask
from taskforge.verify.v2_oracle import CostToGo, cost_to_go
from taskforge.worlds.warehouse import (
    apply_action,
    context_for,
    heuristic,
    initial_state,
    is_dead,
    is_goal,
    stock_of,
)
from taskforge.worlds.warehouse.spec import MAX_ORDERS, MAX_SKUS, MAX_ZONES

# Fixed action layout so a single policy transfers across every task shape.
N_ACTIONS = 4 + MAX_SKUS + MAX_SKUS + MAX_ORDERS + MAX_ORDERS + 1 + MAX_ZONES  # = 25
_DIRS = ("N", "S", "E", "W")


def action_table() -> list[tuple[str, Any]]:
    acts: list[tuple[str, Any]] = [("move", d) for d in _DIRS]
    acts += [("pick", s) for s in range(MAX_SKUS)]
    acts += [("place", s) for s in range(MAX_SKUS)]
    acts += [("pack", o) for o in range(MAX_ORDERS)]
    acts += [("scan", o) for o in range(MAX_ORDERS)]
    acts += [("charge", None)]
    acts += [("unlock", z) for z in range(1, MAX_ZONES + 1)]
    return acts


ACTIONS = action_table()

OBS_DIM = (
    2  # normalised position
    + MAX_SKUS  # held per SKU
    + 1  # held total / capacity
    + MAX_ORDERS * MAX_SKUS  # outstanding need per (order, SKU)
    + MAX_ORDERS * 3  # per order: complete? dispatched? ruined?
    + MAX_ZONES  # zone unlocked flags
    + 1  # battery fraction
    + 1  # steps used fraction
    + MAX_SKUS  # shelf stock per SKU (normalised)
    + MAX_SKUS * 2  # unit vector to each SKU shelf
    + 2  # vector to packing station
    + 2  # vector to nearest charge dock
    + 25  # 5x5 local blocked-cell patch
)


class WarehouseEnv(gym.Env):
    """One verified task, or a curriculum sampled from many."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 12}

    def __init__(
        self,
        tasks: list[VerifiedTask],
        obs_mode: str = "vector",
        shaping: bool = False,
        gamma: float = 1.0,
        seed: int | None = None,
        max_steps: int | None = None,
        horizon_factor: float | None = None,
        vstar_budget: int = 150_000,
        goal_reward: float = 10.0,
        step_cost: float = 0.01,
        ruin_penalty: float = 1.0,
    ):
        super().__init__()
        if not tasks:
            raise ValueError("WarehouseEnv needs at least one verified task")
        self.tasks = tasks
        self.obs_mode = obs_mode
        self.shaping = shaping
        self.gamma = gamma
        self.goal_reward = goal_reward
        self.step_cost = step_cost
        self.ruin_penalty = ruin_penalty
        self.max_steps = max_steps
        self.horizon_factor = horizon_factor
        self._rng = np.random.default_rng(seed)
        self._vstar_budget = vstar_budget
        self._ctg_cache: dict[str, CostToGo] = {}

        self.action_space = spaces.Discrete(N_ACTIONS)
        if obs_mode == "vector":
            self.observation_space = spaces.Box(-1.0, 1.0, (OBS_DIM,), dtype=np.float32)
        elif obs_mode == "pixels":
            self._renderer_cfg = None
            probe = self._render_size(tasks[0].spec)
            self.observation_space = spaces.Box(0, 255, (*probe, 3), dtype=np.uint8)
        else:
            raise ValueError(f"obs_mode must be 'vector' or 'pixels', got {obs_mode!r}")

        self.task: VerifiedTask = tasks[0]
        self.task_spec: TaskSpec = self.task.spec
        self.ctx = context_for(self.task_spec)
        self.state = initial_state(self.ctx)
        self.steps = 0
        self._renderer = None

    # -- helpers -------------------------------------------------------------------
    def _render_size(self, spec: TaskSpec) -> tuple[int, int]:
        from harness.render import RenderConfig, WarehouseRenderer

        r = WarehouseRenderer(spec, RenderConfig(cell=16, show_panel=False, margin=6, header_h=0))
        return r.height, r.width

    def _ctg(self) -> CostToGo | None:
        if not self.shaping:
            return None
        key = self.task_spec.task_id
        if key not in self._ctg_cache:
            self._ctg_cache[key] = cost_to_go(self.task_spec, budget=self._vstar_budget)
        c = self._ctg_cache[key]
        return c if c.exact else None

    def potential(self, state) -> float:
        """Phi(s). Exact -V* when available, else -h(s) from the admissible heuristic.

        Both are valid potentials; only the informativeness differs. Dead states get the
        most negative potential the task can produce, so wandering into an unrecoverable
        state is never rewarded by the shaping term.
        """
        ctg = self._ctg()
        budget = float(self.task_spec.constraints.step_budget)
        if is_dead(self.ctx, state):
            return -budget
        if ctg is not None:
            v = ctg.value(state)
            if v is not None:
                return -float(v)
            return -budget
        return -float(heuristic(self.ctx, state))

    def episode_limit(self) -> int:
        """Steps allowed before truncation.

        Defaults to the spec's step budget, which is what "solvable" is defined against.
        For *training* that is wastefully generous -- a task with an 8-step optimum would
        burn a 120-step episode, so a fixed sample budget buys 15x fewer episodes than it
        should. ``horizon_factor`` scales the horizon off the certificate length instead,
        which is another quiet reuse of the oracle's output: it already knows how long
        this task should take, so it can say how long an attempt is worth running.

        Evaluation always uses the full spec budget, so no agent is judged on a horizon
        the task was not certified against.
        """
        if self.max_steps:
            return self.max_steps
        budget = self.task_spec.constraints.step_budget
        if self.horizon_factor:
            return min(budget, max(16, int(self.horizon_factor * self.task.certificate.cost)))
        return budget

    def valid_action_mask(self) -> np.ndarray:
        """True where the action is legal in the current state. Exposed for maskable
        policies and used by the scripted and LLM agents to score invalid-action rate."""
        mask = np.zeros(N_ACTIONS, dtype=bool)
        for i, a in enumerate(ACTIONS):
            name, arg = a
            if name in ("pick", "place") and arg >= self.ctx.n_skus:
                continue
            if name in ("pack", "scan") and arg >= self.ctx.n_orders:
                continue
            if name == "unlock" and arg not in self.ctx.zone_ids:
                continue
            mask[i] = apply_action(self.ctx, self.state, a) is not None
        return mask

    # -- gym API -------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        idx = int(self._rng.integers(len(self.tasks))) if len(self.tasks) > 1 else 0
        if options and "task_index" in options:
            idx = int(options["task_index"]) % len(self.tasks)
        self.task = self.tasks[idx]
        self.task_spec = self.task.spec
        self.ctx = context_for(self.task_spec)
        self.state = initial_state(self.ctx)
        self.steps = 0
        self._renderer = None
        return self._obs(), self._info()

    def step(self, action: int):
        action = int(action)
        name, arg = ACTIONS[action]
        prev = self.state
        nxt = apply_action(self.ctx, prev, (name, arg))
        invalid = nxt is None
        if not invalid:
            self.state = nxt
        self.steps += 1

        goal = is_goal(self.ctx, self.state)
        dead = is_dead(self.ctx, self.state)
        limit = self.episode_limit()

        reward = -self.step_cost
        if goal:
            reward += self.goal_reward
        if dead and not is_dead(self.ctx, prev):
            reward -= self.ruin_penalty
        if invalid:
            reward -= self.step_cost  # invalid actions still burn a step

        if self.shaping:
            reward += self.gamma * self.potential(self.state) - self.potential(prev)

        terminated = bool(goal or dead)
        truncated = bool(self.steps >= limit and not terminated)
        return self._obs(), float(reward), terminated, truncated, self._info(invalid=invalid)

    def _info(self, invalid: bool = False) -> dict:
        return {
            "task_id": self.task_spec.task_id,
            "difficulty": self.task.difficulty.bucket,
            "optimal": self.task.certificate.cost,
            "steps": self.steps,
            "solved": is_goal(self.ctx, self.state),
            "ruined": is_dead(self.ctx, self.state),
            "invalid_action": invalid,
            "action_mask": self.valid_action_mask(),
        }

    # -- observations ----------------------------------------------------------------
    def _obs(self):
        if self.obs_mode == "pixels":
            return self._obs_pixels()
        return self._obs_vector()

    def _obs_vector(self) -> np.ndarray:
        ctx = self.ctx
        pos, held, filled, dispatched, unlocked, ruined, battery = self.state
        v = np.zeros(OBS_DIM, dtype=np.float32)
        i = 0
        x, y = ctx.xy(pos)
        v[i] = x / max(1, ctx.width - 1)
        v[i + 1] = y / max(1, ctx.height - 1)
        i += 2

        cap = max(1, ctx.capacity)
        for s in range(MAX_SKUS):
            v[i + s] = held[s] / cap if s < ctx.n_skus else 0.0
        i += MAX_SKUS
        v[i] = sum(held) / cap
        i += 1

        for o in range(MAX_ORDERS):
            for s in range(MAX_SKUS):
                if o < ctx.n_orders and s < ctx.n_skus:
                    need = ctx.need_of(o, s)
                    got = filled[o * ctx.n_skus + s]
                    v[i + o * MAX_SKUS + s] = (need - got) / cap
        i += MAX_ORDERS * MAX_SKUS

        for o in range(MAX_ORDERS):
            if o < ctx.n_orders:
                base = o * ctx.n_skus
                complete = all(
                    filled[base + s] == ctx.need[base + s] for s in range(ctx.n_skus)
                )
                v[i + o * 3] = float(complete)
                v[i + o * 3 + 1] = float(dispatched >> o & 1)
                v[i + o * 3 + 2] = float(ruined >> o & 1)
        i += MAX_ORDERS * 3

        for b in range(MAX_ZONES):
            v[i + b] = float(unlocked >> b & 1) if b < len(ctx.zone_ids) else 1.0
        i += MAX_ZONES

        v[i] = (battery / ctx.battery_max) if ctx.battery_max > 0 else 1.0
        i += 1
        v[i] = self.steps / max(1, self.task_spec.constraints.step_budget)
        i += 1

        stock = stock_of(ctx, self.state)
        for s in range(MAX_SKUS):
            v[i + s] = stock[s] / 4.0 if s < ctx.n_skus else 0.0
        i += MAX_SKUS

        # unit vectors to landmarks: gives the MLP spatial structure without a CNN
        for s in range(MAX_SKUS):
            if s < ctx.n_skus and s in ctx.shelf_idx:
                sx, sy = ctx.xy(ctx.shelf_idx[s])
                v[i + s * 2] = np.clip((sx - x) / max(1, ctx.width), -1, 1)
                v[i + s * 2 + 1] = np.clip((sy - y) / max(1, ctx.height), -1, 1)
        i += MAX_SKUS * 2

        pxx, pyy = ctx.xy(ctx.pack_idx)
        v[i] = np.clip((pxx - x) / max(1, ctx.width), -1, 1)
        v[i + 1] = np.clip((pyy - y) / max(1, ctx.height), -1, 1)
        i += 2

        if ctx.dock_idx:
            best = min(ctx.dock_idx, key=lambda c: int(ctx.dist[pos][c]))
            dx, dy = ctx.xy(best)
            v[i] = np.clip((dx - x) / max(1, ctx.width), -1, 1)
            v[i + 1] = np.clip((dy - y) / max(1, ctx.height), -1, 1)
        i += 2

        for oy in range(-2, 3):
            for ox in range(-2, 3):
                nx, ny = x + ox, y + oy
                blocked = 1.0
                if 0 <= nx < ctx.width and 0 <= ny < ctx.height:
                    nidx = ny * ctx.width + nx
                    blocked = 0.0 if ctx.passable[nidx] else 1.0
                v[i] = blocked
                i += 1
        return v

    def _obs_pixels(self) -> np.ndarray:
        from harness.render import RenderConfig, WarehouseRenderer, surf_to_array

        if self._renderer is None:
            self._renderer = WarehouseRenderer(
                self.task_spec, RenderConfig(cell=16, show_panel=False, margin=6, header_h=0)
            )
        surf = self._renderer.render_frame(self.state, step=self.steps)
        return surf_to_array(surf).astype(np.uint8)

    def render(self):
        from harness.render import RenderConfig, WarehouseRenderer, surf_to_array

        if self._renderer is None:
            self._renderer = WarehouseRenderer(self.task_spec, RenderConfig(cell=28))
        surf = self._renderer.render_frame(
            self.state, step=self.steps, optimal=self.task.certificate.cost
        )
        return surf_to_array(surf)

    # -- convenience ---------------------------------------------------------------
    def action_index(self, action: tuple[str, Any]) -> int:
        return ACTIONS.index(action)

    def legal_action_indices(self) -> list[int]:
        return [i for i, ok in enumerate(self.valid_action_mask()) if ok]


def make_env(
    tasks: list[VerifiedTask], shaping: bool = False, seed: int | None = None, **kw
) -> WarehouseEnv:
    return WarehouseEnv(tasks, shaping=shaping, seed=seed, **kw)
