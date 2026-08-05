"""Gym API conformance, shaping invariance, determinism, renderer smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

from taskforge.envs import ACTIONS, N_ACTIONS, OBS_DIM, WarehouseEnv
from taskforge.verify import verify
from taskforge.verify.pipeline import accepted_task
from taskforge.worlds.warehouse import generate


def make_tasks(difficulties=(1, 2), seeds=(0, 1)):
    out = []
    for d in difficulties:
        for s in seeds:
            spec = generate(s, d)
            outcome = verify(spec)
            if outcome.accepted:
                out.append(accepted_task(spec, outcome))
    assert out
    return out


# --------------------------------------------------------------------------------------
# Gym conformance
# --------------------------------------------------------------------------------------


def test_gym_api_conformance() -> None:
    from gymnasium.utils.env_checker import check_env

    env = WarehouseEnv(make_tasks(), obs_mode="vector", seed=0)
    check_env(env, skip_render_check=True)


def test_reset_step_contract() -> None:
    env = WarehouseEnv(make_tasks(), seed=0)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs), "reset observation is outside the declared space"
    assert obs.shape == (OBS_DIM,)
    obs, r, term, trunc, info = env.step(0)
    assert env.observation_space.contains(obs)
    assert isinstance(r, float)
    assert isinstance(term, bool) and isinstance(trunc, bool)
    assert set(["task_id", "difficulty", "optimal", "solved", "ruined"]) <= set(info)


def test_action_space_is_fixed_size_across_tasks() -> None:
    """A single policy has to transfer across task shapes, so the action layout must not
    depend on how many SKUs or orders a particular task has."""
    tasks = make_tasks(difficulties=(1, 5), seeds=(0,))
    env = WarehouseEnv(tasks, seed=0)
    assert env.action_space.n == N_ACTIONS == len(ACTIONS)
    for i in range(len(tasks)):
        env.reset(options={"task_index": i})
        assert env.action_space.n == N_ACTIONS
        assert env.valid_action_mask().shape == (N_ACTIONS,)


def test_validity_mask_agrees_with_the_executor() -> None:
    from taskforge.worlds.warehouse import apply_action

    env = WarehouseEnv(make_tasks(), seed=3)
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(120):
        mask = env.valid_action_mask()
        for i, a in enumerate(ACTIONS):
            name, arg = a
            if name in ("pick", "place") and arg >= env.ctx.n_skus:
                continue
            if name in ("pack", "scan") and arg >= env.ctx.n_orders:
                continue
            if name == "unlock" and arg not in env.ctx.zone_ids:
                continue
            legal = apply_action(env.ctx, env.state, a) is not None
            assert mask[i] == legal, f"mask disagrees with executor on {a}"
        legal_idx = env.legal_action_indices()
        if not legal_idx:
            break
        _, _, term, trunc, _ = env.step(int(rng.choice(legal_idx)))
        if term or trunc:
            break


def test_oracle_plan_solves_the_env() -> None:
    """End-to-end: the certificate replayed through the *Gym* env must reach the goal in
    exactly the certified number of steps."""
    from harness.agents import OracleAgent, run_episode

    tasks = make_tasks(difficulties=(1, 2, 3), seeds=(0, 1))
    env = WarehouseEnv(tasks, seed=0)
    for i, t in enumerate(tasks):
        res = run_episode(env, OracleAgent(), i)
        assert res.solved, f"{t.spec.task_id}: certificate did not solve the env"
        assert res.steps == t.certificate.cost
        assert res.length_ratio == 1.0
        assert res.invalid_actions == 0


# --------------------------------------------------------------------------------------
# Shaping
# --------------------------------------------------------------------------------------


def test_shaping_telescopes_to_a_constant_over_any_trajectory() -> None:
    """Potential-based shaping is policy-invariant because the added terms telescope:
    with gamma = 1 the total shaping bonus over a trajectory depends only on its
    endpoints, never on the route taken. Two different routes between the same pair of
    states must therefore accrue exactly the same shaping total."""
    tasks = make_tasks(difficulties=(1,), seeds=(0,))
    env = WarehouseEnv(tasks, shaping=True, gamma=1.0, seed=0)
    env.reset(options={"task_index": 0})
    start = env.state
    total = 0.0
    rng = np.random.default_rng(2)
    for _ in range(25):
        legal = env.legal_action_indices()
        if not legal:
            break
        prev = env.state
        env.step(int(rng.choice(legal)))
        total += env.gamma * env.potential(env.state) - env.potential(prev)
    end = env.state
    assert total == pytest.approx(env.potential(end) - env.potential(start), abs=1e-6)


def test_shaping_does_not_change_which_states_are_terminal() -> None:
    tasks = make_tasks(difficulties=(1,), seeds=(0,))
    plain = WarehouseEnv(tasks, shaping=False, seed=0)
    shaped = WarehouseEnv(tasks, shaping=True, seed=0)
    from harness.agents import OracleAgent, run_episode

    a = run_episode(plain, OracleAgent(), 0)
    b = run_episode(shaped, OracleAgent(), 0)
    assert a.solved == b.solved
    assert a.steps == b.steps


def test_potential_is_zero_at_goal_and_worst_in_a_dead_state() -> None:
    """Phi(goal) = 0 because no actions remain; a ruined state gets the most negative
    potential the task can produce, so the shaping term can never reward wandering into
    an unrecoverable state."""
    tasks = make_tasks(difficulties=(1, 2), seeds=(0, 1))
    env = WarehouseEnv(tasks, shaping=True, seed=0)
    env.reset(options={"task_index": 0})
    from harness.agents import OracleAgent

    agent = OracleAgent()
    agent.reset(env)
    for _ in range(env.task.certificate.cost):
        env.step(agent.act(env))
    assert env.potential(env.state) == 0.0

    # construct a ruined state directly and confirm it is the worst potential available
    for i in range(len(tasks)):
        env.reset(options={"task_index": i})
        ctx = env.ctx
        surplus = [s for s in range(ctx.n_skus) if ctx.need_of(0, s) == 0]
        if not surplus:
            continue
        held = tuple(1 if k == surplus[0] else 0 for k in range(ctx.n_skus))
        ruined = (ctx.pack_idx, held, (0,) * (ctx.n_orders * ctx.n_skus), 0, 0, 1, ctx.battery_max)
        assert env.potential(ruined) == -float(env.task_spec.constraints.step_budget)
        assert env.potential(ruined) < env.potential(env.state)
        return
    pytest.skip("no task with a surplus SKU available")


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------


def test_same_seed_gives_identical_spec_plan_and_first_frame() -> None:
    from harness.render import RenderConfig, WarehouseRenderer, surf_to_array
    from taskforge.verify import solve
    from taskforge.worlds.warehouse import context_for, initial_state

    a, b = generate(11, 3), generate(11, 3)
    assert a.canonical_json() == b.canonical_json()
    assert solve(a).plan == solve(b).plan

    frames = []
    for spec in (a, b):
        r = WarehouseRenderer(spec, RenderConfig(cell=24))
        frames.append(surf_to_array(r.render_frame(initial_state(context_for(spec)))))
    assert np.array_equal(frames[0], frames[1]), "first frame is not byte-identical"


# --------------------------------------------------------------------------------------
# Renderer
# --------------------------------------------------------------------------------------


def test_renderer_smoke_under_dummy_driver(tmp_path) -> None:
    import os

    assert os.environ.get("SDL_VIDEODRIVER") == "dummy"
    from harness.render import RenderConfig, episode_frames, save_gif, save_png

    spec = generate(0, 2)
    outcome = verify(spec)
    assert outcome.accepted
    frames, final = episode_frames(
        spec, outcome.certificate.as_actions(), RenderConfig(cell=20), tween=2,
        optimal=outcome.certificate.cost,
    )
    assert len(frames) > 10
    save_png(frames[0], tmp_path / "f.png")
    gif = save_gif(frames, tmp_path / "f.gif", fps=12)
    assert gif.exists() and gif.stat().st_size > 0


def test_pixel_observation_mode() -> None:
    env = WarehouseEnv(make_tasks(difficulties=(1,), seeds=(0,)), obs_mode="pixels", seed=0)
    obs, _ = env.reset()
    assert obs.dtype == np.uint8
    assert env.observation_space.contains(obs)
    obs2, _, _, _, _ = env.step(0)
    assert obs2.shape == obs.shape


def test_renderer_handles_every_difficulty() -> None:
    from harness.render import RenderConfig, WarehouseRenderer
    from taskforge.worlds.warehouse import context_for, initial_state

    for d in range(1, 6):
        spec = generate(0, d)
        r = WarehouseRenderer(spec, RenderConfig(cell=18))
        surf = r.render_frame(initial_state(context_for(spec)), step=0, optimal=10)
        assert surf.get_width() > 0 and surf.get_height() > 0
