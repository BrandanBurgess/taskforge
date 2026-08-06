"""The four rendered artifacts: hero GIF, cost-to-go field, three-way comparison, replay.

Kept apart from ``figures.py`` (which draws matplotlib charts from results JSON) because
these run the simulator to produce their content.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from harness.figures import cost_to_go_panels
from harness.palette import LIGHT, Theme  # noqa: F401
from harness.render import (
    RenderConfig,
    WarehouseRenderer,
    episode_frames,
    hstack,
    save_gif,
    save_png,
    surf_to_array,
)
from taskforge.dsl import TaskSpec
from taskforge.verify.v2_oracle import cost_to_go, positional_cost_to_go
from taskforge.worlds.warehouse import apply_action, context_for, initial_state

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"


def hero_gif(
    spec: TaskSpec, plan, out: Path | None = None, theme: Theme = LIGHT, cell: int = 40, fps: int = 20
) -> Path:
    """The certificate plan executing, side panel visible, looping cleanly."""
    out = out or IMG / "hero.gif"
    frames, _ = episode_frames(
        spec,
        plan,
        RenderConfig(cell=cell, theme=theme),
        tween=3,
        hold_start=8,
        hold_end=24,
        optimal=len(plan),
        title=spec.task_id,
        subtitle=f"oracle certificate plan  ·  {len(plan)} steps, proven optimal",
    )
    return save_gif(frames, out, fps=fps)


def cost_to_go_figure(
    spec: TaskSpec, plan, out_name: str = "cost_to_go", theme: Theme = LIGHT, cell: int = 34
) -> Path | None:
    """Project the exact V* table onto grid positions at three points in the plan.

    The field is not static: it is defined over full symbolic states, so picking an item
    or packing a box changes what "distance to done" means from every cell. Showing three
    snapshots is what makes that legible -- a single frame looks like a distance transform
    and hides the whole point.
    """
    ctg = cost_to_go(spec, budget=400_000)
    if not ctg.exact:
        return None
    ctx = context_for(spec)

    # pick three informative moments: the start, just after the first pick, and just
    # after the first pack
    marks = [(0, initial_state(ctx))]
    state = initial_state(ctx)
    seen_pick = seen_pack = False
    for i, action in enumerate(plan):
        nxt = apply_action(ctx, state, action)
        if nxt is None:
            break
        state = nxt
        if action[0] == "pick" and not seen_pick:
            marks.append((i + 1, state))
            seen_pick = True
        elif action[0] == "pack" and not seen_pack:
            marks.append((i + 1, state))
            seen_pack = True
        if len(marks) == 3:
            break

    # the optimal path, for tracing over the field
    path = [initial_state(ctx)[0]]
    s = initial_state(ctx)
    for action in plan:
        n = apply_action(ctx, s, action)
        if n is None:
            break
        if n[0] != s[0]:
            path.append(n[0])
        s = n

    r = WarehouseRenderer(
        spec, RenderConfig(cell=cell, theme=theme, show_panel=False, header_h=0, margin=10)
    )
    snapshots = []
    for step, st in marks:
        heat = positional_cost_to_go(spec, ctg, st)
        if not heat:
            continue
        surf = r.render_frame(st, step=step, heat=heat, path=path if step == 0 else None)
        held = sum(st[1])
        packed = sum(st[2])
        if step == 0:
            title = f"step 0  ·  holding nothing  ·  V* = {ctg.value(st)}"
        else:
            title = (
                f"step {step}  ·  holding {held}, packed {packed}  ·  V* = {ctg.value(st)}"
            )
        snapshots.append((title, surf_to_array(surf)))
    if not snapshots:
        return None
    path_out = cost_to_go_panels(spec, snapshots, theme)
    return path_out.with_name(f"{out_name}{'' if theme.name == 'light' else '_dark'}.png")


def three_way_gif(
    spec_task,
    agents_frames: list[tuple[str, list]],
    out: Path | None = None,
    fps: int = 20,
) -> Path:
    out = out or IMG / "three_way.gif"
    stacked = hstack([f for _, f in agents_frames])
    return save_gif(stacked, out, fps=fps)


def render_agent_episode(
    task, agent, theme: Theme = LIGHT, cell: int = 30, label: str = "", max_steps: int | None = None
):
    """Run an agent on a task and render its trajectory, matching the hero styling."""
    from harness.agents import run_episode
    from taskforge.envs import ACTIONS, WarehouseEnv

    env = WarehouseEnv([task], obs_mode="vector", seed=0)
    env.reset(options={"task_index": 0})
    agent.reset(env)
    limit = max_steps or task.spec.constraints.step_budget
    actions = []
    for _ in range(limit):
        a = agent.act(env)
        prev = env.state
        _, _, term, trunc, _ = env.step(a)
        if env.state != prev:
            actions.append(ACTIONS[a])
        if term or trunc:
            break
    solved = env.task and env.state[3] == env.ctx.all_dispatched and env.state[5] == 0
    ruined = env.state[5] != 0
    status = "solved" if solved else ("RUINED a box" if ruined else "did not finish")
    frames, _ = episode_frames(
        task.spec,
        actions,
        RenderConfig(cell=cell, theme=theme),
        tween=2,
        hold_start=6,
        hold_end=26,
        optimal=task.certificate.cost,
        title=label,
        subtitle=f"{len(actions)} steps  ·  {status}",
    )
    _ = run_episode
    return frames, len(actions), status


def build_three_way(task, model_path: str | None = None, theme: Theme = LIGHT, cell: int = 28):
    """Oracle vs trained PPO vs LLM/scripted agent, same task, side by side."""
    from harness.agents import LLMAgent, OracleAgent, PolicyAgent

    panes: list[tuple[str, list]] = []
    frames, n, _ = render_agent_episode(task, OracleAgent(), theme, cell, "Oracle (optimal)")
    panes.append(("oracle", frames))

    if model_path and Path(model_path).exists():
        from sb3_contrib import MaskablePPO

        try:
            model = MaskablePPO.load(model_path, device="cpu")
        except Exception as e:
            # A checkpoint left over from a different algorithm should degrade the
            # figure to two panes, not abort the whole visual build.
            print(f"  ! skipping PPO pane: {model_path} did not load ({e})")
        else:
            f, n, _ = render_agent_episode(task, PolicyAgent(model), theme, cell, "PPO agent")
            panes.append(("ppo", f))

    llm = LLMAgent()
    label = "LLM agent" if llm.available() else "Scripted agent (no API key)"
    f, n, _ = render_agent_episode(task, llm, theme, cell, label)
    panes.append(("llm", f))
    return panes


def contact_sheet(specs, out: Path | None = None, theme: Theme = LIGHT, cell: int = 22) -> Path:
    """A grid of initial states across the difficulty ladder -- useful for the README's
    'what do these worlds look like' question."""
    out = out or IMG / "ladder.png"
    arrs = []
    for spec in specs:
        r = WarehouseRenderer(
            spec, RenderConfig(cell=cell, theme=theme, show_panel=False, header_h=22, margin=8)
        )
        ctx = context_for(spec)
        r.cfg.title = spec.task_id
        arrs.append(surf_to_array(r.render_frame(initial_state(ctx))))
    h = max(a.shape[0] for a in arrs)
    w = max(a.shape[1] for a in arrs)
    padded = []
    for a in arrs:
        canvas = np.full((h, w, 3), a[0, 0], dtype=a.dtype)
        canvas[: a.shape[0], : a.shape[1]] = a
        padded.append(canvas)
    row = np.hstack(padded)
    import imageio.v2 as imageio

    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out, row)
    return out


__all__ = [
    "build_three_way",
    "contact_sheet",
    "cost_to_go_figure",
    "hero_gif",
    "render_agent_episode",
    "save_png",
    "three_way_gif",
]
