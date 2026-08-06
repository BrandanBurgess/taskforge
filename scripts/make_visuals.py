"""Render the artifacts that need the simulator (and, for the comparison, a model).

    python scripts/make_visuals.py --model checkpoints/ppo_shaped.zip

Writes the hero GIF, the cost-to-go field (light + dark), the three-way agent
comparison GIF, the difficulty-ladder contact sheet, and the HTML replay viewer.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "checkpoints" / "ppo_shaped.zip"))
    ap.add_argument("--hero-difficulty", type=int, default=3)
    ap.add_argument("--hero-seed", type=int, default=0)
    ap.add_argument("--comparison-bucket", type=int, default=2)
    args = ap.parse_args()

    from harness import visuals
    from harness.palette import DARK
    from harness.replay import build_replay
    from taskforge.verify import load_specs, verify
    from taskforge.worlds.warehouse import generate

    spec = generate(args.hero_seed, args.hero_difficulty)
    outcome = verify(spec)
    if not outcome.accepted:
        raise SystemExit(f"hero task rejected: {outcome.reject_reason}")
    plan = outcome.certificate.as_actions()

    hero = visuals.hero_gif(spec, plan, IMG / "hero.gif", cell=40, fps=20)
    print(f"hero.gif              {hero.stat().st_size / 1e6:.2f} MB")

    for theme in (None, DARK):
        p = visuals.cost_to_go_figure(
            spec, plan, theme=theme if theme else visuals.LIGHT
        )
        print(f"cost_to_go            {p}")

    replay = build_replay(
        spec, plan, ROOT / "docs" / "replay.html", optimal=outcome.certificate.cost,
        title=spec.task_id,
    )
    print(f"replay.html           {replay.stat().st_size / 1e6:.2f} MB")

    ladder = visuals.contact_sheet([generate(0, d) for d in range(1, 6)])
    print(f"ladder.png            {ladder}")

    # -- three-way comparison ---------------------------------------------------------
    tasks = load_specs(ROOT / "specs", buckets=(args.comparison_bucket,))
    if tasks:
        # The most demanding task in the bucket the policy was actually trained on:
        # small enough that PPO still performs, large enough to be worth watching.
        task = max(tasks, key=lambda t: t.certificate.cost)
        model = args.model if Path(args.model).exists() else None
        if model is None:
            print("three_way.gif         (no model checkpoint; oracle + scripted only)")
        panes = visuals.build_three_way(task, model, cell=26)
        out = visuals.three_way_gif(task, panes, IMG / "three_way.gif", fps=18)
        print(f"three_way.gif         {out.stat().st_size / 1e6:.2f} MB "
              f"({len(panes)} panes, task {task.spec.task_id})")


if __name__ == "__main__":
    main()
