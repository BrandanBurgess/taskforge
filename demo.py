"""One-command demo. No API key, no network, no GPU.

    python demo.py            # full demo, ~2-3 minutes on an M1
    python demo.py --smoke    # fast path used by CI

Generates tasks, verifies them through all three stages, prints the certificate plan,
renders the hero GIF and the cost-to-go field, trains a short PPO run, evaluates every
agent against the oracle's goal predicate, writes every figure, and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parent
IMG = ROOT / "docs" / "img"
RESULTS = ROOT / "results"

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


def rule(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}\n{DIM}{'-' * min(78, max(30, len(title) + 12))}{RESET}")


def fmt_plan(spec, plan, per_line: int = 6) -> str:
    from harness.render import describe

    steps = [f"{i + 1:>3}. {describe(spec, a)}" for i, a in enumerate(plan)]
    lines = []
    for i in range(0, len(steps), per_line):
        lines.append("   " + "".join(f"{s:<26}" for s in steps[i : i + per_line]))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="fast path for CI")
    ap.add_argument("--steps", type=int, default=60_000, help="PPO steps for the demo run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    smoke = args.smoke
    IMG.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    from harness import figures, visuals
    from harness.agents import GreedyAgent, LLMAgent, OracleAgent, RandomAgent, run_episode
    from harness.replay import build_replay
    from taskforge.verify import load_specs, solve, verify
    from taskforge.verify.pipeline import accepted_task
    from taskforge.worlds.warehouse import generate

    # ---------------------------------------------------------------- 1. generate + verify
    rule("1. Generate and verify")
    difficulty = 2 if smoke else 3
    spec = generate(args.seed, difficulty)
    print(f"generated  {spec.task_id}  (world={spec.world}, seed={spec.seed})")
    p = spec.payload
    print(f"           {p['width']}x{p['height']} grid, {len(p['skus'])} SKUs, "
          f"{len(p['orders'])} orders, capacity {spec.constraints.capacity}")

    t0 = time.time()
    outcome = verify(spec)
    dt = time.time() - t0
    for stage in ("V1", "V2", "V3"):
        res = outcome.stages.get(stage)
        if res is None:
            continue
        mark = "PASS" if res.ok else "FAIL"
        extra = ""
        if stage == "V2" and res.ok:
            extra = f"  (optimal {res.detail['cost']} steps, {res.detail['expanded']} nodes expanded)"
        if stage == "V3" and res.ok:
            extra = f"  (replayed {res.detail['replay_length']} steps, goal reached)"
        print(f"  {stage} {mark}{extra}")
    if not outcome.accepted:
        raise SystemExit(f"demo task rejected at {outcome.reject_stage}: {outcome.reject_reason}")
    task = accepted_task(spec, outcome)
    cert = outcome.certificate
    print(f"ACCEPTED in {dt:.2f}s   difficulty label: bucket {outcome.difficulty.bucket} "
          f"(score {outcome.difficulty.score})")

    # ---------------------------------------------------------------- 2. certificate
    rule("2. The certificate plan (this is the proof)")
    print(fmt_plan(spec, cert.as_actions()))
    print(f"\n   {len(cert.plan)} actions, proven optimal. Branching factor "
          f"{cert.branching}, {cert.states_seen} distinct states reached.")

    # ---------------------------------------------------------------- 3. conservative rejection
    rule("3. Conservative rejection")
    tight = solve(spec, step_budget=max(1, cert.cost - 1))
    print(f"   same task, step budget {cert.cost - 1} (one below optimal): "
          f"{tight.status.value}")
    print("   the oracle answers 'solvable' or 'not proven solvable'. Never 'probably'.")

    # ---------------------------------------------------------------- 4. visuals
    rule("4. Render")
    t0 = time.time()
    hero = visuals.hero_gif(spec, cert.as_actions(), IMG / "hero.gif",
                            cell=34 if smoke else 40, fps=20)
    print(f"   hero GIF            {hero.relative_to(ROOT)}  "
          f"({hero.stat().st_size / 1e6:.2f} MB)")
    ctg_path = visuals.cost_to_go_figure(spec, cert.as_actions())
    if ctg_path:
        print(f"   cost-to-go field    {Path(ctg_path).relative_to(ROOT)}")
    else:
        print("   cost-to-go field    skipped (state space too large to enumerate exactly)")
    replay = build_replay(spec, cert.as_actions(), ROOT / "docs" / "replay.html",
                          optimal=cert.cost, title=spec.task_id)
    print(f"   HTML replay         {replay.relative_to(ROOT)}  "
          f"({replay.stat().st_size / 1e6:.2f} MB)")
    print(f"   rendered in {time.time() - t0:.1f}s")

    # ---------------------------------------------------------------- 5. train
    tasks = load_specs(ROOT / "specs", buckets=(1, 2, 3)) or [task]
    model = None
    if not args.skip_train:
        rule("5. Short PPO run (CPU)")
        steps = 8_000 if smoke else args.steps
        import torch
        from sb3_contrib import MaskablePPO

        from train import build_masked_env, evaluate_policy

        torch.set_num_threads(2)
        t0 = time.time()
        env = build_masked_env(tasks, "shaped", seed=args.seed)
        model = MaskablePPO(
            "MlpPolicy", env, seed=args.seed, device="cpu", n_steps=1024,
            batch_size=256, ent_coef=0.003, n_epochs=10,
            policy_kwargs={"net_arch": [64, 64]}, verbose=0,
        )
        model.learn(total_timesteps=steps, progress_bar=False)
        stats = evaluate_policy(model, tasks, 1)
        print(f"   {steps} steps in {time.time() - t0:.0f}s on {len(tasks)} verified tasks")
        print(f"   success {stats['success_rate']:.2f}   "
              f"plan-length ratio {stats['mean_length_ratio']}   "
              f"ruin rate {stats['ruin_rate']:.2f}")
        print(f"   {DIM}(the committed results in results/ come from a longer multi-seed "
              f"run; see README){RESET}")

    # ---------------------------------------------------------------- 6. evaluate
    rule("6. Evaluate every agent against the oracle's goal predicate")
    from harness.agents import PolicyAgent

    eval_tasks = tasks[: 8 if smoke else len(tasks)]
    agents = [("oracle", OracleAgent()), ("greedy", GreedyAgent(seed=7)),
              ("random", RandomAgent(seed=7))]
    if model is not None:
        agents.insert(1, ("ppo (demo run)", PolicyAgent(model)))
    llm = LLMAgent()
    agents.append(("scripted (LLM fallback)" if not llm.available() else "llm", llm))

    from train import build_env

    env = build_env(eval_tasks, "sparse", seed=7)
    rows = []
    for name, agent in agents:
        res = [run_episode(env, agent, i) for i in range(len(eval_tasks))]
        solved = [r for r in res if r.solved]
        ratios = [r.length_ratio for r in solved if r.length_ratio]
        rows.append((
            name,
            len(solved) / len(res),
            sum(ratios) / len(ratios) if ratios else None,
            sum(r.ruined for r in res) / len(res),
        ))

    print(f"   {'agent':<26}{'pass@1':>9}{'steps/optimal':>16}{'ruined a box':>15}")
    print(f"   {DIM}{'-' * 64}{RESET}")
    for name, pass1, ratio, ruin in rows:
        rs = f"{ratio:.2f}x" if ratio else "-"
        print(f"   {name:<26}{pass1:>9.2f}{rs:>16}{ruin:>15.2f}")
    print(f"\n   {DIM}LLM agent used: {llm.used_llm}"
          f"{' (no ANTHROPIC_API_KEY - scripted fallback ran instead)' if not llm.used_llm else ''}"
          f"{RESET}")

    # ---------------------------------------------------------------- 7. figures
    rule("7. Figures")
    made = figures.build_all(RESULTS)
    for m in made:
        print(f"   {Path(m).relative_to(ROOT)}")
    if not made:
        print("   (no results/*.json yet - run scripts/build_specs.py, train.py, evaluate.py)")

    # ---------------------------------------------------------------- summary
    rule("Summary")
    funnel_path = RESULTS / "verification_funnel.json"
    if funnel_path.exists():
        f = json.loads(funnel_path.read_text())
        for arm in f["arms"]:
            print(f"   {arm['arm']:<12} accepted {arm['accepted']:>4}/{arm['attempts']:<4} "
                  f"({arm['accept_rate'] * 100:.1f}%)   "
                  f"V2/V3 agreement {arm['v2_v3_agreement_rate'] * 100:.2f}%")
    print(f"   specs committed: {len(list((ROOT / 'specs').glob('*.json')))}")
    print(f"\n   total {time.time() - t_start:.0f}s   no API key required.")


if __name__ == "__main__":
    main()
