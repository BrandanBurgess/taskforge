"""PPO training on verified tasks. CPU, small MLP, multi-seed.

    python train.py --steps 200000 --seeds 3 --reward sparse
    python train.py --steps 200000 --seeds 3 --reward shaped

Two reward regimes, reported side by side and labelled honestly:

``sparse``
    Goal-only. A terminal bonus for satisfying the goal predicate, a small per-step
    cost, a penalty for ruining a box. No shaping, no cost-to-go, no state leakage. This
    is the honest headline -- it is what "can an agent learn this task" actually means.

``shaped``
    Adds potential-based shaping with ``Phi(s) = -V*(s)`` from the oracle. Policy-
    invariant by Ng, Harada & Russell (1999), and **easy by construction**: following
    the shaping gradient *is* the optimal policy, because Phi is the exact optimal
    cost-to-go. It proves the plumbing -- oracle to potential to learner -- works. It
    does not prove the task is hard, and it is never quoted as the headline result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

# Training episodes are truncated at this multiple of the certificate length; see
# WarehouseEnv.episode_limit. Evaluation always uses the full spec step budget.
TRAIN_HORIZON_FACTOR = 3.0


def build_env(
    tasks, reward: str, seed: int, max_steps: int | None = None,
    horizon_factor: float | None = None,
):
    from taskforge.envs import WarehouseEnv

    return WarehouseEnv(
        tasks,
        obs_mode="vector",
        shaping=(reward == "shaped"),
        seed=seed,
        max_steps=max_steps,
        horizon_factor=horizon_factor,
    )


def build_masked_env(tasks, reward: str, seed: int, max_steps: int | None = None):
    """Same env, wrapped so MaskablePPO sees the action-validity mask.

    Masking is applied during *training*, not only at inference. Masking at inference
    only would be a train/test mismatch: the policy would spend its capacity learning to
    avoid the 20-odd illegal actions available at any moment, then be handed that for
    free at evaluation time, and the reported invalid-action rate would describe the
    wrapper rather than the policy.
    """
    from sb3_contrib.common.wrappers import ActionMasker

    env = build_env(tasks, reward, seed, max_steps, horizon_factor=TRAIN_HORIZON_FACTOR)
    return ActionMasker(env, lambda e: e.unwrapped.valid_action_mask())


def evaluate_policy(model, tasks, episodes_per_task: int = 1, mask: bool = True) -> dict:
    from harness.agents import PolicyAgent, run_episode

    env = build_env(tasks, "sparse", seed=1234)
    agent = PolicyAgent(model, mask=mask)
    results = []
    for i in range(len(tasks)):
        for _ in range(episodes_per_task):
            results.append(run_episode(env, agent, i))
    return summarize(results)


def summarize(results) -> dict:
    solved = [r for r in results if r.solved]
    ratios = [r.length_ratio for r in solved if r.length_ratio]
    by_diff: dict[int, list] = {}
    for r in results:
        by_diff.setdefault(r.difficulty, []).append(r)
    return {
        "episodes": len(results),
        "success_rate": round(len(solved) / max(1, len(results)), 4),
        "mean_length_ratio": round(float(np.mean(ratios)), 4) if ratios else None,
        "ruin_rate": round(sum(r.ruined for r in results) / max(1, len(results)), 4),
        "invalid_action_rate": round(
            sum(r.invalid_actions for r in results) / max(1, sum(r.steps for r in results)), 4
        ),
        "by_difficulty": {
            str(d): {
                "n": len(rs),
                "success_rate": round(sum(r.solved for r in rs) / len(rs), 4),
                "ruin_rate": round(sum(r.ruined for r in rs) / len(rs), 4),
                "mean_length_ratio": (
                    round(float(np.mean([r.length_ratio for r in rs if r.length_ratio])), 4)
                    if any(r.length_ratio for r in rs)
                    else None
                ),
            }
            for d, rs in sorted(by_diff.items())
        },
    }


class CurveCallback:
    """Periodically evaluates success rate so learning curves have real y-values rather
    than reward proxies. Reward is not comparable across the two regimes; success rate
    is."""

    def __init__(self, tasks, every: int, mask: bool = True):
        self.tasks = tasks
        self.every = every
        self.mask = mask
        self.points: list[tuple[int, float]] = []

    def make(self):
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class _CB(BaseCallback):
            def __init__(self):
                super().__init__()
                self.last = 0

            def _on_step(self) -> bool:
                if self.num_timesteps - self.last >= outer.every:
                    self.last = self.num_timesteps
                    stats = evaluate_policy(self.model, outer.tasks, 1, outer.mask)
                    outer.points.append((self.num_timesteps, stats["success_rate"]))
                return True

        return _CB()


def train_one(tasks, eval_tasks, reward: str, seed: int, steps: int, eval_every: int) -> dict:
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(2)  # 8 GB M1: more threads is slower, not faster
    env = build_masked_env(tasks, reward, seed=seed)
    curve = CurveCallback(eval_tasks, eval_every)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        seed=seed,
        device="cpu",
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.003,
        n_epochs=10,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=0,
    )
    t0 = time.time()
    model.learn(total_timesteps=steps, callback=curve.make(), progress_bar=False)
    elapsed = time.time() - t0
    final = evaluate_policy(model, eval_tasks, episodes_per_task=1)
    return {
        "reward": reward,
        "seed": seed,
        "steps": steps,
        "seconds": round(elapsed, 1),
        "curve": curve.points,
        "final": final,
        "_model": model,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--reward", choices=["sparse", "shaped", "both"], default="both")
    ap.add_argument("--train-buckets", default="1,2,3")
    ap.add_argument("--eval-buckets", default="1,2,3")
    ap.add_argument("--holdout-buckets", default="4,5")
    ap.add_argument("--eval-every", type=int, default=10_000)
    ap.add_argument("--specs", default=str(ROOT / "specs"))
    ap.add_argument("--out", default=str(ROOT / "results" / "rl_training.json"))
    ap.add_argument("--save-model", default="")
    args = ap.parse_args()

    from taskforge.verify import load_specs

    def parse(s):
        return tuple(int(x) for x in s.split(",") if x.strip())

    train_tasks = load_specs(args.specs, buckets=parse(args.train_buckets))
    eval_tasks = load_specs(args.specs, buckets=parse(args.eval_buckets))
    holdout = load_specs(args.specs, buckets=parse(args.holdout_buckets))
    if not train_tasks:
        raise SystemExit("no training specs; run scripts/build_specs.py first")
    print(f"train {len(train_tasks)} tasks | eval {len(eval_tasks)} | holdout {len(holdout)}")

    regimes = ["sparse", "shaped"] if args.reward == "both" else [args.reward]
    runs = []
    best_model = None
    for reward in regimes:
        for seed in range(args.seeds):
            print(f"  training reward={reward} seed={seed} ...", flush=True)
            r = train_one(train_tasks, eval_tasks, reward, seed, args.steps, args.eval_every)
            model = r.pop("_model")
            if reward == "shaped" and seed == 0:
                best_model = model
            if holdout:
                r["holdout"] = evaluate_policy(model, holdout, 1)
            print(
                f"    success {r['final']['success_rate']:.2f} "
                f"ratio {r['final']['mean_length_ratio']} "
                f"({r['seconds']}s)",
                flush=True,
            )
            runs.append(r)
            if args.save_model and reward == "shaped" and seed == 0:
                Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
                model.save(args.save_model)

    # -- baselines on the same tasks ------------------------------------------------
    from harness.agents import GreedyAgent, RandomAgent, run_episode

    baselines = {}
    for name, factory in (("random", RandomAgent), ("greedy", GreedyAgent)):
        env = build_env(eval_tasks, "sparse", seed=7)
        agent = factory(seed=7)
        res = [run_episode(env, agent, i) for i in range(len(eval_tasks))]
        baselines[name] = summarize(res)
        print(f"  baseline {name}: success {baselines[name]['success_rate']:.2f}")
        if holdout:
            envh = build_env(holdout, "sparse", seed=7)
            resh = [run_episode(envh, factory(seed=7), i) for i in range(len(holdout))]
            baselines[name]["holdout"] = summarize(resh)

    payload = {
        "config": {
            "steps": args.steps,
            "seeds": args.seeds,
            "train_buckets": args.train_buckets,
            "eval_buckets": args.eval_buckets,
            "holdout_buckets": args.holdout_buckets,
            "n_train_tasks": len(train_tasks),
            "n_eval_tasks": len(eval_tasks),
            "n_holdout_tasks": len(holdout),
            "device": "cpu",
            "algo": "PPO",
            "net_arch": [128, 128],
        },
        "runs": runs,
        "baselines": baselines,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")
    _ = best_model


if __name__ == "__main__":
    main()
