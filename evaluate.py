"""Agent evaluation, graded by the oracle.

Every agent -- random, greedy, PPO, LLM -- is scored the same way, against the same
goal predicate the verifier used to certify the task:

* **pass@1** -- did the goal predicate hold when the episode ended?
* **plan-length ratio** -- steps taken / oracle-optimal. Only defined on solved
  episodes; a ratio of 1.0 means the agent matched the certificate.
* **invalid-action rate** -- fraction of steps that the executor rejected.
* **irreversible-failure rate** -- fraction of episodes that ended in a ruined box.
  The oracle makes this exactly measurable: "unrecoverable" is not a heuristic judgement
  here, it is a proven property of the state.

All of it buckets by difficulty. With no API key the LLM slot runs the scripted agent
instead, and the output records ``used_llm: false`` so the distinction survives into the
results file.

    python evaluate.py --model checkpoints/ppo_shaped.zip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def summarize(results, label: str) -> dict:
    by: dict[int, list] = {}
    for r in results:
        by.setdefault(r.difficulty, []).append(r)

    def block(rs):
        s = [r for r in rs if r.solved]
        rr = [r.length_ratio for r in s if r.length_ratio]
        return {
            "n": len(rs),
            "pass_at_1": round(len(s) / max(1, len(rs)), 4),
            "mean_length_ratio": round(float(np.mean(rr)), 3) if rr else None,
            "median_length_ratio": round(float(np.median(rr)), 3) if rr else None,
            "irreversible_rate": round(sum(r.ruined for r in rs) / max(1, len(rs)), 4),
            "invalid_action_rate": round(
                sum(r.invalid_actions for r in rs) / max(1, sum(r.steps for r in rs)), 4
            ),
            "mean_steps": round(float(np.mean([r.steps for r in rs])), 1),
        }

    return {
        "agent": label,
        "overall": block(results),
        "by_difficulty": {str(d): block(rs) for d, rs in sorted(by.items())},
        "_raw": [
            {
                "task_id": r.task_id,
                "difficulty": r.difficulty,
                "optimal": r.optimal,
                "steps": r.steps,
                "solved": r.solved,
                "ruined": r.ruined,
            }
            for r in results
        ],
    }


def run_agent(tasks, agent, episodes: int = 1, seed: int = 99):
    from harness.agents import run_episode
    from train import build_env

    env = build_env(tasks, "sparse", seed=seed)
    out = []
    for i in range(len(tasks)):
        for _ in range(episodes):
            out.append(run_episode(env, agent, i))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default=str(ROOT / "specs"))
    ap.add_argument("--buckets", default="1,2,3,4,5")
    ap.add_argument("--model", default="", help="path to a saved SB3 PPO zip")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--llm-tasks", type=int, default=0, help="cap LLM episodes (API cost)")
    ap.add_argument("--out", default=str(ROOT / "results" / "agent_eval.json"))
    args = ap.parse_args()

    from harness.agents import GreedyAgent, LLMAgent, OracleAgent, PolicyAgent, RandomAgent
    from taskforge.verify import load_specs

    buckets = tuple(int(x) for x in args.buckets.split(","))
    tasks = load_specs(args.specs, buckets=buckets)
    if not tasks:
        raise SystemExit("no specs found; run scripts/build_specs.py first")
    print(f"evaluating on {len(tasks)} verified tasks (buckets {buckets})")

    reports = {}

    # The oracle replaying its own certificate is a sanity row, not a result: it must be
    # 100% pass@1 at ratio exactly 1.000. If it is not, the executor and the certificate
    # have drifted and every other number on this table is suspect.
    res = run_agent(tasks, OracleAgent(), 1)
    reports["oracle"] = summarize(res, "oracle")
    assert reports["oracle"]["overall"]["pass_at_1"] == 1.0, (
        "oracle failed to replay its own certificate: V2/V3 drift"
    )

    for name, agent in (("random", RandomAgent(seed=99)), ("greedy", GreedyAgent(seed=99))):
        res = run_agent(tasks, agent, args.episodes)
        reports[name] = summarize(res, name)
        print(f"  {name:8s} pass@1 {reports[name]['overall']['pass_at_1']:.3f}")

    if args.model and Path(args.model).exists():
        from stable_baselines3 import PPO

        model = PPO.load(args.model, device="cpu")
        res = run_agent(tasks, PolicyAgent(model, name="ppo"), args.episodes)
        reports["ppo"] = summarize(res, "ppo")
        print(f"  {'ppo':8s} pass@1 {reports['ppo']['overall']['pass_at_1']:.3f}")

    llm = LLMAgent()
    llm_tasks = tasks if not args.llm_tasks else tasks[: args.llm_tasks]
    res = run_agent(llm_tasks, llm, 1)
    label = "llm" if llm.used_llm else "scripted (LLM fallback, no API key)"
    reports["llm"] = summarize(res, label)
    reports["llm"]["used_llm"] = llm.used_llm
    reports["llm"]["api_key_present"] = llm.available()
    reports["llm"]["llm_calls"] = llm.calls
    print(f"  {'llm':8s} pass@1 {reports['llm']['overall']['pass_at_1']:.3f}  used_llm={llm.used_llm}")

    payload = {
        "n_tasks": len(tasks),
        "buckets": list(buckets),
        "episodes_per_task": args.episodes,
        "agents": reports,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")

    # -- difficulty calibration ---------------------------------------------------------
    calib = calibrate(tasks, reports)
    (Path(args.out).parent / "difficulty_calibration.json").write_text(
        json.dumps(calib, indent=2) + "\n"
    )
    print(f"difficulty label vs measured success: r = {calib['pearson_r']}")


def calibrate(tasks, reports) -> dict:
    """Correlate the oracle-derived difficulty label with measured agent success.

    The label is only worth shipping if it predicts something. We pool the non-oracle
    agents' per-task outcomes, average them per task, and correlate against the label's
    continuous score.
    """
    by_task: dict[str, list[bool]] = {}
    for name, rep in reports.items():
        if name == "oracle":
            continue
        for row in rep["_raw"]:
            by_task.setdefault(row["task_id"], []).append(bool(row["solved"]))
    scores, rates = [], []
    per_bucket: dict[int, list[float]] = {}
    for t in tasks:
        outcomes = by_task.get(t.spec.task_id)
        if not outcomes:
            continue
        rate = sum(outcomes) / len(outcomes)
        scores.append(t.difficulty.score)
        rates.append(rate)
        per_bucket.setdefault(t.difficulty.bucket, []).append(rate)
    r = float(np.corrcoef(scores, rates)[0, 1]) if len(scores) > 2 else float("nan")
    return {
        "n_tasks": len(scores),
        "pearson_r": round(r, 4),
        "note": (
            "Correlation between the oracle-derived difficulty score "
            "(plan length x SKU scatter x branching) and pooled agent success rate. "
            "Negative r is the expected direction: harder tasks, fewer solves."
        ),
        "by_bucket": {
            str(b): {
                "n": len(v),
                "mean_success": round(float(np.mean(v)), 4),
            }
            for b, v in sorted(per_bucket.items())
        },
        "points": [
            {"score": s, "success": round(rt, 4)} for s, rt in zip(scores, rates, strict=True)
        ],
    }


if __name__ == "__main__":
    main()
