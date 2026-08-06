"""Render the RL results table from results/rl_training.json into the README.

Keeps the README's headline RL numbers mechanically tied to the committed run rather
than retyped by hand. Replaces the `<!--RL_TABLE-->` marker, or the block it previously
generated, in place.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
START = "<!--RL_TABLE-->"
END = "<!--/RL_TABLE-->"


def fmt(v, nd=2, suffix=""):
    return "—" if v is None else f"{v:.{nd}f}{suffix}"


def render(data: dict) -> str:
    cfg = data["config"]
    regimes: dict[str, list] = {}
    for run in data["runs"]:
        regimes.setdefault(run["reward"], []).append(run)

    lines = [
        f"Trained on difficulty buckets {cfg['train_buckets']} "
        f"({cfg['n_train_tasks']} verified tasks), {cfg['seeds']} seeds x "
        f"{cfg['steps']:,} steps, MaskablePPO on CPU with a {cfg['net_arch']} MLP.",
        "",
        "| agent | success rate (mean ± spread over seeds) | per-seed | steps / oracle-optimal |",
        "|---|---|---|---|",
    ]

    label = {"sparse": "**PPO, sparse reward** *(the honest headline)*",
             "shaped": "**PPO + oracle shaping** *(easy by construction)*"}
    for reward in ("sparse", "shaped"):
        runs = regimes.get(reward)
        if not runs:
            continue
        succ = [r["final"]["success_rate"] for r in runs]
        ratios = [r["final"]["mean_length_ratio"] for r in runs if r["final"]["mean_length_ratio"]]
        per_seed = " / ".join(f"{s:.2f}" for s in succ)
        spread = f"{np.mean(succ):.2f} ± {(max(succ) - min(succ)) / 2:.2f}"
        ratio = f"{np.mean(ratios):.2f}×" if ratios else "—"
        lines.append(f"| {label[reward]} | {spread} | {per_seed} | {ratio} |")

    for name in ("greedy", "random"):
        b = data["baselines"].get(name)
        if not b:
            continue
        lines.append(
            f"| {name} baseline | {b['success_rate']:.2f} | — | "
            f"{fmt(b['mean_length_ratio'], 2, '×')} |"
        )
    lines.append("| oracle (certificate replay) | 1.00 | — | 1.00× |")
    return "\n".join(lines)


def main() -> None:
    data = json.loads((ROOT / "results" / "rl_training.json").read_text())
    block = f"{START}\n\n{render(data)}\n\n{END}"
    readme = ROOT / "README.md"
    text = readme.read_text()
    if START in text and END in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), block, text, flags=re.S)
    else:
        text = text.replace(START, block)
    readme.write_text(text)
    print(render(data))


if __name__ == "__main__":
    main()
