"""Every figure in the README, drawn from committed results JSON.

One palette, one set of chrome rules, light and dark. Conventions applied throughout:

* Categorical hue = identity (an agent keeps its slot in every chart it appears in).
  Sequential single-hue ramp = magnitude. Status tokens = state, never a series.
* Hairline recessive grid, no dashed rules, thin marks, generous padding.
* A legend whenever two or more series are present, plus selective direct labels -- never
  a number on every point -- so identity is never carried by colour alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from harness.palette import (  # noqa: E402
    AGENT_LABEL,
    DARK,
    LIGHT,
    Theme,
    agent_color,
    apply_matplotlib,
    mix,
)

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"


def ramp_cmap(theme: Theme, name: str = "tf") -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, list(theme.ramp), N=256)


def _save(fig, name: str, theme: Theme) -> Path:
    IMG.mkdir(parents=True, exist_ok=True)
    suffix = "" if theme.name == "light" else "_dark"
    path = IMG / f"{name}{suffix}.png"
    fig.savefig(path, facecolor=theme.surface)
    plt.close(fig)
    return path


def _footnote(ax, text: str, theme: Theme) -> None:
    ax.figure.text(
        0.0, -0.02, text, ha="left", va="top", fontsize=8, color=theme.ink_muted, wrap=True
    )


# --------------------------------------------------------------------------------------
# 1. Verification funnel
# --------------------------------------------------------------------------------------


def funnel(results: dict, theme: Theme = LIGHT) -> Path:
    """Stage-by-stage survival with rejection reasons broken out.

    An ordinal ramp (not categorical) because the stages are ordered, and it starts no
    lighter than the ramp step that still clears contrast on the surface.
    """
    apply_matplotlib(theme)
    arms = results["arms"]
    fig, axes = plt.subplots(
        1, len(arms), figsize=(5.6 * len(arms), 4.4), sharey=False
    )
    if len(arms) == 1:
        axes = [axes]

    for ax, arm in zip(axes, arms, strict=True):
        stages = [
            ("generated", arm["attempts"]),
            ("V1 schema", arm["v1_pass"]),
            ("V2 oracle", arm["v2_pass"]),
            ("V3 replay", arm["v3_pass"]),
            ("accepted", arm["accepted"]),
        ]
        labels = [s[0] for s in stages]
        vals = [s[1] for s in stages]
        # ordinal ramp: light -> dark across ordered stages, starting past the
        # contrast floor
        idxs = np.linspace(3, len(theme.ramp) - 3, len(stages)).round().astype(int)
        colors = [theme.ramp[i] for i in idxs]

        y = np.arange(len(stages))[::-1]
        ax.barh(y, vals, height=0.62, color=colors, zorder=3)
        ax.set_yticks(y, labels)
        ax.set_xlim(0, max(vals) * 1.28)
        ax.grid(axis="x", zorder=0)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)

        for yi, (_lab, v) in zip(y, stages, strict=True):
            pct = 100 * v / max(1, arm["attempts"])
            ax.text(
                v + max(vals) * 0.015,
                yi,
                f"{v}  ({pct:.0f}%)",
                va="center",
                fontsize=9,
                color=theme.ink_secondary,
            )
        # rejection callouts, attached to the stage that did the rejecting
        drop_rows = [
            ("V1 schema", arm["rejections"].get("V1", {})),
            ("V2 oracle", arm["rejections"].get("V2", {})),
            ("V3 replay", arm["rejections"].get("V3", {})),
        ]
        notes = []
        for stage, reasons in drop_rows:
            for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
                notes.append(f"{stage}: {reason} x{n}")
        ax.set_title(
            f"{arm['arm']} generator  -  {arm['accept_rate'] * 100:.1f}% accepted",
            color=theme.ink,
        )
        ax.set_xlabel("tasks")
        # Rejection callouts live below the plot, never inside it -- overlaying them on
        # the bars collided with the count labels.
        ax.text(
            0.0,
            -0.30,
            ("rejected -  " + " · ".join(notes)) if notes else "no rejections",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color=theme.ink_muted,
            linespacing=1.6,
        )
    fig.suptitle(
        "Verification funnel: a task is accepted only if all three stages pass",
        x=0.005, ha="left", fontsize=13, color=theme.ink, weight="bold",
    )
    return _save(fig, "funnel", theme)


# --------------------------------------------------------------------------------------
# 2. Learning curves
# --------------------------------------------------------------------------------------


def learning_curves(training: dict, theme: Theme = LIGHT) -> Path:
    """Multi-seed success rate with variance bands, plus baseline reference rules."""
    apply_matplotlib(theme)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    regimes = {}
    for run in training["runs"]:
        regimes.setdefault(run["reward"], []).append(run)

    for reward, runs in sorted(regimes.items()):
        curves = [r["curve"] for r in runs if r["curve"]]
        if not curves:
            continue
        n = min(len(c) for c in curves)
        xs = np.array([p[0] for p in curves[0][:n]])
        ys = np.array([[p[1] for p in c[:n]] for c in curves])
        mean, lo, hi = ys.mean(0), ys.min(0), ys.max(0)
        key = f"ppo_{reward}"
        col = agent_color(key, theme)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(xs, mean, color=col, zorder=3, label=AGENT_LABEL[key])
        ax.annotate(
            AGENT_LABEL[key],
            (xs[-1], mean[-1]),
            textcoords="offset points",
            xytext=(8, 0),
            fontsize=9,
            color=theme.ink_secondary,
            va="center",
        )

    for name in ("greedy", "random"):
        base = training["baselines"].get(name)
        if not base:
            continue
        ax.axhline(
            base["success_rate"], color=agent_color(name, theme), linewidth=1.4,
            zorder=1, alpha=0.9,
        )
        ax.annotate(
            f"{AGENT_LABEL[name]}  {base['success_rate']:.2f}",
            (0, base["success_rate"]),
            textcoords="offset points",
            xytext=(2, 5),
            fontsize=8.5,
            color=theme.ink_muted,
        )

    ax.set_xlabel("environment steps")
    ax.set_ylabel("success rate (goal predicate)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(left=0)
    n_seeds = training["config"]["seeds"]
    ax.set_title(
        f"PPO on verified tasks, {n_seeds} seeds (band = min-max across seeds)",
        color=theme.ink,
    )
    ax.legend(loc="upper left", ncols=2)
    _footnote(
        ax,
        "Shaped reward uses Phi = -V* from the oracle and is easy by construction: "
        "following the shaping gradient is the optimal policy. The sparse curve is the "
        "honest measure of learnability.",
        theme,
    )
    fig.subplots_adjust(right=0.82)
    return _save(fig, "learning_curves", theme)


# --------------------------------------------------------------------------------------
# 3. Difficulty calibration
# --------------------------------------------------------------------------------------


def difficulty_calibration(calib: dict, theme: Theme = LIGHT) -> Path:
    apply_matplotlib(theme)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2), width_ratios=[1.35, 1])

    xs = np.array([p["score"] for p in calib["points"]])
    ys = np.array([p["success"] for p in calib["points"]])
    ax.scatter(
        xs, ys, s=34, color=theme.series[0], alpha=0.75, linewidths=1.4,
        edgecolors=theme.surface, zorder=3,
    )
    if len(xs) > 2 and xs.std() > 0:
        k, b = np.polyfit(xs, ys, 1)
        gx = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(gx, k * gx + b, color=theme.ink, linewidth=1.4, alpha=0.65, zorder=4)
    ax.set_xlabel("oracle difficulty score  (plan length x SKU scatter x branching)")
    ax.set_ylabel("pooled agent success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("The label predicts agent success", color=theme.ink)
    ax.text(
        0.97, 0.94, f"r = {calib['pearson_r']}\nn = {calib['n_tasks']}",
        transform=ax.transAxes, ha="right", va="top", fontsize=11,
        color=theme.ink, linespacing=1.6,
    )

    buckets = sorted(calib["by_bucket"], key=int)
    vals = [calib["by_bucket"][b]["mean_success"] for b in buckets]
    ns = [calib["by_bucket"][b]["n"] for b in buckets]
    idxs = np.linspace(3, len(theme.ramp) - 3, len(buckets)).round().astype(int)
    ax2.bar(
        range(len(buckets)), vals, width=0.62,
        color=[theme.ramp[i] for i in idxs], zorder=3,
    )
    ax2.set_xticks(range(len(buckets)), [f"D{b}" for b in buckets])
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("mean success rate")
    ax2.set_title("by difficulty bucket", color=theme.ink)
    for i, (v, n) in enumerate(zip(vals, ns, strict=True)):
        ax2.text(
            i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9, color=theme.ink_secondary
        )
        ax2.text(i, 0.02, f"n={n}", ha="center", fontsize=8, color=theme.ink_muted)
    return _save(fig, "difficulty_calibration", theme)


# --------------------------------------------------------------------------------------
# 4. Agent evaluation
# --------------------------------------------------------------------------------------


def agent_eval(evaluation: dict, theme: Theme = LIGHT) -> Path:
    """pass@1 and plan-efficiency by difficulty, plus the irreversibility study."""
    apply_matplotlib(theme)
    agents = [a for a in ("oracle", "greedy", "ppo", "llm", "random") if a in evaluation["agents"]]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3))

    buckets = sorted(
        {b for a in agents for b in evaluation["agents"][a]["by_difficulty"]}, key=int
    )
    width = 0.8 / max(1, len(agents))

    def slot(a: str) -> str:
        return {"ppo": "ppo_shaped", "llm": "llm"}.get(a, a)

    def label(a: str) -> str:
        rep = evaluation["agents"][a]
        if a == "llm" and not rep.get("used_llm", False):
            return "scripted (LLM fallback)"
        return AGENT_LABEL.get(slot(a), a)

    # -- pass@1 --------------------------------------------------------------------
    ax = axes[0]
    for i, a in enumerate(agents):
        rep = evaluation["agents"][a]["by_difficulty"]
        vals = [rep.get(b, {}).get("pass_at_1", 0) for b in buckets]
        ax.bar(
            np.arange(len(buckets)) + i * width - 0.4 + width / 2,
            vals, width=width * 0.88, color=agent_color(slot(a), theme),
            label=label(a), zorder=3,
        )
    ax.set_xticks(range(len(buckets)), [f"D{b}" for b in buckets])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("pass@1")
    ax.set_title("Success by difficulty", color=theme.ink)
    # Legend below the plot: at D1 several agents are at 1.00 and an in-axes legend
    # sits straight on top of the bars.
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)

    # -- plan efficiency ------------------------------------------------------------
    ax = axes[1]
    for i, a in enumerate(agents):
        rep = evaluation["agents"][a]["by_difficulty"]
        vals = [rep.get(b, {}).get("mean_length_ratio") or np.nan for b in buckets]
        ax.bar(
            np.arange(len(buckets)) + i * width - 0.4 + width / 2,
            vals, width=width * 0.88, color=agent_color(slot(a), theme), zorder=3,
        )
    ax.axhline(1.0, color=theme.ink, linewidth=1.2, alpha=0.6, zorder=4)
    ax.annotate(
        "oracle-optimal = 1.0", (0.02, 1.0), xycoords=("axes fraction", "data"),
        textcoords="offset points", xytext=(0, 6), fontsize=8.5, color=theme.ink_muted,
    )
    # Clip the axis to the interesting band and label anything above it in place. The
    # random agent's ~12x on the easiest bucket would otherwise flatten every bar that
    # actually matters into an indistinguishable strip.
    finite = [
        v
        for a in agents
        for v in (
            evaluation["agents"][a]["by_difficulty"].get(b, {}).get("mean_length_ratio")
            for b in buckets
        )
        if v
    ]
    cap = max(2.2, min(3.5, (max(finite) if finite else 2.0)))
    ax.set_ylim(0, cap)
    for i, a in enumerate(agents):
        rep = evaluation["agents"][a]["by_difficulty"]
        for j, b in enumerate(buckets):
            v = rep.get(b, {}).get("mean_length_ratio")
            if v and v > cap:
                ax.annotate(
                    f"{v:.1f}x",
                    (j + i * width - 0.4 + width / 2, cap),
                    ha="center", va="top", fontsize=8, color=theme.surface,
                    textcoords="offset points", xytext=(0, -4), rotation=90,
                )
    ax.set_xticks(range(len(buckets)), [f"D{b}" for b in buckets])
    ax.set_ylabel("steps / oracle-optimal")
    ax.set_title("Plan efficiency (solved episodes only)", color=theme.ink)

    # -- irreversibility -------------------------------------------------------------
    ax = axes[2]
    names = [a for a in agents if a != "oracle"]
    vals = [evaluation["agents"][a]["overall"]["irreversible_rate"] for a in names]
    ax.bar(
        range(len(names)), vals, width=0.6,
        color=[agent_color(slot(a), theme) for a in names], zorder=3,
    )
    ax.set_xticks(range(len(names)), [label(a).split(" (")[0] for a in names],
                  rotation=18, ha="right")
    ax.set_ylabel("episodes ending in a ruined box")
    ax.set_title("Irreversible failures", color=theme.ink)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals + [0.01]) * 0.04, f"{v:.2f}", ha="center",
                fontsize=9, color=theme.ink_secondary)
    _footnote(
        ax,
        "'Unrecoverable' is proven, not estimated: a ruined box makes the goal "
        "predicate unreachable, so the oracle measures this exactly.",
        theme,
    )
    return _save(fig, "agent_eval", theme)


# --------------------------------------------------------------------------------------
# 5. MAP-Elites archive
# --------------------------------------------------------------------------------------


def map_elites(archive: dict, theme: Theme = LIGHT) -> Path:
    apply_matplotlib(theme)
    buckets = archive["grid"]["difficulty_buckets"]
    sigs = archive["grid"]["entity_signatures"]
    grid = np.full((len(sigs), len(buckets)), np.nan)
    depth = np.zeros_like(grid)
    for cell in archive["cells"]:
        r = sigs.index(cell["signature"])
        c = buckets.index(cell["difficulty"])
        grid[r, c] = cell["optimal_cost"]
        depth[r, c] = cell["depth"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6), width_ratios=[1.5, 1])
    cmap = ramp_cmap(theme)
    cmap.set_bad(mix(theme.surface, theme.ink, 0.05))
    im = ax.imshow(grid, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(buckets)), [f"D{b}" for b in buckets])
    ax.set_yticks(range(len(sigs)), sigs, fontsize=8)
    ax.set_xlabel("difficulty bucket")
    ax.set_ylabel("entity multiset signature")
    ax.grid(visible=False)
    for r in range(len(sigs)):
        for c in range(len(buckets)):
            if not np.isnan(grid[r, c]):
                span = float(np.nanmax(grid) - np.nanmin(grid))
                frac = (grid[r, c] - np.nanmin(grid)) / max(1e-9, span)
                ax.text(
                    c, r, f"{int(grid[r, c])}", ha="center", va="center", fontsize=8,
                    color=theme.surface if frac > 0.5 else theme.ink,
                )
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("optimal plan length of the cell elite", fontsize=9,
                 color=theme.ink_secondary)
    cb.outline.set_visible(False)
    ax.set_title(
        f"MAP-Elites archive: {archive['coverage_cells']} cells covered "
        f"({archive['accepted']}/{archive['attempts']} mutants re-verified)",
        color=theme.ink,
    )

    ops = archive["mutation_ops"]
    names = sorted(ops, key=lambda k: -ops[k]["accept_rate"])
    vals = [ops[n]["accept_rate"] for n in names]
    ax2.barh(range(len(names))[::-1], vals, height=0.6, color=theme.series[0], zorder=3)
    ax2.set_yticks(range(len(names))[::-1], names, fontsize=8)
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("share of mutants that survived re-verification")
    ax2.set_title("Which edits preserve solvability", color=theme.ink)
    ax2.grid(axis="y", visible=False)
    for i, (n, v) in enumerate(zip(names, vals, strict=True)):
        ax2.text(
            v + 0.02, len(names) - 1 - i, f"{v:.2f}  (n={ops[n]['tried']})",
            va="center", fontsize=8, color=theme.ink_muted,
        )
    _footnote(
        ax2,
        f"Max lineage depth {archive['max_lineage_depth']}; every archived task is still "
        "provably solvable because failed mutants are never archived.",
        theme,
    )
    return _save(fig, "map_elites", theme)


# --------------------------------------------------------------------------------------
# 6. Cost-to-go field  (the signature image)
# --------------------------------------------------------------------------------------


def cost_to_go_panels(spec, snapshots, theme: Theme = LIGHT) -> Path:
    """Render V* projected onto grid positions at several points in the plan.

    ``snapshots`` is a list of ``(title, rendered RGB array)``.
    """
    apply_matplotlib(theme)
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(4.9 * n, 4.9))
    if n == 1:
        axes = [axes]
    for ax, (title, arr) in zip(axes, snapshots, strict=True):
        ax.imshow(arr)
        ax.set_title(title, fontsize=10.5, color=theme.ink)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(visible=False)
        for s in ax.spines.values():
            s.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=ramp_cmap(theme))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["at the goal", "far from the goal"], fontsize=9)
    cb.set_label("exact actions remaining, V*(s)", fontsize=9, color=theme.ink_secondary)
    cb.outline.set_visible(False)
    fig.suptitle(
        "The oracle's cost-to-go field - this is the proof, and it is also the reward function",
        x=0.005, ha="left", fontsize=13, color=theme.ink, weight="bold",
    )
    return _save(fig, "cost_to_go", theme)


# --------------------------------------------------------------------------------------
# 7. Generalization
# --------------------------------------------------------------------------------------


def generalization(training: dict, theme: Theme = LIGHT) -> Path:
    apply_matplotlib(theme)
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    regimes: dict[str, list] = {}
    for run in training["runs"]:
        if "holdout" in run:
            regimes.setdefault(run["reward"], []).append(run)

    entries = []
    for reward, runs in sorted(regimes.items()):
        seen = [r["final"]["success_rate"] for r in runs]
        held = [r["holdout"]["success_rate"] for r in runs]
        entries.append((f"ppo_{reward}", AGENT_LABEL[f"ppo_{reward}"], seen, held))
    for name in ("greedy", "random"):
        b = training["baselines"].get(name)
        if b and "holdout" in b:
            entries.append(
                (name, AGENT_LABEL[name], [b["success_rate"]], [b["holdout"]["success_rate"]])
            )

    x = np.arange(2)
    # Held-out values cluster near zero and their labels collide; nudge each one to keep
    # a minimum vertical separation instead of stacking four numbers on one pixel row.
    order = sorted(range(len(entries)), key=lambda i: float(np.mean(entries[i][3])))
    label_y: dict[int, float] = {}
    prev = -1e9
    for i in order:
        v = float(np.mean(entries[i][3]))
        v = max(v, prev + 0.055)
        label_y[i] = v
        prev = v

    for idx, (key, lab, seen, held) in enumerate(entries):
        col = agent_color(key, theme)
        m = [float(np.mean(seen)), float(np.mean(held))]
        ax.plot(x, m, color=col, marker="o", markersize=7, markeredgecolor=theme.surface,
                markeredgewidth=1.6, label=lab, zorder=3)
        if len(seen) > 1:
            ax.fill_between(
                x, [min(seen), min(held)], [max(seen), max(held)],
                color=col, alpha=0.14, linewidth=0, zorder=2,
            )
        ax.annotate(
            f"{m[1]:.2f}", (1, label_y[idx]), textcoords="offset points", xytext=(9, 0),
            fontsize=9, color=col, va="center", fontweight="bold",
        )
    tb = training["config"]["train_buckets"]
    hb = training["config"]["holdout_buckets"]
    ax.set_xticks(x, [f"trained on D{tb}", f"held out D{hb}"])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("success rate")
    ax.set_xlim(-0.15, 1.35)
    ax.set_title("Generalization to unseen difficulty", color=theme.ink)
    ax.legend(loc="upper right", ncols=1)
    return _save(fig, "generalization", theme)


# --------------------------------------------------------------------------------------


def legend_patches(agents, theme: Theme) -> list[Patch]:
    return [Patch(facecolor=agent_color(a, theme), label=AGENT_LABEL.get(a, a)) for a in agents]


def build_all(results_dir: Path | None = None, themes=(LIGHT, DARK)) -> list[Path]:
    """Draw every JSON-backed figure that has data on disk."""
    rd = Path(results_dir or ROOT / "results")
    made: list[Path] = []
    jobs = [
        ("verification_funnel.json", funnel),
        ("rl_training.json", learning_curves),
        ("rl_training.json", generalization),
        ("difficulty_calibration.json", difficulty_calibration),
        ("agent_eval.json", agent_eval),
        ("map_elites.json", map_elites),
    ]
    for fname, fn in jobs:
        path = rd / fname
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for theme in themes:
            try:
                made.append(fn(data, theme))
            except Exception as e:  # a missing optional block must not kill the rest
                print(f"  ! {fn.__name__} ({theme.name}): {e}")
    return made


if __name__ == "__main__":
    for p in build_all():
        print(p)
