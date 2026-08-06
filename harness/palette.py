"""One palette, shared by the pygame renderer and every matplotlib figure.

The renderer and the plots are the same designed system, in both light and dark. Colours
are assigned by the *job* they do, which is why this file is organised by role rather
than by hue:

* **Categorical** slots identify agents (oracle / PPO / random / greedy / LLM) and SKUs.
  Assigned in fixed slot order and never cycled -- an entity keeps its hue no matter how
  many series a given figure draws.
* **Sequential** (single blue hue, light to dark) encodes magnitude: the cost-to-go
  field, the MAP-Elites coverage grid.
* **Status** is reserved for state, never for a series: a ruined box is ``critical``, a
  dispatched order is ``good``. Those tokens never stand in for "series 4".

The categorical slots were validated with the dataviz palette validator against both
surfaces (worst adjacent CVD dE 9.1 light / 8.4 dark; worst adjacent normal-vision dE
20.8 light / 19.3 dark). Two light-mode slots sit below 3:1 contrast on the light
surface, so every figure that uses them also carries a direct label or a legend -- colour
never carries meaning on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def mix(a: str, b: str, t: float) -> str:
    """Blend two hex colours in sRGB. Used for tints of an existing role, never to
    invent a new categorical hue."""
    ar, ag, ab = hex_to_rgb(a)
    br, bg, bb = hex_to_rgb(b)
    return f"#{round(ar + (br - ar) * t):02x}{round(ag + (bg - ag) * t):02x}{round(ab + (bb - ab) * t):02x}"


@dataclass(frozen=True)
class Theme:
    name: str
    # surfaces & ink
    surface: str
    plane: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    # categorical slots, fixed order
    series: tuple[str, ...]
    # sequential ramp, light -> dark
    ramp: tuple[str, ...]
    # status, reserved
    good: str
    warning: str
    serious: str
    critical: str
    # warehouse-specific surfaces, derived from the neutrals above
    floor: str = ""
    floor_alt: str = ""
    wall: str = ""
    rack: str = ""
    extras: dict = field(default_factory=dict)


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    plane="#f4f3ef",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"),
    ramp=(
        "#cde2fb",
        "#b7d3f6",
        "#9ec5f4",
        "#86b6ef",
        "#6da7ec",
        "#5598e7",
        "#3987e5",
        "#2a78d6",
        "#256abf",
        "#1c5cab",
        "#184f95",
        "#104281",
        "#0d366b",
    ),
    good="#0ca30c",
    warning="#fab219",
    serious="#ec835a",
    critical="#d03b3b",
    floor="#f0efea",
    floor_alt="#e8e7e1",
    wall="#3c3b38",
    rack="#cfcdc4",
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    plane="#0d0d0d",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767"),
    ramp=(
        "#0d366b",
        "#104281",
        "#184f95",
        "#1c5cab",
        "#256abf",
        "#2a78d6",
        "#3987e5",
        "#5598e7",
        "#6da7ec",
        "#86b6ef",
        "#9ec5f4",
        "#b7d3f6",
        "#cde2fb",
    ),
    good="#0ca30c",
    warning="#fab219",
    serious="#ec835a",
    critical="#d03b3b",
    floor="#232322",
    floor_alt="#1f1f1e",
    wall="#4a4945",
    rack="#33322f",
)

THEMES = {"light": LIGHT, "dark": DARK}


# --------------------------------------------------------------------------------------
# Role assignments (stable across every figure and the renderer)
# --------------------------------------------------------------------------------------

# Agents keep their slot no matter which subset a figure draws.
AGENT_SLOT = {
    "oracle": 0,
    "ppo_shaped": 1,
    "ppo_sparse": 2,
    "greedy": 3,
    "random": 4,
    "llm": 5,
    "scripted": 5,
}

AGENT_LABEL = {
    "oracle": "Oracle (optimal)",
    "ppo_shaped": "PPO + shaping",
    "ppo_sparse": "PPO sparse",
    "greedy": "Greedy baseline",
    "random": "Random",
    "llm": "LLM agent",
    "scripted": "Scripted agent",
}


def agent_color(agent: str, theme: Theme = LIGHT) -> str:
    return theme.series[AGENT_SLOT.get(agent, 6) % len(theme.series)]


def sku_color(sku_index: int, theme: Theme = LIGHT) -> str:
    """SKUs are identities, so they take categorical slots in fixed order."""
    return theme.series[sku_index % len(theme.series)]


def ramp_color(t: float, theme: Theme = LIGHT) -> str:
    """Sample the sequential ramp at ``t`` in [0, 1]."""
    t = min(1.0, max(0.0, t))
    idx = round(t * (len(theme.ramp) - 1))
    return theme.ramp[idx]


def apply_matplotlib(theme: Theme = LIGHT) -> None:
    """Push the theme into matplotlib rcParams: recessive hairline chrome, generous
    padding, no dashed gridlines, system sans throughout."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "axes.edgecolor": theme.axis,
            "axes.labelcolor": theme.ink_secondary,
            "axes.titlecolor": theme.ink,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": theme.grid,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": theme.ink_muted,
            "ytick.color": theme.ink_muted,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "text.color": theme.ink,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": theme.ink_secondary,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
            "figure.constrained_layout.use": True,
        }
    )
