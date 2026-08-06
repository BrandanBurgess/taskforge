"""Headless pygame-ce renderer: grid + side panel, GIF and PNG export.

Design notes that matter for how this reads:

* **Flat vector shapes, one stroke weight.** Tiles are rounded rectangles with a single
  hairline stroke and generous inset, so the grid reads as a floor plan rather than as a
  spreadsheet. No sprites, no gradients, no default pygame primaries.
* **The side panel is the point.** A grid animation alone is unreadable to someone who
  has not read the repo. The panel carries the order manifest with per-SKU fill, what
  the robot is holding, battery, ``step 14 / oracle-optimal 11``, and the current action
  in plain words -- so a GIF is self-explanatory in isolation.
* **Motion is interpolated.** The robot eases between cells over several frames and
  pick/pack/unlock flash briefly. One frame per environment step looks cheap and makes
  it genuinely hard to see what happened.

Everything runs under ``SDL_VIDEODRIVER=dummy``, so CI and ``demo.py`` produce frames
with no display attached.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import pygame  # noqa: E402

from harness.palette import LIGHT, Theme, hex_to_rgb, mix, ramp_color, sku_color  # noqa: E402
from taskforge.dsl import TaskSpec  # noqa: E402
from taskforge.worlds.warehouse import context_for, stock_of  # noqa: E402
from taskforge.worlds.warehouse.spec import CONVEYORS, DOCK, PACK, SHELF_CHARS, WALL  # noqa: E402

_INITIALISED = False


def _init() -> None:
    global _INITIALISED
    if not _INITIALISED:
        pygame.init()
        pygame.font.init()
        _INITIALISED = True


def _font(size: int, bold: bool = False) -> pygame.font.Font:
    _init()
    for name in ("Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"):
        path = pygame.font.match_font(name, bold=bold)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.SysFont("sans", size, bold=bold)


@dataclass
class RenderConfig:
    cell: int = 40
    panel_w: int = 300
    margin: int = 22
    header_h: int = 46
    theme: Theme = LIGHT
    show_panel: bool = True
    title: str = ""
    subtitle: str = ""


C = hex_to_rgb


# --------------------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------------------


def _round_rect(
    surf: pygame.Surface,
    rect: tuple[float, float, float, float],
    fill: str | None,
    stroke: str | None = None,
    radius: int = 5,
    width: int = 1,
) -> None:
    r = pygame.Rect(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
    if fill:
        pygame.draw.rect(surf, C(fill), r, border_radius=radius)
    if stroke:
        pygame.draw.rect(surf, C(stroke), r, width=width, border_radius=radius)


def _text(
    surf: pygame.Surface,
    s: str,
    pos: tuple[int, int],
    size: int = 13,
    color: str = "#000000",
    bold: bool = False,
    right: bool = False,
) -> int:
    img = _font(size, bold).render(s, True, C(color))
    x = pos[0] - img.get_width() if right else pos[0]
    surf.blit(img, (x, pos[1]))
    return img.get_width()


def _arrow(surf: pygame.Surface, cx: float, cy: float, d: tuple[int, int], size: float, color: str):
    """A small chevron showing conveyor travel direction."""
    dx, dy = d
    px, py = -dy, dx
    tip = (cx + dx * size, cy + dy * size)
    a = (cx - dx * size * 0.45 + px * size * 0.75, cy - dy * size * 0.45 + py * size * 0.75)
    b = (cx - dx * size * 0.45 - px * size * 0.75, cy - dy * size * 0.45 - py * size * 0.75)
    pygame.draw.polygon(surf, C(color), [tip, a, b])


# --------------------------------------------------------------------------------------
# Renderer
# --------------------------------------------------------------------------------------


class WarehouseRenderer:
    def __init__(self, spec: TaskSpec, cfg: RenderConfig | None = None):
        _init()
        self.spec = spec
        self.cfg = cfg or RenderConfig()
        self.ctx = context_for(spec)
        self.theme = self.cfg.theme
        cell = self.cfg.cell
        self.grid_w = self.ctx.width * cell
        self.grid_h = self.ctx.height * cell
        self.width = self.cfg.margin * 2 + self.grid_w + (
            self.cfg.panel_w + self.cfg.margin if self.cfg.show_panel else 0
        )
        self.height = self.cfg.margin * 2 + self.grid_h + self.cfg.header_h
        self._static: pygame.Surface | None = None

    # -- geometry ---------------------------------------------------------------------
    def _origin(self) -> tuple[int, int]:
        return self.cfg.margin, self.cfg.margin + self.cfg.header_h

    def cell_rect(self, idx: int) -> tuple[float, float, float, float]:
        ox, oy = self._origin()
        x, y = self.ctx.xy(idx)
        c = self.cfg.cell
        return (ox + x * c, oy + y * c, c, c)

    def cell_center(self, idx: int) -> tuple[float, float]:
        r = self.cell_rect(idx)
        return r[0] + r[2] / 2, r[1] + r[3] / 2

    def xy_center(self, x: float, y: float) -> tuple[float, float]:
        ox, oy = self._origin()
        c = self.cfg.cell
        return ox + (x + 0.5) * c, oy + (y + 0.5) * c

    # -- static layer -----------------------------------------------------------------
    def _draw_static(self) -> pygame.Surface:
        t = self.theme
        surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        surf.fill(C(t.surface))
        ctx, p, c = self.ctx, self.ctx.payload, self.cfg.cell
        inset = 1.5

        for idx in range(ctx.width * ctx.height):
            x, y = ctx.xy(idx)
            ch = p.tile(x, y)
            zone = ctx.zone_of[idx]
            rx, ry, rw, rh = self.cell_rect(idx)
            r = (rx + inset, ry + inset, rw - 2 * inset, rh - 2 * inset)

            if ch == WALL:
                # Perimeter walls are the building; interior blocks are racking. They are
                # identically impassable, but drawing them the same way makes the floor
                # read as a maze instead of a warehouse.
                perimeter = x == 0 or y == 0 or x == ctx.width - 1 or y == ctx.height - 1
                if perimeter:
                    _round_rect(surf, r, t.wall, None, radius=4)
                else:
                    _round_rect(surf, r, t.rack, mix(t.rack, t.ink, 0.16), radius=4)
                    for k in (0.34, 0.66):
                        pygame.draw.line(
                            surf,
                            C(mix(t.rack, t.ink, 0.16)),
                            (r[0] + 4, r[1] + r[3] * k),
                            (r[0] + r[2] - 4, r[1] + r[3] * k),
                            1,
                        )
                continue

            # floor: subtle checker so the grid reads without heavy gridlines
            base = t.floor if (x + y) % 2 == 0 else t.floor_alt
            if zone:
                base = mix(base, t.warning, 0.22 if t.name == "light" else 0.16)
            _round_rect(surf, r, base, None, radius=4)

            if ch in SHELF_CHARS:
                sku = int(ch)
                col = sku_color(sku, t)
                _round_rect(surf, r, mix(t.rack, col, 0.30), None, radius=4)
                # a solid SKU chip: identity is the shelf's whole job
                chip = (rx + rw * 0.22, ry + rh * 0.22, rw * 0.56, rh * 0.56)
                _round_rect(surf, chip, col, None, radius=3)
                _text(
                    surf,
                    p.skus[sku][0].upper(),
                    (int(rx + rw / 2 - 4), int(ry + rh / 2 - 8)),
                    size=max(11, c // 3),
                    color=t.surface,
                    bold=True,
                )
            # Infrastructure is identified by SHAPE and neutral ink, never by a
            # categorical hue. The categorical slots belong to SKU identity, and if the
            # charge dock were "aqua" it would be indistinguishable from whichever SKU
            # holds that slot. Shape is a free channel here; hue is not.
            elif ch == PACK:
                _round_rect(surf, r, mix(t.floor, t.ink, 0.10), t.ink, radius=4, width=2)
                cx, cy = self.cell_center(idx)
                _round_rect(
                    surf, (cx - c * 0.23, cy - c * 0.16, c * 0.46, c * 0.34), t.ink, radius=2
                )
                pygame.draw.line(
                    surf, C(t.surface), (cx - c * 0.23, cy - c * 0.03), (cx + c * 0.23, cy - c * 0.03), 2
                )
            elif ch == DOCK:
                _round_rect(
                    surf, r, mix(t.floor, t.ink, 0.06), mix(t.ink, t.floor, 0.45), radius=4, width=2
                )
                cx, cy = self.cell_center(idx)
                pygame.draw.polygon(
                    surf,
                    C(mix(t.ink, t.floor, 0.20)),
                    [
                        (cx + c * 0.10, cy - c * 0.24),
                        (cx - c * 0.13, cy + c * 0.03),
                        (cx + c * 0.01, cy + c * 0.03),
                        (cx - c * 0.09, cy + c * 0.26),
                        (cx + c * 0.14, cy - c * 0.02),
                        (cx, cy - c * 0.02),
                    ],
                )
            elif ch in CONVEYORS:
                _round_rect(surf, r, mix(t.floor, t.ink, 0.09), None, radius=4)
                cx, cy = self.cell_center(idx)
                d = CONVEYORS[ch]
                for k in (-1, 1):
                    _arrow(
                        surf,
                        cx - d[0] * c * 0.17 * k,
                        cy - d[1] * c * 0.17 * k,
                        d,
                        c * 0.14,
                        mix(t.ink, t.floor, 0.42),
                    )

            if zone:
                # hairline dashed-free ring; locked-ness is also stated in the panel
                _round_rect(surf, r, None, mix(t.warning, t.ink, 0.25), radius=4, width=1)

        return surf

    # -- dynamic layer ----------------------------------------------------------------
    def render_frame(
        self,
        state,
        robot_xy: tuple[float, float] | None = None,
        action: str = "",
        step: int = 0,
        optimal: int | None = None,
        flash: float = 0.0,
        flash_kind: str = "",
        heat: dict[int, int] | None = None,
        path: list[int] | None = None,
        banner: str = "",
    ) -> pygame.Surface:
        t = self.theme
        if self._static is None:
            self._static = self._draw_static()
        surf = self._static.copy()
        ctx, c = self.ctx, self.cfg.cell

        if heat:
            self._draw_heat(surf, heat)
        if path:
            self._draw_path(surf, path)

        # header (only when a header band was reserved -- otherwise it would print
        # straight over the top row of cells)
        ox, oy = self._origin()
        if self.cfg.header_h > 0:
            title = self.cfg.title or self.spec.task_id
            _text(surf, title, (ox, self.cfg.margin - 2), size=15, color=t.ink, bold=True)
            if self.cfg.subtitle:
                _text(
                    surf, self.cfg.subtitle, (ox, self.cfg.margin + 20), size=11,
                    color=t.ink_muted,
                )

        pos, held, filled, dispatched, unlocked, ruined, battery = state

        # unlocked zones lose their amber wash
        for bit, z in enumerate(ctx.zone_ids):
            if unlocked >> bit & 1:
                for idx in range(ctx.width * ctx.height):
                    if ctx.zone_of[idx] == z and ctx.passable[idx]:
                        rx, ry, rw, rh = self.cell_rect(idx)
                        x, y = ctx.xy(idx)
                        base = t.floor if (x + y) % 2 == 0 else t.floor_alt
                        _round_rect(surf, (rx + 1.5, ry + 1.5, rw - 3, rh - 3), base, None, radius=4)

        # robot
        if robot_xy is None:
            robot_xy = ctx.xy(pos)
        cx, cy = self.xy_center(*robot_xy)
        rad = c * 0.30
        if flash > 0:
            glow = {"pick": t.series[2], "pack": t.series[0], "unlock": t.warning,
                    "ruin": t.critical}.get(flash_kind, t.series[0])
            gr = rad + c * 0.30 * flash
            ring = pygame.Surface((int(gr * 2 + 4), int(gr * 2 + 4)), pygame.SRCALPHA)
            pygame.draw.circle(
                ring, (*C(glow), int(150 * flash)), (int(gr + 2), int(gr + 2)), int(gr)
            )
            surf.blit(ring, (cx - gr - 2, cy - gr - 2))
        body = t.critical if ruined else t.ink
        pygame.draw.circle(surf, C(t.surface), (int(cx), int(cy)), int(rad + 2.5))
        pygame.draw.circle(surf, C(body), (int(cx), int(cy)), int(rad))
        # carried-item pips ring the robot, so "what is it holding" is visible on the grid
        carried = [s for s in range(ctx.n_skus) for _ in range(held[s])]
        for i, s in enumerate(carried[:4]):
            ang = -1.2 + i * 0.9
            import math

            px = cx + math.cos(ang) * rad * 1.62
            py = cy + math.sin(ang) * rad * 1.62
            pygame.draw.circle(surf, C(t.surface), (int(px), int(py)), int(c * 0.14) + 2)
            pygame.draw.circle(surf, C(sku_color(s, t)), (int(px), int(py)), int(c * 0.14))

        if self.cfg.show_panel:
            self._draw_panel(surf, state, action, step, optimal, banner)
        return surf

    def _draw_heat(self, surf: pygame.Surface, heat: dict[int, int]) -> None:
        """Sequential single-hue field: light = close to done, dark = far."""
        if not heat:
            return
        vals = [v for v in heat.values() if v < (1 << 19)]
        if not vals:
            return
        lo, hi = min(vals), max(vals)
        span = max(1, hi - lo)
        for idx, v in heat.items():
            if v >= (1 << 19):
                continue
            rx, ry, rw, rh = self.cell_rect(idx)
            col = ramp_color((v - lo) / span, self.theme)
            _round_rect(surf, (rx + 1.5, ry + 1.5, rw - 3, rh - 3), col, None, radius=4)
            _text(
                surf,
                str(v),
                (int(rx + rw / 2 - 7), int(ry + rh / 2 - 7)),
                size=max(9, self.cfg.cell // 4),
                color=self.theme.surface if (v - lo) / span > 0.45 else self.theme.ink,
                bold=True,
            )

    def _draw_path(self, surf: pygame.Surface, path: list[int]) -> None:
        if len(path) < 2:
            return
        # Halo then hairline, so the trace stays legible over the heat field without
        # burying the per-cell V* numbers underneath it.
        pts = [self.cell_center(i) for i in path]
        pygame.draw.lines(surf, C(self.theme.surface), False, pts, 5)
        pygame.draw.lines(surf, C(mix(self.theme.ink, self.theme.surface, 0.25)), False, pts, 2)
        for p, col in ((pts[0], self.theme.ink), (pts[-1], self.theme.ink)):
            pygame.draw.circle(surf, C(self.theme.surface), (int(p[0]), int(p[1])), 5)
            pygame.draw.circle(surf, C(col), (int(p[0]), int(p[1])), 3)

    # -- side panel --------------------------------------------------------------------
    def _draw_panel(
        self,
        surf: pygame.Surface,
        state,
        action: str,
        step: int,
        optimal: int | None,
        banner: str,
    ) -> None:
        t = self.theme
        ctx, p = self.ctx, self.ctx.payload
        pos, held, filled, dispatched, unlocked, ruined, battery = state
        x0 = self.cfg.margin + self.grid_w + self.cfg.margin
        y = self.cfg.margin + self.cfg.header_h
        w = self.cfg.panel_w
        _round_rect(surf, (x0, y, w, self.grid_h), t.plane, t.grid, radius=10)
        px = x0 + 18
        yy = y + 16

        # -- step counter, the headline number ----------------------------------------
        _text(surf, "STEP", (px, yy), size=9, color=t.ink_muted, bold=True)
        yy += 13
        big = _font(26, True).render(str(step), True, C(t.ink))
        surf.blit(big, (px, yy))
        if optimal is not None:
            _text(
                surf,
                f"/ {optimal} oracle-optimal",
                (px + big.get_width() + 8, yy + 12),
                size=11,
                color=t.ink_muted,
            )
        yy += 36

        _text(surf, "ACTION", (px, yy), size=9, color=t.ink_muted, bold=True)
        yy += 13
        _text(surf, action or "-", (px, yy), size=13, color=t.ink_secondary, bold=True)
        yy += 24

        # -- order manifest ------------------------------------------------------------
        _text(surf, "ORDERS", (px, yy), size=9, color=t.ink_muted, bold=True)
        yy += 16
        for o in range(ctx.n_orders):
            done = bool(dispatched >> o & 1)
            wrecked = bool(ruined >> o & 1)
            label = f"order {o}"
            state_txt = "RUINED" if wrecked else ("dispatched" if done else "open")
            state_col = t.critical if wrecked else (t.good if done else t.ink_muted)
            _text(surf, label, (px, yy), size=12, color=t.ink, bold=True)
            _text(surf, state_txt, (x0 + w - 18, yy), size=10, color=state_col, bold=True, right=True)
            yy += 17
            for s in range(ctx.n_skus):
                need = ctx.need_of(o, s)
                if need == 0:
                    continue
                got = filled[o * ctx.n_skus + s]
                _round_rect(surf, (px + 2, yy + 3, 8, 8), sku_color(s, t), None, radius=2)
                _text(surf, p.skus[s], (px + 16, yy), size=11, color=t.ink_secondary)
                # fill meter: 2px surface gap between segments, no borders
                bx = px + 118
                for k in range(need):
                    filled_seg = k < got
                    _round_rect(
                        surf,
                        (bx + k * 14, yy + 3, 12, 8),
                        sku_color(s, t) if filled_seg else mix(t.plane, t.ink, 0.10),
                        None,
                        radius=2,
                    )
                _text(
                    surf, f"{got}/{need}", (x0 + w - 18, yy), size=10,
                    color=t.ink_muted, bold=True, right=True
                )
                yy += 15
            yy += 6

        # -- held items ------------------------------------------------------------------
        yy += 2
        _text(surf, "HOLDING", (px, yy), size=9, color=t.ink_muted, bold=True)
        _text(
            surf, f"{sum(held)} / {ctx.capacity}", (x0 + w - 18, yy), size=9,
            color=t.ink_muted, bold=True, right=True
        )
        yy += 15
        slot = 0
        for s in range(ctx.n_skus):
            for _ in range(held[s]):
                _round_rect(surf, (px + slot * 26, yy, 22, 22), sku_color(s, t), None, radius=4)
                _text(
                    surf, p.skus[s][0].upper(), (px + slot * 26 + 7, yy + 4),
                    size=11, color=t.surface, bold=True
                )
                slot += 1
        for k in range(slot, ctx.capacity):
            _round_rect(surf, (px + k * 26, yy, 22, 22), None, mix(t.plane, t.ink, 0.16), radius=4)
        yy += 32

        # -- battery ----------------------------------------------------------------------
        if ctx.battery_max >= 0:
            _text(surf, "BATTERY", (px, yy), size=9, color=t.ink_muted, bold=True)
            _text(
                surf, f"{battery}/{ctx.battery_max}", (x0 + w - 18, yy), size=9,
                color=t.ink_muted, bold=True, right=True
            )
            yy += 14
            frac = max(0.0, battery / max(1, ctx.battery_max))
            bw = w - 36
            _round_rect(surf, (px, yy, bw, 7), mix(t.plane, t.ink, 0.10), None, radius=3)
            col = t.critical if frac < 0.2 else (t.warning if frac < 0.4 else t.series[2])
            if frac > 0:
                _round_rect(surf, (px, yy, max(3, bw * frac), 7), col, None, radius=3)
            yy += 20

        # -- zones -------------------------------------------------------------------------
        if ctx.zone_ids:
            _text(surf, "ZONES", (px, yy), size=9, color=t.ink_muted, bold=True)
            yy += 14
            for bit, z in enumerate(ctx.zone_ids):
                open_ = bool(unlocked >> bit & 1)
                _text(surf, f"zone {z}", (px, yy), size=11, color=t.ink_secondary)
                _text(
                    surf, "unlocked" if open_ else "locked", (x0 + w - 18, yy),
                    size=10, color=t.good if open_ else t.warning, bold=True, right=True
                )
                yy += 15
            yy += 6

        # -- stock ---------------------------------------------------------------------------
        stock = stock_of(ctx, state)
        _text(surf, "SHELF STOCK", (px, yy), size=9, color=t.ink_muted, bold=True)
        yy += 14
        for s in range(ctx.n_skus):
            _round_rect(surf, (px + 2, yy + 3, 8, 8), sku_color(s, t), None, radius=2)
            _text(surf, p.skus[s], (px + 16, yy), size=11, color=t.ink_secondary)
            _text(
                surf, str(stock[s]), (x0 + w - 18, yy), size=11,
                color=t.ink_secondary, bold=True, right=True
            )
            yy += 15

        if banner:
            _round_rect(surf, (px - 6, yy + 6, w - 24, 26), mix(t.plane, t.critical, 0.18),
                        t.critical, radius=6)
            _text(surf, banner, (px + 4, yy + 13), size=11, color=t.critical, bold=True)


# --------------------------------------------------------------------------------------
# Episode -> frames
# --------------------------------------------------------------------------------------

ACTION_WORDS = {
    "move": "move {arg}",
    "pick": "pick {sku}",
    "place": "place {sku}",
    "pack": "pack into order {arg}",
    "scan": "scan + dispatch order {arg}",
    "charge": "charge at dock",
    "unlock": "unlock zone {arg}",
}


def describe(spec: TaskSpec, action) -> str:
    ctx = context_for(spec)
    name, arg = action
    tpl = ACTION_WORDS.get(name, name)
    sku = ctx.payload.skus[arg] if name in ("pick", "place") and isinstance(arg, int) else ""
    return tpl.format(arg=arg, sku=sku)


def episode_frames(
    spec: TaskSpec,
    plan,
    cfg: RenderConfig | None = None,
    tween: int = 4,
    hold_start: int = 6,
    hold_end: int = 14,
    optimal: int | None = None,
    title: str = "",
    subtitle: str = "",
):
    """Render a plan into a list of RGB frames with interpolated motion.

    ``tween`` frames per action. Movement eases between cell centres; non-movement
    actions hold position and emit a decaying highlight flash instead, so a pick reads
    differently from a step even though both take one action.
    """
    from taskforge.worlds.warehouse import apply_action

    cfg = cfg or RenderConfig()
    if title:
        cfg.title = title
    if subtitle:
        cfg.subtitle = subtitle
    r = WarehouseRenderer(spec, cfg)
    ctx = r.ctx
    state = ctx and __import__(
        "taskforge.worlds.warehouse", fromlist=["initial_state"]
    ).initial_state(ctx)

    frames = []
    xy = ctx.xy(state[0])
    for _ in range(hold_start):
        frames.append(r.render_frame(state, xy, "ready", 0, optimal))

    for i, action in enumerate(plan):
        nxt = apply_action(ctx, state, action)
        if nxt is None:
            break
        name = action[0]
        start_xy = ctx.xy(state[0])
        end_xy = ctx.xy(nxt[0])
        label = describe(spec, action)
        ruined_now = nxt[5] != 0 and state[5] == 0
        kind = "ruin" if ruined_now else name
        banner = "irreversible: wrong SKU packed" if nxt[5] else ""
        for k in range(tween):
            u = (k + 1) / tween
            ease = u * u * (3 - 2 * u)  # smoothstep
            ix = start_xy[0] + (end_xy[0] - start_xy[0]) * ease
            iy = start_xy[1] + (end_xy[1] - start_xy[1]) * ease
            flash = 0.0
            if name in ("pick", "pack", "unlock", "place") or ruined_now:
                flash = max(0.0, 1.0 - u)
            frames.append(
                r.render_frame(
                    nxt, (ix, iy), label, i + 1, optimal, flash, kind, banner=banner
                )
            )
        state = nxt

    final = frames[-1] if frames else r.render_frame(state, xy, "", 0, optimal)
    for _ in range(hold_end):
        frames.append(final)
    return frames, state


def surf_to_array(surf: pygame.Surface):
    import numpy as np

    arr = pygame.surfarray.array3d(surf)
    return np.transpose(arr, (1, 0, 2))


def save_png(surf: pygame.Surface, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surf, str(path))


def save_gif(frames, path: str | Path, fps: int = 18, loop: int = 0) -> Path:
    """Write a looping GIF. Frames are quantised to a shared adaptive palette, which is
    what keeps a 300-frame animation in the low hundreds of KB rather than tens of MB."""
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [surf_to_array(f) if isinstance(f, pygame.Surface) else f for f in frames]
    imageio.mimsave(path, arrays, format="GIF", fps=fps, loop=loop, subrectangles=True)
    return path


def hstack(frame_lists, gap: int = 0):
    """Stack several equal-height frame sequences side by side, padding the short ones
    by holding their last frame -- used for the three-way agent comparison."""
    import numpy as np

    n = max(len(f) for f in frame_lists)
    cols = []
    for seq in frame_lists:
        arrs = [surf_to_array(f) if isinstance(f, pygame.Surface) else f for f in seq]
        arrs = arrs + [arrs[-1]] * (n - len(arrs))
        cols.append(arrs)
    out = []
    h = max(c[0].shape[0] for c in cols)
    for i in range(n):
        row = []
        for c in cols:
            a = c[i]
            if a.shape[0] < h:
                pad = np.full((h - a.shape[0], a.shape[1], 3), a[0, 0], dtype=a.dtype)
                a = np.vstack([a, pad])
            row.append(a)
        out.append(np.hstack(row))
    return out
