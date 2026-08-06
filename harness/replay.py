"""Self-contained HTML replay viewer.

Writes a single ``.html`` file with one embedded PNG per action (base64 data URIs) plus
a scrubber, play/pause and step controls. No CDN, no external assets, no network at all --
open the file from disk and it works.

One frame per *action* rather than per animation frame keeps the page small: the GIFs
interpolate for looks, but a scrubber wants exactly one position per step so that
"step 23" means something.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from harness.palette import LIGHT, Theme
from harness.render import RenderConfig, WarehouseRenderer, describe
from taskforge.dsl import TaskSpec
from taskforge.worlds.warehouse import apply_action, context_for, initial_state


def _png_data_uri(surf) -> str:
    import pygame

    buf = io.BytesIO()
    pygame.image.save(surf, buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_replay(
    spec: TaskSpec,
    plan,
    out: Path,
    theme: Theme = LIGHT,
    cell: int = 34,
    title: str = "",
    optimal: int | None = None,
) -> Path:
    ctx = context_for(spec)
    cfg = RenderConfig(cell=cell, theme=theme, title=title or spec.task_id,
                       subtitle="oracle certificate plan")
    r = WarehouseRenderer(spec, cfg)
    state = initial_state(ctx)

    frames = [_png_data_uri(r.render_frame(state, action="ready", step=0, optimal=optimal))]
    captions = ["ready"]
    for i, action in enumerate(plan):
        nxt = apply_action(ctx, state, action)
        if nxt is None:
            break
        label = describe(spec, action)
        banner = "irreversible: wrong SKU packed" if nxt[5] else ""
        frames.append(
            _png_data_uri(
                r.render_frame(nxt, action=label, step=i + 1, optimal=optimal, banner=banner)
            )
        )
        captions.append(label)
        state = nxt

    payload = json.dumps({"frames": frames, "captions": captions})
    html = _TEMPLATE.replace("__DATA__", payload).replace(
        "__TITLE__", title or spec.task_id
    ).replace("__OPTIMAL__", str(optimal or len(plan)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>taskforge replay - __TITLE__</title>
<style>
  :root {
    color-scheme: light dark;
    --surface: #fcfcfb; --plane: #f4f3ef; --ink: #0b0b0b;
    --ink2: #52514e; --muted: #898781; --line: #e1e0d9; --accent: #2a78d6;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --surface: #1a1a19; --plane: #0d0d0d; --ink: #ffffff;
      --ink2: #c3c2b7; --muted: #898781; --line: #2c2c2a; --accent: #3987e5;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 32px 24px; background: var(--plane); color: var(--ink);
    font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 1000px; margin: 0 auto; }
  h1 { font-size: 18px; margin: 0 0 4px; font-weight: 600; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
  .stage {
    background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
    padding: 14px; overflow-x: auto;
  }
  img { display: block; max-width: 100%; height: auto; image-rendering: auto; }
  .controls {
    display: flex; align-items: center; gap: 14px; margin-top: 16px; flex-wrap: wrap;
  }
  button {
    font: inherit; font-weight: 600; padding: 8px 16px; border-radius: 8px;
    border: 1px solid var(--line); background: var(--surface); color: var(--ink);
    cursor: pointer; min-width: 88px;
  }
  button:hover { background: var(--plane); }
  input[type=range] { flex: 1; min-width: 220px; accent-color: var(--accent); }
  .readout {
    font-variant-numeric: tabular-nums; color: var(--ink2); font-size: 13px;
    min-width: 210px;
  }
  .caption { margin-top: 10px; color: var(--ink2); font-size: 13px; min-height: 20px; }
  kbd {
    font: 11px ui-monospace, monospace; background: var(--plane); padding: 2px 6px;
    border-radius: 4px; border: 1px solid var(--line); color: var(--muted);
  }
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <div class="sub">Oracle certificate plan, one frame per action. Optimal length __OPTIMAL__ steps.</div>
  <div class="stage"><img id="frame" alt="warehouse replay frame"></div>
  <div class="controls">
    <button id="play">Play</button>
    <input type="range" id="scrub" min="0" value="0" step="1">
    <span class="readout" id="readout"></span>
  </div>
  <div class="caption" id="caption"></div>
  <div class="sub" style="margin-top:14px">
    <kbd>&larr;</kbd> <kbd>&rarr;</kbd> step &middot; <kbd>space</kbd> play/pause
  </div>
</div>
<script>
const DATA = __DATA__;
const img = document.getElementById('frame');
const scrub = document.getElementById('scrub');
const readout = document.getElementById('readout');
const caption = document.getElementById('caption');
const play = document.getElementById('play');
let i = 0, timer = null;
scrub.max = DATA.frames.length - 1;

function show(n) {
  i = Math.max(0, Math.min(DATA.frames.length - 1, n));
  img.src = DATA.frames[i];
  scrub.value = i;
  readout.textContent = `step ${i} / ${DATA.frames.length - 1}`;
  caption.textContent = DATA.captions[i] || '';
}
function stop() { clearInterval(timer); timer = null; play.textContent = 'Play'; }
function start() {
  if (i >= DATA.frames.length - 1) show(0);
  play.textContent = 'Pause';
  timer = setInterval(() => {
    if (i >= DATA.frames.length - 1) { stop(); return; }
    show(i + 1);
  }, 220);
}
play.onclick = () => (timer ? stop() : start());
scrub.oninput = () => { stop(); show(+scrub.value); };
document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowRight') { stop(); show(i + 1); }
  else if (e.key === 'ArrowLeft') { stop(); show(i - 1); }
  else if (e.key === ' ') { e.preventDefault(); timer ? stop() : start(); }
});
show(0);
</script>
</body>
</html>
"""
