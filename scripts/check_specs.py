"""Re-verify every committed spec against the current executor.

A committed certificate is only worth anything if it still replays. This catches the
case where a change to the executor's semantics silently invalidates the specs sitting
in ``specs/`` -- they would keep loading fine and keep claiming to be solvable while no
longer being so.
"""

from __future__ import annotations

import sys
from pathlib import Path

from taskforge.verify import load_specs, replay
from taskforge.verify.pipeline import revalidate

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    tasks = load_specs(ROOT / "specs")
    if not tasks:
        print("no specs found in specs/ -- run scripts/build_specs.py", file=sys.stderr)
        return 1

    bad = []
    for t in tasks:
        trace = replay(t.spec, t.certificate.as_actions())
        ok = trace.goal_reached and trace.length == t.certificate.cost and revalidate(t)
        if not ok:
            bad.append((t.spec.task_id, trace.failure or "cost mismatch"))

    print(f"checked {len(tasks)} committed specs")
    if bad:
        print(f"FAILED: {len(bad)} specs no longer replay", file=sys.stderr)
        for tid, why in bad[:10]:
            print(f"  {tid}: {why}", file=sys.stderr)
        return 1
    print("all certificates replay to the goal at their certified length")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
