"""Generate, verify, and commit a corpus of pre-verified tasks.

Writes ``specs/*.json`` (spec + certificate + difficulty label) and
``results/verification_funnel.json`` (per-stage counts and rejection reasons for both
generator arms).

    python scripts/build_specs.py --curriculum 12 --wild 250
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from taskforge.verify import verify
from taskforge.verify.pipeline import accepted_task
from taskforge.worlds.warehouse.generator import generate, random_params

ROOT = Path(__file__).resolve().parents[1]


def run_arm(name: str, specs, node_budget: int) -> dict:
    stages = Counter()
    reasons: dict[str, Counter] = {"V1": Counter(), "V2": Counter(), "V3": Counter()}
    accepted = []
    disagreements = []
    t0 = time.time()

    for spec in specs:
        stages["attempts"] += 1
        outcome = verify(spec, node_budget=node_budget)
        if outcome.reject_stage is None:
            stages["V1_pass"] += 1
            stages["V2_pass"] += 1
            stages["V3_pass"] += 1
            stages["accepted"] += 1
            accepted.append((spec, outcome))
            continue
        if outcome.reject_stage == "V1":
            reasons["V1"][outcome.reason_code()] += 1
            continue
        stages["V1_pass"] += 1
        if outcome.reject_stage == "V2":
            reasons["V2"][outcome.reason_code()] += 1
            continue
        stages["V2_pass"] += 1
        reasons["V3"][outcome.reason_code()] += 1
        if outcome.disagreement:
            disagreements.append(outcome.task_id)

    elapsed = time.time() - t0
    return {
        "arm": name,
        "attempts": stages["attempts"],
        "v1_pass": stages["V1_pass"],
        "v2_pass": stages["V2_pass"],
        "v3_pass": stages["V3_pass"],
        "accepted": stages["accepted"],
        "accept_rate": round(stages["accepted"] / max(1, stages["attempts"]), 4),
        "rejections": {k: dict(v) for k, v in reasons.items()},
        "v2_v3_disagreements": disagreements,
        "v2_v3_agreement_rate": round(
            1.0 - len(disagreements) / max(1, stages["V2_pass"]), 6
        ),
        "seconds": round(elapsed, 2),
        "_accepted_objs": accepted,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curriculum", type=int, default=12, help="seeds per difficulty level")
    ap.add_argument("--wild", type=int, default=250, help="randomized generation attempts")
    ap.add_argument("--node-budget", type=int, default=200_000)
    ap.add_argument("--specs-dir", default=str(ROOT / "specs"))
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    args = ap.parse_args()

    specs_dir = Path(args.specs_dir)
    specs_dir.mkdir(parents=True, exist_ok=True)
    for old in specs_dir.glob("*.json"):
        old.unlink()

    curriculum = [
        generate(seed, d) for d in range(1, 6) for seed in range(args.curriculum)
    ]
    print(f"curriculum arm: {len(curriculum)} attempts")
    cur = run_arm("curriculum", curriculum, args.node_budget)
    print(f"  accepted {cur['accepted']}/{cur['attempts']} in {cur['seconds']}s")

    rng = random.Random(20260805)
    wild_specs = []
    for i in range(args.wild):
        p = random_params(rng)
        wild_specs.append(generate(500_000 + i, 0, p))
    print(f"wild arm: {len(wild_specs)} attempts")
    wild = run_arm("wild", wild_specs, args.node_budget)
    print(f"  accepted {wild['accepted']}/{wild['attempts']} in {wild['seconds']}s")

    # Commit the curriculum tasks: they are what training and evaluation run on.
    saved = 0
    for spec, outcome in cur.pop("_accepted_objs"):
        task = accepted_task(spec, outcome)
        task.save(specs_dir / f"{spec.task_id}.json")
        saved += 1
    wild.pop("_accepted_objs")

    funnel = {
        "generated_at_seed": 20260805,
        "node_budget": args.node_budget,
        "arms": [cur, wild],
        "specs_committed": saved,
    }
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verification_funnel.json").write_text(json.dumps(funnel, indent=2) + "\n")

    agree = min(cur["v2_v3_agreement_rate"], wild["v2_v3_agreement_rate"])
    print(f"\nsaved {saved} verified specs to {specs_dir}")
    print(f"V2/V3 agreement: {agree * 100:.4f}%")
    for arm in (cur, wild):
        print(f"\n{arm['arm']}: accept rate {arm['accept_rate'] * 100:.1f}%")
        for stage in ("V1", "V2", "V3"):
            if arm["rejections"][stage]:
                print(f"  {stage} rejections: {arm['rejections'][stage]}")


if __name__ == "__main__":
    main()
