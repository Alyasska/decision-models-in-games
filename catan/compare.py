"""Head-to-head comparison of two model seats from mirrored runs (play.py --mirror).

  .venv/bin/python compare.py fair-pilot            # runs whose folder name contains this tag
  .venv/bin/python compare.py fair-                 # several runs together (e.g. a series split over processes)

Each deal is played twice with the two models' seats swapped, so seating and board cancel out within a deal.
Reports: wins with a 95% Wilson interval, average points, how often each finished above the other
(with an exact two-sided sign test), the per-deal point difference with a 95% bootstrap interval, speed and cost.
"""
import glob
import json
import math
import random
import statistics
import sys
from collections import defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def sign_test(a, b):
    """Exact two-sided binomial test, ties dropped: P(result at least this lopsided | 50/50)."""
    n, k = a + b, max(a, b)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "fair-"
    runs = [f for f in sorted(glob.glob(f"runs/*{tag}*/summary.json")) if json.load(open(f)).get("mirror")]
    if not runs:
        raise SystemExit(f"no mirrored runs matching {tag!r}")
    games, model_stats = [], defaultdict(lambda: defaultdict(float))
    seats = None
    for f in runs:
        s = json.load(open(f))
        seats = seats or s["seats"]
        for r in s["results"]:
            if r.get("error"):
                continue
            games.append((f.split("/")[1], r))
            for m, st in r.get("models", {}).items():
                ms = model_stats[m]
                ms["games"] += 1
                ms["calls"] += st["calls"]
                ms["tokens"] += st["input_tokens"]
                ms["cost"] += st["cost_usd"] or 0
                ms["seconds"] += (st["avg_s"] or 0) * st["calls"]
    A, B = seats[0], seats[1]
    n = len(games)
    print(f"{len(runs)} run(s), {n} games, models {A} vs {B}, others {seats[2:]}")
    print(f"code: {json.load(open(runs[0])).get('code')}\n")

    print(f"{'seat':12} {'wins':>5} {'win rate (95% CI)':>22} {'avg pts':>8}")
    for seat in seats:
        w = sum(r["winner"] == seat for _, r in games)
        lo, hi = wilson(w, n)
        pts = statistics.mean(r["points"][seat] for _, r in games)
        print(f"{seat:12} {w:>5} {w / n:>8.0%} ({lo:.0%}–{hi:.0%}) {pts:>8.1f}")

    above = {A: 0, B: 0}
    for _, r in games:
        pa, pb = r["points"][A], r["points"][B]
        if pa != pb:
            above[A if pa > pb else B] += 1
    ties = n - above[A] - above[B]
    print(f"\nfinished above the other: {A} {above[A]}, {B} {above[B]}, equal {ties}"
          f"  (sign test p = {sign_test(above[A], above[B]):.2f})")

    deals = defaultdict(list)
    for run, r in games:
        deals[(run, r["deal"])].append(r["points"][A] - r["points"][B])
    diffs = [sum(v) / len(v) for v in deals.values() if len(v) == 2]
    if diffs:
        rng = random.Random(0)
        boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(5000))
        print(f"points {A} minus {B}, per complete deal (both seatings): mean {statistics.mean(diffs):+.2f} "
              f"(95% bootstrap {boots[124]:+.2f} to {boots[4874]:+.2f}), {len(diffs)} deals")

    print(f"\n{'model':12} {'calls/game':>10} {'tokens/game':>12} {'s/call':>7} {'$/game':>8}")
    for m, ms in model_stats.items():
        g = ms["games"]
        print(f"{m:12} {ms['calls'] / g:>10.0f} {ms['tokens'] / g:>12.0f} {ms['seconds'] / max(1, ms['calls']):>7.2f}"
              f" {ms['cost'] / g:>8.4f}")


if __name__ == "__main__":
    main()
