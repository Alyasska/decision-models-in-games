"""Summarize runs: wins, points, speed and cost per seat.

  python report.py                 # every run in runs/
  python report.py drex-vs-bots    # only runs whose folder name contains this text
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    for d in sorted(Path("runs").glob("*/summary.json")):
        if flt not in d.parent.name:
            continue
        s = json.loads(d.read_text())
        res = s.get("results") or s.get("games")
        if not isinstance(res, list) or not res or "points" not in res[0]:
            continue
        stats = defaultdict(lambda: {"games": 0, "wins": 0, "points": 0, "calls": 0, "tokens": 0,
                                     "cost": 0.0, "secs": 0.0})
        for r in res:
            for seat, pts in r["points"].items():
                st = stats[seat]
                st["games"] += 1
                st["points"] += pts
                st["wins"] += r["winner"] == seat
                m = r.get("models", {}).get(seat)
                if m:
                    st["calls"] += m["calls"]
                    st["tokens"] += m["input_tokens"]
                    st["cost"] += m["cost_usd"]
                    st["secs"] += (m["avg_s"] or 0) * m["calls"]
        flags = " (dry)" if s.get("dry") else "" + (" (shuffled options)" if s.get("shuffle_options") else "")
        print(f"\n{d.parent.name}{flags}: {len(res)} games")
        print(f"  {'seat':14} {'wins':>6} {'avg pts':>8} {'calls/g':>8} {'tok/g':>8} {'s/move':>7} {'$/game':>8}")
        for seat, st in stats.items():
            g = st["games"]
            c = st["calls"]
            print(f"  {seat:14} {st['wins']:>3}/{g:<2} {st['points'] / g:>8.1f} "
                  f"{c / g if c else 0:>8.0f} {st['tokens'] / g if c else 0:>8.0f} "
                  f"{st['secs'] / c if c else 0:>7.2f} {st['cost'] / g if c else 0:>8.4f}")


if __name__ == "__main__":
    main()
