"""Report for tables with two seats per model (e.g. drex-max-or#1, jev#1, drex-max-or#2, jev#2) played with --rotate.

  .venv/bin/python selfplay_report.py selfplay

Shows wins per model (both copies together; 50% each if the models are equally strong), the control comparison
copy #1 vs copy #2 of the same model (equal in expectation, so its gap shows the noise level of the series),
average points, and win rate by turn order position. Reads every run folder whose name contains the tag.
"""
import glob
import json
import statistics
import sys
from collections import Counter, defaultdict

from compare import sign_test, wilson


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "selfplay"
    results = []
    for f in sorted(glob.glob(f"runs/*{tag}*/summary.json")):
        s = json.load(open(f))
        results += [r for r in s["results"] if not r.get("error") and r.get("winner")]
    n = len(results)
    if not n:
        raise SystemExit("no finished games")
    seats = sorted(results[0]["points"])
    fam = lambda s: s.split("#")[0]
    models = sorted({fam(s) for s in seats})
    print(f"{n} finished games, seats {seats}\n")

    print("wins per model (both copies together; equal strength = 50%):")
    for m in models:
        w = sum(fam(r["winner"]) == m for r in results)
        lo, hi = wilson(w, n)
        pts = statistics.mean(r["points"][s] for r in results for s in seats if fam(s) == m)
        print(f"  {m:12} {w:3}/{n}  {w / n:5.0%}  (95% CI {lo:.0%}–{hi:.0%})   avg points per seat {pts:.2f}")

    print("\nper seat:")
    for s in seats:
        w = sum(r["winner"] == s for r in results)
        print(f"  {s:14} wins {w:3} ({w / n:4.0%})   avg points {statistics.mean(r['points'][s] for r in results):.2f}")

    print("\ncontrol: copy #1 vs copy #2 of the same model (should be equal):")
    for m in models:
        a, b = f"{m}#1", f"{m}#2"
        if a in seats and b in seats:
            wa = sum(r["winner"] == a for r in results)
            wb = sum(r["winner"] == b for r in results)
            above = Counter((a if r["points"][a] > r["points"][b] else b) for r in results if r["points"][a] != r["points"][b])
            print(f"  {m:12} wins {wa} vs {wb} (p = {sign_test(wa, wb):.2f}); finished above the other {above[a]} vs {above[b]}"
                  f" (p = {sign_test(above[a], above[b]):.2f})")

    print("\nbetween models: how often a seat of each model finished above a seat of the other (all 4 cross pairs):")
    if len(models) == 2:
        x, y = models
        ab = Counter()
        for r in results:
            for s in seats:
                for t in seats:
                    if fam(s) == x and fam(t) == y and r["points"][s] != r["points"][t]:
                        ab[x if r["points"][s] > r["points"][t] else y] += 1
        print(f"  {x} {ab[x]} vs {y} {ab[y]}  (pairs within a game are not independent; read as a tendency)")

    pos, posn = Counter(), Counter()
    for r in results:
        for i, s in enumerate(r["seating"]):
            posn[i] += 1
            pos[i] += r["winner"] == s
    print("\nwin rate by turn order position:", ", ".join(f"{i + 1}: {pos[i] / posn[i]:.0%}" for i in sorted(posn)))
    deals = defaultdict(set)
    for r in results:
        deals[r.get("deal")].add(fam(r["winner"]))
    print(f"deals where both models won at least one of the 4 rotations: "
          f"{sum(len(v) == 2 for v in deals.values())} of {len(deals)}")


if __name__ == "__main__":
    main()
