"""How do the models actually play? Behaviour statistics from every real (non-dry) run's decision traces.

  .venv/bin/python analyze.py            # all model seats
  .venv/bin/python analyze.py examples   # plus example decisions per category

Counts are per game in which the model played. The traces keep the chosen option and the top 5 options
with probabilities (Claude: the chosen option and its one-line reason), not the full option list.
"""
import collections
import glob
import json
import re
import statistics
import sys

SEATS = ["drex-max", "drex-max-la", "drex-v2", "drex-finance", "jev", "claude"]


def category(text):
    if text.startswith("Roll"): return "roll"
    if text.startswith("End your turn"): return "end turn"
    if text.startswith("Build a settlement (free)"): return "setup settlement"
    if text.startswith("Build a road (free)"): return "free road"
    if text.startswith("Build a settlement"): return "settlement"
    if text.startswith("Build a road"): return "road"
    if text.startswith("Upgrade"): return "city"
    if text.startswith("Buy a development"): return "buy dev card"
    if text.startswith("Play a knight"): return "knight"
    if text.startswith("Play"): return "other dev card"
    if text.startswith("Trade"): return "bank/harbor trade"
    if text.startswith("Put the robber"): return "robber"
    if text.startswith("Discard"): return "discard"
    return "other"


def load():
    games = []  # (seat label, run, game index, [records of that seat])
    for sf in sorted(glob.glob("runs/*/summary.json")):
        s = json.load(open(sf))
        if s.get("dry"):
            continue
        run = sf.split("/")[1]
        for r in s.get("results", []):
            g = r["game"]
            try:
                rows = [json.loads(l) for l in open(f"runs/{run}/game_{g}.jsonl")]
            except FileNotFoundError:
                continue
            for seat in r.get("models", {}):
                mine = [x for x in rows if x["player"] == seat]
                if mine:
                    games.append((seat, run, g, mine, r))
    return games


def main():
    games = load()
    by = collections.defaultdict(list)
    for seat, run, g, rows, res in games:
        by[seat].append((run, g, rows, res))
    show = "examples" in sys.argv[1:]
    for seat in [s for s in SEATS if s in by] + [s for s in by if s not in SEATS]:
        gs = by[seat]
        n = len(gs)
        rows = [x for _, _, rr, _ in gs for x in rr]
        wins = sum(res["winner"] == seat for *_, res in gs)
        print(f"\n=== {seat}: {n} games, {wins} wins, {len(rows)} decisions ({len(rows) / n:.0f} per game)")

        cat = collections.Counter(category(x["chosen"]) for x in rows)
        print("  chosen per game:", ", ".join(f"{k} {v / n:.1f}" for k, v in cat.most_common()))

        # setup: pips of the starting settlements
        setup = [int(m.group(1)) for x in rows if category(x["chosen"]) == "setup settlement"
                 for m in [re.search(r"total (\d+) pips", x["chosen"])] if m]
        best_seen = [max(int(re.search(r"total (\d+) pips", t).group(1)) for t, _ in x["top"])
                     for x in rows if category(x["chosen"]) == "setup settlement" and x.get("top")]
        if setup:
            print(f"  starting settlements: avg {statistics.mean(setup):.1f} pips"
                  + (f" (best among its top-5 options: avg {statistics.mean(best_seen):.1f})" if best_seen else ""))

        # trades
        tr = [x for x in rows if category(x["chosen"]) == "bank/harbor trade"]
        if tr:
            useless = sum("unlocks nothing new" in x["chosen"] for x in tr)
            kinds = collections.Counter(re.search(r"\((bank 4:1|3:1 harbor|2:1 harbor)\)", x["chosen"]).group(1)
                                        for x in tr)
            print(f"  trades: {len(tr) / n:.1f}/game, {dict(kinds)}, unlock nothing new: {useless}/{len(tr)}")

        # robber (robber prompt and knights)
        rb = [x for x in rows if category(x["chosen"]) == "robber"]
        if rb:
            selfb = sum("YOUR" in x["chosen"].split("; steal")[0] for x in rb)
            nobody = sum("blocks nobody" in x["chosen"] for x in rb)
            desert = sum("the desert" in x["chosen"] for x in rb)
            nosteal = sum("steal from" not in x["chosen"] for x in rb)
            leader = 0
            for x in rb:  # did it steal from the opponent with the most points among the options it saw?
                pts = lambda t: int(m.group(1)) if (m := re.search(r"steal from \w+ \((\d+) points", t)) else -1
                seen = [pts(t) for t, _ in x.get("top", [])] or [pts(x["chosen"])]
                leader += pts(x["chosen"]) >= 0 and pts(x["chosen"]) == max(seen)
            print(f"  robber moves: {len(rb)}, blocks own building {selfb}, blocks nobody {nobody} "
                  f"(desert {desert}), no steal {nosteal}, steals from top-scorer among shown {leader}")

        # discards = cards lost to a 7 (hoarding)
        d = cat.get("discard", 0)
        print(f"  cards discarded to 7s: {d / n:.1f}/game")

        # passing on a build: ended the turn while a city/settlement option was among its top 5
        passed = [x for x in rows if category(x["chosen"]) == "end turn"
                  and any(category(t) in ("city", "settlement") for t, _ in x.get("top", []))]
        ends = cat.get("end turn", 0)
        if any(x.get("top") for x in rows):
            print(f"  ended turn although a city/settlement was among top-5 options: {len(passed)}/{ends}")

        # confidence by phase (models with probabilities)
        conf = collections.defaultdict(list)
        for x in rows:
            if x.get("top"):
                conf[x["prompt"]].append(x["top"][0][1])
        if conf:
            print("  top probability by phase:", ", ".join(
                f"{k.lower()} {statistics.mean(v):.2f} (n={len(v)})" for k, v in sorted(conf.items())))

        if show:
            for c in ("city", "settlement", "bank/harbor trade", "robber", "end turn"):
                ex = [x for x in rows if category(x["chosen"]) == c][:2]
                for x in ex:
                    top = " | ".join(f"{p:.2f} {t[:60]}" for t, p in x.get("top", [])[:3])
                    print(f"    [{c}] {x['chosen'][:110]}" + (f"\n        why: {x['reason']}" if x.get("reason") else "")
                          + (f"\n        top: {top}" if top else ""))


if __name__ == "__main__":
    main()
