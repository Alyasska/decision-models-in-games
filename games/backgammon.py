"""Backgammon (OpenSpiel): models vs models vs a heuristic bot, duplicate dice, win/loss scoring (no cube).

  ../catan/.venv/bin/python backgammon.py --a drex-max-or --b jev --deals 10
  ../catan/.venv/bin/python backgammon.py --a jev --b heuristic --deals 10

Each deal is played twice with the same dice sequence and the seats swapped, so dice luck cancels.
The board is described from the mover's side (points 1-24, you move toward 1 and bear off from 1-6).
Every option says whether it hits, which of your blots are left in direct range, and your pip count after.
Bots: random; heuristic (pip race + hits + made points - exposed blots, one move deep).
"""
import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import pyspiel

from common import ModelSeat, game_client

SYSTEM = ("You are playing backgammon against one opponent and want to win the game. Each message gives the "
          "position and the legal moves for your dice, each with an id. Pick the single best move. Reply with JSON: "
          "`reason` is one short sentence, `choice` is the id.")
RULES = ("Backgammon, no doubling cube, a win counts 1. Points are numbered from your side: you move your checkers "
         "from 24 toward 1, your home board is points 1-6, and you bear off from there once all 15 are home. "
         "The opponent moves the other way (toward your 24). A single checker (a blot) can be hit and sent to the "
         "bar; checkers on the bar must re-enter first. Two or more checkers make a point the opponent cannot land on.")


def mine(p, i):
    """absolute index 0..23 -> point number from player p's side (p=1 'o' moves 23->0, p=0 'x' moves 0->23)."""
    return i + 1 if p == 1 else 24 - i


def position(s, p):
    txt = str(s)
    bar = re.search(r"Bar:(.*)", txt).group(1)
    sc = re.search(r"Scores, X: (\d+), O: (\d+)", txt)
    off = [int(sc.group(1)), int(sc.group(2))]
    sym = "xo"
    me = {mine(p, i): s.board(p, i) for i in range(24) if s.board(p, i)}
    op = {mine(p, i): s.board(1 - p, i) for i in range(24) if s.board(1 - p, i)}
    return {"me": me, "op": op, "my_bar": bar.count(sym[p]), "op_bar": bar.count(sym[1 - p]),
            "my_off": off[p], "op_off": off[1 - p]}


def pips(pos):
    me = sum(k * n for k, n in pos["me"].items()) + 25 * pos["my_bar"]
    op = sum((25 - k) * n for k, n in pos["op"].items()) + 25 * pos["op_bar"]
    return me, op


def exposed(pos):
    """my blots that an opponent checker (or one entering from the bar) could hit with one die."""
    out = []
    for q, n in pos["me"].items():
        if n != 1:
            continue
        sources = [r for r in pos["op"]] + ([0] if pos["op_bar"] else [])
        if any(0 < q - r <= 6 for r in sources):
            out.append(q)
    return sorted(out)


def heuristic_value(pos):
    me, op = pips(pos)
    made = [k for k, n in pos["me"].items() if n >= 2]
    return ((op - me) / 10 + 1.5 * pos["op_bar"] - 1.2 * len(exposed(pos)) + 0.3 * len(made)
            + 0.5 * sum(1 for k in made if k <= 6) + 0.5 * pos["my_off"])


def describe_move(s, p, a):
    before = position(s, p)
    c = s.clone()
    c.apply_action(a)
    after = position(c, p)
    hit = after["op_bar"] > before["op_bar"]
    blots = exposed(after)
    me, op = pips(after)
    txt = s.action_to_string(p, a).split(" - ", 1)[-1]
    return (f"Move {txt}; " + ("hits an opponent blot; " if hit else "")
            + (f"leaves your blots in direct range on {', '.join(map(str, blots))}; " if blots else "no blots in direct range; ")
            + f"your pip count after {me} (opponent {op})" + ("; bears off" if after["my_off"] > before["my_off"] else ""))


class Bot:
    def __init__(self, kind, seed=0):
        self.label, self.kind, self.rng = kind, kind, random.Random(seed)
        self.cost = 0.0

    def act(self, s, p, legal):
        if self.kind == "random":
            return self.rng.choice(legal)
        best, val = legal[0], -1e9
        for a in legal:
            c = s.clone()
            c.apply_action(a)
            v = heuristic_value(position(c, p)) + self.rng.random() * 1e-6
            if v > val:
                best, val = a, v
        return best


def play_game(seats, seed, trace):
    g = pyspiel.load_game("backgammon")
    s = g.new_initial_state()
    rng = random.Random(seed)
    turns = 0
    while not s.is_terminal():
        if s.is_chance_node():
            outs = s.chance_outcomes()
            s.apply_action(rng.choices([o for o, _ in outs], weights=[w for _, w in outs])[0])
            continue
        p = s.current_player()
        legal = s.legal_actions()
        seat = seats[p]
        turns += 1
        if isinstance(seat, Bot) or len(legal) == 1:
            a = seat.act(s, p, legal) if isinstance(seat, Bot) else legal[0]
        else:
            pos = position(s, p)
            me, op = pips(pos)
            dice = re.search(r"Dice: (\d*)", str(s)).group(1)
            state = {"rules": RULES, "dice": " and ".join(dice) + (" (doubles: you will move twice)" if len(set(dice)) == 1 and len(dice) == 2 else ""),
                     "your_checkers": ", ".join(f"{n} on {k}" for k, n in sorted(pos["me"].items(), reverse=True)) or "none on the board",
                     "opponent_checkers": ", ".join(f"{n} on {k}" for k, n in sorted(pos["op"].items(), reverse=True)) or "none on the board",
                     "on_the_bar": f"you {pos['my_bar']}, opponent {pos['op_bar']}",
                     "borne_off": f"you {pos['my_off']} of 15, opponent {pos['op_off']} of 15",
                     "pip_counts": f"you {me}, opponent {op} (lower is ahead in the race)"}
            texts = [describe_move(s, p, a) for a in legal]
            k = seat.pick(state, "Pick the best backgammon move now.", texts,
                          info={"seed": seed, "turn": turns, "dice": dice, "pips": [me, op], "n_options": len(texts)})
            a = legal[k]
        s.apply_action(a)
    return s.returns()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--deals", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7000)
    a = ap.parse_args()
    out = Path(__file__).parent / "runs" / (time.strftime("%Y%m%d-%H%M%S") + f"-backgammon-{a.a}-vs-{a.b}")
    out.mkdir(parents=True, exist_ok=True)
    trace = []

    def seat(spec, n):
        return Bot(spec, n) if spec in ("random", "heuristic") else ModelSeat(spec, game_client(spec, SYSTEM), trace, n)

    A, B = seat(a.a, 1), seat(a.b, 2)
    wins, per_deal = 0, []
    for d in range(a.deals):
        r1 = play_game([A, B], a.seed + d, trace)
        r2 = play_game([B, A], a.seed + d, trace)
        w = (r1[0] > 0) + (r2[1] > 0)
        wins += w
        per_deal.append(w)
        print(f"deal {d}: {a.a} won {w} of 2", flush=True)
    n = 2 * a.deals
    se = math.sqrt(max(1e-9, (wins / n) * (1 - wins / n) / n))
    summ = {"a": a.a, "b": a.b, "games": n, "a_wins": wins, "a_win_rate": round(wins / n, 3),
            "per_deal": per_deal, "cost_usd": {x.label: round(x.cost, 4) for x in (A, B)}}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    with open(out / "decisions.jsonl", "w") as f:
        for row in trace:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n{a.a} vs {a.b}: {a.a} won {wins} of {n} ({wins / n:.0%}, ±{2 * se:.0%}) | cost {summ['cost_usd']} | {out}")


if __name__ == "__main__":
    main()
