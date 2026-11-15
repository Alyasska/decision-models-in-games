"""Heads-up no-limit Hold'em (OpenSpiel universal_poker): models vs models vs simple bots, duplicate format.

  ../catan/.venv/bin/python poker.py --a drex-max-or --b jev --deals 20
  ../catan/.venv/bin/python poker.py --a jev --b equity --deals 20

Blinds 50/100, every hand starts with 20,000 chips each (200 big blinds). Actions: fold, check/call, bet half
the pot, bet the pot, all-in. Duplicate: each deal is played twice with the same cards and the seats swapped,
so card luck cancels. Bots: random, station (always check/call), equity (Monte Carlo hand strength vs a random
hand, bets and calls by pot odds). Model seats ending in -mix sample from their probabilities instead of
taking the top choice. Result: big blinds won per 100 hands by --a against --b.
"""
import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import eval7
import pyspiel

from common import ModelSeat, game_client

PARAMS = {"betting": "nolimit", "numPlayers": 2, "numRounds": 4, "blind": "100 50", "firstPlayer": "2 1 1 1",
          "numSuits": 4, "numRanks": 13, "numHoleCards": 2, "numBoardCards": "0 3 1 1",
          "stack": "20000 20000", "bettingAbstraction": "fchpa"}
STACK, BB = 20000, 100
STREETS = ["preflop", "flop", "turn", "river"]
SUIT = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
SYSTEM = ("You are playing heads-up no-limit Texas Hold'em poker against one opponent over many hands and want to "
          "win as many chips as possible. Each message gives the current situation and the legal actions, each with "
          "an id. Pick the single best action. Reply with JSON: `reason` is one short sentence, `choice` is the id.")
RULES = ("Heads-up no-limit Texas Hold'em. Blinds 50/100. Every hand starts fresh with 20,000 chips each "
         "(200 big blinds). You play many hands against the same opponent; the goal is to win as many chips "
         "as possible over all hands. Best 5-card hand from your 2 cards plus the board wins at showdown.")


def info(state, p):
    s = state.information_state_string(p)
    f = dict(re.match(r"(\w+):? ?(.*)", part).groups() for part in s.strip("[]").split("]["))
    money = [int(x) for x in f["Money"].split()]
    cards = lambda t: [t[i:i + 2] for i in range(0, len(t), 2)]
    return {"round": int(f["Round"]), "money": money, "private": cards(f.get("Private", "")),
            "public": cards(f.get("Public", ""))}


def pretty(cs):
    return " ".join(c[0].replace("T", "10") + SUIT[c[1]] for c in cs) or "none yet"


def hand_name(cards):
    if len(cards) < 5:
        return None
    return eval7.handtype(eval7.evaluate([eval7.Card(c) for c in cards]))


def equity(hole, board, n=250, rng=random):
    known = set(hole + board)
    deck = [r + s for r in "23456789TJQKA" for s in "shdc" if r + s not in known]
    h = [eval7.Card(c) for c in hole]
    b = [eval7.Card(c) for c in board]
    score = 0.0
    for _ in range(n):
        draw = rng.sample(deck, 2 + 5 - len(board))
        opp = [eval7.Card(c) for c in draw[:2]]
        full = b + [eval7.Card(c) for c in draw[2:]]
        a, o = eval7.evaluate(h + full), eval7.evaluate(opp + full)
        score += 1.0 if a > o else 0.5 if a == o else 0.0
    return score / n


class Bot:
    def __init__(self, kind, seed=0):
        self.label, self.kind, self.rng = kind, kind, random.Random(seed)
        self.calls, self.cost = 0, 0.0

    def act(self, state, p, legal, texts):
        if self.kind == "random":
            return self.rng.choice(legal)
        if self.kind == "station":
            return 1 if 1 in legal else legal[0]
        i = info(state, p)
        eq = equity(i["private"], i["public"], rng=self.rng)
        mine, theirs = STACK - i["money"][p], STACK - i["money"][1 - p]
        to_call, pot = theirs - mine, mine + theirs
        if to_call > 0:
            if eq > 0.8 and 2 in legal:
                return 2
            if eq > to_call / (pot + to_call) + 0.05:
                return 1
            return 0 if 0 in legal else 1
        if eq > 0.68 and 2 in legal:
            return 2
        if eq > 0.55 and 4 in legal:
            return 4
        return 1


def option_texts(state, p, legal):
    before = STACK - info(state, p)["money"][p]
    opp = STACK - info(state, p)["money"][1 - p]
    out = []
    for a in legal:
        c = state.clone()
        c.apply_action(a)
        put = (STACK - info(c, p)["money"][p]) - before if not c.is_terminal() else 0
        if a == 0:
            out.append("Fold (give up this hand)")
        elif a == 1:
            out.append("Check" if opp == before else f"Call {opp - before}")
        elif a == 3:
            out.append(f"All-in: put in all your remaining {STACK - before} chips")
        else:
            out.append(f"{'Raise' if opp > before else 'Bet'} {'half the pot' if a == 4 else 'the pot'}: "
                       f"put in {put} more chips (your total in this hand becomes {before + put})")
    return out


def play_hand(seats, seed, trace):
    """seats[0] is OpenSpiel player 0 (big blind), seats[1] player 1 (small blind / button). Returns chips won."""
    g = pyspiel.load_game("universal_poker", PARAMS)
    s = g.new_initial_state()
    rng = random.Random(seed)
    log = []
    while not s.is_terminal():
        if s.is_chance_node():
            s.apply_action(rng.choice(s.legal_actions()))
            continue
        p = s.current_player()
        legal = s.legal_actions()
        texts = option_texts(s, p, legal)
        seat = seats[p]
        i = info(s, p)
        street = STREETS[i["round"]]
        if isinstance(seat, Bot):
            a = seat.act(s, p, legal, texts)
        else:
            mine, theirs = STACK - i["money"][p], STACK - i["money"][1 - p]
            hn = hand_name(i["private"] + i["public"])
            state = {"rules": RULES,
                     "position": ("You are the small blind and the button: you act first before the flop and last "
                                  "on the flop, turn and river.") if p == 1 else
                                 ("You are the big blind: you act last before the flop and first on the flop, turn "
                                  "and river."),
                     "street": street, "your_cards": pretty(i["private"]), "board": pretty(i["public"]),
                     **({"your_best_hand_now": hn} if hn else {}),
                     "pot": mine + theirs, "your_chips_in_this_hand": mine, "opponent_chips_in_this_hand": theirs,
                     "to_call": theirs - mine, "your_chips_behind": STACK - mine, "opponent_chips_behind": STACK - theirs,
                     "action_so_far": "; ".join(("you " if who == p else "opponent ") + what for who, what in log)
                                      or "only the blinds are posted"}
            k = seat.pick(state, "Pick the best poker action now.", texts,
                          info={"seed": seed, "street": street, "cards": pretty(i["private"]),
                                "board": pretty(i["public"]), "to_call": theirs - mine, "pot": mine + theirs})
            a = legal[k]
        log.append((p, f"{texts[legal.index(a)].split(':')[0].lower()} ({street})"))
        s.apply_action(a)
    return s.returns()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--deals", type=int, default=20)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--shuffle", action="store_true", help="show model seats the options in a random order")
    ap.add_argument("--max-usd", type=float, default=None,
                    help="stop after the deal where OpenRouter spend of this match passes this (Claude CLI not counted)")
    a = ap.parse_args()
    out = Path(__file__).parent / "runs" / (time.strftime("%Y%m%d-%H%M%S") + f"-poker-{a.a}-vs-{a.b}")
    out.mkdir(parents=True, exist_ok=True)
    trace = []

    def seat(spec, n):
        if spec in ("random", "station", "equity"):
            return Bot(spec, seed=n)
        return ModelSeat(spec, game_client(spec, SYSTEM), trace, rng_seed=n, shuffle=a.shuffle)

    A, B = seat(a.a, 1), seat(a.b, 2)
    per_deal = []
    for d in range(a.deals):
        seed = a.seed + d
        r1 = play_hand([A, B], seed, trace)  # A big blind, B button
        r2 = play_hand([B, A], seed, trace)  # same cards, seats swapped
        won = (r1[0] + r2[1]) / BB  # A's big blinds over the pair
        per_deal.append(won)
        print(f"deal {d}: {a.a} {won:+.1f} bb", flush=True)
        paid = sum(x.cost for x in (A, B) if type(getattr(x, "client", None)).__name__ != "ClaudeCLIClient")
        if a.max_usd is not None and paid > a.max_usd:
            print(f"stopping: OpenRouter spend ${paid:.3f} passed --max-usd {a.max_usd}", flush=True)
            break
    n = len(per_deal)
    m = sum(per_deal) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in per_deal) / max(1, n - 1))
    bb100 = m / 2 * 100
    se100 = sd / math.sqrt(n) / 2 * 100
    summ = {"a": a.a, "b": a.b, "deals": n, "hands": 2 * n, "a_bb_total": round(sum(per_deal), 1),
            "a_bb_per_100": round(bb100, 1), "ci95": [round(bb100 - 2 * se100, 1), round(bb100 + 2 * se100, 1)],
            "cost_usd": {x.label: round(x.cost, 4) for x in (A, B)}, "per_deal": per_deal}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    with open(out / "decisions.jsonl", "w") as f:
        for row in trace:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n{a.a} vs {a.b}: {sum(per_deal):+.1f} bb over {2 * n} hands = {bb100:+.0f} bb/100 "
          f"(95% CI {bb100 - 2 * se100:+.0f} to {bb100 + 2 * se100:+.0f}) | cost {summ['cost_usd']} | {out}")


if __name__ == "__main__":
    main()
