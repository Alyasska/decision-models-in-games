"""Pokémon battles (gen9randombattle singles) on a local Showdown server via poke-env: models vs bots / each other.

  node 22: ~/.local/node22/bin/node pokemon-showdown start --no-security     (in pokemon-showdown/)
  ../catan/.venv/bin/python pokemon.py --a drex-max-or --b heuristic --battles 6
  ../catan/.venv/bin/python pokemon.py --a drex-max-or --b jev --battles 6

Each turn the model sees both active Pokémon, its bench, field conditions and every legal action (moves with
type, power, accuracy and effectiveness against the current opponent; Terastallize variants; switches with how
the opponent's types hit the switch-in). Bots from poke-env: random, maxpower (strongest move), heuristic
(SimpleHeuristicsPlayer). Teams are random every battle, so results are noisy: small samples are for a look only.
"""
import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer

from common import ModelSeat, game_client

SYSTEM = ("You are playing a Pokémon battle (Pokémon Showdown, Gen 9 random battle, singles) and want to win. "
          "Each message gives the battle state and your legal actions, each with an id. Pick the single best action. "
          "Reply with JSON: `reason` is one short sentence, `choice` is the id.")
RULES = ("Gen 9 random battle, singles, 6 vs 6 with random teams. Each turn pick a move for your active Pokémon or "
         "switch to a healthy benched Pokémon (switching uses the turn). A Pokémon faints at 0 HP; you win when all six "
         "opponent Pokémon faint. Once per battle you may Terastallize, changing your active Pokémon's type to its tera "
         "type (moves of that type get a bonus). Damage depends on type effectiveness (0x, 0.25x, 0.5x, 1x, 2x, 4x), "
         "same-type attack bonus, power, stats and stat boosts.")
BOTS = {"random": RandomPlayer, "maxpower": MaxBasePowerPlayer, "heuristic": SimpleHeuristicsPlayer}


def tname(t):
    return t.name.lower() if t is not None else "?"


def mon(p, own=False):
    if p is None:
        return "none"
    s = f"{p.species} ({'/'.join(tname(t) for t in p.types if t)}), HP {round(p.current_hp_fraction * 100)}%"
    if p.status:
        s += f", status {p.status.name.lower()}"
    b = {k: v for k, v in (p.boosts or {}).items() if v}
    if b:
        s += ", boosts " + ", ".join(f"{k} {v:+d}" for k, v in b.items())
    if own and p.item:
        s += f", item {p.item}"
    if own and p.ability:
        s += f", ability {p.ability}"
    if p.is_terastallized:
        s += f", terastallized ({tname(p.tera_type)})"
    return s


def acc(m):
    a = m.accuracy
    return "never misses" if a is True or a is None else f"accuracy {round(a * 100)}%"


class ModelPlayer(Player):
    def __init__(self, seat, trace, **kw):
        super().__init__(**kw)
        self.seat, self.trace = seat, trace

    async def choose_move(self, battle):
        me, opp = battle.active_pokemon, battle.opponent_active_pokemon
        orders, texts = [], []
        for m in battle.available_moves:
            t = f"Use {m.id}: {tname(m.type)} {m.category.name.lower()} move, power {m.base_power}, {acc(m)}"
            if m.base_power and opp is not None:
                t += f"; {opp.damage_multiplier(m):g}x effective against {opp.species}"
            if me is not None and m.type in me.types and m.base_power:
                t += "; same-type bonus"
            orders.append(self.create_order(m))
            texts.append(t)
            if battle.can_tera and me is not None and me.tera_type is not None:
                orders.append(self.create_order(m, terastallize=True))
                texts.append(t + f"; and Terastallize into {tname(me.tera_type)} type first")
        for p in battle.available_switches:
            worst = max((p.damage_multiplier(t) for t in (opp.types if opp else []) if t), default=1)
            orders.append(self.create_order(p))
            texts.append(f"Switch to {mon(p, own=True)}; the opponent's {opp.species if opp else '?'} types hit it "
                         f"at up to {worst:g}x")
        if not orders:
            return self.choose_random_move(battle)
        bench = [mon(p, own=True) for p in battle.team.values() if not p.active]
        seen = [mon(p) for p in battle.opponent_team.values() if not p.active]
        state = {"rules": RULES, "turn": battle.turn,
                 "your_active": mon(me, own=True), "opponent_active": mon(opp),
                 "your_bench": "; ".join(bench) or "none",
                 "opponent_pokemon_seen_on_bench": "; ".join(seen) or "none yet",
                 "opponent_pokemon_left": 6 - sum(p.fainted for p in battle.opponent_team.values()),
                 "you_can_terastallize": bool(battle.can_tera),
                 "weather": ", ".join(w.name.lower() for w in battle.weather) or "none",
                 "your_side": ", ".join(c.name.lower() for c in battle.side_conditions) or "nothing",
                 "opponent_side": ", ".join(c.name.lower() for c in battle.opponent_side_conditions) or "nothing"}
        k = await asyncio.to_thread(self.seat.pick, state, "Pick the best action this turn.", texts,
                                    {"battle": battle.battle_tag, "turn": battle.turn, "active": me.species if me else None,
                                     "opponent": opp.species if opp else None})
        return orders[k]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--battles", type=int, default=6)
    a = ap.parse_args()
    out = Path(__file__).parent / "runs" / (time.strftime("%Y%m%d-%H%M%S") + f"-pokemon-{a.a}-vs-{a.b}")
    out.mkdir(parents=True, exist_ok=True)
    trace = []
    tag = f"{random.randrange(10**4):04d}"

    def player(spec, n):
        acct = AccountConfiguration(f"{spec.replace('-', '')[:10]}{n}{tag}"[:18], None)
        kw = dict(account_configuration=acct, battle_format="gen9randombattle",
                  server_configuration=LocalhostServerConfiguration, max_concurrent_battles=1)
        if spec in BOTS:
            return BOTS[spec](**kw)
        return ModelPlayer(ModelSeat(spec, game_client(spec, SYSTEM), trace, n), trace, **kw)

    A, B = player(a.a, 1), player(a.b, 2)
    t = time.time()
    await A.battle_against(B, n_battles=a.battles)
    won = A.n_won_battles
    n = A.n_finished_battles
    summ = {"a": a.a, "b": a.b, "battles": n, "a_wins": won, "minutes": round((time.time() - t) / 60, 1),
            "turns": [b.turn for b in A.battles.values()],
            "cost_usd": {x.seat.label: round(x.seat.cost, 4) for x in (A, B) if isinstance(x, ModelPlayer)}}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    with open(out / "decisions.jsonl", "w") as f:
        for row in trace:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{a.a} vs {a.b}: {a.a} won {won} of {n} | {summ['minutes']} min | cost {summ['cost_usd']} | {out}")


if __name__ == "__main__":
    asyncio.run(main())
