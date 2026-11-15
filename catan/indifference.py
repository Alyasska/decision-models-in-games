"""Ore or harbour? Positions where both settlement spots are (nearly) equally good, and what the models pick.

  .venv/bin/python indifference.py find  --games 60             # bot games -> runs/<ts>-indiff/snaps/*.pkl
  .venv/bin/python rollout.py runs/<ts>-indiff --n 200          # verify: paired playouts of both spots
  .venv/bin/python indifference.py ask runs/<ts>-indiff [snap]   # ask models (both option orders) + board page

find: ValueFunction bots play; a candidate is a turn where a player can afford exactly one settlement and has
two legal spots: one touching an ore tile (3+ pips of ore, no harbour) and one on a harbour (no ore).
ask: the most even position by rollouts (or the named snapshot) is put to the models as a forced choice between
the two spots, in both orders (a calibrated, order-blind model gives the same split both times), and a board
page with both spots marked is written next to it (position.html, screenshot position.png).
"""
import json
import pickle
import sys
import time
from pathlib import Path

from catanatron import serialization
from catanatron.game import Game
from catanatron.models.enums import RESOURCES, ActionPrompt, ActionType
from catanatron.models.player import Color
from catanatron.players.value import ValueFunctionPlayer
from catanatron.state_functions import get_actual_victory_points, player_key

from drex_player import ModelPlayer, affordable, pips

COLORS = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]


def spot(game, node):
    m = game.state.board.map
    port = next((("3:1" if r is None else f"2:1 {r.lower()}") for r, ns in m.port_nodes.items() if node in ns), None)
    ore = sum(pips(t.number) for t in m.adjacent_tiles.get(node, []) if t.resource == "ORE")
    return port, ore


def hand(game, color):
    k = player_key(game.state, color)
    return [game.state.player_state[f"{k}_{r}_IN_HAND"] for r in RESOURCES]


class Finder(ValueFunctionPlayer):
    def __init__(self, color, out):
        super().__init__(color)
        self.out, self.found = out, False

    def decide(self, game, playable_actions):
        s = game.state
        if not self.found and s.current_prompt == ActionPrompt.PLAY_TURN and not s.is_initial_build_phase:
            sets = [a for a in playable_actions if a.action_type == ActionType.BUILD_SETTLEMENT]
            h = hand(game, self.color)
            after = [x - c for x, c in zip(h, [1, 1, 1, 1, 0])]
            if len(sets) >= 2 and "settlement" not in affordable(after):
                ore = sorted((a for a in sets if spot(game, a.value)[1] >= 3 and not spot(game, a.value)[0]),
                             key=lambda a: -spot(game, a.value)[1])
                port = [a for a in sets if spot(game, a.value)[0] and spot(game, a.value)[1] == 0]
                if ore and port:
                    self.found = True
                    snap = game.copy()
                    snap.state = snap.state.copy()
                    snap.state.players = [ValueFunctionPlayer(p.color) for p in s.players]
                    mp = ModelPlayer(self.color, None, trace=[], label="probe")
                    name = f"{game.seed}_{self.color.value}.pkl"
                    with open(self.out / name, "wb") as f:
                        pickle.dump({"game": snap, "color": self.color,
                                     "player": {"model": "ore", "action": ore[0]},
                                     "shadow": {"model": "port", "action": port[0]},
                                     "record": {"idx": len(s.action_records), "turn": s.num_turns, "prompt": "PLAY_TURN",
                                                "chosen": mp.action_text(game, ore[0]),
                                                "shadow": mp.action_text(game, port[0])}}, f)
        return super().decide(game, playable_actions)


def find(n_games):
    run = Path("runs") / (time.strftime("%Y%m%d-%H%M%S") + "-indiff")
    (run / "snaps").mkdir(parents=True)
    for seed in range(n_games):
        Game([Finder(c, run / "snaps") for c in COLORS], seed=20_000 + seed).play()
    print(f"{len(list((run / 'snaps').glob('*.pkl')))} candidate positions -> {run}")


def ask(run, name=None):
    run = Path(run)
    rolls = {o["snapshot"]: o for o in json.loads((run / "rollouts.json").read_text())}
    if name is None:  # most even: smallest |difference| relative to its noise, both spots reasonably played
        open_ = [k for k in rolls if 0.25 <= rolls[k]["win_player_move"] <= 0.75]  # skip already-decided games
        name = min(open_ or rolls, key=lambda k: abs(rolls[k]["diff"]) + rolls[k]["se"])
    r = rolls[name]
    snap = pickle.loads((run / "snaps" / name).read_bytes())
    game, color = snap["game"], snap["color"]
    sys.path.insert(0, str(Path(__file__).parent.parent / "games"))
    from common import game_client
    from claude_client import SYSTEM
    mp = ModelPlayer(color, None, trace=[], label="probe")
    state = mp.describe(game)
    texts = {"ore": snap["record"]["chosen"], "port": snap["record"]["shadow"]}
    instr = (f"You are playing Catan as {color.value} and want to win. You can afford exactly one settlement this "
             "turn. Pick the better spot.")
    answers = {}
    for spec in ("drex-max-or", "jev", "clef", "clef-flash", "gpt-sol", "claude"):
        client = game_client(spec, SYSTEM)
        answers[spec] = []
        for order in (("ore", "port"), ("port", "ore")):
            opts = {f"a{i}": texts[k] for i, k in enumerate(order)}
            ans, _, _ = client.choose(state, instr, opts)
            pr = ans.get("probabilities") or {}
            pick = order[int(ans["choice"][1:])] if ans.get("choice") else None
            answers[spec].append({"order": list(order), "pick": pick,
                                  "p_ore": round(pr.get(f"a{order.index('ore')}", float(pick == 'ore')), 3),
                                  "reason": ans.get("reason")})
            print(spec, order, "->", pick, answers[spec][-1]["p_ore"], ans.get("reason") or "")
    out = {"snapshot": name, "color": color.value, "rollouts": r, "options": texts, "answers": answers,
           "hand": dict(zip([x.lower() for x in RESOURCES], hand(game, color))),
           "points": get_actual_victory_points(game.state, color)}
    (run / "answers.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    page(run, game, color, snap, out)


def page(run, game, color, snap, out):
    sys.path.insert(0, str(Path(__file__).parent))
    from make_replay import TEMPLATE
    st = serialization.state_to_json(game)
    recs = st["action_records"]
    vps = [[get_actual_victory_points(game.state, c) for c in game.state.colors]]
    labels = {c.value: ("you: " if c == color else "") + c.value.lower() for c in game.state.colors}
    r = out["rollouts"]
    data = {"labels": labels, "vps": vps, "geometry": serialization.geometry(game),
            "state": {"colors": [c.value for c in game.state.colors], "action_records": recs},
            "trace": [], "winner": None, "title": "Ore or harbour?",
            "subtitle": (f"{color.value.lower()} has {out['points']} points and cards for exactly one settlement · "
                         f"bot playouts: ore spot wins {r['win_player_move']:.0%}, harbour spot {r['win_shadow_move']:.0%} "
                         f"(200 games each, same dice)")}
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    rows = "".join(
        f"<div class='row'><span class='opt'>{m}</span><span class='pct'>{sum(a['p_ore'] for a in v) / 2:.0%} ore</span>"
        f"<div class='track'><div class='fill' style='width:{sum(a['p_ore'] for a in v) / 2 * 100:.0f}%'></div></div></div>"
        f"<div class='reason' style='margin:-2px 0 8px'>ore listed first: {v[0]['pick']} ({v[0]['p_ore']:.0%} ore) · "
        f"harbour listed first: {v[1]['pick']} ({v[1]['p_ore']:.0%} ore)</div>"
        for m, v in out["answers"].items())
    ore_n, port_n = snap["player"]["action"].value, snap["shadow"]["action"].value
    extra = f"""<script>
go(recs.length);
for (const [n, label, col] of [[{ore_n}, "A ore", "#7f8c8d"], [{port_n}, "B harbour", "#2980b9"]]) {{
  const [x, y] = nodePos[n];
  el("circle", {{cx: x, cy: y, r: 17, fill: "none", stroke: col, "stroke-width": 5}});
  const t = el("text", {{x: x, y: y - 24, "text-anchor": "middle", "font-size": 15, "font-weight": 800, fill: "#fff",
    stroke: "#000", "stroke-width": 3, "paint-order": "stroke"}}); t.textContent = label;
}}
document.getElementById("action").textContent = "A: {out['options']['ore'].replace('"', "'")}  |  B: {out['options']['port'].replace('"', "'")}";
const box = document.getElementById("decision");
box.innerHTML = "<div class='label'>Forced choice, asked twice with the order swapped: share for the ore spot</div>" + {json.dumps(rows)};
document.getElementById("log").innerHTML = "<div>hand: {', '.join(f'{v} {k}' for k, v in out['hand'].items())}</div>";
</script></body>"""
    (run / "position.html").write_text(html.replace("</body>", extra, 1))
    print(run / "position.html")


if __name__ == "__main__":
    if sys.argv[1] == "find":
        find(int(sys.argv[sys.argv.index("--games") + 1]) if "--games" in sys.argv else 60)
    else:
        ask(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
