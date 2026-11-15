"""Shadow games: one model plays, a second model is asked the very same question at every decision.

  .venv/bin/python shadow.py --seats "drex-max-or>jev,jev>drex-max-or,value,vp" --seed 1100 --games 4 --mirror --snapshots

"A>B" = A decides, B shadows (same state text, same options in the same order; its answer is only recorded).
Writes runs/<ts>-shadow/game_<n>.jsonl where each decision of a shadowed seat also has `shadow`,
`shadow_top`, `agree`, and on disagreement `options` with both models' probabilities. Prints agreement rates.

--snapshots: at every disagreement also saves snaps/<game>_<idx>.pkl = the position (a game copy whose players
are all ValueFunction bots, so no model client or key is inside) plus both candidate actions; rollout.py then
plays each candidate out many times with bots to see whose choice was better.
"""
import argparse
import collections
import json
import pickle
import time
from pathlib import Path

from catanatron.game import Game
from catanatron.models.enums import ActionType
from catanatron.players.value import ValueFunctionPlayer
from catanatron.state_functions import get_actual_victory_points

from drex_player import MAX_OPTIONS, ModelPlayer
from play import BOTS, COLORS, make_client


class ShadowPlayer(ModelPlayer):
    def __init__(self, color, client, shadow_client, shadow_label, snap_dir=None, game_no=0, **kw):
        super().__init__(color, client, **kw)
        self.shadow_client, self.shadow_label = shadow_client, shadow_label
        self.snap_dir, self.game_no = snap_dir, game_no

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        actions = list(playable_actions)  # same option list as ModelPlayer.decide
        if len(actions) > MAX_OPTIONS:
            roads = [a for a in actions if a.action_type == ActionType.BUILD_ROAD]
            rest = [a for a in actions if a.action_type != ActionType.BUILD_ROAD]
            actions = rest + roads[: max(0, MAX_OPTIONS - len(rest))]
        if self.shuffle:
            self.rng.shuffle(actions)
        options = {f"a{i}": self.action_text(game, a) for i, a in enumerate(actions)}
        state = self.describe(game)
        instructions = (f"You are playing Catan as {self.color.value} and want to win. "
                        "Pick the single best action to take right now.")
        tok0, cost0 = self.client.input_tokens, getattr(self.client, "cost", 0.0)
        ans, raw, wall = self.client.choose(state, instructions, options)
        self.calls += 1
        self.input_tokens += self.client.input_tokens - tok0
        self.cost += getattr(self.client, "cost", 0.0) - cost0
        self.wall += wall
        try:
            s_ans, _, _ = self.shadow_client.choose(state, instructions, options)
        except Exception as e:  # a failed shadow call must not stop the game
            s_ans = {"choice": None, "error": str(e)[:200]}

        def index(k):
            return int(k[1:]) if k and k.startswith("a") and k[1:].isdigit() and int(k[1:]) < len(actions) else None
        mi, si = index(ans.get("choice")), index(s_ans.get("choice"))
        chosen = actions[mi if mi is not None else 0]
        mp, sp = ans.get("probabilities", {}), s_ans.get("probabilities", {})
        rec = {"player": self.label, "color": self.color.value, "idx": len(game.state.action_records),
               "turn": game.state.num_turns, "prompt": game.state.current_prompt.value,
               "n_options": len(actions), "chosen": options.get(ans.get("choice"), "?"),
               "confidence": ans.get("confidence"),
               "top": [(options[k], round(p, 4)) for k, p in sorted(mp.items(), key=lambda kv: -kv[1])[:5] if k in options],
               "input_tokens": raw.get("usage", {}).get("input_tokens"), "wall_s": round(wall, 2),
               "shadow_model": self.shadow_label, "shadow": options.get(s_ans.get("choice")),
               "shadow_top": [(options[k], round(p, 4)) for k, p in sorted(sp.items(), key=lambda kv: -kv[1])[:5] if k in options],
               "shadow_error": s_ans.get("error"), "agree": mi == si}
        if mi != si and si is not None:
            rec["options"] = {k: [t, round(mp.get(k, 0), 4), round(sp.get(k, 0), 4)] for k, t in options.items()}
            if self.snap_dir:
                snap = game.copy()  # position before the move; bots in every seat, no clients inside
                snap.state = snap.state.copy()
                snap.state.players = [ValueFunctionPlayer(p.color) for p in game.state.players]
                name = f"{self.game_no}_{rec['idx']}.pkl"
                with open(self.snap_dir / name, "wb") as f:
                    pickle.dump({"game": snap, "color": self.color,
                                 "player": {"model": self.label, "action": chosen},
                                 "shadow": {"model": self.shadow_label, "action": actions[si]},
                                 "record": {k: rec[k] for k in ("idx", "turn", "prompt", "chosen", "shadow")}}, f)
                rec["snapshot"] = name
        self.trace.append(rec)
        return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seats", required=True)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--snapshots", action="store_true")
    ap.add_argument("--budget", type=int, default=3_000_000)
    ap.add_argument("--tag", default="shadow")
    a = ap.parse_args()
    seats = a.seats.split(",")
    out = Path("runs") / (time.strftime("%Y%m%d-%H%M%S") + f"-{a.tag}")
    out.mkdir(parents=True, exist_ok=True)
    snap_dir = None
    if a.snapshots:
        snap_dir = out / "snaps"
        snap_dir.mkdir()
    clients = {}
    for s in seats:
        if s not in BOTS:
            m, sh = s.split(">")
            clients[s] = (make_client(m, a.budget), make_client(sh, a.budget))
    results, allrows = [], []
    for g in range(a.games):
        deal = g // 2 if a.mirror else g
        order = list(seats)
        if a.mirror and g % 2:
            order[0], order[1] = order[1], order[0]
        trace, players, label = [], [], {}
        for i, (spec, color) in enumerate(zip(order, COLORS)):
            if spec in BOTS:
                p = BOTS[spec](color)
            else:
                m, sh = spec.split(">")
                p = ShadowPlayer(color, clients[spec][0], clients[spec][1], sh, snap_dir=snap_dir, game_no=g,
                                 trace=trace, label=m, shuffle_options=True, seed=1000 * g + i)
            players.append(p)
            label[color.value] = spec.split(">")[0]
        game = Game(players, seed=a.seed + deal)
        t0 = time.time()
        try:
            winner, err = game.play(), None
        except RuntimeError as e:
            winner, err = None, str(e)
        res = {"game": g, "deal": deal, "seed": game.seed, "winner": label[winner.value] if winner else None,
               "error": err, "turns": game.state.num_turns,
               "points": {label[c.value]: get_actual_victory_points(game.state, c) for c in game.state.colors},
               "seating": [label[c.value] for c in game.state.colors], "minutes": round((time.time() - t0) / 60, 1)}
        results.append(res)
        with open(out / f"game_{g}.jsonl", "w") as f:
            for row in trace:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        allrows += trace
        print(json.dumps(res, ensure_ascii=False), flush=True)
        if err:
            break
    stats = collections.defaultdict(lambda: collections.Counter())
    for r in allrows:
        if r.get("shadow") is not None:
            key = f"{r['player']} plays, {r['shadow_model']} shadows"
            stats[key]["n"] += 1
            stats[key]["agree"] += r["agree"]
            stats[key][f"n:{r['prompt']}"] += 1
            stats[key][f"agree:{r['prompt']}"] += r["agree"]
    summary = {"seats": seats, "games": len(results), "results": results,
               "agreement": {k: {"decisions": v["n"], "agree": round(v["agree"] / v["n"], 3),
                                 "by_phase": {p[2:]: round(v["agree:" + p[2:]] / v[p], 3)
                                              for p in v if p.startswith("n:")}}
                             for k, v in stats.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary["agreement"], indent=1), "\nsaved", out)


if __name__ == "__main__":
    main()
