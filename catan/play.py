"""Run Catan games between models and Catanatron bots.

  python play.py --seats drex-max,alphabeta --games 3
  python play.py --seats drex-max,drex-v2,jev,claude --games 5
  python play.py --seats drex-max,value,weighted,vp --shuffle-options
  python play.py --seats fake,alphabeta --dry                 # no API calls

  python play.py --seats drex-max-or,jev,value,vp --seed 700 --games 10 --mirror --shuffle-options

  python play.py --seats drex-max-or,jev,claude,gpt-sol --seed 800 --games 4 --rotate --shuffle-options

Seats: drex-max, drex-max-or (OpenRouter), drex-v2, drex-finance, jev (any of these + "-la" = with lookahead),
claude / claude-sonnet / claude-fable (Claude Code CLI), claude-api, gpt-astra / gpt-sol / gpt-terra or or:<model>
(OpenRouter chat), fake,
and bots: random, weighted, vp, value, alphabeta. Catanatron shuffles the seating order each game.
Writes runs/<timestamp>/game_<n>.jsonl (every model decision) and summary.json.
"""
import argparse
import json
import random
import time
from pathlib import Path

from catanatron import serialization
from catanatron.game import Game, GameAccumulator
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.state_functions import get_actual_victory_points

from drex_player import DrexClient, DrexORClient, JevClient, LookaheadPlayer, ModelPlayer

BOTS = {"random": RandomPlayer, "weighted": WeightedRandomPlayer, "vp": VictoryPointPlayer,
        "value": ValueFunctionPlayer, "alphabeta": AlphaBetaPlayer}
COLORS = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
CLAUDE_CLI = {"claude": "opus", "claude-sonnet": "sonnet", "claude-fable": "claude-fable-5-2"}
DECISION_OR = {"clef": "cloudflare/clef", "clef-flash": "cloudflare/clef-flash", "jev-1.13": "typesafe/jev-1.13",
               "drex-v2-or": "nace-ai/drex-v2.0", "drex-finance-or": "nace-ai/drex-max-finance"}
CHAT = {"gpt-astra": "openai/gpt-6-astra", "gpt-sol": "openai/gpt-6.1-sol", "gpt-terra": "openai/gpt-6.1-terra"}


class VPRecorder(GameAccumulator):
    """Victory points of every color before each action (aligned with state.action_records)."""

    def __init__(self, *args, **kwargs):
        self.vps = []

    def step(self, game_before_action, action):
        st = game_before_action.state
        self.vps.append([get_actual_victory_points(st, c) for c in st.colors])

    def after(self, game):
        st = game.state
        self.vps.append([get_actual_victory_points(st, c) for c in st.colors])


class FakeClient:
    """Same interface as the model clients, picks at random; for testing without spending tokens."""

    def __init__(self, seed=1):
        self.rng = random.Random(seed)
        self.calls = self.input_tokens = 0
        self.cost = self.model_ms = 0.0

    def ask(self, state, questions):
        self.calls += 1
        self.input_tokens += len(json.dumps([state, questions])) // 4
        out = {}
        for name, q in questions.items():
            if q["type"] == "choice":
                ks = list(q["criteria"]); self.rng.shuffle(ks)
                pr = {k: 1.0 / (i + 2) for i, k in enumerate(ks)}
                out[name] = {"type": "choice", "choice": ks[0], "probabilities": pr, "confidence": 0.0}
            else:
                out[name] = {"type": "score", "score": self.rng.uniform(0, 4)}
        return out, {"usage": {}}, 0.0

    def choose(self, state, instructions, options):
        self.calls += 1
        self.input_tokens += len(json.dumps([state, instructions, options])) // 4  # rough estimate
        k = self.rng.choice(list(options))
        return {"choice": k, "confidence": 0.0, "probabilities": {k: 1.0}}, {"usage": {}}, 0.0


def make_client(spec, budget):
    spec = spec.split("#")[0]  # "jev#2" = a second, separate seat of the same model
    if spec.endswith("-la"):  # same client, LookaheadPlayer on top
        return make_client(spec[:-3], budget)
    if spec == "drex-max":
        return DrexClient("drex-max-v1.1", budget)
    if spec == "drex-v2":
        return DrexClient("drex-v2.0", budget)
    if spec == "drex-finance":
        return DrexClient("drex-max-finance", budget)
    if spec == "drex-max-or":  # Drex Max via OpenRouter
        return DrexORClient("nace-ai/drex-max", budget)
    if spec in DECISION_OR or spec.startswith("dm:"):  # other decision models on OpenRouter, e.g. dm:cloudflare/clef
        return DrexORClient(DECISION_OR.get(spec) or spec[3:], budget)
    if spec == "jev":
        return JevClient("typesafe/jev-1.14", budget)
    if spec in CLAUDE_CLI:  # Claude Code CLI, the account logged in on this laptop (subscription)
        from claude_client import ClaudeCLIClient
        return ClaudeCLIClient(CLAUDE_CLI[spec])
    if spec in CHAT or spec.startswith("or:"):  # any chat model on OpenRouter, e.g. or:openai/gpt-6.1-sol
        from chat_client import OpenRouterChatClient
        return OpenRouterChatClient(CHAT.get(spec) or spec[3:])
    if spec == "claude-api":  # Anthropic API key in ~/.anthropic_key
        from claude_client import ClaudeClient
        return ClaudeClient("claude-opus-5-5")
    if spec == "fake":
        return FakeClient()
    raise SystemExit(f"unknown seat {spec!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seats", default="drex-max,alphabeta")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--start", type=int, default=0, help="index of the first game (to split a series over processes)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--budget", type=int, default=600_000, help="max input tokens per decision-model client")
    ap.add_argument("--shuffle-options", action="store_true", help="shuffle the order of legal actions")
    ap.add_argument("--mirror", action="store_true",
                    help="play each deal twice with the first two seats swapped (same board, same seed)")
    ap.add_argument("--rotate", action="store_true",
                    help="play each deal once per seat rotation (4 seats -> 4 games), so every seat gets every place")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry", action="store_true", help="replace every model seat with a random fake")
    a = ap.parse_args()

    seats = a.seats.split(",")
    assert 2 <= len(seats) <= 4, "2 to 4 seats"
    clients = {}
    for s in seats:
        if s not in BOTS and s not in clients:
            clients[s] = make_client("fake" if a.dry else s, a.budget)

    out = Path("runs") / (time.strftime("%Y%m%d-%H%M%S") + (f"-{a.tag}" if a.tag else "") + (f"-s{a.start}" if a.start else ""))
    out.mkdir(parents=True, exist_ok=True)
    import hashlib
    code = {f: hashlib.sha256(Path(f).read_bytes()).hexdigest()[:12] for f in ("play.py", "drex_player.py")}
    results = []
    for g in range(a.start, a.start + a.games):
        trace, players, label = [], [], {}
        deal = g // 2 if a.mirror else g // len(seats) if a.rotate else g
        order = list(seats)
        if a.mirror and g % 2:
            order[0], order[1] = order[1], order[0]
        if a.rotate:
            k = g % len(seats)
            order = seats[k:] + seats[:k]
        for i, (spec, color) in enumerate(zip(order, COLORS)):
            if spec in BOTS:
                p = BOTS[spec](color)
            else:
                cls = LookaheadPlayer if spec.endswith("-la") else ModelPlayer
                p = cls(color, clients[spec], trace=trace, label=spec,
                        shuffle_options=a.shuffle_options, seed=1000 * g + i)
            players.append(p)
            label[color.value] = spec
        seed = (a.seed + deal) if a.seed is not None else None
        game = Game(players, seed=seed)
        t0 = time.time()
        rec = VPRecorder()
        try:
            winner, err = game.play(accumulators=[rec]), None
        except RuntimeError as e:
            winner, err = None, str(e)
        vps = {label[c.value]: get_actual_victory_points(game.state, c) for c in game.state.colors}
        per = {p.label: {"calls": p.calls, "input_tokens": p.input_tokens, "cost_usd": round(p.cost, 4),
                         "avg_s": round(p.wall / p.calls, 2) if p.calls else None}
               for p in players if isinstance(p, ModelPlayer)}
        res = {"game": g, "deal": deal, "seed": game.seed, "winner": label[winner.value] if winner else None, "error": err,
               "turns": game.state.num_turns, "points": vps,
               "seating": [label[c.value] for c in game.state.colors], "models": per,
               "minutes": round((time.time() - t0) / 60, 1)}
        results.append(res)
        with open(out / f"game_{g}.jsonl", "w") as f:
            for row in trace:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        try:  # full record for replays: board geometry + every action of every player
            replay = {"labels": label, "vps": rec.vps, "geometry": serialization.geometry(game),
                      "state": serialization.state_to_json(game)}
            (out / f"game_{g}_replay.json").write_text(json.dumps(replay, default=str))
        except Exception as e:  # a replay is a bonus; never lose the game over it
            print("replay not saved:", e)
        print(json.dumps(res, ensure_ascii=False), flush=True)
        if err:
            break

    wins = {s: sum(r["winner"] == s for r in results) for s in dict.fromkeys(seats)}
    summary = {"seats": seats, "dry": a.dry, "shuffle_options": a.shuffle_options, "mirror": a.mirror, "rotate": a.rotate,
               "code": code,
               "games": len(results), "wins": wins, "results": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("wins:", wins, "| saved", out)


if __name__ == "__main__":
    main()
