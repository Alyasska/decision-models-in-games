"""Drex plays Catan: a Catanatron Player that asks Nace's Drex decision model to pick each move.

Every decision with more than one legal action becomes one Drex `choice` question:
the state is described in plain text, each legal action is one option, Drex returns
probabilities and we take the most likely option. Single-option decisions (roll, forced
discard, ...) are played without a call.

The key is read from ~/.nace_key and never printed.
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import dotenv_values

from catanatron.models.enums import (
    ActionPrompt, ActionType, CITY, DEVELOPMENT_CARDS, RESOURCES, SETTLEMENT,
)
from catanatron.models.board import STATIC_GRAPH
from catanatron.models.player import Player
from catanatron.state_functions import (
    get_longest_road_length, get_visible_victory_points, player_key,
    player_num_dev_cards, player_num_resource_cards,
)

MAX_OPTIONS = 255
RES = {"WOOD": "wood", "BRICK": "brick", "SHEEP": "sheep", "WHEAT": "wheat", "ORE": "ore"}
COSTS = ("Costs: road = 1 wood + 1 brick; settlement = 1 wood + 1 brick + 1 sheep + 1 wheat; "
         "city = 2 wheat + 3 ore (upgrades a settlement); development card = 1 sheep + 1 wheat + 1 ore.")
RULES = ("Catan, base game. First player to 10 victory points wins. Settlement = 1 point, city = 2, "
         "longest road (5+) = 2, largest army (3+ knights) = 2, some development cards = 1. "
         "A dice roll of N pays every settlement (1 card) and city (2 cards) touching a tile numbered N. "
         "Pips = how often a number is rolled (6 and 8 = 5 pips, 2 and 12 = 1 pip). "
         "A 7 moves the robber, which blocks a tile and steals a card; players with more than 7 cards discard half.")


BUILD_COSTS = {"road": [1, 1, 0, 0, 0], "settlement": [1, 1, 1, 1, 0],
               "city": [0, 0, 0, 2, 3], "development card": [0, 0, 1, 1, 1]}


def pips(number):
    return 0 if number is None else 6 - abs(7 - number)


def affordable(hand):
    return [b for b, c in BUILD_COSTS.items() if all(h >= x for h, x in zip(hand, c))]


def missing(hand, build):
    need = [max(0, x - h) for h, x in zip(hand, BUILD_COSTS[build])]
    return ", ".join(f"{n} {RES[r]}" for n, r in zip(need, RESOURCES) if n)


NACE_URL = "https://console.nace.ai/v1/systemone"
OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"


# .env files of this project only (catan/ first, then nace-ai/); never walks further up.
ENV_FILES = [Path(__file__).resolve().parent / ".env", Path(__file__).resolve().parent.parent / ".env"]


def load_key(env_var, key_path=None):
    """API key from the environment, else the project .env, else a key file. The value is never printed.

    Only this one variable is read from .env (nothing is copied into os.environ, so subprocesses such as
    `claude -p` don't inherit other keys from the file).
    """
    import logging
    logging.getLogger("dotenv.main").setLevel(logging.ERROR)  # malformed lines elsewhere in .env: stay quiet
    key = os.environ.get(env_var, "").strip()
    for f in ENV_FILES:
        if not key and f.is_file():
            key = (dotenv_values(f).get(env_var) or "").strip()
    if not key and key_path and os.path.exists(os.path.expanduser(key_path)):
        key = open(os.path.expanduser(key_path)).read().strip()
    if not key:
        raise SystemExit(f"no {env_var} in the environment / .env" + (f" and no {key_path}" if key_path else ""))
    return key


class SystemOneClient:
    """Decision-model client for the System One format (Drex on Nace, Jev on OpenRouter)."""

    def __init__(self, url, key, model, max_input_tokens=None):
        self.key = key
        self.url, self.model = url, model
        self.max_input_tokens = max_input_tokens
        self.calls = 0
        self.input_tokens = 0
        self.model_ms = 0.0
        self.cost = 0.0

    def choose(self, state, instructions, options):
        answers, out, wall = self.ask(state, {"move": {"type": "choice", "instructions": instructions,
                                                       "criteria": options}})
        ans = answers["move"]
        if ans.get("type") == "refusal" or "choice" not in ans:
            ans = {"choice": None, "confidence": None, "probabilities": {}, "refused": True}
        return ans, out, wall

    def ask(self, state, questions):
        """Any set of named System One questions in one request; returns (answers, raw response, seconds)."""
        if self.max_input_tokens and self.input_tokens >= self.max_input_tokens:
            raise RuntimeError(f"token budget reached: {self.input_tokens} input tokens")
        body = {"model": self.model, "state": state, "questions": questions}
        data = json.dumps(body).encode()
        for attempt in range(5):
            req = urllib.request.Request(self.url, data=data, headers={
                "Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
            t = time.time()
            try:
                out = json.loads(urllib.request.urlopen(req, timeout=120).read().decode())
                break
            except urllib.error.HTTPError as e:
                msg = e.read().decode()[:300]
                if e.code in (408, 429, 500, 502, 503, 504) and attempt < 4:
                    time.sleep(2 ** attempt * 2)
                    continue
                raise RuntimeError(f"{self.model} HTTP {e.code}: {msg}")
            except (urllib.error.URLError, TimeoutError):
                if attempt < 4:
                    time.sleep(2 ** attempt * 2)
                    continue
                raise
        wall = time.time() - t
        usage = out.get("usage", {})
        self.calls += 1
        self.input_tokens += usage.get("input_tokens", 0)
        self.model_ms += out.get("evaluation_time_ms", 0) or 0
        self.cost += usage.get("cost", 0) or 0
        return out["answers"], out, wall


def DrexClient(model="drex-latest", max_input_tokens=None):
    return SystemOneClient(NACE_URL, load_key("NACE_API_KEY", "~/.nace_key"), model, max_input_tokens)


def DrexORClient(model="nace-ai/drex-max", max_input_tokens=None):
    """Drex through OpenRouter: same gateway and billing as Jev, so speed and cost compare directly."""
    return SystemOneClient(OPENROUTER_URL, load_key("OPENROUTER_API_KEY", "~/.openrouter_key"), model,
                           max_input_tokens)


def JevClient(model="typesafe/jev-1.14", max_input_tokens=None):
    return SystemOneClient(OPENROUTER_URL, load_key("OPENROUTER_API_KEY", "~/.openrouter_key"), model,
                           max_input_tokens)


class ModelPlayer(Player):
    """A Catanatron player whose every real decision is made by a model client."""

    def __init__(self, color, client, trace=None, label="model", shuffle_options=False, seed=0):
        super().__init__(color)
        self.client = client
        self.trace = trace  # list to append decision records to
        self.label = label
        self.shuffle = shuffle_options
        self.rng = __import__("random").Random(seed)
        self.calls = self.input_tokens = 0
        self.cost = self.wall = 0.0

    # ---------- describing the game ----------
    def node_text(self, game, node_id):
        m = game.state.board.map
        tiles = m.adjacent_tiles.get(node_id, [])
        parts, total = [], 0
        for t in tiles:
            if t.resource is None:
                parts.append("desert")
                continue
            p = pips(t.number)
            total += p
            parts.append(f"{RES[t.resource]} {t.number} ({p} pips)")
        port = None
        for res, nodes in m.port_nodes.items():
            if node_id in nodes:
                port = "3:1 any" if res is None else f"2:1 {RES[res]}"
        txt = f"spot {node_id}: " + (", ".join(parts) if parts else "no land") + f"; total {total} pips"
        if port:
            txt += f"; harbor {port}"
        return txt, total

    def production(self, game, color):
        b = game.state.buildings_by_color[color]
        per = {r: 0 for r in RESOURCES}
        for kind, mult in ((SETTLEMENT, 1), (CITY, 2)):
            for n in b.get(kind, []):
                for t in game.state.board.map.adjacent_tiles.get(n, []):
                    if t.resource is not None and t.number is not None:
                        per[t.resource] += pips(t.number) * mult
        return per

    def hand(self, game):
        k = player_key(game.state, self.color)
        return [game.state.player_state[f"{k}_{r}_IN_HAND"] for r in RESOURCES]

    def describe(self, game):
        s = game.state
        me = self.color
        k = player_key(s, me)
        ps = s.player_state
        hand = ", ".join(f"{ps[f'{k}_{r}_IN_HAND']} {RES[r]}" for r in RESOURCES)
        devs = [f"{ps[f'{k}_{d}_IN_HAND']} {d.lower().replace('_', ' ')}" for d in DEVELOPMENT_CARDS
                if ps[f"{k}_{d}_IN_HAND"]]
        b = s.buildings_by_color[me]
        prod = self.production(game, me)
        my_ports = sorted(("3:1 any" if r is None else f"2:1 {RES[r]}")
                          for r in game.state.board.get_player_port_resources(me))
        robber = s.board.map.land_tiles.get(s.board.robber_coordinate)
        robber_txt = ("desert" if robber is None or robber.resource is None
                      else f"{RES[robber.resource]} {robber.number}")
        mine = (f"You are {me.value}. Victory points: {ps[f'{k}_ACTUAL_VICTORY_POINTS']} of {game.vps_to_win}. "
                f"Hand: {hand}. Development cards: {', '.join(devs) if devs else 'none'}. "
                f"Knights played: {ps[f'{k}_PLAYED_KNIGHT']}. "
                f"Settlements on: {', '.join(self.node_text(game, n)[0] for n in b.get(SETTLEMENT, [])) or 'none'}. "
                f"Cities on: {', '.join(self.node_text(game, n)[0] for n in b.get(CITY, [])) or 'none'}. "
                f"Income in pips per resource: {', '.join(f'{RES[r]} {prod[r]}' for r in RESOURCES)}. "
                f"Harbors: {', '.join(my_ports) or 'none'}. "
                f"Pieces left: {ps[f'{k}_ROADS_AVAILABLE']} roads, {ps[f'{k}_SETTLEMENTS_AVAILABLE']} settlements, "
                f"{ps[f'{k}_CITIES_AVAILABLE']} cities. Longest road: {get_longest_road_length(s, me)}"
                + (" (you hold Longest Road)" if ps[f"{k}_HAS_ROAD"] else "")
                + (" You hold Largest Army." if ps[f"{k}_HAS_ARMY"] else ""))
        opps = []
        for c in s.colors:
            if c == me:
                continue
            ok = player_key(s, c)
            ob = s.buildings_by_color[c]
            opps.append(
                f"{c.value}: {get_visible_victory_points(s, c)} visible points, "
                f"{player_num_resource_cards(s, c)} cards in hand, {player_num_dev_cards(s, c)} development cards, "
                f"{len(ob.get(SETTLEMENT, []))} settlements, {len(ob.get(CITY, []))} cities, "
                f"longest road {get_longest_road_length(s, c)}, knights played {ps[f'{ok}_PLAYED_KNIGHT']}"
                + (", holds Longest Road" if ps[f"{ok}_HAS_ROAD"] else "")
                + (", holds Largest Army" if ps[f"{ok}_HAS_ARMY"] else ""))
        phase = {
            ActionPrompt.BUILD_INITIAL_SETTLEMENT: "Setup: place a starting settlement (free).",
            ActionPrompt.BUILD_INITIAL_ROAD: "Setup: place a starting road (free) next to your new settlement.",
            ActionPrompt.PLAY_TURN: "Your turn.",
            ActionPrompt.DISCARD: "You must discard.",
            ActionPrompt.MOVE_ROBBER: "Move the robber.",
        }.get(s.current_prompt, str(s.current_prompt))
        hv = self.hand(game)
        can = affordable(hv)
        need = "; ".join(f"{b} needs {missing(hv, b)} more" for b in BUILD_COSTS if b not in can)
        return {"rules": RULES + " " + COSTS, "phase": phase, "you": mine,
                "you_can_afford_now": ", ".join(can) or "nothing", "you_are_missing": need or "nothing",
                "opponents": " | ".join(opps), "robber_on": robber_txt, "turn": str(s.num_turns)}

    def action_text(self, game, a):
        t, v = a.action_type, a.value
        s = game.state
        if t == ActionType.ROLL:
            return "Roll the dice"
        if t == ActionType.END_TURN:
            return "End your turn"
        if t == ActionType.BUILD_SETTLEMENT:
            free = " (free)" if s.is_initial_build_phase else ""
            prod = self.production(game, self.color)
            new = sorted({RES[t.resource] for t in s.board.map.adjacent_tiles.get(v, [])
                          if t.resource is not None and prod[t.resource] == 0})
            return (f"Build a settlement{free} on " + self.node_text(game, v)[0]
                    + (f"; new resources for you: {', '.join(new)}" if new else "; no new resource types"))
        if t == ActionType.BUILD_CITY:
            return "Upgrade your settlement to a city (doubles its income) on " + self.node_text(game, v)[0]
        if t == ActionType.BUILD_ROAD:
            a_, b_ = v
            mine = set()
            for kind in (SETTLEMENT, CITY):
                mine |= set(s.buildings_by_color[self.color].get(kind, []))
            for (x, y), col in s.board.roads.items():
                if col == self.color:
                    mine |= {x, y}
            far = b_ if a_ in mine else a_
            txt, _ = self.node_text(game, far)
            free = " (free)" if s.is_initial_build_phase or s.is_road_building else ""
            land = s.board.map.land_nodes
            if far in s.board.board_buildable_ids:
                reach = "you could build a settlement right there next"
            else:
                nxt = [(self.node_text(game, n)[1], n) for n in STATIC_GRAPH.neighbors(far)
                       if n not in mine and n in land and n in s.board.board_buildable_ids]
                reach = (f"one more road reaches an open settlement spot ({max(nxt)[0]} pips)" if nxt
                         else "no open settlement spot within reach")
            return f"Build a road{free} toward {txt}; {reach}"
        if t == ActionType.BUY_DEVELOPMENT_CARD:
            return "Buy a development card (knight, progress card or 1 victory point)"
        if t == ActionType.PLAY_KNIGHT_CARD:
            return "Play a knight: move the robber and steal a card (counts toward Largest Army)"
        if t == ActionType.PLAY_YEAR_OF_PLENTY:
            return "Play Year of Plenty: take " + " + ".join(RES[r] for r in v if r) + " from the bank"
        if t == ActionType.PLAY_MONOPOLY:
            return f"Play Monopoly on {RES[v]}: every opponent gives you all their {RES[v]}"
        if t == ActionType.PLAY_ROAD_BUILDING:
            return "Play Road Building: build 2 free roads"
        if t == ActionType.MARITIME_TRADE:
            give = [r for r in v[:-1] if r]
            how = {4: "bank 4:1", 3: "3:1 harbor", 2: "2:1 harbor"}.get(len(give), "")
            hv = self.hand(game)
            after = list(hv)
            after[RESOURCES.index(give[0])] -= len(give)
            after[RESOURCES.index(v[-1])] += 1
            gained = [b for b in affordable(after) if b not in affordable(hv)]
            return (f"Trade {len(give)} {RES[give[0]]} for 1 {RES[v[-1]]} ({how})"
                    + (f"; afterwards you can afford: {', '.join(gained)}" if gained else "; unlocks nothing new"))
        if t == ActionType.MOVE_ROBBER:
            coord, victim = v[0], v[1]
            tile = s.board.map.land_tiles.get(coord)
            where = "the desert" if tile is None or tile.resource is None else \
                f"{RES[tile.resource]} {tile.number} ({pips(tile.number)} pips)"
            hit = []
            for c in s.colors:
                for kind in (SETTLEMENT, CITY):
                    for n in s.buildings_by_color[c].get(kind, []):
                        if tile is not None and n in tile.nodes.values():
                            hit.append(("YOUR " if c == self.color else c.value + " ") + kind.lower())
            txt = f"Put the robber on {where}"
            txt += "; blocks " + ", ".join(hit) if hit else "; blocks nobody"
            if victim is not None:
                txt += (f"; steal from {victim.value} ({get_visible_victory_points(s, victim)} points, "
                        f"{player_num_resource_cards(s, victim)} cards)")
            return txt
        if t == ActionType.DISCARD_RESOURCE:
            return f"Discard 1 {RES[v]}"
        if t in (ActionType.OFFER_TRADE, ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE,
                 ActionType.CONFIRM_TRADE):
            give = ", ".join(f"{n} {RES[r]}" for n, r in zip(v[:5], RESOURCES) if n) or "nothing"
            get = ", ".join(f"{n} {RES[r]}" for n, r in zip(v[5:10], RESOURCES) if n) or "nothing"
            verb = {ActionType.OFFER_TRADE: "Offer a trade", ActionType.ACCEPT_TRADE: "Accept the trade",
                    ActionType.REJECT_TRADE: "Reject the trade", ActionType.CONFIRM_TRADE: "Confirm the trade"}[t]
            who = f" with {v[10].value}" if t == ActionType.CONFIRM_TRADE and len(v) > 10 else ""
            return f"{verb}{who}: the offering player gives {give} and gets {get}"
        if t == ActionType.CANCEL_TRADE:
            return "Cancel the trade"
        return f"{t.value} {v}"

    # ---------- deciding ----------
    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        actions = list(playable_actions)
        if len(actions) > MAX_OPTIONS:  # rare: keep everything but the weakest roads
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
        pick = ans.get("choice")
        idx = int(pick[1:]) if pick and pick.startswith("a") and pick[1:].isdigit() else 0
        chosen = actions[idx] if idx < len(actions) else actions[0]
        if self.trace is not None:
            probs = ans.get("probabilities", {})
            top = sorted(probs.items(), key=lambda kv: -kv[1])[:5]
            self.trace.append({
                "player": self.label, "color": self.color.value,
                "idx": len(game.state.action_records),
                "turn": game.state.num_turns, "prompt": game.state.current_prompt.value,
                "n_options": len(actions), "chosen": options.get(pick, "?"),
                "confidence": ans.get("confidence"),
                "top": [(options[k], round(p, 4)) for k, p in top if k in options],
                "input_tokens": raw.get("usage", {}).get("input_tokens"),
                "model_ms": raw.get("evaluation_time_ms"), "wall_s": round(wall, 2),
                "reason": ans.get("reason"), "refused": ans.get("refused", False),
            })
        return chosen



SCORE_LEVELS = ["far behind, very unlikely to win", "behind", "roughly even", "ahead", "far ahead, likely to win"]


class LookaheadPlayer(ModelPlayer):
    """Two-stage player for decision models.

    1. Drex ranks every legal move (one `choice` question), as ModelPlayer does.
    2. If it is not already sure, the engine plays each of its top-k moves on a copy of the game, and Drex
       scores every resulting position in one batched request (one `score` question per position).
    The move with the best position score is played; Drex's first-stage probability breaks ties.
    """

    def __init__(self, *args, k=3, sure=0.85, **kwargs):
        super().__init__(*args, **kwargs)
        self.k, self.sure = k, sure

    def _ask(self, state, questions):
        tok0 = self.client.input_tokens
        answers, raw, wall = self.client.ask(state, questions)
        self.calls += 1
        self.input_tokens += self.client.input_tokens - tok0
        self.wall += wall
        return answers, raw, wall

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        actions = list(playable_actions)
        if len(actions) > MAX_OPTIONS:
            roads = [a for a in actions if a.action_type == ActionType.BUILD_ROAD]
            rest = [a for a in actions if a.action_type != ActionType.BUILD_ROAD]
            actions = rest + roads[: max(0, MAX_OPTIONS - len(rest))]
        if self.shuffle:
            self.rng.shuffle(actions)
        options = {f"a{i}": self.action_text(game, a) for i, a in enumerate(actions)}
        me = self.color.value
        answers, raw, wall1 = self._ask(self.describe(game), {"move": {
            "type": "choice", "criteria": options,
            "instructions": f"You are playing Catan as {me} and want to win. Pick the single best action to take right now."}})
        ans = answers["move"]
        probs = ans.get("probabilities", {}) if ans.get("type") == "choice" else {}
        ranked = sorted(probs, key=lambda k: -probs[k]) or list(options)
        cands = [k for k in ranked[: self.k] if k in options]
        scores, wall2 = {}, 0.0
        if len(cands) > 1 and probs.get(cands[0], 0) < self.sure:
            state = {"rules": self.describe(game)["rules"],
                     "note": f"Each position below is the game right after {me} makes one candidate move."}
            questions = {}
            for j, key in enumerate(cands, 1):
                sim = game.copy()
                sim.execute(actions[int(key[1:])])
                d = self.describe(sim)
                d.pop("rules", None)
                d.pop("you_can_afford_now", None)
                d.pop("you_are_missing", None)
                state[f"position_{j}"] = {"after_move": options[key], **d}
                questions[f"p{j}"] = {"type": "score", "criteria": SCORE_LEVELS,
                                      "instructions": f"In position {j}, how likely is {me} to win the game?"}
            answers2, _, wall2 = self._ask(state, questions)
            for j, key in enumerate(cands, 1):
                a2 = answers2.get(f"p{j}", {})
                if a2.get("type") == "score":
                    scores[key] = a2["score"]
        if scores:
            pick = max(scores, key=lambda k: (round(scores[k], 2), probs.get(k, 0)))
        else:
            pick = cands[0] if cands else "a0"
        chosen = actions[int(pick[1:])]
        if self.trace is not None:
            top = [(options[k], round(probs.get(k, 0), 4)) for k in ranked[:5] if k in options]
            self.trace.append({
                "player": self.label, "color": me, "idx": len(game.state.action_records),
                "turn": game.state.num_turns, "prompt": game.state.current_prompt.value,
                "n_options": len(actions), "chosen": options[pick], "confidence": ans.get("confidence"),
                "top": top, "lookahead": [(options[k], round(v, 3)) for k, v in scores.items()],
                "input_tokens": raw.get("usage", {}).get("input_tokens"), "wall_s": round(wall1 + wall2, 2),
            })
        return chosen


DrexPlayer = ModelPlayer  # old name
