"""Shared bits for the extra games: model clients with a game-specific system prompt, and asking for a move.

Reuses the Catan harness clients (../catan): Drex and Jev (System One choice questions), Claude (CLI) and chat
models on OpenRouter. Keys come from the project .env via load_key and are never printed.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "catan"))

from play import CHAT, CLAUDE_CLI, make_client  # noqa: E402


def game_client(spec, system, budget=3_000_000):
    spec = spec.split("#")[0]
    if spec.endswith("-mix"):
        spec = spec[:-4]
    if spec in CLAUDE_CLI:
        from claude_client import ClaudeCLIClient
        return ClaudeCLIClient(CLAUDE_CLI[spec], system=system)
    if spec in CHAT or spec.startswith("or:"):
        from chat_client import OpenRouterChatClient
        return OpenRouterChatClient(CHAT.get(spec) or spec[3:], system=system)
    return make_client(spec, budget)


class ModelSeat:
    """A model making one choice among labelled options. `-mix` seats sample from the model's probabilities."""

    def __init__(self, label, client, trace, rng_seed=0, shuffle=False):
        self.label, self.client, self.trace = label, client, trace
        self.shuffle = shuffle
        self.mix = label.split("#")[0].endswith("-mix")
        self.rng = random.Random(rng_seed)
        self.calls = 0
        self.cost0 = getattr(client, "cost", 0.0)

    def pick(self, state, instructions, texts, info=None):
        """texts: list of option strings; returns the chosen index."""
        if len(texts) == 1:
            return 0
        order = list(range(len(texts)))
        if self.shuffle:  # show the options in a random order, map the answer back
            self.rng.shuffle(order)
        options = {f"a{j}": texts[i] for j, i in enumerate(order)}
        ans, raw, wall = self.client.choose(state, instructions, options)
        self.calls += 1
        probs = ans.get("probabilities") or {}
        key = ans.get("choice")
        if self.mix and probs:
            keys = [k for k in options if k in probs]
            key = self.rng.choices(keys, weights=[probs[k] for k in keys])[0]
        j = int(key[1:]) if key and key[1:].isdigit() and int(key[1:]) < len(texts) else 0
        i = order[j]
        self.trace.append({"player": self.label, **(info or {}), "chosen": texts[i],
                           "probs": {options[k]: round(p, 3) for k, p in sorted(probs.items(), key=lambda kv: -kv[1])[:5]
                                     if k in options},
                           "reason": ans.get("reason"), "wall_s": round(wall, 2)})
        return i

    @property
    def cost(self):
        return getattr(self.client, "cost", 0.0) - self.cost0
