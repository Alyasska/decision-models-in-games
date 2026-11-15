"""Claude as a Catan player: same `choose(state, instructions, options)` interface as the decision models.

Claude reads the same state text and the same option list, and answers with structured JSON:
the option id plus one short sentence of reasoning (kept for the replay). The key is read from
~/.anthropic_key and never printed.
"""
import json
import os
import time

import anthropic

PRICE = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}

SYSTEM = ("You are playing a game of Catan against other players and you want to win. "
          "Each message gives the current game state and the legal actions, each with an id. "
          "Pick the single best action for this moment. Reply with JSON: `reason` is one short sentence, "
          "`choice` is the id of the action.")


class ClaudeClient:
    def __init__(self, model="claude-opus-5-5", effort="medium", key_path="~/.anthropic_key", system=None):
        self.system = system or SYSTEM
        self.client = anthropic.Anthropic(api_key=open(os.path.expanduser(key_path)).read().strip(),
                                          max_retries=4)
        self.model, self.effort = model, effort
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0
        self.model_ms = 0.0

    def choose(self, state, instructions, options):
        schema = {"type": "object",
                  "properties": {"reason": {"type": "string"},
                                 "choice": {"type": "string", "enum": list(options)}},
                  "required": ["reason", "choice"], "additionalProperties": False}
        prompt = json.dumps({"task": instructions, "state": state, "actions": options}, indent=1)
        t = time.time()
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
            system=self.system,
            messages=[{"role": "user", "content": prompt}],
        )
        wall = time.time() - t
        u = resp.usage
        self.calls += 1
        self.input_tokens += u.input_tokens
        self.output_tokens += u.output_tokens
        pin, pout = PRICE.get(self.model, (4.0, 20.0))
        self.cost += (u.input_tokens * pin + u.output_tokens * pout) / 1e6
        if resp.stop_reason == "refusal":
            return {"choice": None, "confidence": None, "probabilities": {}, "refused": True}, {}, wall
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        data = json.loads(text)
        ans = {"choice": data.get("choice"), "confidence": None, "probabilities": {},
               "reason": data.get("reason")}
        raw = {"usage": {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens}}
        return ans, raw, wall


class ClaudeCLIClient:
    """Claude through the local Claude Code CLI (`claude -p`), using the account already logged in there.

    No tools, no MCP servers, no saved sessions; runs in an empty folder so no project CLAUDE.md is loaded.
    The answer comes back as structured JSON validated against the same schema as the API client.
    """

    def __init__(self, model="opus", effort="medium", workdir=None, system=None):
        import pathlib
        self.system = system or SYSTEM
        self.model, self.effort = model, effort
        self.workdir = workdir or str(pathlib.Path(__file__).parent / "claude_cwd")
        pathlib.Path(self.workdir).mkdir(exist_ok=True)
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0
        self.model_ms = 0.0

    def choose(self, state, instructions, options):
        import subprocess
        schema = {"type": "object",
                  "properties": {"reason": {"type": "string"},
                                 "choice": {"type": "string", "enum": list(options)}},
                  "required": ["reason", "choice"], "additionalProperties": False}
        prompt = json.dumps({"task": instructions, "state": state, "actions": options}, indent=1)
        cmd = ["claude", "-p", "--output-format", "json", "--model", self.model, "--effort", self.effort,
               "--system-prompt", self.system, "--tools", "", "--strict-mcp-config", "--no-session-persistence",
               "--json-schema", json.dumps(schema)]
        t = time.time()
        for attempt in range(3):
            p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=self.workdir, timeout=600)
            try:
                out = json.loads(p.stdout)
            except json.JSONDecodeError:
                out = {"is_error": True, "result": (p.stdout or p.stderr)[:300]}
            if not out.get("is_error") and out.get("structured_output"):
                break
            time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"claude CLI failed: {str(out.get('result'))[:300]}")
        wall = time.time() - t
        u = out.get("usage", {})
        tin = u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0)
        self.calls += 1
        self.input_tokens += tin
        self.output_tokens += u.get("output_tokens", 0)
        self.cost += out.get("total_cost_usd", 0) or 0
        self.model_ms += out.get("duration_api_ms", 0) or 0
        data = out["structured_output"]
        ans = {"choice": data.get("choice"), "confidence": None, "probabilities": {}, "reason": data.get("reason")}
        return ans, {"usage": {"input_tokens": tin, "output_tokens": u.get("output_tokens", 0)}}, wall
