"""Chat models (GPT, Gemini, Grok, ...) as Catan players through OpenRouter's chat completions endpoint.

Same `choose(state, instructions, options)` interface and the same system prompt and input JSON as the Claude
client, so every LLM seat sees exactly the same text. The answer is structured JSON (`reason`, `choice`)
validated against an enum of option ids. The key comes from OPENROUTER_API_KEY (environment / project .env)
and is never printed. Cost per call comes from OpenRouter's `usage.cost`.
"""
import json
import time
import urllib.error
import urllib.request

from claude_client import SYSTEM
from drex_player import load_key

URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterChatClient:
    def __init__(self, model, effort="medium", system=None):
        self.system = system or SYSTEM
        self.key = load_key("OPENROUTER_API_KEY", "~/.openrouter_key")
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
        body = {"model": self.model,
                "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": prompt}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "move", "strict": True, "schema": schema}},
                "reasoning": {"effort": self.effort},
                "usage": {"include": True}}
        data = json.dumps(body).encode()
        t = time.time()
        for attempt in range(5):
            req = urllib.request.Request(URL, data=data, headers={
                "Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
            try:
                out = json.loads(urllib.request.urlopen(req, timeout=600).read().decode())
                if "choices" not in out:
                    raise ValueError(str(out.get("error", out))[:200])
                content = out["choices"][0]["message"].get("content") or ""
                ans = json.loads(content)
                if ans.get("choice") not in options:
                    raise ValueError(f"choice {ans.get('choice')!r} not among the options")
                break
            except urllib.error.HTTPError as e:
                msg = e.read().decode()[:300].replace(self.key, "***")
                if e.code in (408, 429, 500, 502, 503, 504) and attempt < 4:
                    time.sleep(2 ** attempt * 3)
                    continue
                raise RuntimeError(f"{self.model} HTTP {e.code}: {msg}")
            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as e:
                if attempt < 4:
                    time.sleep(2 ** attempt * 3)
                    continue
                raise RuntimeError(f"{self.model}: {str(e)[:200]}")
        wall = time.time() - t
        u = out.get("usage", {})
        self.calls += 1
        self.input_tokens += u.get("prompt_tokens", 0)
        self.output_tokens += u.get("completion_tokens", 0)
        self.cost += u.get("cost", 0) or 0
        result = {"choice": ans["choice"], "confidence": None, "probabilities": {}, "reason": ans.get("reason")}
        return result, {"usage": {"input_tokens": u.get("prompt_tokens", 0),
                                  "output_tokens": u.get("completion_tokens", 0)}}, wall
