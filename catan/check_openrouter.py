"""Check that the OpenRouter key works, without printing it.

  .venv/bin/python check_openrouter.py

Loads OPENROUTER_API_KEY from the environment / project .env (see drex_player.load_key), then makes two free
GET requests: /api/v1/models (with the key) and /api/v1/key (key status: credit used / limit; the key's label
is not printed). Any error text is scrubbed of the key before it is shown.
"""
import json
import os
import urllib.error
import urllib.request

from drex_player import ENV_FILES, load_key

key = load_key("OPENROUTER_API_KEY")
print("key loaded:", "environment" if os.environ.get("OPENROUTER_API_KEY") else
      next((f"{f.parent.name}/.env" for f in ENV_FILES if f.is_file()), "?"))


def get(path):
    req = urllib.request.Request("https://openrouter.ai" + path, headers={"Authorization": "Bearer " + key})
    try:
        return 200, json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300].replace(key, "***")


code, models = get("/api/v1/models")  # public list: answers 200 even with a bad key
if code == 200:
    ids = [m["id"] for m in models["data"]]
    jev = [m for m in models["data"] if "jev" in m["id"] or m["id"].startswith("typesafe/")]
    print(f"/api/v1/models: HTTP 200, {len(ids)} models")
    for m in jev:
        p = {k: float(v) for k, v in m.get("pricing", {}).items() if k in ("prompt", "completion")}
        price = ("variable (router)" if min(p.values(), default=0) < 0 else
                 f"prompt ${p.get('prompt', 0) * 1e6:.3f}/M, completion ${p.get('completion', 0) * 1e6:.3f}/M")
        print(f"  {m['id']}: {price}")
    if not jev:
        print("  no typesafe/jev models in the list")
else:
    print(f"/api/v1/models: HTTP {code}: {models}")

code, info = get("/api/v1/key")  # this one actually checks the key
if code == 200:
    d = info.get("data", {})
    print(f"/api/v1/key: HTTP 200 -> key valid | usage ${d.get('usage', 0):.4f} | limit {d.get('limit')} "
          f"| limit_remaining {d.get('limit_remaining')} | free tier {d.get('is_free_tier')}")
else:
    print(f"/api/v1/key: HTTP {code}: {info}")
