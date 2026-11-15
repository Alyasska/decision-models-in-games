"""Rebuild runs/<dir>/summary.json from the per-game JSON lines in several play.py logs.

Needed when a series is split over processes that start in the same second: they share one run folder and
each overwrites summary.json at exit. The per-game files (game_<n>.jsonl, game_<n>_replay.json) are distinct.
  .venv/bin/python merge_logs.py runs/<dir> runs/a.log runs/b.log ...
"""
import json
import sys
from pathlib import Path

run, logs = Path(sys.argv[1]), sys.argv[2:]
old = json.loads((run / "summary.json").read_text())
results = sorted((json.loads(l) for f in logs for l in open(f) if l.startswith('{"game"')), key=lambda r: r["game"])
seats = old["seats"]
old.update(games=len(results), results=results, wins={s: sum(r["winner"] == s for r in results) for s in seats},
           merged_from=logs)
(run / "summary.json").write_text(json.dumps(old, indent=2, ensure_ascii=False))
print(f"{run}: {len(results)} games, wins {old['wins']}")
