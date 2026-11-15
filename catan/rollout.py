"""Play out each disagreement snapshot many times with bots: whose move was better?

  .venv/bin/python rollout.py runs/<ts>-shadow [--n 100] [--workers 12]

For every snapshot (a position where the two models chose differently) both candidate moves are applied and the
game is played to the end by ValueFunction bots in all seats, n times each. The two candidates share dice seeds
(rollout i uses the same seed for both), and the remaining development card deck is reshuffled per rollout so the
bots cannot know future cards. Measures the win rate of the deciding player's colour after each candidate.
Writes rollouts.json in the run folder and prints a summary.
"""
import argparse
import collections
import json
import math
import pickle
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from catanatron.state_functions import get_actual_victory_points

from analyze import category


def one(task):
    path, which, seed = task
    snap = pickle.loads(Path(path).read_bytes())
    g = snap["game"].copy()
    g.state = g.state.copy()
    rng = random.Random(seed)
    g.random = g.state.random = rng
    g.state.development_listdeck = list(g.state.development_listdeck)
    rng.shuffle(g.state.development_listdeck)
    g.execute(snap[which]["action"])
    winner = g.play()
    me = snap["color"]
    return path, which, seed, winner == me, get_actual_victory_points(g.state, me), winner is None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    snaps = sorted(Path(a.run, "snaps").glob("*.pkl"))
    if not snaps:
        sys.exit("no snapshots")
    raw = Path(a.run, "rollouts_raw.jsonl")  # every finished playout; a restart skips what is already done
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    done = set()
    if raw.exists():
        for line in raw.read_text().splitlines():
            d = json.loads(line)
            res[d["path"]][d["which"]].append((d["seed"], d["won"], d["vp"], d["unfinished"]))
            done.add((d["path"], d["which"], d["seed"]))
    tasks = [(str(p), w, 10_000 + i) for p in snaps for w in ("player", "shadow") for i in range(a.n)]
    todo = [t for t in tasks if t not in done]
    print(f"{len(snaps)} positions x 2 moves x {a.n} rollouts = {len(tasks)} games; {len(tasks) - len(todo)} already done, "
          f"{len(todo)} to play on {a.workers} workers", flush=True)
    t = time.time()
    with Pool(a.workers) as pool, open(raw, "a") as f:
        for k, (path, which, seed, won, vp, unfinished) in enumerate(pool.imap_unordered(one, todo, chunksize=8)):
            res[path][which].append((seed, won, vp, unfinished))
            f.write(json.dumps({"path": path, "which": which, "seed": seed, "won": won, "vp": vp,
                                "unfinished": unfinished}) + "\n")
            if (k + 1) % 2000 == 0:
                f.flush()
                print(f"  {k + 1}/{len(todo)} ({time.time() - t:.0f}s)", flush=True)
    out = []
    for p in snaps:
        snap = pickle.loads(p.read_bytes())
        r = res[str(p)]
        a_ = dict((s, (w, v)) for s, w, v, _ in r["player"])
        b_ = dict((s, (w, v)) for s, w, v, _ in r["shadow"])
        seeds = sorted(set(a_) & set(b_))
        wa = sum(a_[s][0] for s in seeds) / len(seeds)
        wb = sum(b_[s][0] for s in seeds) / len(seeds)
        diffs = [a_[s][0] - b_[s][0] for s in seeds]  # paired: same dice
        m = sum(diffs) / len(diffs)
        sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / max(1, len(diffs) - 1))
        se = sd / math.sqrt(len(diffs))
        rec = snap["record"]
        out.append({"snapshot": p.name, "turn": rec["turn"], "prompt": rec["prompt"],
                    "player_model": snap["player"]["model"], "player_move": rec["chosen"],
                    "shadow_model": snap["shadow"]["model"], "shadow_move": rec["shadow"],
                    "kind": "same kind" if category(rec["chosen"]) == category(rec["shadow"]) else "different kind",
                    "win_player_move": round(wa, 3), "win_shadow_move": round(wb, 3),
                    "diff": round(m, 3), "se": round(se, 3),
                    "vp_player_move": round(sum(a_[s][1] for s in seeds) / len(seeds), 2),
                    "vp_shadow_move": round(sum(b_[s][1] for s in seeds) / len(seeds), 2)})
    Path(a.run, "rollouts.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"done in {time.time() - t:.0f}s -> {a.run}/rollouts.json\n")

    # per model: when its move was compared with the other model's move, how often was it better?
    models = sorted({o["player_model"] for o in out} | {o["shadow_model"] for o in out})
    tally = {m: collections.Counter() for m in models}
    gain = {m: [] for m in models}
    for o in out:
        for me, other, d in ((o["player_model"], o["shadow_model"], o["diff"]), (o["shadow_model"], o["player_model"], -o["diff"])):
            if me != o["player_model"] and me != o["shadow_model"]:
                continue
            gain[me].append(d)
        sig = abs(o["diff"]) > 2 * o["se"] and o["se"] > 0
        better = o["player_model"] if o["diff"] > 0 else o["shadow_model"]
        for m in (o["player_model"], o["shadow_model"]):
            tally[m]["positions"] += 1
        if sig:
            tally[better]["clearly better"] += 1
        else:
            for m in (o["player_model"], o["shadow_model"]):
                tally[m]["no clear difference"] += 1
    print(f"{len(out)} disagreement positions")
    for m in models:
        g = gain[m]
        print(f"  {m:12} its move clearly better in {tally[m]['clearly better']} positions; "
              f"average win-chance change from choosing its move instead of the other's: {sum(g) / len(g) * 100:+.1f} pts")
    print(f"  no clear difference (within 2 standard errors): {tally[models[0]]['no clear difference']}")
    for kind in ("different kind", "same kind"):
        sub = [o for o in out if o["kind"] == kind]
        if sub:
            print(f"\n{kind} ({len(sub)}): " + ", ".join(
                f"{m} avg {sum((o['diff'] if o['player_model'] == m else -o['diff']) for o in sub) / len(sub) * 100:+.1f} pts"
                for m in models))
    print("\nbiggest gaps:")
    for o in sorted(out, key=lambda o: -abs(o["diff"]))[:8]:
        w = o["player_model"] if o["diff"] > 0 else o["shadow_model"]
        print(f"  {o['diff'] * 100:+5.0f} pts (±{2 * o['se'] * 100:.0f}) better: {w:12} | {o['player_model']}: {o['player_move'][:60]}"
              f"\n{'':38}| {o['shadow_model']}: {o['shadow_move'][:60]}")


if __name__ == "__main__":
    main()
