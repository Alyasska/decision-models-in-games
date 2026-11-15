# Decision models in games

A small harness for watching **decision models** (Jev, Drex, Clef: models that pick one of the options you give
them and return probabilities, without writing text) play games against **reasoning LLMs** (Claude, GPT) and
simple bots. Every player gets the same game-state text and the same list of legal actions.

Games:

| Game | Engine | Bots to compare against |
|---|---|---|
| Heads-up no-limit hold'em | [OpenSpiel](https://github.com/google-deepmind/open_spiel) `universal_poker` | random, calling station, equity bot (Monte Carlo hand strength) |
| Catan | [Catanatron](https://github.com/bcollazo/catanatron) | Catanatron's ValueFunction, AlphaBeta, VictoryPoint, WeightedRandom |
| Backgammon | OpenSpiel `backgammon` | random, one-move heuristic |
| Pokémon (gen 9 random battles) | local [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) via [poke-env](https://github.com/hsahovic/poke-env) | poke-env's random, max-power and heuristic players |

This is a quick look, not a benchmark: the samples are small and the README says so wherever it gives numbers.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Keys are read from the environment or from a `.env` file in `catan/` or the repo root (only the one variable
that is needed is read, nothing else is loaded):

- `OPENROUTER_API_KEY`: Jev, Drex, Clef (OpenRouter's decision endpoint) and GPT (chat completions)
- `NACE_API_KEY`: optional, Drex directly from Nace's API
- Claude seats use the local Claude Code CLI (`claude -p`) with your logged-in account; `claude-api` uses `~/.anthropic_key`

`catan/check_openrouter.py` checks that the OpenRouter key works without printing it.

## Seats

Decision models: `jev` (Jev 1.14), `jev-1.13`, `drex-max-or` (Drex Max via OpenRouter), `drex-v2-or`,
`drex-finance-or`, `drex-max` (Drex via Nace), `clef`, `clef-flash`, or any `dm:<openrouter id>`.
Reasoning LLMs: `claude`, `claude-sonnet`, `claude-fable` (Claude Code CLI), `gpt-sol`, `gpt-terra`, `gpt-astra`,
or any `or:<openrouter chat model>`. `fake` picks at random and costs nothing (for testing).

## Poker

```bash
cd games
python poker.py --a gpt-terra --b jev --deals 25 --shuffle --max-usd 0.25
python poker.py --a drex-max-or --b equity --deals 20
```

Every deal is played twice with the same cards and the seats swapped, so card luck cancels. Blinds 50/100,
200 big blinds deep, actions fold / check-call / half pot / pot / all-in. `--shuffle` shows the model the options
in a random order; `--max-usd` stops a match when its OpenRouter spend passes the limit. The result is big
blinds per 100 hands with a 95% interval; every decision (with the model's probabilities or its one-line reason)
goes to `games/runs/<timestamp>/decisions.jsonl`.

## Catan

```bash
cd catan
python play.py --seats drex-max-or,jev,value,vp --seed 900 --games 26 --mirror --shuffle-options
python play.py --seats claude-fable,gpt-sol,drex-max-or,jev --games 4 --rotate --shuffle-options
python compare.py <run tag>            # paired comparison for --mirror runs
python make_replay.py runs/<run> 0     # HTML replay with the model's probabilities on every move
```

`--mirror` plays each deal twice with the first two seats swapped (same board), `--rotate` gives every seat every
place. Other tools: `shadow.py` (a second model answers every decision the first one makes; disagreements are
saved as snapshots), `rollout.py` (plays both moves of every disagreement out with bots, same dice for both),
`indifference.py` (finds positions where two settlement spots are equally good and asks the models),
`analyze.py` and `selfplay_report.py` (play-style statistics).

## Backgammon and Pokémon

```bash
cd games
python backgammon.py --a jev --b heuristic --deals 8
# Pokémon: start a local Showdown server first (Node.js 22+), listening on localhost only
BIND_IP=127.0.0.1 node pokemon-showdown start --no-security
python pokemon.py --a drex-max-or --b heuristic --battles 6
```

## What I saw (small samples)

Poker, 24 to 50 hands per match: no clear winner between decision models and LLMs. Play styles differed a lot:
when bet into, Jev folded 71%, Clef 62%, Drex 56%, GPT-6.1 Terra 21%, Claude Sonnet 12%; Jev made 0 bluffs in
35 postflop bets, GPT bluffed on 9 of 25. Shuffling the option order did not change the fold rates.

Catan: with seats rotated, Claude won 3 of 4 games at a table with GPT, Drex and Jev; decision models buy a
development card at almost every chance and rarely build. Drex and Jev were even over 104 mirrored games.

## License

GPL-3.0, because the Catan part builds on Catanatron (GPL-3.0).
