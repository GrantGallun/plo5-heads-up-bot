# RangeRider: a heads-up Short Deck 5-card PLO bot

TAMU Datathon poker bot challenge (heads-up Short Deck 5-card Pot-Limit Omaha).
The bot is `agent.py` (the starter web server, unchanged) calling `strategy.py`.

## How it plays

- **Equilibrium blueprint.** A strategy for the exact betting tree of the game
  (every legal half-pot / pot raise sequence up to the 4-raise cap), computed
  offline with Discounted CFR on a GPU. Hands are grouped by a
  distribution-aware card abstraction: on the flop and turn by the
  distribution of their final strength over the remaining cards, on the river
  by final strength and a blocker feature, preflop by clustered final-strength
  distributions (256 / 1,024 / 1,024 / 800 buckets per street).
- **Opponent model.** During each half the bot learns the opponent's fold,
  call and raise frequencies and the hands it shows down, and prices every
  action by expected chips against the opponent's likely range (Monte Carlo
  over opponent hands and runouts).
- **Safe exploitation.** It plays close to the blueprint while the opponent
  looks like an equilibrium player and moves toward the model-based best
  response as the evidence of a real leak grows.
- **Time management.** Monte Carlo sample sizes shrink smoothly when the
  half's time budget runs ahead of schedule, so a slower machine plays a
  noisier version of the same strategy. Showdowns use a precomputed table of
  the best hand each pair of hole cards makes on every board (10 lookups per
  hand instead of 100).

## Files

| File | Purpose |
|---|---|
| `HANDOFF.md` | Current state, results and the remaining steps before submission |
| `agent.py` | Starter web server; sets the team and bot name and calls `strategy.py` |
| `strategy.py` | Decision logic: opponent model, EV pricing, blueprint blend, pacing |
| `abs3_runtime.py` | Card abstraction used by the blueprint (CPU, numpy) |
| `abstraction.py` | Betting tree and street-strength helpers |
| `fasteval.py` | Vectorised hand evaluation and lookup tables |
| `data/blueprint_v3.npz` | The blueprint strategy |
| `data/abs3/` | Abstraction tables (per-board opponent rank samples, clusters, river edges) |
| `data/pairbest/` | Best rank of each hole-card pair on every board (speed table) |
| `preflop_equity.npy` | Preflop equity of every 5-card holding |
| `blueprint.npz` | Small fallback blueprint, used only if the files in `data/` cannot be loaded |
| `game.py`, `evaluator.py`, `judge_engine.py`, `baseline-bots/`, `docker/`, `tools/`, `test_agent.py` | Organizers' starter files |

## Running

```bash
pip install -r requirements.txt
python agent.py                     # the bot, port 5008
python baseline-bots/baseline_random.py
python judge_engine.py --deals 200  # play a match
python test_agent.py
./tools/check_build.sh              # tournament build + smoke test (needs Docker)
```

Requirements: Python 3.10+, `fastapi`, `uvicorn`, `numpy`. The bot uses about
1 GB of memory, starts in a few seconds and stays well inside the 8 s time
bank per half.
