# Handoff: state of the bot (2026-11-08)

Read this first if you are picking up the work.

## What is in this repo
The tournament bot "RangeRider" (heads-up Short Deck 5-card PLO). `agent.py` is the
starter web server (plumbing unchanged) and calls `strategy.py`. See `README.md` for how it
plays and what each file does.

Live configuration:
- **Blueprint v3** (`data/blueprint_v3.npz`): Discounted CFR over the exact betting tree with a
  distribution-aware card abstraction (256 / 1,024 / 1,024 / 800 buckets per street). The
  abstraction tables are in `data/abs3/`; `abs3_runtime.py` computes buckets on the CPU.
- **Opponent model + safe exploitation**: plays near the blueprint until the evidence of a leak
  grows. The trust references in `strategy.py` (`BP_REF`, `BP_REF_CTX`) were measured with v1 and
  re-checked against v3's self-play stats: they still fit.
- **Smooth time pacing** (`BUDGET_FRAC=0.80`, `SMOOTH_PACE=True`, `N_FLOOR=40`): Monte Carlo
  samples shrink smoothly when a half runs ahead of its time budget.
- **PAIRBEST speed table** (`data/pairbest/`, about 450 MB): showdown strength of every hole-card
  pair on every board, about 1.4 ms per decision.
- `blueprint.npz` is a small fallback used only if `data/` fails to load.

## Results
- Against the field of public bots (official judge, fresh servers per half): v3 went **24 W / 6 D /
  0 L, +1.84 bb/deal**, versus 21-9-2, +1.22 for the previous version.
- Against the strongest blueprint-style opponent: 16 draws, 0 losses (the previous version lost 3
  of 24).
- Time use: at most about 2.4 s of the 8 s budget per half.
- Bug fixed before release: a river spot was looked up in the preflop table, and the exception
  turned into a FOLD (it folded a royal flush). The blueprint lookup now runs only when the tree
  node's street matches the observation's street.
- `test_agent.py` and the official judge (`python judge_engine.py`) run cleanly from this folder:
  +4.45 bb/deal against the random baseline, no errors.

## Checked against the participant guide
- `bash tools/check_build.sh`: **ALL CHECKS PASSED** (image adds 782 MB of the 3,072 MB cap; the
  smoke test answers in 1-3 ms per action).
- Fresh server per half, 1,000 deals, official judge: 0 errors, 0.84-1.09 s of the 8 s bank
  against the random baseline, peak memory 0.9 GB (cap 4 GB), startup 2.3 s.
- `agent.py` changes are only the names and the code between the YOUR CODE HERE markers; the
  SERVER PLUMBING half, `game.py`, `evaluator.py` and `judge_engine.py` match the starter repo.
- No file writes, no network use; `requirements.txt` (fastapi, uvicorn, numpy) is in the root;
  every hook is wrapped in try/except with a safe legal fallback.
- On Windows, pass the judge `http://127.0.0.1:<port>` instead of a bare port: `localhost` can
  resolve to IPv6 first and the health check times out, which looks like a forfeit.

## Not done yet (before submission)
1. **Set `PARTICIPANT`** in `agent.py` (line 42; it is still `"Datathon Team"`) to the team's
   registered name. The team must be registered, and no person may be on two teams.
2. Keep the history to clean release commits. Never push to the organizers'
   `tamu-datathon-org` repo.
3. Shortly before the deadline: make the repo **public**, get the full 40-character SHA with
   `git rev-parse HEAD`, and submit team name (same name on every resubmission) + repo URL +
   SHA on the submission form.

## Tournament rules to keep in mind
8 s thinking per 1,000-hand half, at most 1 s per request, 5 errors = forfeit = match loss,
1 CPU, 4 GB RAM, 3 GB image, no GPU. Don't edit the SERVER PLUMBING half of `agent.py`. Ties on
points are broken by average bb/deal over all matches.

## Testing notes
- Machine load changes results a lot. Run A/B arms at the same time, with few workers and nothing
  else heavy running.
- An 8-match A/B can't resolve 0.2 bb/deal; use 24+ matches per arm.
