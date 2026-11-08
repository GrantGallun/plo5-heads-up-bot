"""
strategy.py - "RangeRider": an adaptive, EV-driven Short Deck PLO5 bot.

How it decides (every turn):
  1. Simulate N (opponent hand, runout) pairs with the numpy evaluator and
     record, per sample, who wins at showdown and how strong the opponent's
     hand was on each street (as a percentile among random hands).
  2. Weight every sample by how likely THIS opponent's actions in this hand
     are for a hand of that strength. Those likelihoods are learned during
     the half from showdowns (what the opponent showed after betting,
     raising, calling or checking on each street), starting from a prior.
  3. Price each legal action as expected chips relative to folding now:
       fold  = 0
       call  = equity * (pot + call) - call
       raise = f * pot + (1 - f) * (equity_vs_callers * final_pot - my_chips)
     where f is the opponent's learned fold rate to our bets/raises on that
     street, and equity_vs_callers drops the weakest f of their range.
     The highest-EV action is played.

Only public information is used, and all learning lives in `memory`, so
nothing survives into the next half (each half also runs in a fresh process).
"""
from __future__ import annotations

import os
from time import perf_counter

import numpy as np

import abstraction as ab
import fasteval as fe
from game import Action, AgentConfig, HandResult, Observation, Street

fe.build()
USE_PAIRBEST = True
if USE_PAIRBEST:
    fe.load_pairbest()      # optional speed table; without it the bot computes the same ranks slowly

# ---------------------------------------------------------------------------
# Equilibrium blueprint (trained offline): Discounted CFR over the
# exact betting tree with NB_BP strength buckets per street. The live policy
# samples from  q(a) ~ blueprint(a) * exp(EV(a) / tau): close to the
# blueprint while little is known about the opponent, moving toward the
# model-based best response as evidence accumulates (tau shrinks).
# ---------------------------------------------------------------------------
BP = BP_TREE = None
BP_VERSION = 0


def load_blueprint(path: str) -> bool:
    """Load a v1 (one table, 10 buckets) or v2 (per-street tables,
    potential-aware buckets) blueprint. On any problem, play without one."""
    global BP, BP_TREE, BP_VERSION
    try:
        z = np.load(path)
        tree = BP_TREE or ab.Tree()
        if "version" in z and int(z["version"]) == 3:
            # v3: distribution-aware buckets from abs3_runtime (data/abs3)
            import abs3_runtime
            if abs3_runtime.TABLE is None and not abs3_runtime.load():
                raise ValueError("v3 abstraction data missing")
            tables = [z[k].astype(np.float32) for k in ("pre", "flop", "turn", "river")]
            nb = [int(x) for x in z["nb"]]
            if any(t.shape != (tree.street_rows[s], nb[s]) for s, t in enumerate(tables)):
                raise ValueError("v3 blueprint does not match the tree")
            BP, BP_TREE, BP_VERSION = tables, tree, 3
        elif "version" in z and int(z["version"]) == 2:
            tables = [z[k].astype(np.float32) for k in ("pre", "flop", "turn", "river")]
            if any(t.shape != (tree.street_rows[s], ab.NB_STREET[s]) for s, t in enumerate(tables)):
                raise ValueError("v2 blueprint does not match the tree")
            BP, BP_TREE, BP_VERSION = tables, tree, 2
        else:
            table = z["strategy"].astype(np.float32)
            if table.shape != (int(tree.offsets[-1]), ab.NB_BP):
                raise ValueError("v1 blueprint does not match the tree")
            BP, BP_TREE, BP_VERSION = table, tree, 1
        return True
    except (OSError, KeyError, ValueError, ImportError, MemoryError):
        BP, BP_VERSION = None, 0
        return False


_HERE = os.path.dirname(os.path.abspath(__file__))
# v3 blueprint (distribution-aware abstraction, trained offline on a GPU); the
# v1 blueprint is the fallback if the v3 files cannot be loaded.
if not load_blueprint(os.path.join(_HERE, "data", "blueprint_v3.npz")):
    load_blueprint(os.path.join(_HERE, "blueprint.npz"))
BLEND_KAPPA = 0.4       # tau = kappa * pot * trust(deviation); 0 = pure exploit
BP_ONLY = False         # True: play the blueprint alone (for testing)
# Exploit only on evidence: the opponent's rates are compared with what the
# blueprint itself does in self-play (fold to a raise preflop, fold to a bet
# postflop, raise when bet into, postflop aggression). Below DEV_Z0 standard
# errors of deviation the bot stays anchored to the blueprint; above DEV_Z1 it
# plays the model's best response; in between it interpolates.
BP_REF = (0.322, 0.446, 0.239, 0.479)      # v3 blueprint self-play statistics
DEV_Z0, DEV_Z1 = 2.0, 4.0
# "magnitude" (default): trust falls as the SIZE of the deviation grows,
# using a lower 95% bound on |observed rate - blueprint rate|. A z-score grows
# with every hand even for a balanced opponent that merely differs in style,
# so against a strong equilibrium bot "z" mode drifted into exploitation and
# lost; the bound levels off at that opponent's true distance instead.
DEV_MODE = "magnitude"
DEV_MAG0, DEV_MAG1 = 0.12, 0.18     # anchored below MAG0, full best response above MAG1
# Per-situation trust (off by default): a balanced opponent may still leak in
# specific spots. Each decision also looks at the deviation of the opponent
# behaviour it depends on - their responses if we bet, their aggression if we
# face a bet - per [preflop, postflop], against the blueprint's self-play
# rates (bet when unopened, fold and raise when facing), and the smaller
# trust wins.
CTX_TRUST = False
CTX_MAG0, CTX_MAG1 = 0.06, 0.10
CTX_MIN_N = 20
BP_REF_CTX = ((0.484, 0.238, 0.436), (0.434, 0.446, 0.233))

CHECK, CALL, BET, RAISE = 0, 1, 2, 3          # opponent action classes per street
# Strength-percentile buckets, finer near the top where betting decisions live.
EDGES = np.array([0.0, 0.25, 0.45, 0.60, 0.72, 0.82, 0.90, 0.95, 1.0])
NB = len(EDGES) - 1
WIDTH = np.diff(EDGES)
_MID = (EDGES[:-1] + EDGES[1:]) / 2
_dens = np.array([
    1.35 - 0.70 * _MID,                        # check: skews weak
    0.55 + 0.90 * _MID,                        # call
    0.45 + 1.10 * _MID,                        # bet into a check
    0.15 + 1.90 * _MID ** 1.5,                 # raise over a bet
])
PRIOR_MASS = _dens * WIDTH / (_dens * WIDTH).sum(axis=1, keepdims=True)
PRIOR_N = 10.0                                 # pseudo-showdowns behind the prior
SIZE_PRIOR_N = 8.0                           # shrink aggression size toward pooled other-size evidence

# Frequency-based range priors: an opponent that bets a fraction b of the
# times it can is read as betting roughly its top b (softly), and the more
# often it bets the less a bet says about its cards ("selectivity").
FREQ_TEMP, FREQ_EPS = 0.10, 0.08
FREQ_PRIOR_OPEN = np.array([0.0, 5.0, 3.0])     # [fold, check, bet] pseudo-counts
FREQ_PRIOR_FACING = np.array([2.4, 3.6, 2.0])   # [fold, call, raise]
USE_FREQ_PRIOR = True
# Re-raise lookahead: continuing hands may re-raise our raise (top q of them).
REREAISE_PRIOR = (2.0, 6.0)                     # pseudo (re-raises, continues)
REREAISE_MAX = 0.35                             # above this, re-raises say little: ignore
USE_LOOKAHEAD = True

# Value of information: a showdown reveals the opponent's cards, which sharpen
# the range model for every later hand of the half. An action is credited
#   P(showdown | action) * INFO_KAPPA * (share of the half left)
#   * sum over this hand's opponent actions of street_weight * uncertainty,
# where uncertainty = PRIOR_N / (PRIOR_N + showdowns already seen for that
# street and action type). It fades as the model fills and the half runs out.
INFO_KAPPA = 0.0      # chips per fully-uncertain action type; tested 1-6: no gain (off)
INFO_STREET_W = (0.4, 0.6, 0.8, 1.0)
INFO_CALLS_ONLY = False                        # True: only bluff-catching river calls

# Variance control for the confidence-interval verdict. Points depend on
# t = mean / SE, not on margin, and maximising t is (locally) maximising
# EV - lambda * Var. Small edges only occur against equilibrium-like
# opponents, so the penalty is scaled by the same blueprint trust that
# anchors us there (and vanishes against clearly exploitable bots).
VAR_LAMBDA = 0.0          # per chip; 0 = off (scaled by trust when VAR_BY_TRUST)
VAR_EXPLOIT = 0.0         # per chip, scaled by (1 - trust): prefer low variance against exploitable bots
TRAP = 0.0                # weight of the "opponent bets after our check" term in the check EV (0 = off)
VAR_BY_TRUST = True
LOOK_SELECTIVE = True      # loose raisers' re-raises carry less strength: blend toward the range

# Fold model: the opponent folds to our bet/raise when its strength percentile
# on the street is below a threshold T. A posterior over T is kept per street.
T_GRID = np.linspace(0.0, 0.95, 20)
_A, _B = 1.4, 2.6                              # weak prior, mode near T = 0.35
FOLD_PRIOR_LL = _A * np.log(T_GRID + 0.02) + _B * np.log(1.02 - T_GRID)
FOLD_POOL = 0.4                                # weight of other streets' evidence

# Equity realisation before the river: 1 - aggression * penalty (IP / OOP).
REAL_IP, REAL_OOP = 0.10, 0.27
AGGR_PRIOR = (3.0, 10.0)                       # pseudo (aggressive, total) actions
# Simulation sizes: preflop unopened, normal, big pot.
N_PRE, N_NORMAL, N_BIG, BIG_POT = 450, 650, 1100, 40
BUDGET_FRAC = 0.80     # share of the time bank the simulations may use (the rest is margin)
N_FLOOR = 40           # fewest samples while pacing (instead of falling back to the table bot)
SMOOTH_PACE = True     # scale samples by how far ahead of schedule we are (no cliff)

# Rank windows (as 9-bit presence rows) for straight-draw detection, incl. the A-6-7-8-9 wheel.
WINDOWS = np.zeros((6, 9), dtype=bool)
for _w, _ranks in enumerate([range(0, 5), range(1, 6), range(2, 7), range(3, 8), range(4, 9),
                             (8, 0, 1, 2, 3)]):
    WINDOWS[_w, list(_ranks)] = True
SUITS = np.arange(4)


# ---------------------------------------------------------------------------
# Hand strength on a street (percentile machinery)
# ---------------------------------------------------------------------------

def street_strength(holes: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Strength of each (N, 5) holding on a visible board of 3-5 cards (made
    hand + draw bonus). Shared with the blueprint's card abstraction."""
    return ab.strength(holes, board)


def rank_pct(values: np.ndarray) -> np.ndarray:
    """Mid-rank percentile of each value among the sample (ties share a rank)."""
    srt = np.sort(values)
    lo = np.searchsorted(srt, values, "left")
    hi = np.searchsorted(srt, values, "right")
    return (lo + hi) / (2.0 * len(values))


def to_bucket(pct: np.ndarray) -> np.ndarray:
    return np.searchsorted(EDGES[1:-1], pct, "right")


# ---------------------------------------------------------------------------
# Half-local state
# ---------------------------------------------------------------------------

def new_half(config: AgentConfig, memory: dict) -> None:
    memory.clear()
    memory.update(
        rng=np.random.default_rng(20261107),
        hist=np.zeros((4, 4, NB)),             # [street, class, bucket] showdown counts
        hist_size=np.zeros((4, 4, 2, NB)),     # [street,class,half/pot,bucket], own showdown evidence only
        resp=np.zeros((4, 2, 3)),              # [street, bet/raise, fold/call/raise] (stats)
        fold_ll=np.zeros((4, len(T_GRID))),    # log-likelihood of each threshold T
        aggr=np.zeros(2),                      # opponent postflop [passive, aggressive] actions
        freq=np.zeros((4, 2, 3)),              # [street, facing a bet, fold/passive/aggressive]
        num_hands=max(1, config.num_hands),
        budget=max(0.0, config.time_bank_seconds * BUDGET_FRAC),
        cpu=0.0, decisions=0, hands_done=0, errors=0,
        hand=None,
    )


def _new_hand_state(obs: Observation) -> dict:
    return {"index": obs.hand_index, "hole": obs.hole_cards, "button": obs.is_button,
            "seen": 0, "cls": {}, "sizes": {}, "pending": None, "cache": None, "last": None,
            "raise_F": {}}


def _replay(obs: Observation, memory: dict) -> None:
    """Walk the action history; count only records not seen before."""
    h = memory["hand"]
    me_tot, opp_tot = (1, 2) if obs.is_button else (2, 1)
    street = Street.PREFLOP
    pending = None                                   # (street, kind) of our unanswered aggression
    for i, rec in enumerate(obs.action_history):
        if rec.street != street:
            street, me_tot, opp_tot, pending = rec.street, 0, 0, None
        new = i >= h["seen"]
        if rec.is_me:
            if rec.action in (Action.RAISE_HALF, Action.RAISE_POT):
                pending = (int(street), 0 if me_tot == opp_tot else 1)
            me_tot = rec.street_total
            continue
        facing = me_tot > opp_tot
        if rec.action == Action.FOLD:
            cls, outcome = None, 0
        elif rec.action == Action.CHECK_CALL:
            cls, outcome = (CALL if facing else CHECK), 1
        else:
            cls, outcome = (RAISE if facing else BET), 2
        if pending is not None and facing:
            if new:
                _record_response(memory, h, pending, outcome)
            pending = None
        if cls is not None:
            if cls >= BET and cls >= h["cls"].get(int(street), -1):
                h["sizes"][int(street)] = int(rec.action == Action.RAISE_POT)
            h["cls"][int(street)] = max(h["cls"].get(int(street), -1), cls)
        if new and cls is not None and street > Street.PREFLOP:
            memory["aggr"][1 if cls >= BET else 0] += 1
        if new:
            memory["freq"][int(street), int(facing), outcome] += 1
        opp_tot = rec.street_total
    h["seen"] = len(obs.action_history)
    h["pending"] = pending


# ---------------------------------------------------------------------------
# Simulation and range weighting
# ---------------------------------------------------------------------------

def _simulate(obs: Observation, memory: dict, n: int) -> dict:
    rng = memory["rng"]
    hole = np.array(obs.hole_cards, dtype=np.int64)
    board = np.array(obs.board, dtype=np.int64)
    used = np.zeros(36, dtype=bool)
    used[hole] = True
    used[board] = True
    rest = np.flatnonzero(~used)
    need = 5 - len(board)
    idx = rng.random((n, len(rest))).argsort(axis=1)[:, :5 + need]
    draw = rest[idx]
    opp = draw[:, :5]
    full = np.concatenate([np.broadcast_to(board, (n, len(board))), draw[:, 5:]], axis=1)
    if fe.PAIRBEST is not None:          # 10 table lookups per hand instead of 100
        hero = fe.full_ranks(hole[None, :], full)
        villain = fe.full_ranks(opp, full)
    else:
        tri = fe.triple_indices(full)
        hero = fe.best_scores(fe.pair_indices(hole[None, :]), tri)
        villain = fe.best_scores(fe.pair_indices(opp), tri)
    share = (hero > villain) + 0.5 * (hero == villain)
    pcts = [fe.preflop_percentile(opp)]
    raw = None
    for street, k in ((Street.FLOP, 3), (Street.TURN, 4)):
        if street > obs.street:
            break
        raw = street_strength(opp, board[:k])
        pcts.append(rank_pct(raw))
    if obs.street == Street.RIVER:
        raw = villain
        pcts.append(rank_pct(villain))      # river strength is the final score
    # Our own blueprint bucket: percentile among random holdings on this board.
    if obs.street == Street.PREFLOP:
        hero_pct = float(fe.preflop_percentile(hole[None, :])[0])
    else:
        mine = hero[0] if obs.street == Street.RIVER else street_strength(hole[None, :], board)[0]
        hero_pct = float((raw < mine).mean() + 0.5 * (raw == mine).mean())
    if BP_VERSION == 3:
        import abs3_runtime
        bucket = abs3_runtime.bucket(int(obs.street), hole, board, rng)
    elif BP_VERSION == 2:
        bucket = int(ab.bucket2(int(obs.street), hero_pct, float(share.mean())))
    else:
        bucket = int(ab.bp_bucket(hero_pct))
    return {"share": share, "pcts": pcts, "n": n, "bucket": bucket}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _freq_rates(memory: dict, street: int):
    """(bet rate when unopened, fold rate and raise rate when facing a bet),
    pooled across postflop streets with extra weight on this street."""
    fr = memory["freq"]
    c = fr[0] + 0.2 * fr[1:].sum(axis=0) if street == 0 else fr[1:].sum(axis=0) + fr[street]
    o = c[0] + FREQ_PRIOR_OPEN
    f = c[1] + FREQ_PRIOR_FACING
    return o[2] / o[1:].sum(), f[0] / f.sum(), f[2] / f.sum()


def _freq_prior(memory: dict, street: int) -> np.ndarray:
    """Prior bucket masses [class, bucket] shaped by observed frequencies."""
    b, fo, ra = _freq_rates(memory, street)
    u = _MID
    sel_b = min(1.0, max(0.0, (0.85 - b) / 0.55))
    sel_r = min(1.0, max(0.0, (0.85 - ra) / 0.55))
    top_b = _sigmoid((u - (1.0 - b)) / FREQ_TEMP)
    top_r = _sigmoid((u - (1.0 - ra)) / FREQ_TEMP)
    like = np.array([
        1.0 - sel_b * (1.0 - FREQ_EPS) * top_b,                                  # check
        FREQ_EPS + (1.0 - FREQ_EPS) * _sigmoid((u - fo) / FREQ_TEMP)
        * (1.0 - 0.7 * sel_r * top_r),                                            # call
        (1.0 - sel_b) + sel_b * (FREQ_EPS + (1.0 - FREQ_EPS) * top_b),           # bet
        (1.0 - sel_r) + sel_r * (FREQ_EPS + (1.0 - FREQ_EPS) * top_r),           # raise
    ])
    mass = like * WIDTH
    return mass / mass.sum(axis=1, keepdims=True)


def _weights(sim: dict, cls_by_street: dict, memory: dict, sizes=None) -> np.ndarray:
    w = np.ones(sim["n"])
    hist = memory["hist"]
    for street, cls in cls_by_street.items():
        counts = hist[street, cls]
        prior = _freq_prior(memory, street)[cls] if USE_FREQ_PRIOR else PRIOR_MASS[cls]
        if sizes is not None and cls >= BET and street in sizes:
            sized = memory["hist_size"][street, cls, sizes[street]]
            other = np.maximum(0., counts - sized)
            pooled = (other + PRIOR_N * prior) / (other.sum() + PRIOR_N)
            # Disjoint other-size observations supply shrinkage. No evidence
            # for this size reproduces the previous pooled likelihood exactly.
            like = (sized + SIZE_PRIOR_N * pooled) / (sized.sum() + SIZE_PRIOR_N) / WIDTH
        else:
            like = (counts + PRIOR_N * prior) / (counts.sum() + PRIOR_N) / WIDTH
        w = w * like[to_bucket(sim["pcts"][street])]
    return w


def _threshold_posterior(memory: dict, street: int) -> np.ndarray:
    ll = memory["fold_ll"]
    total = FOLD_PRIOR_LL + ll[street] + FOLD_POOL * (ll.sum(axis=0) - ll[street])
    p = np.exp(total - total.max())
    return p / p.sum()


def _range_by_threshold(share, w, cur_pct, q: float = 0.0):
    """For every threshold T: F(T) = weighted share of the range below T (it
    folds), our equity against the part at or above T (it continues), and,
    when q > 0, our equity against the top q of the continuing weight (it
    re-raises) and against the rest (it flat-calls)."""
    order = np.argsort(cur_pct, kind="stable")
    sp = cur_pct[order]
    cw = np.concatenate([[0.0], np.cumsum(w[order])])
    cs = np.concatenate([[0.0], np.cumsum((w * share)[order])])
    k = np.searchsorted(sp, T_GRID, "left")
    W, S = cw[-1], cs[-1]
    below_w, below_s = cw[k], cs[k]
    F = below_w / W
    above = W - below_w
    s_above = S - below_s
    eq_above = np.where(above > 1e-12, s_above / np.maximum(above, 1e-12), S / W)
    if q <= 0.0:
        return F, eq_above, None, None
    j = np.clip(np.searchsorted(cw, W - q * above, "left"), 0, len(cw) - 1)
    rr_w, rr_s = W - cw[j], S - cs[j]
    eq_rr = np.where(rr_w > 1e-12, rr_s / np.maximum(rr_w, 1e-12), eq_above)
    flat_w, flat_s = above - rr_w, s_above - rr_s
    eq_flat = np.where(flat_w > 1e-12, flat_s / np.maximum(flat_w, 1e-12), eq_above)
    return F, eq_above, eq_rr, eq_flat


def _reraise_rate(memory: dict, kind: int) -> float:
    resp = memory["resp"][:, kind].sum(axis=0)          # [fold, call, raise]
    q = (resp[2] + REREAISE_PRIOR[0]) / (resp[1] + resp[2] + REREAISE_PRIOR[0] + REREAISE_PRIOR[1])
    return 0.0 if q > REREAISE_MAX else float(q)


def _record_response(memory: dict, h: dict, pending, outcome: int) -> None:
    street, kind = pending
    memory["resp"][street, kind, outcome] += 1
    F = h["raise_F"].get(street)
    if F is not None:
        p = F if outcome == 0 else 1.0 - F
        memory["fold_ll"][street] += np.log(np.clip(p, 0.02, 1.0))


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def _samples_for(obs: Observation, memory: dict) -> int:
    done = obs.hand_index / memory["num_hands"]
    spent = memory["cpu"] / memory["budget"] if memory["budget"] > 0 else 2.0
    if spent >= 1.0:
        return 0
    n = N_NORMAL if obs.pot < BIG_POT else N_BIG
    if obs.street == Street.PREFLOP and obs.pot <= 4:
        n = N_PRE
    if SMOOTH_PACE:
        # Projected use of the budget at this pace; above 1 the samples shrink
        # smoothly so the budget lasts the whole half (slow machines play a
        # noisier version of the same strategy, not the fallback table bot).
        projected = spent / max(done, 0.05)
        if projected > 0.9:
            n = int(n * min(1.0, (0.9 / projected) ** 2))
        return max(N_FLOOR, n)
    if spent > done + 0.10:              # running ahead of schedule: economise
        n //= 3
    return max(60, n)


def _decide(obs: Observation, memory: dict) -> Action:
    h = memory["hand"]
    if h is None or h["index"] != obs.hand_index:
        h = memory["hand"] = _new_hand_state(obs)
    _replay(obs, memory)
    legal = obs.legal_actions
    if len(legal) == 1:
        return legal[0]

    key = int(obs.street)
    if h["cache"] is None or h["cache"][0] != key:
        n = _samples_for(obs, memory)
        h["cache"] = (key, _simulate(obs, memory, n) if n else None)
    sim = h["cache"][1]
    if sim is None:                       # out of budget: table-driven fallback
        eq = float(fe.preflop_equity(np.array([obs.hole_cards]))[0]) if not obs.board else 0.5
        return _simple(obs, eq)

    w = _weights(sim, h["cls"], memory, h["sizes"])
    share = sim["share"]
    eq = float((w * share).sum() / w.sum())
    cur_pct = sim["pcts"][key]

    pot, cost = obs.pot, min(obs.to_call, obs.my_stack)
    river = obs.street == Street.RIVER
    passive, aggressive = memory["aggr"]
    aggr = (aggressive + AGGR_PRIOR[0]) / (passive + aggressive + AGGR_PRIOR[1])
    realize = 1.0 if river else 1.0 - aggr * (REAL_IP if obs.is_button else REAL_OOP)

    evs = {}
    if Action.FOLD in legal:
        evs[Action.FOLD] = 0.0
    all_in_call = cost >= obs.my_stack
    r_call = 1.0 if all_in_call else realize
    evs[Action.CHECK_CALL] = eq * r_call * (pot + cost) - cost
    if TRAP > 0 and cost == 0 and obs.street > Street.PREFLOP and not obs.is_button and not any(
            r.street == obs.street and not r.is_me for r in obs.action_history):
        # We check first, so the street does not simply check through: the
        # opponent bets with its observed checked-to bet rate beta. Facing a
        # bet of ~0.75 pot we call when it pays (the trap: strong hands win
        # the extra bet) or fold (weak hands lose their equity share).
        beta = _freq_rates(memory, key)[0]
        bet = 0.75 * pot
        after = max(0.0, eq * realize * (pot + 2 * bet) - bet)
        evs[Action.CHECK_CALL] += TRAP * beta * (after - eq * realize * pot)
    F = None
    if Action.RAISE_HALF in legal or Action.RAISE_POT in legal:
        q = _reraise_rate(memory, 1 if obs.to_call > 0 else 0) if USE_LOOKAHEAD else 0.0
        F, eq_above, eq_rr, eq_flat = _range_by_threshold(share, w, cur_pct, q)
        if eq_rr is not None and LOOK_SELECTIVE:
            sel = min(1.0, max(0.0, (0.85 - _freq_rates(memory, key)[2]) / 0.55))
            eq_rr = sel * eq_rr + (1.0 - sel) * eq_above
            eq_flat = sel * eq_flat + (1.0 - sel) * eq_above
        post = _threshold_posterior(memory, key)
    for a in (Action.RAISE_HALF, Action.RAISE_POT):
        if a not in legal:
            continue
        target = obs.raise_to[a]
        mine = target - obs.my_bet
        theirs = min(target - obs.opp_bet, obs.opp_stack)
        jam = mine >= obs.my_stack
        r = 1.0 if (river or jam) else realize
        can_rr = (eq_rr is not None and obs.raises_this_street + 1 < 4
                  and mine < obs.my_stack and theirs < obs.opp_stack)
        if can_rr:
            # Facing their re-raise we fold or get it all in (pot-limit at 40bb
            # commits within one more raise).
            allin = np.maximum(-mine, eq_rr * (pot + obs.my_stack + obs.opp_stack) - obs.my_stack)
            cont = (1.0 - q) * (eq_flat * r * (pot + mine + theirs) - mine) + q * allin
        else:
            cont = eq_above * r * (pot + mine + theirs) - mine
        ev_t = F * pot + (1.0 - F) * cont
        evs[a] = float(post @ ev_t)
    if INFO_KAPPA > 0:
        _add_information_value(evs, obs, h, memory, F, post if F is not None else None)
    if VAR_LAMBDA != 0 or VAR_EXPLOIT != 0:
        # > 0: prefer low variance. < 0: prefer high variance, which keeps the
        # match CI wide; scaled by trust, so it only acts against blueprint-like
        # (equilibrium) opponents, where we cannot gain and a wide CI turns a
        # small deficit into a chop instead of a loss.
        tr = trust(memory)
        lam = VAR_LAMBDA * (tr if VAR_BY_TRUST else 1.0) + VAR_EXPLOIT * (1.0 - tr)
        if lam != 0:
            var = _action_variances(obs, pot, cost, eq, F, eq_above if F is not None else None,
                                    post if F is not None else None)
            for a in evs:
                evs[a] -= lam * var.get(a, 0.0)
    best = max(evs, key=evs.get)
    if BP is not None and (BP_ONLY or BLEND_KAPPA > 0):
        bp = _blueprint_probs(obs, sim)
        if bp is not None:
            best = _blend(evs, bp, obs, memory)
    if best in (Action.RAISE_HALF, Action.RAISE_POT):
        h["raise_F"][key] = F
    h["last"] = (key, eq, evs)
    return best


def _blueprint_probs(obs: Observation, sim: dict):
    """Blueprint action probabilities at this exact node, or None."""
    if BP is None:
        return None
    node = BP_TREE.index.get(ab.history_key(obs.action_history))
    if (node is None or BP_TREE.player[node] != (0 if obs.is_button else 1)
            or BP_TREE.street[node] != int(obs.street)):
        return None
    acts = BP_TREE.actions[node]
    if tuple(acts) != tuple(int(a) for a in obs.legal_actions):
        return None
    if BP_VERSION >= 2:
        lo = BP_TREE.street_off[node]
        probs = BP[BP_TREE.street[node]][lo:lo + len(acts), sim["bucket"]].astype(np.float64)
    else:
        lo = BP_TREE.offsets[node]
        probs = BP[lo:lo + len(acts), sim["bucket"]].astype(np.float64)
    total = probs.sum()
    if total <= 0:
        return None
    return {Action(a): p / total for a, p in zip(acts, probs)}


def deviation(memory: dict) -> float:
    """Largest z-score between the opponent's observed rates and the blueprint's."""
    r, a = memory["resp"], memory["aggr"]
    stats = ((r[0, :, 0].sum(), r[0].sum()), (r[1:, :, 0].sum(), r[1:].sum()),
             (r[..., 2].sum(), r.sum()), (a[1], a.sum()))
    z = 0.0
    for (hits, n), p in zip(stats, BP_REF):
        if n >= 5:
            z = max(z, abs(hits / n - p) / np.sqrt(p * (1 - p) / n))
    return float(z)


def deviation_size(memory: dict) -> float:
    """Largest lower-95%-bound distance between the opponent's rates and the blueprint's."""
    r, a = memory["resp"], memory["aggr"]
    stats = ((r[0, :, 0].sum(), r[0].sum()), (r[1:, :, 0].sum(), r[1:].sum()),
             (r[..., 2].sum(), r.sum()), (a[1], a.sum()))
    size = 0.0
    for (hits, n), p in zip(stats, BP_REF):
        if n >= 5:
            rate = hits / n
            se = np.sqrt(max(rate * (1 - rate), 0.02) / n)
            size = max(size, abs(rate - p) - 1.96 * se)
    return float(size)


def _lcb_gap(hits, n, p):
    rate = hits / n
    return abs(rate - p) - 1.96 * np.sqrt(max(rate * (1 - rate), 0.02) / n)


def situation_trust(memory: dict, street: int, facing: bool) -> float:
    """Trust for one decision, from the opponent behaviour that decision depends on."""
    g = 0 if street == 0 else 1
    f = memory["freq"][0:1].sum(axis=0) if g == 0 else memory["freq"][1:4].sum(axis=0)
    ref = BP_REF_CTX[g]
    if facing:      # they were aggressive: how often do they bet and raise?
        stats = ((f[0, 2], f[0, 1:].sum(), ref[0]), (f[1, 2], f[1].sum(), ref[2]))
    else:           # we may bet: how do they respond to bets?
        stats = ((f[1, 0], f[1].sum(), ref[1]), (f[1, 2], f[1].sum(), ref[2]))
    size = max([_lcb_gap(h, n, p) for h, n, p in stats if n >= CTX_MIN_N] + [0.0])
    return min(1.0, max(0.0, (CTX_MAG1 - size) / (CTX_MAG1 - CTX_MAG0)))


def trust(memory: dict) -> float:
    """1 while the opponent looks like the blueprint, 0 once it clearly deviates."""
    if DEV_MODE == "magnitude":
        return min(1.0, max(0.0, (DEV_MAG1 - deviation_size(memory)) / (DEV_MAG1 - DEV_MAG0)))
    return min(1.0, max(0.0, (DEV_Z1 - deviation(memory)) / (DEV_Z1 - DEV_Z0)))


def _action_variances(obs, pot, cost, eq, F, eq_above, post) -> dict:
    """Variance (chips^2) of each action's result relative to folding now,
    treating the hand as settled after the action (a deliberate simplification)."""
    var = {Action.FOLD: 0.0, Action.CHECK_CALL: eq * (1 - eq) * (pot + cost) ** 2}
    if F is None:
        return var
    f = float(post @ F)
    ec = float(post @ eq_above)
    for a in (Action.RAISE_HALF, Action.RAISE_POT):
        if a not in obs.legal_actions:
            continue
        mine = obs.raise_to[a] - obs.my_bet
        theirs = min(obs.raise_to[a] - obs.opp_bet, obs.opp_stack)
        win, lose = pot + theirs, -mine
        ev = f * pot + (1 - f) * (ec * win + (1 - ec) * lose)
        ex2 = f * pot ** 2 + (1 - f) * (ec * win ** 2 + (1 - ec) * lose ** 2)
        var[a] = max(0.0, ex2 - ev ** 2)
    return var


def _blend(evs: dict, bp: dict, obs: Observation, memory: dict) -> Action:
    acts = list(evs)
    pi = np.array([bp.get(a, 0.0) for a in acts])
    trust_ = trust(memory)
    if CTX_TRUST:
        trust_ = min(trust_, situation_trust(memory, int(obs.street), obs.to_call > 0))
    if not BP_ONLY and trust_ <= 0.0:
        return max(evs, key=evs.get)                  # clear deviation: best response
    if BP_ONLY:
        q = pi / pi.sum()
    else:
        tau = BLEND_KAPPA * max(2.0, obs.pot) * trust_
        ev = np.array([evs[a] for a in acts])
        logits = np.log(pi + 1e-4) + (ev - ev.max()) / tau
        q = np.exp(logits - logits.max())
        q /= q.sum()
    return acts[int(memory["rng"].choice(len(acts), p=q))]


def information_value(memory: dict, h: dict) -> float:
    """Chips one more showdown of this hand is worth to the opponent model."""
    left = max(0.0, 1.0 - memory["hands_done"] / memory["num_hands"])
    hist = memory["hist"]
    u = sum(INFO_STREET_W[st] * PRIOR_N / (PRIOR_N + hist[st, cls].sum())
            for st, cls in h["cls"].items())
    return INFO_KAPPA * left * u


def _add_information_value(evs: dict, obs: Observation, h: dict, memory: dict, F, post) -> None:
    """Credit each action with P(showdown | action) * information value."""
    river = obs.street == Street.RIVER
    all_in_call = obs.to_call > 0 and obs.to_call >= obs.my_stack
    closes = (obs.to_call > 0) or obs.is_button        # call, or check behind in position
    if INFO_CALLS_ONLY and not (river and obs.to_call > 0):
        return
    if not (all_in_call or (river and closes) or river):
        return
    value = information_value(memory, h)
    if value <= 0:
        return
    if all_in_call or (river and closes):
        evs[Action.CHECK_CALL] += value
    if river and F is not None and not INFO_CALLS_ONLY:
        p_call = 1.0 - float(post @ F)                    # a called river raise shows down
        for a in (Action.RAISE_HALF, Action.RAISE_POT):
            if a in evs:
                evs[a] += value * p_call


def _simple(obs: Observation, eq: float) -> Action:
    cost = obs.to_call
    price = cost / max(1, obs.pot + cost)
    if cost and eq < price + 0.05 and Action.FOLD in obs.legal_actions:
        return Action.FOLD
    return Action.CHECK_CALL


def choose_action(obs: Observation, memory: dict) -> Action:
    start = perf_counter()
    try:
        if "rng" not in memory:
            new_half(AgentConfig(), memory)
        action = _decide(obs, memory)
        if action not in obs.legal_actions:
            raise ValueError("illegal action from policy")
        h = memory["hand"]
        h["last_act"] = (int(obs.street), action, obs.to_call)
        if action in (Action.RAISE_HALF, Action.RAISE_POT):
            # Our raise is not in the history yet; if the opponent folds to it
            # the hand ends and only on_hand_end will tell us.
            h["pending"] = (int(obs.street), 1 if obs.to_call > 0 else 0)
    except Exception:  # never crash the server: count it and play safe
        memory["errors"] = memory.get("errors", 0) + 1
        action = Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
        if action not in obs.legal_actions:
            action = obs.legal_actions[0]
    memory["cpu"] = memory.get("cpu", 0.0) + perf_counter() - start
    memory["decisions"] = memory.get("decisions", 0) + 1
    return action


# ---------------------------------------------------------------------------
# Learning from finished hands
# ---------------------------------------------------------------------------

def _learn_showdown(result: HandResult, h: dict, memory: dict) -> None:
    opp = np.array(result.opp_hole, dtype=np.int64)
    board = np.array(result.board, dtype=np.int64)
    hole = np.array(h["hole"], dtype=np.int64)
    rng = memory["rng"]
    for street, cls in h["cls"].items():
        if street == 0:
            pct = float(fe.preflop_percentile(opp[None, :])[0])
        else:
            k = 2 + street
            vis = board[:k]
            used = np.zeros(36, dtype=bool)
            used[hole] = True
            used[vis] = True
            rest = np.flatnonzero(~used)
            ref = rest[rng.random((200, len(rest))).argsort(axis=1)[:, :5]]
            s_ref = street_strength(ref, vis)
            s_opp = street_strength(opp[None, :], vis)[0]
            pct = float((s_ref < s_opp).mean() + 0.5 * (s_ref == s_opp).mean())
        bucket = int(to_bucket(np.array(pct)))
        memory["hist"][street, cls, bucket] += 1
        if cls >= BET and street in h.get("sizes", {}):
            memory["hist_size"][street, cls, h["sizes"][street], bucket] += 1


def on_hand_end(result: HandResult, memory: dict) -> None:
    start = perf_counter()
    try:
        if "rng" not in memory:
            new_half(AgentConfig(), memory)
        memory["hands_done"] += 1
        h = memory.get("hand")
        if h is None or h["index"] != result.hand_index:
            if not result.went_to_showdown and result.winner == "me":
                memory["freq"][0, 1, 0] += 1          # button folded its first action
            return
        pending = h["pending"]
        if pending is not None:
            if not result.went_to_showdown and result.winner == "me":
                _record_response(memory, h, pending, 0)                 # they folded
                memory["freq"][pending[0], 1, 0] += 1
            elif result.went_to_showdown:
                _record_response(memory, h, pending, 1)                 # they called
                memory["freq"][pending[0], 1, 1] += 1
                h["cls"][pending[0]] = max(h["cls"].get(pending[0], -1), CALL)
        last = h.get("last_act")
        if (result.went_to_showdown and pending is None and last is not None and last[0] == 3
                and last[1] == Action.CHECK_CALL and last[2] == 0 and 3 not in h["cls"]):
            h["cls"][3] = CHECK                                         # they checked behind
            memory["freq"][3, 0, 1] += 1
        if result.went_to_showdown and result.opp_hole is not None and len(result.board) == 5:
            _learn_showdown(result, h, memory)
        memory["hand"] = None
    except Exception:
        memory["errors"] = memory.get("errors", 0) + 1
    finally:
        memory["cpu"] = memory.get("cpu", 0.0) + perf_counter() - start


def warm_start(hands: int = 6) -> None:
    """Run a few throwaway decisions at startup (outside the time bank) so the
    first real requests - /new-half above all, which forfeits the half if it
    takes over a second - never pay one-time first-call costs."""
    from game import Game, make_deal
    memory: dict = {}
    new_half(AgentConfig(), memory)
    for d in range(hands):
        game = Game(make_deal("warm-start", d), d % 2, hand_index=d)
        while not game.is_terminal:
            obs = Observation.from_dict(game.observation(game.to_act).to_dict())
            action = choose_action(obs, memory)
            game.apply(action if action in obs.legal_actions else obs.legal_actions[0])
        on_hand_end(HandResult.from_dict(game.hand_result(0).to_dict()), memory)
    new_half(AgentConfig(), {})
