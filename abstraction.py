"""
abstraction.py - card abstraction and the exact betting tree, shared by the
offline CFR trainer and the live bot (strategy.py).

Card abstraction: on every street a hand is put in one of NB_BP buckets by its
strength percentile among random opponent holdings on the same board
(preflop: the precomputed preflop-equity percentile). Strength is the made
hand plus a bonus for flush / straight draws, exactly as the live bot
measures it, so training and play use the same buckets.

Betting tree: the real game's tree (pot-limit sizes, 4-raise cap, 80-chip
stacks) built with game.Game. The button is seat 0. There is no action
abstraction because the game only has four actions.
"""
from __future__ import annotations

import numpy as np

import fasteval as fe
from game import Action, Game, make_deal

BP_EDGES = np.array([0.0, 0.20, 0.35, 0.50, 0.62, 0.72, 0.80, 0.87, 0.93, 0.97, 1.0])
NB_BP = len(BP_EDGES) - 1

WINDOWS = np.zeros((6, 9), dtype=bool)
for _w, _ranks in enumerate([range(0, 5), range(1, 6), range(2, 7), range(3, 8), range(4, 9),
                             (8, 0, 1, 2, 3)]):
    WINDOWS[_w, list(_ranks)] = True
SUITS = np.arange(4)


def bp_bucket(pct):
    return np.searchsorted(BP_EDGES[1:-1], pct, "right")


# ---- v2: potential-aware buckets --------------------------------------------
# Preflop and river: strength percentile, finer near the top. Flop and turn:
# current-strength percentile (8) x equity against a random hand (6), so a big
# draw (weak now, strong equity) and a made hand of the same strength differ.
PRE_EDGES = np.array([0, .15, .30, .42, .52, .60, .67, .73, .78, .83, .87, .90, .93, .95, .97, .985, 1.0])
STR_EDGES = np.array([0, .25, .45, .60, .72, .82, .90, .95, 1.0])
EQ_EDGES = np.array([0, .30, .40, .50, .60, .70, 1.0])
RIV_EDGES = np.array([0, .20, .35, .50, .60, .70, .78, .85, .90, .93, .95, .97, .985, .995, 1.0])
NB_STREET = (len(PRE_EDGES) - 1, (len(STR_EDGES) - 1) * (len(EQ_EDGES) - 1),
             (len(STR_EDGES) - 1) * (len(EQ_EDGES) - 1), len(RIV_EDGES) - 1)


def bucket2(street: int, pct, eq=None):
    """v2 bucket for a street from strength percentile (and equity on flop/turn)."""
    if street == 0:
        return np.searchsorted(PRE_EDGES[1:-1], pct, "right")
    if street == 3:
        return np.searchsorted(RIV_EDGES[1:-1], pct, "right")
    n_eq = len(EQ_EDGES) - 1
    return (np.searchsorted(STR_EDGES[1:-1], pct, "right") * n_eq
            + np.searchsorted(EQ_EDGES[1:-1], eq, "right"))


def strength_rows(holes: np.ndarray, boards: np.ndarray) -> np.ndarray:
    """Strength of each (M, 5) holding on its own (M, k) board, k = 3, 4 or 5:
    made-hand category + kicker fraction, raised for flush/straight draws."""
    m, k = boards.shape
    s = fe.best_scores(fe.pair_indices(holes), fe.triple_indices(boards)).astype(np.float64) / (1 << 20)
    if k == 5:
        return s
    flop = k == 3
    board_suits = ((boards[:, :, None] & 3) == SUITS).sum(axis=1)            # (M, 4)
    hole_suits = ((holes[:, :, None] & 3) == SUITS).sum(axis=1)
    flush_draw = ((hole_suits >= 2) & (board_suits == 2)).any(axis=1)
    rows = np.arange(m)[:, None]
    hole_ranks = np.zeros((m, 9), dtype=bool)
    hole_ranks[rows, holes >> 2] = True
    board_ranks = np.zeros((m, 9), dtype=bool)
    board_ranks[rows, boards >> 2] = True
    b_in = np.minimum(3, (WINDOWS[None] & board_ranks[:, None, :]).sum(axis=2))          # (M, 6)
    h_in = np.minimum(2, (hole_ranks[:, None, :] & WINDOWS[None] & ~board_ranks[:, None, :]).sum(axis=2))
    draws4 = ((b_in + h_in) == 4).sum(axis=1)
    bonus = np.where(flush_draw, 3.6 if flop else 3.3, 0.0)
    bonus = np.maximum(bonus, np.where(draws4 >= 2, 3.3 if flop else 3.0,
                                       np.where(draws4 == 1, 2.6 if flop else 2.35, 0.0)))
    bonus = bonus + np.where(flush_draw & (draws4 > 0), 0.3, 0.0)
    return np.maximum(s, bonus)


def strength(holes: np.ndarray, board: np.ndarray) -> np.ndarray:
    """strength_rows with one shared board (1-D, 3-5 cards)."""
    return strength_rows(holes, np.broadcast_to(board, (len(holes), len(board))))


# ---------------------------------------------------------------------------
# Betting tree
# ---------------------------------------------------------------------------

class Tree:
    """Arrays describing every decision node and leaf.

    node ids 0..N-1 are decision nodes; children[n] is a list of
    (action, child) where child >= 0 is a node and child < 0 is leaf ~child.
    hist[n] is the action history string ('0'..'3' per action) -> n in index.
    """

    def __init__(self) -> None:
        self.player, self.street, self.actions, self.children, self.hist = [], [], [], [], []
        self.leaf_kind, self.leaf_u0, self.leaf_c = [], [], []
        self.index: dict[str, int] = {}
        self._build(Game(make_deal("tree", 0), 0), "")
        self.offsets = np.cumsum([0] + [len(a) for a in self.actions])
        self.n = len(self.player)
        # Per-street layout (v2 blueprints): node n's actions occupy rows
        # street_off[n] .. +len(actions) of its street's (rows, buckets) table.
        self.street_off = [0] * self.n
        self.street_rows = [0, 0, 0, 0]
        self.street_starts = [[], [], [], []]
        for n in range(self.n):
            st = self.street[n]
            self.street_off[n] = self.street_rows[st]
            self.street_starts[st].append(self.street_rows[st])
            self.street_rows[st] += len(self.actions[n])

    def _build(self, g: Game, h: str) -> int:
        if g.is_terminal:
            res = g.result()
            if res.went_to_showdown:
                self.leaf_kind.append(1)
                self.leaf_u0.append(0.0)
                self.leaf_c.append(float(g.contributed[0]))
            else:
                self.leaf_kind.append(0)
                self.leaf_u0.append(float(res.nets[0]))
                self.leaf_c.append(0.0)
            return ~(len(self.leaf_kind) - 1)
        n = len(self.player)
        self.player.append(g.to_act)
        self.street.append(int(g.street))
        legal = list(g.legal_actions())
        self.actions.append([int(a) for a in legal])
        self.children.append(None)
        self.hist.append(h)
        self.index[h] = n
        kids = []
        for a in legal:
            c = g.clone()
            c.apply(a)
            kids.append(self._build(c, h + str(int(a))))
        self.children[n] = kids
        return n


def history_key(action_history) -> str:
    return "".join(str(int(r.action)) for r in action_history)
