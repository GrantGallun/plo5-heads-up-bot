"""
abs3_runtime.py - the v3 card abstraction at play time (numpy, 1 CPU).

Computes exactly the same buckets as abstraction3.Abs3.bucket (the GPU code
the v3 blueprint was trained with), from the same files:
    data/abs3/board_table_*.npy   sorted ranks of 256 random holdings on every board
    data/abs3/abs3.npz            preflop table, flop/turn centroids, river edges
Costs (one holding): flop ~0.3 ms, turn ~0.1 ms, river ~0.2 ms.
"""
from __future__ import annotations

import glob
import os
from itertools import combinations

import numpy as np

import fasteval as fe

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "abs3")
S_BOARD = 256
NBINS = 20
N_RIV_B = 4
TOP_Q = 0.10
BLOCK_SAMPLES = 128
FLOP_STRIDE = 3

RANK = None
TABLE = None
PRE_LUT = FLOP_C = TURN_C = RIV_P = RIV_B = None
RUNOUTS = np.array(list(combinations(range(28), 2)), dtype=np.int64)[::FLOP_STRIDE]   # (126, 2)
_FLOP_CN = _TURN_CN = None


def load(data_dir: str = DATA) -> bool:
    """Load the tables; False if they are missing (the bot then plays without v3)."""
    global RANK, TABLE, PRE_LUT, FLOP_C, TURN_C, RIV_P, RIV_B, _FLOP_CN, _TURN_CN
    parts = sorted(glob.glob(os.path.join(data_dir, "board_table_*.npy")))
    meta = os.path.join(data_dir, "abs3.npz")
    if not parts or not os.path.exists(meta):
        return False
    fe.build()
    valid = fe.SCORE_FLAT >= 0
    uniq = np.unique(fe.SCORE_FLAT[valid])
    rank = np.full(fe.SCORE_FLAT.shape, -1, np.int32)
    rank[valid] = np.searchsorted(uniq, fe.SCORE_FLAT[valid]).astype(np.int32)
    RANK = rank
    TABLE = np.concatenate([np.load(p) for p in parts])
    z = np.load(meta)
    PRE_LUT = z["pre_lut"].astype(np.int64)
    FLOP_C, TURN_C = z["flop_c"].astype(np.float32), z["turn_c"].astype(np.float32)
    RIV_P, RIV_B = z["riv_p_edges"].astype(np.float32), z["riv_b_edges"].astype(np.float32)
    _FLOP_CN = (FLOP_C * FLOP_C).sum(1)
    _TURN_CN = (TURN_C * TURN_C).sum(1)
    return TABLE.shape == (376992, S_BOARD)


def _best(holes, boards):
    """Best dense rank of each holding (M, 5) on its board (M, 5) (or one board)."""
    if fe.PAIRBEST is not None:
        return fe.full_ranks(holes, boards)
    pr = fe.pair_indices(holes)                                   # (M, 10)
    tr = fe.triple_indices(boards)                                # (M or 1, 10)
    return RANK[pr[:, :, None] + tr[:, None, :]].max(axis=(1, 2))


def _colex(cards):
    s = np.sort(cards, axis=-1)
    return fe.colex_index(s)


def river_p(hole, boards):
    """Percentile of one holding against random holdings on each board (B, 5)."""
    holes = np.broadcast_to(np.asarray(hole, dtype=np.int64), (len(boards), 5))
    own = _best(np.ascontiguousarray(holes), boards).astype(np.int16)[:, None]
    tab = TABLE[_colex(boards)]
    return ((tab < own).sum(1) + 0.5 * (tab == own).sum(1)) / S_BOARD


def _cdf(p):
    idx = np.clip((p * NBINS).astype(np.int64), 0, NBINS - 1)
    hist = np.bincount(idx, minlength=NBINS).astype(np.float32) / len(p)
    return np.cumsum(hist)[:-1]


def _remaining(known):
    used = np.zeros(36, dtype=bool)
    used[list(known)] = True
    return np.flatnonzero(~used)


def flop_feature(hole, flop):
    rem = _remaining(list(hole) + list(flop))                    # (28,)
    ro = rem[RUNOUTS]                                             # (126, 2)
    boards = np.concatenate([np.broadcast_to(np.asarray(flop), (len(ro), 3)), ro], axis=1)
    return _cdf(river_p(hole, boards))


def turn_feature(hole, board4):
    rem = _remaining(list(hole) + list(board4))                  # (27,)
    boards = np.concatenate([np.broadcast_to(np.asarray(board4), (len(rem), 4)), rem[:, None]], axis=1)
    return _cdf(river_p(hole, boards))


def river_block(hole, board, rng):
    tab = TABLE[_colex(np.asarray(board)[None])][0]
    thr = tab[int((1 - TOP_Q) * S_BOARD)]
    base = (tab >= thr).mean()
    rest = _remaining(list(hole) + list(board))
    opp = rest[np.argsort(rng.random((BLOCK_SAMPLES, len(rest))), axis=1)[:, :5]]
    sc = _best(opp, np.asarray(board, dtype=np.int64)[None]).astype(np.int16)
    return float((sc >= thr).mean() / base)


def bucket(street: int, hole, board, rng) -> int:
    hole = np.asarray(hole, dtype=np.int64)
    if street == 0:
        return int(PRE_LUT[_colex(hole[None])[0]])
    if street == 1:
        f = flop_feature(hole, board[:3])
        return int(np.argmin(_FLOP_CN - 2 * FLOP_C @ f))
    if street == 2:
        f = turn_feature(hole, board[:4])
        return int(np.argmin(_TURN_CN - 2 * TURN_C @ f))
    b = np.asarray(board[:5], dtype=np.int64)
    p = float(river_p(hole, b[None])[0])
    bl = river_block(hole, b, rng)
    return int(np.searchsorted(RIV_P, p, "right")) * N_RIV_B + int(np.searchsorted(RIV_B, bl, "right"))
