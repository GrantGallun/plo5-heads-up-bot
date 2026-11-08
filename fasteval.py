"""
fasteval.py - vectorised (numpy) Short Deck Omaha evaluation.

Every 5-card hand is split as (2 hole cards, 3 board cards). We store one
int32 score for every (hole pair, board triple) combination:

    SCORE[pair_index, triple_index]      630 x 7140 table (18 MB)

so the best Omaha hand for N sampled hands is one fancy-index gather of shape
(N, 10 pairs, 10 triples) followed by a max. Scores are identical to
evaluator.omaha_score (same table, same ordering).

Also provides:
    PAIR_IDX[a, b], TRIPLE_IDX[a, b, c]   symmetric index lookups
    colex_index(cards)                    0..376991 index of sorted 5-card sets
    PREFLOP_EQ                            equity vs a random hand for every
                                          5-card holding (from preflop_equity.npy)
"""
from __future__ import annotations

import os
from itertools import combinations
from math import comb

import numpy as np

from evaluator import get_table

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS5 = np.array(list(combinations(range(5), 2)), dtype=np.int64)    # (10, 2)
TRIPLES5 = np.array(list(combinations(range(5), 3)), dtype=np.int64)  # (10, 3)
TRIPLES4 = np.array(list(combinations(range(4), 3)), dtype=np.int64)  # (4, 3)

PAIR_IDX = np.full((36, 36), -1, dtype=np.int64)
TRIPLE_IDX = np.full((36, 36, 36), -1, dtype=np.int64)
BINOM = np.array([[comb(n, k) for k in range(6)] for n in range(36)], dtype=np.int64)
SCORE: np.ndarray | None = None
SCORE_FLAT: np.ndarray | None = None      # SCORE.ravel(): index pair * 7140 + triple
NT = 7140
PREFLOP_EQ: np.ndarray | None = None
PREFLOP_PCT: np.ndarray | None = None

for _i, (_a, _b) in enumerate(combinations(range(36), 2)):
    PAIR_IDX[_a, _b] = PAIR_IDX[_b, _a] = _i
for _i, (_a, _b, _c) in enumerate(combinations(range(36), 3)):
    for _p in ((_a, _b, _c), (_a, _c, _b), (_b, _a, _c), (_b, _c, _a), (_c, _a, _b), (_c, _b, _a)):
        TRIPLE_IDX[_p] = _i


PAIR_OFF = np.where(PAIR_IDX >= 0, PAIR_IDX * NT, -1)


def colex_index(sorted_cards: np.ndarray) -> np.ndarray:
    """Colex rank of ascending 5-card rows: sum C(c_k, k+1). Shape (..., 5)."""
    out = BINOM[sorted_cards[..., 0], 1]
    for k in range(1, 5):
        out = out + BINOM[sorted_cards[..., k], k + 1]
    return out


def build() -> None:
    """Build SCORE (about 1 s) and load the preflop table if present."""
    global SCORE, SCORE_FLAT, PREFLOP_EQ, PREFLOP_PCT
    if SCORE is not None:
        return
    table = get_table()
    combos = np.array(list(combinations(range(36), 5)), dtype=np.int64)  # (376992, 5)
    masks = (np.int64(1) << combos).sum(axis=1)
    scores = np.array([table[m] for m in masks.tolist()], dtype=np.int32)
    score = np.full((630, 7140), -1, dtype=np.int32)
    for pi, (i, j) in enumerate(combinations(range(5), 2)):
        rest = [k for k in range(5) if k not in (i, j)]
        p = PAIR_IDX[combos[:, i], combos[:, j]]
        t = TRIPLE_IDX[combos[:, rest[0]], combos[:, rest[1]], combos[:, rest[2]]]
        score[p, t] = scores
    SCORE = score
    SCORE_FLAT = score.ravel()
    path = os.path.join(HERE, "preflop_equity.npy")
    if os.path.exists(path):
        eq = np.load(path).astype(np.float32)
        PREFLOP_EQ = eq
        order = np.argsort(eq, kind="stable")
        pct = np.empty_like(eq)
        pct[order] = (np.arange(len(eq), dtype=np.float32) + 0.5) / len(eq)
        PREFLOP_PCT = pct


def pair_indices(holes: np.ndarray) -> np.ndarray:
    """(N, 5) hole cards -> (N, 10) row offsets (pair index * 7140) into SCORE_FLAT."""
    return PAIR_OFF[holes[:, PAIRS5[:, 0]], holes[:, PAIRS5[:, 1]]]


def triple_indices(boards: np.ndarray) -> np.ndarray:
    """(N, k) boards (k = 3, 4, 5) -> (N, C(k,3)) triple indices."""
    k = boards.shape[1]
    tri = TRIPLES5 if k == 5 else TRIPLES4 if k == 4 else np.array([[0, 1, 2]])
    return TRIPLE_IDX[boards[:, tri[:, 0]], boards[:, tri[:, 1]], boards[:, tri[:, 2]]]


def best_scores(pairs: np.ndarray, triples: np.ndarray) -> np.ndarray:
    """pairs (N or 1, 10) from pair_indices, triples (N or 1, T) -> (N,) best score."""
    return SCORE_FLAT[pairs[:, :, None] + triples[:, None, :]].max(axis=(1, 2))


# ---- PAIRBEST: best dense rank of each of the 630 pairs on every 5-card board ----
# data/pairbest/pairbest_*.npy (precomputed offline, ~450 MB). With it the
# best hand of a holding on a full board is 10 lookups instead of 100. Dense
# ranks keep the order of SCORE exactly, so every comparison is unchanged.
PAIRBEST: np.ndarray | None = None


def load_pairbest(path: str | None = None) -> bool:
    global PAIRBEST
    import glob
    d = path or os.path.join(HERE, "data", "pairbest")
    parts = sorted(glob.glob(os.path.join(d, "pairbest_*.npy")))
    if not parts:
        return False
    try:
        pb = np.concatenate([np.load(p) for p in parts])
    except (OSError, ValueError, MemoryError):
        return False
    if pb.shape != (376992, 630):
        return False
    PAIRBEST = pb
    return True


def pair_ids(holes: np.ndarray) -> np.ndarray:
    """(N, 5) hole cards -> (N, 10) pair ids 0..629."""
    return PAIR_IDX[holes[:, PAIRS5[:, 0]], holes[:, PAIRS5[:, 1]]]


def board_ids(boards: np.ndarray) -> np.ndarray:
    """(N, 5) boards -> (N,) colex ids of the sorted board."""
    return colex_index(np.sort(boards, axis=1))


def full_ranks(holes: np.ndarray, boards: np.ndarray) -> np.ndarray:
    """Best dense rank of each (N or 1, 5) holding on each (N or 1, 5) full board
    (needs PAIRBEST). Same order as omaha_scores."""
    return PAIRBEST[board_ids(boards)[:, None], pair_ids(holes)].max(axis=1)


def omaha_scores(holes: np.ndarray, boards: np.ndarray) -> np.ndarray:
    """Best Omaha score for each row of (N, 5) holes and (N, 5) boards."""
    return best_scores(pair_indices(holes), triple_indices(boards))


def preflop_equity(holes: np.ndarray) -> np.ndarray:
    return PREFLOP_EQ[colex_index(np.sort(holes, axis=1))]


def preflop_percentile(holes: np.ndarray) -> np.ndarray:
    return PREFLOP_PCT[colex_index(np.sort(holes, axis=1))]
