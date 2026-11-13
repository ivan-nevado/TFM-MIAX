"""Five-block monthly split (paper, section 3.1 and online appendix figure IA.1).

Each month the observations are randomly allocated to five blocks. Each block is the test set
once; of the four others, one is the validation set and three are the training set (60:20:20).
Paper silent, defaults used: the seed of a month is ``base_seed + yyyymm`` and the validation
block for test block ``k`` is ``(k + 1) mod 5``. Every model and benchmark uses these blocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_blocks(n: int, yyyymm: int, base_seed: int, n_blocks: int = 5) -> np.ndarray:
    """Block (0..n_blocks-1) of each of ``n`` observations, given in a fixed order (sorted by id).

    A random permutation is dealt round-robin, so block sizes differ by at most one.
    """
    rng = np.random.default_rng(int(base_seed) + int(yyyymm))
    perm = rng.permutation(n)
    blocks = np.empty(n, dtype=np.int8)
    blocks[perm] = np.arange(n) % n_blocks
    return blocks


def add_blocks(month: pd.DataFrame, base_seed: int, n_blocks: int = 5) -> pd.DataFrame:
    """Sort one month's sample by id and add the ``block`` column."""
    month = month.sort_values("id", kind="stable").reset_index(drop=True)
    date = month["date"].iloc[0]
    month["block"] = assign_blocks(len(month), date.year * 100 + date.month, base_seed, n_blocks)
    return month


def roles(
    blocks: np.ndarray, test_block: int, n_blocks: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boolean masks (train, validation, test) for a test block."""
    test = blocks == test_block
    val = blocks == (test_block + 1) % n_blocks
    train = ~test & ~val
    return train, val, test
