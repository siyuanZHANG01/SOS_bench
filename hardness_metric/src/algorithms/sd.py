from __future__ import annotations
import numpy as np
from src.utils import ols_std_segment_internal

def _as_float_array(keys: np.ndarray) -> np.ndarray:
    x = np.asarray(keys).reshape(-1)
    if np.issubdtype(x.dtype, np.integer):
        return x.astype(np.float64, copy=False)
    if x.dtype in (np.float16, np.float32):
        return x.astype(np.float64, copy=False)
    return x

def _second_derivative_scores(keys: np.ndarray) -> np.ndarray:
    x = _as_float_array(keys)
    n = x.size
    score = np.zeros(n, dtype=float)
    if n <= 2:
        return score
    gaps = np.diff(x)                  # size n-1
    dd = np.abs(gaps[1:] - gaps[:-1])  # size n-2, corresponding to i=2..n-1
    score[2:] = dd
    score[0] = 0.0
    score[1] = 0.0
    return score

def _select_splits_by_second_derivative(keys: np.ndarray, seg_budget: int,
                                        window: int = 128, radius: int = 3) -> list[int]:
    x = _as_float_array(keys)
    n = x.size
    if n <= 1 or seg_budget <= 0:
        return [n - 1] if n > 0 else []

    score = _second_derivative_scores(x)
    noSeg = int(max(1, min(seg_budget, n)))
    halfSegSize = max(1, int(round(n / (2.0 * noSeg))))
    remaining = max(0, noSeg - 1)

    splits: list[int] = []
    while remaining > 0:
        i = int(np.argmax(score))
        best = score[i]
        if best <= 0.0:
            break

        choose = i
        if 1 <= i-1 <= n-2 and 1 <= i+1 <= n-2:
            choose = i-1 if score[i+1] < score[i-1] else i
        elif 1 <= i-1 <= n-2:
            choose = i - 1
        elif 1 <= i+1 <= n-2:
            choose = i
        else:
            choose = min(max(i, 1), n-2)

        splits.append(int(choose))
        remaining -= 1

        lo = max(0, choose - halfSegSize)
        hi = min(n - 1, choose + halfSegSize)
        score[lo:hi + 1] = -1.0

    if n - 1 not in splits:
        splits.append(n - 1)
    splits = sorted(set(splits))
    return splits

def sd_partition_and_stats(keys: np.ndarray, seg_budget: int, window: int = 128, radius: int = 3):
    x = _as_float_array(keys)
    n = x.size
    if n == 0:
        return [], float('nan'), float('nan')

    seg_budget = max(1, min(int(seg_budget), n))
    splits = _select_splits_by_second_derivative(x, seg_budget, window=window, radius=radius)

    bounds = [0] + [int(s) + 1 for s in splits]
    if bounds[-1] != n:
        bounds[-1] = n

    errs = []
    starts = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        xs = x[a:b]
        m = b - a
        ys = np.arange(m, dtype=float)                      # Original Rank
        errs.append(ols_std_segment_internal(xs, ys))
        starts.append(xs[0])

    local_avg = float(np.mean(errs)) if errs else float('nan')
    local_std = float(np.std(errs))  if errs else float('nan')
    return starts, local_avg, local_std
