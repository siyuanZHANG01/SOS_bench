from __future__ import annotations
import math
import numpy as np
from src.utils import (
    ols_line_and_std,
    ols_std_segment_internal,
)

def eqpop_edges(n: int, m_override: int | None = None) -> np.ndarray:
    if n <= 0:
        return np.array([0, 0], dtype=int)
    if m_override is None:
        m = int(math.floor(math.sqrt(n)))
    else:
        m = int(m_override)
    m = max(1, min(m, n))
    edges = np.linspace(0, n, m+1, dtype=int)
    edges = np.unique(edges)
    if edges[-1] != n:
        edges[-1] = n
    return edges

def global_local_mse(x_sorted: np.ndarray, m_override: int) -> tuple[float, float, int, float]:
    n = x_sorted.size
    if n == 0:
        return np.nan, np.nan, 0, np.nan

    edges = eqpop_edges(n, m_override)
    m_used = max(0, edges.size - 1)
    if m_used == 0:
        return np.nan, np.nan, 0, np.nan

    # Representative points
    reps_x = []
    for j in range(m_used):
        lo, hi = edges[j], edges[j+1]
        if hi <= lo:
            continue
        # Modified: Use start key (minimum key) of the segment
        reps_x.append(x_sorted[lo])
        
    reps_x = np.array(reps_x, dtype=x_sorted.dtype if x_sorted.dtype.kind=='f' else np.float64)
    reps_y = np.arange(reps_x.size, dtype=float)              # Original Rank
    _, _, global_std = ols_line_and_std(reps_x, reps_y)

    # Segment internal L-infinity
    std_list = []
    for j in range(m_used):
        lo, hi = edges[j], edges[j+1]
        if hi <= lo:
            continue
        xs = x_sorted[lo:hi]
        ys = np.arange(xs.size, dtype=float)                  # Original Rank
        std_j = ols_std_segment_internal(xs, ys)
        std_list.append(std_j)

    if std_list:
        local_mean = float(np.mean(std_list))
        local_std  = float(np.std(std_list))
    else:
        local_mean = np.nan
        local_std  = np.nan

    return float(global_std), local_mean, m_used, local_std