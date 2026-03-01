from __future__ import annotations
import numpy as np
from math import inf

def shrinkingcone_segments_and_starts(keys: np.ndarray, eps: int) -> tuple[int, list]:
    n_keys = keys.size
    if n_keys == 0:
        return 0, []
    if n_keys == 1:
        return 1, [keys[0]]

    segs = 1
    starts = [keys[0]]

    key_start = keys[0]
    last_key  = keys[0]
    slope_low  = -inf
    slope_high =  inf
    n = 1

    for i in range(1, n_keys):
        key = keys[i]
        if key == last_key:
            continue
        if key < last_key:
            raise ValueError("Key must be largest to be appended")

        delta = key - key_start
        # Keep native dtype (e.g., longdouble) to avoid forcing float64.
        center = n / delta

        if slope_low <= center <= slope_high:
            slope_high_temp = (n + eps) / delta
            slope_low_temp  = (n - eps) / delta
            slope_high = min(slope_high, slope_high_temp)
            slope_low  = max(slope_low,  slope_low_temp)
            last_key = key
            n += 1
        else:
            segs += 1
            starts.append(key)
            key_start = key
            last_key  = key
            slope_low  = -inf
            slope_high =  inf
            n = 1
    return segs, starts