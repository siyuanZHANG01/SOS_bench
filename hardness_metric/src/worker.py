from __future__ import annotations
from __future__ import annotations
import os
import numpy as np
from src.utils import (
    load_series_npy, load_series_bin_uint64, 
    need_high_precision, make_strictly_increasing_with_count,
    ols_line_and_std, ols_std_segment_internal,
    eqpop_edges, calculate_sliding_median_std
)
from src.algorithms.pgm import pgm_segments_and_starts
from src.algorithms.sc import shrinkingcone_segments_and_starts
from src.algorithms.sd import sd_partition_and_stats, _select_splits_by_second_derivative

def recover_bounds_from_keys_sequential(x: np.ndarray, keys: list) -> list[int]:
    n = x.size
    bounds = [0]
    current_idx = 0
    targets = keys[1:] 
    for k in targets:
        sub_x = x[current_idx:]
        idx_in_sub = np.searchsorted(sub_x, k, side='left')
        real_idx = current_idx + idx_in_sub
        if real_idx > n: real_idx = n
        bounds.append(real_idx)
        current_idx = real_idx
        if current_idx >= n: break
    if bounds[-1] != n: bounds.append(n)
    return bounds

def compute_local_l2_from_bounds(x: np.ndarray, bounds: list) -> float:
    if len(bounds) < 2: return 0.0
    errs = []
    for i in range(len(bounds) - 1):
        lo, hi = int(bounds[i]), int(bounds[i+1])
        if hi <= lo: continue
        seg_x = x[lo:hi]
        if seg_x.size > 0:
            seg_x = seg_x - seg_x[0]
        seg_y = np.arange(hi - lo, dtype=float)
        errs.append(ols_std_segment_internal(seg_x, seg_y))
    return float(np.mean(errs)) if errs else 0.0

def _search_eps_for_target_segs(
    x: np.ndarray,
    target_segs: int,
    seg_func,
    eps_min: int = 1,
    eps_max: int = 1 << 30,
    tolerance: float = 0.2, 
) -> tuple[int, int, list]:
    n = x.size
    if n == 0: return 0, 0, []
    target = max(1, int(target_segs))
    tol_min = int(target * (1.0 - tolerance))
    tol_max = int(target * (1.0 + tolerance))

    eps = max(eps_min, 1)
    segs, starts = seg_func(eps)
    
    if segs < target: return eps, segs, starts
    if tol_min <= segs <= tol_max: return eps, segs, starts

    low_eps = eps
    high_eps = eps * 2
    best_eps = eps
    best_segs = segs
    best_starts = starts
    min_diff = abs(segs - target)
    found_upper = False

    for _ in range(32):
        if high_eps > eps_max: high_eps = eps_max
        segs_high, starts_high = seg_func(high_eps)
        
        if tol_min <= segs_high <= tol_max:
            return high_eps, segs_high, starts_high
            
        diff = abs(segs_high - target)
        if diff < min_diff:
            min_diff = diff
            best_eps = high_eps
            best_segs = segs_high
            best_starts = starts_high

        if segs_high < target:
            found_upper = True
            break
        if high_eps >= eps_max: break
        low_eps = high_eps
        high_eps *= 2

    if not found_upper: return best_eps, best_segs, best_starts

    lo = low_eps
    hi = high_eps
    max_steps = 20
    steps = 0
    while lo <= hi and steps < max_steps:
        steps += 1
        mid = (lo + hi) // 2
        if mid == 0: mid = 1
        segs_mid, starts_mid = seg_func(mid)
        
        if tol_min <= segs_mid <= tol_max:
            return mid, segs_mid, starts_mid
            
        diff = abs(segs_mid - target)
        if diff < min_diff:
            min_diff = diff
            best_eps = mid
            best_segs = segs_mid
            best_starts = starts_mid
        
        if segs_mid > tol_max: lo = mid + 1
        elif segs_mid < tol_min: hi = mid - 1
        else: return mid, segs_mid, starts_mid

    return best_eps, best_segs, best_starts

def force_equi_depth_fallback(x: np.ndarray, target_m: int):
    m = max(1, min(int(target_m), x.size))
    bounds = eqpop_edges(x.size, m_override=m)
    rep_indices = bounds[:-1]
    xs = x[rep_indices]
    if xs.size > 0:
        xs = xs - xs[0]
    ys = np.arange(len(xs), dtype=float)

    _, _, p_global = ols_line_and_std(xs, ys)
    p_global_med = calculate_sliding_median_std(xs, ys)
    p_local = compute_local_l2_from_bounds(x, bounds)
    
    return p_global, p_global_med, p_local, len(rep_indices)

def compute_one_file(fpath: str,
                     file_idx: int,
                     input_format: str,
                     bin_endian: str,
                     npz_key: str | None,
                     mmap: bool,
                     mode: str,
                     jitter_duplicates: bool,
                     dtype_spec: str,
                     pgm_errors: list[float] | None,
                     mse_segs: list[int] | None,
                     sd_segs: list[int] | None,
                     sd_window: int,
                     sd_radius: int):
    try:
        if input_format == "npy":
            raw = load_series_npy(fpath, npz_key=npz_key, mmap=mmap)
        else:
            raw = load_series_bin_uint64(fpath, endian=bin_endian)

        if raw.size == 0: return None

        use_high_precision = need_high_precision(dtype_spec, raw)
        x = raw.astype(np.float128, copy=False) if use_high_precision else raw.astype(np.float64, copy=False)

        x.sort(kind="quicksort")

        if jitter_duplicates:
            x, jittered = make_strictly_increasing_with_count(x)
        else:
            jittered = 0

        if x.size > 0:
            x_base = x[0]
            x_normalized = x - x_base
        else:
            x_normalized = x

        p_global = 0.0
        p_global_med = 0.0
        p_local = 0.0
        m_used = 0
        info = {}

        if mode == "pgm":
            if pgm_errors is None: raise ValueError("Requires --pgm_error_file")
            target_segs = int(pgm_errors[file_idx])

            def _func(e): return pgm_segments_and_starts(x_normalized, int(e))

            best_eps, m_used, starts = _search_eps_for_target_segs(x_normalized, target_segs, _func)
            
            if m_used < target_segs / 2 or m_used > target_segs * 2:
                print(f"[WARN] PGM failed to match target segments for {os.path.basename(fpath)}. "
                      f"Target={target_segs}, Got={m_used}. Fallback to Equi-depth.")
                
                p_global, p_global_med, p_local, m_used = force_equi_depth_fallback(x, target_segs)
                info = {"m_used": m_used, "eps_used": -1, "fallback": True} 
            else:
                bounds = recover_bounds_from_keys_sequential(x_normalized, starts)
                rep_indices = bounds[:-1]
                xs = x_normalized[rep_indices]
                ys = np.arange(len(xs), dtype=float)
                
                _, _, p_global = ols_line_and_std(xs, ys)
                p_global_med = calculate_sliding_median_std(xs, ys)
                p_local = compute_local_l2_from_bounds(x_normalized, bounds)
                info = {"m_used": len(rep_indices), "eps_used": int(best_eps), "fallback": False}

        elif mode == "sc":
            if pgm_errors is None: raise ValueError("Requires --pgm_error_file")
            target_segs = int(pgm_errors[file_idx])

            def _func(e): return shrinkingcone_segments_and_starts(x_normalized, int(e))
            
            best_eps, m_used, starts = _search_eps_for_target_segs(x_normalized, target_segs, _func)
            
            if m_used < target_segs / 2 or m_used > target_segs * 2:
                print(f"[WARN] SC failed to match target segments for {os.path.basename(fpath)}. "
                      f"Target={target_segs}, Got={m_used}. Fallback to Equi-depth.")
                
                p_global, p_global_med, p_local, m_used = force_equi_depth_fallback(x, target_segs)
                info = {"m_used": m_used, "eps_used": -1, "fallback": True}
            else:
                bounds = recover_bounds_from_keys_sequential(x_normalized, starts)
                rep_indices = bounds[:-1]
                xs = x_normalized[rep_indices]
                ys = np.arange(len(xs), dtype=float)
                
                _, _, p_global = ols_line_and_std(xs, ys)
                p_global_med = calculate_sliding_median_std(xs, ys)
                p_local = compute_local_l2_from_bounds(x_normalized, bounds)
                info = {"m_used": len(rep_indices), "eps_used": int(best_eps), "fallback": False}

        elif mode == "mse":
            if mse_segs is None: raise ValueError("Requires --mse_segs_file")
            m = max(1, min(int(mse_segs[file_idx]), x.size))
            
            bounds = eqpop_edges(x.size, m_override=m)
            rep_indices = bounds[:-1]
            xs = x_normalized[rep_indices]
            ys = np.arange(len(xs), dtype=float)
            
            _, _, p_global = ols_line_and_std(xs, ys)
            p_global_med = calculate_sliding_median_std(xs, ys)
            p_local = compute_local_l2_from_bounds(x_normalized, bounds)
            
            info = {"m_used": len(rep_indices)}

        elif mode == "sd":
            if sd_segs is None: raise ValueError("Requires --sd_segs_file")
            m = max(1, min(int(sd_segs[file_idx]), x.size))
            
            splits = _select_splits_by_second_derivative(x_normalized, m, window=sd_window, radius=sd_radius)
            bounds = [0] + [int(s) + 1 for s in splits]
            if bounds[-1] != x.size: bounds[-1] = x.size
            bounds = sorted(list(set(bounds)))
            bounds = np.array(bounds, dtype=int)
            
            rep_indices = bounds[:-1]
            if len(rep_indices) == 0: rep_indices = [0]
            
            xs = x_normalized[rep_indices]
            ys = np.arange(len(xs), dtype=float)
            
            _, _, p_global = ols_line_and_std(xs, ys)
            p_global_med = calculate_sliding_median_std(xs, ys)
            p_local = compute_local_l2_from_bounds(x_normalized, bounds)
            
            info = {"m_used": len(rep_indices)}

        else:
            raise ValueError(f"Unknown mode: {mode}")

        name = os.path.splitext(os.path.basename(fpath))[0]
        return (name, p_global, p_global_med, p_local, int(x.size), int(jittered), info)
    except Exception as ex:
        import traceback
        traceback.print_exc()
        return ("__ERROR__:"+os.path.basename(fpath), ex, None, 0, 0, 0, {})