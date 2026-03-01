#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
import numpy as np
import math
from dataclasses import dataclass, replace
from typing import Tuple, Dict, List, Optional
from numpy.lib.format import open_memmap
from scipy.interpolate import PchipInterpolator
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def convert_float_to_uint64(data: np.ndarray, scale_factor: float, rng_seed: int) -> np.ndarray:
    norm = np.clip(data * scale_factor, 0.0, 1.0)

    MAX_53_BITS = (1 << 53) - 1
    high_part = (norm * MAX_53_BITS).astype(np.uint64)

    high_part <<= np.uint64(11)

    local_rng = np.random.default_rng(rng_seed + 999)
    low_noise = local_rng.integers(0, 1 << 11, size=data.shape, dtype=np.uint64)

    return high_part | low_noise


def even_split_counts(total: int, k: int) -> List[int]:
    base = total // k
    rem = total % k
    return [base + (1 if i < rem else 0) for i in range(k)]


def rounded_mse_from_samples(x: np.ndarray, A: float, B: float, use_midpoint: bool = False) -> float:
    xs = np.sort(np.asarray(x, dtype=np.float64))
    n = xs.size
    if n == 0:
        return 0.0
    if use_midpoint:
        y = (np.arange(1, n + 1, dtype=np.float64) - 0.5) / n
    else:
        y = (np.arange(1, n + 1, dtype=np.float64)) / n
    xm = xs.mean()
    ym = y.mean()
    varx = np.mean((xs - xm) ** 2) + 1e-18
    covxy = np.mean((xs - xm) * (y - ym))
    a = covxy / varx
    b = ym - a * xm
    yhat = a * xs + b
    yhat = np.clip(yhat, 0.0, 1.0)
    yhat_round = yhat
    mse = float(np.mean((yhat_round - y) ** 2))
    return mse


@dataclass
class SmallBasisCfg:
    K: int = 16
    grid: int = 8192
    ell: float = 0.03
    seed: int = 7

def build_z_smallgrid(cfg: SmallBasisCfg) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    t = (np.arange(cfg.grid, dtype=np.float64) + 0.5) / cfg.grid
    freqs = rng.normal(0.0, 1.0 / (cfg.ell + 1e-12), size=cfg.K).astype(np.float64)
    phases = rng.uniform(0.0, 2 * np.pi, size=cfg.K).astype(np.float64)
    amps = (rng.normal(0.0, 1.0, size=cfg.K) / np.sqrt(cfg.K)).astype(np.float64)
    Z = np.zeros_like(t, dtype=np.float64)
    two_pi_t = 2.0 * np.pi * t
    Z += (amps[:, None] * np.cos(freqs[:, None] * two_pi_t[None, :] + phases[:, None])).sum(axis=0, dtype=np.float64)
    Z = (Z - Z.mean()) / (Z.std() + 1e-12)
    return t, Z


def build_T_from_Z_alpha(t: np.ndarray, Z: np.ndarray, alpha: float) -> np.ndarray:
    z_shift = Z - Z.max()
    f = np.exp(alpha * z_shift, dtype=np.float64)
    F = np.cumsum(f) / len(t)
    T = F / (F[-1] + 1e-300)
    T[0] = 0.0
    T[-1] = 1.0
    return T


def build_inv_cdf_pchip(t: np.ndarray, T: np.ndarray) -> PchipInterpolator:
    Tm = np.maximum.accumulate(T)
    idx = np.r_[0, 1 + np.where(np.diff(Tm) > 1e-15)[0], len(Tm) - 1]
    U = Tm[idx].astype(np.float64, copy=False)
    X = t[idx].astype(np.float64, copy=False)
    U[0], X[0] = 0.0, 0.0
    U[-1], X[-1] = 1.0, 1.0
    if not np.all(np.diff(U) > 0):
        jitter = 1e-15 * np.arange(U.size, dtype=np.float64)
        U = U + jitter
        U[0] = 0.0
        U[-1] = 1.0
        if not np.all(np.diff(U) > 0):
            keep = np.r_[True, np.diff(U) > 0]
            U, X = U[keep], X[keep]
    return PchipInterpolator(U, X, extrapolate=True)


def quantile_samples(n: int, inv: PchipInterpolator, A: float, B: float) -> np.ndarray:
    u = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    x_unit = inv(u)
    np.clip(x_unit, 0.0, 1.0, out=x_unit)
    return A + (B - A) * x_unit


@dataclass
class AlphaCalibCfg:
    tau: float
    tol: float = 8e-7
    alpha_hi: float = 18.0
    iters: int = 40
    max_expand: int = 8
    bins_eval: Optional[int] = None


def calibrate_alpha_by_samples(t: np.ndarray, Z: np.ndarray, n_eval: int,
                               A: float, B: float, cfg: AlphaCalibCfg
                               ) -> Tuple[float, PchipInterpolator, float]:
    def eval_alpha(alpha: float) -> Tuple[float, PchipInterpolator, float]:
        T = build_T_from_Z_alpha(t, Z, alpha)
        inv = build_inv_cdf_pchip(t, T)
        x = quantile_samples(n_eval, inv, A, B)
        w2 = rounded_mse_from_samples(x, A=A, B=B, use_midpoint=False)
        return w2, inv, alpha

    lo, hi = 0.0, float(cfg.alpha_hi)
    w_lo, inv_lo, _ = eval_alpha(lo)
    w_hi, inv_hi, _ = eval_alpha(hi)

    expand = 0
    while w_hi < cfg.tau and expand < cfg.max_expand and hi < 1e3:
        lo, w_lo, inv_lo = hi, w_hi, inv_hi
        hi *= 2.0
        w_hi, inv_hi, _ = eval_alpha(hi)
        expand += 1

    best = (abs(w_lo - cfg.tau), lo, w_lo, inv_lo)
    cand = (abs(w_hi - cfg.tau), hi, w_hi, inv_hi)
    if cand[0] < best[0]:
        best = cand

    for _ in range(cfg.iters):
        mid = 0.5 * (lo + hi)
        w_mid, inv_mid, _ = eval_alpha(mid)
        err = abs(w_mid - cfg.tau)
        if err < best[0]:
            best = (err, mid, w_mid, inv_mid)
        if err <= cfg.tol:
            lo = hi = mid
            break
        if w_mid < cfg.tau:
            lo = mid
        else:
            hi = mid

    _, alpha_star, w2_star, inv_star = best
    return float(alpha_star), inv_star, float(w2_star)


@dataclass
class LayerParams:
    tau: float
    ell: float
    K: int
    grid: int
    seed: int
    alpha_hi: float = 18.0
    tol: float = 8e-7
    bins_eval: Optional[int] = None


@dataclass
class HierSpec:
    top: LayerParams
    second: LayerParams
    m_leaf: int


def rng_from_base_and_index(base_seed: int, j: int) -> np.random.Generator:
    ss = np.random.SeedSequence([int(base_seed), int(j)])
    return np.random.default_rng(ss)


def _recursive_execution(interval_quota: int,
                         next_level_quotas: List[int],
                         A: float, B: float,
                         spec_top: LayerParams,
                         seed_seq: np.random.SeedSequence) -> List[Tuple[float, float, int]]:
    K_splits = interval_quota
    if K_splits <= 0:
        return []

    my_rng = np.random.default_rng(seed_seq)
    node_seed = int(my_rng.integers(1, 1 << 31))

    t, Z = build_z_smallgrid(SmallBasisCfg(K=spec_top.K, grid=spec_top.grid,
                                           ell=spec_top.ell, seed=node_seed))

    n_eval_needed = max(K_splits * 4, 2048)
    if spec_top.bins_eval:
        n_eval_needed = max(n_eval_needed, spec_top.bins_eval)

    alpha, inv, _ = calibrate_alpha_by_samples(
        t, Z, n_eval=n_eval_needed, A=A, B=B,
        cfg=AlphaCalibCfg(tau=spec_top.tau, tol=spec_top.tol,
                          alpha_hi=spec_top.alpha_hi, bins_eval=spec_top.bins_eval)
    )

    u_boundaries = (np.arange(1, K_splits, dtype=np.float64)) / float(K_splits)
    x_units = inv(u_boundaries)
    np.clip(x_units, 0.0, 1.0, out=x_units)
    internal_cuts = A + (B - A) * x_units
    all_cuts = np.concatenate(([A], internal_cuts, [B]))

    results = []
    if len(next_level_quotas) > 0:
        global_next_total = next_level_quotas[0]
        remaining_levels = next_level_quotas[1:]
        sub_quotas = even_split_counts(global_next_total, K_splits)
        children_seeds = seed_seq.spawn(K_splits)
        for i in range(K_splits):
            if sub_quotas[i] > 0:
                sub_res = _recursive_execution(
                    interval_quota=sub_quotas[i],
                    next_level_quotas=remaining_levels,
                    A=all_cuts[i],
                    B=all_cuts[i + 1],
                    spec_top=spec_top,
                    seed_seq=children_seeds[i]
                )
                results.extend(sub_res)
    else:
        for i in range(K_splits):
            results.append((all_cuts[i], all_cuts[i + 1], 0))
    return results


def generate_hierarchical_buckets(N_total: int, A: float, B: float,
                                  spec_top: LayerParams, m_leaf: int,
                                  wsize: int,
                                  top_seed: int) -> List[Tuple[float, float, int]]:

    target_leaf_count = N_total // max(1, m_leaf)
    if target_leaf_count < 1:
        target_leaf_count = 1
    seed_seq = np.random.SeedSequence(top_seed)
    if target_leaf_count < 300:
        return _recursive_execution(
            interval_quota=target_leaf_count,
            next_level_quotas=[],
            A=A, B=B, spec_top=spec_top, seed_seq=seed_seq
        )
    branching_factor = max(2, wsize)
    plan_stack = [target_leaf_count]

    while True:
        c = plan_stack[-1]
        if c <= branching_factor:
            break
        parent_count = c // branching_factor
        if parent_count < 2:
            break

        plan_stack.append(parent_count)

    plan_levels = plan_stack[::-1]

    return _recursive_execution(
        interval_quota=plan_levels[0],
        next_level_quotas=plan_levels[1:] if len(plan_levels) > 1 else [],
        A=A, B=B, spec_top=spec_top, seed_seq=seed_seq
    )


def generate_two_layer_exact(A: float, B: float, N_total: int,
                             spec: HierSpec, rng_seed_for_second: int) -> np.ndarray:
    if N_total <= 0:
        return np.empty(0, dtype=np.float64)
    target_leaf_count = N_total // max(1, spec.m_leaf)
    if target_leaf_count < 1:
        target_leaf_count = 1
    val1 = 8.0 * math.sqrt(target_leaf_count)
    val2 = target_leaf_count / 16.0
    w_size = int(min(val1, val2))
    leaf_buckets_info = generate_hierarchical_buckets(
        N_total=N_total, A=A, B=B, spec_top=spec.top,
        m_leaf=spec.m_leaf, wsize=w_size, top_seed=spec.top.seed
    )
    num_buckets = len(leaf_buckets_info)
    if num_buckets == 0:
        return np.empty(0, dtype=np.float64)
    real_counts = even_split_counts(N_total, num_buckets)
    leaf_buckets = []
    for i, (L, R, _) in enumerate(leaf_buckets_info):
        leaf_buckets.append((L, R, real_counts[i]))

    out = np.empty(N_total, dtype=np.float64)
    pos = 0
    for j, (Lj, Rj, m_j) in enumerate(leaf_buckets):
        if m_j <= 0:
            continue
        if not (Lj < Rj):
            x_sub = np.full(m_j, Lj, dtype=np.float64)
        else:
            rng_j = rng_from_base_and_index(rng_seed_for_second, j)
            seed_j = int(rng_j.integers(1, 1 << 31))
            t2, Z2 = build_z_smallgrid(SmallBasisCfg(K=spec.second.K, grid=spec.second.grid,
                                                     ell=spec.second.ell, seed=seed_j))
            alpha2, inv2, _ = calibrate_alpha_by_samples(
                t2, Z2, n_eval=m_j, A=0.0, B=1.0,
                cfg=AlphaCalibCfg(tau=spec.second.tau, tol=spec.second.tol,
                                  alpha_hi=spec.second.alpha_hi, bins_eval=spec.second.bins_eval)
            )
            u = (np.arange(m_j, dtype=np.float64) + 0.5) / float(m_j)
            x_unit = inv2(u)
            np.clip(x_unit, 0.0, 1.0, out=x_unit)
            x_sub = Lj + (Rj - Lj) * x_unit
        out[pos:pos + m_j] = x_sub
        pos += m_j
    return out


@dataclass
class SchedulePhase:
    kind: str
    count: int
    tau_start: float
    tau_end: float


class DriftScheduler:
    def __init__(self, spec_template: HierSpec, phases: List[SchedulePhase]):
        self.spec = spec_template
        self.phases = phases
        self.phase_idx = 0
        self.generated_in_phase = 0
        self.finished = False

    def generate_batch(self, n_req: int, rng_seed_base: int,
                       dist_idx: int, A: float, B: float) -> np.ndarray:
        out_buffer = []
        remaining = n_req
        current_rng_seed = rng_seed_base
        while remaining > 0:
            if self.finished:
                last_tau = self.phases[-1].tau_end
                current_phase = SchedulePhase('stable', int(1e18), last_tau, last_tau)
            else:
                current_phase = self.phases[self.phase_idx]
            phase_left = current_phase.count - self.generated_in_phase
            take = min(remaining, phase_left)
            if take > 0:
                spec_now = self.spec
                if current_phase.kind == 'stable':
                    spec_now.top.tau = current_phase.tau_start
                    spec_now.second.tau = current_phase.tau_start
                    data = generate_two_layer_exact(
                        A=A, B=B, N_total=take, spec=spec_now,
                        rng_seed_for_second=current_rng_seed
                    )
                else:
                    spec_a = replace(spec_now)
                    spec_a.top = replace(spec_now.top)
                    spec_a.second = replace(spec_now.second)
                    spec_b = replace(spec_now)
                    spec_b.top = replace(spec_now.top)
                    spec_b.second = replace(spec_now.second)
                    spec_a.top.tau = current_phase.tau_start
                    spec_a.second.tau = current_phase.tau_start
                    spec_b.top.tau = current_phase.tau_end
                    spec_b.second.tau = current_phase.tau_end
                    data_a = generate_two_layer_exact(
                        A=A, B=B, N_total=take, spec=spec_a,
                        rng_seed_for_second=current_rng_seed
                    )
                    data_b = generate_two_layer_exact(
                        A=A, B=B, N_total=take, spec=spec_b,
                        rng_seed_for_second=current_rng_seed
                    )
                    start_j = self.generated_in_phase
                    end_j = start_j + take
                    p = np.arange(start_j, end_j, dtype=np.float64) / float(current_phase.count)
                    rng_mix = np.random.default_rng(current_rng_seed + 99999)
                    mask = rng_mix.random(take) < p
                    data = np.where(mask, data_b, data_a)
                out_buffer.append(data)
                remaining -= take
                self.generated_in_phase += take
                current_rng_seed += 137
            if not self.finished and self.generated_in_phase >= current_phase.count:
                self.phase_idx += 1
                self.generated_in_phase = 0
                if self.phase_idx >= len(self.phases):
                    self.finished = True
        if len(out_buffer) == 1:
            return out_buffer[0]
        return np.concatenate(out_buffer)


@dataclass
class DistState:
    top: LayerParams
    second: LayerParams
    m_leaf: int


def load_mixture(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["configs"]


def load_time_axis(path: str) -> List[Tuple[int, int, int]]:
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    out = []
    for row in arr:
        if not (isinstance(row, (list, tuple)) and len(row) == 3):
            raise ValueError("time_axis.json must be list of triples: [stable_len, transit_len, avg_window_size]")
        out.append((int(row[0]), int(row[1]), int(row[2])))
    return out


def load_tau_axis(path: str) -> List[List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    if not isinstance(arr, list):
        raise ValueError("tau_axis.json must be a list of rows.")
    out: List[List[float]] = []
    for r, row in enumerate(arr):
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"tau_axis.json row {r} must be a list.")
        out.append([float(x) for x in row])
    return out


def _safe_tau_str(x: float) -> str:
    s = f"{x:.6g}"
    s = s.replace('.', 'p').replace('-', 'm')
    return s


def _fmt_count_tag(n: int) -> str:
    n = int(n)
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n % 1_000 == 0:
        return f"{n // 1_000}K"
    return str(n)

def _norm_bins_eval(x):
    return None if x is None else int(x)


def _pooled_cfg_key(comp_idx: int, top_lp: LayerParams, second_lp: LayerParams) -> Tuple:
    return (
        int(comp_idx),
        float(top_lp.tau), float(top_lp.ell), int(top_lp.K), int(top_lp.grid), int(top_lp.seed),
        float(top_lp.alpha_hi), float(top_lp.tol), _norm_bins_eval(top_lp.bins_eval),
        float(second_lp.tau), float(second_lp.ell), int(second_lp.K), int(second_lp.grid), int(second_lp.seed),
        float(second_lp.alpha_hi), float(second_lp.tol), _norm_bins_eval(second_lp.bins_eval),
    )


def _pooled_cfg_desc(comp_idx: int, top_lp: LayerParams, second_lp: LayerParams) -> str:
    return (
        f"comp={comp_idx}, "
        f"top(tau={top_lp.tau}, ell={top_lp.ell}, K={top_lp.K}, grid={top_lp.grid}, seed={top_lp.seed}), "
        f"second(tau={second_lp.tau}, ell={second_lp.ell}, K={second_lp.K}, grid={second_lp.grid}, seed={second_lp.seed})"
    )


def parse_args():
    p = argparse.ArgumentParser(description="Two-layer generator with Drifting Schedule.")
    p.add_argument("--mixture", type=str, required=True, help="mixture.json")
    p.add_argument("--time_axis", type=str, default=None, help="time_axis.json: [[s1, n1], ...]")
    p.add_argument("--tau_axis", type=str, default=None,
                   help="tau_axis.json: rows of per-component taus for each stable block (applies to BOTH top & second). "
                        "Length can be num_blocks or num_blocks+1.")
    p.add_argument("--tau_axis_top", type=str, default=None,
                   help="tau_axis_top.json: rows of per-component taus for each stable block, for TOP layer. "
                        "Length can be num_blocks or num_blocks+1.")
    p.add_argument("--tau_axis_second", type=str, default=None,
                   help="tau_axis_second.json: rows of per-component taus for each stable block, for SECOND layer. "
                        "Length can be num_blocks or num_blocks+1.")

    p.add_argument("--outfile", type=str, required=True, help="Output file/dir")

    p.add_argument("--dtype", type=str, default="uint64", choices=["float32", "float64", "uint64"])
    p.add_argument("--shuffle_seed", type=int, default=2025, help="Global shuffle seed")
    p.add_argument("--seed_min", type=int, default=1)
    p.add_argument("--seed_max", type=int, default=10_000_000)
    p.add_argument("--mode", type=str, default="Com", choices=["Com", "Div"])
    p.add_argument("--total_num", type=int, default=None, help="For Div mode (final saved N2)")
    p.add_argument("--gen_num", type=int, default=None,
                   help="For Div mode: pre-generate N1 samples, then randomly sample N2 (=total_num) to save. "
                        "If omitted, N1=N2.")
    p.add_argument("--total", type=int, default=None, help="Optional total override")
    p.add_argument("--workers", type=int, default=None, help="Number of workers for Div mode")
    p.add_argument("--jitter_duplicates", action="store_true",
                   help="If set, minimally push duplicates upward by 1 ULP (or +1 for uint64) while preserving order; prints pushed count.")
    p.add_argument("--avg_w_global", type=int, default=None,
                   help="Com mode required. Global avg_window_size used to compute m_leaf for pooled full datasets.")

    return p.parse_args()


def build_schedule_phases(initial_tau: float, time_axis_data: List[Tuple[int, int]],
                          rng: np.random.Generator, k: int) -> List[SchedulePhase]:
    TARGET_TAUS = [0.01, 0.001, 0.0001]
    phases = []
    current_tau = initial_tau
    for s_count_global, n_count_global in time_axis_data:
        s_count_local = int(s_count_global / k)
        n_count_local = int(n_count_global / k)
        phases.append(SchedulePhase('stable', s_count_local, current_tau, current_tau))
        if n_count_local > 0:
            next_tau = rng.choice(TARGET_TAUS)
            phases.append(SchedulePhase('transition', n_count_local, current_tau, next_tau))
            current_tau = next_tau
    return phases


def _alloc_scaled_ints(values: List[int], target_sum: int) -> List[int]:
    total = int(sum(values))
    if total <= 0:
        return [0] * len(values)

    r = float(target_sum) / float(total)
    raw = [v * r for v in values]
    base = [int(math.floor(x)) for x in raw]
    frac = [x - b for x, b in zip(raw, base)]
    cur = sum(base)
    diff = target_sum - cur

    if diff > 0:
        order = sorted(range(len(values)), key=lambda i: frac[i], reverse=True)
        idx = 0
        while diff > 0 and idx < len(order):
            base[order[idx]] += 1
            diff -= 1
            idx += 1
        idx = 0
        while diff > 0:
            base[order[idx % len(order)]] += 1
            diff -= 1
            idx += 1
    elif diff < 0:
        diff = -diff
        order = sorted(range(len(values)), key=lambda i: frac[i])
        idx = 0
        while diff > 0 and idx < len(order):
            i = order[idx]
            if base[i] > 0:
                base[i] -= 1
                diff -= 1
            idx += 1
        idx = 0
        while diff > 0:
            i = idx % len(base)
            if base[i] > 0:
                base[i] -= 1
                diff -= 1
            idx += 1

    return base


def _alloc_scaled_stable_trans(stables: List[int], trans: List[int], target_sum: int) -> Tuple[List[int], List[int]]:
    assert len(stables) == len(trans)
    total = int(sum(stables) + sum(trans))
    if total <= 0:
        return [0] * len(stables), [0] * len(trans)

    r = float(target_sum) / float(total)

    items = []
    s_base, n_base = [], []
    for i in range(len(stables)):
        rs = stables[i] * r
        rn = trans[i] * r
        bs = int(math.floor(rs)); fs = rs - bs
        bn = int(math.floor(rn)); fn = rn - bn
        s_base.append(bs)
        n_base.append(bn)
        items.append((fs, 's', i))
        items.append((fn, 'n', i))

    cur = sum(s_base) + sum(n_base)
    diff = target_sum - cur

    if diff > 0:
        items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
        t = 0
        while diff > 0:
            frac, kind, i = items_sorted[t % len(items_sorted)]
            if kind == 's':
                s_base[i] += 1
            else:
                n_base[i] += 1
            diff -= 1
            t += 1
    elif diff < 0:
        diff = -diff
        items_sorted = sorted(items, key=lambda x: x[0])
        t = 0
        while diff > 0 and t < 10 * len(items_sorted):
            frac, kind, i = items_sorted[t % len(items_sorted)]
            if kind == 's' and s_base[i] > 0:
                s_base[i] -= 1
                diff -= 1
            elif kind == 'n' and n_base[i] > 0:
                n_base[i] -= 1
                diff -= 1
            t += 1
        t = 0
        while diff > 0:
            i = t % len(s_base)
            if s_base[i] > 0:
                s_base[i] -= 1
                diff -= 1
            elif n_base[i] > 0:
                n_base[i] -= 1
                diff -= 1
            t += 1

    return s_base, n_base


def _mix_transition_from_halves(prev_half: np.ndarray, next_half: np.ndarray, T: int, seed: int) -> np.ndarray:
    if T <= 0:
        return np.empty(0, dtype=np.float64)

    rng = np.random.default_rng(seed)

    G1 = np.asarray(prev_half, dtype=np.float64, order="C")
    G2 = np.asarray(next_half, dtype=np.float64, order="C")

    n1 = int(G1.size)
    n2 = int(G2.size)

    if (n1 + n2) != int(T):
        raise ValueError(f"Transition pool sizes mismatch: |G1|+|G2|={n1+n2} != T={T}")

    if n1 == 0 and n2 == 0:
        return np.empty(T, dtype=np.float64)

    scores = np.empty(n1 + n2, dtype=np.float64)
    values = np.empty(n1 + n2, dtype=np.float64)

    if n1 > 0:
        u1 = rng.random(n1)
        v1 = rng.random(n1)
        s1 = np.minimum(u1, v1)
        scores[:n1] = s1
        values[:n1] = G1

    if n2 > 0:
        u2 = rng.random(n2)
        v2 = rng.random(n2)
        s2 = np.maximum(u2, v2)
        scores[n1:] = s2
        values[n1:] = G2

    scores = scores + (rng.random(scores.size) - 0.5) * 1e-15

    order = np.argsort(scores, kind="mergesort")
    out = values[order]
    return out


def _transition_scores_values(prev_half: np.ndarray, next_half: np.ndarray, T: int, seed: int
                             ) -> Tuple[np.ndarray, np.ndarray]:
    if T <= 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    rng = np.random.default_rng(seed)

    G1 = np.asarray(prev_half, dtype=np.float64, order="C")
    G2 = np.asarray(next_half, dtype=np.float64, order="C")

    n1 = int(G1.size)
    n2 = int(G2.size)

    if (n1 + n2) != int(T):
        raise ValueError(f"Transition pool sizes mismatch: |G1|+|G2|={n1+n2} != T={T}")

    if n1 == 0 and n2 == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    scores = np.empty(n1 + n2, dtype=np.float64)
    values = np.empty(n1 + n2, dtype=np.float64)

    if n1 > 0:
        u1 = rng.random(n1)
        v1 = rng.random(n1)
        scores[:n1] = np.minimum(u1, v1)
        values[:n1] = G1

    if n2 > 0:
        u2 = rng.random(n2)
        v2 = rng.random(n2)
        scores[n1:] = np.maximum(u2, v2)
        values[n1:] = G2

    scores = scores + (rng.random(scores.size) - 0.5) * 1e-15
    return scores, values


def jitter_duplicates_inplace(arr: np.ndarray) -> int:
    pushed = 0

    if arr.dtype == np.uint64:
        seen: set[int] = set()
        UINT64_MAX = (1 << 64) - 1
        for i in range(arr.size):
            v = int(arr[i])
            if v not in seen:
                seen.add(v)
                continue
            newv = v
            while newv < UINT64_MAX:
                newv += 1
                if newv not in seen:
                    arr[i] = np.uint64(newv)
                    seen.add(newv)
                    pushed += 1
                    break
        return pushed

    if arr.dtype == np.float64:
        bits_view = arr.view(np.uint64)
        seen: set[int] = set()
        for i in range(arr.size):
            b = int(bits_view[i])
            if b not in seen:
                seen.add(b)
                continue
            x = float(arr[i])
            while True:
                x = float(np.nextafter(x, np.inf))
                b2 = int(np.array(x, dtype=np.float64).view(np.uint64)[0])
                if b2 not in seen:
                    arr[i] = np.float64(x)
                    seen.add(b2)
                    pushed += 1
                    break
                if np.isinf(x):
                    break
        return pushed

    if arr.dtype == np.float32:
        bits_view = arr.view(np.uint32)
        seen: set[int] = set()
        for i in range(arr.size):
            b = int(bits_view[i])
            if b not in seen:
                seen.add(b)
                continue
            x = np.float32(arr[i])
            while True:
                x = np.nextafter(x, np.float32(np.inf)).astype(np.float32)
                b2 = int(np.array(x, dtype=np.float32).view(np.uint32)[0])
                if b2 not in seen:
                    arr[i] = x
                    seen.add(b2)
                    pushed += 1
                    break
                if np.isinf(x):
                    break
        return pushed

    raise TypeError(f"Unsupported dtype for jitter_duplicates: {arr.dtype}")


def run_div_task(args_tuple):
    j, spec, gen_num, outdir, dtype_str, seed_val, jitter_flag = args_tuple

    samples_full = generate_two_layer_exact(
        A=0.0, B=1.0,
        N_total=int(gen_num),
        spec=spec,
        rng_seed_for_second=seed_val
    )

    if dtype_str == "uint64":
        samples_full = convert_float_to_uint64(samples_full, scale_factor=1.0, rng_seed=seed_val)
        samples_full = samples_full.astype(np.uint64, copy=False)
    else:
        out_dtype = np.float32 if dtype_str == "float32" else np.float64
        samples_full = samples_full.astype(out_dtype, copy=False)

    pushed = 0
    if jitter_flag:
        pushed = jitter_duplicates_inplace(samples_full)

    top_tau_str = _safe_tau_str(spec.top.tau)
    second_tau_str = _safe_tau_str(spec.second.tau)
    tag_full = _fmt_count_tag(gen_num)

    fname_full = f"data_{top_tau_str}_{second_tau_str}_{tag_full}.npy"
    fpath_full = os.path.join(outdir, fname_full)
    np.save(fpath_full, samples_full)

    return (f"[Process {os.getpid()}] Saved dist #{j} -> {fname_full} "
            f"(n={samples_full.size}, m_leaf={spec.m_leaf}, jitter_pushed={pushed})")


def main():
    args = parse_args()
    cfgs = load_mixture(args.mixture)
    k = len(cfgs)

    scheduler_templates = []
    for c in cfgs:
        top = LayerParams(
            tau=float(c["top"]["tau"]),
            ell=float(c["top"]["ell"]),
            K=int(c["top"].get("K", 16)),
            grid=int(c["top"].get("grid", 8192)),
            seed=int(c["top"].get("seed", 7)),
            alpha_hi=float(c["top"].get("alpha_hi", 18.0)),
            tol=float(c["top"].get("tol", 8e-7)),
            bins_eval=c["top"].get("bins_eval", None),
        )
        second = LayerParams(
            tau=float(c["second"]["tau"]),
            ell=float(c["second"]["ell"]),
            K=int(c["second"].get("K", 16)),
            grid=int(c["second"].get("grid", 8192)),
            seed=int(c["second"].get("seed", 77)),
            alpha_hi=float(c["second"].get("alpha_hi", 18.0)),
            tol=float(c["second"].get("tol", 8e-7)),
            bins_eval=c["second"].get("bins_eval", None),
        )
        m_leaf = int(c.get("m_leaf", 3000))
        scheduler_templates.append(HierSpec(top, second, m_leaf))

    rng_step = np.random.default_rng(args.shuffle_seed)

    if args.mode == "Div":
        if args.total_num is None or args.total_num <= 0:
            raise ValueError("Div mode requires --total_num (>0)")

        gen_num = args.gen_num if args.gen_num is not None else args.total_num

        outdir = os.path.abspath(args.outfile)
        os.makedirs(outdir, exist_ok=True)

        if gen_num >= args.total_num:
            scale = float(gen_num) / float(args.total_num)
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid scale gen_num/total_num = {scale}")

            scaled_templates = []
            for spec in scheduler_templates:
                new_m_leaf = max(1, int(round(spec.m_leaf * scale)))
                scaled_templates.append(HierSpec(spec.top, spec.second, new_m_leaf))
            scheduler_templates = scaled_templates

            print(f"Div mode: gen_num >= total_num, scale m_leaf by {gen_num}/{args.total_num}={scale:.6g}, "
                  f"then generate N={gen_num} (total_num ignored after scaling).")
        else:
            print(f"Div mode: gen_num < total_num, keep m_leaf unchanged, generate N={gen_num} "
                  f"(total_num ignored).")

        task_seeds = [int(rng_step.integers(1, 1 << 31)) for _ in scheduler_templates]

        tasks = []
        for j, spec in enumerate(scheduler_templates):
            tasks.append((j, spec, int(gen_num), outdir, args.dtype, task_seeds[j], bool(args.jitter_duplicates)))
        num_workers = args.workers if args.workers else min(len(tasks), multiprocessing.cpu_count())
        print(f"Starting parallel generation for {len(tasks)} distributions using {num_workers} workers...")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(run_div_task, tasks)
            for res_msg in results:
                print(res_msg)

        print("All separate files saved to:", outdir)
        return

    if not args.time_axis:
        raise ValueError("Com mode requires --time_axis")
    if args.avg_w_global is None or int(args.avg_w_global) <= 0:
        raise ValueError("Com mode requires --avg_w_global (>0)")
    avg_w_global = int(args.avg_w_global)

    time_axis_data = load_time_axis(args.time_axis)
    total_N = sum(s + n for (s, n, w) in time_axis_data)
    if args.total:
        print(f"Warning: --total {args.total} is ignored. Using sum of time_axis: {total_N}")

    print(f"Start Com Generation: Total={total_N}, Output Type={args.dtype}, avg_w_global={avg_w_global}")

    if args.dtype == "uint64":
        out_dtype = np.uint64
    else:
        out_dtype = np.float32 if args.dtype == "float32" else np.float64

    outfile = os.path.abspath(args.outfile)
    mm = open_memmap(outfile, mode='w+', dtype=out_dtype, shape=(total_N,))

    TARGET_TAUS = [0.01, 0.001, 0.0001]

    num_blocks = len(time_axis_data)
    stable_counts_by_block = []
    trans_counts_by_block = []
    avgw_by_block = []
    for (s, n, w) in time_axis_data:
        stable_counts_by_block.append(even_split_counts(int(s), k))
        trans_counts_by_block.append(even_split_counts(int(n), k))
        avgw_by_block.append(max(1, int(w)))

    comp_total_counts = [0] * k
    for j in range(k):
        total_j = 0
        for bi in range(num_blocks):
            total_j += int(stable_counts_by_block[bi][j])
            total_j += int(trans_counts_by_block[bi][j])
        comp_total_counts[j] = total_j

    tau_top: Optional[List[List[float]]] = None
    tau_second: Optional[List[List[float]]] = None

    if args.tau_axis:
        base = load_tau_axis(args.tau_axis)
        tau_top = base
        tau_second = base

    if args.tau_axis_top:
        tau_top = load_tau_axis(args.tau_axis_top)
    if args.tau_axis_second:
        tau_second = load_tau_axis(args.tau_axis_second)

    def _check_tau_matrix(name: str, mat: Optional[List[List[float]]]):
        if mat is None:
            return
        if len(mat) not in (num_blocks, num_blocks + 1):
            raise ValueError(
                f"{name} must have length num_blocks ({num_blocks}) or num_blocks+1 ({num_blocks+1}), got {len(mat)}."
            )
        for i, row in enumerate(mat):
            if len(row) != k:
                raise ValueError(f"{name} row {i} must have length k={k}, got {len(row)}.")

        last_trans_global = int(time_axis_data[-1][1])
        if last_trans_global == 0 and len(mat) == (num_blocks + 1):
            print(f"[Warning][{name}] last block transition_len==0, but {name} has an extra final row "
                  f"(num_blocks+1). The final tau row will be unused.")

    _check_tau_matrix("tau_axis_top", tau_top)
    _check_tau_matrix("tau_axis_second", tau_second)

    all_state_head: List[List[np.ndarray]] = []
    all_state_stable: List[List[np.ndarray]] = []
    all_state_tail: List[List[np.ndarray]] = []
    all_trans_len_list: List[List[int]] = []

    state_requests: List[List[Dict]] = []

    pooled_demands: Dict[Tuple, Dict] = {}

    for j, tmpl in enumerate(scheduler_templates):
        rng_tau = np.random.default_rng(np.random.SeedSequence([args.shuffle_seed, j, 314159]))

        use_tau_axis_top = (tau_top is not None)
        use_tau_axis_second = (tau_second is not None)

        tau_list_top: List[float] = []
        tau_list_second: List[float] = []

        stable_len_list: List[int] = []
        avgw_list: List[int] = []
        trans_len_list: List[int] = []

        current_tau_top = float(tmpl.top.tau)        # fallback only
        current_tau_second = float(tmpl.second.tau)  # fallback only

        for bi in range(num_blocks):
            s_len = int(stable_counts_by_block[bi][j])
            t_len = int(trans_counts_by_block[bi][j])
            w_len = int(avgw_by_block[bi])

            if use_tau_axis_top:
                tau_here_top = float(tau_top[bi][j])
            else:
                tau_here_top = current_tau_top

            if use_tau_axis_second:
                tau_here_second = float(tau_second[bi][j])
            else:
                tau_here_second = current_tau_second

            tau_list_top.append(tau_here_top)
            tau_list_second.append(tau_here_second)

            stable_len_list.append(max(0, s_len))
            avgw_list.append(max(1, w_len))

            if t_len > 0:
                trans_len_list.append(max(0, t_len))
            else:
                trans_len_list.append(0)

            # fallback drift when not using tau_axis for that layer
            if t_len > 0:
                if not use_tau_axis_top:
                    current_tau_top = float(rng_tau.choice(TARGET_TAUS))
                if not use_tau_axis_second:
                    current_tau_second = float(rng_tau.choice(TARGET_TAUS))

        # warning: tau jump when transition_len==0 -> warn and skip (no error)
        for bi in range(num_blocks - 1):
            if int(trans_counts_by_block[bi][j]) != 0:
                continue

            if use_tau_axis_top:
                t0 = float(tau_top[bi][j])
                t1 = float(tau_top[bi + 1][j])
                if abs(t1 - t0) > 1e-15:
                    print(f"[Warning][tau_axis_top] comp={j}, block={bi}: transition_len=0 but tau jumps "
                          f"{t0} -> {t1}. Will skip transition and start next stable with new tau.")

            if use_tau_axis_second:
                t0 = float(tau_second[bi][j])
                t1 = float(tau_second[bi + 1][j])
                if abs(t1 - t0) > 1e-15:
                    print(f"[Warning][tau_axis_second] comp={j}, block={bi}: transition_len=0 but tau jumps "
                          f"{t0} -> {t1}. Will skip transition and start next stable with new tau.")

        # final state only needed if last block has transition > 0
        if trans_len_list and trans_len_list[-1] > 0:
            # TOP final
            if use_tau_axis_top:
                if len(tau_top) >= num_blocks + 1:
                    tau_final_top = float(tau_top[num_blocks][j])
                else:
                    tau_final_top = float(tau_list_top[-1])
                    print(f"[Warning][tau_axis_top] last block has transition>0 but tau_axis_top has no final row. "
                          f"Use tau_final_top=tau_last={tau_final_top} for comp={j}.")
            else:
                tau_final_top = float(current_tau_top)

            # SECOND final
            if use_tau_axis_second:
                if len(tau_second) >= num_blocks + 1:
                    tau_final_second = float(tau_second[num_blocks][j])
                else:
                    tau_final_second = float(tau_list_second[-1])
                    print(f"[Warning][tau_axis_second] last block has transition>0 but tau_axis_second has no final row. "
                          f"Use tau_final_second=tau_last={tau_final_second} for comp={j}.")
            else:
                tau_final_second = float(current_tau_second)

            tau_list_top.append(tau_final_top)
            tau_list_second.append(tau_final_second)

            stable_len_list.append(0)
            avgw_list.append(avgw_list[-1])

        num_states = len(tau_list_top)
        if len(tau_list_second) != num_states:
            raise RuntimeError("Internal error: tau_list_top and tau_list_second length mismatch.")

        # transitions between states
        trans_between = []
        for tlen in trans_len_list:
            if len(trans_between) >= num_states - 1:
                break
            trans_between.append(int(tlen))
        while len(trans_between) < num_states - 1:
            trans_between.append(0)

        # init placeholders
        state_head: List[np.ndarray] = []
        state_stable: List[np.ndarray] = []
        state_tail: List[np.ndarray] = []
        reqs_j: List[Dict] = []

        for si in range(num_states):
            prev_T = trans_between[si - 1] if si - 1 >= 0 else 0
            next_T = trans_between[si] if si < (num_states - 1) else 0

            head_len = int(prev_T - (prev_T // 2))  # ceil(prev_T/2)
            tail_len = int(next_T // 2)             # floor(next_T/2)
            stable_len = int(stable_len_list[si])

            N_need = head_len + stable_len + tail_len

            # placeholders (filled later from pooled buffers)
            state_head.append(np.empty(0, dtype=np.float64))
            state_stable.append(np.empty(0, dtype=np.float64))
            state_tail.append(np.empty(0, dtype=np.float64))

            if N_need <= 0:
                reqs_j.append({
                    "key": None,
                    "need": 0,
                    "head_len": head_len,
                    "stable_len": stable_len,
                    "tail_len": tail_len,
                    "state_idx": si,
                    "avgw_local": int(avgw_list[si]),
                })
                continue

            top_local = replace(tmpl.top)
            top_local.tau = float(tau_list_top[si])

            sec_local = replace(tmpl.second)
            sec_local.tau = float(tau_list_second[si])

            key = _pooled_cfg_key(j, top_local, sec_local)
            desc = _pooled_cfg_desc(j, top_local, sec_local)

            reqs_j.append({
                "key": key,
                "need": int(N_need),
                "head_len": head_len,
                "stable_len": stable_len,
                "tail_len": tail_len,
                "state_idx": si,
                "avgw_local": int(avgw_list[si]),
            })

            if key not in pooled_demands:
                pooled_demands[key] = {
                    "j": int(j),
                    "top_local": top_local,
                    "sec_local": sec_local,
                    "base_m_leaf": int(tmpl.m_leaf),
                    "desc": desc,
                    "states": [],
                    "states_need_sum": 0,  # only for logs/check
                }

            pooled_demands[key]["states"].append((int(j), int(si), int(N_need)))
            pooled_demands[key]["states_need_sum"] += int(N_need)

        all_state_head.append(state_head)
        all_state_stable.append(state_stable)
        all_state_tail.append(state_tail)
        all_trans_len_list.append(trans_len_list)
        state_requests.append(reqs_j)

    # Build pooled full datasets:
    # one pool per unique (comp + top cfg + second cfg)
    # pool length = comp_total_counts[comp]  <-- as requested
    pooled_buffers: Dict[Tuple, Dict] = {}
    pooled_keys_sorted = sorted(pooled_demands.keys(), key=lambda x: str(x))

    print(f"[Pool] unique config pools = {len(pooled_keys_sorted)}")
    print(f"[Pool] comp total counts = {comp_total_counts}")

    for pi, key in enumerate(pooled_keys_sorted):
        info = pooled_demands[key]
        j = int(info["j"])
        N_pool = int(comp_total_counts[j])              # <-- requested behavior
        states_need_sum = int(info["states_need_sum"])  # just for logging

        if N_pool <= 0:
            continue

        top_local: LayerParams = info["top_local"]
        sec_local: LayerParams = info["sec_local"]
        base_m_leaf = int(info["base_m_leaf"])

        if N_pool < avg_w_global:
            use_m_leaf = base_m_leaf
        else:
            use_m_leaf = max(1, int(round(base_m_leaf * (float(N_pool) / float(avg_w_global)))))

        spec_local = HierSpec(top_local, sec_local, use_m_leaf)

        # deterministic generation seed per pool
        seed_pool = int(np.random.default_rng(
            np.random.SeedSequence([args.shuffle_seed, 424242, pi])
        ).integers(1, 1 << 31))

        x_pool = generate_two_layer_exact(
            A=float(j), B=float(j + 1),
            N_total=int(N_pool),
            spec=spec_local,
            rng_seed_for_second=seed_pool
        )

        # shuffle immediately after generation (as requested)
        seed_pool_shuffle = int(np.random.default_rng(
            np.random.SeedSequence([args.shuffle_seed, 515151, pi])
        ).integers(1, 1 << 31))
        rng_pool_shuffle = np.random.default_rng(seed_pool_shuffle)
        rng_pool_shuffle.shuffle(x_pool)

        pooled_buffers[key] = {
            "data": x_pool.astype(np.float64, copy=False),
            "cursor": 0,
            "total": int(N_pool),
            "desc": info["desc"],
            "use_m_leaf": int(use_m_leaf),
            "states_need_sum": states_need_sum,
        }

        print(f"[Pool Build] pool_N={N_pool}, states_need_sum={states_need_sum}, "
              f"m_leaf={use_m_leaf}, avg_w_global={avg_w_global} | {info['desc']}")

    # Assign pooled slices to each state and split into head/stable/tail
    for j in range(k):
        for si, req in enumerate(state_requests[j]):
            need = int(req["need"])
            head_len = int(req["head_len"])
            stable_len = int(req["stable_len"])
            tail_len = int(req["tail_len"])
            key = req["key"]

            if need <= 0:
                print(f"[State Use] comp={j}, state={si}, need=0 (skip)")
                continue

            if key not in pooled_buffers:
                raise RuntimeError(f"Missing pooled buffer for comp={j}, state={si}")

            pb = pooled_buffers[key]
            start = int(pb["cursor"])
            end = start + need
            total = int(pb["total"])

            if end > total:
                raise RuntimeError(
                    f"Pooled buffer exhausted for comp={j}, state={si}: "
                    f"need={need}, cursor={start}, total={total}"
                )

            x = pb["data"][start:end]
            pb["cursor"] = end

            if head_len > 0:
                h = x[:head_len]
            else:
                h = np.empty(0, dtype=np.float64)

            if stable_len > 0:
                s = x[head_len: head_len + stable_len]
            else:
                s = np.empty(0, dtype=np.float64)

            if tail_len > 0:
                t = x[head_len + stable_len:]
            else:
                t = np.empty(0, dtype=np.float64)

            all_state_head[j][si] = h
            all_state_stable[j][si] = s
            all_state_tail[j][si] = t

            print(
                f"[State Use] comp={j}, state={si}, need={need} "
                f"(head={head_len}, stable={stable_len}, tail={tail_len}) "
                f"from pool[{pb['desc']}] slice[{start}:{end}] / {total}"
            )

    for _, pb in pooled_buffers.items():
        print(f"[Pool Summary] used={pb['cursor']}/{pb['total']} "
              f"(states_need_sum={pb['states_need_sum']}) | {pb['desc']}")

    cursor = 0
    base_seed_gen = 42

    for bi in range(num_blocks):
        # stable: cross-component shuffle
        stable_parts = []
        for j in range(k):
            s = all_state_stable[j][bi]
            if s.size > 0:
                stable_parts.append(s)

        if stable_parts:
            stable_block = np.concatenate(stable_parts).astype(np.float64, copy=False)

            rng_stable = np.random.default_rng(np.random.SeedSequence([args.shuffle_seed, 99991, bi]))
            rng_stable.shuffle(stable_block)

            if args.dtype == "uint64":
                u64 = convert_float_to_uint64(stable_block, scale_factor=1.0 / float(k), rng_seed=base_seed_gen)
                mm[cursor: cursor + u64.size] = u64
            else:
                mm[cursor: cursor + stable_block.size] = stable_block.astype(out_dtype, copy=False)

            cursor += stable_block.size
            base_seed_gen += 1337

        # transition: scheme A cross-component mixing + keep old->new
        score_list = []
        value_list = []

        for j in range(k):
            Tj = int(all_trans_len_list[j][bi])
            if Tj <= 0:
                continue

            prev_half = all_state_tail[j][bi]
            next_half = all_state_head[j][bi + 1] if (bi + 1) < len(all_state_head[j]) else np.empty(0, dtype=np.float64)

            seed_mix = int(np.random.default_rng(np.random.SeedSequence([args.shuffle_seed, 77777, bi, j])).integers(1, 1 << 31))

            sj, vj = _transition_scores_values(prev_half, next_half, T=Tj, seed=seed_mix)
            if vj.size > 0:
                score_list.append(sj)
                value_list.append(vj)

        if value_list:
            S = np.concatenate(score_list)
            V = np.concatenate(value_list)

            order = np.argsort(S, kind="mergesort")
            trans_block = V[order].astype(np.float64, copy=False)

            if args.dtype == "uint64":
                u64 = convert_float_to_uint64(trans_block, scale_factor=1.0 / float(k), rng_seed=base_seed_gen)
                mm[cursor: cursor + u64.size] = u64
            else:
                mm[cursor: cursor + trans_block.size] = trans_block.astype(out_dtype, copy=False)

            cursor += trans_block.size
            base_seed_gen += 1337

        if cursor % 1_000_000 == 0 or bi == num_blocks - 1:
            print(f"[Progress] {cursor/total_N*100:.1f}% ({cursor:,}/{total_N:,})")

    if cursor != total_N:
        raise RuntimeError(f"[Com] Size mismatch: wrote {cursor} samples, expected {total_N}.")

    if args.jitter_duplicates:
        pushed = jitter_duplicates_inplace(mm)
        print(f"[Jitter] duplicates pushed: {pushed}")

    del mm
    print("Saved to:", outfile)


if __name__ == "__main__":
    main()