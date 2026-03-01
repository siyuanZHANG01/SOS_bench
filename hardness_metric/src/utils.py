import os
import math
import numpy as np

def read_list_file(path: str, as_type):
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            token = s.split()[0].split(",")[0]
            vals.append(as_type(token))
    return vals

def load_series_npy(path: str, npz_key=None, mmap=False) -> np.ndarray:
    if path.endswith(".npz"):
        arrs = np.load(path, allow_pickle=False, mmap_mode=("r" if mmap else None))
        key = list(arrs.keys())[0] if npz_key is None else npz_key
        x = arrs[key]
    elif path.endswith(".npy"):
        x = np.load(path, allow_pickle=False, mmap_mode=("r" if mmap else None))
    else:
        raise ValueError(f"Unsupported file type (npy mode): {path}")
    if x.ndim != 1:
        x = np.array(x).reshape(-1, order="C")
    x = x[np.isfinite(x)]
    return x

def load_series_bin_uint64(path: str, endian: str = "little") -> np.ndarray:
    dt = "<u8" if endian == "little" else ">u8"
    arr = np.fromfile(path, dtype=dt)
    if arr.size == 0:
        return arr
    L = int(arr[0])
    data = arr[1:]
    if L <= 0:
        return np.array([], dtype=np.uint64)
    if L > data.size:
        print(f"[WARN] {os.path.basename(path)} declared length {L} > actual {data.size}, using only available part.")
        L = data.size
    return data[:L]

def need_high_precision(spec_dtype: str, data: np.ndarray) -> bool:
    if spec_dtype == "uint64":
        return True
    if spec_dtype == "float64":
        return False
    if data.dtype == np.uint64:
        return data.size > 0 and int(data.max()) > (2**53 - 1)
    return False

def make_strictly_increasing_with_count(x_sorted: np.ndarray):
    if x_sorted.size <= 1:
        return x_sorted.copy(), 0
    out = x_sorted.copy()
    prev = out[0]
    cnt = 0
    for i in range(1, out.size):
        v = out[i]
        if v <= prev:
            v = np.nextafter(prev, np.inf, dtype=out.dtype) if hasattr(np, "nextafter") else np.nextafter(prev, np.inf)
            cnt += 1
        out[i] = v
        prev = v
    return out, cnt

def _ols_centered_coeffs(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    n = x.size
    if n == 0:
        return 0.0, 0.0

    dt = np.result_type(x.dtype, y.dtype, np.float64)
    xx = x.astype(dt, copy=False)
    yy = y.astype(dt, copy=False)

    xm = xx.mean(dtype=dt)
    ym = yy.mean(dtype=dt)
    dx = xx - xm
    dy = yy - ym

    denom = np.dot(dx, dx)
    if denom == 0:
        a = dt.type(0.0)
    else:
        a = np.dot(dx, dy) / denom
    b = ym - a * xm
    return a, b

def ols_line_and_std(x: np.ndarray, y: np.ndarray) -> tuple:
    n = x.size
    if n == 0:
        return 0.0, 0.0, np.nan
    a, b = _ols_centered_coeffs(x, y)
    err = y - (a * x + b)
    std_val = float(np.std(err)) if n > 1 else 0.0
    return a, b, std_val

def ols_std_segment_internal(x: np.ndarray, y: np.ndarray) -> float:
    n = x.size
    if n <= 1:
        return 0.0
    a, b = _ols_centered_coeffs(x, y)
    err = y - (a * x + b)
    return float(np.std(err))

def calculate_sliding_median_std(x: np.ndarray, y: np.ndarray) -> float:
    n = x.size
    
    if n < 300:
        print("The data volume is too small; the sliding window for calculating top-level error has been disabled.")
        _, _, std_val = ols_line_and_std(x, y)
        return float(std_val)

    val1 = 8.0 * math.sqrt(n)
    val2 = n / 16.0
    
    raw_window = min(val1, val2)
    window = max(1, int(raw_window))
    step = max(1, int(window / 4))
    
    errs = []
    for i in range(0, n - window + 1, step):
        wx = x[i : i + window]
        wy = y[i : i + window]
        std_val = ols_std_segment_internal(wx, wy)
        errs.append(std_val)
    
    if not errs:
        _, _, std_val = ols_line_and_std(x, y)
        return float(std_val)

    return float(np.median(errs))


def eqpop_edges(n: int, m_override=None) -> np.ndarray:
    if n <= 0:
        return np.array([0, 0], dtype=int)
    if m_override is None:
        m = int(math.floor(math.sqrt(n)))
    else:
        m = int(m_override)
    m = max(1, min(m, n))
    edges = np.linspace(0, n, m + 1, dtype=int)
    edges = np.unique(edges)
    if edges[-1] != n:
        edges[-1] = n
    return edges
