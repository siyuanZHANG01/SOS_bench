#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
import os
from pathlib import Path
import numpy as np


def make_strictly_increasing_with_count_inplace(sorted_vals: np.ndarray) -> int:
    """
    In-place make sorted_vals strictly increasing with minimal upward jitter.

    - float32/float64: if v <= prev -> v = nextafter(prev, +inf)
    - uint64: if v <= prev -> v = prev + 1

    Returns: number of elements changed.
    Notes:
      * Assumes sorted_vals is non-decreasing initially.
      * Rejects NaN/Inf for float.
    """
    n = int(sorted_vals.size)
    if n <= 1:
        return 0

    cnt = 0

    if sorted_vals.dtype == np.uint64:
        prev = int(sorted_vals[0])
        UINT64_MAX = (1 << 64) - 1
        for i in range(1, n):
            v = int(sorted_vals[i])
            if v <= prev:
                if prev == UINT64_MAX:
                    raise OverflowError("uint64 overflow during jitter: cannot push beyond UINT64_MAX")
                v = prev + 1
                sorted_vals[i] = np.uint64(v)
                cnt += 1
            prev = int(sorted_vals[i])
        return cnt

    if sorted_vals.dtype == np.float32:
        prev = np.float32(sorted_vals[0])
        if not np.isfinite(prev):
            raise ValueError("Found non-finite value (NaN/Inf); not supported for float jitter.")
        for i in range(1, n):
            v = np.float32(sorted_vals[i])
            if not np.isfinite(v):
                raise ValueError("Found non-finite value (NaN/Inf); not supported for float jitter.")
            if v <= prev:
                nv = np.nextafter(prev, np.float32(np.inf)).astype(np.float32)
                if not (nv > prev) or not np.isfinite(nv):
                    raise OverflowError("float32 overflow/stuck during jitter (nextafter did not increase).")
                sorted_vals[i] = nv
                prev = nv
                cnt += 1
            else:
                prev = v
        return cnt

    if sorted_vals.dtype == np.float64:
        prev = np.float64(sorted_vals[0])
        if not np.isfinite(prev):
            raise ValueError("Found non-finite value (NaN/Inf); not supported for float jitter.")
        for i in range(1, n):
            v = np.float64(sorted_vals[i])
            if not np.isfinite(v):
                raise ValueError("Found non-finite value (NaN/Inf); not supported for float jitter.")
            if v <= prev:
                nv = np.nextafter(prev, np.float64(np.inf)).astype(np.float64)
                if not (nv > prev) or not np.isfinite(nv):
                    raise OverflowError("float64 overflow/stuck during jitter (nextafter did not increase).")
                sorted_vals[i] = nv
                prev = nv
                cnt += 1
            else:
                prev = v
        return cnt

    raise TypeError(f"Unsupported dtype: {sorted_vals.dtype}")


def write_header(f, n: int, header_dtype: str):
    # little-endian header
    if header_dtype == "uint64":
        np.array([n], dtype="<u8").tofile(f)
    elif header_dtype == "uint32":
        if n >= (1 << 32):
            raise ValueError(f"N={n} does not fit into uint32 header")
        np.array([n], dtype="<u4").tofile(f)
    else:
        raise ValueError("header_dtype must be uint64 or uint32")


def list_npy_files(root: Path, recursive: bool) -> list[Path]:
    return sorted(root.rglob("*.npy") if recursive else root.glob("*.npy"))


def build_out_path(npy_path: Path, out_suffix: str, out_ext: str) -> Path:
    # 同名：stem 相同；输出为 stem + out_suffix + out_ext
    stem = npy_path.stem
    if not out_ext.startswith("."):
        out_ext = "." + out_ext
    return npy_path.with_name(f"{stem}{out_suffix}{out_ext}")


def maybe_to_little_endian_view(buf: np.ndarray) -> np.ndarray:
    """
    Ensure written bytes are little-endian for portability.
    If already little-endian/native little-endian -> zero-copy view.
    If big-endian -> byteswap (copy).
    """
    dt = buf.dtype
    dt_le = dt.newbyteorder("<")
    if dt == dt_le:
        return buf
    # big-endian -> swap to little-endian (copy)
    return buf.byteswap().newbyteorder("<")


def jitter_npy_to_binary(
    npy_path: Path,
    out_path: Path,
    header_dtype: str,
    chunk_elems: int,
    overwrite: bool,
    quiet: bool,
) -> int:
    """
    Read npy (mmap), global sort-based jitter de-dup (preserve original order),
    write binary: [header N] + raw array (dtype same as input, little-endian for portability).
    Return: pushed/changed count. Return -1 if skipped.
    """
    if out_path.exists() and not overwrite:
        if not quiet:
            print(f"[SKIP] {npy_path.name}: output exists -> {out_path.name}")
        return -1

    x = np.load(str(npy_path), mmap_mode="r")
    if x.ndim != 1:
        x = x.reshape(-1)

    N = int(x.size)
    dtype = x.dtype

    if dtype not in (np.float32, np.float64, np.uint64):
        if not quiet:
            print(f"[SKIP] {npy_path.name}: dtype={dtype} not supported (float32/float64/uint64 only).")
        del x
        gc.collect()
        return -1

    if not quiet:
        print(f"[FILE] {npy_path.name}  N={N:,}  dtype={dtype}  -> {out_path.name}")

    # 1) stable argsort
    idx = np.argsort(x, kind="mergesort")

    # 2) sorted values copy + jitter
    sorted_vals = np.asarray(x[idx]).copy()
    pushed = make_strictly_increasing_with_count_inplace(sorted_vals)

    # 3) inverse permutation (map back to original order)
    inv = np.empty_like(idx)
    inv[idx] = np.arange(N, dtype=idx.dtype)

    # 4) write binary (atomic): tmp then replace
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")

    with open(tmp_path, "wb") as f:
        write_header(f, N, header_dtype)

        chunk = int(max(1, chunk_elems))
        for start in range(0, N, chunk):
            end = min(N, start + chunk)
            ranks = inv[start:end]
            buf = sorted_vals[ranks]

            # write little-endian bytes for portability
            buf_le = maybe_to_little_endian_view(buf)
            buf_le.tofile(f)

    os.replace(str(tmp_path), str(out_path))

    if not quiet:
        print(f"  -> wrote, jitter_pushed={pushed:,}")

    # free memory aggressively
    del x, idx, sorted_vals, inv
    gc.collect()

    return pushed


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch: read all .npy in a directory, global jitter de-dup (sort-based), keep order, write binary per file (keep original .npy)."
    )
    p.add_argument("--dir", required=True, type=str, help="Directory containing .npy files")
    p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    p.add_argument("--out_ext", type=str, default=".bin", help="Output extension (default: .bin)")
    p.add_argument("--out_suffix", type=str, default="", help="Extra suffix before extension (e.g. .dedup)")
    p.add_argument("--header_dtype", default="uint64", choices=["uint64", "uint32"],
                   help="Binary header type for max_size=N (little-endian). Default: uint64")
    p.add_argument("--chunk_elems", type=int, default=2_000_000,
                   help="Chunk elements for gather+write (default: 2,000,000)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output binary if exists")
    p.add_argument("--quiet", action="store_true", help="Less logging")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"--dir must be an existing directory: {root}")

    files = list_npy_files(root, args.recursive)
    if not files:
        print(f"No .npy files found in: {root}")
        return

    total = len(files)
    processed = 0
    skipped = 0
    errors = 0
    pushed_total = 0

    if not args.quiet:
        print(f"Found {total} .npy files under: {root} (recursive={args.recursive})")
        print(f"Output naming: stem + '{args.out_suffix}' + '{args.out_ext}'")
        print(f"Header dtype: {args.header_dtype}")

    for i, fp in enumerate(files, 1):
        outp = build_out_path(fp, args.out_suffix, args.out_ext)
        try:
            pushed = jitter_npy_to_binary(
                npy_path=fp,
                out_path=outp,
                header_dtype=args.header_dtype,
                chunk_elems=args.chunk_elems,
                overwrite=args.overwrite,
                quiet=args.quiet,
            )
            if pushed < 0:
                skipped += 1
            else:
                processed += 1
                pushed_total += pushed
        except Exception as e:
            errors += 1
            # cleanup temp if present
            tmp = outp.with_name(outp.name + ".tmp")
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass
            print(f"[ERROR] {fp}: {e}")

        if not args.quiet:
            print(f"[Progress] {i}/{total}")

    print(f"Done. processed={processed}, skipped={skipped}, errors={errors}, total_pushed={pushed_total:,}")


if __name__ == "__main__":
    main()
