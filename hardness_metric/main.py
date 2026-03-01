#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import csv
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from src.utils import read_list_file
from src.worker import compute_one_file
from src.plotting import generate_plot

def main():
    ap = argparse.ArgumentParser(
        description="Read directory (.npy/.npz or raw binary uint64), compute Global/Local hardness (L2/Std Metric) and output scatter plot and CSV."
    )
    ap.add_argument("--data_dir", required=True, help="Directory containing data files")
    ap.add_argument("--input_format", choices=["npy", "bin"], default="npy",
                    help="Input file format: npy (.npy/.npz) or bin (raw binary uint64, first value is length L)")
    ap.add_argument("--bin_endian", choices=["little", "big"], default="little",
                    help="Endianness for --input_format bin (default: little)")
    ap.add_argument("--npz_key", default=None, help="Key name for .npz files when input_format=npy (default: first key)")
    ap.add_argument("--mmap", action="store_true", help="Enable memory mapping for .npy/.npz when input_format=npy")

    ap.add_argument("--mode", choices=["pgm", "sc", "sd", "mse"], default="pgm",
                    help="pgm/sc: uses --pgm_error_file; sd: uses --sd_segs_file; mse: uses --mse_segs_file")

    ap.add_argument("--pgm_error_file", type=str, default=None,
                    help="Error list file for pgm/sc mode (one per line, order matches data files)")
    ap.add_argument("--mse_segs_file", type=str, default=None,
                    help="Segment count list file for mse mode (one per line, order matches data files)")
    ap.add_argument("--sd_segs_file", type=str, default=None,
                    help="Segment count list file for sd mode (one per line, order matches data files)")

    ap.add_argument("--sd_window", type=int, default=128, help="(Compatibility param) Sliding window length for legacy SD")
    ap.add_argument("--sd_radius", type=int, default=3, help="NMS radius for sd mode splitting (default: 3)")

    ap.add_argument("--pattern", default=None,
                    help="File matching pattern, comma separated; default='*.npy,*.npz' for npy, '*' for bin")
    ap.add_argument("--jitter_duplicates", action="store_true",
                    help="Apply minimal ULP push to duplicate keys to make them strictly increasing (recommended)")
    ap.add_argument("--dtype", choices=["auto", "float64", "uint64"], default="auto",
                    help="Specify semantic data type: auto(default)/float64/uint64; determines if high precision calculation is used")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="Number of parallel processes (default: CPU count)")
    ap.add_argument("--out", default="hardness_scatter_l2.png", help="Output image path")
    ap.add_argument("--csv_out", default="hardness_points_l2.csv", help="Output result CSV path")
    args = ap.parse_args()

    if args.pattern is None:
        args.pattern = "*.npy,*.npz" if args.input_format == "npy" else "*"

    patterns = [p.strip() for p in args.pattern.split(",") if p.strip()]
    files = []
    for pat in patterns:
        files += glob.glob(os.path.join(args.data_dir, pat))
    files = sorted(set(files))
    if not files:
        raise SystemExit(f"No matching files found in directory: {args.data_dir} / pattern={args.pattern}")

    if len(files) > 9:
        print(f"[INFO] Found {len(files)} files; processing only the first 9 as requested (sorted by filename).")
        files = files[:9]

    pgm_errors = None
    mse_segs = None
    sd_segs = None

    if args.mode in ("pgm", "sc"):
        if not args.pgm_error_file:
            raise SystemExit("pgm/sc mode requires --pgm_error_file")
        pgm_errors = read_list_file(args.pgm_error_file, float)
        if len(pgm_errors) != len(files):
            raise SystemExit(f"--pgm_error_file lines ({len(pgm_errors)}) must match number of data files ({len(files)})")
    elif args.mode == "mse":
        if not args.mse_segs_file:
            raise SystemExit("mse mode requires --mse_segs_file")
        mse_segs = read_list_file(args.mse_segs_file, int)
        if len(mse_segs) != len(files):
            raise SystemExit(f"--mse_segs_file lines ({len(mse_segs)}) must match number of data files ({len(files)})")
    elif args.mode == "sd":
        if not args.sd_segs_file:
            raise SystemExit("sd mode requires --sd_segs_file")
        sd_segs = read_list_file(args.sd_segs_file, int)
        if len(sd_segs) != len(files):
            raise SystemExit(f"--sd_segs_file lines ({len(sd_segs)}) must match number of data files ({len(files)})")

    points = []
    worker = partial(
        compute_one_file,
        input_format=args.input_format,
        bin_endian=args.bin_endian,
        npz_key=args.npz_key,
        mmap=args.mmap,
        mode=args.mode,
        jitter_duplicates=args.jitter_duplicates,
        dtype_spec=args.dtype,
        pgm_errors=pgm_errors,
        mse_segs=mse_segs,
        sd_segs=sd_segs,
        sd_window=args.sd_window,
        sd_radius=args.sd_radius,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fut2file = {ex.submit(worker, f, idx): (f, idx) for idx, f in enumerate(files)}
        for fut in as_completed(fut2file):
            res = fut.result()
            if res is None:
                continue
            if isinstance(res, tuple) and len(res)>=2 and isinstance(res[1], Exception):
                print(f"[WARN] Processing failed: {res[0]}  Reason: {res[1]}")
                continue
            points.append(res)

    if not points:
        raise SystemExit("No valid data points available.")

    points.sort(key=lambda t: t[0])

    with open(args.csv_out, "w", newline="") as f:
        w = csv.writer(f)
        headers = [
            "dataset",
            "Global=Std(start keys)",
            "Global=SlidingMedianStd",
            "Local=Avg Std(segments)",
            "n", "jittered", "m_used"
        ]
        w.writerow(headers)

        for tup in points:
            name, p_global, p_global_med, p_local, n, jit, info = tup
            w.writerow([name, p_global, p_global_med, p_local, n, jit, info.get("m_used")])

    generate_plot(points, args.mode, args.out)

    print(f"Done.\nOutput plot: {args.out}\nResult CSV: {args.csv_out}\nPoints: {len(points)}  Parallel: {min(args.workers, len(files))} processes")

if __name__ == "__main__":
    main()
