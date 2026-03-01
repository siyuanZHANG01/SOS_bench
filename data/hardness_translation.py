#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import sys
from typing import Literal, Optional

TauKind = Literal["global", "local"]


def tau_absolute(
    tau: float,
    kind: TauKind,
    workset_size: int,
    leaf_size: int,
    *,
    divisor: float = 3000.0,
) -> float:

    if not math.isfinite(float(tau)):
        raise ValueError(f"tau must be finite, got {tau!r}")
    if workset_size <= 0:
        raise ValueError(f"workset_size must be > 0, got {workset_size}")
    if leaf_size <= 0:
        raise ValueError(f"leaf_size must be > 0, got {leaf_size}")
    if divisor <= 0:
        raise ValueError(f"divisor must be > 0, got {divisor}")

    if kind == "local":
        # Local/second layer: divide by divisor and square.
        x = float(tau) / float(divisor)
        return x * x

    if kind == "global":
        # Global/top layer:
        # 1) Derive the "upper-level bucket count" from workset_size/divisor.
        target_leaf_count = int(math.ceil(float(workset_size) / float(divisor)))
        if target_leaf_count < 1:
            target_leaf_count = 1

        # 2) Compute w_size using the provided rule.
        val1 = 8.0 * math.sqrt(target_leaf_count)
        val2 = target_leaf_count / 16.0
        w_size = int(min(val1, val2))
        if w_size < 1:
            w_size = 1

        # 3) Top tau: divide by w_size and square (do NOT divide by divisor).
        x = float(tau) / float(w_size)
        return x * x

    raise ValueError(f"Unknown kind: {kind!r} (expected 'global' or 'local')")


def _parse_args(argv):
    p = argparse.ArgumentParser(
        description="Compute 'absolute' tau transform for global(top) or local(second)."
    )
    p.add_argument("tau", type=float, help="Input tau value (float).")
    p.add_argument("kind", choices=["global", "local"], help='Mode: "global" (top) or "local" (second).')
    p.add_argument("workset_size", type=int, help="Workset size (int > 0).")
    p.add_argument("leaf_size", type=int, help="Leaf size (int > 0).")
    p.add_argument("--divisor", type=float, default=3000.0, help="Divisor for local and bucket-count derivation (default: 3000).")
    p.add_argument(
        "--json",
        action="store_true",
        help="Output JSON object instead of plain number.",
    )
    return p.parse_args(argv)


def main(argv) -> int:
    args = _parse_args(argv)
    try:
        out = tau_absolute(
            tau=args.tau,
            kind=args.kind,
            workset_size=args.workset_size,
            leaf_size=args.leaf_size,
            divisor=args.divisor,
        )
        if args.json:
            sys.stdout.write(json.dumps({"tau_in": args.tau, "kind": args.kind, "tau_out": out}, ensure_ascii=False) + "\n")
        else:
            sys.stdout.write(f"{out}\n")
        return 0
    except Exception as e:
        sys.stderr.write(f"[tau_absolute] ERROR: {e}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))