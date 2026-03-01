#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import bisect
import json
import math
import struct
import sys
from dataclasses import dataclass
from typing import Any, List, Tuple


@dataclass(frozen=True)
class Event:
    ts: int
    num: int


class InputError(RuntimeError):
    pass


def round_half_up_positive(x: float) -> int:
    return int(math.floor(x + 0.5))


def _as_int(name: str, v: Any) -> int:
    if isinstance(v, bool):
        raise InputError(f"{name} must be an integer, got bool")
    if isinstance(v, (int,)):
        return int(v)
    if isinstance(v, (float,)):
        if not math.isfinite(v):
            raise InputError(f"{name} must be finite, got {v!r}")
        if abs(v - int(v)) > 1e-12:
            raise InputError(f"{name} must be an integer, got {v!r}")
        return int(v)
    if isinstance(v, str):
        try:
            return int(v.strip())
        except Exception as e:
            raise InputError(f"{name} must be an integer string, got {v!r}") from e
    raise InputError(f"{name} must be an integer, got {type(v).__name__}")


def _extract_events(obj: Any) -> List[Event]:
    if isinstance(obj, dict) and "events" in obj:
        obj = obj["events"]

    events: List[Tuple[int, int]] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            ts = _as_int("timestamp", k)
            num = _as_int(f"num@{ts}", v)
            events.append((ts, num))
    elif isinstance(obj, list):
        for i, it in enumerate(obj):
            if isinstance(it, dict):
                if "timestamp" in it:
                    ts = _as_int(f"events[{i}].timestamp", it["timestamp"])
                elif "ts" in it:
                    ts = _as_int(f"events[{i}].ts", it["ts"])
                elif "t" in it:
                    ts = _as_int(f"events[{i}].t", it["t"])
                else:
                    raise InputError(f"events[{i}] missing timestamp field (expected timestamp/ts/t)")

                if "num" in it:
                    num = _as_int(f"events[{i}].num", it["num"])
                elif "n" in it:
                    num = _as_int(f"events[{i}].n", it["n"])
                else:
                    raise InputError(f"events[{i}] missing num field (expected num/n)")
                events.append((ts, num))
            elif isinstance(it, (list, tuple)) and len(it) == 2:
                ts = _as_int(f"events[{i}][0]", it[0])
                num = _as_int(f"events[{i}][1]", it[1])
                events.append((ts, num))
            else:
                raise InputError(
                    f"events[{i}] must be an object {{timestamp, num}} or [timestamp, num], got {type(it).__name__}"
                )
    elif obj is None:
        events = []
    else:
        raise InputError(f"Unsupported JSON root type: {type(obj).__name__}")

    out: List[Event] = []
    for ts, num in events:
        if ts < 1:
            raise InputError(f"timestamp must be >= 1, got {ts}")
        if num <= 0:
            raise InputError(f"num must be > 0, got {num} at timestamp={ts}")
        out.append(Event(ts=ts, num=num))
    out.sort(key=lambda e: e.ts)

    for i in range(1, len(out)):
        if out[i].ts <= out[i - 1].ts:
            raise InputError(f"timestamps must be strictly increasing, got {out[i-1].ts} then {out[i].ts}")
    return out


def _extract_dis_timestamps(obj: Any) -> List[int]:
    if isinstance(obj, dict):
        if "dis_events" in obj:
            obj = obj["dis_events"]
        elif "timestamps" in obj:
            obj = obj["timestamps"]
        elif "events" in obj and not ("num" in obj):
            obj = obj["events"]

    ts_list: List[int] = []

    if obj is None:
        ts_list = []
    elif isinstance(obj, list):
        for i, it in enumerate(obj):
            if isinstance(it, dict):
                if "timestamp" in it:
                    t = _as_int(f"dis_events[{i}].timestamp", it["timestamp"])
                elif "ts" in it:
                    t = _as_int(f"dis_events[{i}].ts", it["ts"])
                elif "t" in it:
                    t = _as_int(f"dis_events[{i}].t", it["t"])
                else:
                    raise InputError(f"dis_events[{i}] missing timestamp field (expected timestamp/ts/t)")
                ts_list.append(t)
            elif isinstance(it, (int, float, str)):
                t = _as_int(f"dis_events[{i}]", it)
                ts_list.append(t)
            elif isinstance(it, (list, tuple)) and len(it) >= 1:
                t = _as_int(f"dis_events[{i}][0]", it[0])
                ts_list.append(t)
            else:
                raise InputError(f"dis_events[{i}] unsupported item type: {type(it).__name__}")
    elif isinstance(obj, dict):
        for k in obj.keys():
            t = _as_int("dis_timestamp", k)
            ts_list.append(t)
    else:
        raise InputError(f"Unsupported dis_events JSON root type: {type(obj).__name__}")

    for t in ts_list:
        if t < 1:
            raise InputError(f"dis timestamp must be >= 1, got {t}")

    return ts_list


def _validate_event_gaps(events: List[Event], window_size: int) -> None:
    prev = 1
    for e in events:
        if e.ts - prev < window_size:
            raise InputError(
                f"Invalid event gap: timestamp {e.ts} - prev {prev} = {e.ts - prev} < window_size {window_size}"
            )
        prev = e.ts


def _interval_or_error(window_size: int, num: int, context: str) -> float:
    interval = window_size / float(num)
    if interval < 1.0:
        raise InputError(f"interval < 1: window_size/num = {window_size}/{num} = {interval} ({context})")
    return interval


def generate_timestamps(
    *,
    window_size: int,
    total_size: int,
    initial_num: int,
    events: List[Event],
    trim_forced_tail: bool,
) -> List[int]:
    if window_size <= 0:
        raise InputError("window_size must be > 0")
    if total_size < 1:
        raise InputError("total_size must be >= 1")
    if initial_num <= 0:
        raise InputError("initial_num must be > 0")

    _validate_event_gaps(events, window_size)

    _ = _interval_or_error(window_size, initial_num, "initial")
    for e in events:
        _ = _interval_or_error(window_size, e.num, f"event@{e.ts}")

    step = _interval_or_error(window_size, initial_num, "initial")

    switch_events: List[Event] = [Event(ts=e.ts - window_size, num=e.num) for e in events]
    for i in range(1, len(switch_events)):
        if switch_events[i].ts <= switch_events[i - 1].ts:
            raise InputError(
                f"Derived switch timestamps must be strictly increasing; got {switch_events[i-1].ts} then {switch_events[i].ts}"
            )

    cur_real = 1.0
    cur_ts = 1
    out = [1]

    idx = 0
    forced_cnt = 0

    while True:
        while idx < len(switch_events) and switch_events[idx].ts == cur_ts:
            step = _interval_or_error(window_size, switch_events[idx].num, f"switch@{switch_events[idx].ts}")
            idx += 1

        next_switch_ts = switch_events[idx].ts if idx < len(switch_events) else None

        cur_real += step
        nxt = round_half_up_positive(cur_real)

        if nxt <= cur_ts:
            nxt = cur_ts + 1
            cur_real = float(nxt)

        if next_switch_ts is not None and cur_ts < next_switch_ts and nxt > next_switch_ts:
            nxt = next_switch_ts
            cur_real = float(nxt)
            forced_cnt += 1

        out.append(nxt)
        cur_ts = nxt

        if cur_ts > total_size:
            break

    if trim_forced_tail and forced_cnt > 0:
        keep = max(1, len(out) - forced_cnt)
        out = out[:keep]

    return out


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate timestamps with dynamic interval changes driven by a JSON (timestamp,num) schedule."
    )
    p.add_argument("window_size", type=int, help="window_size (int > 0)")
    p.add_argument("total_size", type=int, help="Stop after generating a timestamp > total_size (int >= 1)")
    p.add_argument("initial_num", type=int, help="initial_num (int > 0). Initial interval = window_size/initial_num")
    p.add_argument("json_path", type=str, help="Path to JSON file containing (timestamp,num) pairs")

    p.add_argument(
        "--ts-file-type",
        choices=["text", "binary"],
        default="text",
        help=(
            "ts_file output type. text: one timestamp per line (default). "
            "binary: uint64 header(max_size) + uint64 array (little-endian)."
        ),
    )
    p.add_argument(
        "--format",
        choices=["one_line", "lines", "json"],
        default="lines",
        help="Output format. lines: one per line (default). one_line: space-separated in one line. json: JSON array.",
    )
    p.add_argument("--out", type=str, default="", help="Optional output file path. If omitted, write to stdout.")

    p.add_argument(
        "--dis-events",
        type=str,
        default="",
        help="Optional path to dis_events.json containing special timestamps for post-processing.",
    )
    p.add_argument(
        "--dis-out",
        type=str,
        default="",
        help=(
            "Output path for post-processing JSON result. "
            "If provided, script will output global diffs. "
            "If --dis-events is omitted/empty, diffs are computed using only [1, len(ts)]."
        ),
    )

    p.add_argument(
        "--trim-forced-tail",
        action="store_true",
        help="Drop as many timestamps from the END as the number of forced truncations to switch points.",
    )

    return p.parse_args(argv)


def _lower_bound_pos_1based(ts: List[int], x: int, *, name: str) -> int:
    i0 = bisect.bisect_left(ts, x)  # 0-based
    if i0 >= len(ts):
        raise InputError(f"{name}: no timestamp >= {x}; x is beyond generated range (max={ts[-1]})")
    return i0 + 1


def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    try:
        with open(args.json_path, "r", encoding="utf-8-sig") as f:
            obj = json.load(f)
        events = _extract_events(obj)

        ts = generate_timestamps(
            window_size=args.window_size,
            total_size=args.total_size,
            initial_num=args.initial_num,
            events=events,
            trim_forced_tail=args.trim_forced_tail,
        )

        if args.ts_file_type == "binary":
            if args.format != "lines":
                raise InputError("--ts-file-type=binary requires --format=lines (each timestamp is a uint64 element)")
            if not args.out:
                raise InputError("--ts-file-type=binary requires --out to write a binary file")
            max_size = len(ts)
            with open(args.out, "wb") as f:
                f.write(struct.pack("<Q", max_size))
                for x in ts:
                    if x < 0 or x > 0xFFFFFFFFFFFFFFFF:
                        raise InputError(f"timestamp out of uint64 range: {x}")
                    f.write(struct.pack("<Q", int(x)))
        else:
            if args.format == "one_line":
                content = " ".join(str(x) for x in ts) + "\n"
            elif args.format == "lines":
                content = "".join(f"{x}\n" for x in ts)
            else:
                content = json.dumps(ts, ensure_ascii=False) + "\n"

            if args.out:
                with open(args.out, "w", encoding="utf-8", newline="\n") as f:
                    f.write(content)
            else:
                sys.stdout.write(content)

        if args.dis_out:
            dis_ts: List[int] = []
            if args.dis_events:
                with open(args.dis_events, "r", encoding="utf-8-sig") as f:
                    dis_obj = json.load(f)
                dis_ts = _extract_dis_timestamps(dis_obj)

            switch_ts_list = [e.ts - args.window_size for e in events]
            positions: List[int] = [1]

            for d in dis_ts:
                pos_d = _lower_bound_pos_1based(ts, d, name="pos_dis")
                positions.append(pos_d)

                k = bisect.bisect_left(switch_ts_list, d)
                if k >= len(switch_ts_list):
                    raise InputError(
                        f"dis timestamp {d}: no switch_ts (=event.ts-window_size) >= {d} in events.json; cannot compute diffs"
                    )
                switch_ts = switch_ts_list[k]
                pos_switch = _lower_bound_pos_1based(ts, switch_ts, name="pos_switch")
                positions.append(pos_switch)

            positions.append(len(ts))

            positions.sort()
            diffs: List[int] = []
            for i in range(len(positions) - 1):
                diffs.append(positions[i + 1] - positions[i])

            rows: List[List[int]] = []
            i = 0
            while i < len(diffs):
                a = diffs[i]
                b = diffs[i + 1] if i + 1 < len(diffs) else 0
                rows.append([int(a), int(b), int(args.initial_num)])
                i += 2

            with open(args.dis_out, "w", encoding="utf-8", newline="\n") as f:
                json.dump(rows, f, ensure_ascii=False)
                f.write("\n")

        return 0

    except InputError as e:
        sys.stderr.write(f"[generate_ts] ERROR: {e}\n")
        return 2
    except FileNotFoundError as e:
        sys.stderr.write(f"[generate_ts] ERROR: file not found: {e}\n")
        return 2
    except json.JSONDecodeError as e:
        sys.stderr.write(f"[generate_ts] ERROR: JSON parse error: {e}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))