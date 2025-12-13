#!/usr/bin/env python
"""
debug_segments_horizon.py

Inspect the segment "time horizon" from the producer log.

This script parses lines like:

    [clip] indexed segment: start=1765493315733 end=1765493328806 ms location=D:/media/security_camera_app/segments/seg_08494.mp4

and prints:

- A summary of the most recent N indexed segments.
- The overall horizon: earliest_start_ms, latest_end_ms.
- If an event window is provided, where it falls relative to that horizon.

Usage examples:

    python tools/debug_segments_horizon.py --log D:/logs/producer.log

    python tools/debug_segments_horizon.py \\
        --log D:/logs/producer.log \\
        --last-n 50 \\
        --event-start-ms 1765488291770.511 \\
        --event-stop-ms 1765488294770.511
"""

import argparse
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

SEGMENT_LINE_RE = re.compile(
    r"\[clip\]\s+indexed segment:\s+"
    r"start=(?P<start>\d+(?:\.\d+)?)\s+"
    r"end=(?P<end>\d+(?:\.\d+)?)\s+ms\s+"
    r"location=(?P<loc>\S+)"
)

SEG_INDEX_RE = re.compile(r"seg_(?P<idx>\d+)\.mp4$", re.IGNORECASE)


@dataclass
class Segment:
    start_ms: float
    end_ms: float
    path: str
    index: Optional[int] = None  # extracted from filename if present


def _read_log_text(path: str) -> str:
    """
    Read the log file as text, handling both UTF-8 and UTF-16 (with BOM).
    """
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        print(f"ERROR: log file not found: {path}", file=sys.stderr)
        sys.exit(1)

    # Detect UTF-16 BOM (LE or BE)
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            # Fall back to utf-8 if something is really weird
            return raw.decode("utf-8", errors="replace")

    # Default: assume UTF-8
    return raw.decode("utf-8", errors="replace")


def parse_segments_from_log(path: str) -> List[Segment]:
    segments: List[Segment] = []
    text = _read_log_text(path)

    for line in text.splitlines():
        m = SEGMENT_LINE_RE.search(line)
        if not m:
            continue
        start_ms = float(m.group("start"))
        end_ms = float(m.group("end"))
        loc = m.group("loc")
        idx_match = SEG_INDEX_RE.search(loc)
        seg_idx: Optional[int] = None
        if idx_match:
            try:
                seg_idx = int(idx_match.group("idx"))
            except ValueError:
                seg_idx = None
        segments.append(Segment(start_ms=start_ms, end_ms=end_ms, path=loc, index=seg_idx))

    # Sort by start_ms (and index as a tie breaker if present)
    segments.sort(key=lambda s: (s.start_ms, s.index if s.index is not None else -1))
    return segments


def classify_event_vs_horizon(
    event_start_ms: float,
    event_stop_ms: float,
    earliest_ms: float,
    latest_ms: float,
) -> str:
    """
    Classify where the event window sits relative to the segment horizon.
    """
    if event_stop_ms < earliest_ms:
        delta = earliest_ms - event_stop_ms
        return f"event is COMPLETELY BEFORE horizon by {delta:.3f} ms"
    if event_start_ms > latest_ms:
        delta = event_start_ms - latest_ms
        return f"event is COMPLETELY AFTER horizon by {delta:.3f} ms"
    if event_start_ms >= earliest_ms and event_stop_ms <= latest_ms:
        return "event window is FULLY CONTAINED within horizon"
    return "event window PARTIALLY OVERLAPS horizon"


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect segment horizon from producer log.")
    ap.add_argument(
        "--log",
        required=True,
        help="Path to producer log file containing '[clip] indexed segment' lines.",
    )
    ap.add_argument(
        "--last-n",
        type=int,
        default=50,
        help="How many most recent segments to display (default: 50).",
    )
    ap.add_argument(
        "--event-start-ms",
        type=float,
        default=None,
        help="Optional motion event start_ms (epoch ms) to compare to horizon.",
    )
    ap.add_argument(
        "--event-stop-ms",
        type=float,
        default=None,
        help="Optional motion event stop_ms (epoch ms) to compare to horizon.",
    )

    args = ap.parse_args()

    segments = parse_segments_from_log(args.log)
    if not segments:
        print("No '[clip] indexed segment' lines found in log.")
        sys.exit(0)

    total = len(segments)
    print(f"Found {total} indexed segments in log.")

    # Compute overall horizon
    earliest_ms = min(s.start_ms for s in segments)
    latest_ms = max(s.end_ms for s in segments)
    print()
    print("=== Overall Horizon ===")
    print(f"  earliest start_ms: {earliest_ms:.3f}")
    print(f"  latest  end_ms   : {latest_ms:.3f}")
    print()

    # Show last N segments
    last_n = max(1, args.last_n)
    tail = segments[-last_n:]

    print(f"=== Last {len(tail)} segments (by start_ms) ===")
    print(f"{'idx':>6}  {'start_ms':>16}  {'end_ms':>16}  {'duration_ms':>12}  path")
    for s in tail:
        duration = s.end_ms - s.start_ms
        idx_str = str(s.index) if s.index is not None else "-"
        print(f"{idx_str:>6}  {s.start_ms:16.3f}  {s.end_ms:16.3f}  {duration:12.3f}  {s.path}")

    # If event window provided, classify it
    if args.event_start_ms is not None and args.event_stop_ms is not None:
        es = args.event_start_ms
        ee = args.event_stop_ms
        print()
        print("=== Event vs Horizon ===")
        print(f"  event_start_ms: {es:.3f}")
        print(f"  event_stop_ms : {ee:.3f}")
        relation = classify_event_vs_horizon(es, ee, earliest_ms, latest_ms)
        print(f"  classification: {relation}")
    elif (args.event_start_ms is not None) ^ (args.event_stop_ms is not None):
        print(
            "\nWARNING: You provided only one of --event-start-ms / --event-stop-ms. "
            "Provide both to get classification.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
