#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

__version__ = "0.1.0"


def _sample_md() -> str:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return (
        "# Sync Report (sample)\n\n"
        f"Generated: {now}\n\n"
        "- video: demo.mp4\n"
        "- audio: demo.m4a\n"
        "- offset_ms: 12\n"
        "- drift_ppm: 2.3\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_sync",
        description="Validate A/V sync logs and emit reports (stub).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    p_sample = sub.add_parser("sample", help="Emit a sample sync report to stdout or a file.")
    p_sample.add_argument("-o", "--output", help="Write report to this path (Markdown).")

    args = parser.parse_args(argv)

    if args.cmd == "sample":
        md = _sample_md()
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(md)
        else:
            sys.stdout.write(md)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
