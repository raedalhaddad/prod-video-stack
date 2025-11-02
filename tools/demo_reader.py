from __future__ import annotations

import argparse
import contextlib
import json
import re
import time

from capture.nonblocking_adapter import wrap_nonblocking
from capture.reader import ReaderConfig, ReaderFactory
from capture.stats_adapter import make_pts_mapper_from_health, wrap_with_stats


def _fetch_producer_stats(url: str):
    txt = None
    try:
        import requests

        txt = requests.get(url, timeout=1.0).text.strip()
    except Exception:
        try:
            from urllib.request import urlopen

            with urlopen(url, timeout=1.0) as resp:
                txt = resp.read().decode("utf-8").strip()
        except Exception:
            return None
    if not txt:
        return None
    # Try JSON first
    try:
        return json.loads(txt)
    except Exception:
        pass
    # Fallback: parse whitespace/table-ish output (two+ spaces or tabs)
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    if len(lines) >= 2:
        headers = re.split(r"\s{2,}|\t+", lines[0].strip())
        values = re.split(r"\s{2,}|\t+", lines[-1].strip())
        if len(values) >= len(headers):
            return dict(zip(headers, values))
    return {"raw": txt}


def _dump_chain(obj, maxdepth=5):
    i, cur = 0, obj
    while i < maxdepth and cur is not None:
        print(f"[demo] layer{i}: {type(cur).__name__} mod={getattr(cur,'__module__','')}")
        cur = getattr(cur, "_inner", None)
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--health", type=str, default="http://127.0.0.1:8765/health")
    ap.add_argument("--stats-url", type=str, default="http://127.0.0.1:8765/stats")
    ap.add_argument("--prefer", type=str, default="shm", choices=["shm", "null"])
    ap.add_argument("--ctrl-pipe", type=str, default=r"\\.\pipe\rtva_cam0_ctrl")
    args = ap.parse_args()

    print("[demo] file =", __file__)
    print("[demo] parsed seconds =", args.seconds)

    cfg = ReaderConfig(prefer=args.prefer, health_url=args.health, ctrl_pipe=args.ctrl_pipe)
    reader = ReaderFactory.from_config(cfg)
    try:
        # best-effort: show the format we discovered
        import json as _json
        import urllib.request

        with urllib.request.urlopen(args.health, timeout=1.0) as _r:
            _h = _json.loads(_r.read().decode("utf-8"))
            _fmt = _h.get("shm", {}).get("format")
            if _fmt:
                print(f"[demo] shm.format = {_fmt}")
    except Exception:
        pass
    # Ensure non-blocking reads regardless of native behavior
    reader = wrap_nonblocking(reader, queue_max=3, drop_policy="drop_old", start_timeout_s=6.0)
    # Normalize timebase + add uniform stats
    pts_map = make_pts_mapper_from_health(args.health)
    desc = getattr(pts_map, "_desc", None)
    if desc:
        print(f"[timebase] {desc}")
    reader = wrap_with_stats(reader, pts_mapper=pts_map)
    _dump_chain(reader)
    print(f"[demo] transport={type(reader).__name__} mod={getattr(reader,'__module__','')}")
    reader.start()
    try:
        t_end = time.time() + args.seconds
        frames = 0
        last_fid = None
        stats = None
        warn_thresh_ms = 1500
        while time.time() < t_end:
            item = reader.read()
            if item is None:
                continue
            _, pts_ms, fid = item
            frames += 1
            last_fid = fid
            # lightweight watchdog
            age = reader.stats().last_frame_age_ms
            if age > warn_thresh_ms:
                print(f"[watchdog] last_frame_age_ms={age:.0f} (> {warn_thresh_ms} ms)")
                stats = reader.stats()
    except KeyboardInterrupt:
        print("[demo] KeyboardInterrupt")
    finally:
        try:
            import time as _t

            # best-effort: show mapping of the last raw PTS we saw (stats wrapper can expose it)
            last_raw = getattr(reader, "_last_pts_raw", None)
            last_mapped = getattr(reader, "_last_pts_ms", None)
            if last_raw is not None and last_mapped is not None:
                now_ms = _t.time() * 1000.0
                age = now_ms - last_mapped
                print(
                    f"[diag] last_raw_pts={last_raw:.0f} → mapped_ms={last_mapped:.1f} now_ms={now_ms:.1f} age_ms={age:.1f}"
                )
        except Exception:
            pass
        # Print summary even if something went sideways
        try:
            if stats is None:
                stats = reader.stats()
            fps = frames / max(args.seconds, 1)
            print("[demo] seconds=", args.seconds)
            print(f"[demo] frames_out={frames} (~{fps:.2f} fps)")
            if stats is not None:
                print(
                    f"[demo] in={getattr(stats,'frames_in',0)} out={getattr(stats,'frames_out',0)} "
                    f"drops={getattr(stats,'drops',0)} last_age_ms={getattr(stats,'last_frame_age_ms',0.0):.1f}"
                )
                print(
                    f"[demo] read_loop_us mean={getattr(stats,'read_loop_us_mean',0.0):.0f} "
                    f"p95={getattr(stats,'read_loop_us_p95',0.0):.0f}"
                )
                # Producer stats (optional, best effort)
                prod = _fetch_producer_stats(args.stats_url)
                if prod:
                    enc_fps = prod.get("encoder_fps") or prod.get("encoder-fps")
                    seg_cnt = (
                        prod.get("segment_count")
                        or prod.get("segments")
                        or prod.get("segment-count")
                    )
                    last_seg = prod.get("last_segment") or prod.get("last-segment")
                    latest_id = prod.get("latest_frame_id") or prod.get("latest-frame-id")
                    print(
                        "[producer] encoder_fps=",
                        enc_fps,
                        "segment_count=",
                        seg_cnt,
                        "last_segment=",
                        last_seg,
                    )
                    if latest_id is not None and last_fid is not None:
                        try:
                            lag = int(latest_id) - int(last_fid)
                            print(
                                f"[telemetry] producer_latest_id={latest_id} consumer_last_id={last_fid} lag_fids={lag}"
                            )
                        except Exception:
                            print(
                                f"[telemetry] producer_latest_id={latest_id} consumer_last_id={last_fid} (non-numeric)"
                            )
            else:
                print("[demo] transport exposes no stats(); only frames_out reported.")
        finally:
            # Close with safeguards (NonBlocking.close is already timeout-safe)
            with contextlib.suppress(Exception):
                reader.close()


if __name__ == "__main__":
    main()
