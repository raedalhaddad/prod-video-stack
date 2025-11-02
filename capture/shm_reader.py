import json
import struct
import time
from typing import Optional, Tuple

import numpy as np

try:
    import mmap

    import pywintypes
    import win32file  # type: ignore
    import win32pipe  # type: ignore
except Exception:
    pass

HEADER_FMT = "<I H B B H H I Q Q I I"  # per spec
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def _normalize_pipe_path(p: Optional[str]) -> Optional[str]:
    """Return a normalized Windows named-pipe path, or None to disable."""
    if p is None:
        return None
    p = str(p).strip()
    if not p:
        return None
    if p.startswith(r"\\.\pipe\\"):
        return p
    return r"\\.\pipe\\" + p


class ShmTransport:
    def __init__(self, map_name: str, ctrl_pipe: Optional[str], timeout_s: float = 1.5, **kwargs):
        self._map_name = map_name
        self._ctrl_pipe = _normalize_pipe_path(ctrl_pipe)
        self.timeout_s = timeout_s
        self._mmap = None
        # Ring/meta (can be provided via kwargs to avoid pipe handshake)
        self._slot_bytes = int(kwargs.get("slot_bytes") or 0)
        self._slot_count = int(kwargs.get("slot_count") or 0)
        self._fmt = str(kwargs.get("fmt") or "BGRx")
        self._width = int(kwargs.get("width") or 0)
        self._height = int(kwargs.get("height") or 0)
        self._stride = int(kwargs.get("stride") or 0)
        self._last_seen_id = -1

    def start(self) -> None:
        # If a control pipe is provided, fetch ring meta from it; otherwise
        # rely on kwargs the factory passes (slot_count/slot_bytes/width/height/stride/fmt).
        if self._ctrl_pipe:
            meta = self._connect_ctrl()
            self._slot_count = int(meta["slot_count"])
            self._slot_bytes = int(meta["slot_bytes"])
            self._fmt = meta["fmt"]
            self._width = int(meta["width"])
            self._height = int(meta["height"])
            self._stride = int(meta["stride"])
            if meta.get("map_name"):
                self._map_name = str(meta["map_name"])
        else:
            if not (
                self._slot_count
                and self._slot_bytes
                and self._width
                and self._height
                and self._stride
                and self._fmt
            ):
                raise RuntimeError(
                    "SHM: ctrl_pipe disabled but ring metadata missing (need slot_count, slot_bytes, width, height, stride, fmt)"
                )
        self._open_mapping()

    def _connect_ctrl(self):
        """
        Connect to the control pipe and read one JSON blob with ring metadata.
        Robust against either a bare name (e.g., 'rtva_cam0_ctrl') or a full path
        (e.g., '\\\\.\\pipe\\rtva_cam0_ctrl').
        """
        # Normalize to canonical Windows pipe path
        path = self._ctrl_pipe
        if not path:
            return {}  # handshake disabled; caller provided kwargs

        deadline = time.time() + self.timeout_s
        last_err = None
        while time.time() < deadline:
            try:
                # Block up to 1000ms for the server side to create/accept
                try:
                    win32pipe.WaitNamedPipe(path, 1000)
                except pywintypes.error as e:
                    # 2=FILE_NOT_FOUND, 231=ALL_PIPE_INSTANCES_BUSY; just retry until deadline
                    last_err = e
                    time.sleep(0.05)
                    continue

                # Open for READ only (server writes one JSON then closes)
                handle = win32file.CreateFile(
                    path,
                    win32file.GENERIC_READ,
                    0,  # no sharing
                    None,
                    win32file.OPEN_EXISTING,
                    0,
                    None,
                )
                # Read a single JSON blob and close
                _, raw = win32file.ReadFile(handle, 4096)
                win32file.CloseHandle(handle)

                meta = json.loads(raw.decode("utf-8", errors="ignore"))
                # Cache fields
                self._slot_count = int(meta["slot_count"])
                self._slot_bytes = int(meta["slot_bytes"])
                self._fmt = meta["fmt"]
                self._width = int(meta["width"])
                self._height = int(meta["height"])
                self._stride = int(meta["stride"])
                # IMPORTANT: adopt the producer-advertised name for the mapping
                if meta.get("map_name"):
                    self.map_name = meta["map_name"]
                return meta

            except (pywintypes.error, OSError, ValueError) as e:
                last_err = e
                time.sleep(0.05)

        raise RuntimeError(f"Failed to connect control pipe: {last_err}")

    def _open_mapping(self):
        total = int(self._slot_bytes) * int(self._slot_count)
        # On Windows, open existing named mapping via tagname
        self._mmap = mmap.mmap(-1, total, tagname=self._map_name)

    def read(self) -> Optional[Tuple[np.ndarray, float, int]]:
        if self._mmap is None:
            return None
        # Scan all slots once
        next_id = None
        next_off = None
        newest_id = -1
        newest_off = None

        for i in range(self._slot_count):
            off = i * self._slot_bytes
            hdr = self._mmap[off : off + HEADER_SIZE]
            if len(hdr) != HEADER_SIZE:
                continue
            (magic, version, fmt_code, _r0, w, h, stride, frame_id, pts_ns, payload_len, _r2) = (
                struct.unpack(HEADER_FMT, hdr)
            )
            if magic != 0x53484D46 or version != 1:
                continue

            # Track newest regardless
            if frame_id > newest_id:
                newest_id = frame_id
                newest_off = off

            # If there is a frame immediately after what we last saw, pick the smallest such (> last_seen_id)
            if self._last_seen_id < frame_id and (next_id is None or frame_id < next_id):
                next_id = frame_id
                next_off = off

        # Live-first policy:
        # - On first run or if we're >2 frames behind, jump straight to NEWEST to kill backlog.
        # - Otherwise take the next sequential frame (smooth path).
        if newest_off is None:
            return None
        backlog = (newest_id - self._last_seen_id) if self._last_seen_id >= 0 else 0
        if self._last_seen_id < 0 or backlog > 2:
            sel_off, sel_id = newest_off, newest_id
        elif next_off is not None:
            sel_off, sel_id = next_off, next_id
        else:
            sel_off, sel_id = newest_off, newest_id

        if sel_off is None or sel_id == self._last_seen_id:
            return None

        # --- keep the existing double-read guard below, but use sel_off/sel_id ---
        hdr = self._mmap[sel_off : sel_off + HEADER_SIZE]
        (magic, version, fmt_code, _r0, w, h, stride, frame_id0, pts_ns, payload_len, _r2) = (
            struct.unpack(HEADER_FMT, hdr)
        )
        payload = self._mmap[sel_off + HEADER_SIZE : sel_off + HEADER_SIZE + payload_len]
        hdr2 = self._mmap[sel_off : sel_off + HEADER_SIZE]
        frame_id1 = struct.unpack(HEADER_FMT, hdr2)[7]
        if frame_id0 != frame_id1:
            # one retry
            hdr = self._mmap[sel_off : sel_off + HEADER_SIZE]
            (magic, version, fmt_code, _r0, w, h, stride, frame_id0, pts_ns, payload_len, _r2) = (
                struct.unpack(HEADER_FMT, hdr)
            )
            payload = self._mmap[sel_off + HEADER_SIZE : sel_off + HEADER_SIZE + payload_len]
            hdr2 = self._mmap[sel_off : sel_off + HEADER_SIZE]
            frame_id1 = struct.unpack(HEADER_FMT, hdr2)[7]
        if frame_id0 != frame_id1:
            return None  # torn frame, skip this round

        self._last_seen_id = frame_id1
        # ... proceed to reshape to BGR and return ...

        h = int(h)
        w = int(w)
        stride = int(stride)
        if fmt_code == 0:  # BGRx
            arr = np.frombuffer(payload, dtype=np.uint8).reshape(h, stride)[:, : w * 4]
            bgr = arr.reshape(h, w, 4)[:, :, :3].copy()
        else:
            bgr = (
                np.frombuffer(payload, dtype=np.uint8)
                .reshape(h, stride)[:, : w * 3]
                .reshape(h, w, 3)
                .copy()
            )

        # Return raw PTS in nanoseconds; the stats adapter maps it to epoch-ms via /health timebase.
        return bgr, float(pts_ns), int(frame_id1)

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
