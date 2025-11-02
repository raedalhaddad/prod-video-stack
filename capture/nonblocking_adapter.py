from __future__ import annotations

import contextlib
import sys
import threading
import time
from collections import deque
from typing import Deque, Literal, Optional, Protocol, Tuple

import numpy as np  # type: ignore

DropPolicy = Literal["drop_new", "drop_old"]


class FrameStream(Protocol):
    def start(self) -> None: ...
    def read(self) -> Optional[Tuple[np.ndarray, float, int]]: ...
    def close(self) -> None: ...


class NonBlocking:
    """
    Wrap any FrameStream so that:
      - start() runs in a worker thread (won't block the main thread)
      - read() is non-blocking via a small deque
      - close() attempts to stop gracefully with a short timeout (won't hang)
    """

    def __init__(
        self,
        inner: FrameStream,
        queue_max: int = 3,
        drop_policy: DropPolicy = "drop_new",
        start_timeout_s: float = 2.0,
        close_timeout_s: float = 0.75,
    ):
        self._inner = inner
        self._q: Deque[Tuple[np.ndarray, float, int]] = deque(maxlen=queue_max)
        self._drop = drop_policy
        self._ev = threading.Event()
        self._thr: Optional[threading.Thread] = None
        self._start_timeout_s = start_timeout_s
        self._close_timeout_s = close_timeout_s
        self._started = threading.Event()
        self._start_exc: Optional[BaseException] = None

    def start(self) -> None:
        # Kick off the worker; it will run inner.start() first, then ingest.
        if self._thr and self._thr.is_alive():
            return
        self._ev.set()
        self._thr = threading.Thread(target=self._worker, name="reader-nonblock", daemon=True)
        self._thr.start()
        # Wait briefly for inner.start(); don't block forever
        if not self._started.wait(self._start_timeout_s):
            # We continue anyway; worker might still be in a long start()
            print(
                f"[nonblock] WARN: inner.start() not ready after {self._start_timeout_s:.2f}s — continuing",
                file=sys.stderr,
            )

        if self._start_exc:
            # Propagate start error if one occurred
            raise self._start_exc

    def _worker(self) -> None:
        try:
            # Run inner.start() here; if it blocks, the main thread isn't affected.
            try:
                self._inner.start()
            except BaseException as e:
                self._start_exc = e
            finally:
                self._started.set()

            # Ingest loop
            while self._ev.is_set() and self._start_exc is None:
                item = None
                try:
                    item = self._inner.read()  # may block; that's fine here
                except Exception:
                    # tiny backoff on exceptions
                    time.sleep(0.001)
                if item is None:
                    # yield a tick to avoid hot spinning
                    time.sleep(0.0)
                    continue
                if len(self._q) == self._q.maxlen:
                    if self._drop == "drop_old":
                        self._q.popleft()
                        self._q.append(item)
                    else:
                        # drop_new
                        pass
                else:
                    self._q.append(item)
        finally:
            # Best-effort close in worker if main didn't call close()
            with contextlib.suppress(Exception):
                self._inner.close()

    def read(self) -> Optional[Tuple[np.ndarray, float, int]]:
        if not self._thr or not self._thr.is_alive():
            return None
        if not self._q:
            return None
        return self._q.popleft()

    def close(self) -> None:
        # Signal stop and try to close inner without hanging the caller
        self._ev.clear()

        # Close inner in a short-lived thread so we don't block
        def _closer():
            with contextlib.suppress(Exception):
                self._inner.close()

        t = threading.Thread(target=_closer, name="reader-close", daemon=True)
        t.start()
        t.join(timeout=self._close_timeout_s)
        # Don't join the ingest thread forever; give it a moment only
        if self._thr:
            self._thr.join(timeout=0.5)
            self._thr = None


def wrap_nonblocking(
    stream: FrameStream,
    queue_max: int = 3,
    drop_policy: DropPolicy = "drop_new",
    start_timeout_s: float = 2.0,
    close_timeout_s: float = 0.75,
) -> FrameStream:
    return NonBlocking(
        stream,
        queue_max=queue_max,
        drop_policy=drop_policy,
        start_timeout_s=start_timeout_s,
        close_timeout_s=close_timeout_s,
    )
