from __future__ import annotations

from capture.reader import ReaderConfig, ReaderFactory

if __name__ == "__main__":
    reader = ReaderFactory.from_config(ReaderConfig())
    reader.start()
    # give it a moment to connect
    for _ in range(10000):
        reader.read()
    stats = reader.stats()
    print(vars(stats))
    reader.close()
