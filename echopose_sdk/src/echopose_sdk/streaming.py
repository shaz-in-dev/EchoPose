"""echopose_sdk.streaming — CSI bundle streaming reader/writer.

Provides context-manager-based readers and writers for JSONL (newline-
delimited JSON) CSI bundle files produced by the EchoPose pipeline.

Usage example
-------------
Read all frames from a session log:

    from echopose_sdk.streaming import BundleReader

    with BundleReader("session.jsonl") as reader:
        for bundle in reader:
            print(bundle["timestamp_ms"])

Write frames captured from a live WebSocket feed:

    from echopose_sdk.streaming import BundleWriter

    with BundleWriter("capture.jsonl") as writer:
        for bundle in ws_feed():
            writer.write(bundle)
"""

from __future__ import annotations

import gzip
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, Generator, IO, Iterator, Optional, Union


PathLike = Union[str, Path]


class BundleReader:
    """Streaming reader for newline-delimited JSON bundle files.

    Supports plain ``.jsonl`` and gzip-compressed ``.jsonl.gz`` files.

    Parameters
    ----------
    path:
        Path to the JSONL file.
    skip_invalid:
        If ``True``, silently skip lines that fail to parse (default: ``False``).
    """

    def __init__(self, path: PathLike, skip_invalid: bool = False) -> None:
        self._path = Path(path)
        self._skip = skip_invalid
        self._fh: Optional[IO] = None
        self._count = 0

    def __enter__(self) -> "BundleReader":
        if self._path.suffix == ".gz":
            self._fh = gzip.open(self._path, "rt", encoding="utf-8")
        else:
            self._fh = self._path.open("r", encoding="utf-8")
        return self

    def __exit__(self, *_) -> None:
        if self._fh:
            self._fh.close()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self._fh is None:
            raise RuntimeError("Use BundleReader as a context manager.")
        for line in self._fh:
            line = line.strip()
            if not line:
                continue
            try:
                bundle = json.loads(line)
                self._count += 1
                yield bundle
            except json.JSONDecodeError as exc:
                if not self._skip:
                    raise
                continue

    @property
    def bundles_read(self) -> int:
        """Number of bundles yielded so far."""
        return self._count


class BundleWriter:
    """Streaming writer for newline-delimited JSON bundle files.

    Parameters
    ----------
    path:
        Destination path.  Use ``.jsonl.gz`` suffix for transparent gzip.
    append:
        If ``True``, open in append mode (default: ``False`` = overwrite).
    flush_every:
        Flush the underlying file handle every N bundles (default: 100).
    """

    def __init__(
        self,
        path: PathLike,
        append: bool = False,
        flush_every: int = 100,
    ) -> None:
        self._path = Path(path)
        self._append = append
        self._flush_every = flush_every
        self._fh: Optional[IO] = None
        self._count = 0

    def __enter__(self) -> "BundleWriter":
        mode = "at" if self._append else "wt"
        if self._path.suffix == ".gz":
            self._fh = gzip.open(self._path, mode, encoding="utf-8")
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._path.open(mode, encoding="utf-8")
        return self

    def __exit__(self, *_) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()

    def write(self, bundle: Dict[str, Any]) -> None:
        """Serialise a bundle dict as one JSON line."""
        if self._fh is None:
            raise RuntimeError("Use BundleWriter as a context manager.")
        self._fh.write(json.dumps(bundle, separators=(",", ":")) + "\n")
        self._count += 1
        if self._count % self._flush_every == 0:
            self._fh.flush()

    @property
    def bundles_written(self) -> int:
        """Number of bundles written so far."""
        return self._count


# ── frame-level utilities ─────────────────────────────────────────────────────

def stream_frames(
    source: PathLike,
    node_filter: Optional[str] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Yield individual CSI frames from a JSONL bundle file.

    Each bundle may contain multiple frames (one per node).  This generator
    flattens bundles into individual frame dicts, adding a ``"_bundle_ts"``
    key with the bundle's timestamp.

    Parameters
    ----------
    source:
        Path to a ``.jsonl`` or ``.jsonl.gz`` file.
    node_filter:
        If given, only yield frames whose ``"node_id"`` matches this value.
    """
    with BundleReader(source, skip_invalid=True) as reader:
        for bundle in reader:
            ts = bundle.get("timestamp_ms", bundle.get("ts", None))
            for frame in bundle.get("frames", []):
                if node_filter is not None and frame.get("node_id") != node_filter:
                    continue
                frame = dict(frame)
                frame["_bundle_ts"] = ts
                yield frame


def filter_by_time(
    source: PathLike,
    start_ms: Optional[float] = None,
    end_ms: Optional[float] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Yield bundles within a millisecond time window.

    Parameters
    ----------
    source:
        Path to JSONL file.
    start_ms, end_ms:
        Inclusive time bounds in milliseconds.  ``None`` means no bound.
    """
    with BundleReader(source, skip_invalid=True) as reader:
        for bundle in reader:
            ts = bundle.get("timestamp_ms", bundle.get("ts", None))
            if ts is None:
                yield bundle
                continue
            if start_ms is not None and ts < start_ms:
                continue
            if end_ms is not None and ts > end_ms:
                continue
            yield bundle


def count_bundles(source: PathLike) -> int:
    """Count the number of valid bundles in a JSONL file."""
    total = 0
    with BundleReader(source, skip_invalid=True) as reader:
        for _ in reader:
            total += 1
    return total


def split_train_test(
    source: PathLike,
    out_train: PathLike,
    out_test: PathLike,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Dict[str, int]:
    """Split a JSONL file into train and test sets.

    Bundles are shuffled deterministically before splitting.

    Parameters
    ----------
    source:
        Input JSONL file.
    out_train, out_test:
        Output paths for train and test JSONL files.
    test_fraction:
        Fraction of bundles allocated to the test set (default 0.2).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict with keys ``"train"`` and ``"test"`` holding bundle counts.
    """
    import random

    bundles = []
    with BundleReader(source, skip_invalid=True) as reader:
        for b in reader:
            bundles.append(b)

    rng = random.Random(seed)
    rng.shuffle(bundles)

    split_idx = int(len(bundles) * (1 - test_fraction))
    train_set = bundles[:split_idx]
    test_set = bundles[split_idx:]

    with BundleWriter(out_train) as w:
        for b in train_set:
            w.write(b)

    with BundleWriter(out_test) as w:
        for b in test_set:
            w.write(b)

    return {"train": len(train_set), "test": len(test_set)}
