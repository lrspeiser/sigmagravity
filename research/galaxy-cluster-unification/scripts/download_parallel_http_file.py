#!/usr/bin/env python3
"""Download one immutable HTTP file with verified parallel byte ranges."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def request(url: str, *, method: str = "GET", headers: dict | None = None):
    return urllib.request.urlopen(
        urllib.request.Request(url, method=method, headers=headers or {}), timeout=120
    )


def download_range(url: str, part: Path, start: int, end: int, total: int) -> None:
    expected = end - start + 1
    if part.exists() and part.stat().st_size == expected:
        return
    temporary = part.with_suffix(part.suffix + ".tmp")
    for attempt in range(5):
        try:
            with request(url, headers={"Range": f"bytes={start}-{end}"}) as response:
                content_range = response.headers.get("Content-Range", "")
                if response.status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError(
                        f"range {start}-{end} returned {response.status} {content_range!r}"
                    )
                with temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
            if temporary.stat().st_size != expected:
                raise RuntimeError(
                    f"range {start}-{end} has {temporary.stat().st_size} bytes, expected {expected}"
                )
            os.replace(temporary, part)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-mib", type=int, default=64)
    args = parser.parse_args()

    with request(args.url, method="HEAD") as response:
        total = int(response.headers["Content-Length"])
        accept_ranges = response.headers.get("Accept-Ranges", "").lower()
        etag = response.headers.get("ETag")
        last_modified = response.headers.get("Last-Modified")
    if accept_ranges != "bytes":
        raise RuntimeError(f"server does not advertise byte ranges: {accept_ranges!r}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.stat().st_size == total:
        print(f"existing {args.output} {total} bytes sha256={sha256(args.output)}")
        return

    parts_dir = args.output.with_name(args.output.name + ".ranges")
    parts_dir.mkdir(parents=True, exist_ok=True)
    chunk_bytes = args.chunk_mib * 1024 * 1024
    ranges = []
    for index, start in enumerate(range(0, total, chunk_bytes)):
        end = min(total - 1, start + chunk_bytes - 1)
        ranges.append((parts_dir / f"part-{index:04d}", start, end))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_range, args.url, part, start, end, total): (start, end)
            for part, start, end in ranges
        }
        done = 0
        for future in as_completed(futures):
            future.result()
            done += 1
            print(f"completed ranges {done}/{len(ranges)}", flush=True)

    assembling = args.output.with_name(args.output.name + ".assembling")
    with assembling.open("wb") as destination:
        for part, _, _ in ranges:
            with part.open("rb") as source:
                shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
    if assembling.stat().st_size != total:
        raise RuntimeError(f"assembled size {assembling.stat().st_size} != {total}")
    os.replace(assembling, args.output)
    print(
        f"downloaded {args.output} {total} bytes sha256={sha256(args.output)} "
        f"etag={etag!r} last_modified={last_modified!r}"
    )


if __name__ == "__main__":
    main()
