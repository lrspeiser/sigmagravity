#!/usr/bin/env python3
"""Mirror the frozen CCF root with size-checked concurrent downloads."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse


BASE = "https://heasarc.gsfc.nasa.gov/FTP/xmm/data/CCF/"


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def urlopen(url: str, *, method: str = "GET"):
    return urllib.request.urlopen(urllib.request.Request(url, method=method), timeout=120)


def remote_size(url: str) -> int:
    with urlopen(url, method="HEAD") as response:
        return int(response.headers["Content-Length"])


def download_one(url: str, target: Path) -> tuple[str, int, str]:
    for attempt in range(5):
        try:
            expected = remote_size(url)
            if target.exists() and target.stat().st_size == expected:
                return target.name, expected, "existing"
            temporary = target.with_suffix(target.suffix + ".part")
            with urlopen(url) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if temporary.stat().st_size != expected:
                raise RuntimeError(
                    f"{target.name}: {temporary.stat().st_size} bytes != {expected}"
                )
            os.replace(temporary, target)
            return target.name, expected, "downloaded"
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with urlopen(BASE) as response:
        listing = response.read().decode("utf-8", errors="replace")
    parser_html = Links()
    parser_html.feed(listing)
    names = sorted(
        {
            Path(urlparse(href).path).name
            for href in parser_html.hrefs
            if urlparse(href).path.endswith(".CCF")
        }
    )
    if len(names) < 1000:
        raise RuntimeError(f"CCF index exposed only {len(names)} constituents")
    print(f"frozen index exposes {len(names)} CCF constituents", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, urljoin(BASE, name), args.output / name): name
            for name in names
        }
        downloaded = 0
        existing = 0
        for completed, future in enumerate(as_completed(futures), start=1):
            _, _, action = future.result()
            downloaded += action == "downloaded"
            existing += action == "existing"
            if completed % 100 == 0 or completed == len(futures):
                print(
                    f"verified {completed}/{len(futures)} downloaded={downloaded} existing={existing}",
                    flush=True,
                )

    manifest = args.output / "CCF_MANIFEST.sha256"
    with manifest.open("w", encoding="ascii", newline="\n") as stream:
        for name in names:
            stream.write(f"{sha256(args.output / name)}  {name}\n")
    total_bytes = sum((args.output / name).stat().st_size for name in names)
    print(
        f"snapshot files={len(names)} bytes={total_bytes} "
        f"manifest_sha256={sha256(manifest)}"
    )


if __name__ == "__main__":
    main()
