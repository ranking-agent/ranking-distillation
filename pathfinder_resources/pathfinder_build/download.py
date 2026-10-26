"""Fetching and unpacking the inputs that come as URLs."""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

CHUNK = 1 << 20


def download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`, through a partial file renamed into place
    only once the whole body has arrived."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    with urllib.request.urlopen(url) as resp:
        total = resp.headers.get("Content-Length")
        with open(partial, "wb") as out, tqdm(
                total=int(total) if total else None, unit="B", unit_scale=True,
                desc=f"downloading {dest.name}") as progress:
            while chunk := resp.read(CHUNK):
                out.write(chunk)
                progress.update(len(chunk))
    if total and partial.stat().st_size != int(total):
        raise OSError(f"{url} sent {partial.stat().st_size:,} of "
                      f"{int(total):,} bytes")
    partial.replace(dest)


def extract(archive: Path, dest: Path) -> None:
    """Unpack a `.tar.gz` or `.tar.zst` into `dest`, streaming, and refusing
    any member that would land outside it."""
    dest.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zst"):
        import zstandard

        with open(archive, "rb") as raw:
            reader = zstandard.ZstdDecompressor().stream_reader(raw)
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                tar.extractall(dest, filter="data")
        return
    with tarfile.open(archive, "r|gz") as tar:
        tar.extractall(dest, filter="data")
