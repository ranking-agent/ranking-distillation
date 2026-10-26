"""Reading a KGX release: its jsonl files and its release metadata."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import orjson

RELEASE_METADATA = "release-metadata.json"
NODES = "nodes.jsonl"
EDGES = "edges.jsonl"


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """One dict per line of a jsonl file.

    A malformed line stops the read, naming the file and line, and whether
    the line looks cut off: the usual cause is a truncated download.
    """
    with open(path, "rb") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                truncated = not line.endswith(b"\n")
                raise ValueError(
                    f"{path} line {number:,} is not valid JSON: {exc}"
                    + ("  (the line has no trailing newline, so the file is "
                       "very likely truncated -- check the download)"
                       if truncated else "")
                ) from exc


def release_metadata(kgx_dir: Path) -> dict:
    """`release-metadata.json` of a release directory."""
    return json.loads((kgx_dir / RELEASE_METADATA).read_text())


def build_day(metadata: dict) -> str:
    """The release's build date as `YYYY_MM_DD`: what each output directory
    is named by."""
    built = datetime.fromisoformat(metadata["build_date"])
    return built.strftime("%Y_%m_%d")


def biolink_version(metadata: dict) -> str:
    """The biolink model release the graph was built against."""
    version = metadata.get("biolink_version")
    if not version:
        raise ValueError(f"{RELEASE_METADATA} names no biolink_version")
    return str(version)
