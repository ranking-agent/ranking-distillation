"""Vectors on disk: a table keyed by string, and one keyed by node pair.

An `EmbeddingCache` is a directory:

- `vectors.npy`: a memmap of shape (N, D);
- `keys.jsonl`: one `{"key": ..., "row": int}` per line;
- `meta.json`: `{"dim": D, "dtype": ..., "count": N}`.

A `PairTable` holds a vector for every node pair of a graph, tens of
millions, as a sorted array of each pair's 64-bit hash beside the vectors,
both memory-mapped: it opens at once and a lookup is one binary search.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import orjson


def pair_key(a: str, b: str) -> str:
    """Direction-agnostic key for a node pair."""
    return "|".join(sorted((a, b)))


def pair_hash(key: str) -> int:
    """The 64-bit hash of a `pair_key` that a `PairTable` is sorted by."""
    return int.from_bytes(
        hashlib.blake2b(key.encode(), digest_size=8).digest(), "little")


#  meta.json's "kind" for a PairTable; an EmbeddingCache's has none
PAIR_TABLE_KIND = "pair_table"


@dataclass
class PairTable:
    """Vectors by node pair, read as an `EmbeddingCache` keyed by
    `pair_key` is: `key in table`, `table[key]` and `dim`. Two pairs sharing
    a hash are refused when it is written."""

    dir: Path
    hashes: np.ndarray
    vectors: np.ndarray
    dim: int

    @classmethod
    def create(cls, dir: str | Path, hashes: np.ndarray,
               vectors: np.ndarray) -> PairTable:
        """Write `vectors[i]` under `hashes[i]`, in any order."""
        hashes = np.asarray(hashes, dtype=np.uint64)
        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or len(vectors) != len(hashes):
            raise ValueError(f"{len(hashes)} hashes for vectors of shape "
                             f"{vectors.shape}")
        order = np.argsort(hashes, kind="stable")
        hashes, vectors = hashes[order], vectors[order]
        if len(hashes) > 1 and bool((hashes[1:] == hashes[:-1]).any()):
            raise ValueError(
                f"{int((hashes[1:] == hashes[:-1]).sum())} pair(s) share a "
                "hash with another: one pair given twice, or two whose 64-bit "
                "hashes collide")
        d = Path(dir)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "hashes.npy", hashes)
        np.save(d / "vectors.npy", vectors)
        (d / "meta.json").write_text(json.dumps({
            "kind": PAIR_TABLE_KIND, "dim": int(vectors.shape[1]),
            "dtype": str(vectors.dtype), "count": len(hashes),
            "hash": "blake2b-64"}))
        return cls.load(d)

    @classmethod
    def load(cls, dir: str | Path) -> PairTable:
        d = Path(dir)
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("kind") != PAIR_TABLE_KIND:
            raise ValueError(f"{d} is not a pair table")
        return cls(dir=d, hashes=np.load(d / "hashes.npy", mmap_mode="r"),
                   vectors=np.load(d / "vectors.npy", mmap_mode="r"),
                   dim=meta["dim"])

    def _row(self, key: str) -> int | None:
        h = np.uint64(pair_hash(key))
        i = int(np.searchsorted(self.hashes, h))
        if i < len(self.hashes) and self.hashes[i] == h:
            return i
        return None

    def __getitem__(self, key: str) -> np.ndarray:
        row = self._row(key)
        if row is None:
            raise KeyError(key)
        return self.vectors[row]

    def __contains__(self, key: str) -> bool:
        return self._row(key) is not None

    def __len__(self) -> int:
        return len(self.hashes)


@dataclass
class EmbeddingCache:
    dir: Path
    vectors: np.memmap
    key_to_row: dict[str, int]
    dim: int

    @classmethod
    def create(cls, dir: str | Path, keys: list[str], dim: int,
               dtype: str = "float16") -> EmbeddingCache:
        """Allocate a cache on disk; the caller fills `vectors[row]`, then
        calls `flush()`."""
        d = Path(dir)
        d.mkdir(parents=True, exist_ok=True)
        n = len(keys)
        vectors = np.memmap(d / "vectors.npy", dtype=dtype, mode="w+",
                            shape=(n, dim))
        key_to_row = {k: i for i, k in enumerate(keys)}
        with open(d / "keys.jsonl", "wb") as f:
            for i, k in enumerate(keys):
                f.write(orjson.dumps({"key": k, "row": i}))
                f.write(b"\n")
        (d / "meta.json").write_text(json.dumps({"dim": dim, "dtype": dtype,
                                                 "count": n}))
        return cls(dir=d, vectors=vectors, key_to_row=key_to_row, dim=dim)

    @classmethod
    def load(cls, dir: str | Path) -> EmbeddingCache:
        d = Path(dir)
        meta = json.loads((d / "meta.json").read_text())
        vectors = np.memmap(d / "vectors.npy", dtype=meta["dtype"], mode="r",
                            shape=(meta["count"], meta["dim"]))
        key_to_row: dict[str, int] = {}
        with open(d / "keys.jsonl", "rb") as f:
            for line in f:
                rec = orjson.loads(line)
                key_to_row[rec["key"]] = rec["row"]
        return cls(dir=d, vectors=vectors, key_to_row=key_to_row, dim=meta["dim"])

    def flush(self) -> None:
        self.vectors.flush()

    def __getitem__(self, key: str) -> np.ndarray:
        return self.vectors[self.key_to_row[key]]

    def __contains__(self, key: str) -> bool:
        return key in self.key_to_row

    def __len__(self) -> int:
        return len(self.key_to_row)
