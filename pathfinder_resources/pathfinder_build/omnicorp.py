"""Omnicorp literature co-occurrence, joined onto the edges the graph has.

The Translator KG ships no literature co-occurrence edges, but co-occurrence
is a property of a node pair, not of an edge record, so omnicorp's counts
fold into the pricing without adding an edge. Two LMDBs, both in omnicorp's
tarball (`omnicorp_lmdb/` in omnicorp_data_05_05_2026.tar.gz):

    curies.lmdb         CURIE -> {"index": int, "pmc": int}
    shared_counts.lmdb  "<index>_<index>" -> shared publication count, ASCII

`pmc` is each node's own publication count, which the KGX carries none of:
without it the coverage correction could not discount a pair for both
endpoints merely being well studied.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import lmdb
import orjson

SHARED_COUNTS = "shared_counts.lmdb"
CURIES = "curies.lmdb"


def omnicorp_files(directory: Path) -> tuple[Path, Path]:
    """`(shared_counts.lmdb, curies.lmdb)` under `directory`, at its top or
    one directory down, as the tarball unpacks them."""
    found = {}
    for name in (SHARED_COUNTS, CURIES):
        matches = sorted([*directory.glob(name), *directory.glob(f"*/{name}")])
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one {name} in {directory} or a directory inside "
                f"it, found {len(matches)}")
        found[name] = matches[0]
    return found[SHARED_COUNTS], found[CURIES]


def _open(path: Path) -> lmdb.Environment:
    return lmdb.open(str(path), readonly=True, subdir=path.is_dir(),
                     lock=False, meminit=False)


def load_curies(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    """(CURIE -> omnicorp index, CURIE -> that node's publication count)."""
    env = _open(path)
    index: dict[str, int] = {}
    pubs: dict[str, int] = {}
    with env.begin() as txn:
        for key, raw in txn.cursor():
            curie = key.decode("utf-8")
            rec = orjson.loads(raw)
            index[curie] = rec["index"]
            if rec.get("pmc"):
                pubs[curie] = rec["pmc"]
    env.close()
    return index, pubs


def detect_key_order(path: Path, sample: int = 100_000) -> str:
    """Whether shared_counts stores a pair once ("ascending") or under both
    orderings ("both"), from a sample of its keys."""
    env = _open(path)
    descending = 0
    seen = 0
    with env.begin() as txn:
        for key, _ in txn.cursor():
            left, _, right = key.decode("utf-8").partition("_")
            if int(left) > int(right):
                descending += 1
            seen += 1
            if seen >= sample:
                break
    env.close()
    if not seen:
        raise ValueError(f"{path} is empty")
    return "both" if descending else "ascending"


def open_shared_counts(
    shared_counts_path: Path, curies_path: Path,
) -> tuple[Callable[[str, str], int | None], dict[str, int], dict]:
    """`(lookup, node publication counts, stats)`. The lookup takes two
    CURIEs in either order and returns their shared publication count, or
    None when omnicorp does not know one of them or the pair never
    co-occurs. Its read transaction stays open for the caller's whole pass:
    it is asked once per edge, tens of millions of times."""
    index, pubs = load_curies(curies_path)
    order = detect_key_order(shared_counts_path)
    env = _open(shared_counts_path)
    txn = env.begin()
    stats = {"curies": len(index), "curies_with_publications": len(pubs),
             "key_order": order, "lookups": 0, "hits": 0,
             "unresolved_curie": 0}

    def lookup(a: str, b: str) -> int | None:
        stats["lookups"] += 1
        ia, ib = index.get(a), index.get(b)
        if ia is None or ib is None:
            stats["unresolved_curie"] += 1
            return None
        lo, hi = (ia, ib) if ia < ib else (ib, ia)
        raw = txn.get(f"{lo}_{hi}".encode())
        if raw is None and order == "both":
            raw = txn.get(f"{hi}_{lo}".encode())
        if raw is None:
            return None
        stats["hits"] += 1
        return int(raw)

    return lookup, pubs, stats
