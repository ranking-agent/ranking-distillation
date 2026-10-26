"""Every KGX edge's evidence, captured once, and its relationships priced.

Two LMDB stores, each written once:

- the **evidence store** (`capture_evidence`): one record per Translator
  edge -- unique on its canonicalized (subject, predicate, object,
  qualifiers) plus primary knowledge source, knowledge_level and
  agent_type -- holding the raw ingredients and metric bag the pricing
  reads. Duplicate KGX lines of one edge merge into one record (their
  metric bags union), so the pricing reads each edge once.
- the **priced store** (`price_relationships`): one record per
  relationship -- edges merged across source, level and agent -- whose
  admittance sums its edges' confidence prices.

Keys are 16-byte blake2b hashes (`evidence.edge_key`, `relationship_key`),
values orjson documents. The backbone reads the relationships in key order,
which is what fixes the order of the search graph's nodes.

The relationship sums are held in RAM, roughly 150 bytes per distinct
relationship. Every pass reads in chunks under short read transactions, and
a write that fills the map doubles it and retries.
"""

from __future__ import annotations

import datetime
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import lmdb
import orjson
from tqdm import tqdm

from .confidence import edge_admittance
from .evidence import (
    TOTAL_PUBS,
    canonicalize_edge,
    edge_key,
    extract_ingredients,
    extract_metrics,
    extract_primary_source,
    extract_qualifiers,
    merge_metrics,
    relationship_key,
)
from .kgx import iter_jsonl

DB_NAMES = ("edges", "relationships", "meta")

_NODE_PUB_FIELDS = ("num_publications", "omnicorp_article_count")

# A batch that still does not fit after this many doublings is not a sizing
# problem (one oversized value, or no disk left); fail instead of spinning.
_MAX_MAP_GROWTHS_PER_BATCH = 12


def _open(path: Path, *, readonly: bool, map_size: int = 64 * 1024**3):
    return lmdb.open(str(path), map_size=map_size, subdir=True,
                     readonly=readonly, lock=not readonly,
                     max_dbs=len(DB_NAMES), meminit=False)


def _get_record(env, db_name: str, key: bytes) -> dict | None:
    db = env.open_db(db_name.encode())
    with env.begin(db=db, buffers=True) as txn:
        raw = txn.get(key)
        return None if raw is None else orjson.loads(bytes(raw))


def capture_finished(path: Path) -> bool:
    """Whether `path` holds an evidence store whose KGX pass ran to the end."""
    if not (path / "data.mdb").exists():
        return False
    with _open(path, readonly=True) as env:
        return _get_record(env, "meta", b"pass1") is not None


def load_node_pubs(nodes_path: Path) -> dict[str, int]:
    """Node CURIE -> its own publication count, where the KGX gives one."""
    node_pubs: dict[str, int] = {}
    for rec in tqdm(iter_jsonl(nodes_path),
                    desc="nodes.jsonl: publication counts", unit=" nodes"):
        for field in _NODE_PUB_FIELDS:
            raw = rec.get(field)
            if raw is None:
                continue
            try:
                count = int(raw)
            except (TypeError, ValueError):
                continue
            if count > 0:
                node_pubs[rec["id"]] = count
            break
    return node_pubs


def _grow_map(env) -> None:
    """Double the environment's map size. No transaction may be open."""
    env.set_mapsize(env.info()["map_size"] * 2)


def _write_batch(env, db, items) -> None:
    """Write (key, value) pairs in one transaction, growing the map if it
    fills; an aborted transaction applies nothing, so the replay cannot
    double-write."""
    for _ in range(_MAX_MAP_GROWTHS_PER_BATCH + 1):
        txn = env.begin(write=True, db=db)
        try:
            for key, value in items:
                txn.put(key, value)
            txn.commit()
            return
        except lmdb.MapFullError:
            txn.abort()
            _grow_map(env)
        except BaseException:
            txn.abort()
            raise
    raise lmdb.MapFullError(f"a batch of {len(items)} records still does not "
                            "fit after the map grew; check free disk space")


def _iter_chunks(env, db, chunk_size: int, total: int | None, desc: str, unit: str):
    """Lists of (key, value) bytes in key order, each read under its own
    short transaction, so writes between chunks can reuse freed pages."""
    last_key: bytes | None = None
    with tqdm(total=total, desc=desc, unit=unit) as progress:
        while True:
            chunk = []
            with env.begin(db=db, buffers=True) as read_txn:
                cursor = read_txn.cursor()
                if last_key is None:
                    positioned = cursor.first()
                else:
                    positioned = cursor.set_range(last_key)
                    if positioned and bytes(cursor.key()) == last_key:
                        positioned = cursor.next()
                while positioned and len(chunk) < chunk_size:
                    chunk.append((bytes(cursor.key()), bytes(cursor.value())))
                    positioned = cursor.next()
            if not chunk:
                return
            last_key = chunk[-1][0]
            progress.update(len(chunk))
            yield chunk


def _flush_edge_batch(env, db, pending: dict) -> None:
    """Write one batch of edge records, merging each with any record of the
    same edge already in the store."""
    for _ in range(_MAX_MAP_GROWTHS_PER_BATCH + 1):
        txn = env.begin(write=True, db=db)
        try:
            merged = []
            for key, record in pending.items():
                raw = txn.get(key)
                if raw is None:
                    merged.append((key, orjson.dumps(record)))
                    continue
                existing = orjson.loads(raw)
                existing["n_records"] += record["n_records"]
                if "metrics" in record:
                    merge_metrics(existing.setdefault("metrics", {}),
                                  record["metrics"])
                merged.append((key, orjson.dumps(existing)))
            for key, value in merged:
                txn.put(key, value)
            txn.commit()
            return
        except lmdb.MapFullError:
            txn.abort()
            _grow_map(env)
        except BaseException:
            txn.abort()
            raise
    raise lmdb.MapFullError(f"an edge batch of {len(pending)} records still "
                            "does not fit after the map grew; check free disk "
                            "space")


def capture_evidence(edges_path: Path, nodes_path: Path, out: Path, *,
                     commit_every: int = 200_000) -> dict:
    """Stream a KGX edges.jsonl into the evidence store at `out`.

    A store at `out` that did not finish is removed and captured again: one
    cut off partway covers an arbitrary prefix of the file.
    """
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    env = _open(out, readonly=False)
    edges_db = env.open_db(b"edges")
    meta_db = env.open_db(b"meta")
    node_pubs = load_node_pubs(nodes_path)
    stats = {"edges_seen": 0, "edges_skipped_missing_fields": 0,
             "duplicate_edge_records": 0, "node_pub_entries": len(node_pubs)}

    pending: dict[bytes, dict] = {}
    for edge in tqdm(iter_jsonl(edges_path), desc="edges.jsonl -> evidence",
                     unit=" edges"):
        subject = edge.get("subject")
        predicate = edge.get("predicate")
        object_ = edge.get("object")
        if not (subject and predicate and object_):
            stats["edges_skipped_missing_fields"] += 1
            continue
        stats["edges_seen"] += 1
        s, p, o, qualifiers = canonicalize_edge(
            subject, predicate, object_, extract_qualifiers(edge))
        primary_source = extract_primary_source(edge)
        knowledge_level = edge.get("knowledge_level")
        agent_type = edge.get("agent_type")
        metrics = extract_metrics(edge)
        key = edge_key(s, p, o, qualifiers, primary_source, knowledge_level,
                       agent_type)
        previous = pending.get(key)
        if previous is not None:
            # a duplicate within this batch: merge now, no store read needed
            previous["n_records"] += 1
            if metrics:
                merge_metrics(previous.setdefault("metrics", {}), metrics)
            stats["duplicate_edge_records"] += 1
            continue
        record = {
            "subject": s,
            "predicate": p,
            "object": o,
            "qualifiers": [[name, value] for name, value in qualifiers],
            "primary_source": primary_source,
            "knowledge_level": knowledge_level,
            "agent_type": agent_type,
            "ingredients": extract_ingredients(edge, node_pubs),
            "n_records": 1,
        }
        if metrics:
            record["metrics"] = metrics
        pending[key] = record
        if len(pending) >= commit_every:
            _flush_edge_batch(env, edges_db, pending)
            pending = {}
    if pending:
        _flush_edge_batch(env, edges_db, pending)
    with env.begin(db=edges_db) as txn:
        stats["edge_entries"] = txn.stat()["entries"]
    #  Written last: only a capture that reached the end of the file has it.
    _write_batch(env, meta_db, [(b"pass1", orjson.dumps({"stats": stats}))])
    env.sync()
    env.close()
    return stats


def _price_edge(rec, profile, confidence, shared_counts, node_pubs) -> float:
    """One edge's admittance under the confidence pricing: nothing for a NOT
    annotation, and omnicorp's co-occurrence for its pair folded in as the
    worker's coverage formula reads it."""
    negated = (rec.get("metrics") or {}).get("negated")
    if negated is not None and not isinstance(negated, bool):
        raise ValueError(f"negated is {negated!r} on {rec['subject']} "
                         f"{rec['predicate']} {rec['object']}; expected a boolean")
    if negated:
        return 0.0
    ingredients = dict(rec.get("ingredients") or {})
    shared = shared_counts(rec["subject"], rec["object"])
    if shared:
        cov = (shared / TOTAL_PUBS) - (
            node_pubs.get(rec["subject"], 0) / TOTAL_PUBS
        ) * (node_pubs.get(rec["object"], 0) / TOTAL_PUBS)
        ingredients["literature_coocurrence"] = max(cov, 0.0) * TOTAL_PUBS
    return edge_admittance(
        ingredients, rec.get("metrics") or {}, rec["primary_source"],
        profile, confidence,
        knowledge_level=rec.get("knowledge_level"),
        agent_type=rec.get("agent_type"),
        predicate=rec["predicate"], subject=rec["subject"], object_=rec["object"],
    )


def _relationship_key_of(rec) -> bytes:
    qualifiers = tuple((n, v) for n, v in rec["qualifiers"])
    return relationship_key(rec["subject"], rec["predicate"], rec["object"],
                            qualifiers)


def price_relationships(
    evidence: Path, out: Path, profile: Mapping[str, Any],
    confidence: Mapping[str, Any],
    shared_counts: Callable[[str, str], int | None],
    node_pubs: Mapping[str, int], *, commit_every: int = 200_000,
) -> dict:
    """Price every edge of the evidence store and write each relationship's
    summed admittance to the priced store at `out`."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    with _open(evidence, readonly=True) as src:
        src_edges = src.open_db(b"edges", create=False)
        total = (_get_record(src, "meta", b"pass1") or {})["stats"]["edge_entries"]
        env = _open(out, readonly=False)
        relationships_db = env.open_db(b"relationships")
        meta_db = env.open_db(b"meta")

        # pass 1: each edge's admittance, summed per relationship
        sums: dict[bytes, list] = {}
        for chunk in _iter_chunks(src, src_edges, commit_every, total,
                                  "evidence -> priced sums", " edges"):
            for _, raw in chunk:
                rec = orjson.loads(raw)
                admittance = _price_edge(rec, profile, confidence,
                                         shared_counts, node_pubs)
                key = _relationship_key_of(rec)
                acc = sums.get(key)
                if acc is None:
                    sums[key] = [admittance, 1]
                else:
                    acc[0] += admittance
                    acc[1] += 1
        n_relationships = len(sums)

        # pass 2: one record per relationship, at its first edge
        for chunk in _iter_chunks(src, src_edges, commit_every, total,
                                  "evidence -> relationships", " edges"):
            batch = []
            for _, raw in chunk:
                rec = orjson.loads(raw)
                key = _relationship_key_of(rec)
                acc = sums.pop(key, None)
                if acc is None:
                    continue  # written from an earlier edge of the same one
                admittance, n_edges = acc
                batch.append((key, orjson.dumps({
                    "subject": rec["subject"],
                    "predicate": rec["predicate"],
                    "object": rec["object"],
                    "qualifiers": rec["qualifiers"],
                    "n_edges": n_edges,
                    "admittance": admittance,
                })))
            if batch:
                _write_batch(env, relationships_db, batch)
    stats = {"edges_priced": total, "relationship_entries": n_relationships,
             "finished_utc": datetime.datetime.now(datetime.UTC).isoformat()}
    _write_batch(env, meta_db, [(b"build", orjson.dumps({"stats": stats}))])
    env.sync()
    env.close()
    return stats


def iter_relationships(path: Path):
    """`(total, records)`: every priced relationship, in key order."""
    env = _open(path, readonly=True)
    total = (_get_record(env, "meta", b"build") or {})["stats"]["relationship_entries"]

    def records():
        db = env.open_db(b"relationships", create=False)
        with env.begin(db=db, buffers=True) as txn:
            for _, value in txn.cursor():
                yield orjson.loads(bytes(value))
        env.close()
    return total, records()
