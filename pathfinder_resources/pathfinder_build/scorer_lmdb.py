"""The LMDB shepherd's score_paths worker scores from, in the layout the
deployed worker reads (version 1).

The worker has no SAPBERT, graph or gandalf call per path: everything it
reads is in one LMDB --

- the model as TorchScript, and its config: the length scale, the scalars
  it reads, whether it reads each context table, the embedding width, and
  the biolink model the paths are read by;
- each node's SAPBERT vector by CURIE, each category's by CURIE, and each
  hop phrase's, as SAPBERT gives them (nodes and categories in float16);
- the node context table by CURIE, and the edge context table over every
  pair by the pair's 64-bit hash.

A key longer than LMDB allows is stored by its SHA-256 digest, as the worker
looks it up.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lmdb
import numpy as np
from tqdm import tqdm

from .tables import EmbeddingCache, PairTable

NODE, CATEGORY, PHRASE = "node", "category", "phrase"
NODE_CONTEXT, EDGE_CONTEXT = "node_context", "edge_context"
# Nodes and categories keep the SAPBERT caches' float16
DTYPES = {
    NODE: np.float16,
    CATEGORY: np.float16,
    PHRASE: np.float32,
    NODE_CONTEXT: np.float32,
    EDGE_CONTEXT: np.float32,
}
MODEL_KEY = b"reranker:model.pt"
CONFIG_KEY = b"reranker:config.json"
# LMDB refuses keys over 511 bytes
MAX_KEY_BYTES = 480
#  Puts per write transaction
WRITE_BATCH = 50_000


def lmdb_key(kind: str, text: str) -> bytes:
    key = f"{kind}:{text}".encode()
    if len(key) > MAX_KEY_BYTES:
        key = f"{kind}#".encode() + hashlib.sha256(key).hexdigest().encode()
    return key


def read_model(path: Path) -> tuple[bytes, dict]:
    """`(TorchScript bytes, config)` of the reranker an LMDB holds."""
    env = lmdb.open(str(path), readonly=True, lock=False, subdir=True)
    with env.begin() as txn:
        blob, config = txn.get(MODEL_KEY), txn.get(CONFIG_KEY)
    env.close()
    if blob is None or config is None:
        raise SystemExit(f"{path} holds no reranker")
    return blob, json.loads(config)


def _write(env, items, total: int, desc: str) -> None:
    """Put `(key, value bytes)` pairs, a batch per transaction."""
    txn = env.begin(write=True)
    for n, (key, value) in enumerate(tqdm(items, total=total, desc=desc,
                                          unit=" key"), start=1):
        txn.put(key, value)
        if n % WRITE_BATCH == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()


def write_lmdb(out: Path, model: bytes, config: dict, *,
               names: dict[str, str], categories: dict[str, str],
               phrases: set[str], embeddings: EmbeddingCache,
               category_texts: dict[str, str],
               node_context: EmbeddingCache | None,
               edge_context: PairTable | None,
               map_size: int = 1 << 40) -> dict:
    """Write the LMDB at `out`; returns how many of each it holds.

    `names` gives each node's text by CURIE and `category_texts` each
    category's; `embeddings` holds the vectors of every text.
    """
    env = lmdb.open(str(out), map_size=map_size, subdir=True)
    with env.begin(write=True) as txn:
        txn.put(MODEL_KEY, model)
        txn.put(CONFIG_KEY, json.dumps(config).encode())

    def vectors(kind, items):
        dtype = DTYPES[kind]
        for key, text in items:
            yield (lmdb_key(kind, key),
                   np.ascontiguousarray(embeddings[text], dtype=dtype).tobytes())

    _write(env, vectors(NODE, sorted(names.items())), len(names), NODE)
    held_categories = sorted(set(categories.values()))
    _write(env, vectors(CATEGORY, ((c, category_texts[c])
                                   for c in held_categories)),
           len(held_categories), CATEGORY)
    _write(env, vectors(PHRASE, ((p, p) for p in sorted(phrases))),
           len(phrases), PHRASE)
    held = {"nodes": len(names), "categories": len(held_categories),
            "phrases": len(phrases)}
    if node_context is not None:
        _write(env, ((lmdb_key(NODE_CONTEXT, c),
                      np.ascontiguousarray(node_context[c], np.float32).tobytes())
                     for c in sorted(node_context.key_to_row)),
               len(node_context), NODE_CONTEXT)
        held["node_context"] = len(node_context)
    if edge_context is not None:
        _write(env, ((f"{EDGE_CONTEXT}#".encode() + int(h).to_bytes(8, "little"),
                      np.ascontiguousarray(v, np.float32).tobytes())
                     for h, v in zip(edge_context.hashes, edge_context.vectors)),
               len(edge_context), EDGE_CONTEXT)
        held["edge_context"] = len(edge_context)
    env.sync()
    env.close()
    return held
