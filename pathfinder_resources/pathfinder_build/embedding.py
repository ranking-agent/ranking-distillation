"""SAPBERT embeddings of the texts the reranker reads, cached by the text.

Every node name, category and hop phrase is embedded as SAPBERT was
trained: its [CLS] output, L2-normalized, at most 32 tokens, kept as float16.
The cache is an `EmbeddingCache` keyed by the text itself, so a cache built
for one release serves the next exactly: a node renamed is a new text, and
an unchanged one is not embedded again.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .phrases import PHRASE_MAX_LENGTH
from .tables import EmbeddingCache

SAPBERT_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
BATCH_SIZE = 256
#  Texts embedded and written at once
WRITE_CHUNK = 65_536

Embed = Callable[[list[str]], np.ndarray]


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sapbert(device: str | None = None) -> Embed:
    """`embed(texts) -> (len(texts), 768)` float32, SAPBERT loaded on first
    use: a cache that already holds every text never needs it."""
    loaded = []

    def embed(texts: list[str]) -> np.ndarray:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not loaded:
            where = device or pick_device()
            model = AutoModel.from_pretrained(SAPBERT_NAME).to(where)
            model.eval()
            loaded.append((AutoTokenizer.from_pretrained(SAPBERT_NAME), model,
                           where))
        tokenizer, model, where = loaded[0]
        out = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
        with torch.inference_mode():
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]
                enc = tokenizer(batch, padding=True, truncation=True,
                                max_length=PHRASE_MAX_LENGTH,
                                return_tensors="pt").to(where)
                vecs = model(**enc).last_hidden_state[:, 0]
                vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
                out[i : i + len(batch)] = vecs.float().cpu().numpy()
        return out

    return embed


def ensure_cache(path: Path, texts: Iterable[str], embed: Embed,
                 provided: Path | None = None) -> EmbeddingCache:
    """A cache at `path` holding every one of `texts`.

    One already there that holds them all is used as it is. Otherwise it is
    written again: each text from it or from `provided` (a cache an earlier
    build wrote) where either holds it, and embedded where neither does.
    """
    wanted = sorted(set(texts))
    have = EmbeddingCache.load(path) if (path / "meta.json").exists() else None
    if have is not None and all(t in have for t in wanted):
        return have
    sources = [c for c in (have, EmbeddingCache.load(provided) if provided
                           else None) if c is not None]
    lacking = [t for t in wanted if not any(t in c for c in sources)]
    print(f"{len(wanted):,} texts: {len(wanted) - len(lacking):,} cached, "
          f"{len(lacking):,} for SAPBERT")
    first = embed(lacking[:WRITE_CHUNK]) if lacking else None
    dims = {c.dim for c in sources} | ({first.shape[1]} if lacking else set())
    if len(dims) != 1:
        raise ValueError(f"the cached and new vectors differ in width: {dims}")
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    cache = EmbeddingCache.create(partial, wanted, dims.pop(), dtype="float16")
    row_of = cache.key_to_row
    for source in sources:
        for text, row in source.key_to_row.items():
            if text in row_of:
                cache.vectors[row_of[text]] = source.vectors[row]
    for i in tqdm(range(0, len(lacking), WRITE_CHUNK), desc="embedding",
                  unit=" chunk"):
        chunk = lacking[i : i + WRITE_CHUNK]
        vectors = first if i == 0 else embed(chunk)
        cache.vectors[[row_of[t] for t in chunk]] = vectors
    cache.flush()
    del cache
    if path.exists():
        shutil.rmtree(path)
    partial.replace(path)
    return EmbeddingCache.load(path)
