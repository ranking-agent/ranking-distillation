"""Retraining the reranker on graded paths, as the deployed one was trained.

The graded paths are labelled parquets: per row a path (`path_curies`,
`categories`, one predicate list per hop), its `tier` (1 is best) and a soft
label (`tier_weights`), its query's endpoints (`source_curie`,
`target_curie`) and the rubric it was graded under.

The recipe:

1. Hold out about 15% of the queries that have expected nodes in the
   graded queries' TSVs, in groups that share an endpoint or expect one
   another's endpoint, so no held-out query leaks into training. The
   held-out queries choose the checkpoint (NDCG@10) and stop training.
2. Train three teachers (4 layers of width 256, seeds 0, 1000 and 2000) by
   RankNet on pairs of a top-tier path against one below it within a query,
   each pair's target its soft labels' chance of that order.
3. Train the student (4 layers of width 128, seed 0) on the same pairs, each
   pair's target the logistic of the teachers' mean score gap: one model
   that ranks as the three do together, at one model's cost.

Each training is checkpointed every evaluation and resumes from there.
"""

from __future__ import annotations

import ast
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .phrases import category_phrase, row_hop_phrase
from .reranker import (
    CHECKPOINT,
    N_HOPS_SCALE,
    SCALAR_NAMES,
    PathSequenceScorer,
    load_scorer,
    save_scorer,
)
from .tables import EmbeddingCache, PairTable, pair_key

HOP_COLUMNS = [
    "first_hop_predicates",
    "second_hop_predicates",
    "third_hop_predicates",
    "fourth_hop_predicates",
    "fifth_hop_predicates",
]
REQUIRED_PARQUET_COLUMNS = {"path", "categories", "path_curies"}
ARROW_SEP = " --> "

MIN_TIER, MAX_TIER = 1, 5
#  The tiers a consumer keeps: the top of top-vs-rest.
TOP_TIERS = frozenset({1, 2})
#  Tiers per rubric; a row that names none was graded under "upstream".
RUBRIC_TIERS = {"gates": 5, "upstream": 4}
UNRECORDED_RUBRIC = "upstream"
WAYPOINT_SEPARATOR = "|"

#  A pair is drawn only when one path beats the other with at least this much
#  more than an even chance.
MIN_PAIR_MARGIN = 0.05

#  The recipe
SPLIT_SEED = 0
EVAL_FRAC = 0.15
TEACHER_SEEDS = (0, 1000, 2000)
STUDENT_SEED = 0
TEACHER_D_MODEL = 256
STUDENT_D_MODEL = 128


@dataclass
class Settings:
    """One training's settings: the deployed recipe's unless given."""

    d_model: int
    seed: int
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    head_hidden: int = 256
    max_positions: int = 64
    lr: float = 1e-4
    batch_size: int = 256
    max_steps: int = 20_000
    eval_every: int = 250
    patience: int = 10


@dataclass
class LabelledRow:
    """One graded path."""

    n_hops: int
    node_curies: list[str]
    node_categories: list[str]
    hop_phrases: list[str]
    tier: int
    source: str
    tier_weights: tuple[float, ...] | None = None


# --- the graded paths -------------------------------------------------------------


def _parse_predicate_list(raw) -> list[str]:
    """A hop's predicates, as a list or the string of a Python list."""
    if raw is None:
        return []
    if not isinstance(raw, (str, bytes)):
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        try:
            return [str(p) for p in raw]
        except TypeError:
            raise ValueError(f"unrecognised predicate column type: {type(raw).__name__}")
    raw = raw.strip() if isinstance(raw, str) else raw.decode().strip()
    if not raw:
        return []
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"could not parse predicate list {raw!r}: {e}") from e
    if not isinstance(value, list):
        raise ValueError(f"predicate column did not contain a list: {raw!r}")
    return [str(p) for p in value]


def _parquet_rows(path: Path):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if not REQUIRED_PARQUET_COLUMNS.issubset(set(table.column_names)):
        return
    yield from table.to_pylist()


def _rubric_of(row) -> str:
    rubric = row.get("rubric") or UNRECORDED_RUBRIC
    if rubric not in RUBRIC_TIERS:
        raise ValueError(f"unknown rubric {rubric!r}; expected one of "
                         f"{sorted(RUBRIC_TIERS)}")
    return rubric


def _checked_tier_weights(raw, rubric: str) -> tuple[float, ...]:
    weights = tuple(float(w) for w in raw)
    n_tiers = RUBRIC_TIERS[rubric]
    if (len(weights) != n_tiers or any(w < 0 for w in weights)
            or not math.isclose(sum(weights), 1.0, abs_tol=1e-6)):
        raise ValueError(f"tier_weights {list(weights)} is not a weight for "
                         f"each of the {n_tiers} tiers the {rubric} rubric "
                         "grades, summing to one")
    return weights


def _query_id(row: dict, parquet_path: Path) -> str:
    source, target = row.get("source_curie"), row.get("target_curie")
    if isinstance(source, str) and isinstance(target, str):
        return f"{source} -> {target}"
    return parquet_path.name


def graded_rows(graded: Path, names: dict[str, str]) -> list[LabelledRow]:
    """Every usable graded path in the parquets of `graded`, in file and row
    order; a path through a node the graph does not hold is left out."""
    files = sorted(p for p in graded.iterdir()
                   if p.is_file() and p.suffix.lower() == ".parquet")
    if not files:
        raise SystemExit(f"no *.parquet files in {graded}")
    rows: list[LabelledRow] = []
    skipped_not_in_graph = skipped_bad_row = 0
    rubrics_by_query: dict[str, Counter] = defaultdict(Counter)
    for path in tqdm(files, desc="graded paths"):
        for row in _parquet_rows(path):
            curies_raw, cats_raw, tier_raw = (row.get("path_curies"),
                                              row.get("categories"), row.get("tier"))
            if (not isinstance(curies_raw, str) or not isinstance(cats_raw, str)
                    or tier_raw is None):
                skipped_bad_row += 1
                continue
            try:
                tier = int(tier_raw)
            except (TypeError, ValueError):
                skipped_bad_row += 1
                continue
            curies, cats = curies_raw.split(ARROW_SEP), cats_raw.split(ARROW_SEP)
            n_hops = len(curies) - 1
            if len(curies) != len(cats) or not 1 <= n_hops <= len(HOP_COLUMNS):
                skipped_bad_row += 1
                continue
            phrases = []
            for h in range(n_hops):
                try:
                    preds = _parse_predicate_list(row.get(HOP_COLUMNS[h]))
                except ValueError:
                    preds = []
                if not preds:
                    break
                phrases.append(row_hop_phrase(preds))
            if len(phrases) != n_hops:
                skipped_bad_row += 1
                continue
            if any(c not in names for c in curies):
                skipped_not_in_graph += 1
                continue
            weights = row.get("tier_weights")
            rubric = _rubric_of(row)
            source = _query_id(row, path)
            rubrics_by_query[source][rubric] += 1
            rows.append(LabelledRow(
                n_hops=n_hops, node_curies=curies, node_categories=cats,
                hop_phrases=phrases, tier=tier, source=source,
                tier_weights=(None if weights is None
                              else _checked_tier_weights(weights, rubric))))
    print(f"  {len(rows):,} graded paths; left out {skipped_not_in_graph:,} "
          f"through a node not in the graph, {skipped_bad_row:,} malformed")
    mixed = sorted(q for q, found in rubrics_by_query.items() if len(found) > 1)
    if mixed:
        raise SystemExit(f"{len(mixed):,} quer(y/ies) hold rows graded under "
                         f"different rubrics, e.g. {mixed[0]}; their tiers are "
                         "on different scales")
    if not rows:
        raise SystemExit("no usable graded paths")
    return rows


def row_texts(rows: Sequence[LabelledRow], names: dict[str, str]) -> set[str]:
    """Every text the rows' features embed."""
    return ({names[c] for r in rows for c in r.node_curies}
            | {category_phrase(c) for r in rows for c in r.node_categories}
            | {p for r in rows for p in r.hop_phrases})


def expected_nodes(paths: Sequence[Path]) -> dict[str, set[str]]:
    """`{"SOURCE -> TARGET": expected node CURIEs}` from query TSVs (source,
    target, '|'-separated expected nodes; '#' comments), or a JSON list of
    `[source, target, expected...]`; a query in several files keeps the
    union."""
    queries: dict[tuple[str, str], set[str]] = {}
    for path in paths:
        text = Path(path).read_text().strip()
        if text.startswith("["):
            pairs = [(str(item[0]).strip(), str(item[1]).strip(),
                      tuple(str(w).strip() for w in (item[2:] or [])))
                     for item in json.loads(text)]
        else:
            pairs = []
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", "\t").split("\t")
                if len(parts) < 2:
                    raise SystemExit(f"malformed query line: {line!r}")
                waypoints = tuple(w for w in (
                    parts[2].strip() if len(parts) > 2 else "").split(
                        WAYPOINT_SEPARATOR) if w)
                pairs.append((parts[0].strip(), parts[1].strip(), waypoints))
        for source, target, waypoints in pairs:
            queries.setdefault((source, target), set()).update(waypoints)
    if not queries:
        raise SystemExit("no query pairs found in "
                         + ", ".join(str(p) for p in paths))
    return {f"{s} -> {t}": nodes for (s, t), nodes in queries.items()}


# --- the held-out queries ----------------------------------------------------------


def _query_endpoints(query: str) -> tuple[str, ...]:
    source, arrow, target = query.partition(" -> ")
    return (source, target) if arrow else (query,)


def query_components(queries, expected: dict[str, set[str]] | None
                     ) -> list[list[str]]:
    """The queries grouped so no group leaks into another across the split:
    two belong together when they share an endpoint, or when an expected
    node of one is an endpoint of the other."""
    queries = sorted(set(queries))
    parent = {q: q for q in queries}

    def find(q: str) -> str:
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    by_endpoint: dict[str, list[str]] = defaultdict(list)
    for q in queries:
        for endpoint in _query_endpoints(q):
            by_endpoint[endpoint].append(q)
    for group in by_endpoint.values():
        for q in group[1:]:
            union(group[0], q)
    for q in queries:
        for node in (expected or {}).get(q, ()):
            for other in by_endpoint.get(node, ()):
                union(q, other)

    groups: dict[str, list[str]] = defaultdict(list)
    for q in queries:
        groups[find(q)].append(q)
    return sorted(groups.values(), key=lambda g: g[0])


def split_queries(rows: list[LabelledRow], expected: dict[str, set[str]] | None
                  ) -> tuple[list[LabelledRow], list[LabelledRow], list[str]]:
    """`(training rows, held-out rows, the held-out queries)`: whole groups
    of queries held out until EVAL_FRAC of the queries with expected nodes
    are; without `expected`, every query can be."""
    queries = sorted({r.source for r in rows})
    if len(queries) < 2:
        raise SystemExit(f"only {len(queries)} query in the graded paths; need "
                         "at least 2 to hold one out")
    scorable = (set(queries) if expected is None
                else {q for q in queries if expected.get(q)})
    groups = [g for g in query_components(queries, expected)
              if scorable.intersection(g)]
    if not groups:
        raise SystemExit("no graded query has expected nodes in the graded "
                         "queries, so none can be held out")
    random.Random(SPLIT_SEED).shuffle(groups)
    n_eval = max(1, round(len(scorable) * EVAL_FRAC))
    held: list[list[str]] = []
    n_held = 0
    for group in groups:
        if n_held >= n_eval:
            break
        held.append(group)
        n_held += len(scorable.intersection(group))
    if sum(map(len, held)) >= len(queries):
        if len(held) < 2:
            raise SystemExit("every query shares a neighbourhood with every "
                             "other, so holding any out leaves nothing to "
                             "train on")
        held.pop()
    withheld = {q for group in held for q in group}
    train = [r for r in rows if r.source not in withheld]
    evaluation = [r for r in rows if r.source in withheld and r.source in scorable]
    return train, evaluation, sorted(withheld & scorable)


# --- features -----------------------------------------------------------------------


@dataclass
class PathFeatures:
    """Ragged per-path arrays, padded into one batch by `collate`."""

    node_x: list[np.ndarray]
    edge_x: list[np.ndarray]
    n_hops: np.ndarray        # (N,) int64
    scalars: np.ndarray       # (N, len(SCALAR_NAMES)) float32
    embed_dim: int
    tiers: np.ndarray         # (N,) int64
    sources: list[str]
    tier_weights: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.node_x)


def scalar_features(rows: Sequence[LabelledRow], node_context: EmbeddingCache,
                    edge_context: PairTable) -> np.ndarray:
    """(N, len(SCALAR_NAMES)): the path length, and the context tables'
    degree, information content and support along the path; a node or pair
    a table lacks reads as zeros."""
    out = np.zeros((len(rows), len(SCALAR_NAMES)), dtype=np.float32)
    col = {name: j for j, name in enumerate(SCALAR_NAMES)}
    for i, row in enumerate(rows):
        out[i, col["n_hops_scaled"]] = row.n_hops / N_HOPS_SCALE
        vecs = np.stack([
            np.asarray(node_context[c], dtype=np.float32)
            if c in node_context else np.zeros(node_context.dim, np.float32)
            for c in row.node_curies
        ])
        out[i, col["deg_out_max"]] = vecs[:, 0].max()
        out[i, col["deg_out_mean"]] = vecs[:, 0].mean()
        out[i, col["deg_in_max"]] = vecs[:, 1].max()
        out[i, col["deg_in_mean"]] = vecs[:, 1].mean()
        out[i, col["ic_mean"]] = vecs[:, 2].mean()
        out[i, col["ic_present_frac"]] = vecs[:, 3].mean()
        hops = np.stack([
            np.asarray(edge_context[k], dtype=np.float32)
            if (k := pair_key(a, b)) in edge_context
            else np.zeros(edge_context.dim, np.float32)
            for a, b in zip(row.node_curies[:-1], row.node_curies[1:])
        ])
        for j, base in enumerate(["edge_count", "pubs", "primary_sources",
                                  "distinct_sources", "attributes_seen"]):
            out[i, col[f"{base}_mean"]] = hops[:, j].mean()
            out[i, col[f"{base}_min"]] = hops[:, j].min()
    return out


def path_features(rows: list[LabelledRow], names: dict[str, str],
                  embeddings: EmbeddingCache, node_context: EmbeddingCache,
                  edge_context: PairTable) -> PathFeatures:
    """Each row's node, category and hop vectors and its scalars."""
    dim = embeddings.dim
    node_x, edge_x = [], []
    for row in rows:
        fused = np.empty((row.n_hops + 1, 2 * dim), dtype=np.float32)
        for j, (curie, cat) in enumerate(
                zip(row.node_curies, row.node_categories, strict=True)):
            fused[j, :dim] = embeddings[names[curie]]
            fused[j, dim:] = embeddings[category_phrase(cat)]
        node_x.append(fused)
        edge_x.append(np.stack([np.asarray(embeddings[p], dtype=np.float32)
                                for p in row.hop_phrases]))
    widths = {len(r.tier_weights) for r in rows if r.tier_weights is not None}
    if len(widths) > 1:
        raise ValueError(f"soft labels over {sorted(widths)} tiers in one set")
    tier_weights = None
    if widths:
        (n_tiers,) = widths
        tier_weights = np.array([r.tier_weights if r.tier_weights is not None
                                 else [np.nan] * n_tiers for r in rows],
                                dtype=np.float64)
    return PathFeatures(
        node_x=node_x, edge_x=edge_x,
        n_hops=np.array([r.n_hops for r in rows], dtype=np.int64),
        scalars=scalar_features(rows, node_context, edge_context),
        embed_dim=dim,
        tiers=np.array([r.tier for r in rows], dtype=np.int64),
        sources=[r.source for r in rows],
        tier_weights=tier_weights,
    )


def collate(features: PathFeatures, indices, device: str) -> dict[str, torch.Tensor]:
    """The selected paths padded into dense tensors for one forward pass."""
    idx = np.asarray(indices)
    max_hops = int(features.n_hops[idx].max())
    dim = features.embed_dim
    b = len(idx)
    node_x = np.zeros((b, max_hops + 1, 2 * dim), dtype=np.float32)
    edge_x = np.zeros((b, max_hops, dim), dtype=np.float32)
    for row_pos, i in enumerate(idx):
        n = int(features.n_hops[i])
        node_x[row_pos, : n + 1] = features.node_x[i]
        edge_x[row_pos, :n] = features.edge_x[i]
    return {
        "node_x": torch.from_numpy(node_x).to(device),
        "edge_x": torch.from_numpy(edge_x).to(device),
        "n_hops": torch.from_numpy(features.n_hops[idx]).to(device),
        "scalars": torch.from_numpy(features.scalars[idx]).to(device),
        "metapath_ids": torch.zeros(b, dtype=torch.int64, device=device),
    }


def score_paths(model: PathSequenceScorer, features: PathFeatures, device: str,
                batch_size: int = 512) -> np.ndarray:
    """Every path's raw score, (N,) float32."""
    model.eval()
    out = np.empty(len(features), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            idx = np.arange(start, min(start + batch_size, len(features)))
            out[idx] = model(**collate(features, idx, device)).float().cpu().numpy()
    return out


# --- pairs -------------------------------------------------------------------------


@dataclass
class PairIndex:
    """Rows grouped for drawing pairs within a query, a hop count at a time.

    `targets[i, j]` is the chance a path of label class i is the better of a
    pair with one of class j; `eligible` marks the class pairs worth drawing.
    """

    strata: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]
    classes: np.ndarray
    targets: np.ndarray
    eligible: np.ndarray


def build_pair_index(features: PathFeatures) -> PairIndex:
    """Pairs of a top-tier path against one below it, by soft label."""
    w = features.tier_weights
    missing = len(features) if w is None else int(np.isnan(w).any(axis=1).sum())
    if missing:
        raise SystemExit(f"{missing:,} of {len(features):,} graded paths carry "
                         "no tier_weights, which training reads")
    weights, classes = np.unique(np.round(w, 12), axis=0, return_inverse=True)
    classes = classes.reshape(-1)
    #  Each class merged to its chance of a top tier against one below
    top = weights[:, [t - MIN_TIER for t in TOP_TIERS]].sum(axis=1)
    merged, inverse = np.unique(np.round(top, 12), return_inverse=True)
    classes, weights = inverse[classes], np.stack([merged, 1.0 - merged], axis=1)
    at_least = np.cumsum(weights, axis=1)
    targets = weights @ (1.0 - at_least).T + 0.5 * (weights @ weights.T)
    eligible = np.abs(targets - 0.5) >= MIN_PAIR_MARGIN

    by_query: dict[str, list[int]] = defaultdict(list)
    for i, src in enumerate(features.sources):
        by_query[src].append(i)
    strata: dict[int, list] = defaultdict(list)
    for rows in by_query.values():
        rows_arr = np.asarray(rows)
        row_classes = classes[rows_arr]
        present = np.unique(row_classes)
        usable = present[eligible[np.ix_(present, present)].any(axis=1)]
        if not len(usable):
            continue
        can_anchor = np.isin(row_classes, usable)
        hops = features.n_hops[rows_arr]
        for h in np.unique(hops):
            anchors = rows_arr[(hops == h) & can_anchor]
            if len(anchors):
                strata[int(h)].append((anchors, rows_arr, row_classes))
    if not strata:
        raise SystemExit("no query holds two paths whose labels differ; "
                         "pairwise training is impossible on this data")
    return PairIndex(strata=dict(strata), classes=classes, targets=targets,
                     eligible=eligible)


def sample_pairs(index: PairIndex, batch_size: int, rng: random.Random
                 ) -> tuple[np.ndarray, np.ndarray]:
    """`batch_size` (better, worse) pairs: hop counts in turn, then a query,
    an anchor of that length and a partner it can be drawn against."""
    hop_counts = sorted(index.strata)
    better = np.empty(batch_size, dtype=np.int64)
    worse = np.empty(batch_size, dtype=np.int64)
    for k in range(batch_size):
        entries = index.strata[hop_counts[k % len(hop_counts)]]
        anchors, rows_arr, row_classes = entries[rng.randrange(len(entries))]
        a = int(anchors[rng.randrange(len(anchors))])
        a_class = row_classes[rows_arr == a][0]
        partners = rows_arr[index.eligible[a_class, row_classes]]
        b = int(partners[rng.randrange(len(partners))])
        b_class = row_classes[rows_arr == b][0]
        if index.targets[a_class, b_class] > 0.5:
            better[k], worse[k] = a, b
        else:
            better[k], worse[k] = b, a
    return better, worse


def teacher_pair_targets(teacher_scores: np.ndarray, better: np.ndarray,
                         worse: np.ndarray) -> np.ndarray:
    """The logistic of the teachers' mean score gap for each pair."""
    gap = (teacher_scores[:, better] - teacher_scores[:, worse]).mean(axis=0)
    return 0.5 * (1.0 + np.tanh(0.5 * gap))


# --- the held-out yardstick ----------------------------------------------------------


def ndcg_at_10(tiers: np.ndarray, scores: np.ndarray, sources: list[str]
               ) -> float | None:
    """Mean per-query NDCG@10 against relevance `max tier - tier`, ties in
    score broken by a fixed shuffle."""
    tiers = [int(t) for t in tiers]
    scores = [float(s) for s in scores]
    max_tier = max(tiers) if tiers else 1
    by_query: dict[str, list[int]] = defaultdict(list)
    for i, q in enumerate(sources):
        by_query[q].append(i)
    rng = random.Random(0)
    ndcgs = []
    for _, idx in sorted(by_query.items()):
        shuffled = list(idx)
        rng.shuffle(shuffled)
        top = sorted(shuffled, key=lambda i: -scores[i])[:10]
        ideal = sorted((max(0, max_tier - tiers[i]) for i in idx),
                       reverse=True)[:10]
        got = [max(0, max_tier - tiers[i]) for i in top]
        ideal_dcg = _dcg(ideal)
        if ideal_dcg > 0:
            ndcgs.append(_dcg(got) / ideal_dcg)
    return statistics.fmean(ndcgs) if ndcgs else None


def _dcg(rels) -> float:
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def fit_length_scale(scores: np.ndarray, n_hops: np.ndarray
                     ) -> dict[int, tuple[float, float]]:
    """Each path length's mean score and spread, over the training paths."""
    scale = {}
    for length in np.unique(n_hops):
        picked = np.asarray(scores, dtype=np.float64)[n_hops == length]
        spread = float(picked.std())
        if not spread > 0:
            raise ValueError(f"the {int(length)}-hop paths are all scored "
                             f"{float(picked[0])}: no spread to fit a scale on")
        scale[int(length)] = (float(picked.mean()), spread)
    return scale


# --- one training ---------------------------------------------------------------------


def _save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def fit(train: PathFeatures, held_out: PathFeatures, settings: Settings,
        device: str, checkpoint: Path,
        teacher_scores: np.ndarray | None = None) -> tuple[PathSequenceScorer, dict]:
    """Train one model until NDCG@10 on the held-out queries stops rising
    (or the step budget runs out); returns the best model and its kwargs.

    RankNet on soft-label pairs, or on the teachers' targets when
    `teacher_scores` (teachers, training paths) are given. `checkpoint` is
    written every evaluation, and a run that finds one of its own settings
    there resumes from it.
    """
    torch.manual_seed(settings.seed)
    rng = random.Random(settings.seed)
    max_hops = int(max(train.n_hops.max(), held_out.n_hops.max()))
    model_kwargs = {
        "embed_dim": train.embed_dim,
        "d_model": settings.d_model,
        "n_heads": settings.n_heads,
        "n_layers": settings.n_layers,
        "dim_feedforward": None,
        "dropout": settings.dropout,
        "head_hidden": settings.head_hidden,
        "n_scalars": len(SCALAR_NAMES),
        "metapath_vocab_size": 0,
        "metapath_dim": 32,
        "max_positions": max(settings.max_positions, 2 * max_hops + 2),
    }
    model = PathSequenceScorer(**model_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=settings.lr)
    index = build_pair_index(train)
    fingerprint = {"model_kwargs": model_kwargs, "settings": vars(settings),
                   "teachers": None if teacher_scores is None
                   else len(teacher_scores)}

    best_value, best_state, evals_since_best, start_step = -float("inf"), None, 0, 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint:
            raise SystemExit(f"{checkpoint} was written by another training; "
                             "give this build its own work directory")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        best_value, best_state = state["best_value"], state["best_state"]
        evals_since_best, start_step = state["evals_since_best"], state["step"]
        print(f"  resumed at step {start_step:,}")

    for step in tqdm(range(start_step, settings.max_steps), desc="training",
                     initial=start_step, total=settings.max_steps, leave=False):
        model.train()
        better, worse = sample_pairs(index, settings.batch_size, rng)
        scores = model(**collate(train, np.concatenate([better, worse]), device))
        s_better, s_worse = scores[: settings.batch_size], scores[settings.batch_size:]
        if teacher_scores is None:
            target = index.targets[index.classes[better], index.classes[worse]]
        else:
            target = teacher_pair_targets(teacher_scores, better, worse)
        target = torch.as_tensor(target, dtype=s_better.dtype, device=s_better.device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            s_better - s_worse, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % settings.eval_every == 0:
            value = ndcg_at_10(held_out.tiers, score_paths(model, held_out, device),
                               held_out.sources)
            if value is not None and value > best_value:
                best_value = value
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                evals_since_best = 0
            else:
                evals_since_best += 1
            _save_state(checkpoint, {
                "fingerprint": fingerprint, "step": step + 1,
                "model": model.state_dict(), "optimizer": opt.state_dict(),
                "rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                "best_value": best_value, "best_state": best_state,
                "evals_since_best": evals_since_best})
            if evals_since_best >= settings.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, model_kwargs


def train_one(name: str, train: PathFeatures, held_out: PathFeatures,
              held_out_queries: list[str], settings: Settings, device: str,
              work: Path, teachers: Sequence[Path] = ()) -> Path:
    """Train one model of the recipe into `work/name/best.pt`, unless it is
    already there; returns the checkpoint."""
    out = work / name
    best = out / CHECKPOINT
    if best.exists():
        return best
    out.mkdir(parents=True, exist_ok=True)
    teacher_scores = None
    if teachers:
        teacher_scores = np.stack([
            score_paths(load_scorer(t, device)[0], train, device)
            for t in teachers])
    print(f"training {name} (width {settings.d_model}, seed {settings.seed}"
          + (f", taught by {len(teachers)}" if teachers else "") + ")")
    model, model_kwargs = fit(train, held_out, settings, device,
                              out / "last.pt", teacher_scores)
    length_scale = fit_length_scale(score_paths(model, train, device),
                                    train.n_hops)
    partial = out / (CHECKPOINT + ".partial")
    save_scorer(partial, model, model_kwargs, {
        "length_scale": length_scale,
        "eval_sources": held_out_queries,
        "select_metric": "ndcg@10",
        "args": {"node_context": True, "edge_context": True,
                 "seed": SPLIT_SEED, "train_seed": settings.seed,
                 "teachers": [str(t) for t in teachers]},
    })
    partial.replace(best)
    return best


def train_recipe(rows: list[LabelledRow], expected: dict[str, set[str]] | None,
                 names: dict[str, str], embeddings: EmbeddingCache,
                 node_context: EmbeddingCache, edge_context: PairTable,
                 work: Path, device: str) -> Path:
    """The three teachers and the student; returns the student's checkpoint."""
    train_rows, held_rows, held_queries = split_queries(rows, expected)
    print(f"  {len(held_queries)} queries held out: {len(train_rows):,} paths "
          f"to train on, {len(held_rows):,} to choose by")
    train = path_features(train_rows, names, embeddings, node_context, edge_context)
    held_out = path_features(held_rows, names, embeddings, node_context,
                             edge_context)
    teachers = [train_one(f"teacher_{seed}", train, held_out, held_queries,
                          Settings(d_model=TEACHER_D_MODEL, seed=seed),
                          device, work)
                for seed in TEACHER_SEEDS]
    return train_one("student", train, held_out, held_queries,
                     Settings(d_model=STUDENT_D_MODEL, seed=STUDENT_SEED),
                     device, work, teachers)

