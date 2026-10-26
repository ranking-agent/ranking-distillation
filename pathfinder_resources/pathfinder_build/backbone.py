"""The priced relationships as the pathfinder's search graph.

Three steps, all vectorized numpy over the relationships:

1. **Backbone.** The relationships, `subclass_of` and `superclass_of` left
   out (the mechanistic backbone: a path through an ontology parent explains
   nothing), are summed per unordered node pair. The disparity filter
   (Serrano, Boguna, Vespignani 2009) runs at significance 1.0, which keeps
   a pair either endpoint finds significant, any pair of a node below degree
   2, and each node's two strongest pairs ranked by the degree-damped
   weight at alpha 0.15 -- at that significance nearly every pair.
2. **Weights.** Each relationship's Tier-1 weight is

       w(e) = -log(evidence(e) / Z(e)) + alpha * log(deg_a deg_b) + lambda * G(e)

   with evidence(e) = exp(-1 / admittance(e)), Z(e) the largest evidence in
   its (predicate, qualifier set) group (the bare predicate's for a group
   under 24 relationships), alpha 0.25, and G(e) in [0, 1] its genericity
   from the biolink hierarchy, at lambda 1.0, less 0.25 per qualifier. A
   pair weighs its cheapest relationship.
3. **Search graph.** The pair graph as an undirected CSR, with the node
   CURIEs in index order: what shepherd's aragorn_pathfinder worker loads.
"""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from tqdm import tqdm

from .biolink import try_predicate_depth

QUALIFIED_PREDICATE = "qualified_predicate"

#  The settings the deployed search graph was built with
DROP_PREDICATES = ("biolink:subclass_of", "biolink:superclass_of")
SIGNIFICANCE = 1.0
MIN_DEGREE = 2
KEEP_TOP_K = 2
RESCUE_ALPHA = 0.15
ALPHA = 0.25
SPECIFICITY_LAMBDA = 1.0
QUALIFIER_MU = 0.25
MIN_GROUP = 24


@dataclass
class RelationshipTable:
    """The priced relationships as parallel arrays plus vocabularies."""

    node_ids: list[str]
    predicates: list[str]
    qualifier_sets: list[list[list[str]]]  # per id: [[name, value], ...]
    subject_idx: np.ndarray  # int32
    object_idx: np.ndarray  # int32
    predicate_id: np.ndarray  # int32
    qualset_id: np.ndarray  # int32
    admittance: np.ndarray  # float32


def relationship_table(total: int, records) -> RelationshipTable:
    """One scan of the priced relationships into arrays, in the order they
    come; a relationship with no admittance carries no evidence and is left
    out."""
    node_vocab: dict[str, int] = {}
    predicate_vocab: dict[str, int] = {}
    qualset_vocab: dict[tuple, int] = {}
    qualifier_sets: list[list[list[str]]] = []
    subject_idx, object_idx = array("i"), array("i")
    predicate_id, qualset_id = array("i"), array("i")
    admittance = array("f")
    for rec in tqdm(records, total=total, desc="relationships -> arrays",
                    unit=" rels"):
        if rec["admittance"] <= 0.0:
            continue
        subject_idx.append(node_vocab.setdefault(rec["subject"], len(node_vocab)))
        object_idx.append(node_vocab.setdefault(rec["object"], len(node_vocab)))
        predicate_id.append(
            predicate_vocab.setdefault(rec["predicate"], len(predicate_vocab)))
        qual_key = tuple((q[0], q[1]) for q in rec["qualifiers"])
        qid = qualset_vocab.get(qual_key)
        if qid is None:
            qid = len(qualifier_sets)
            qualset_vocab[qual_key] = qid
            qualifier_sets.append([[name, value] for name, value in qual_key])
        qualset_id.append(qid)
        admittance.append(rec["admittance"])
    return RelationshipTable(
        node_ids=list(node_vocab),
        predicates=list(predicate_vocab),
        qualifier_sets=qualifier_sets,
        subject_idx=np.frombuffer(subject_idx, dtype=np.int32),
        object_idx=np.frombuffer(object_idx, dtype=np.int32),
        predicate_id=np.frombuffer(predicate_id, dtype=np.int32),
        qualset_id=np.frombuffer(qualset_id, dtype=np.int32),
        admittance=np.frombuffer(admittance, dtype=np.float32),
    )


def without_predicates(rels: RelationshipTable,
                       names: Sequence[str]) -> RelationshipTable:
    """The table less every relationship of those predicates; the
    vocabularies stay as they are."""
    drop_ids = np.array(
        [i for i, name in enumerate(rels.predicates) if name in set(names)],
        dtype=np.int64,
    )
    keep = ~np.isin(rels.predicate_id, drop_ids)
    return replace(
        rels,
        subject_idx=rels.subject_idx[keep],
        object_idx=rels.object_idx[keep],
        predicate_id=rels.predicate_id[keep],
        qualset_id=rels.qualset_id[keep],
        admittance=rels.admittance[keep],
    )


def pair_table(rels: RelationshipTable):
    """`(pair_a, pair_b, pair admittance, each relationship's pair row)`:
    the unordered node pairs, a < b, each summing its relationships."""
    a = np.minimum(rels.subject_idx, rels.object_idx).astype(np.int64)
    b = np.maximum(rels.subject_idx, rels.object_idx).astype(np.int64)
    keys = (a << 32) | b
    unique_keys, rel_pair = np.unique(keys, return_inverse=True)
    pair_a = (unique_keys >> 32).astype(np.int32)
    pair_b = (unique_keys & 0xFFFFFFFF).astype(np.int32)
    pair_admittance = np.bincount(
        rel_pair, weights=rels.admittance.astype(np.float64)
    ).astype(np.float32)
    return pair_a, pair_b, pair_admittance, rel_pair.astype(np.int32)


def disparity_keep_mask(pair_a, pair_b, pair_weight, num_nodes: int,
                        significance: float, min_degree: int) -> np.ndarray:
    """The disparity filter: a pair is kept if either endpoint finds it
    significant ((1 - w/s)^(k-1) < significance) or either endpoint is below
    `min_degree`."""
    weight = pair_weight.astype(np.float64)
    degree = np.bincount(pair_a, minlength=num_nodes) + np.bincount(
        pair_b, minlength=num_nodes
    )
    strength = np.bincount(
        pair_a, weights=weight, minlength=num_nodes
    ) + np.bincount(pair_b, weights=weight, minlength=num_nodes)

    def endpoint_significant(nodes: np.ndarray) -> np.ndarray:
        k = degree[nodes]
        s = strength[nodes]
        p = np.divide(weight, s, out=np.ones_like(weight), where=s > 0)
        alpha = np.power(1.0 - np.minimum(p, 1.0), k - 1)
        return (k > 1) & (alpha < significance)

    exempt = (degree[pair_a] < min_degree) | (degree[pair_b] < min_degree)
    return endpoint_significant(pair_a) | endpoint_significant(pair_b) | exempt


def top_k_keep_mask(pair_a, pair_b, pair_weight, num_nodes: int, k: int,
                    rescue_alpha: float) -> np.ndarray:
    """Each node's `k` strongest pairs, ranked by the damped weight
    `exp(-1/admittance) / partner degree**rescue_alpha`, the order the
    search would take them in."""
    endpoint = np.concatenate([pair_a, pair_b])
    pair_idx = np.concatenate([np.arange(len(pair_a)), np.arange(len(pair_a))])
    weight = np.concatenate([pair_weight, pair_weight]).astype(np.float64)
    partner = np.concatenate([pair_b, pair_a])
    degree = np.bincount(endpoint, minlength=num_nodes).astype(np.float64)
    weight = np.exp(-1.0 / weight) / np.maximum(degree[partner], 1.0) ** rescue_alpha
    # strongest first within each node
    order = np.lexsort((-weight, endpoint))
    endpoint = endpoint[order]
    pair_idx = pair_idx[order]
    is_first = np.concatenate(([True], endpoint[1:] != endpoint[:-1]))
    positions = np.arange(len(endpoint))
    group_start = np.maximum.accumulate(np.where(is_first, positions, 0))
    rank = positions - group_start
    keep = np.zeros(len(pair_a), dtype=bool)
    keep[pair_idx[rank < k]] = True
    return keep


@dataclass
class Backbone:
    """The pairs the filter kept and the relationships on them."""

    node_ids: list[str]
    predicates: list[str]
    qualifier_sets: list[list[list[str]]]
    pair_a: np.ndarray            # int32
    pair_b: np.ndarray            # int32
    degrees: np.ndarray           # int32, per node, over the kept pairs
    rel_pair: np.ndarray          # int32
    rel_predicate: np.ndarray     # int32
    rel_qualset: np.ndarray       # int32
    rel_admittance: np.ndarray    # float32


def build_backbone(rels: RelationshipTable) -> Backbone:
    """The mechanistic backbone of the priced relationships."""
    rels = without_predicates(rels, DROP_PREDICATES)
    num_nodes = len(rels.node_ids)
    pair_a, pair_b, pair_admittance, rel_pair = pair_table(rels)
    keep = disparity_keep_mask(pair_a, pair_b, pair_admittance, num_nodes,
                               SIGNIFICANCE, MIN_DEGREE)
    keep = keep | top_k_keep_mask(pair_a, pair_b, pair_admittance, num_nodes,
                                  KEEP_TOP_K, RESCUE_ALPHA)
    # Re-number the surviving pairs; the node vocabulary stays as it is.
    new_pair_index = np.full(len(pair_a), -1, dtype=np.int32)
    new_pair_index[keep] = np.arange(int(keep.sum()), dtype=np.int32)
    pair_a, pair_b = pair_a[keep], pair_b[keep]
    rel_keep = keep[rel_pair]
    degrees = (np.bincount(pair_a, minlength=num_nodes)
               + np.bincount(pair_b, minlength=num_nodes)).astype(np.int32)
    return Backbone(
        node_ids=rels.node_ids,
        predicates=rels.predicates,
        qualifier_sets=rels.qualifier_sets,
        pair_a=pair_a,
        pair_b=pair_b,
        degrees=degrees,
        rel_pair=new_pair_index[rel_pair[rel_keep]],
        rel_predicate=rels.predicate_id[rel_keep],
        rel_qualset=rels.qualset_id[rel_keep],
        rel_admittance=rels.admittance[rel_keep],
    )


def _group_max(values: np.ndarray, groups: np.ndarray, num_groups: int) -> np.ndarray:
    """Max of `values` per group id, via one sort + reduceat."""
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    boundaries = np.flatnonzero(np.diff(sorted_groups)) + 1
    starts = np.concatenate(([0], boundaries))
    maxima = np.maximum.reduceat(values[order], starts)
    out = np.zeros(num_groups, dtype=values.dtype)
    out[sorted_groups[starts]] = maxima
    return out


def genericity(predicates, qualifier_sets, predicate_id, qualset_id,
               qualifier_mu: float) -> np.ndarray:
    """Per-relationship genericity in [0, 1]: one less the depth of its
    effective predicate (the qualified predicate where there is one) over
    the deepest observed, less `qualifier_mu` per other qualifier. A
    predicate the model gives no depth counts as depth 0."""
    pred_depths = np.array(
        [try_predicate_depth(p) or 0 for p in predicates], dtype=np.float64
    )
    qualset_depth_override = np.full(len(qualifier_sets), -1.0)
    qualset_n_quals = np.zeros(len(qualifier_sets))
    for qid, quals in enumerate(qualifier_sets):
        n = 0
        for name, value in quals:
            if name == QUALIFIED_PREDICATE:
                depth = try_predicate_depth(value)
                if depth is not None:
                    qualset_depth_override[qid] = depth
            else:
                n += 1
        qualset_n_quals[qid] = n

    depths = pred_depths[predicate_id]
    override = qualset_depth_override[qualset_id]
    depths = np.where(override >= 0, override, depths)
    max_depth = max(
        float(pred_depths.max()),
        float(qualset_depth_override.max()),
        1.0,
    )
    g = 1.0 - depths / max_depth - qualifier_mu * qualset_n_quals[qualset_id]
    return np.clip(g, 0.0, 1.0)


# The four additive terms a relationship weight is the sum of.
COST_TERMS = ("evidence", "promiscuity", "group", "genericity")


def pair_weights(bb: Backbone) -> np.ndarray:
    """Each pair's Tier-1 weight, float32: its cheapest relationship's."""
    evidence = np.exp(-1.0 / bb.rel_admittance.astype(np.float64))
    end_a = bb.pair_a[bb.rel_pair]
    end_b = bb.pair_b[bb.rel_pair]
    deg_a = bb.degrees[end_a].astype(np.float64)
    deg_b = bb.degrees[end_b].astype(np.float64)
    promiscuity = ALPHA * (np.log(deg_a) + np.log(deg_b))
    #  The normalizer is taken from the evidence itself, so the degree tax
    #  is charged in full.
    normalized = evidence

    group_keys = (bb.rel_predicate.astype(np.int64) << 32) | bb.rel_qualset.astype(
        np.int64
    )
    _, group_id = np.unique(group_keys, return_inverse=True)
    num_groups = int(group_id.max()) + 1
    group_size = np.bincount(group_id, minlength=num_groups)
    group_max = _group_max(normalized, group_id, num_groups)
    pred_max = _group_max(
        normalized, bb.rel_predicate.astype(np.int64), len(bb.predicates)
    )
    backoff = group_size < MIN_GROUP
    group_pred = np.zeros(num_groups, dtype=np.int64)
    group_pred[group_id] = np.asarray(bb.rel_predicate, dtype=np.int64)
    z_base = np.where(backoff, pred_max[group_pred], group_max)[group_id]

    terms = {
        "evidence": -np.log(evidence),
        "promiscuity": promiscuity,
        "group": np.log(z_base),
        "genericity": SPECIFICITY_LAMBDA * genericity(
            bb.predicates, bb.qualifier_sets, np.asarray(bb.rel_predicate),
            np.asarray(bb.rel_qualset), QUALIFIER_MU),
    }
    w = np.maximum(sum(terms[name] for name in COST_TERMS), 0.0)
    w_pair = np.full(len(bb.pair_a), np.inf)
    np.minimum.at(w_pair, bb.rel_pair, w)
    return w_pair.astype(np.float32)


def search_csr(bb: Backbone, w_pair: np.ndarray):
    """`(offsets, neighbors, weights)`: the undirected CSR over the pairs,
    every pair in both directions."""
    num_nodes = len(bb.degrees)
    src = np.concatenate([bb.pair_a, bb.pair_b]).astype(np.int64)
    dst = np.concatenate([bb.pair_b, bb.pair_a]).astype(np.int32)
    w = np.concatenate([w_pair, w_pair]).astype(np.float64)
    order = np.argsort(src, kind="stable")
    offsets = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=num_nodes), out=offsets[1:])
    return offsets, dst[order], w[order]
