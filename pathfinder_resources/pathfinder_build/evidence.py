"""One KGX edge's evidence, and the weight a pricing profile gives it.

Ported from the edge-value extraction and weight combination of shepherd's
aragorn_score worker (itself from ranking-agent/aragorn-ranker), with the
input side adapted to raw KGX jsonl edges, where everything the TRAPI
attribute loop looks for is a top-level field.

Per edge:

1. `canonicalize_edge` normalizes direction: a symmetric predicate orders its
   endpoints lexically, a non-canonical one (by the biolink model's
   `canonical_predicate` annotation) flips to its canonical inverse, swapping
   subject_/object_ qualifiers and inverting a `qualified_predicate`.
2. `extract_ingredients` pulls the raw values the profile prices
   (publications, p-value, literature co-occurrence, affinity), and
   `extract_metrics` the rest of the evidence the confidence pricing reads.
3. `edge_property_weights` maps ingredients to weights in [0, 1) under a
   profile: a base weight for the edge's source, plus sigmoid(value) times
   the source's weight for each ingredient present.
4. `admittance_from_weights` combines them as ELROND does: each weight w is
   the admittance -1/ln(w), and parallel evidence adds.

Two deliberate departures from the worker: a weight <= 0 contributes nothing
(the worker would take its log), and weights are clamped to MAX_WEIGHT by
value, since the worker's `w >= 1` test never fires on a weight that
saturates just below 1 and so prices well-cited edges as weaker than
less-cited ones.

Predicates the biolink model does not know are kept as asserted, with no
direction canonicalization, since their symmetry and inverse are unknowable.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from .biolink import (
    get_inverse_predicate,
    name_to_predicate_curie,
    predicate_to_name,
    toolkit,
)

# Global publication count the literature co-occurrence formula assumes.
TOTAL_PUBS = 27_840_000

DEFAULT_WEIGHT = 1e-2

# Largest weight admitted into -1/ln(w), which diverges at w = 1.
MAX_WEIGHT = 0.9999999


def get_base_weight(source: str, base_weights: Mapping[str, float]) -> float:
    return base_weights.get(source, base_weights.get("default_weight", DEFAULT_WEIGHT))


def get_source_weight(
    source: str,
    property: str,
    source_weights: Mapping[str, Mapping[str, float]],
    unknown_source_weight: Mapping[str, float],
) -> float:
    return source_weights.get(source, unknown_source_weight).get(
        property, unknown_source_weight["unknown_property"]
    )


def get_source_sigmoid(
    value: float,
    source: str,
    property: str,
    source_transformation: Mapping[str, Mapping[str, Mapping[str, float]]],
    unknown_source_transformation: Mapping[str, Mapping[str, float]],
) -> float:
    """Generalized logistic squashing a raw property value into a weight."""
    parameters = source_transformation.get(source, unknown_source_transformation).get(
        property, unknown_source_transformation["unknown_property"]
    )
    upper = parameters["upper"]
    lower = parameters["lower"]
    midpoint = parameters["midpoint"]
    rate = parameters["rate"]
    return lower + ((upper - lower) / (1 + math.exp(-1 * rate * (value - midpoint))))


# ---------------------------------------------------------------------------
# KGX field handling
# ---------------------------------------------------------------------------

# Substrings the ranker matches against attribute names to spot a p-value.
_P_VALUE_MARKERS = ("p_value", "p-value", "pValue", "fisher_exact_p", "gwas_pvalue")

LIT_COOC_PREDICATE = "biolink:occurs_together_in_literature_with"


@lru_cache(maxsize=1)
def qualifier_fields() -> frozenset[str]:
    """The top-level KGX edge fields that are biolink qualifiers: every
    descendant of the abstract `qualifier` slot, as gandalf's loader derives
    them."""
    descendants = toolkit().get_descendants("qualifier", formatted=True)
    fields = {d.split(":", 1)[-1] for d in descendants}
    fields.discard("qualifier")  # abstract root, never a real edge field
    if not fields:
        raise RuntimeError("biolink model returned no qualifier slots")
    return frozenset(fields)


def extract_qualifiers(edge: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """`(qualifier name, value)` pairs from a KGX edge, sorted by name.

    Values that are not strings are coerced with str(), and a
    `qualified_predicate` is normalized to a `biolink:` CURIE so it keys as
    gandalf's normalized form does.
    """
    quals = []
    for field in sorted(qualifier_fields() & edge.keys()):
        value = edge[field]
        if not isinstance(value, str):
            value = str(value)
        if field == "qualified_predicate":
            value = name_to_predicate_curie(predicate_to_name(value))
        quals.append((field, value))
    return tuple(quals)


def extract_primary_source(edge: Mapping[str, Any]) -> str:
    """The primary knowledge source, or "unspecified" (the ranker's default),
    from the flat KGX field or a TRAPI-style `sources` list."""
    primary = edge.get("primary_knowledge_source")
    if primary:
        return primary
    for source in edge.get("sources") or []:
        if source.get("resource_role") == "primary_knowledge_source":
            return source.get("resource_id", "unspecified")
    return "unspecified"


@lru_cache(maxsize=4096)
def _predicate_form(predicate: str) -> tuple[str, bool, bool, bool]:
    """(canonical CURIE, flip?, symmetric?, known?) for a predicate CURIE.

    Canonical if the model annotates it `canonical_predicate: true`, if it is
    symmetric, or if neither it nor its inverse carries the annotation. A
    predicate the model does not know keeps its asserted form.
    """
    tk = toolkit()
    name = predicate_to_name(predicate)
    curie = name_to_predicate_curie(name)
    if not tk.is_predicate(name):
        return curie, False, False, False
    if tk.is_symmetric(name):
        return curie, False, True, True
    if _has_canonical_annotation(tk, name):
        return curie, False, False, True
    inverse = tk.get_inverse(name)
    if inverse is not None and _has_canonical_annotation(tk, inverse):
        return name_to_predicate_curie(inverse), True, False, True
    return curie, False, False, True


def _has_canonical_annotation(tk, name: str) -> bool:
    element = tk.get_element(name)
    if element is None:
        return False
    annotations = getattr(element, "annotations", None) or {}
    annotation = annotations.get("canonical_predicate")
    return bool(annotation and getattr(annotation, "value", False))


def _swap_qualifier_sides(
    qualifiers: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    swapped = []
    for name, value in qualifiers:
        if name.startswith("subject_"):
            name = "object_" + name[len("subject_") :]
        elif name.startswith("object_"):
            name = "subject_" + name[len("object_") :]
        elif name == "qualified_predicate":
            value = get_inverse_predicate(value)
        swapped.append((name, value))
    return tuple(sorted(swapped))


def canonicalize_edge(
    subject: str,
    predicate: str,
    object_: str,
    qualifiers: tuple[tuple[str, str], ...],
) -> tuple[str, str, str, tuple[tuple[str, str], ...]]:
    """An edge in the direction both assertions of one fact key alike."""
    canonical_pred, flip, symmetric, _known = _predicate_form(predicate)
    if flip or (symmetric and object_ < subject):
        return object_, canonical_pred, subject, _swap_qualifier_sides(qualifiers)
    return subject, canonical_pred, object_, qualifiers


# ---------------------------------------------------------------------------
# Ingredients: the worker's attribute loop, on KGX fields
# ---------------------------------------------------------------------------


def _parse_publications(raw: Any) -> list[str]:
    """The worker's publication parsing: strings, pipe- and comma-joined lists."""
    pubs = raw if raw is not None else []
    if isinstance(pubs, str):
        pubs = [pubs]
    if len(pubs) == 1 and isinstance(pubs[0], str):
        if "|" in pubs[0]:
            pubs = pubs[0].split("|")
        elif "," in pubs[0]:
            pubs = pubs[0].split(",")
    return list(pubs)


def parse_p_value(raw: Any) -> float | None:
    p_value = raw
    if isinstance(p_value, list):
        p_value = p_value[0] if len(p_value) > 0 else None
    if isinstance(p_value, str):
        try:
            p_value = float(p_value)
        except ValueError:
            p_value = None
    return p_value


def extract_ingredients(
    edge: Mapping[str, Any], node_pubs: Mapping[str, int]
) -> dict[str, Any]:
    """The rankable raw values present on one KGX edge line:
    `num_publications`, `p_value`, `literature_coocurrence`, `affinity`,
    `confidence_score` (kept, never weighted, as in the worker).

    `node_pubs` maps a node CURIE to its publication count, read only for a
    literature co-occurrence edge.
    """
    ingredients: dict[str, Any] = {}

    pubs_raw = edge.get("publications") or edge.get("supporting_document")
    if pubs_raw is not None:
        ingredients["num_publications"] = len(_parse_publications(pubs_raw))

    if "evidence_count" in edge:
        ingredients["num_publications"] = edge["evidence_count"]

    for field in sorted(edge.keys()):
        if any(marker in field for marker in _P_VALUE_MARKERS):
            p_value = parse_p_value(edge[field])
            if p_value is not None:
                ingredients["p_value"] = p_value
            break

    if edge.get("predicate") == LIT_COOC_PREDICATE and "has_count" in edge:
        try:
            num_pubs = int(edge["has_count"])
        except (TypeError, ValueError):
            num_pubs = 0
        subject_pubs = node_pubs.get(edge.get("subject"), 0)
        object_pubs = node_pubs.get(edge.get("object"), 0)
        cov = (num_pubs / TOTAL_PUBS) - (subject_pubs / TOTAL_PUBS) * (
            object_pubs / TOTAL_PUBS
        )
        ingredients["literature_coocurrence"] = max(cov, 0.0) * TOTAL_PUBS

    if "affinity" in edge:
        ingredients["affinity"] = edge["affinity"]

    confidence = edge.get("tmkp_confidence_score")
    if confidence is None:
        confidence = edge.get("biolink:tmkp_confidence_score")
    if confidence is not None:
        ingredients["confidence_score"] = confidence

    return ingredients


# ---------------------------------------------------------------------------
# The metric bag: the evidence the confidence pricing reads
# ---------------------------------------------------------------------------

#: Top-level KGX fields captured verbatim when present.
SCALAR_METRIC_FIELDS = (
    "has_confidence_score",
    "dgidb_evidence_score",
    "dgidb_interaction_score",
    "diseases_confidence_score",
    "gene2phenotype_confidence_category",
    "max_research_phase",
    "clinical_approval_status",
    "FDA_regulatory_approvals",
    "number_of_cases",
    "has_count",
    "has_total",
    "has_percentage",
    "has_quotient",
    "has_evidence",
    "qualifier",
    "evidence_count",
    "negated",
    "has_evidence_of_type",
)

#: Fields of a supporting-study result that are free text or identifiers.
_STUDY_SKIP_FIELDS = frozenset({
    "id", "name", "category", "xref", "has_study_results", "supporting_text",
    "supporting_documents", "supporting_document", "supporting_text_located_in",
    "subject_location_in_text", "object_location_in_text",
})

#: A string longer than this in a study result is prose, not a category.
_STUDY_CATEGORY_MAX_LENGTH = 40

#: Distinct values kept per categorical study field.
_STUDY_CATEGORY_MAX_VALUES = 32


def _publication_id(pub: Any) -> Any:
    """A PMID as its integer, anything else as given."""
    if isinstance(pub, str) and pub.startswith("PMID:"):
        digits = pub[5:]
        if digits.isdigit():
            return int(digits)
    return pub


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _affinity_measurements(raw: Any) -> list[list[Any]]:
    """BindingDB's `has_affinity` as [parameter, value, relation] per
    measurement with a numeric value."""
    items = raw if isinstance(raw, list) else [raw]
    out = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        value = item.get("affinity")
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                value = None
        value = _numeric(value)
        if value is None:
            continue
        out.append([item.get("affinity_parameter"), value,
                    item.get("has_binary_relation")])
    return out


def _supporting_studies(raw: Any) -> list[Mapping[str, Any]]:
    """The Study objects in `has_supporting_studies`, a dict keyed by study
    id or a list."""
    studies = list(raw.values()) if isinstance(raw, Mapping) else (
        raw if isinstance(raw, list) else [])
    return [study for study in studies if isinstance(study, Mapping)]


def _study_results(raw: Any) -> tuple[int, list[Mapping[str, Any]]]:
    """(studies, result records): a study's `has_study_results`, or the
    study itself when it lists none."""
    studies = _supporting_studies(raw)
    records: list[Mapping[str, Any]] = []
    for study in studies:
        results = study.get("has_study_results")
        if isinstance(results, list) and results:
            records.extend(r for r in results if isinstance(r, Mapping))
        else:
            records.append(study)
    return len(studies), records


def _interval(value: Any) -> tuple[float, float] | None:
    """A two-number list -- a confidence interval -- as its bounds."""
    if isinstance(value, list) and len(value) == 2:
        low, high = _numeric(value[0]), _numeric(value[1])
        if low is not None and high is not None:
            return low, high
    return None


def _fold_number(fields: dict[str, dict], name: str, number: float) -> None:
    agg = fields.setdefault(name, {"n": 0, "min": number, "max": number, "sum": 0.0})
    if "n" not in agg:
        return  # already categorical under this name
    agg["n"] += 1
    agg["min"] = min(agg["min"], number)
    agg["max"] = max(agg["max"], number)
    agg["sum"] += number


def _aggregate_study_fields(records: list[Mapping[str, Any]]) -> dict:
    """Per field over the study results: numbers as n/min/max/sum, an
    interval as one aggregate per bound, short strings as a histogram."""
    fields: dict[str, dict] = {}
    for record in records:
        for name, value in record.items():
            if name in _STUDY_SKIP_FIELDS:
                continue
            number = _numeric(value)
            if number is not None:
                _fold_number(fields, name, number)
                continue
            bounds = _interval(value)
            if bounds is not None:
                _fold_number(fields, f"{name}[0]", bounds[0])
                _fold_number(fields, f"{name}[1]", bounds[1])
                continue
            values = value if isinstance(value, list) else [value]
            for item in values:
                if not isinstance(item, str) or len(item) > _STUDY_CATEGORY_MAX_LENGTH:
                    continue
                agg = fields.setdefault(name, {"values": {}})
                if "values" not in agg:
                    continue  # already numeric under this name
                _count_value(agg, item)
    return fields


def _count_value(agg: dict, item: str, count: int = 1) -> None:
    histogram = agg["values"]
    if item in histogram:
        histogram[item] += count
    elif len(histogram) < _STUDY_CATEGORY_MAX_VALUES:
        histogram[item] = count
    else:
        agg["overflow"] = True


def extract_metrics(edge: Mapping[str, Any]) -> dict[str, Any]:
    """The evidence an edge carries beyond the profile's ingredients: each
    `SCALAR_METRIC_FIELDS` field and every p-value-like field verbatim,
    BindingDB's affinity measurements, the supporting studies' result
    fields, and the publication ids."""
    metrics: dict[str, Any] = {}
    for field in SCALAR_METRIC_FIELDS:
        value = edge.get(field)
        if value is None:
            value = edge.get(f"biolink:{field}")
        if value is not None:
            metrics[field] = value
    for field in sorted(edge.keys()):
        if any(marker in field for marker in _P_VALUE_MARKERS):
            metrics[field] = edge[field]

    affinity = edge.get("has_affinity")
    if affinity is not None:
        measurements = _affinity_measurements(affinity)
        if measurements:
            metrics["affinity_measurements"] = measurements

    supporting = edge.get("has_supporting_studies")
    if supporting:
        studies, records = _study_results(supporting)
        if records:
            metrics["studies"] = studies
            metrics["study_results"] = len(records)
            fields = _aggregate_study_fields(records)
            if fields:
                metrics["study_fields"] = fields

    pubs_raw = edge.get("publications") or edge.get("supporting_document")
    if pubs_raw is not None:
        ids = [_publication_id(pub) for pub in _parse_publications(pubs_raw)]
        if ids:
            metrics["publication_ids"] = ids
    return metrics


def merge_metrics(existing: dict[str, Any], incoming: Mapping[str, Any]) -> None:
    """Fold a duplicate record's metric bag into `existing`, in place:
    publication ids and affinity measurements union, study counts add, study
    field aggregates combine, and a scalar present in both keeps the first
    line's value."""
    for name, value in incoming.items():
        if name not in existing:
            existing[name] = value
            continue
        if name in ("publication_ids", "affinity_measurements"):
            seen = {_hashable(item) for item in existing[name]}
            for item in value:
                if _hashable(item) not in seen:
                    existing[name].append(item)
                    seen.add(_hashable(item))
        elif name in ("studies", "study_results"):
            existing[name] += value
        elif name == "study_fields":
            _merge_study_fields(existing[name], value)


def _hashable(item: Any) -> Any:
    """A hashable stand-in for a JSON value (lists become tuples)."""
    return tuple(item) if isinstance(item, list) else item


def _merge_study_fields(existing: dict, incoming: Mapping[str, dict]) -> None:
    for name, agg in incoming.items():
        mine = existing.get(name)
        if mine is None:
            existing[name] = agg
        elif "n" in mine and "n" in agg:
            mine["n"] += agg["n"]
            mine["min"] = min(mine["min"], agg["min"])
            mine["max"] = max(mine["max"], agg["max"])
            mine["sum"] += agg["sum"]
        elif "values" in mine and "values" in agg:
            for item, count in agg["values"].items():
                _count_value(mine, item, count)
            if agg.get("overflow"):
                mine["overflow"] = True


# ---------------------------------------------------------------------------
# Weights and their combination
# ---------------------------------------------------------------------------


def edge_property_weights(
    ingredients: Mapping[str, Any],
    primary_source: str,
    profile: Mapping[str, Any],
    *,
    knowledge_level: str | None = None,
    agent_type: str | None = None,
) -> dict[str, float]:
    """Per-property weights in [0, 1) for one edge: a base weight for its
    source, sigmoid(value) times the source's weight for each ingredient,
    and the profile's knowledge-level and agent-type weights where it has
    them (an absent value weighted as `not_provided`)."""
    source_weights = profile["source_weights"]
    unknown_source_weight = profile["unknown_source_weight"]
    source_transformation = profile["source_transformation"]
    unknown_source_transformation = profile["unknown_source_transformation"]

    def weighted(property: str, value: float) -> float:
        property_w = get_source_sigmoid(
            value,
            primary_source,
            property,
            source_transformation,
            unknown_source_transformation,
        )
        source_w = get_source_weight(
            primary_source, property, source_weights, unknown_source_weight
        )
        return property_w * source_w

    weights = {"base_weight": get_base_weight(primary_source, profile["base_weights"])}
    for field, value in (
        ("knowledge_level", knowledge_level), ("agent_type", agent_type)
    ):
        table = profile.get(f"{field}_weights")
        if not table:
            continue
        weight = table.get(value if value is not None else "not_provided")
        if weight is None:
            weight = table.get("default_weight")
        if weight is not None:
            weights[field] = weight
    if ingredients.get("p_value") is not None:
        weights["p_value"] = weighted("p_value", ingredients["p_value"])
    if ingredients.get("num_publications"):
        weights["publications"] = weighted(
            "publications", ingredients["num_publications"]
        )
    if ingredients.get("literature_coocurrence") is not None:
        weights["literature_coocurrence"] = weighted(
            "literature_co-occurrence", ingredients["literature_coocurrence"]
        )
    if ingredients.get("affinity") is not None:
        weights["affinity"] = weighted("affinity", ingredients["affinity"])
    return weights


def admittance_from_weights(weights: Mapping[str, float]) -> float:
    """ELROND's combination: each weight w as the admittance -1/ln(w),
    clamped to MAX_WEIGHT, summed; weights <= 0 contribute nothing."""
    total = 0.0
    for w in weights.values():
        if w <= 0:
            continue
        total += -1.0 / math.log(min(w, MAX_WEIGHT))
    return total


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def _hash_key(*parts: str) -> bytes:
    return hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=16).digest()


def _qualifiers_token(qualifiers: tuple[tuple[str, str], ...]) -> str:
    return "\x1e".join(f"{name}={value}" for name, value in sorted(qualifiers))


def edge_key(
    subject: str,
    predicate: str,
    object_: str,
    qualifiers: tuple[tuple[str, str], ...],
    primary_source: str,
    knowledge_level: str | None,
    agent_type: str | None,
) -> bytes:
    """One Translator edge, in its canonical direction."""
    return _hash_key(
        "edge",
        subject,
        predicate,
        object_,
        _qualifiers_token(qualifiers),
        primary_source,
        knowledge_level or "",
        agent_type or "",
    )


def relationship_key(
    subject: str,
    predicate: str,
    object_: str,
    qualifiers: tuple[tuple[str, str], ...],
) -> bytes:
    """A qualified relationship, merged across source, level and agent."""
    return _hash_key(
        "relationship", subject, predicate, object_, _qualifiers_token(qualifiers)
    )
