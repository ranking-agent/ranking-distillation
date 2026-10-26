"""The Translator edge-confidence framework as a pricing, in log-odds.

The framework prices an edge as

    L  = logit(prior) + sum over metrics of M * R * Q * S
    CW = 1 / (1 + exp(-L))

with `M` a metric's value normalised onto a signed scale by curator anchors,
`R` its relevance to edge truth, `Q` its quality and `S = ln(max odds
effect)` the most it may move the odds. The search wants an additive,
non-negative cost, and `-ln(CW)` is one; it is written as the admittance
`-1 / ln(CW)`, so the backbone's evidence term `-ln(exp(-1/admittance))` is
exactly `-ln(CW)`.

`pricing/confidence.json` keeps ELROND's combination of the profile's
ingredients as the starting log-odds (`ingredients: elrond`), adds the
curated metrics' `M * R * Q * S` on top, and multiplies the odds of HPOA's
phenotype edges by a source reliability factor. A curated metric that
`replaces` one of ELROND's ingredients takes its place for its sources. A
missing metric contributes nothing.

A continuous metric maps to `M = cap * tanh(k * (t(x) - neutral))` on each
side of its neutral value, `t` its pre-transform, with `k` solved from the
side's anchor `(x, M)`; a side without an anchor gives 0. Ordered categories
and small discrete counts map by table.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .evidence import admittance_from_weights, edge_property_weights, parse_p_value

#: The weights an edge carries whatever its evidence: its prior under ELROND.
PRIOR_TERMS = ("base_weight", "knowledge_level", "agent_type")


def load_config(path: str | Path) -> dict:
    """A confidence config, refused unless it prices as this module does."""
    config = json.loads(Path(path).read_text())
    if config.get("prior") != "elrond" or config.get("ingredients") != "elrond":
        raise ValueError(f"{path} must start from ELROND's prior and "
                         "combination (prior and ingredients both 'elrond')")
    return config


def _logit_of_admittance(admittance: float) -> float:
    """logit(exp(-1/A)), computed without forming p near 0 or 1."""
    log_p = -1.0 / admittance
    return log_p - math.log(-math.expm1(log_p))


def admittance_of_log_odds(log_odds: float) -> float:
    """-1 / ln(sigmoid(L)): the admittance whose ELROND evidence is CW."""
    #  -ln(sigmoid(L)) = softplus(-L), written to stay finite for large |L|.
    softplus = (math.log1p(math.exp(-log_odds)) if log_odds > -30
                else -log_odds)
    return 1.0 / softplus


def _scalar(value: Any, name: str) -> float:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"metric {name!r} expects a number, got {value!r}")
    return float(value)


def _transform(value: Any, spec: Mapping[str, Any]) -> float:
    if spec.get("transform", "none") == "neg_log10":
        p = parse_p_value(value)
        if p is None:
            raise ValueError(f"metric {spec['name']!r} expects a p-value, got {value!r}")
        return -math.log10(max(p, spec["floor"]))
    return _scalar(value, spec["name"])


def _side(t: float, neutral: float, side: Mapping[str, Any] | None) -> float:
    """cap * tanh(k * (t - neutral)), k solved from the side's anchor."""
    if side is None:
        return 0.0
    (anchor_x, anchor_m), cap = side["anchor"], side["cap"]
    k = math.atanh(anchor_m / cap) / (anchor_x - neutral)
    return cap * math.tanh(k * (t - neutral))


def normalize(value: Any, spec: Mapping[str, Any]) -> float:
    """A metric's raw value on the framework's signed scale, M."""
    if "categories" in spec:
        key = str(value).lower()
        if key not in spec["categories"]:
            raise ValueError(f"metric {spec['name']!r} has no mapping for "
                             f"{value!r}; known: {sorted(spec['categories'])}")
        return float(spec["categories"][key])
    if "steps" in spec:
        x = _scalar(value, spec["name"])
        m = 0.0
        for threshold, step in spec["steps"]:
            if x >= threshold:
                m = step
        return m
    t = _transform(value, spec)
    neutral = spec["neutral"]
    return _side(t, neutral, spec.get("positive") if t >= neutral
                 else spec.get("negative"))


def contribution(value: Any, spec: Mapping[str, Any]) -> float:
    """M * R * Q * S in log-odds."""
    return normalize(value, spec) * spec["R"] * spec["Q"] * math.log(spec["max_odds"])


def _applies(spec: Mapping[str, Any], source: str, predicate: str | None,
             ends: tuple[str, ...]) -> bool:
    """Whether a spec prices this edge: its source and, where the spec names
    them, its predicate and a CURIE prefix one of its endpoints must have."""
    if source not in spec["sources"]:
        return False
    if "predicates" in spec and predicate not in spec["predicates"]:
        return False
    return "endpoint_prefix" not in spec or spec["endpoint_prefix"] in ends


def edge_admittance(
    ingredients: Mapping[str, Any],
    metrics: Mapping[str, Any],
    primary_source: str,
    profile: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    knowledge_level: str | None,
    agent_type: str | None,
    predicate: str,
    subject: str,
    object_: str,
) -> float:
    """-1 / ln(CW) for one edge: ELROND's log-odds for the whole edge, less
    any ingredient a curated metric replaces, plus the curated metrics."""
    weights = edge_property_weights(
        ingredients, primary_source, profile,
        knowledge_level=knowledge_level, agent_type=agent_type,
    )
    prior = admittance_from_weights({k: w for k, w in weights.items()
                                     if k in PRIOR_TERMS})
    if prior <= 0:
        raise ValueError(f"{primary_source} has no prior under this profile "
                         f"(its base weight is 0), which log-odds cannot "
                         f"start from")
    ends = tuple(curie.split(":", 1)[0] for curie in (subject, object_) if curie)
    applies = [spec for spec in config["metrics"]
               if _applies(spec, primary_source, predicate, ends)]
    replaced = {spec["replaces"] for spec in applies if spec.get("replaces")}
    whole = admittance_from_weights({k: w for k, w in weights.items()
                                     if k not in replaced})
    terms = {"elrond": _logit_of_admittance(whole)}
    for spec in applies:
        if "srf" in spec:
            #  The framework's source reliability factor: an odds multiplier
            #  on every edge of the source (and edge type) it names.
            terms[spec["name"]] = math.log(spec["srf"])
            continue
        bag = ingredients if spec["from"] == "ingredients" else metrics
        value = bag.get(spec["field"])
        if value is not None:
            terms[spec["name"]] = contribution(value, spec)
    return admittance_of_log_odds(sum(terms.values()))
