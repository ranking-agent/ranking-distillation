"""The words a path's hops and categories are embedded as.

A hop's edge reads as a phrase in the direction walked:

1. a plain predicate: `treats`, walked backwards `treated by`;
2. a qualified predicate with an aspect and a direction:
   `causes decreased expression of`, backwards
   `has decreased expression caused by`;
3. an aspect alone, read with the edge's own predicate:
   `affects expression of`, backwards `has expression affected by`.

Subject-side qualifiers and a direction with no aspect have no wording; such
an edge reads by its bare predicate. Context qualifiers (species, anatomy,
...) never change how a hop reads.

A hop lists its edges as labels: a plain edge as its predicate CURIE read in
the direction walked, a qualified one with its qualifiers on the side of the
node it reaches --

    biolink:causes{object_aspect_qualifier=activity,object_direction_qualifier=decreased}
    biolink:caused_by{subject_aspect_qualifier=activity,subject_direction_qualifier=decreased}

-- the second being the first walked backwards. A hop whose edges read
several ways is the bracketed phrase `[a or b]`, sorted. These strings, and a
category's bare class name, are what the embeddings are keyed by; the
score_paths worker reads a path into the same strings.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .biolink import get_inverse_predicate, name_to_predicate_curie, predicate_to_name

# The token budget a hop phrase, a category and a node name are embedded at.
# A multi-predicate hop reaches 16 words; truncating near there would give
# different hops one vector.
PHRASE_MAX_LENGTH = 32


class UnsupportedQualifierPattern(ValueError):
    """An edge uses a qualifier combination that has no wording."""


@dataclass(frozen=True)
class EdgeSpec:
    """What an edge's phrase is made of: the predicate it reads with (the
    qualified predicate when there is a direction), and an aspect with a
    direction or not."""

    predicate: str  # biolink CURIE
    object_aspect: str | None = None
    object_direction: str | None = None

    def __post_init__(self) -> None:
        if self.object_direction is not None and self.object_aspect is None:
            raise UnsupportedQualifierPattern(
                f"object_direction needs an object_aspect "
                f"(got direction={self.object_direction!r} and no aspect)")


def unworded_pattern(qualifiers: Mapping[str, object]) -> str | None:
    """What an edge's qualifiers say that no pattern words, or None."""
    if (qualifiers.get("subject_aspect_qualifier") is not None
            or qualifiers.get("subject_direction_qualifier") is not None):
        return "subject-side qualifiers"
    if (qualifiers.get("object_direction_qualifier") is not None
            and qualifiers.get("object_aspect_qualifier") is None):
        return "a direction but no aspect"
    return None


#  The slots that decide how an edge reads
READING_SLOTS = (
    "qualified_predicate", "object_aspect_qualifier",
    "object_direction_qualifier", "subject_aspect_qualifier",
    "subject_direction_qualifier",
)


def edge_spec(predicate: str, qualifiers: Mapping[str, object]) -> EdgeSpec:
    """The EdgeSpec an edge reads as; raises `UnsupportedQualifierPattern`
    for the patterns with no wording."""
    pattern = unworded_pattern(qualifiers)
    if pattern is not None:
        raise UnsupportedQualifierPattern(
            f"an edge with {pattern} has no wording; fix at source")
    aspect = qualifiers.get("object_aspect_qualifier")
    direction = qualifiers.get("object_direction_qualifier")
    qualified = qualifiers.get("qualified_predicate")
    if qualified is not None and (direction is not None or aspect is None):
        predicate = name_to_predicate_curie(predicate_to_name(str(qualified)))
    return EdgeSpec(
        predicate=predicate,
        object_aspect=None if aspect is None else str(aspect),
        object_direction=None if direction is None else str(direction),
    )


def hop_spec(predicate: str, qualifiers: Mapping[str, object]) -> EdgeSpec:
    """The spec a hop reads an edge as: its bare predicate where its
    qualifiers have no wording."""
    if unworded_pattern(qualifiers) is not None:
        return EdgeSpec(predicate)
    return edge_spec(predicate, qualifiers)


def _worded(verb: str, aspect: str | None, direction: str | None,
            walked_back: bool) -> str:
    """The phrase for an edge whose `verb` is already the predicate read in
    the direction walked."""
    word = predicate_to_name(verb)
    if aspect is None:
        return word
    change = aspect.replace("_", " ")
    if direction is not None:
        change = f"{direction.replace('_', ' ')} {change}"
    return f"has {change} {word}" if walked_back else f"{word} {change} of"


#  A value goes between "=" and "," in a label, so it may hold neither, nor
#  a brace.
_VALUE = re.compile(r"[\w.:-]+( [\w.:-]+)*")
_QUALIFIED = re.compile(r"(?P<verb>[^{}]+)\{(?P<qualifiers>[^{}]*)\}")


def _label(verb: str, side: str, aspect: str, direction: str | None) -> str:
    parts = [(f"{side}_aspect_qualifier", aspect)]
    if direction is not None:
        parts.append((f"{side}_direction_qualifier", direction))
    for slot, value in parts:
        if not _VALUE.fullmatch(value):
            raise ValueError(
                f"{slot}={value!r} cannot be written in a hop label; a "
                "qualifier value is expected to be a Biolink enum value")
    return verb + "{" + ",".join(f"{slot}={value}" for slot, value in parts) + "}"


def _parsed(label: str) -> tuple[str, str | None, str | None, str | None]:
    """`(verb, side, aspect, direction)`; side is None for a plain label."""
    match = _QUALIFIED.fullmatch(label)
    if match is None:
        if "{" in label or "}" in label:
            raise ValueError(f"{label!r} is not a hop label")
        return label, None, None, None
    fields: dict[str, str] = {}
    for item in match["qualifiers"].split(","):
        slot, sep, value = item.partition("=")
        if not sep or slot in fields:
            raise ValueError(f"{label!r} is not a hop label")
        fields[slot] = value
    sides = {slot.split("_", 1)[0] for slot in fields}
    side = sides.pop() if len(sides) == 1 else None
    aspect = fields.pop(f"{side}_aspect_qualifier", None)
    direction = fields.pop(f"{side}_direction_qualifier", None)
    if side not in ("object", "subject") or aspect is None or fields:
        raise ValueError(
            f"{label!r} is not a hop label: a qualified label carries an "
            "aspect, and a direction or not, all on one side")
    return match["verb"], side, aspect, direction


def hop_label(spec: EdgeSpec) -> str:
    """An edge walked as stored, as a hop lists it."""
    if spec.object_aspect is None:
        return spec.predicate
    return _label(spec.predicate, "object", spec.object_aspect,
                  spec.object_direction)


def label_phrase(label: str) -> str:
    """The phrase a hop label reads as."""
    verb, side, aspect, direction = _parsed(label)
    return _worded(verb, aspect, direction, side == "subject")


def reverse_label(label: str) -> str:
    """The same edge read the other way: the inverse predicate, and the
    qualifiers moved to the other side. Raises ValueError for a predicate the
    biolink model cannot invert."""
    verb, side, aspect, direction = _parsed(label)
    inverse = get_inverse_predicate(verb)
    if side is None:
        return inverse
    return _label(inverse, "object" if side == "subject" else "subject",
                  aspect, direction)


def predicates_phrase(predicates: Iterable[str]) -> str:
    """One hop's phrase: its one phrase, or several sorted as `[a or b]`."""
    preds = {label_phrase(p) for p in predicates}
    if not preds:
        raise ValueError("at least one predicate is required for a hop")
    deduped = sorted(preds)
    if len(deduped) == 1:
        return deduped[0]
    return "[" + " or ".join(deduped) + "]"


def row_hop_phrase(predicate_curies: Iterable[str]) -> str:
    """A graded path's hop, a list of predicates, as its phrase."""
    names = [label_phrase(p) for p in predicate_curies]
    seen: set[str] = set()
    deduped = [n for n in names if not (n in seen or seen.add(n))]
    return predicates_phrase(deduped)


def category_phrase(category: str) -> str:
    """A category as it is embedded: the bare class name."""
    if not category:
        raise ValueError("category is empty")
    return category.removeprefix("biolink:")
