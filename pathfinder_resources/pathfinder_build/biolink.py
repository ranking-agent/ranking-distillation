"""Biolink model lookups: predicate inverse and symmetry, hierarchy depth,
category specificity.

Wraps the biolink-model-toolkit (`bmt`). KGX files store predicates and
categories as CURIEs (`biolink:treats`, `biolink:Gene`); the toolkit's slot
API wants the spaced name (`treats`, `has part`), so names are normalized on
the way in.

The model release is pinned to the one the graph was built against
(`set_biolink_version`, from the release's `release-metadata.json`): a bare
`Toolkit()` loads whatever bmt's default is, under which retired predicates
are unknown. Pin before the first lookup; changing it after the model has
loaded raises.
"""

from __future__ import annotations

from functools import lru_cache

from bmt import Toolkit

# Same tagged-release URLs gandalf uses, so one version string classifies
# alike across the stack.
_SCHEMA_URL = (
    "https://raw.githubusercontent.com/biolink/biolink-model/"
    "refs/tags/v{version}/biolink-model.yaml"
)
_PREDICATE_MAP_URL = (
    "https://raw.githubusercontent.com/biolink/biolink-model/"
    "refs/tags/v{version}/predicate_mapping.yaml"
)

_biolink_version: str | None = None


def set_biolink_version(version: str) -> None:
    """Pin the biolink model release. Call before any model lookup."""
    global _biolink_version
    version = version.lstrip("vV")
    if toolkit.cache_info().currsize and version != _biolink_version:
        raise RuntimeError(
            f"biolink model {_biolink_version or 'default'!r} is already "
            f"loaded; set version {version!r} before the first model lookup")
    _biolink_version = version


@lru_cache(maxsize=1)
def toolkit() -> Toolkit:
    if not _biolink_version:
        raise RuntimeError("no biolink model pinned; call set_biolink_version "
                           "with the graph's biolink_version first")
    return Toolkit(
        schema=_SCHEMA_URL.format(version=_biolink_version),
        predicate_map=_PREDICATE_MAP_URL.format(version=_biolink_version),
    )


def predicate_to_name(pred: str) -> str:
    """`biolink:has_part` or `has_part` -> `has part`."""
    pred = pred.removeprefix("biolink:")
    return pred.replace("_", " ")


def name_to_predicate_curie(name: str) -> str:
    """`treated by` -> `biolink:treated_by`."""
    return "biolink:" + name.replace(" ", "_")


def get_inverse_predicate(pred: str) -> str:
    """The inverse predicate as a CURIE.

    A predicate is its own inverse when it is symmetric; otherwise the model
    has to name one, and ValueError says it does not.
    """
    # Temporary workaround: fake inverse for has_chemical_role
    if pred in ("biolink:has_chemical_role", "has_chemical_role"):
        return "biolink:is_chemical_role_of"
    if pred in ("biolink:is_chemical_role_of", "is_chemical_role_of"):
        return "biolink:has_chemical_role"

    tk = toolkit()
    name = predicate_to_name(pred)
    if not tk.is_predicate(name):
        raise ValueError(f"Not a biolink predicate: {pred!r} (normalized {name!r})")
    if tk.is_symmetric(name):
        return name_to_predicate_curie(name)
    try:
        inv = tk.get_inverse(name)
    except AttributeError as exc:
        #  bmt hands the slot's declared inverse to linkml, which dereferences
        #  it without checking that it resolves; for a caller that is no
        #  inverse at all.
        raise ValueError(
            f"Predicate {pred!r} (normalized {name!r}) declares an inverse "
            f"that biolink model {tk.get_model_version()} cannot resolve: "
            f"{exc!r}") from exc
    if inv is None:
        raise ValueError(f"Predicate {pred!r} has no inverse and is not "
                         "symmetric")
    return name_to_predicate_curie(inv)


def try_predicate_depth(pred: str) -> int | None:
    """Depth of a predicate in the slot hierarchy (related_to is 0), or None
    for one the model does not know."""
    tk = toolkit()
    name = predicate_to_name(pred)
    if not tk.is_predicate(name):
        return None
    return tk.get_element_depth(name)


def most_specific_category(categories: list[str]) -> str:
    """The deepest biolink class among a node's categories, ties to the
    first listed; a node with mixins alone keeps its first mixin. Raises
    ValueError when none is a known category or mixin."""
    tk = toolkit()
    best: tuple[int, str] | None = None
    for cat in categories:
        if not tk.is_category(cat):
            continue
        depth = tk.get_element_depth(cat)
        if depth is None:
            continue
        if best is None or depth > best[0]:
            best = (depth, cat)
    if best is not None:
        return _normalize_category(best[1])
    mixins = [cat for cat in categories if tk.is_mixin(cat)]
    if mixins:
        return _normalize_category(mixins[0])
    raise ValueError(f"No biolink categories found in {categories!r}")


def _normalize_category(cat: str) -> str:
    """A category as a `biolink:ClassName` CURIE."""
    if cat.startswith("biolink:"):
        return cat
    return "biolink:" + cat.replace(" ", "")
