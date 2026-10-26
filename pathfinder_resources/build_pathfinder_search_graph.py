"""Build the search graph shepherd's aragorn_pathfinder worker loads, from a
KGX release.

    pathfinder_search_graph_<build date>_v<VERSION>/
        offsets.npy     int64, one past each node's last neighbour slot
        neighbors.npy   int32, the neighbour in each slot
        weights.npy     float64, the pair's Tier-1 weight in each slot
        nodes.txt       node CURIE per index
        meta.json       what it was built from

Each edge of the KGX is priced by the Translator edge-confidence framework
(`pricing/confidence.json` over `pricing/profile.json`): ELROND's
combination of its evidence, the curated metrics, a reliability factor on
HPOA's phenotype edges, omnicorp's literature co-occurrence for its node
pair, and nothing for a NOT annotation. Its relationships then make the
mechanistic backbone and its Tier-1 weights (`pathfinder_build.backbone`).

Stages, each kept in the work directory so an interrupted build resumes:

    evidence   every edge's evidence, captured from edges.jsonl  (hours)
    priced     every relationship's priced admittance
    then the backbone, its weights and the search graph, in memory

`--omnicorp` is omnicorp's LMDB directory, or the URL of its tarball, which
is downloaded and unpacked into the work directory once.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pathfinder_build.backbone import (
    ALPHA,
    DROP_PREDICATES,
    KEEP_TOP_K,
    MIN_DEGREE,
    MIN_GROUP,
    QUALIFIER_MU,
    RESCUE_ALPHA,
    SIGNIFICANCE,
    SPECIFICITY_LAMBDA,
    build_backbone,
    pair_weights,
    relationship_table,
    search_csr,
)
from pathfinder_build.biolink import set_biolink_version
from pathfinder_build.confidence import load_config
from pathfinder_build.download import download, extract
from pathfinder_build.kgx import biolink_version, build_day
from pathfinder_build.omnicorp import omnicorp_files, open_shared_counts
from pathfinder_build.stages import is_done, mark_done
from pathfinder_build.weight_cache import (
    capture_evidence,
    capture_finished,
    iter_relationships,
    price_relationships,
)

#  This part of the build, by what it writes: a new number is a new
#  directory name beside the old.
VERSION = 1

PRICING = Path(__file__).resolve().parent / "pricing"
PROFILE = PRICING / "profile.json"
CONFIDENCE = PRICING / "confidence.json"


def search_graph_name(build_day: str) -> str:
    return f"pathfinder_search_graph_{build_day}_v{VERSION}"


def omnicorp_dir(omnicorp: str, work: Path) -> Path:
    """omnicorp's LMDB directory: as given, or its tarball downloaded and
    unpacked under `work`."""
    if not omnicorp.startswith(("http://", "https://")):
        return Path(omnicorp)
    unpacked = work / "omnicorp"
    inputs = {"url": omnicorp}
    if not is_done(unpacked, inputs):
        archive = work / omnicorp.rsplit("/", 1)[-1]
        download(omnicorp, archive)
        if unpacked.exists():
            shutil.rmtree(unpacked)
        extract(archive, unpacked)
        archive.unlink()
        mark_done(unpacked, inputs)
    return unpacked


def build(nodes: Path, edges: Path, metadata: dict, omnicorp: str,
          work: Path, out: Path) -> Path:
    """Write the search graph to `out`, resuming from `work`; returns `out`.

    A finished `out` is left as it is.
    """
    if out.exists():
        print(f"{out} exists; nothing to do")
        return out
    set_biolink_version(biolink_version(metadata))
    work = work / f"search_graph_v{VERSION}"
    work.mkdir(parents=True, exist_ok=True)
    release = {"build_version": metadata["build_version"],
               "release_version": metadata["release_version"],
               "search_graph_version": VERSION}

    evidence = work / "evidence"
    if not (is_done(evidence, release) and capture_finished(evidence)):
        print(f"capturing every edge's evidence from {edges} ...")
        capture_evidence(edges, nodes, evidence)
        mark_done(evidence, release)

    priced = work / "priced"
    if not is_done(priced, release):
        shared_counts_path, curies_path = omnicorp_files(
            omnicorp_dir(omnicorp, work.parent))
        shared_counts, node_pubs, omnicorp_stats = open_shared_counts(
            shared_counts_path, curies_path)
        print(f"omnicorp: {omnicorp_stats['curies']:,} curies")
        stats = price_relationships(
            evidence, priced, json.loads(PROFILE.read_text()),
            load_config(CONFIDENCE), shared_counts, node_pubs)
        print(f"priced {stats['edges_priced']:,} edges into "
              f"{stats['relationship_entries']:,} relationships; omnicorp "
              f"joined {omnicorp_stats['hits']:,} of "
              f"{omnicorp_stats['lookups']:,}")
        mark_done(priced, release)

    backbone = build_backbone(relationship_table(*iter_relationships(priced)))
    offsets, neighbors, weights = search_csr(backbone, pair_weights(backbone))
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    np.save(partial / "offsets.npy", offsets)
    np.save(partial / "neighbors.npy", neighbors)
    np.save(partial / "weights.npy", weights)
    (partial / "nodes.txt").write_text("\n".join(backbone.node_ids) + "\n")
    (partial / "meta.json").write_text(json.dumps({
        "version": VERSION,
        "release": metadata,
        "nodes": len(backbone.node_ids),
        "pairs": len(backbone.pair_a),
        "backbone": {"drop_predicates": list(DROP_PREDICATES),
                     "significance": SIGNIFICANCE, "min_degree": MIN_DEGREE,
                     "keep_top_k": KEEP_TOP_K, "rescue_alpha": RESCUE_ALPHA},
        "weights": {"alpha": ALPHA, "specificity_lambda": SPECIFICITY_LAMBDA,
                    "qualifier_mu": QUALIFIER_MU, "min_group": MIN_GROUP,
                    "normalize_on": "evidence", "promiscuity": "degree"},
        "built_utc": datetime.now(UTC).isoformat(),
    }, indent=2))
    partial.replace(out)
    print(f"search graph over {len(backbone.node_ids):,} nodes and "
          f"{len(backbone.pair_a):,} pairs -> {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", type=Path, required=True,
                    help="the release's nodes.jsonl")
    ap.add_argument("--edges", type=Path, required=True,
                    help="the release's edges.jsonl")
    ap.add_argument("--release-metadata", type=Path, required=True,
                    help="the release's release-metadata.json")
    ap.add_argument("--omnicorp", required=True,
                    help="omnicorp's LMDB directory (shared_counts.lmdb and "
                         "curies.lmdb), or the URL of its .tar.gz")
    ap.add_argument("--work", type=Path, required=True,
                    help="where the stages are kept, so a build resumes")
    ap.add_argument("--out", type=Path, required=True,
                    help="the directory that gets the search graph")
    args = ap.parse_args(argv)
    metadata = json.loads(args.release_metadata.read_text())
    build(args.nodes, args.edges, metadata, args.omnicorp, args.work,
          args.out / search_graph_name(build_day(metadata)))


if __name__ == "__main__":
    main()
