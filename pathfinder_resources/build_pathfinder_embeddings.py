"""Build the embeddings LMDB shepherd's score_paths worker scores from, from
a KGX release.

    pathfinder_embeddings_<build date>_v<VERSION>/
        data.mdb    the reranker and everything it reads

Version 1 is the layout the deployed worker reads: the reranker as
TorchScript and its config, each node's, category's and hop phrase's SAPBERT
vector, and the node and edge context tables (`pathfinder_build.scorer_lmdb`).

The reranker is not retrained for a new graph unless asked: `--model` names
the one to serve, an earlier embeddings LMDB (its model and config are copied
in) or a checkpoint a retrain wrote (`<work>/embeddings_v1/training/student/
best.pt`). `--graded-paths` retrains it instead, on labelled parquets, by the
deployed recipe (`pathfinder_build.training`); `--graded-queries` names the
query TSVs whose expected nodes decide which queries are held out to choose
the checkpoint.

Stages, each kept in the work directory so an interrupted build resumes:

    gandalf_graph    the release loaded by gandalf (skipped with --gandalf-graph)
    node_context     degree and information content per node
    edge_context     edges, publications and sources per node pair
    embedding_cache  SAPBERT over every name, category and phrase (a GPU, in
                     practice), each text from --embedding-cache where it is
    training         three teachers and a student, with --graded-paths
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from pathfinder_build.biolink import set_biolink_version, toolkit
from pathfinder_build.embedding import ensure_cache, pick_device, sapbert
from pathfinder_build.gandalf_graph import (
    build_graph,
    edge_context,
    graph_nodes,
    graph_phrases,
    load_graph,
    node_context,
)
from pathfinder_build.kgx import biolink_version, build_day
from pathfinder_build.phrases import category_phrase
from pathfinder_build.reranker import (
    CHECKPOINT,
    deployment_problems,
    load_scorer,
    scripted,
)
from pathfinder_build.scorer_lmdb import read_model, write_lmdb
from pathfinder_build.stages import is_done, mark_done
from pathfinder_build.tables import EmbeddingCache, PairTable
from pathfinder_build.training import (
    expected_nodes,
    graded_rows,
    row_texts,
    train_recipe,
)

#  This part of the build, by what it writes: a new number is a new
#  directory name beside the old.
VERSION = 1


def embeddings_name(build_day: str) -> str:
    return f"pathfinder_embeddings_{build_day}_v{VERSION}"


def served_model(model: Path) -> tuple[bytes, dict]:
    """`(TorchScript bytes, config)` of the reranker at `model`: an
    embeddings LMDB, or a checkpoint (its file or its run directory)."""
    if (model / "data.mdb").exists():
        blob, source = read_model(model)
        config = {key: source[key] for key in (
            "embed_dim", "length_scale", "scalar_names", "node_context",
            "edge_context")}
        return blob, {**config, "checkpoint": source.get("checkpoint", str(model))}
    path = model / CHECKPOINT if model.is_dir() else model
    scorer, ckpt = load_scorer(path)
    problems = deployment_problems(ckpt)
    if problems:
        raise SystemExit(f"{path} cannot be served: " + "; ".join(problems))
    return scripted(scorer), {
        "embed_dim": ckpt["model_kwargs"]["embed_dim"],
        "length_scale": {str(n): [float(m), float(s)]
                         for n, (m, s) in ckpt["length_scale"].items()},
        "scalar_names": list(ckpt["scalar_names"]),
        "node_context": True,
        "edge_context": True,
        "checkpoint": str(path),
    }


def build(nodes: Path, edges: Path, metadata: dict, work: Path, out: Path, *,
          model: Path | None = None,
          graded_paths: Path | None = None,
          graded_queries: list[Path] | None = None,
          gandalf_graph: Path | None = None,
          embedding_cache: Path | None = None,
          device: str | None = None) -> Path:
    """Write the embeddings LMDB to `out`, resuming from `work`; returns
    `out`. A finished `out` is left as it is."""
    if out.exists():
        print(f"{out} exists; nothing to do")
        return out
    if (model is None) == (graded_paths is None):
        raise SystemExit("name the reranker to serve (--model) or the graded "
                         "paths to retrain it on (--graded-paths), one of them")
    if graded_queries and graded_paths is None:
        raise SystemExit("--graded-queries goes with --graded-paths")
    set_biolink_version(biolink_version(metadata))
    work = work / f"embeddings_v{VERSION}"
    work.mkdir(parents=True, exist_ok=True)
    release = {"build_version": metadata["build_version"],
               "release_version": metadata["release_version"],
               "embeddings_version": VERSION}

    if gandalf_graph is None:
        gandalf_graph = work / "gandalf_graph"
        if not is_done(gandalf_graph, release):
            if gandalf_graph.exists():
                shutil.rmtree(gandalf_graph)
            print("loading the release into gandalf ...")
            build_graph(nodes, edges, gandalf_graph)
            mark_done(gandalf_graph, release)
    graph = load_graph(gandalf_graph)

    node_table = work / "node_context"
    if not is_done(node_table, release):
        if node_table.exists():
            shutil.rmtree(node_table)
        print(f"node context over {node_context(graph, nodes, node_table):,} nodes")
        mark_done(node_table, release)
    edge_table = work / "edge_context"
    if not is_done(edge_table, release):
        if edge_table.exists():
            shutil.rmtree(edge_table)
        print(f"edge context over {edge_context(graph, edge_table):,} pairs")
        mark_done(edge_table, release)
    node_ctx, edge_ctx = EmbeddingCache.load(node_table), PairTable.load(edge_table)

    names, categories = graph_nodes(graph)
    category_texts = {c: category_phrase(c) for c in set(categories.values())}
    phrases = graph_phrases(graph)
    texts = {*names.values(), *category_texts.values(), *phrases}
    rows = graded_rows(graded_paths, names) if graded_paths else []
    texts |= row_texts(rows, names)
    device = device or pick_device()
    embeddings = ensure_cache(work / "embedding_cache", texts, sapbert(device),
                              embedding_cache)

    if graded_paths is not None:
        model = train_recipe(
            rows, expected_nodes(graded_queries) if graded_queries else None,
            names, embeddings, node_ctx, edge_ctx, work / "training", device)
    blob, config = served_model(model)
    if config["embed_dim"] != embeddings.dim:
        raise SystemExit(f"the reranker reads {config['embed_dim']}-d vectors "
                         f"and the embeddings are {embeddings.dim}-d")
    config.update(biolink_version=toolkit().get_model_version(),
                  version=VERSION, release=metadata,
                  built_utc=datetime.now(UTC).isoformat())

    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    held = write_lmdb(
        partial, blob, config, names=names, categories=categories,
        phrases=phrases, embeddings=embeddings, category_texts=category_texts,
        node_context=node_ctx if config["node_context"] else None,
        edge_context=edge_ctx if config["edge_context"] else None)
    (partial / "lock.mdb").unlink(missing_ok=True)
    partial.replace(out)
    print(f"wrote {out}: the reranker (lengths "
          f"{sorted(int(n) for n in config['length_scale'])}, biolink "
          f"{config['biolink_version']}), "
          + ", ".join(f"{n:,} {kind.replace('_', ' ')}"
                      for kind, n in held.items()))
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
    ap.add_argument("--work", type=Path, required=True,
                    help="where the stages are kept, so a build resumes")
    ap.add_argument("--out", type=Path, required=True,
                    help="the directory that gets the embeddings LMDB")
    ap.add_argument("--model", type=Path,
                    help="the reranker to serve: an earlier embeddings LMDB "
                         "directory, or a checkpoint (best.pt or its "
                         "directory). Required unless --graded-paths")
    ap.add_argument("--graded-paths", type=Path,
                    help="labelled parquets to retrain the reranker on")
    ap.add_argument("--graded-queries", type=Path, nargs="+",
                    help="query TSVs (source, target, '|'-separated expected "
                         "nodes) whose queries with expected nodes may be held "
                         "out to choose the retrained checkpoint")
    ap.add_argument("--gandalf-graph", type=Path,
                    help="a gandalf graph (memmap directory) already loaded "
                         "from this release, by the gandalf-csr version "
                         "requirements.txt installs, instead of loading it")
    ap.add_argument("--embedding-cache", type=Path,
                    help="an embedding cache an earlier build wrote "
                         "(<work>/embeddings_v1/embedding_cache): its texts "
                         "are not embedded again")
    ap.add_argument("--device",
                    help="where SAPBERT and training run (default: a GPU "
                         "when there is one)")
    args = ap.parse_args(argv)
    metadata = json.loads(args.release_metadata.read_text())
    build(args.nodes, args.edges, metadata, args.work,
          args.out / embeddings_name(build_day(metadata)), model=args.model,
          graded_paths=args.graded_paths, graded_queries=args.graded_queries,
          gandalf_graph=args.gandalf_graph,
          embedding_cache=args.embedding_cache, device=args.device)


if __name__ == "__main__":
    main()
