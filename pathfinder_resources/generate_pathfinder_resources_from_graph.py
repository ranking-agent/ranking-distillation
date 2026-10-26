"""Generate everything shepherd's pathfinder needs for one KGX release.

    <out>/<build date>_v<release version>/
        pathfinder_search_graph_<build date>_v<search graph version>/
        pathfinder_embeddings_<build date>_v<embeddings version>/

from a release URL (kgx-storage's, e.g.
https://kgx-storage.ci.transltr.io/releases/translator_kg/1.0.2/), whose files
are downloaded, or a directory holding them already: release-metadata.json
and the graph, as `translator_kg.tar.zst` or already unpacked to nodes.jsonl
and edges.jsonl. The dates and versions come from release-metadata.json; the
two parts carry their own version numbers (`build_pathfinder_search_graph.py`,
`build_pathfinder_embeddings.py`).

Everything else is kept in `--work`, so an interrupted run resumes where it
stopped. `--tarballs` also writes each part as the .tar.gz shepherd
downloads, its files at the top level.
"""

from __future__ import annotations

import argparse
import re
import tarfile
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

import build_pathfinder_embeddings
import build_pathfinder_search_graph
from pathfinder_build.download import download, extract
from pathfinder_build.kgx import (
    EDGES,
    NODES,
    RELEASE_METADATA,
    build_day,
    release_metadata,
)
from pathfinder_build.stages import is_done, mark_done

#  What each part's tarball holds, at its top level
TARBALL_FILES = {
    "search_graph": ("offsets.npy", "neighbors.npy", "weights.npy", "nodes.txt",
                     "meta.json"),
    "embeddings": ("data.mdb",),
}


def release_files(url: str) -> list[str]:
    """The URLs of the files a release directory's listing links to."""
    with urllib.request.urlopen(url) as resp:
        listing = resp.read().decode()
    files = []
    for href in re.findall(r'href="([^"]+)"', listing):
        target = urljoin(url, href)
        name = target[len(url):]
        if target.startswith(url) and name and "/" not in name and "?" not in name:
            files.append(target)
    if not files:
        raise SystemExit(f"{url} lists no files")
    return sorted(set(files))


def fetch_release(url: str, work: Path) -> Path:
    """Every file of the release at `url`, downloaded into `work/release`."""
    url = url if url.endswith("/") else url + "/"
    release = work / "release"
    for file_url in release_files(url):
        dest = release / file_url.rsplit("/", 1)[-1]
        if not dest.exists():
            download(file_url, dest)
    return release


def jsonl_files(release: Path, metadata: dict, work: Path) -> tuple[Path, Path]:
    """`(nodes.jsonl, edges.jsonl)`: beside the release's files, or unpacked
    from its archive into `work/kgx` once."""
    if (release / NODES).exists() and (release / EDGES).exists():
        return release / NODES, release / EDGES
    archives = sorted(release.glob("*.tar.zst"))
    if len(archives) != 1:
        raise SystemExit(f"{release} holds neither {NODES} and {EDGES} nor one "
                         f".tar.zst to unpack them from")
    unpacked = work / "kgx"
    inputs = {"archive": archives[0].name,
              "build_version": metadata["build_version"],
              "release_version": metadata["release_version"]}
    if not is_done(unpacked, inputs):
        print(f"unpacking {archives[0]} ...")
        extract(archives[0], unpacked)
        mark_done(unpacked, inputs)
    found = [sorted(unpacked.rglob(name)) for name in (NODES, EDGES)]
    if any(len(paths) != 1 for paths in found):
        raise SystemExit(f"{archives[0]} did not hold one {NODES} and one {EDGES}")
    return found[0][0], found[1][0]


def tarball(directory: Path, members: tuple[str, ...]) -> Path:
    """`<directory>.tar.gz` holding `members` at its top level."""
    out = directory.with_name(directory.name + ".tar.gz")
    if out.exists():
        return out
    partial = out.with_name(out.name + ".partial")
    with tarfile.open(partial, "w:gz") as tar:
        for name in members:
            tar.add(directory / name, arcname=name)
    partial.replace(out)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph",
                    help="a KGX release URL, or a directory holding its files")
    ap.add_argument("--out", type=Path, required=True,
                    help="where the release's resource directory is written")
    ap.add_argument("--work", type=Path, required=True,
                    help="where everything in between is kept, so a run "
                         "resumes; tens of GB")
    ap.add_argument("--omnicorp", required=True,
                    help="omnicorp's LMDB directory (shared_counts.lmdb and "
                         "curies.lmdb), or the URL of its .tar.gz")
    ap.add_argument("--model", type=Path,
                    help="the reranker to serve: an earlier pathfinder "
                         "embeddings directory, or a checkpoint a retrain "
                         "wrote. Required unless --graded-paths")
    ap.add_argument("--graded-paths", type=Path,
                    help="a directory of labelled parquets to retrain the "
                         "reranker on, instead of --model")
    ap.add_argument("--graded-queries", type=Path, nargs="+",
                    help="with --graded-paths: query TSVs (source, target, "
                         "'|'-separated expected nodes) whose queries with "
                         "expected nodes may be held out to choose the "
                         "checkpoint")
    ap.add_argument("--gandalf-graph", type=Path,
                    help="a gandalf graph already loaded from this release, by "
                         "the gandalf-csr version requirements.txt installs, "
                         "instead of loading it again")
    ap.add_argument("--embedding-cache", type=Path,
                    help="an embedding cache an earlier run wrote "
                         "(<work>/embeddings_v1/embedding_cache): its texts "
                         "are not embedded again")
    ap.add_argument("--tarballs", action="store_true",
                    help="also write each part as a .tar.gz beside it")
    ap.add_argument("--device",
                    help="where SAPBERT and training run (default: a GPU "
                         "when there is one)")
    args = ap.parse_args(argv)
    if (args.model is None) == (args.graded_paths is None):
        ap.error("give --model or --graded-paths, one of them")

    args.work.mkdir(parents=True, exist_ok=True)
    release = (fetch_release(args.graph, args.work)
               if args.graph.startswith(("http://", "https://"))
               else Path(args.graph))
    if not (release / RELEASE_METADATA).exists():
        raise SystemExit(f"{release} holds no {RELEASE_METADATA}")
    metadata = release_metadata(release)
    day = build_day(metadata)
    nodes, edges = jsonl_files(release, metadata, args.work)
    out = args.out / f"{day}_v{metadata['release_version'].replace('.', '_')}"
    out.mkdir(parents=True, exist_ok=True)

    search_graph = build_pathfinder_search_graph.build(
        nodes, edges, metadata, args.omnicorp, args.work,
        out / build_pathfinder_search_graph.search_graph_name(day))
    embeddings = build_pathfinder_embeddings.build(
        nodes, edges, metadata, args.work,
        out / build_pathfinder_embeddings.embeddings_name(day),
        model=args.model, graded_paths=args.graded_paths,
        graded_queries=args.graded_queries, gandalf_graph=args.gandalf_graph,
        embedding_cache=args.embedding_cache, device=args.device)
    if args.tarballs:
        for part, directory in (("search_graph", search_graph),
                                ("embeddings", embeddings)):
            print(f"wrote {tarball(directory, TARBALL_FILES[part])}")
    print(f"done: {out}")


if __name__ == "__main__":
    main()
