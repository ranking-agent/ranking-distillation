# Pathfinder resources from a KGX release

Builds the search graph and the embeddings database shepherd's pathfinder
workers load, for one KGX release, into
`<output_directory>/<build date>_v<release version>/`.

## Install

Python 3.11 or newer.

```
pip install -r requirements.txt
```

## Run

Serving an existing reranker:

```
python generate_pathfinder_resources_from_graph.py <kgx_release_url_or_directory> \
    --out <output_directory> \
    --work <work_directory> \
    --omnicorp <omnicorp_directory_or_tarball_url> \
    --model <embeddings_directory_or_checkpoint>
```

Retraining the reranker on graded paths instead:

```
python generate_pathfinder_resources_from_graph.py <kgx_release_url_or_directory> \
    --out <output_directory> \
    --work <work_directory> \
    --omnicorp <omnicorp_directory_or_tarball_url> \
    --graded-paths <graded_parquet_directory> \
    --graded-queries <query_tsv> [<query_tsv> ...]
```

Optional:

- `--gandalf-graph <gandalf_graph_directory>`: a gandalf graph already loaded from this release
- `--embedding-cache <embedding_cache_directory>`: an embedding cache an earlier run wrote
- `--tarballs`: also write each part as a `.tar.gz`
- `--device <device>`: where SAPBERT and training run

Run with the same `--work` to resume an interrupted run.
