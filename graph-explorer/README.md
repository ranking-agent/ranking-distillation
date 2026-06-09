# GANDALF Explorer

A prototype GUI for walking the NIH Translator knowledge graph one triple at a
time, via the GANDALF TRAPI endpoint at RENCI (`https://gandalf.renci.org`).

Pick a starting node → see its neighbours grouped by relationship (predicate +
direction) as connectors fanning out to the right → click a relationship to list
the nodes reached through it (ranked by node degree) → click a node to step into
it. The chosen node becomes the root of the next triple; the path you build is
shown in the filmstrip up top, with a frame around the triple you're viewing.
Use **‹ Back / Forward ›** (or the arrow keys, or click any node in the
filmstrip) to swipe along the path.

## Run it

No dependencies — just Python 3.8+.

```bash
cd gandalf-explorer
python3 server.py
```

Then open <http://localhost:8000>.

The tiny `server.py` proxies the TRAPI calls (avoids CORS) and fans out the
node-degree lookups concurrently. Configuration is via environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GANDALF_PORT` | `8000` | local port |
| `GANDALF_BASE` | `https://gandalf.renci.org` | TRAPI server to query |
| `GANDALF_SUBCLASS` | `false` | enable biolink subclass inference on lookups |
| `GANDALF_TIMEOUT` | `60` | per-request timeout (seconds) |
| `GANDALF_DEGREE_WORKERS` | `8` | concurrent `/node_degree` lookups |

## How the queries work

For a node `X`, the backend issues two predicate-agnostic one-hop TRAPI queries
to `/query`:

- **outgoing:** `X --?--> n1`
- **incoming:** `n1 --?--> X`

Edges come back in `message.knowledge_graph`. They're grouped by
`(predicate, direction)`; each group's neighbour count and node list are sent to
the browser. When you open a relationship, the browser asks the backend for the
neighbours' degrees (`GET /node_degree/{curie}`, cached + parallel) and sorts the
list descending.

## Known limitations (it's a prototype)

- **Degree ranking is sampled.** A relationship can have thousands of neighbours;
  fetching a degree per node is one request each. The client degree-ranks the
  first `DEGREE_CAP` (300) of a relationship and labels the list accordingly. So
  the *true* highest-degree nodes of a huge hub may not all surface — the status
  line above each list is honest about this. A production version would push
  degree-sorting server-side (or have the KP return degrees in the lookup).
- **No name search.** Starting nodes are a curated example list plus a CURIE
  box. Real name→CURIE resolution would need the Translator Name Resolver /
  Node Normalizer (separate services, not in this KP's spec).
- **Two queries per node.** Direction matters in TRAPI, so neighbours are
  fetched once per direction. Results are cached for the session.
- `subclass` inference is **off** by default so each step shows that exact
  node's literal edges; flip `GANDALF_SUBCLASS=true` to include subclasses.

## Files

- `server.py` — stdlib proxy + static server
- `index.html`, `style.css`, `app.js` — the frontend
