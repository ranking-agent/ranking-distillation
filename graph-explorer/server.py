#!/usr/bin/env python3
"""
GANDALF Explorer — local backend.

A tiny, dependency-free server (Python standard library only) that sits between
the browser frontend and the GANDALF TRAPI knowledge-graph endpoint at RENCI.

Why a backend at all (instead of calling TRAPI straight from the browser)?
  - Avoids CORS headaches when hitting a third-party API.
  - Lets us fan out node-degree lookups concurrently and cache them.
  - Keeps the TRAPI request shaping in one place.

Run:
    python3 server.py
then open http://localhost:8000

Configuration via environment variables (all optional):
    GANDALF_BASE          base URL of the TRAPI server   (default https://gandalf.renci.org)
    GANDALF_PORT          local port to serve on          (default 8000)
    GANDALF_SUBCLASS      "true"/"false" subclass infer    (default false — literal neighbours)
    GANDALF_TIMEOUT       per-request timeout in seconds   (default 60)
    GANDALF_DEGREE_WORKERS  concurrent degree lookups      (default 8)
"""

import json
import os
import sys
import urllib.parse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
GANDALF_BASE = os.environ.get("GANDALF_BASE", "https://gandalf.renci.org").rstrip("/")
PORT = int(os.environ.get("GANDALF_PORT", "8000"))
SUBCLASS = os.environ.get("GANDALF_SUBCLASS", "false").lower() in ("1", "true", "yes")
TIMEOUT = float(os.environ.get("GANDALF_TIMEOUT", "60"))
DEGREE_WORKERS = int(os.environ.get("GANDALF_DEGREE_WORKERS", "8"))

STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}

# ---------------------------------------------------------------------------
# Simple in-memory caches (process lifetime)
# ---------------------------------------------------------------------------
_degree_cache = {}
_degree_lock = Lock()
_expand_cache = {}
_expand_lock = Lock()


# ---------------------------------------------------------------------------
# Outbound helpers
# ---------------------------------------------------------------------------
def _post_trapi(query):
    """POST a TRAPI message to GANDALF /query and return the parsed JSON."""
    url = f"{GANDALF_BASE}/query"
    data = json.dumps(query).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_degree(curie):
    """GET /node_degree/{curie}; cached. Returns int degree or None on failure."""
    with _degree_lock:
        if curie in _degree_cache:
            return _degree_cache[curie]
    url = f"{GANDALF_BASE}/node_degree/{urllib.parse.quote(curie, safe='')}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        degree = int(body.get("degree")) if body.get("degree") is not None else None
    except Exception:
        degree = None
    with _degree_lock:
        _degree_cache[curie] = degree
    return degree


def _trapi_neighbor_query(curie, root_is_subject):
    """Build a one-hop, predicate-agnostic neighbour query in one direction."""
    if root_is_subject:
        edge = {"subject": "n0", "object": "n1"}
    else:
        edge = {"subject": "n1", "object": "n0"}
    return {
        "message": {
            "query_graph": {
                "nodes": {
                    "n0": {"ids": [curie]},
                    "n1": {},  # unconstrained — match any neighbour
                },
                "edges": {"e0": edge},
            }
        },
        "parameters": {"subclass": SUBCLASS},
    }


def _node_label(kg_nodes, node_id):
    node = kg_nodes.get(node_id) or {}
    name = node.get("name") or node_id
    cats = node.get("categories") or []
    return name, cats


def expand(curie):
    """
    Fetch all neighbours of `curie` in both directions, grouped by
    (predicate, direction). Returns a dict ready for the frontend.
    """
    with _expand_lock:
        if curie in _expand_cache:
            return _expand_cache[curie]

    root_name, root_cats = curie, []
    # key -> {predicate, direction, neighbors: {id: {id,name,categories}}}
    groups = {}
    errors = []

    for root_is_subject in (True, False):
        direction = "out" if root_is_subject else "in"
        try:
            resp = _post_trapi(_trapi_neighbor_query(curie, root_is_subject))
        except urllib.error.HTTPError as e:
            errors.append(f"{direction}: HTTP {e.code}")
            continue
        except Exception as e:  # noqa: BLE001
            errors.append(f"{direction}: {e}")
            continue

        msg = (resp or {}).get("message", {}) or {}
        kg = msg.get("knowledge_graph") or {}
        kg_nodes = kg.get("nodes") or {}
        kg_edges = kg.get("edges") or {}

        # capture the root's own label if present
        if curie in kg_nodes:
            rn, rc = _node_label(kg_nodes, curie)
            root_name, root_cats = rn, rc

        for edge in kg_edges.values():
            subj = edge.get("subject")
            obj = edge.get("object")
            pred = edge.get("predicate") or "biolink:related_to"
            # neighbour is the endpoint that isn't the root
            if root_is_subject:
                neighbor = obj
            else:
                neighbor = subj
            if neighbor is None or neighbor == curie:
                continue
            key = f"{direction}|{pred}"
            grp = groups.get(key)
            if grp is None:
                grp = {"predicate": pred, "direction": direction, "neighbors": {}}
                groups[key] = grp
            if neighbor not in grp["neighbors"]:
                nname, ncats = _node_label(kg_nodes, neighbor)
                grp["neighbors"][neighbor] = {
                    "id": neighbor,
                    "name": nname,
                    "categories": ncats,
                }

    out_groups = []
    for key, grp in groups.items():
        neighbors = list(grp["neighbors"].values())
        out_groups.append(
            {
                "key": key,
                "predicate": grp["predicate"],
                "direction": grp["direction"],
                "count": len(neighbors),
                "neighbors": neighbors,
            }
        )
    # Sort predicate groups by neighbour count (descending) for a sensible default.
    out_groups.sort(key=lambda g: g["count"], reverse=True)

    result = {
        "curie": curie,
        "name": root_name,
        "categories": root_cats,
        "groups": out_groups,
        "errors": errors,
    }
    with _expand_lock:
        _expand_cache[curie] = result
    return result


def degrees(curies):
    """Fetch degrees for a list of curies concurrently. Returns {curie: degree|null}."""
    out = {}
    todo = [c for c in curies if c]
    if not todo:
        return out
    with ThreadPoolExecutor(max_workers=DEGREE_WORKERS) as pool:
        futs = {pool.submit(_get_degree, c): c for c in todo}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                out[c] = fut.result()
            except Exception:  # noqa: BLE001
                out[c] = None
    return out


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "GandalfExplorer/1.0"

    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write("  %s\n" % (fmt % args))

    # -- helpers ----------------------------------------------------------
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, filename, content_type):
        path = os.path.join(HERE, filename)
        try:
            with open(path, "rb") as f:
                body = f.read()
        except FileNotFoundError:
            self.send_error(404, "Not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- routes -----------------------------------------------------------
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path

        if route in STATIC:
            filename, ctype = STATIC[route]
            self._send_static(filename, ctype)
            return

        if route == "/api/expand":
            qs = urllib.parse.parse_qs(parsed.query)
            curie = (qs.get("curie") or [""])[0].strip()
            if not curie:
                self._send_json({"error": "missing 'curie' parameter"}, 400)
                return
            try:
                self._send_json(expand(curie))
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, 502)
            return

        if route == "/api/config":
            self._send_json(
                {"base": GANDALF_BASE, "subclass": SUBCLASS, "timeout": TIMEOUT}
            )
            return

        self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path

        if route == "/api/degrees":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                curies = payload.get("curies") or []
                if not isinstance(curies, list):
                    raise ValueError("'curies' must be a list")
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": f"bad request: {e}"}, 400)
                return
            try:
                self._send_json({"degrees": degrees(curies)})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, 502)
            return

        self.send_error(404, "Not found")


def main():
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"GANDALF Explorer running at  http://localhost:{PORT}")
    print(f"  proxying TRAPI -> {GANDALF_BASE}")
    print(f"  subclass inference: {SUBCLASS}")
    print("  press Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        httpd.shutdown()


if __name__ == "__main__":
    main()
