"""Read-only graph query sidecar for the Apple Notes Graphiti/Kuzu knowledge graph.

The notes UI (Bun) auto-spawns this and proxies /api/related to it. Kuzu is strictly
single-process, so this holds ONE read_only connection for its whole life and the UI
must be the only reader. No LLM / API key needed — pure Cypher.

Config (env):
  GRAPH_DB    path to the .kuzu file   (default: ~/dev/exp-notes-indexing/graphiti_notes.kuzu)
  GRAPH_PORT  port to listen on        (default: 3743)

Schema (graphiti-core + Kuzu): notes are :Episodic {name=title, content=body},
linked to :Entity via MENTIONS; entities link via RELATES_TO -> :RelatesToNode_ -> RELATES_TO.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import kuzu

DB_PATH = os.environ.get("GRAPH_DB") or str(Path.home() / "dev/exp-notes-indexing/graphiti_notes.kuzu")
PORT = int(os.environ.get("GRAPH_PORT", "3743"))

if not Path(DB_PATH).exists():
    raise SystemExit(f"graph DB not found: {DB_PATH}")

# one read-only connection for the process lifetime (Kuzu is single-process)
_conn = kuzu.Connection(kuzu.Database(DB_PATH, read_only=True))


def _rows(res):
    out = []
    while res.has_next():
        out.append(res.get_next())
    return out


def related_notes(title, limit=15):
    # exact title match first
    q = """
    MATCH (src:Episodic {name: $title})-[:MENTIONS]->(e1:Entity)
    OPTIONAL MATCH (e1)-[:RELATES_TO]->(:RelatesToNode_)-[:RELATES_TO]->(e2:Entity)
    WITH src, collect(DISTINCT e1.uuid) + collect(DISTINCT e2.uuid) AS ents
    UNWIND ents AS euid
    MATCH (other:Episodic)-[:MENTIONS]->(e:Entity {uuid: euid})
    WHERE other.uuid <> src.uuid
    RETURN other.name AS note, count(DISTINCT e) AS shared
    ORDER BY shared DESC LIMIT $limit
    """
    rows = _rows(_conn.execute(q, {"title": title, "limit": limit}))
    return [{"note": n, "shared": s} for n, s in rows]


def notes_for_entity(name, limit=15):
    q = """
    MATCH (e:Entity {name: $name})
    OPTIONAL MATCH (e)-[:RELATES_TO]->(:RelatesToNode_)-[:RELATES_TO]->(nb:Entity)
    WITH collect(DISTINCT e.uuid) + collect(DISTINCT nb.uuid) AS ents
    UNWIND ents AS euid
    MATCH (note:Episodic)-[:MENTIONS]->(x:Entity {uuid: euid})
    RETURN note.name AS note, count(DISTINCT x) AS hits
    ORDER BY hits DESC LIMIT $limit
    """
    rows = _rows(_conn.execute(q, {"name": name, "limit": limit}))
    return [{"note": n, "hits": h} for n, h in rows]


def entities_in_note(title, limit=20):
    q = """
    MATCH (:Episodic {name: $title})-[:MENTIONS]->(e:Entity)
    RETURN e.name AS name LIMIT $limit
    """
    return [r[0] for r in _rows(_conn.execute(q, {"title": title, "limit": limit}))]


def health():
    def c(q):
        try:
            return _rows(_conn.execute(q))[0][0]
        except Exception:
            return 0
    return {
        "ok": True, "db": DB_PATH,
        "episodes": c("MATCH (e:Episodic) RETURN count(e)"),
        "entities": c("MATCH (n:Entity) RETURN count(n)"),
        "edges": c("MATCH ()-[r:RELATES_TO]->() RETURN count(r)"),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        try:
            if u.path == "/health":
                self._send(200, health())
            elif u.path == "/related":
                title = (qs.get("title") or [""])[0]
                self._send(200, {"title": title, "related": related_notes(title), "entities": entities_in_note(title)})
            elif u.path == "/entity":
                name = (qs.get("name") or [""])[0]
                self._send(200, {"entity": name, "notes": notes_for_entity(name)})
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": str(e)})


if __name__ == "__main__":
    print(f"graph sidecar on http://127.0.0.1:{PORT} (db={DB_PATH})", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
