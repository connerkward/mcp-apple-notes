"""Read-only graph query sidecar for the Apple Notes knowledge graph (FalkorDB).

The notes UI (Bun) auto-spawns this and proxies /api/related to it. FalkorDB is a
server (Redis-based) so concurrent reads are fine — no single-process constraint.
Pure Cypher, no LLM / API key needed.

Config (env):
  FALKOR_HOST  default localhost
  FALKOR_PORT  default 6379
  FALKOR_DB    graph name, default "notes"
  GRAPH_PORT   HTTP port to listen on, default 3743

Schema (graphiti-core): notes are :Episodic {name=title}, linked to :Entity via
MENTIONS; entities link to each other via RELATES_TO.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from falkordb import FalkorDB

PORT = int(os.environ.get("GRAPH_PORT", "3743"))
_db = FalkorDB(host=os.environ.get("FALKOR_HOST", "localhost"),
               port=int(os.environ.get("FALKOR_PORT", "6379")))
_g = _db.select_graph(os.environ.get("FALKOR_DB", "notes"))


def q(cypher, params=None):
    return _g.query(cypher, params or {}).result_set


def related_notes(title, limit=15):
    rows = q(
        "MATCH (src:Episodic {name:$t})-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(other:Episodic) "
        "WHERE other.name <> src.name "
        "RETURN other.name AS note, count(DISTINCT e) AS shared ORDER BY shared DESC LIMIT $l",
        {"t": title, "l": limit})
    return [{"note": r[0], "shared": r[1]} for r in rows]


def notes_for_entity(name, limit=15):
    rows = q(
        "MATCH (e:Entity)<-[:MENTIONS]-(note:Episodic) WHERE e.name = $n "
        "RETURN note.name AS note, count(e) AS hits ORDER BY hits DESC LIMIT $l",
        {"n": name, "l": limit})
    return [{"note": r[0], "hits": r[1]} for r in rows]


def entities_in_note(title, limit=20):
    rows = q("MATCH (:Episodic {name:$t})-[:MENTIONS]->(e:Entity) RETURN e.name LIMIT $l",
             {"t": title, "l": limit})
    return [r[0] for r in rows]


def health():
    def c(cypher):
        try:
            return q(cypher)[0][0]
        except Exception:
            return 0
    return {
        "ok": True, "backend": "falkordb",
        "episodes": c("MATCH (e:Episodic) RETURN count(e)"),
        "entities": c("MATCH (n:Entity) RETURN count(n)"),
        "edges": c("MATCH ()-[r:RELATES_TO]->() RETURN count(r)"),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
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
    print(f"graph sidecar (falkordb) on http://127.0.0.1:{PORT}", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
