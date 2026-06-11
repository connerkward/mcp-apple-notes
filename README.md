# MCP Apple Notes

![MCP Apple Notes](./images/logo.png)

A [Model Context Protocol (MCP)](https://www.anthropic.com/news/model-context-protocol) server for fast, accurate, always-fresh semantic search over your Apple Notes — fully local, no API keys.

> **Other MCP Notes servers break at scale.** They use JXA (AppleScript automation) to read notes one-by-one — fine for 50 notes, unusable at 500+. At 1,800 notes, JXA takes ~49 minutes just to fetch content. On macOS Sequoia it's worse: Apple silently denies Automation permission to processes without a bundle ID, so JXA-based servers fail entirely. This fork reads the SQLite database directly, decodes the protobuf blobs for real note text, and indexes 1,800 notes in under 5 seconds.

![MCP Apple Notes](./images/demo.png)

## Comparison

| Feature | **This fork** | [RafalWilinski](https://github.com/RafalWilinski/mcp-apple-notes) (base) | [disco-trooper](https://github.com/disco-trooper/apple-notes-mcp) | [sirmews](https://github.com/sirmews/apple-notes-mcp) | [dhravya](https://github.com/Dhravya/apple-mcp) |
|---|---|---|---|---|---|
| Notes access | SQLite (direct) | JXA | JXA | SQLite | JXA |
| Fetch 1800 notes | **~430ms** | ~49 min (est.) | ~49 min (est.) | fast | slow |
| Sequoia compatible | ✅ | ⚠️ JXA denied | ⚠️ JXA denied | ✅ | ⚠️ |
| Content quality | **Protobuf decoded** | Raw HTML | Raw HTML | Partial | Raw |
| Semantic search | ✅ | ✅ | ✅ | ❌ | ❌ |
| Auto re-index on search | ✅ | ❌ | ❌ | ❌ | ❌ |
| Folder-aware search | ✅ Full path | ❌ | ✅ | ✅ | ✅ |
| Note chunking | ✅ 1500 chars | ❌ | ✅ | ❌ | ❌ |
| Re-ranking (RRF × title × recency) | ✅ | RRF only | RRF only | ❌ | ❌ |
| Non-blocking index UI | ✅ Live progress | ❌ | ❌ | ❌ | ❌ |
| Incremental indexing | ✅ | ❌ | ✅ | ❌ | ❌ |
| Update existing notes | ✅ | ❌ | ✅ | ❌ | ❌ |
| Tags + wikilinks | ✅ | ❌ | ❌ | ❌ | ❌ |
| Related notes | ✅ Multi-signal | ❌ | ❌ | ❌ | ❌ |

> JXA on macOS Sequoia: processes without a bundle ID are silently auto-denied Automation permission. This fork reads SQLite directly.

## Features

- 🔍 Semantic search via [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) on-device embeddings
- 📝 BM25 full-text search combined via Reciprocal Rank Fusion
- 🧠 Re-ranking: `RRF × title_boost × recency_factor`
- 📂 Full folder hierarchy — filter by any path segment
- 🕐 Auto re-index on every search (~1ms change detection)
- ✂️ 1500-char chunking for long notes
- 🍎 Direct SQLite + protobuf decode — real note text, not garbled HTML
- 🏃‍♂️ Fully local — no API keys, no cloud

## Local web app — not just search, but connections & synthesis

Run the server directly and open the browser UI — no MCP client needed:

```bash
bun index.ts            # → http://localhost:3741/   (also serves /mcp)
```

Four modes, building from retrieval toward sensemaking:

- **Search** — the hybrid semantic + BM25 search, in a paper-themed UI with query highlighting and folder filter. Each result can expand a **related (graph)** panel (see graph below).
- **Map** — a topic map of every note: spherical k-means clusters (TF-IDF labels) over the embeddings, projected to 2D with PCA. Hover a note for the **neighbor-lens** — lines to its nearest-by-meaning notes, which often cross clusters (the connection-finding payload). Endpoint: `GET /api/clusters?k=`.
- **Synthesize** — *"what do I think about X across everything I've written?"* Query-expansion → relevance-gated retrieval (no recency bias) → MMR diversification → an LLM writes a grounded answer with inline `[n]` citations back to the source notes; provenance is post-checked. Endpoint: `GET /api/synthesize?q=`.
- **Graph** — entity-based related notes from a [Graphiti](https://github.com/getzep/graphiti) / Kuzu knowledge graph (people/places/concepts extracted per note). Graph **queries need no LLM** (read-only Cypher via a small Python sidecar that the server auto-spawns). Endpoints: `/api/related`, `/api/graph-entity`, `/api/graph-status`.

### Enabling synthesis (needs an LLM)

Embeddings/search/clustering/graph-queries are fully local. Only **synthesis generation** and **building the graph** need an LLM. Point at a local OpenAI-compatible server to keep notes private:

```bash
# LM Studio / Ollama (zero API cost, notes stay local):
SYNTH_BASE_URL=http://localhost:1234/v1 SYNTH_MODEL=<loaded-model> OPENAI_API_KEY=local bun index.ts
# …or real OpenAI: set a funded OPENAI_API_KEY (defaults to gpt-4o-mini).
```

### The knowledge graph

The graph is built by the companion [`exp-notes-indexing`](https://github.com/connerkward/exp-notes-indexing) pipeline (Apple Notes → entity/relationship extraction → Kuzu). The web app reads it via `graph/server.py`; configure with `GRAPH_DB` (path to the `.kuzu`), `GRAPH_PY` (python with `kuzu` installed), `GRAPH_PORT`. If no graph is present the UI simply omits the graph panel.

## Installation

1. Clone and install:

```bash
git clone https://github.com/connerkward/mcp-apple-notes
cd mcp-apple-notes
bun install
```

2. Add **bun** (`~/.bun/bin/bun`) to **Full Disk Access** in System Settings.

3. Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "apple-notes": {
      "command": "/Users/<YOUR_USER_NAME>/.bun/bin/bun",
      "args": ["/Users/<YOUR_USER_NAME>/mcp-apple-notes/index.ts", "--stdio"]
    }
  }
}
```

4. Restart Claude Desktop and ask: *"Index my Apple Notes"*.

## Tools

| Tool | Description |
|------|-------------|
| `index-notes` | Background indexing with live progress UI |
| `search-notes` | Semantic + FTS search; optional `folder`, `modifiedAfter`, `modifiedBefore` |
| `find-notes` | Exact substring search (like Apple Notes built-in); optional `folder`, date range |
| `get-note` | Full note by title; fuzzy fallback on no exact match |
| `list-notes` | Notes sorted by recency; optional `folder`, date range, `limit` |
| `list-folders` | All folders with note counts |
| `list-tags` | All `#hashtags` across notes, sorted by frequency |
| `search-by-tag` | Notes containing a specific hashtag |
| `related-notes` | Related notes via shared tags, `[[wikilinks]]`, and vector similarity |
| `get-tables` | Extract pipe/tab-separated tables from a note |
| `create-note` | Create a note |
| `update-note` | Edit an existing note |
| `check-changes` | Check if notes changed since last index (without triggering re-index) |
| `index-health` | Sync status, last indexed time, note count |

## Search & Ranking

Every search detects changes (~1ms) and incrementally re-indexes before returning — always in sync.

```
score = RRF(vector, BM25) × title_boost × recency_factor
```

Temporal queries (`recent`, `latest`, `today`) automatically shift to a 1-day recency half-life at 70% weight. Normal queries use 90-day half-life at 10% so relevance stays primary.

## Benchmarks

1,806 notes, Apple Silicon:

| Approach | Time | Body included |
|---|---|---|
| JXA metadata only | 4,463ms | No |
| JXA with body | ~49 min | Yes |
| **SQLite direct** | **430ms** | **Yes** |

## Troubleshooting

```bash
tail -n 50 -f ~/Library/Logs/Claude/mcp-server-apple-notes.log
```

Permissions error → ensure **bun** (`~/.bun/bin/bun`) has Full Disk Access in System Settings → Privacy & Security.
