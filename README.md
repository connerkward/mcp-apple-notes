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
- 🕐 Auto re-index: every search runs ~1ms change detection and, if notes changed, kicks ONE background index job (single-flight) — search itself never blocks on indexing
- ✂️ 1500-char chunking for long notes
- 🍎 Direct SQLite + protobuf decode — real note text, not garbled HTML
- 🏃‍♂️ Fully local — no API keys, no cloud

## Local web app — not just search, but connections & synthesis

Run the server directly and open the browser UI — no MCP client needed:

```bash
bun index.ts            # → http://localhost:3741/   (also serves /mcp)
```

Six modes, building from retrieval toward sensemaking:

- **Feed** (default) — one ranked, evidence-first stream mixing three connection kinds: **bridges** (A ↔ C via B, with the connecting phrase mined from B as plain-text evidence), **abstraction pairs** (two notes whose LLM-consolidated abstractions from `~/.mcp-apple-notes/consolidated.jsonl` make the same underlying move), and **entity threads** ("X notes across Y folders mention Z", from the entity graph). Every item shows its evidence in plain text. 👍/👎 on any item tunes the ranking: a tiny online logistic regression (pure TS, no deps) takes a few SGD steps per vote; votes append to `~/.mcp-apple-notes/votes.jsonl` (with the item's feature vector) and replay on boot, so the ranking survives restarts. Diversity constraints: ≤2 consecutive items of one kind, no note more than twice per page of 20; "show more" paginates. Endpoints: `GET /api/feed?offset=&limit=`, `POST /api/vote` `{id, vote: 1|-1}`.
- **Search** — the hybrid semantic + BM25 search, in a paper-themed UI with query highlighting and folder filter.
- **Map** — a topic map of every note: spherical k-means clusters (TF-IDF labels) over the embeddings, projected to 2D with PCA. Hover a note for the **neighbor-lens** — lines to its nearest-by-meaning notes, which often cross clusters (the connection-finding payload). Endpoint: `GET /api/clusters?k=`.
- **Synthesize** — *"what do I think about X across everything I've written?"* Query-expansion → relevance-gated retrieval (no recency bias) → MMR diversification → an LLM writes a grounded answer with inline `[n]` citations back to the source notes; provenance is post-checked. Endpoint: `GET /api/synthesize?q=`.
- **Bridges** — Swanson-ABC discovery over your own notes: pairs (A, C) that are **not** directly similar but are both strongly similar to a shared bridge note B — connections the corpus only makes through an intermediary. Pure arithmetic over the embeddings (score = sim(A,B)·sim(B,C)·(1−sim(A,C)); cross-folder, no shared tags/wikilinks; thresholds auto-relax; hub-penalized selection so no note dominates the list). No LLM, $0. Endpoint: `GET /api/bridges?limit=40`.
- **Entities** — *"where else do I talk about Mercedes?"* Ranked entity chips (people, orgs, concepts) → the notes that mention them, by mention weight. Reads an optional sqlite graph db (see below). Endpoints: `GET /api/entities?q=&limit=`, `GET /api/entity-notes?entity=`.

### Precomputed disk caches (instant bridges & map)

Bridge mining (~2 min of embedding) and clustering are computed once and persisted, keyed by a corpus fingerprint (note count + max modification date):

- `~/.mcp-apple-notes/bridges_cache.json` — the full mined bridge pool
- `~/.mcp-apple-notes/clusters_cache.json` — cluster output per `k`

Fresh cache → served from disk instantly (<300ms). Stale (notes changed) → the stale copy is served instantly with a "recomputing…" pill in the UI while ONE background job (single-flight) refreshes it. Absent → computed inline once.

### Enabling synthesis (needs an LLM)

Embeddings/search/clustering/bridges/entities are fully local. Only **synthesis generation** needs an LLM. Point at a local OpenAI-compatible server to keep notes private:

```bash
# LM Studio / Ollama (zero API cost, notes stay local):
SYNTH_BASE_URL=http://localhost:1234/v1 SYNTH_MODEL=<loaded-model> OPENAI_API_KEY=local bun index.ts
# …or real OpenAI: set a funded OPENAI_API_KEY (defaults to gpt-4o-mini).
```

### The entity graph (optional)

The entity layer reads `~/.mcp-apple-notes/layered_graph.db` (override with `LAYERED_DB`) — a plain sqlite file produced by the companion [`exp-notes-indexing`](https://github.com/connerkward/exp-notes-indexing) benchmark harness (`layered_graph.py`). No extra dependencies (bun ships `bun:sqlite`); if the file is absent the entity tools and tab simply report how to generate it.

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

### Claude Code / CLI registration

```json
// .mcp.json (project) or `claude mcp add apple-notes -- bun /path/to/mcp-apple-notes/index.ts --stdio`
{
  "mcpServers": {
    "apple-notes": {
      "command": "bun",
      "args": ["/path/to/mcp-apple-notes/index.ts", "--stdio"]
    }
  }
}
```

The same bridges/entities tools power the web UI tabs at the local app (`bun index.ts` → http://localhost:3741/); the entity graph db is optional, generated by the `exp-notes-indexing` benchmark harness.

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
| `bridge-notes` | Swanson-ABC bridges: non-similar note pairs connected via a shared intermediary; optional `folder`, `limit` |
| `feed` | The ranked evidence-first connection feed (bridges + abstraction pairs + entity threads) as JSON; optional `limit` |
| `entity-notes` | Notes mentioning an entity (e.g. "Mercedes"), by mention weight — needs the optional entity graph db |
| `list-entities` | Entities ranked by mention count; optional substring `query`, `limit` |
| `get-tables` | Extract pipe/tab-separated tables from a note |
| `create-note` | Create a note |
| `update-note` | Edit an existing note |
| `check-changes` | Check if notes changed since last index (without triggering re-index) |
| `index-health` | Sync status, last indexed time, note count |

## Search & Ranking

Every search detects changes (~1ms). If notes changed, ONE background incremental index job is kicked (single-flight — duplicate triggers are dropped) and the search returns immediately from the current index; results catch up when the job lands. The last-indexed watermark persists to `~/.mcp-apple-notes/index_state.json` across restarts. Change detection compares the per-title *set* of modification dates (Apple Notes titles are not unique — "TODO" ×10 — and a title→single-date map can never converge for duplicates).

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
