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
| `get-note` | Full note by title; fuzzy fallback on no exact match |
| `list-notes` | Notes sorted by recency; optional `folder`, date range, `limit` |
| `create-note` | Create a note |
| `update-note` | Edit an existing note |
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
